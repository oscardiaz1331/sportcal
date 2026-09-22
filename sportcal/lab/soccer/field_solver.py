"""Soccer field homography from a binary LINE MASK, three hypothesis generators + a shared scorer.

  * search_pose():    "chamfer without pairing lines". A grid of pinhole camera poses (pan, tilt, focal, a few
                      typical camera positions), each scored by how well the projected template lands on the
                      mask; the best ones are refined with a free 8-parameter H.
  * search_lines():   "intersections". Straight lines are detected in the mask; every quadruple of lines
                      (2 parallel to the field X axis + 2 to the Y axis) against every pair of template lines
                      gives 4 POINT correspondences (their 4 crossings) -> one H.
  * search_ellipse(): hypotheses from a detected centre-circle ellipse (a conic = 5 constraints on the pose).

All of them are scored the same way and refined the same way. Template, world frame and the 180-degree
symmetry convention: sportcal/sports/soccer/field.py. What was measured and what failed:
docs/experiments/soccer.md.

Score (length-weighted Dice, in px): 2*M / (V + Mlen), with V the visible length of the projected template,
M the part of it lying on the mask (weighted by distance to the mask, truncated at tau) and Mlen the length of
line in the mask. Weighting by IMAGE length (not by number of world samples) stops a tiny template squashed onto
a dense area from scoring high just because many samples fall close together.
"""
import itertools

import cv2
import numpy as np
from scipy.optimize import minimize

from sportcal.core.camera import (
    bilinear, is_pinhole_consistent, pinhole_residual, pose_to_H, project,
)
from sportcal.core.fitting import detect_lines
from sportcal.sports.soccer.field import (
    CIRCLE_CENTERS, CIRCLE_RADIUS, CORNERS, X_LINES, Y_LINES, canonicalize_H, sample_template,
)


def _line_length(mask):
    """Length (px) of the lines in a mask = sum of 1 / local width over its pixels (width = 2 x distance
    to the background): a 2 px wide line of length L gives ~L, a filled blob adds ~its diameter, not its
    area. Independent of topology (a 1 px skeleton or a ridge count both misbehave on noisy masks)."""
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return float(max((1.0 / np.maximum(2.0 * dist_in[mask > 0], 1.0)).sum(), 1.0))


def _circle_conic(cx, cy):
    """3x3 conic matrix of the world circle of radius CIRCLE_RADIUS centred at (cx, cy)."""
    return np.array([[1, 0, -cx], [0, 1, -cy], [-cx, -cy, cx * cx + cy * cy - CIRCLE_RADIUS ** 2]], float)


class FieldSolver:
    """Solves H for ONE line mask (uint8 0/1) of size (h, w)."""

    def __init__(self, mask, tau=3.0):
        self.h, self.w = mask.shape
        self.mask = (mask > 0).astype(np.uint8)
        # distance of every pixel to the mask (0 on it)
        self.dist = cv2.distanceTransform((self.mask == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        self.mask_len = _line_length(self.mask)
        self.tau = tau
        self.pinhole_penalty = 0.5
        self.ellipse_pts = self.corner_pts = None  # point evidence, see set_evidence
        self.w_ellipse = self.w_corner = 0.0
        self.pts_coarse, self.cont_coarse = sample_template(1.0)
        self.pts_fine, self.cont_fine = sample_template(0.25)

    # -- point evidence ---------------------------------------------------------
    def set_evidence(self, ellipse_pts=None, corners=None, w_ellipse=0.6, w_corner=0.4, n_pts=60, seed=0):
        """Add what a detected ellipse and detected corners say to the score (working pixels).

        ellipse_pts (n, 2): ellipse edge points. Term = fraction of them lying (Sampson distance < tau) on
            any of the 3 possible projected circles (centre / penalty arcs).
        corners (m, 2): detected line crossings. Term = fraction of them lying (< tau) on a visible
            projected template corner.
        score = Dice + w_ellipse * ellipse term + w_corner * corner term. Without it, refining against a
        dirty mask (low Dice) drags away hypotheses the ellipse had already got right.
        """
        if ellipse_pts is not None and len(ellipse_pts) >= 12:
            E = np.asarray(ellipse_pts, float)
            E = E[np.random.default_rng(seed).choice(len(E), min(n_pts, len(E)), replace=False)]
            self.ellipse_pts, self.w_ellipse = np.c_[E, np.ones(len(E))], w_ellipse
        else:
            self.ellipse_pts, self.w_ellipse = None, 0.0
        if corners is not None and len(corners) >= 1:
            self.corner_pts, self.w_corner = np.asarray(corners, float).reshape(-1, 2), w_corner
        else:
            self.corner_pts, self.w_corner = None, 0.0

    def _ellipse_term(self, Hs, tau):
        n = len(Hs)
        det = np.linalg.det(Hs)
        bad = ~(np.abs(det) > 1e-12)
        Hi = np.linalg.inv(np.where(bad[:, None, None], np.eye(3), Hs))
        best = np.zeros(n)
        for cx, cy in CIRCLE_CENTERS:
            Ci = np.einsum("nji,jk,nkl->nil", Hi, _circle_conic(cx, cy), Hi)  # Hi^T C Hi
            CX = np.einsum("nij,mj->nmi", Ci, self.ellipse_pts)
            q = np.einsum("nmi,mi->nm", CX, self.ellipse_pts)
            g = np.maximum(2.0 * np.hypot(CX[..., 0], CX[..., 1]), 1e-9)
            best = np.maximum(best, np.mean(np.clip(1.0 - np.abs(q / g) / tau, 0, 1), axis=1))
        return np.where(bad, 0.0, best)

    def _corner_term(self, Hs, tau):
        xy, depth = project(Hs, CORNERS)  # (n, 22, 2)
        ok = ((depth > 0) & np.isfinite(xy).all(-1) & (xy[..., 0] > -tau) & (xy[..., 0] < self.w + tau)
              & (xy[..., 1] > -tau) & (xy[..., 1] < self.h + tau))
        d = np.hypot(xy[:, None, :, 0] - self.corner_pts[None, :, None, 0],
                     xy[:, None, :, 1] - self.corner_pts[None, :, None, 1])  # (n, m, 22)
        d = np.where(ok[:, None, :], d, np.inf)
        return np.mean(np.clip(1.0 - d.min(axis=2) / tau, 0, 1), axis=1)

    # -- scoring ------------------------------------------------------------------
    def score(self, Hs, fine=False, tau=None, chunk=1500):
        """Score of each H in Hs (n, 3, 3) -> (n,). `fine` samples the template every 0.25 m instead of 1 m."""
        pts, cont = (self.pts_fine, self.cont_fine) if fine else (self.pts_coarse, self.cont_coarse)
        tau = self.tau if tau is None else tau
        Hs = np.asarray(Hs, float).reshape(-1, 3, 3)
        out = np.empty(len(Hs))
        for i in range(0, len(Hs), chunk):
            xy, depth = project(Hs[i:i + chunk], pts)
            ok = ((depth > 1e-6) & (xy[..., 0] >= 0) & (xy[..., 0] <= self.w - 1)
                  & (xy[..., 1] >= 0) & (xy[..., 1] <= self.h - 1))
            seg = ok[:, 1:] & ok[:, :-1] & cont[1:]
            xy = np.where(ok[..., None], xy, 0.0)
            length = np.hypot(xy[:, 1:, 0] - xy[:, :-1, 0], xy[:, 1:, 1] - xy[:, :-1, 1]) * seg
            mid = 0.5 * (xy[:, 1:] + xy[:, :-1])
            weight = 1.0 - np.minimum(bilinear(self.dist, mid) / tau, 1.0)
            on_mask, visible = (length * weight).sum(1), length.sum(1)
            out[i:i + chunk] = 2.0 * on_mask / (visible + self.mask_len)
            if self.ellipse_pts is not None:
                out[i:i + chunk] += self.w_ellipse * self._ellipse_term(Hs[i:i + chunk], tau)
            if self.corner_pts is not None:
                out[i:i + chunk] += self.w_corner * self._corner_term(Hs[i:i + chunk], tau)
        return out

    # -- generator 1: chamfer over a pose grid --------------------------------------
    def pose_grid(self):
        positions = [(cx, cy, cz) for cx in (-20, 0, 20) for cy in (-50, -70) for cz in (15, 25)]
        pan = np.radians(np.arange(-50, 51, 4))
        tilt = np.radians([8, 14, 20, 26, 32, 38])
        f = self.w * np.geomspace(0.6, 4.5, 8)
        return np.array([(*p, a, t, ff) for p in positions for a in pan for t in tilt for ff in f])

    def search_pose(self, top=12):
        Hs = pose_to_H(self.pose_grid(), self.w, self.h)
        # wide tau: the grid is coarse (the nearest cell is ~100 px away); with a few px of tau every
        # score between cells is 0 and there is nothing to compare
        return self._best_distinct(Hs, self.score(Hs, tau=0.04 * self.w), top)

    # -- generator 2: line quadruples -> 4 points -> H ------------------------------
    def search_lines(self, top=8, n_max=7):
        lines = detect_lines(self.mask, n_max)
        if len(lines) < 4:
            return []
        Hs = line_hypotheses(lines, self.w, self.h)
        if len(Hs) == 0:
            return []
        return self._best_distinct(Hs, self.score(Hs, tau=2.0 * self.tau), top)

    # -- generator 3: hypotheses from the circle's ELLIPSE ---------------------------
    def search_ellipse(self, e_pts, top=8, n_starts=40, seed=0, n_pts=60):
        """A detected ellipse is a conic: 5 constraints on the pose. With the pinhole model (6 parameters,
        no roll, centred principal point) they almost fix it and the mask breaks the remaining ties.

        e_pts (n, 2): ellipse edge points in working pixels. For each possible circle (centre or either
        penalty arc) n_starts local fits are launched from random poses, minimising the Sampson distance of
        the points to the projected circle (with a weak prior for a broadcast camera); poses converging to
        < 4 px go on to mask refinement. Returns like search_pose / search_lines.
        """
        from scipy.optimize import least_squares
        rng = np.random.default_rng(seed)
        E = np.asarray(e_pts, float)
        if len(E) < 12:
            return []
        E = E[rng.choice(len(E), min(n_pts, len(E)), replace=False)]
        X = np.c_[E, np.ones(len(E))]
        lo = np.array([-40, -95, 6, np.radians(-70), np.radians(3), 0.5 * self.w])
        hi = np.array([40, -30, 40, np.radians(70), np.radians(60), 9.0 * self.w])
        Hs = []
        for cx, cy in CIRCLE_CENTERS:
            Cw = _circle_conic(cx, cy)

            def resid(p):
                H = pose_to_H(p[None], self.w, self.h)[0]
                try:
                    Hi = np.linalg.inv(H)
                except np.linalg.LinAlgError:
                    return np.full(len(E) + 3, 1e3)
                Ci = Hi.T @ Cw @ Hi
                CX = X @ Ci
                q = np.einsum("ni,ni->n", CX, X)
                g = np.maximum(2.0 * np.hypot(CX[:, 0], CX[:, 1]), 1e-9)
                d = np.clip(q / g, -300, 300)  # Sampson distance (px)
                return np.r_[d, 0.2 * (p[2] - 18.0), 0.1 * (p[1] + 58.0), 0.1 * p[0]]

            for _ in range(n_starts):
                x0 = np.array([rng.uniform(-25, 25), rng.uniform(-75, -42), rng.uniform(10, 28),
                               rng.uniform(-np.radians(50), np.radians(50)),
                               rng.uniform(np.radians(8), np.radians(40)),
                               self.w * np.exp(rng.uniform(np.log(0.7), np.log(5.0)))])
                try:
                    r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=60, xtol=1e-6, ftol=1e-6)
                except Exception:
                    continue
                if np.sqrt(np.mean(r.fun[:len(E)] ** 2)) < 4.0:
                    Hs.append(pose_to_H(r.x[None], self.w, self.h)[0])
        if not Hs:
            return []
        Hs = np.array(Hs)
        return self._best_distinct(Hs, self.score(Hs, tau=0.04 * self.w), top)

    def search_ellipse_fixed_center(self, center, e_pts, top=4, n_starts=24, seed=0, n_pts=60):
        """Like `search_ellipse`, but with the camera CENTRE known: only pan / tilt / focal are free (3
        unknowns against the ellipse's 5 constraints, so it is over-determined and needs no mask and no
        point clicks at all). Returns like `search_lines` / `search_pose`."""
        from scipy.optimize import least_squares
        rng = np.random.default_rng(seed)
        c = np.asarray(center, float)
        E = np.asarray(e_pts, float)
        if len(E) < 5:
            return []
        E = E[rng.choice(len(E), min(n_pts, len(E)), replace=False)]
        X = np.c_[E, np.ones(len(E))]
        lo = np.array([-np.pi, np.radians(2), 0.4 * self.w])
        hi = np.array([np.pi, np.radians(65), 9.0 * self.w])
        Hs, ptf = [], []
        for cx, cy in CIRCLE_CENTERS:
            Cw = _circle_conic(cx, cy)

            def resid(q):
                H = pose_to_H(np.r_[c, q[0], q[1], q[2]][None], self.w, self.h)[0]
                try:
                    Hi = np.linalg.inv(H)
                except np.linalg.LinAlgError:
                    return np.full(len(E), 1e3)
                Ci = Hi.T @ Cw @ Hi
                CX = X @ Ci
                g = np.maximum(2.0 * np.hypot(CX[:, 0], CX[:, 1]), 1e-9)
                return np.clip(np.einsum("ni,ni->n", CX, X) / g, -300, 300)

            for _ in range(n_starts):
                x0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(np.radians(8), np.radians(45)),
                               self.w * np.exp(rng.uniform(np.log(0.6), np.log(4.0)))])
                try:
                    r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=60, xtol=1e-7, ftol=1e-7)
                except Exception:
                    continue
                if np.sqrt(np.mean(r.fun ** 2)) < 3.0:
                    Hs.append(pose_to_H(np.r_[c, r.x][None], self.w, self.h)[0])
                    ptf.append(r.x)
        if not Hs:
            return []
        Hs = np.array(Hs)
        # anchor ranking AND refinement to the ellipse itself: on a close-up the line mask is thin or empty and
        # would otherwise drag the fit back towards whatever little it sees (see docs/experiments/soccer.md 13)
        saved = (self.ellipse_pts, self.w_ellipse)
        self.set_evidence(E, self.corner_pts, w_ellipse=max(1.5, self.w_ellipse), w_corner=self.w_corner)
        try:
            return self._best_distinct(Hs, self.score(Hs, tau=0.04 * self.w), top,
                                       refiner=lambda i: self.refine_ptz(c, *ptf[i]))
        finally:
            self.ellipse_pts, self.w_ellipse = saved

    # -- corners as point correspondences ----------------------------------------------
    def snap_corners(self, H, Q, tol_px=None, min_match=4):
        """Re-fit H with detected corners Q (m, 2, working pixels): each visible template corner is paired
        with the nearest detected one (mutual nearest neighbours closer than tol_px) and H is re-fitted on
        those pairs. Breaks sliding along a line, which chamfer cannot see. Returns (new H, n pairs), or
        (H, 0) when fewer than min_match pairs exist."""
        Q = np.asarray(Q, float).reshape(-1, 2)
        tol = tol_px if tol_px is not None else 0.03 * self.w
        if len(Q) < min_match:
            return H, 0
        xy, depth = project(np.asarray(H, float)[None], CORNERS)
        xy, depth = xy[0], depth[0]
        visible = np.flatnonzero((depth > 0) & np.isfinite(xy).all(1) & (xy[:, 0] > -tol) & (xy[:, 0] < self.w + tol)
                                 & (xy[:, 1] > -tol) & (xy[:, 1] < self.h + tol))
        if len(visible) < min_match:
            return H, 0
        D = np.hypot(xy[visible, None, 0] - Q[None, :, 0], xy[visible, None, 1] - Q[None, :, 1])
        a = np.argmin(D, axis=1)
        b = np.argmin(D, axis=0)
        pairs = [(visible[i], a[i]) for i in range(len(visible)) if D[i, a[i]] < tol and b[a[i]] == i]
        if len(pairs) < min_match:
            return H, 0
        world = CORNERS[[i for i, _ in pairs]]
        image = Q[[j for _, j in pairs]]
        Hn, _ = cv2.findHomography(world.astype(np.float32), image.astype(np.float32),
                                   cv2.RANSAC if len(pairs) > 4 else 0, 3.0)
        if Hn is None or not np.all(np.isfinite(Hn)):
            return H, 0
        return Hn, len(pairs)

    # -- shared: pick distinct candidates, refine ---------------------------------------
    def _best_distinct(self, Hs, scores, top, min_dist=0.03, refiner=None):
        """Top-`top` DISTINCT hypotheses (their 4 image anchors land > 2 m apart in the world), refined.

        `refiner(i)` -> (H, score) refines hypothesis i; the default is the free 8-parameter `refine`."""
        anchors = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]) * [self.w, self.h]
        order = np.argsort(-scores)[:400]
        chosen, signatures = [], []
        for i in order:
            Hi = np.linalg.inv(Hs[i])
            xy = (Hi @ np.c_[anchors, np.ones(4)].T).T
            with np.errstate(divide="ignore", invalid="ignore"):
                signature = (xy[:, :2] / xy[:, 2:]).ravel()  # where the image anchors fall in the WORLD
            if not np.isfinite(signature).all():
                continue
            if all(np.abs(signature - s).max() > 2.0 for s in signatures):
                signatures.append(signature)
                chosen.append(i)
            if len(chosen) >= top:
                break
        results = []
        for i in chosen:
            Hr, s = refiner(i) if refiner is not None else self.refine(Hs[i])
            results.append({"H": canonicalize_H(Hr), "score": s, "score0": float(scores[i]), "H0": Hs[i]})
        return sorted(results, key=lambda r: -r["score"])

    def refine(self, H0, taus=None):
        """Perturb 4 fixed image points (8 parameters, px) to maximise the fine score. Cascade of
        decreasing tau: a wide tau makes the score smooth and recovers 100+ px of error, the last
        passes sharpen it; only the last one uses the (more expensive) fine sampling. `taus` (fractions of the
        width, the last one is always self.tau) replaces the default cascade, e.g. a short one to track a start
        that is only a few pixels off."""
        src = (np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]) * [self.w, self.h]).astype(np.float32)

        def H_of(d):
            P = cv2.getPerspectiveTransform(src, src + d.reshape(4, 2).astype(np.float32))
            return P @ H0

        best = H0
        cascade = (0.04 * self.w, 0.015 * self.w, 0.006 * self.w) if taus is None else tuple(t * self.w for t in taus)
        for tau in cascade + (self.tau,):
            fine = tau == self.tau
            # pinhole penalty: a free 8-parameter H can "win" over the truth by deforming into
            # something no camera produces
            r = minimize(lambda d: -(self.score(H_of(d)[None], fine, tau)[0]
                                     - self.pinhole_penalty * pinhole_residual(H_of(d)[None], self.w, self.h)[1][0]),
                         np.zeros(8), method="Powell", options={"xtol": 0.05, "ftol": 1e-4, "maxfev": 400})
            best = H_of(r.x)
            H0 = best
        return best, float(self.score(best[None], True)[0])

    # -- generator 4: FIXED camera centre, only pan / tilt / zoom vary ---------------------
    def refine_ptz(self, center, pan, tilt, f):
        """Maximise the fine score over (pan, tilt, focal) with the camera centre held fixed.

        Same tau cascade as `refine`. Parametrised by (pan, tilt, log f) so one step moves the image by a comparable
        amount in each. The result is always an exact pinhole camera at `center`, unlike the free 8-parameter fit."""
        c = np.asarray(center, float)

        def H_of(q):
            return pose_to_H(np.r_[c, q[0], q[1], np.exp(q[2])][None], self.w, self.h)[0]

        q = np.array([pan, tilt, np.log(f)])
        for tau in (0.04 * self.w, 0.015 * self.w, 0.006 * self.w, self.tau):
            fine = tau == self.tau
            r = minimize(lambda x: -self.score(H_of(x)[None], fine, tau)[0], q, method="Powell",
                         options={"xtol": 1e-4, "ftol": 1e-5, "maxfev": 300, "direc": np.diag([0.02, 0.02, 0.05])})
            q = r.x
        H = H_of(q)
        return H, float(self.score(H[None], True)[0])

    def search_fixed_center(self, center, top=4, pan_range=(-60.0, 60.0), tilt_range=(4.0, 40.0), f_range=(0.6, 6.0),
                            steps=(2.0, 2.0, 30)):
        """Hypotheses for a broadcast camera whose CENTRE is known (it only pans, tilts and zooms): a 3-D grid over
        pan (deg), tilt (deg) and focal length (in image widths, log-spaced), scored with the wide tau, then each
        distinct winner refined with `refine_ptz`. `steps` = (pan step deg, tilt step deg, number of focal values).
        With a known centre a single ellipse or a couple of lines is enough to fix the pose."""
        c = np.asarray(center, float)
        pans = np.radians(np.linspace(pan_range[0], pan_range[1], max(2, int(round((pan_range[1] - pan_range[0]) / steps[0])) + 1)))
        tilts = np.radians(np.linspace(tilt_range[0], tilt_range[1], max(2, int(round((tilt_range[1] - tilt_range[0]) / steps[1])) + 1)))
        focals = self.w * np.geomspace(f_range[0], f_range[1], int(steps[2]))
        A, T, F = np.meshgrid(pans, tilts, focals, indexing="ij")
        grid = np.column_stack([np.tile(c, (A.size, 1)), A.ravel(), T.ravel(), F.ravel()])
        Hs = pose_to_H(grid, self.w, self.h)
        return self._best_distinct(Hs, self.score(Hs, tau=0.04 * self.w), top,
                                   refiner=lambda i: self.refine_ptz(c, grid[i, 3], grid[i, 4], grid[i, 5]))


# ---------------------------------------------------------------- hypotheses from detected lines

def line_hypotheses(lines, w, h, xs=X_LINES, ys=Y_LINES):
    """All H obtained by assigning 2 image lines to 2 field lines X=const and another 2 to 2 field lines
    Y=const (both orderings of each pair). Drops mirrored fields, hypotheses with a point behind the camera
    and degenerate ones. (n, 3, 3)."""
    N = len(lines)
    world_v = [(i, j) for i in range(len(xs)) for j in range(len(xs)) if i != j]
    world_h = [(k, m) for k in range(len(ys)) for m in range(len(ys)) if k != m]
    wv = np.array([(a, b) for a in world_v for b in world_h])  # (1260, 2, 2) indices
    X = np.array(xs)[wv[:, 0]]  # (1260, 2): x of the 2 vertical lines
    Y = np.array(ys)[wv[:, 1]]  # (1260, 2): y of the 2 horizontal lines
    # 4 world points per hypothesis: (xi, yk), (xi, yl), (xj, yk), (xj, yl)
    Wp = np.stack([np.stack([X[:, 0], Y[:, 0]], 1), np.stack([X[:, 0], Y[:, 1]], 1),
                   np.stack([X[:, 1], Y[:, 0]], 1), np.stack([X[:, 1], Y[:, 1]], 1)], 1)  # (1260, 4, 2)
    out = []
    for A in itertools.combinations(range(N), 2):
        for B in itertools.combinations(range(N), 2):
            if set(A) & set(B):
                continue
            crossings = np.array([np.cross(lines[a], lines[b]) for a in A for b in B])  # a1b1, a1b2, a2b1, a2b2
            if np.abs(crossings[:, 2]).min() < 1e-9:
                continue
            uv = crossings[:, :2] / crossings[:, 2:]
            if not np.isfinite(uv).all() or np.abs(uv).max() > 20 * max(w, h):
                continue
            out.append(_dlt4(Wp, uv))
    if not out:
        return np.zeros((0, 3, 3))
    return _filter(np.concatenate(out), w, h)


def _dlt4(Wp, uv):
    """Wp (n, 4, 2) world, uv (4, 2) image (the same 4 points for all n) -> H (n, 3, 3)."""
    n = len(Wp)
    A = np.zeros((n, 8, 8))
    b = np.zeros((n, 8))
    for k in range(4):
        X, Y = Wp[:, k, 0], Wp[:, k, 1]
        u, v = uv[k]
        A[:, 2 * k, 0], A[:, 2 * k, 1], A[:, 2 * k, 2] = X, Y, 1
        A[:, 2 * k, 6], A[:, 2 * k, 7] = -u * X, -u * Y
        A[:, 2 * k + 1, 3], A[:, 2 * k + 1, 4], A[:, 2 * k + 1, 5] = X, Y, 1
        A[:, 2 * k + 1, 6], A[:, 2 * k + 1, 7] = -v * X, -v * Y
        b[:, 2 * k], b[:, 2 * k + 1] = u, v
    det = np.linalg.det(A)
    bad = ~(np.abs(det) > 1e-6 * np.abs(det).max() + 1e-300)
    A[bad] = np.eye(8)
    h8 = np.linalg.solve(A, b[..., None])[..., 0]
    H = np.concatenate([h8, np.ones((n, 1))], 1).reshape(n, 3, 3)
    H[bad] = np.nan
    return H


def _filter(Hs, w, h):
    """Drop NaNs, mirrored fields, H with the field centre behind the camera and those that no reasonable
    pinhole camera can produce."""
    Hs = Hs[np.isfinite(Hs).all((1, 2))]
    if len(Hs) == 0:
        return Hs
    # sign: world point (0, 0) is in front of the camera when depth > 0; probe with 3 points
    c = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    _, depth = project(Hs, c)
    s = np.sign(depth[:, 0])
    s[s == 0] = 1
    Hs = Hs * s[:, None, None]
    xy, depth = project(Hs, c)
    # orientation: +X to the image right and +Y UP (v decreases): cross product < 0. The field is equal under
    # a 180-degree turn, so H and H@Rot180 draw the SAME lines with the SAME score; without this every solution
    # is duplicated and sometimes the twin wins, with every template point on its antipode
    dx, dy = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    cross = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    ok = np.isfinite(cross) & (cross < 0) & (depth > 0).all(1) & (dy[:, 1] < 0)
    Hs = Hs[ok]
    return Hs[is_pinhole_consistent(Hs, w, h)]


# ---------------------------------------------------------------- evaluation

def reprojection_error(H_est, H_gt, w, h, step=1.0):
    """Median distance (px) between where H_est and H_gt project the template points that H_gt puts
    INSIDE the image. 1e4 if fewer than 5 qualify."""
    pts, _ = sample_template(step)
    xy_g, depth_g = project(H_gt[None], pts)
    xy_e, depth_e = project(H_est[None], pts)
    ok = ((depth_g[0] > 0) & (xy_g[0, :, 0] >= 0) & (xy_g[0, :, 0] < w)
          & (xy_g[0, :, 1] >= 0) & (xy_g[0, :, 1] < h))
    if ok.sum() < 5:
        return 1e4
    e = np.hypot(*(xy_e[0, ok] - xy_g[0, ok]).T)
    e = np.where((depth_e[0, ok] > 0) & np.isfinite(e), e, 1e4)
    return float(np.median(e))
