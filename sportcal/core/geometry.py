"""Planar-homography geometry: projection helpers and a DLT solver for point AND line
correspondences, followed by a non-linear reprojection refinement.

Everything here is sport-agnostic: the world is a plane in metres, the image is in
pixels, and H maps world (X, Y, 1) to image (x, y, 1).

Why the solver looks the way it does (each item was a real bug, measured in
docs/experiments/hockey.md section 6):
  * inputs are Hartley-normalised (`scale_transform`) - un-normalised DLT is numerically useless;
  * a line correspondence is used as a line (dual DLT), never sampled into fake points;
  * degenerate configurations are rejected by the SVD condition number;
  * the refinement runs in two passes (linear loss, then robust loss).
"""
import cv2
import numpy as np

REF_WIDTH = 1920.0  # pixel errors are reported as if the frame were this wide


# --------------------------------------------------------------- projection helpers

def project_points(H, world, w, h):
    """Project world points through H. Returns (xy, ok).

    `ok` marks the drawable points. A homography also projects points behind the
    camera: their third coordinate flips sign and they reappear on the wrong side of
    the image plane, which would draw a polyline crossing the horizon as a bogus
    straight line. Points far outside the frame are dropped too (int32 overflow in
    cv2.polylines, and they carry no information).
    """
    P = np.c_[world, np.ones(len(world))] @ H.T
    z = P[:, 2]
    finite = np.isfinite(z) & (np.abs(z) > 1e-9)
    xy = np.full((len(world), 2), np.nan)
    if not finite.any():
        return xy, np.zeros(len(world), bool)
    sign = np.sign(np.median(z[finite])) or 1.0
    ok = finite & (z * sign > 1e-9)
    xy[ok] = P[ok, :2] / z[ok, None]
    ok &= np.isfinite(xy).all(1) & (np.abs(xy[:, 0]) < 4 * w) & (np.abs(xy[:, 1]) < 4 * h)
    return xy, ok


def in_front(H, world):
    """Boolean mask of world points that lie in front of the camera."""
    w = world @ H[2, :2] + H[2, 2]
    if not len(w):
        return np.zeros(0, bool)
    sign = np.sign(np.median(w[np.abs(w) > 1e-9])) or 1.0
    return w * sign > 1e-9


def geom_error(Ha, Hb, points, w, h):
    """Median pixel distance between two homographies over the template points that
    Hb places inside the frame (scaled to REF_WIDTH). inf if fewer than 4 qualify."""
    xa, oka = project_points(Ha, points, w, h)
    xb, okb = project_points(Hb, points, w, h)
    inb = oka & okb & (xb[:, 0] > 0) & (xb[:, 0] < w) & (xb[:, 1] > 0) & (xb[:, 1] < h)
    if inb.sum() < 4:
        return np.inf
    return float(np.median(np.linalg.norm(xa[inb] - xb[inb], axis=1)) * REF_WIDTH / w)


# --------------------------------------------------------------- DLT

def dlt_point_rows(Xw, Yw, xi, yi):
    """The two standard DLT rows of a point correspondence x_i ~ H [Xw, Yw, 1]^T."""
    return np.array([
        [Xw, Yw, 1, 0, 0, 0, -xi * Xw, -xi * Yw, -xi],
        [0, 0, 0, Xw, Yw, 1, -yi * Xw, -yi * Yw, -yi],
    ])


def dlt_line_rows(a, b, c, p, q, r):
    """Two DLT rows for a world line (a, b, c) <-> image line (p, q, r), from
    l_w ~ H^T l_i.

    Same cross-product trick as the point DLT (l_w x (H^T l_i) = 0), written in the
    coefficients of H (not H^T) so the rows can be stacked with the point rows and a
    single h solved for.
    """
    return np.array([
        [0, -c * p, b * p, 0, -c * q, b * q, 0, -c * r, b * r],
        [c * p, 0, -a * p, c * q, 0, -a * q, c * r, 0, -a * r],
    ])


def scale_transform(ref):
    """Isotropic T mapping [0, ref] to [-1, 1] (Hartley 2004, 4.4.4).

    Without it the DLT mixes rows of magnitude ~1000 (image px) with ~10-60 (world
    metres) and the SVD null vector is numerically useless.
    """
    s = 2.0 / ref
    return np.array([[s, 0, -1], [0, s, -1], [0, 0, 1]])


def transform_point(T, X, Y):
    x, y, w = T @ [X, Y, 1.0]
    return x / w, y / w


def transform_line(T, a, b, c):
    """Lines transform with the inverse transpose of the point transform."""
    return np.linalg.inv(T).T @ [a, b, c]


def _null_homography(rows):
    """Row-normalise, SVD, return (H normalised so H[2,2]=1, singular values)."""
    A = np.vstack(rows).astype(np.float64)
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / np.where(norm > 1e-12, norm, 1.0)
    _, S, VT = np.linalg.svd(A)
    H = VT[-1].reshape(3, 3)
    return H / H[2, 2], S


def solve_dlt(rows, return_cond=False):
    """Solve stacked DLT rows for H (H[2,2]=1).

    With return_cond, also returns S[-1]/S[-2]: a well-posed system has ONE
    near-zero singular value well separated from the next. If the last two are
    similar the null space is not isolated (typical of parallel lines + 1 point:
    8 dof by count, but the parallel family adds redundant information) and small
    noise blows up into a completely different H.
    """
    H, S = _null_homography(rows)
    if not return_cond:
        return H
    return H, S[-1] / (S[-2] or 1e-12)


def _correspondence_rows(points, lines, T_world, T_img):
    """DLT rows of all correspondences, in normalised coordinates.

    points: [(Xw, Yw, xi, yi), ...]   lines: [(a, b, c, p, q, r), ...]  (un-normalised)
    """
    rows = []
    for Xw, Yw, xi, yi in points:
        rows += list(dlt_point_rows(*transform_point(T_world, Xw, Yw),
                                    *transform_point(T_img, xi, yi)))
    for a, b, c, p, q, r in lines:
        rows += list(dlt_line_rows(*transform_line(T_world, a, b, c),
                                   *transform_line(T_img, p, q, r)))
    return rows


def solve_dlt_correspondences(points, lines, T_world, T_img, max_cond=0.05):
    """Normalise, solve, de-normalise. None if the configuration is too degenerate."""
    Hn, cond = solve_dlt(_correspondence_rows(points, lines, T_world, T_img), return_cond=True)
    if cond > max_cond:
        return None
    H = np.linalg.inv(T_img) @ Hn @ T_world
    return H / H[2, 2]


def solve_dlt_with_residual(points, lines, T_world, T_img):
    """Like solve_dlt_correspondences but also returns the algebraic residual S[-1].

    Note: the algebraic residual does NOT discriminate good from bad candidates (it
    picked the right board sign 0/27 times); use the refinement cost instead.
    """
    rows = _correspondence_rows(points, lines, T_world, T_img)
    if len(rows) < 8:
        return None, np.inf
    Hn, S = _null_homography(rows)
    H = np.linalg.inv(T_img) @ Hn @ T_world
    return H / H[2, 2], float(S[-1])


# --------------------------------------------------------------- non-linear refinement

def sample_world_line(a, b, c, L, W, n=5):
    """Points along the world line (a, b, c) over its REAL extent on the field
    (X=const for y in [0, W], or Y=const for x in [0, L])."""
    if abs(a) >= abs(b):
        x0 = -c / a
        return [(x0, t) for t in np.linspace(0, W, n)]
    y0 = -c / b
    return [(t, y0) for t in np.linspace(0, L, n)]


def refine_residuals(h8, points_n, line_samples_n):
    """Point reprojection errors, then point-to-line distances for sampled line points."""
    H = np.array(list(h8) + [1.0]).reshape(3, 3)
    res = []
    for Xn, Yn, xn, yn in points_n:
        proj = H @ [Xn, Yn, 1.0]
        px, py = proj[:2] / proj[2]
        res.append(px - xn)
        res.append(py - yn)
    for (pn, qn, rn), pts in line_samples_n:
        norm_pq = np.hypot(pn, qn) or 1.0
        for Xn, Yn in pts:
            proj = H @ [Xn, Yn, 1.0]
            px, py = proj[:2] / proj[2]
            res.append((pn * px + qn * py + rn) / norm_pq)
    return np.array(res)


def refine_nonlinear(H0, points, lines, T_world, T_img, L, W):
    """Minimise the REAL reprojection distance starting from the DLT seed H0.

    Returns (H, median final residual in normalised space). The residual measures
    geometric consistency and is the best available confidence signal (see
    docs/experiments/hockey.md section 6 and ADR 0002).

    Two passes, on purpose: a robust loss with a badly calibrated f_scale saturates
    the gradient at the seed and the optimiser "converges" without moving. So pass 1
    is plain least squares (full gradient over the whole range); pass 2 is soft_l1
    with f_scale calibrated to the residual already near the optimum.
    """
    from scipy.optimize import least_squares

    points_n = [(*transform_point(T_world, Xw, Yw), *transform_point(T_img, xi, yi))
                for Xw, Yw, xi, yi in points]
    line_samples_n = []
    for a, b, c, p, q, r in lines:
        pts_n = [transform_point(T_world, Xs, Ys) for Xs, Ys in sample_world_line(a, b, c, L, W)]
        line_samples_n.append((tuple(transform_line(T_img, p, q, r)), pts_n))

    Hn0 = T_img @ H0 @ np.linalg.inv(T_world)
    h8_0 = (Hn0 / Hn0[2, 2]).flatten()[:8]
    args = (points_n, line_samples_n)
    sol = least_squares(refine_residuals, h8_0, args=args, loss="linear", method="trf",
                        max_nfev=3000, xtol=1e-12, ftol=1e-12)
    f_scale = max(0.01, 1.5 * np.median(np.abs(refine_residuals(sol.x, *args))))
    sol = least_squares(refine_residuals, sol.x, args=args, loss="soft_l1", f_scale=f_scale,
                        method="trf", max_nfev=2000)
    Hn = np.array(list(sol.x) + [1.0]).reshape(3, 3)
    H = np.linalg.inv(T_img) @ Hn @ T_world
    return H / H[2, 2], float(np.median(np.abs(refine_residuals(sol.x, *args))))


# --------------------------------------------------------------- fitting and inversion

def fit_homography_ransac(world_xy, img_xy, ransac_px):
    """World (m) -> image (px) homography by RANSAC. Returns (H, inlier mask) or (None, None)."""
    if len(world_xy) < 4:
        return None, None
    H, mask = cv2.findHomography(world_xy.astype(np.float32), img_xy.astype(np.float32),
                                 cv2.RANSAC, ransac_px)
    if H is None:
        return None, None
    return H, mask.ravel().astype(bool)


def image_to_world(H, img_xy):
    """Back-project image pixels onto the field plane (metres). Only meaningful for pixels
    below the horizon; the caller decides which detections to trust."""
    pts = np.asarray(img_xy, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)


def line_through(p, q):
    """Homogeneous line (a, b, c) through two 2-D points."""
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def solve_points_lines(world_pts, img_pts, world_lines, img_lines, ref_world, ref_img, ransac_px):
    """World -> image H from matched points and lines, or None. RANSAC over the points drops their outliers, then one
    Hartley-normalised DLT takes the inlier points AND the lines (as lines - the dual DLT - never sampled into fake
    points). The joint H replaces the points-only one unless it fits the inlier points clearly worse: a wrong line.
    ponytail: lines are only vetoed as a group, never RANSAC'd one by one; per-line consensus is the upgrade."""
    wp = np.asarray(world_pts, float).reshape(-1, 2)
    ip = np.asarray(img_pts, float).reshape(-1, 2)
    H_pts, inl = fit_homography_ransac(wp, ip, ransac_px)
    if inl is not None:
        wp, ip, H_pts = wp[inl], ip[inl], H_pts / H_pts[2, 2]
    rows = [(*a, *b) for a, b in zip(wp, ip)]
    lines = [(*a, *b) for a, b in zip(world_lines, img_lines)]
    H_joint = None
    if len(rows) + len(lines) >= 4:
        H_joint = solve_dlt_correspondences(rows, lines, scale_transform(ref_world), scale_transform(ref_img))
    if H_joint is None or H_pts is None:
        return H_pts if H_joint is None else H_joint

    def resid(H):
        q = np.c_[wp, np.ones(len(wp))] @ H.T
        return np.median(np.linalg.norm(q[:, :2] / q[:, 2:] - ip, axis=1))
    return H_joint if resid(H_joint) <= resid(H_pts) + ransac_px / 2 else H_pts
