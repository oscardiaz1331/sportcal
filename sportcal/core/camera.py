"""Pinhole-camera model for a planar field, batched over hypotheses.

A pose is (cx, cy, cz, pan, tilt, f): camera position in metres (Z up), pan/tilt in radians and
focal length in pixels; no roll, principal point at the image centre. The camera looks towards +Y
from the Y<0 side, so "up" in the image is +Y. H maps world (X, Y, 1) to image pixels.

These functions work on arrays of hypotheses so a search can score thousands of poses at once.
"""
import numpy as np


def pose_to_H(p, w, h):
    """Poses (n, 6) -> homographies (n, 3, 3), with positive homogeneous depth in front of the camera."""
    p = np.atleast_2d(p).astype(float)
    cx, cy, cz, pan, tilt, f = p.T
    sp, cp, st, ct = np.sin(pan), np.cos(pan), np.sin(tilt), np.cos(tilt)
    z = np.zeros_like(pan)
    right = np.stack([cp, -sp, z], 1)
    forward = np.stack([sp * ct, cp * ct, -st], 1)
    down = np.stack([-st * sp, -st * cp, -ct], 1)
    R = np.stack([right, down, forward], 1)  # (n, 3, 3): rows right, down, forward
    C = np.stack([cx, cy, cz], 1)
    t = -np.einsum("nij,nj->ni", R, C)
    M = np.stack([R[:, :, 0], R[:, :, 1], t], 2)  # plane Z=0: drop the third column of R
    K = np.zeros((len(p), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2.0, h / 2.0, 1.0
    return K @ M


def project(H, pts):
    """H (n, 3, 3), world points (m, 2) -> (xy (n, m, 2), depth (n, m))."""
    P = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    q = np.einsum("nij,mj->nmi", H, P)
    depth = q[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[..., :2] / depth[..., None]
    return xy, depth


def bilinear(img, xy):
    """Bilinear sample of a 2-D map at float pixel positions (clamped to the image)."""
    x = np.clip(xy[..., 0], 0, img.shape[1] - 1.001)
    y = np.clip(xy[..., 1], 0, img.shape[0] - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - x0, y - y0
    a, b, c, d = img[y0, x0], img[y0, x0 + 1], img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def pinhole_residual(Hs, w, h):
    """(f2, mismatch) per homography: how far it is from ANY square-pixel pinhole camera.

    For H = K [r1 r2 t] with K having the principal point at the centre, the columns' orthogonality
    gives f^2 = -a1.a2 / (z1 z2) (a_i: the x,y part of column i relative to the centre, z_i its third
    component). `mismatch` in [0, 1] measures how much the norms of K^-1 h1 and K^-1 h2 differ (they
    must be equal); it is 1 when f2 is not a valid focal length (outside 0.4-8 image widths).
    A free 8-parameter H can score as well as the truth while deforming into something no camera
    produces, so searches penalise this mismatch.
    """
    a1 = Hs[:, :2, 0] - np.array([w / 2.0, h / 2.0]) * Hs[:, 2:3, 0]
    a2 = Hs[:, :2, 1] - np.array([w / 2.0, h / 2.0]) * Hs[:, 2:3, 1]
    z1, z2 = Hs[:, 2, 0], Hs[:, 2, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = -(a1 * a2).sum(1) / (z1 * z2)
        n1 = (a1 ** 2).sum(1) / f2 + z1 ** 2
        n2 = (a2 ** 2).sum(1) / f2 + z2 ** 2
        mismatch = np.abs(n1 - n2) / (n1 + n2)
    valid = np.isfinite(f2) & np.isfinite(mismatch) & (f2 > (0.4 * w) ** 2) & (f2 < (8 * w) ** 2)
    return f2, np.where(valid, np.clip(mismatch, 0, 1), 1.0)


def is_pinhole_consistent(Hs, w, h, tol=0.4):
    """Bool (n,): the homography can come from a reasonable pinhole camera."""
    return pinhole_residual(Hs, w, h)[1] < tol


def decompose_H(H, w, h):
    """Split ONE field homography into a pinhole pose, the inverse of `pose_to_H`.

    Returns {"f", "C", "pan", "tilt", "mismatch"}: focal length in pixels, camera centre (3,) in the units of the world,
    pan and tilt in radians, and the pinhole mismatch (0 = exactly a pinhole camera). None when H is not a pinhole camera
    at all (no real focal length). The focal length comes from the orthogonality of the plane's axes, so it is poorly
    conditioned when the view holds few primitives: the centre of such a frame can be wildly off even if the projection
    looks right, which is why a centre is best calibrated from several rich frames of the same camera."""
    H = np.asarray(H, float)
    f2, mismatch = pinhole_residual(H[None], w, h)
    if not np.isfinite(f2[0]) or f2[0] <= 0:
        return None
    f = float(np.sqrt(f2[0]))
    B = np.linalg.inv(np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])) @ H
    lam = 1.0 / np.linalg.norm(B[:, 0])
    r1, r2, t = lam * B[:, 0], lam * B[:, 1], lam * B[:, 2]
    if t[2] < 0:                                      # the plane must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, np.cross(r1, r2)], 1))
    R = U @ Vt                                        # rows: right, down, forward (as in pose_to_H)
    forward = R[2, :]
    return {"f": f, "C": -R.T @ t, "pan": float(np.arctan2(forward[0], forward[1])), "tilt": float(np.arcsin(-forward[2])),
            "mismatch": float(mismatch[0])}

def project_distorted(pose, k1, k2, world_pts, w, h):
    """Like project(pose_to_H(pose), world_pts), but with radial distortion (k1, k2) applied in normalized
    camera coordinates before K. k1, k2: scalar or (n,), one pair per pose. Returns (xy (n, m, 2), depth (n, m)),
    same shape convention as project(). No H exists once k1 or k2 is nonzero, so this does not go through pose_to_H."""
    pose = np.atleast_2d(pose).astype(float)
    k1, k2 = np.atleast_1d(k1).astype(float), np.atleast_1d(k2).astype(float)   # accept a scalar or one per pose
    cx, cy, cz, pan, tilt, f = pose.T
    sp, cp, st, ct = np.sin(pan), np.cos(pan), np.sin(tilt), np.cos(tilt)
    z = np.zeros_like(pan)
    right = np.stack([cp, -sp, z], 1)
    forward = np.stack([sp * ct, cp * ct, -st], 1)
    down = np.stack([-st * sp, -st * cp, -ct], 1)
    R = np.stack([right, down, forward], 1)  # (n, 3, 3): rows right, down, forward
    C = np.stack([cx, cy, cz], 1)
    t = -np.einsum("nij,nj->ni", R, C)
    P = np.asarray(world_pts, float)                       # (m, 2) = [X, Y]; Z=0, so only r1, r2 matter, t added below
    q = np.einsum("nij,mj->nmi", R[:, :, :2], P) + t[:, None]
    x = q[..., 0] / q[..., 2]
    y = q[..., 1] / q[..., 2]
    r2 = x ** 2 + y ** 2
    radial_distortion = 1 + k1[:, None] * r2 + k2[:, None] * r2 ** 2
    x_distorted = x * radial_distortion
    y_distorted = y * radial_distortion
    u = f[:, None] * x_distorted + w / 2.0
    v = f[:, None] * y_distorted + h / 2.0
    return (np.stack([u, v], axis=-1), q[..., 2])  # (n, m, 2), (n, m)


def ptz_warp(H, w, h, pan=0.0, tilt=0.0, roll=0.0, zoom=1.0):
    """(G, G @ H): the image homography G of the SAME camera turned about its own centre (radians; pan about the image
    vertical, tilt > 0 looks down as in pose_to_H, roll about the optical axis) and zoomed by `zoom`, and the field
    homography of the new view. Turning a camera about its centre moves every pixel by K' R K^-1 whatever the depth of the
    scene, so a real frame warped by G, labelled with G @ H, is an exact new view - players, boards and stands included -
    not an approximation; only the frame border goes missing. The focal length comes from `decompose_H`.
    ponytail: principal point at the centre and no distortion, like the rest of this module; an H that is no pinhole
    camera falls back to f = w, which keeps the label exact and only makes the turn slightly unphysical."""
    pose = decompose_H(H, w, h)
    f = pose["f"] if pose else float(w)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    Kz = np.array([[f * zoom, 0, w / 2.0], [0, f * zoom, h / 2.0], [0, 0, 1.0]])
    c, s = np.cos, np.sin
    Rx = np.array([[1, 0, 0], [0, c(tilt), -s(tilt)], [0, s(tilt), c(tilt)]])
    Ry = np.array([[c(pan), 0, s(pan)], [0, 1, 0], [-s(pan), 0, c(pan)]])
    Rz = np.array([[c(roll), -s(roll), 0], [s(roll), c(roll), 0], [0, 0, 1]])
    G = Kz @ Rz @ Rx @ Ry @ np.linalg.inv(K)
    return G, G @ np.asarray(H, float)


def is_plausible_view(H, w, h, field_box, n=(25, 11)):
    """False when no real camera can produce the field homography H: part of the field lies behind the camera AND H has
    no valid focal length (`pinhole_residual`). field_box: (x0, x1, y0, y1), the world extent of the field.

    A free 8-parameter H can fit its points while folding the plane through the horizon, so that, drawn, a line crosses a
    circle. Neither sign alone is enough: a real camera panned along the field has part of it behind, and the focal length
    of a real view is poorly conditioned when few primitives are seen (`decompose_H`), so correct hand labels with no valid
    focal, or an implausibly short one, exist. Both together are the tell. Measured: docs/experiments/hockey.md section 14d.
    ponytail: a sampled grid of the field box, not an exact horizon-vs-box test; a sliver behind the camera that falls
    between samples goes unseen."""
    H = np.asarray(H, float)
    x0, x1, y0, y1 = field_box
    X, Y = np.meshgrid(np.linspace(x0, x1, n[0]), np.linspace(y0, y1, n[1]))
    z = X.ravel() * H[2, 0] + Y.ravel() * H[2, 1] + H[2, 2]
    behind = bool((np.sign(z) != np.sign(np.median(z))).any())
    return not (behind and pinhole_residual(H[None], w, h)[1][0] >= 1.0)


def view_angle(H, w, h):
    """Angle in degrees (0-90) between the image horizontal and the field's long axis (world X) at the image centre: small
    for the usual side camera, large when the camera looks down the length of the field."""
    H = np.asarray(H, float)
    c = np.linalg.solve(H, [w / 2.0, h / 2.0, 1.0])
    c = c[:2] / c[2]
    a, b = (H @ [c[0], c[1], 1.0]), (H @ [c[0] + 0.5, c[1], 1.0])
    d = b[:2] / b[2] - a[:2] / a[2]
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))


def canonical_mirror(H, centre, y_down=True, end_angle=45.0):
    """(H @ S, flips): of the 4 namings of a field that is symmetric about x = centre[0] and about y = centre[1] (each
    mirror draws the same lines and renames every point), the one an image rule picks, so that the same view always gets
    the same names - a model of named points cannot learn names the labels give at random. flips: "", "x", "y" or "xy".

    The rule looks at where the world axes point in the image at the field centre. Side view (the long axis, world x,
    within `end_angle` degrees of the image horizontal): +x points right, +y points down (`y_down`) or up. End view: +x
    points down, i.e. the end the camera sits behind is the high-x one, and +y points left - the side rule turned a
    quarter, so the two agree for a far end at the upper left. It names the view, not the arena: which physical end or
    side a frame shows needs context the frame does not have.
    ponytail: a view whose long axis sits near `end_angle` can switch names between two nearly equal frames; broadcast
    views are far from it (hockey.md section 14g)."""
    H = np.asarray(H, float)
    cx, cy = centre

    def image_step(ex, ey):
        a, b = H @ [cx, cy, 1.0], H @ [cx + ex, cy + ey, 1.0]
        return b[:2] / b[2] - a[:2] / a[2]
    dx, dy = image_step(1.0, 0.0), image_step(0.0, 1.0)
    if np.degrees(np.arctan2(abs(dx[1]), abs(dx[0]))) >= end_angle:
        fx, fy = dx[1] < 0, dy[0] > 0
    else:
        fx, fy = dx[0] < 0, (dy[1] < 0) == y_down
    S = np.array([[-1.0 if fx else 1.0, 0, 2 * cx if fx else 0], [0, -1.0 if fy else 1.0, 2 * cy if fy else 0], [0, 0, 1]])
    Hc = H @ S
    return Hc / Hc[2, 2], "x" * bool(fx) + "y" * bool(fy)


def robust_centre(centres, max_dev=4.0):
    """(median centre, per-axis spread, number of votes) over an array of (n, 3) camera centres, or None when empty.

    A fixed broadcast camera gives one centre per decomposed frame (`decompose_H`); frames with few primitives vote wild
    centres. Votes farther than `max_dev` (world units) from the median of all are dropped once, then the median and the
    median absolute deviation (scaled to a standard deviation) are recomputed on the rest."""
    C = np.asarray(centres, float).reshape(-1, 3)
    if len(C) == 0:
        return None
    keep = np.linalg.norm(C - np.median(C, axis=0), axis=1) <= max_dev
    C = C[keep] if keep.any() else C
    med = np.median(C, axis=0)
    return med, 1.4826 * np.median(np.abs(C - med), axis=0), len(C)
