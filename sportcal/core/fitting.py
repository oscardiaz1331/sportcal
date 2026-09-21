"""Robust line and circle fitting on a binary mask. Pure cv2/numpy: no torch, no model.

The functions do not care where the mask comes from (a segmentation network or a
classical colour threshold), only that it is a 2-D boolean / 0-1 array.
"""
import cv2
import numpy as np

MIN_LINE_PX = 150
MIN_CIRCLE_PX = 80
MAX_ANGLE_GAP_DEG = 150.0  # the visible arc must span at least 360 - this many degrees
BORDER_MARGIN_PX = 3


def _line_from_fit(pts):
    """Unit-normalised implicit line (p, q, r), p*x + q*y + r = 0, least-squares through pts."""
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    p, q = vy, -vx
    r = -(p * x0 + q * y0)
    return np.array([p, q, r]) / (np.hypot(p, q) or 1.0)


def skeleton_points(mask, n_bins=24):
    """Centre points of the mask, evenly spread along its long axis, or None.

    Do NOT fit the line to raw pixels: a painted stripe is much wider (more pixels)
    on the camera side than on the far side, so an unweighted fit lets the near side
    dominate the angle and the slope error explodes when extrapolated. Binning along
    the long axis and taking the median of the short axis makes every stretch of the
    curve weigh by LENGTH, not area.
    """
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return None
    horizontal = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    along, across = (xs, ys) if horizontal else (ys, xs)
    edges = np.linspace(along.min(), along.max(), n_bins + 1)
    idx = np.clip(np.digitize(along, edges) - 1, 0, n_bins - 1)
    pts = []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 3:
            continue
        a_med, c_med = np.median(along[sel]), np.median(across[sel])
        pts.append((a_med, c_med) if horizontal else (c_med, a_med))
    return np.array(pts, np.float32) if len(pts) >= 4 else None


def fit_line_px(mask):
    """Image line (p, q, r) fitted to a class's skeleton points, or None."""
    if int(mask.sum()) < MIN_LINE_PX:
        return None
    pts = skeleton_points(mask)
    return None if pts is None else _line_from_fit(pts)


def fit_line_ransac(mask, tol_px=4.0, n_iter=400, min_inlier_frac=0.25, seed=0):
    """Dominant line in a MIXED mask (e.g. boards: 2 straight sides + 4 corner arcs).

    Classic RANSAC followed by a least-squares refit on the inliers. `fit_line_px`
    assumes one clean line and would blend the sides and corners into nonsense.
    """
    ys, xs = np.where(mask)
    n = len(xs)
    if n < MIN_LINE_PX:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float64)
    rng = np.random.default_rng(seed)
    sample = pts if n <= 4000 else pts[rng.choice(n, 4000, replace=False)]
    best_n, best_line = 0, None
    for _ in range(n_iter):
        i, j = rng.choice(len(sample), 2, replace=False)
        (x1, y1), (x2, y2) = sample[i], sample[j]
        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)
        if norm < 5:
            continue
        p, q = dy / norm, -dx / norm
        r = -(p * x1 + q * y1)
        n_in = int((np.abs(p * pts[:, 0] + q * pts[:, 1] + r) < tol_px).sum())
        if n_in > best_n:
            best_n, best_line = n_in, (p, q, r)
    if best_line is None or best_n < max(30, min_inlier_frac * n):
        return None
    p, q, r = best_line
    inliers = pts[np.abs(p * pts[:, 0] + q * pts[:, 1] + r) < tol_px].astype(np.float32)
    return _line_from_fit(inliers)


def fit_circle_center(mask, w=None, h=None):
    """Centre of an ellipse fitted to the mask, or None if the visible arc is untrustworthy.

    A circle whose centre is off-frame shows only a partial arc, often cut by the
    image border; fitting an ellipse to that is a classic ill-conditioned problem
    (centres hundreds of px off). Two cheap filters catch it without knowing H:
    the arc must wrap around the fitted centre over a wide angular range, and a mask
    touching the image border is rejected as clipped.
    """
    ys, xs = np.where(mask)
    if len(xs) < MIN_CIRCLE_PX:
        return None
    if w is not None and h is not None:
        if xs.min() <= BORDER_MARGIN_PX or xs.max() >= w - 1 - BORDER_MARGIN_PX:
            return None
        if ys.min() <= BORDER_MARGIN_PX or ys.max() >= h - 1 - BORDER_MARGIN_PX:
            return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (cx, cy), _, _ = cv2.fitEllipse(pts)
    angles = np.sort(np.degrees(np.arctan2(ys - cy, xs - cx)))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360]))
    if gaps.max() > MAX_ANGLE_GAP_DEG:
        return None
    return np.array([cx, cy])


def detect_lines(mask, n_max=7):
    """Dominant straight lines of a mask: (n, 3) rows (a, b, c) with a*x + b*y + c = 0, a^2 + b^2 = 1.

    Probabilistic Hough -> segments grouped by angle and offset (longest first) -> one least-squares
    line per group, returned ordered by total segment length. Thresholds scale with the image width.
    """
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, int(0.03 * w),
                           minLineLength=int(0.06 * w), maxLineGap=int(0.02 * w))
    if segs is None:
        return np.zeros((0, 3))
    S = segs.reshape(-1, 4).astype(float)
    length = np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1])
    angle = np.arctan2(S[:, 3] - S[:, 1], S[:, 2] - S[:, 0]) % np.pi
    free = np.ones(len(S), bool)
    groups = []
    for i in np.argsort(-length):
        if not free[i]:
            continue
        n = np.array([-np.sin(angle[i]), np.cos(angle[i])])
        c = -(n @ S[i, :2])
        d1 = np.abs(S[:, :2] @ n + c)
        d2 = np.abs(S[:, 2:] @ n + c)
        d_angle = np.abs((angle - angle[i] + np.pi / 2) % np.pi - np.pi / 2)
        member = free & (d1 < 0.012 * w) & (d2 < 0.012 * w) & (d_angle < np.radians(4))
        free &= ~member
        pts = np.concatenate([S[member][:, :2], S[member][:, 2:]]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        nn = np.array([-vy, vx])
        groups.append((length[member].sum(), np.array([nn[0], nn[1], -(nn @ [x0, y0])])))
    groups.sort(key=lambda g: -g[0])
    return np.array([g[1] for g in groups[:n_max]]) if groups else np.zeros((0, 3))
