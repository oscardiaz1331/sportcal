"""Frame-to-frame image motion of a static, distant background.

A camera that only pans, tilts and zooms about a fixed centre relates any two frames by a homography
(K2 R K1^-1) whatever the scene depth, so a distant textured background (stands, advertising boards)
gives the image-to-image motion even when the playing surface has almost no usable primitives.

Pipeline: pick corners inside a mask -> track them forward AND backward with pyramidal Lucas-Kanade ->
drop tracks whose round trip does not come back to the start (unstable) -> fit one motion model with
RANSAC, which keeps the points that move consistently (players, static overlays and animated boards
move differently and fall out as outliers).

Results and thresholds that worked live in docs/experiments/, not here.
"""
import cv2
import numpy as np

MODELS = ("homography", "affine", "similarity")
_MIN_POINTS = {"homography": 4, "affine": 3, "similarity": 2}
_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def select_features(gray, mask=None, max_corners=800, quality=0.01, min_distance=8, border=8):
    """Shi-Tomasi corners of `gray` (uint8) restricted to `mask` (nonzero = allowed), (n, 2) float32.

    `border` pixels around the image are excluded so a track can not leave the frame in the first step."""
    allowed = np.full(gray.shape, 255, np.uint8) if mask is None else ((np.asarray(mask) > 0).astype(np.uint8) * 255)
    if border > 0:
        allowed[:border], allowed[-border:], allowed[:, :border], allowed[:, -border:] = 0, 0, 0, 0
    pts = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_distance, mask=allowed)
    return np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2).astype(np.float32)


def track_forward_backward(gray0, gray1, pts, win=21, levels=3, max_fb_error=1.0):
    """Track `pts` from gray0 to gray1 and back. Returns {"p1", "fb_error", "ok"}.

    fb_error is the distance between a point and where it lands after the round trip; `ok` also requires
    both passes to succeed and the forward result to stay inside the frame."""
    n = len(pts)
    if n == 0:
        return {"p1": np.zeros((0, 2), np.float32), "fb_error": np.zeros(0), "ok": np.zeros(0, bool)}
    lk = dict(winSize=(int(win), int(win)), maxLevel=int(levels), criteria=_LK_CRITERIA)
    p0 = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    p1, st_f, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk)
    p0_back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **lk)
    p1 = p1.reshape(-1, 2)
    fb = np.linalg.norm(p0_back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
    h, w = gray1.shape[:2]
    inside = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    ok = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb <= max_fb_error) & inside
    return {"p1": p1, "fb_error": fb, "ok": ok}


def apply_motion(M, pts):
    """Apply a 3x3 motion to (n, 2) points."""
    q = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(pts))] @ np.asarray(M, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def fit_motion(p0, p1, model="homography", ransac_thresh=3.0):
    """RANSAC fit of the motion p0 -> p1. Returns (M 3x3 or None, inliers bool per input point)."""
    if model not in MODELS:
        raise ValueError("model must be one of {}".format(MODELS))
    p0, p1 = np.asarray(p0, np.float32).reshape(-1, 2), np.asarray(p1, np.float32).reshape(-1, 2)
    none = np.zeros(len(p0), bool)
    if len(p0) < max(2 * _MIN_POINTS[model], 8):
        return None, none
    if model == "homography":
        M, inl = cv2.findHomography(p0, p1, cv2.RANSAC, ransac_thresh, maxIters=3000, confidence=0.999)
    else:
        fit = cv2.estimateAffine2D if model == "affine" else cv2.estimateAffinePartial2D
        A, inl = fit(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=3000, confidence=0.999)
        M = None if A is None else np.vstack([A, [0.0, 0.0, 1.0]])
    if M is None or inl is None or not np.all(np.isfinite(M)):
        return None, none
    M = M / M[2, 2]
    return M, np.linalg.norm(apply_motion(M, p0) - p1, axis=1) <= ransac_thresh


def estimate_motion(gray0, gray1, mask=None, model="homography", max_corners=800, quality=0.01, min_distance=8,
                    win=21, levels=3, max_fb_error=1.0, ransac_thresh=3.0, border=8, min_inliers=30):
    """Motion of the masked background from gray0 to gray1 (uint8 images of the same size).

    Returns a dict: "M" (3x3, maps frame-0 pixels to frame-1 pixels, or None when there were too few
    consistent tracks), "p0"/"p1" (every selected point and where it landed), "fb_error", "tracked" (survived
    the forward-backward check), "inliers" (also survived RANSAC), the counts "n_selected"/"n_tracked"/"n_inliers"
    and "median_residual" (pixels, over the inliers). A fit supported by fewer than `min_inliers` points is not trusted and
    gives M = None: a shot cut or a close-up leaves a handful of accidental matches that can fit a wild motion."""
    p0 = select_features(gray0, mask, max_corners, quality, min_distance, border)
    tr = track_forward_backward(gray0, gray1, p0, win, levels, max_fb_error)
    idx = np.flatnonzero(tr["ok"])
    M, inl = fit_motion(p0[idx], tr["p1"][idx], model, ransac_thresh)
    inliers = np.zeros(len(p0), bool)
    if M is not None:
        inliers[idx[inl]] = True
        if inliers.sum() < min_inliers:
            M = None
    res = np.linalg.norm(apply_motion(M, p0[inliers]) - tr["p1"][inliers], axis=1) if M is not None else np.zeros(0)
    return {"M": M, "p0": p0, "p1": tr["p1"], "fb_error": tr["fb_error"], "tracked": tr["ok"], "inliers": inliers,
            "n_selected": len(p0), "n_tracked": int(tr["ok"].sum()), "n_inliers": int(inliers.sum()),
            "median_residual": float(np.median(res)) if len(res) else float("nan")}


def motion_summary(M, w, h):
    """Human-readable view of a motion at the image centre: shift (dx, dy) in pixels, zoom factor (>1 zooms in),
    rotation in degrees (image axes, y down), from the local affine part of M."""
    c = np.array([[w / 2.0, h / 2.0]])
    c1 = apply_motion(M, c)[0]
    e = 1.0
    J = np.stack([(apply_motion(M, c + [[e, 0.0]])[0] - c1) / e, (apply_motion(M, c + [[0.0, e]])[0] - c1) / e], 1)
    return {"dx": float(c1[0] - c[0, 0]), "dy": float(c1[1] - c[0, 1]), "zoom": float(np.sqrt(abs(np.linalg.det(J)))),
            "rotation_deg": float(np.degrees(np.arctan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1])))}


def chain(motions):
    """Compose consecutive motions [M01, M12, ...] into the motion from the first frame to the last."""
    out = np.eye(3)
    for M in motions:
        out = np.asarray(M, float) @ out
    return out


def propagate_homography(H0, M):
    """Field homography of the next frame: world -> frame-0 pixels (H0) followed by frame 0 -> frame 1 (M)."""
    return np.asarray(M, float) @ np.asarray(H0, float)
