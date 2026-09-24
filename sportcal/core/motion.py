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

from sportcal.core import surface as SUR

MODELS = ("homography", "affine", "similarity")
REGIONS = ("background", "field", "all")
W_WORK = 960            # width the frames are tracked at
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


def to_work(frame, w=W_WORK):
    """Frame resized to the working width (INTER_AREA)."""
    h0, w0 = frame.shape[:2]
    return cv2.resize(frame, (w, int(round(h0 * w / w0))), interpolation=cv2.INTER_AREA)


def to_native_motion(M, w_native, w_work=W_WORK):
    """A motion estimated in working pixels, expressed in native pixels (the ones a native H lives in)."""
    s = w_native / float(w_work)
    S = np.diag([s, s, 1.0])
    return S @ np.asarray(M, float) @ np.linalg.inv(S)


def surface_region(work_bgr, surface, chi2=10.17):
    """Boolean play region of `surface` ("ice" | "grass", `core.surface`): robust Gaussian, then filled."""
    return SUR.play_region(SUR.robust_surface(work_bgr, chi2=chi2, surface=surface)) > 0


def background_mask(work_bgr, surface, erode_px=6, chi2=10.17):
    """(mask uint8, region bool): everything that is NOT the play surface, eroded so no feature sits on the boundary."""
    region = surface_region(work_bgr, surface, chi2)
    k = 2 * int(erode_px) + 1
    return cv2.erode((~region).astype(np.uint8), np.ones((k, k), np.uint8)), region


def field_mask(work_bgr, surface, erode_px=6, chi2=10.17, close_px=9):
    """(mask uint8, region bool): the play surface INCLUDING the painted lines, eroded away from its border.

    The robust Gaussian leaves the lines (and the players) as holes in the surface region; closing fills the thin ones, so
    a feature can sit on a line or on a line crossing."""
    region = surface_region(work_bgr, surface, chi2)
    k = 2 * int(close_px) + 1
    filled = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    e = 2 * int(erode_px) + 1
    return cv2.erode(filled, np.ones((e, e), np.uint8)), region


def track_pair(frame0, frame1, surface, erode_px=6, region_kind="background", **kw):
    """Motion frame0 -> frame1 (both native BGR) from the chosen part of the picture, in working pixels.

    region_kind: "background" = everything that is not play surface (stands, boards), "field" = the play surface with its
    painted lines, "all" = no restriction. Returns the dict of `estimate_motion` plus "work0", "work1" (resized frames),
    "region" (play region of frame 0) and "mask" (where features were allowed). Extra keyword arguments go to
    `estimate_motion`."""
    if region_kind not in REGIONS:
        raise ValueError("region_kind must be one of {}".format(REGIONS))
    w0, w1 = to_work(frame0), to_work(frame1)
    if region_kind == "background":
        mask, region = background_mask(w0, surface, erode_px)
    elif region_kind == "field":
        mask, region = field_mask(w0, surface, erode_px)
    else:
        region = surface_region(w0, surface)
        mask = np.ones(region.shape, np.uint8)
    res = estimate_motion(cv2.cvtColor(w0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY), mask=mask, **kw)
    res.update({"work0": w0, "work1": w1, "region": region, "mask": mask})
    return res


def track_video(get_frame, frame0, H0, surface, step=5, n=40, refine=None, region_kind="background", erode_px=6, **kw):
    """Carry a homography (or just the camera motion, when H0 is None) through a clip, one short step at a time.

    get_frame(i) -> native BGR frame or None. H0 maps world to NATIVE pixels of frame0. At every step the background
    motion is estimated (`track_pair`) and chained onto H; `refine(native_frame, H) -> (H, score)`, if given, then pulls H
    back onto what the sport can see in that frame (its painted lines). Yields one dict per frame, the first being frame0
    itself: {"frame", "H" (native, or None), "M" (step motion in working pixels, None at frame0), "cum" (frame0 -> this
    frame, working pixels), "n_selected", "n_inliers", "median_residual", "points" (working-pixel inlier positions in the
    previous frame), "score" (from refine), "lost"}. Stops after a step that loses the background (a shot cut or a
    close-up), which is reported with "lost": True."""
    prev = get_frame(frame0)
    if prev is None:
        return
    w_native = prev.shape[1]
    H = None if H0 is None else np.asarray(H0, float)
    cum = np.eye(3)
    yield {"frame": frame0, "H": H, "M": None, "cum": cum, "n_selected": 0, "n_inliers": 0, "median_residual": float("nan"),
           "points": np.zeros((0, 2)), "score": None, "lost": False}
    for k in range(1, n + 1):
        i = frame0 + k * step
        cur = get_frame(i)
        if cur is None:
            return
        r = track_pair(prev, cur, surface, erode_px=erode_px, region_kind=region_kind, **kw)
        row = {"frame": i, "H": H, "M": r["M"], "n_selected": r["n_selected"], "n_inliers": r["n_inliers"],
               "median_residual": r["median_residual"], "points": r["p0"][r["inliers"]], "score": None, "lost": r["M"] is None}
        if r["M"] is None:
            row["cum"] = cum
            yield row
            return
        cum = r["M"] @ cum
        row["cum"] = cum
        if H is not None:
            H = propagate_homography(H, to_native_motion(r["M"], w_native))
            if refine is not None:
                H, row["score"] = refine(cur, H)
            row["H"] = H
        prev = cur
        yield row
