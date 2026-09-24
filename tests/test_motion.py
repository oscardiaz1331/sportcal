"""Background motion from tracked corners: forward-backward check + RANSAC (sportcal/core/motion.py).

Synthetic frames only. The motion is the exact image-to-image map of a pure camera rotation plus zoom, and the scene has the
things that break a naive fit: a static overlay (logo) and independently moving patches (players).
"""
import cv2
import numpy as np

from sportcal.core import motion

W, H = 640, 360


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 110, np.uint8)
    for _ in range(420):
        x, y = int(rng.integers(0, W - 40)), int(rng.integers(0, H - 40))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(6, 40)), y + int(rng.integers(6, 40))), int(rng.integers(20, 235)), -1)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def _rotation_zoom(pan_deg, tilt_deg, zoom, f=700.0):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    K2 = K.copy()
    K2[0, 0] = K2[1, 1] = f * zoom
    a, b = np.radians(pan_deg), np.radians(tilt_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    M = K2 @ Rx @ Ry @ np.linalg.inv(K)
    return M / M[2, 2]


def _pair(M_true, movers=True, overlay=True, seed=0):
    img0 = _scene(seed)
    img1 = cv2.warpPerspective(img0, M_true, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rng = np.random.default_rng(seed + 1)
    if movers:  # independent movers: the same textured patch appears somewhere else in frame 1
        for _ in range(14):
            x, y = int(rng.integers(30, W - 100)), int(rng.integers(30, H - 130))
            patch = rng.integers(30, 220, (70, 40)).astype(np.uint8)
            img0[y:y + 70, x:x + 40] = patch
            img1[y + 12:y + 82, x + 30:x + 70] = patch
    if overlay:  # a logo that does not move with the scene
        for img in (img0, img1):
            cv2.rectangle(img, (20, 14), (200, 60), 255, -1)
            cv2.putText(img, "LOGO 12", (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    return img0, img1


def _corner_error(M_est, M_true):
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [W / 2, H / 2]], float)
    return float(np.mean(np.linalg.norm(motion.apply_motion(M_est, corners) - motion.apply_motion(M_true, corners), axis=1)))


def test_recovers_rotation_and_zoom_despite_movers_and_overlay():
    M_true = _rotation_zoom(1.6, 0.9, 1.03)
    img0, img1 = _pair(M_true)
    r = motion.estimate_motion(img0, img1)
    assert r["M"] is not None
    assert _corner_error(r["M"], M_true) < 1.0
    assert r["n_inliers"] > 0.5 * r["n_tracked"]


def test_ransac_is_what_rejects_the_outliers():
    """A least-squares fit over the same tracked points is dragged by the movers and the overlay: this is what makes
    the previous test a real check of the RANSAC step and not of the tracker alone."""
    M_true = _rotation_zoom(1.6, 0.9, 1.03)
    img0, img1 = _pair(M_true)
    r = motion.estimate_motion(img0, img1)
    ok = r["tracked"]
    M_ls, _ = cv2.findHomography(r["p0"][ok], r["p1"][ok], 0)
    assert _corner_error(M_ls, M_true) > 3 * _corner_error(r["M"], M_true)


def test_forward_backward_check_rejects_unstable_tracks():
    """Frame 1 is frame 0 shifted, except a noisy band: tracks there do not survive the round trip."""
    img0 = _scene(3)
    M_true = np.array([[1, 0, 6.0], [0, 1, 3.0], [0, 0, 1.0]])
    img1 = cv2.warpPerspective(img0, M_true, (W, H))
    rng = np.random.default_rng(5)
    img1[150:260, :] = rng.integers(0, 255, (110, W)).astype(np.uint8)
    p0 = motion.select_features(img0, max_corners=1500, min_distance=6)
    tr = motion.track_forward_backward(img0, img1, p0, max_fb_error=0.7)
    noisy = (p0[:, 1] > 160) & (p0[:, 1] < 250)
    clean = (p0[:, 1] < 130) | (p0[:, 1] > 290)
    assert noisy.sum() > 20 and clean.sum() > 20
    assert tr["ok"][noisy].mean() < 0.3 < 0.7 < tr["ok"][clean].mean()


def test_features_stay_inside_the_mask():
    img = _scene(1)
    mask = np.zeros((H, W), np.uint8)
    mask[:, W // 2:] = 1
    pts = motion.select_features(img, mask, max_corners=300)
    assert len(pts) > 30 and (pts[:, 0] >= W // 2).all()


def test_too_few_points_gives_no_motion():
    flat = np.full((H, W), 90, np.uint8)
    r = motion.estimate_motion(flat, flat)
    assert r["M"] is None and r["n_inliers"] == 0


def test_summary_recovers_shift_zoom_rotation():
    th, s = np.radians(4.0), 1.15
    M = np.array([[s * np.cos(th), -s * np.sin(th), 0], [s * np.sin(th), s * np.cos(th), 0], [0, 0, 1.0]])
    c = np.array([W / 2, H / 2])
    M[:2, 2] = c + [12.0, -7.0] - M[:2, :2] @ c          # the centre moves by (12, -7)
    out = motion.motion_summary(M, W, H)
    assert abs(out["dx"] - 12.0) < 1e-6 and abs(out["dy"] + 7.0) < 1e-6
    assert abs(out["zoom"] - s) < 1e-6 and abs(out["rotation_deg"] - 4.0) < 1e-6


def test_chain_and_propagation_compose_in_frame_order():
    A = np.array([[1, 0, 5.0], [0, 1, 0], [0, 0, 1.0]])
    B = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
    assert np.allclose(motion.chain([A, B]), B @ A)              # A happens first
    H0 = np.array([[3.0, 0, 1.0], [0, 3.0, 2.0], [0, 0, 1.0]])
    assert np.allclose(motion.propagate_homography(H0, A), A @ H0)


def test_a_weakly_supported_fit_is_rejected():
    """A close-up or a shot cut leaves a handful of consistent-looking matches; a fit on them can be wild. With few
    corners available the motion is well defined but poorly supported: it exists without the guard and must not be returned with it."""
    M_true = _rotation_zoom(1.0, 0.5, 1.01)
    img0, img1 = _pair(M_true, movers=False, overlay=False)
    few = np.zeros((H, W), np.uint8)
    few[120:240, 200:420] = 1
    unguarded = motion.estimate_motion(img0, img1, mask=few, max_corners=22, min_inliers=0)
    assert unguarded["M"] is not None and 8 <= unguarded["n_inliers"] < 30
    guarded = motion.estimate_motion(img0, img1, mask=few, max_corners=22)
    assert guarded["M"] is None


def test_track_video_chains_the_homography_through_native_frames():
    """Frames are windows sliding over one bigger scene (a pure pan): after the steps, H must be the start H followed by
    the total shift in NATIVE pixels, though tracking runs at the working width."""
    big = cv2.cvtColor(cv2.resize(_scene(3), (1600, 900)), cv2.COLOR_GRAY2BGR)
    dx, dy = 6, 2                                     # native px per frame index

    def get_frame(i):
        return big[50 - i * dy:50 - i * dy + 720, 100 - i * dx:100 - i * dx + 1280] if i <= 12 else None
    H0 = np.array([[20.0, 3.0, 200.0], [0.5, -9.0, 600.0], [0.0, 0.002, 1.0]])
    rows = list(motion.track_video(get_frame, 0, H0, "ice", step=3, n=5, region_kind="all"))
    assert [r["frame"] for r in rows] == [0, 3, 6, 9, 12] and not any(r["lost"] for r in rows)
    world = np.array([[0.0, 0.0], [30.0, 10.0], [60.0, 25.0]])
    want = motion.apply_motion(H0, world) + [12 * dx, 12 * dy]
    assert np.abs(motion.apply_motion(rows[-1]["H"], world) - want).max() < 0.5
