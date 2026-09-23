import cv2
import numpy as np

from sportcal.core import camera as C
from sportcal.core.fitting import detect_lines

W, H = 960, 540
POSE = np.array([[3.0, -60.0, 20.0, np.radians(10), np.radians(22), 1.8 * W]])  # cx cy cz pan tilt f


def test_pose_homography_puts_the_field_in_front_of_a_centred_camera():
    Hs = C.pose_to_H(POSE, W, H)
    xy, depth = C.project(Hs, np.array([[0.0, 0.0], [0.0, 30.0]]))
    assert (depth > 0).all() and np.isfinite(xy).all()
    assert 0 < xy[0, 0, 0] < W and 0 < xy[0, 0, 1] < H     # field centre is in view
    assert xy[0, 1, 1] < xy[0, 0, 1]                        # +Y (far side) is UP in the image


def test_a_pose_homography_is_exactly_a_pinhole_camera():
    """pinhole_residual must report ~0 mismatch and recover f^2 for any H built from a pose."""
    f2, mism = C.pinhole_residual(C.pose_to_H(POSE, W, H), W, H)
    assert mism[0] < 1e-9 and np.isclose(f2[0], (1.8 * W) ** 2, rtol=1e-6)
    assert C.is_pinhole_consistent(C.pose_to_H(POSE, W, H), W, H).all()


def test_a_shear_is_not_a_pinhole_camera():
    shear = np.array([[[1.0, 3.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    assert not C.is_pinhole_consistent(shear, W, H).any()


def test_batched_project_matches_a_single_projection():
    poses = np.tile(POSE, (3, 1)) * [1, 1, 1, 1, 1, [1.0], [1.0], [1.0]][0]
    poses[:, 3] += [0.0, 0.1, -0.1]
    Hs = C.pose_to_H(poses, W, H)
    pts = np.array([[5.0, 5.0], [-10.0, 20.0]])
    xy, _ = C.project(Hs, pts)
    for i in range(3):
        one, _ = C.project(Hs[i:i + 1], pts)
        assert np.allclose(xy[i], one[0])


def test_bilinear_is_exact_on_a_linear_ramp():
    ramp = np.add.outer(np.arange(10.0), 2.0 * np.arange(10.0))  # value = row + 2*col
    xy = np.array([[3.5, 2.25], [0.0, 0.0], [8.0, 7.5]])
    assert np.allclose(C.bilinear(ramp, xy), xy[:, 1] + 2.0 * xy[:, 0])


def test_detect_lines_finds_two_crossing_lines():
    m = np.zeros((300, 600), np.uint8)
    cv2.line(m, (20, 220), (580, 220), 1, 3)     # horizontal
    cv2.line(m, (100, 20), (400, 280), 1, 3)     # diagonal
    lines = detect_lines(m, n_max=5)
    assert len(lines) == 2
    assert np.allclose(np.hypot(lines[:, 0], lines[:, 1]), 1.0)
    for (x, y) in [(300, 220), (250, 150)]:      # a point on each true line
        assert min(abs(a * x + b * y + c) for a, b, c in lines) < 2.5


def test_detect_lines_of_an_empty_mask_is_empty():
    assert detect_lines(np.zeros((100, 200), np.uint8)).shape == (0, 3)


def test_decompose_is_the_inverse_of_pose_to_H():
    rng = np.random.default_rng(0)
    for _ in range(8):
        pose = np.array([rng.uniform(-30, 30), rng.uniform(-80, -40), rng.uniform(8, 30), rng.uniform(-0.8, 0.8),
                         rng.uniform(0.15, 0.7), W * rng.uniform(0.8, 4.0)])
        d = C.decompose_H(C.pose_to_H(pose, W, H)[0], W, H)
        assert np.allclose(d["C"], pose[:3], atol=1e-6)
        assert np.allclose([d["pan"], d["tilt"], d["f"]], pose[3:], rtol=1e-6, atol=1e-8) and d["mismatch"] < 1e-9


def test_decompose_ignores_the_arbitrary_sign_of_a_homography_and_rejects_non_cameras():
    Hp = C.pose_to_H(POSE, W, H)[0]
    assert np.allclose(C.decompose_H(-3.0 * Hp, W, H)["C"], POSE[0, :3], atol=1e-6)
    assert C.decompose_H(np.array([[1.0, 3.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), W, H) is None


def test_project_distorted_with_zero_distortion_matches_the_plain_pinhole():
    """docs/plans/lens-distortion.md step 1's anchor: k1 = k2 = 0 must reproduce project(pose_to_H(...)) exactly."""
    pts = np.array([[0.0, 0.0], [10.0, 5.0], [-20.0, 15.0], [30.0, -10.0]])
    xy_d, depth_d = C.project_distorted(POSE, 0.0, 0.0, pts, W, H)
    xy, depth = C.project(C.pose_to_H(POSE, W, H), pts)
    assert np.allclose(xy_d, xy, atol=1e-9) and np.allclose(depth_d, depth, atol=1e-9)


def test_project_distorted_accepts_a_scalar_or_one_k_per_pose():
    poses = np.tile(POSE, (2, 1))
    poses[1, 3] += 0.2
    xy_scalar, _ = C.project_distorted(poses, 0.01, 0.0, np.array([[0.0, 0.0]]), W, H)
    xy_array, _ = C.project_distorted(poses, np.array([0.01, 0.01]), np.array([0.0, 0.0]), np.array([[0.0, 0.0]]), W, H)
    assert np.allclose(xy_scalar, xy_array)


def test_a_nonzero_k1_bends_collinear_world_points_off_a_straight_line():
    """A world-straight line must stay straight through project() (pure pinhole) and bend under distortion,
    growing with |k1| -- the sanity check of docs/plans/lens-distortion.md step 1."""
    line_pts = np.c_[np.linspace(-40.0, 40.0, 9), np.full(9, 5.0)]  # a straight line in the world

    def bend(xy):
        # perpendicular deviation (px) of the projected points from the best-fit line through them
        d = xy[-1] - xy[0]
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        return np.abs((xy - xy[0]) @ n).max()

    xy0, _ = C.project_distorted(POSE, 0.0, 0.0, line_pts, W, H)
    assert bend(xy0[0]) < 1e-6                                      # undistorted: exactly straight
    bends = []
    for k1 in (0.05, 0.15, 0.3):
        xy, _ = C.project_distorted(POSE, k1, 0.0, line_pts, W, H)
        bends.append(bend(xy[0]))
    assert bends[0] > 1.0                                            # already visibly curved
    assert bends[1] > bends[0] and bends[2] > bends[1]                # grows with |k1|


def test_ptz_warp_is_the_same_camera_turned_and_zoomed():
    """Warping by G must give exactly the homography of the tilted / zoomed pose: what makes the augmentation exact."""
    H0 = C.pose_to_H(POSE, W, H)[0]
    for dtilt, zoom in ((np.radians(3), 1.0), (0.0, 1.25), (np.radians(-2), 0.9)):
        G, H1 = C.ptz_warp(H0, W, H, tilt=dtilt, zoom=zoom)
        p = POSE.copy()
        p[0, 4] += dtilt
        p[0, 5] *= zoom
        want = C.pose_to_H(p, W, H)[0]
        assert np.allclose(H1 / H1[2, 2], want / want[2, 2], rtol=1e-6, atol=1e-6), (dtilt, zoom)
        assert np.allclose(G @ H0, H1)
