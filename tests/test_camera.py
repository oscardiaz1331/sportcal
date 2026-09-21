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
