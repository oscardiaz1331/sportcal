"""sportcal/lab/soccer/lens_distortion.py: pooling line straightness across frames into one shared k1."""
import cv2
import numpy as np
import pytest

from sportcal.lab.soccer import lens_distortion as LD
from sportcal.lab.soccer.evaluation import H_IMG, W, dibuja, pose_aleatoria
from sportcal.core.camera import pose_to_H

K = np.array([[1900.0, 0, 960.0], [0, 1900.0, 540.0], [0, 0, 1.0]])


def _distorted_line(x0, y0, x1, y1, k1, n=24):
    """Pixel points of a WORLD-straight line: straight in normalized coordinates (true for any pinhole map),
    then radially distorted -- the inverse of what `_residuals` undoes with `cv2.undistortPoints`."""
    t = np.linspace(0, 1, n)
    x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)          # straight in normalized camera coordinates
    r2 = x ** 2 + y ** 2
    xd, yd = x * (1 + k1 * r2), y * (1 + k1 * r2)
    px = np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)
    return px


def test_residuals_are_zero_for_a_straight_line_at_zero_distortion():
    pts = _distorted_line(-0.3, -0.1, 0.5, 0.2, 0.0)
    r = LD._residuals(np.array([0.0]), [{"pts": pts, "K": K}])
    assert np.abs(r).max() < 1e-3     # cv2's float32 round trip through undistortPoints, not exactly 0


def test_a_line_through_the_optical_centre_never_bows():
    """A purely radial distortion moves every point along its own radius, which for a point ON a line through
    the centre IS the line's own direction -- no perpendicular displacement, whatever k1 is. A real regression
    test: an earlier version of this test file picked exactly such a line by accident and could not see any
    distortion at all."""
    through_centre = _distorted_line(-0.5, -0.35, 0.5, 0.35, 0.2)   # endpoints are +-(0.5, 0.35): passes through (0, 0)
    assert LD._residuals(np.array([0.0]), [{"pts": through_centre, "K": K}]).max() < 1e-3


def test_a_nonzero_k1_bows_an_off_centre_line_and_grows_with_leverage():
    central = _distorted_line(-0.05, 0.2, 0.05, 0.25, 0.15)        # close to the principal point: little leverage
    peripheral = _distorted_line(-0.5, 0.45, 0.5, 0.55, 0.15)      # reaches far from it: much more leverage
    r_c = LD._residuals(np.array([0.0]), [{"pts": central, "K": K}])
    r_p = LD._residuals(np.array([0.0]), [{"pts": peripheral, "K": K}])
    assert np.abs(r_p).max() > 5.0 * np.abs(r_c).max()


def test_fit_shared_k1_recovers_the_true_value_from_several_lines():
    k1_true = 0.18
    rng = np.random.default_rng(0)
    lines = []
    for _ in range(6):
        x0, y0 = rng.uniform(-0.55, 0.55, 2)
        x1, y1 = rng.uniform(-0.55, 0.55, 2)
        if np.hypot(x1 - x0, y1 - y0) < 0.6:                # keep only lines with real leverage, like real long lines
            continue
        pts = _distorted_line(x0, y0, x1, y1, k1_true) + rng.normal(0, 0.3, (24, 2))   # a bit of pixel noise
        lines.append({"pts": pts, "K": K})
    assert len(lines) >= 3
    fit = LD.fit_shared_k1(lines)
    assert abs(fit["k1"] - k1_true) < 0.02
    assert fit["rms_after"] < fit["rms_before"]
    assert fit["n_lines"] == len(lines)


def test_fit_shared_k1_of_no_lines_is_none():
    assert LD.fit_shared_k1([]) is None


def test_longest_line_points_picks_the_longer_of_two_and_ignores_short_ones():
    mask = np.zeros((400, 900), np.uint8)
    cv2.line(mask, (20, 200), (850, 210), 1, 3)     # long, spans most of the width
    cv2.line(mask, (400, 50), (500, 55), 1, 3)      # short, well under MIN_LINE_FRAC of the width
    pts = LD.longest_line_points(mask)
    assert pts is not None and len(pts) >= LD.N_BINS // 2
    assert pts[:, 1].std() < 5.0                     # the picked line is the near-horizontal long one


def test_longest_line_points_of_an_empty_mask_is_none():
    assert LD.longest_line_points(np.zeros((300, 600), np.uint8)) is None


@pytest.mark.slow
def test_is_broadcast_frame_accepts_the_known_centre_and_rejects_a_distant_one():
    rng = np.random.default_rng(1)
    pose, H = pose_aleatoria(rng, W, H_IMG)
    mask = dibuja(H, rng, 0, W, H_IMG)

    class _FakeCap:
        def get(self, *a):
            return 1

        def set(self, *a):
            pass

        def read(self):
            return True, cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

        def release(self):
            pass

    import sportcal.lab.soccer.evaluation as EV
    orig = EV.etapas_mascara
    EV.etapas_mascara = lambda img: {"lineas": mask}
    try:
        acc = LD.is_broadcast_frame(mask, pose[:3])
        assert acc is not None and np.allclose(acc["pose"]["C"], pose[:3], atol=1.0)
        assert LD.is_broadcast_frame(mask, pose[:3] + [20.0, 0.0, 0.0]) is None
    finally:
        EV.etapas_mascara = orig
