import cv2
import numpy as np

from sportcal.core import fitting as F


def _line_mask(p0, p1, shape=(400, 600), thickness=5):
    m = np.zeros(shape, np.uint8)
    cv2.line(m, p0, p1, 1, thickness)
    return m


def test_fit_line_px_recovers_the_line():
    p, q, r = F.fit_line_px(_line_mask((50, 300), (550, 100)))
    for x, y in [(50, 300), (300, 200), (550, 100)]:
        assert abs(p * x + q * y + r) < 1.5
    assert abs(np.hypot(p, q) - 1) < 1e-6


def test_fit_line_px_rejects_tiny_masks():
    assert F.fit_line_px(_line_mask((10, 10), (20, 12), thickness=1)) is None


def test_ransac_picks_the_dominant_line_in_a_mixed_mask():
    m = _line_mask((20, 350), (580, 350), thickness=4)  # long board
    cv2.circle(m, (300, 150), 60, 1, 4)                 # clutter: a ring
    p, q, r = F.fit_line_ransac(m)
    assert abs(q) > 0.95 and abs(p * 300 + q * 350 + r) < 2.0  # nearly horizontal, at y=350


def test_circle_center_is_found():
    m = np.zeros((400, 600), np.uint8)
    cv2.ellipse(m, (300, 200), (90, 60), 0, 0, 360, 1, 4)
    cx, cy = F.fit_circle_center(m, 600, 400)
    assert abs(cx - 300) < 2 and abs(cy - 200) < 2


def test_circle_touching_the_border_is_rejected_as_clipped():
    m = np.zeros((400, 600), np.uint8)
    cv2.ellipse(m, (30, 200), (90, 60), 0, 0, 360, 1, 4)  # runs off the left edge
    assert F.fit_circle_center(m, 600, 400) is None


def test_partial_arc_is_rejected():
    m = np.zeros((400, 600), np.uint8)
    cv2.ellipse(m, (300, 200), (90, 60), 0, 0, 100, 1, 4)  # only 100 degrees visible
    assert F.fit_circle_center(m, 600, 400) is None
