"""End-to-end check of segmentation -> homography on a synthetic, noise-free probability map.

No weights or GPU: the "network output" is rendered from the template through a known H. The
result was verified bit-identical (max |diff| = 0.0) against the pre-restructure code, so this
guards the refactor that moved the DLT into core/. Error is not ~0 even on clean input because
an ellipse centre is not the projection of the circle centre (known bias, docs/experiments/hockey.md
section 6), hence the loose bound.
"""
import cv2
import numpy as np
import pytest

pytest.importorskip("torch")  # seg_to_homography imports torch at module level

from sportcal.core.geometry import geom_error, project_points  # noqa: E402
from sportcal.lab.hockey.seg_to_homography import solve_from_probs  # noqa: E402
from sportcal.sports import get  # noqa: E402
from sportcal.sports.hockey import rink  # noqa: E402

W, H = 1280, 720


def _synthetic_probs(H_true, sport):
    probs = np.zeros((12, H, W), np.float32)
    for cls, poly in sport.polylines():
        xy, ok = project_points(H_true, poly, W, H)
        if ok.sum() > 1:
            m = np.zeros((H, W), np.uint8)
            cv2.polylines(m, [xy[ok].astype(np.int32).reshape(-1, 1, 2)], False, 1, 5, cv2.LINE_AA)
            probs[cls] = np.maximum(probs[cls], m.astype(np.float32))
    probs[0] = 1 - probs[1:].max(0)
    return probs


def test_solver_recovers_a_wide_shot_from_clean_probabilities():
    sport = get("hockey-nhl")
    world = np.float32([(8, 0), (53, 0), (53, 25.9), (8, 25.9)])  # most of the rink in view
    image = np.float32([(60, 690), (1220, 690), (960, 90), (320, 90)])
    H_true = cv2.getPerspectiveTransform(world, image)
    H_true /= H_true[2, 2]

    H_est, n_lines, n_circles, cost = solve_from_probs(_synthetic_probs(H_true, sport), rink.RINK_NHL, W, H)

    assert H_est is not None
    assert (n_lines, n_circles) == (5, 5)
    assert cost < 1e-3
    assert geom_error(H_est, H_true, sport.keypoints(), W, H) < 25.0


def test_solver_refuses_an_empty_probability_map():
    probs = np.zeros((12, H, W), np.float32)
    probs[0] = 1.0
    assert solve_from_probs(probs, rink.RINK_NHL, W, H)[0] is None
