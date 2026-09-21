"""Soccer field solver on synthetic masks with a known camera (CPU, no data).

The scenario generator is the lab's own (`evaluation.pose_aleatoria` / `dibuja`). The old script and this
package produced IDENTICAL evaluation tables for the same seed (checked when the code was ported), so the
bounds below are the measured behaviour on clean masks, with slack: line intersections ~1.4 px, refining
from the truth ~0.5 px (errors at 1920 px width).
"""
import numpy as np
import pytest

from sportcal.core.camera import project
from sportcal.lab.soccer.evaluation import H_IMG, W, dibuja, pose_aleatoria
from sportcal.lab.soccer.field_solver import FieldSolver, line_hypotheses, reprojection_error
from sportcal.sports.soccer.field import CORNERS

K1920 = 1920.0 / W


@pytest.fixture(scope="module")
def scenes():
    rng = np.random.default_rng(0)
    out = []
    for _ in range(2):
        _, H_true = pose_aleatoria(rng)
        out.append((H_true, dibuja(H_true, rng, 0)))
    return out


@pytest.mark.slow
def test_line_intersections_recover_the_camera_on_clean_masks(scenes):
    for H_true, mask in scenes:
        best = FieldSolver(mask).search_lines(top=4)[0]
        assert reprojection_error(best["H"], H_true, W, H_IMG) * K1920 < 10.0
        assert best["score"] > 0.75


def test_refining_from_the_truth_stays_at_the_truth(scenes):
    H_true, mask = scenes[0]
    H_ref, score = FieldSolver(mask).refine(H_true)
    assert reprojection_error(H_ref, H_true, W, H_IMG) * K1920 < 2.0 and score > 0.75


def test_score_falls_sharply_as_the_camera_moves_off_the_truth(scenes):
    """Measured on this scene: 0.83 at the truth, 0.66 at 5 px, 0.10 at 19 px, 0.05 at 77 px."""
    H_true, mask = scenes[0]
    solver = FieldSolver(mask)

    def shifted(px):
        return np.array([[1, 0, px], [0, 1, 0], [0, 0, 1.0]]) @ H_true

    s0, s5, s19 = solver.score(np.stack([H_true, shifted(5.0), shifted(19.0)]), fine=True)
    assert s0 > 0.75 and s0 > s5 > s19 and s19 < 0.3


def test_corner_evidence_adds_to_the_score_and_snapping_corrects_a_small_shift(scenes):
    H_true, mask = scenes[0]
    solver = FieldSolver(mask)
    xy, depth = project(H_true[None], CORNERS)
    xy, depth = xy[0], depth[0]
    seen = (depth > 0) & (xy[:, 0] > 5) & (xy[:, 0] < W - 5) & (xy[:, 1] > 5) & (xy[:, 1] < H_IMG - 5)
    Q = xy[seen]
    assert len(Q) >= 4

    base = solver.score(H_true[None], fine=True)[0]
    solver.set_evidence(corners=Q)
    assert solver.score(H_true[None], fine=True)[0] > base + 0.3   # w_corner = 0.4 x ~1.0

    off = np.array([[1, 0, 0.01 * W], [0, 1, 0.01 * H_IMG], [0, 0, 1]]) @ H_true   # ~10 px off in the image
    H_snap, n_pairs = solver.snap_corners(off, Q)
    assert n_pairs >= 4
    assert reprojection_error(H_snap, H_true, W, H_IMG) < reprojection_error(off, H_true, W, H_IMG)


def test_no_hypotheses_from_too_few_lines_and_solver_rejects_a_blank_mask():
    assert len(line_hypotheses(np.zeros((3, 3)), W, H_IMG)) == 0
    solver = FieldSolver(np.zeros((H_IMG, W), np.uint8))
    assert solver.search_lines() == []


def _strip(mask, x0_frac=0.3, width_frac=0.35):
    """Keep only a vertical strip of the mask: a zoomed-in view with few primitives."""
    out = np.zeros_like(mask)
    x0, w = int(W * x0_frac), int(W * width_frac)
    out[:, x0:x0 + w] = mask[:, x0:x0 + w]
    return out


def _local_search(solver, center, pose, **kw):
    """Fixed-centre search on a small grid around the true pan / tilt / focal (fast enough for a test)."""
    pan, tilt = np.degrees(pose["pan"]), np.degrees(pose["tilt"])
    return solver.search_fixed_center(center, top=2, pan_range=(pan - 8, pan + 8), tilt_range=(tilt - 8, tilt + 8),
                                      f_range=(pose["f"] / W / 1.5, pose["f"] / W * 1.5), steps=(2.0, 2.0, 11), **kw)


@pytest.mark.slow
def test_fixed_centre_solves_a_view_with_few_primitives_and_a_wrong_centre_does_not(scenes):
    from sportcal.core.camera import decompose_H
    for H_true, mask in scenes:
        pose = decompose_H(H_true, W, H_IMG)
        strip = _strip(mask)
        good = _local_search(FieldSolver(strip), pose["C"], pose)[0]
        assert reprojection_error(good["H"], H_true, W, H_IMG) * K1920 < 10.0
        bad = _local_search(FieldSolver(strip), pose["C"] + np.array([0.0, 25.0, 0.0]), pose)[0]
        assert reprojection_error(bad["H"], H_true, W, H_IMG) * K1920 > 5 * reprojection_error(good["H"], H_true, W, H_IMG) * K1920


@pytest.mark.slow
def test_fixed_centre_result_is_an_exact_pinhole_camera_at_that_centre(scenes):
    from sportcal.core.camera import decompose_H
    H_true, mask = scenes[0]
    pose = decompose_H(H_true, W, H_IMG)
    best = _local_search(FieldSolver(_strip(mask)), pose["C"], pose)[0]
    d = decompose_H(best["H"], W, H_IMG)
    assert d["mismatch"] < 1e-6 and np.allclose(d["C"], pose["C"], atol=1e-4)


def test_robust_centre_drops_a_wild_vote():
    from sportcal.lab.soccer.camera_center import robust_centre
    votes = np.array([[-3.3, -67.9, 13.3], [-3.2, -68.0, 13.4], [-3.4, -67.8, 13.2], [30.3, -34.2, 5.6]])
    centre, spread, n = robust_centre(votes)
    assert n == 3 and np.allclose(centre, [-3.3, -67.9, 13.3], atol=0.11) and (spread < 0.3).all()
