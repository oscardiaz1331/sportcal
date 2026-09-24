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


@pytest.mark.slow
def test_ellipse_fixed_center_recovers_a_ptz_pose_with_no_mask_and_no_clicks():
    from sportcal.core.camera import decompose_H, pose_to_H
    from sportcal.sports.soccer.field import CIRCLE_RADIUS
    center = np.array([-3.0, -68.0, 13.5])
    pose = np.r_[center, np.radians(-14.0), np.radians(11.0), 1.9 * W]
    H_true = pose_to_H(pose[None], W, H_IMG)[0]
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False) + 0.2
    ell = project(H_true[None], np.c_[CIRCLE_RADIUS * np.cos(ang), CIRCLE_RADIUS * np.sin(ang)])[0][0]
    solver = FieldSolver(np.zeros((H_IMG, W), np.uint8))    # empty mask: only the ellipse decides
    best = solver.search_ellipse_fixed_center(center, ell, top=1)
    assert best and reprojection_error(best[0]["H"], H_true, W, H_IMG) * K1920 < 1.0
    # a 2-metre-off centre must fit visibly worse (checks the ellipse actually constrains the centre, not just pan/tilt/f)
    wrong = solver.search_ellipse_fixed_center(center + [2.0, 0.0, 0.0], ell, top=1)
    d_wrong = reprojection_error(wrong[0]["H"], H_true, W, H_IMG) * K1920 if wrong else float("inf")
    assert d_wrong > 5.0 * (reprojection_error(best[0]["H"], H_true, W, H_IMG) * K1920)


def _dibuja_distorted(pose, k1, k2, rng, w=W, h=H_IMG):
    """Like evaluation.dibuja, but rendered with project_distorted (radial distortion) instead of an H:
    a clean synthetic mask, no dirt, for testing FieldSolver.refine_distorted."""
    import cv2
    from sportcal.core.camera import project_distorted
    from sportcal.sports.soccer import field as TPL
    m = np.zeros((h, w), np.uint8)
    for pl in TPL.polylines().values():
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
        s = np.arange(0, d[-1], 0.1)
        p = np.stack([np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])], 1)
        xy, den = project_distorted(pose, k1, k2, p, w, h)
        xy, den = xy[0], den[0]
        ok = (den > 1e-3) & np.isfinite(xy).all(1) & (np.abs(xy) < 5 * w).all(1)
        for run in np.split(np.arange(len(p)), np.where(~ok)[0]):
            run = run[ok[run]]
            if len(run) > 1:
                cv2.polylines(m, [np.round(xy[run]).astype(np.int32)], False, 1, 2)
    return m


def _template_error(pose_a, k1_a, k2_a, pose_b, k1_b, k2_b, w=W, h=H_IMG):
    """Median px distance between two distorted projections of a pitch-wide grid, on points the second one sees."""
    from sportcal.core.camera import project_distorted
    grid = np.array([[x, y] for x in np.linspace(-45, 45, 13) for y in np.linspace(-28, 28, 9)], float)
    a, da = project_distorted(pose_a, k1_a, k2_a, grid, w, h)
    b, db = project_distorted(pose_b, k1_b, k2_b, grid, w, h)
    a, b, da, db = a[0], b[0], da[0], db[0]
    ok = (db > 0) & (da > 0) & (b[:, 0] > 0) & (b[:, 0] < w) & (b[:, 1] > 0) & (b[:, 1] < h)
    return float(np.median(np.linalg.norm(a[ok] - b[ok], axis=1))) if ok.sum() >= 8 else float("nan")


@pytest.mark.slow
def test_refine_distorted_recovers_a_known_distortion_from_an_undistorted_start():
    """docs/plans/lens-distortion.md step 3: seed from the (wrong, undistorted) pose that already solves the
    clean field lines, refine pose + k1 jointly, and check the fit gets much closer to the true (distorted)
    field than the undistorted start -- NOT that k1 itself is recovered precisely: k1 trades off against
    pan/tilt/position over the radius range one field of view covers, so from a single frame the exact value is
    only loosely determined even though the resulting projection is accurate (measured on several synthetic
    scenes; a tight tolerance on k1 alone made this test flaky across seeds)."""
    from sportcal.core.camera import decompose_H
    rng = np.random.default_rng(6)
    pose_true, H_true = pose_aleatoria(rng)
    k1_true = 0.2
    mask = _dibuja_distorted(pose_true, k1_true, 0.0, rng)
    solver = FieldSolver(mask)
    start = solver.search_lines()
    assert start and start[0]["score"] > 0.6            # the undistorted solver still finds a usable rough fit
    d = decompose_H(start[0]["H"], W, H_IMG)
    pose0 = np.array([d["C"][0], d["C"][1], d["C"][2], d["pan"], d["tilt"], d["f"]])
    e_before = _template_error(pose0, 0.0, 0.0, pose_true, k1_true, 0.0)
    pose, k1, k2, score = solver.refine_distorted(pose0)
    e_after = _template_error(pose, k1, k2, pose_true, k1_true, 0.0)
    assert e_after < 0.7 * e_before and e_after < 5.0
    assert 0.05 < k1 < 0.45 and k2 == 0.0                # right order of magnitude and sign; k2 left untouched


@pytest.mark.slow
def test_refine_distorted_does_not_invent_distortion_on_a_clean_undistorted_mask():
    rng = np.random.default_rng(5)
    pose_true, H_true = pose_aleatoria(rng)
    mask = dibuja(H_true, rng, 0)
    solver = FieldSolver(mask)
    start = solver.search_lines()
    from sportcal.core.camera import decompose_H
    d = decompose_H(start[0]["H"], W, H_IMG)
    pose0 = np.array([d["C"][0], d["C"][1], d["C"][2], d["pan"], d["tilt"], d["f"]])
    pose, k1, k2, score = solver.refine_distorted(pose0)
    assert abs(k1) < 0.025 and abs(k2) < 0.01
