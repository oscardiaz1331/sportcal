import cv2
import numpy as np

from sportcal.core import geometry as G

TW, TI = G.scale_transform(60.0), G.scale_transform(1920.0)

# A camera-like homography: the visible part of a 60x30 m field (x 10..50, y 3..27) maps to
# a trapezoid inside a 1920x1080 frame (near side wider than the far side).
WORLD_QUAD = np.float32([(10, 3), (50, 3), (50, 27), (10, 27)])
IMAGE_QUAD = np.float32([(300, 900), (1620, 900), (1300, 300), (620, 300)])
H_TRUE = cv2.getPerspectiveTransform(WORLD_QUAD, IMAGE_QUAD)
H_TRUE /= H_TRUE[2, 2]
INSIDE = np.array([[15.0, 8.0], [30.0, 15.0], [45.0, 22.0], [25.0, 25.0]])  # world points in view


def _correspondences(H, rng, n_pts=4, n_lines=4):
    """Point correspondences plus AXIS-ALIGNED line correspondences (X=const / Y=const),
    which is the only kind the field templates contain."""
    points, lines = [], []
    for _ in range(n_pts):
        X, Y = rng.uniform([12, 5], [48, 25])
        x, y, z = H @ [X, Y, 1]
        points.append((X, Y, x / z, y / z))
    for k in range(n_lines):
        a, b, c = (1.0, 0.0, -rng.uniform(12, 48)) if k % 2 == 0 else (0.0, 1.0, -rng.uniform(5, 25))
        lines.append((a, b, c, *(np.linalg.inv(H).T @ [a, b, c])))
    return points, lines


def test_dlt_recovers_H_at_real_scale():
    """The normalised DLT must be exact on noiseless data at metres/pixels magnitudes
    (un-normalised, this same input gave garbage: docs/experiments/hockey.md)."""
    pts, lns = _correspondences(H_TRUE, np.random.default_rng(0))
    Hest = G.solve_dlt_correspondences(pts, lns, TW, TI)
    assert np.abs(Hest - H_TRUE).max() < 1e-3


def test_hartley_normalisation_bounds_the_worst_case_under_noise():
    """With 1 px of noise, over 20 fixed seeds: worst-case error ~21 px normalised versus
    ~71 px un-normalised (measured). Exact data cannot tell the two apart, noise can."""
    worst = 0.0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        pts, lns = _correspondences(H_TRUE, rng, 6, 6)
        pts = [(X, Y, x + rng.normal(0, 1.0), y + rng.normal(0, 1.0)) for X, Y, x, y in pts]
        lns = [(a, b, c, p, q, r + rng.normal(0, 0.5)) for a, b, c, p, q, r in lns]
        H = G.solve_dlt_correspondences(pts, lns, TW, TI, max_cond=1.0)
        assert H is not None
        worst = max(worst, G.geom_error(H, H_TRUE, INSIDE, 1920, 1080))
    assert worst < 35.0


def test_dlt_line_rows_annihilate_a_consistent_line_pair():
    """A line correspondence generated from H must lie in the null space of its DLT rows."""
    rng = np.random.default_rng(1)
    H = np.eye(3) + rng.normal(0, 0.1, (3, 3))
    a, b, c = rng.uniform(-1, 1, 3)
    p, q, r = np.linalg.inv(H).T @ [a, b, c]
    assert np.abs(G.dlt_line_rows(a, b, c, p, q, r) @ H.ravel()).max() < 1e-9


def test_parallel_lines_are_rejected_as_degenerate():
    """4 parallel lines (all X=const) + 1 point cannot fix the transverse direction; the
    condition-number gate must refuse instead of returning a plausible-looking wrong H."""
    lines = []
    for x0 in (15.0, 25.0, 35.0, 45.0):
        a, b, c = 1.0, 0.0, -x0
        lines.append((a, b, c, *(np.linalg.inv(H_TRUE).T @ [a, b, c])))
    x, y, z = H_TRUE @ [25.0, 10.0, 1]
    assert G.solve_dlt_correspondences([(25.0, 10.0, x / z, y / z)], lines, TW, TI) is None


def test_solve_with_residual_needs_eight_rows():
    H, res = G.solve_dlt_with_residual([(1, 2, 3, 4)] * 3, [], TW, TI)
    assert H is None and np.isinf(res)


def test_refinement_pulls_a_perturbed_seed_back():
    pts, lns = _correspondences(H_TRUE, np.random.default_rng(3), n_pts=4, n_lines=4)
    seed = H_TRUE.copy()
    seed[0, 2] += 15.0  # 15 px shift
    assert G.geom_error(seed, H_TRUE, INSIDE, 1920, 1080) > 5.0  # visibly off before refining
    Href, cost = G.refine_nonlinear(seed, pts, lns, TW, TI, 60.0, 30.0)
    assert G.geom_error(Href, H_TRUE, INSIDE, 1920, 1080) < 1.0 and cost < 1e-4


def test_project_points_drops_points_behind_the_camera():
    H = np.array([[100.0, 0, 500], [0, 100.0, 300], [0.0, 0.05, 1.0]])
    world = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, -100.0]])  # last: w = 1 - 5 < 0, behind
    _, ok = G.project_points(H, world, 1920, 1080)
    assert ok.tolist() == [True, True, False]


def test_geom_error_is_zero_for_identical_H_and_inf_when_nothing_is_in_frame():
    assert G.geom_error(H_TRUE, H_TRUE, INSIDE, 1920, 1080) == 0.0
    assert G.geom_error(H_TRUE, H_TRUE, INSIDE + 1e4, 1920, 1080) == np.inf


def test_image_to_world_inverts_projection():
    img, ok = G.project_points(H_TRUE, INSIDE, 1920, 1080)
    assert ok.all() and np.allclose(G.image_to_world(H_TRUE, img), INSIDE, atol=1e-6)


def test_ransac_rejects_outliers_and_needs_four_points():
    rng = np.random.default_rng(5)
    world = rng.uniform([12, 5], [48, 25], (20, 2))
    img = cv2.perspectiveTransform(world.reshape(-1, 1, 2).astype(np.float32), H_TRUE).reshape(-1, 2)
    img[:4] += 300.0  # 4 gross outliers
    _, inliers = G.fit_homography_ransac(world, img, 3.0)
    assert inliers[4:].all() and not inliers[:4].any()
    assert G.fit_homography_ransac(world[:3], img[:3], 3.0) == (None, None)


def test_clustered_clicks_are_unstable_and_off_field_pixels_do_not_count():
    def clicks(world):
        return cv2.perspectiveTransform(np.float32(world)[None], H_TRUE)[0]
    spread = np.vstack([WORLD_QUAD, [[30.0, 15.0]]])
    cluster = np.array([[30.0, 15], [31, 15], [31, 16], [30, 16], [30.5, 15.5]])
    box = (0.0, 60.0, 0.0, 30.0)
    s, c = (G.click_sensitivity(p, clicks(p), H_TRUE, 1920, 1080, box) for p in (spread, cluster))
    assert s < 5 and c > 100
    assert G.click_sensitivity(spread, clicks(spread), H_TRUE, 1920, 1080, (500.0, 560.0, 0.0, 30.0)) == np.inf
