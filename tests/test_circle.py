"""Ellipse of a world circle as a constraint on the homography (core/circle.py)."""
import numpy as np

from sportcal.core import circle as CI
from sportcal.core.camera import pose_to_H

W, HI = 1920, 1080
R = 9.15
POSE = np.array([[-3.0, -68.0, 13.5, np.radians(-10), np.radians(12), 1.9 * W]])


def _scene(pose=POSE, noise=0.0, seed=0):
    H = pose_to_H(pose, W, HI)[0]
    H = H / H[2, 2]
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False) + 0.1
    ell = CI._project(H, np.c_[R * np.cos(ang), R * np.sin(ang)])[0] + rng.normal(0, noise, (16, 2))
    return H, ell


def _grid_error(H, Ht):
    P = np.array([[x, y] for x in np.linspace(-30, 30, 7) for y in np.linspace(-20, 20, 5)], float)
    return float(np.median(np.linalg.norm(CI._project(H, P)[0] - CI._project(Ht, P)[0], axis=1)))


def test_conic_fit_and_sampson_distance():
    H, ell = _scene()
    C = CI.conic_from_points(ell)
    assert CI.is_ellipse(C)
    assert np.abs(CI.sampson(C, ell)).max() < 1e-6
    assert abs(CI.sampson(C, ell[:1] + [10.0, 0.0])[0]) > 2.0            # a point moved off the outline is far from it


def test_a_real_camera_view_has_the_orientation_the_fit_assumes():
    H, _ = _scene()
    assert CI.orientation(H) == -1.0


def test_family_members_are_never_mirror_images_whatever_the_ellipse():
    for i, pose in enumerate([POSE, POSE + [[8, 3, -2, 0.5, 0.1, 400]], POSE + [[-9, 5, 1, -0.8, 0.2, -300]], POSE + [[2, -4, 0, 0.2, -0.05, 900]]]):
        H, ell = _scene(pose)
        fam = CI.circle_family(CI.conic_from_points(ell), R)
        assert all(CI.orientation(fam(p)) == CI.REAL_VIEW for p in [(0, 0, 0), (2.0, 0.4, -0.3), (-1.0, -0.6, 0.5)]), i


def test_every_member_of_the_family_maps_the_circle_onto_the_ellipse():
    H, ell = _scene()
    fam = CI.circle_family(CI.conic_from_points(ell), R)
    ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    C = CI.conic_from_points(ell)
    for p in [(0.0, 0.0, 0.0), (1.0, 0.3, -0.2), (-2.0, -0.5, 0.4)]:
        xy = CI._project(fam(p), np.c_[R * np.cos(ang), R * np.sin(ang)])[0]
        assert np.abs(CI.sampson(C, xy)).max() < 1e-6


def test_ellipse_plus_two_clicks_fix_the_whole_field():
    Ht, ell = _scene()
    world = np.array([[0.0, 0.0], [0.0, R]])                                  # centre spot and centre circle x halfway line
    img = CI._project(Ht, world)[0]
    out = CI.fit_points_ellipse(world, img, ell, R)
    assert out is not None and _grid_error(out["H"], Ht) < 0.5
    assert out["n_solutions"] == 1
    # two clicks alone (without the ellipse) say nothing about the rest of the field: 4 correspondences are the minimum
    assert CI.fit_points_ellipse(world[:1], img[:1], ell, R) is None or CI.fit_points_ellipse(world[:1], img[:1], ell, R)["n_solutions"] > 1


def test_noisy_ellipse_and_clicks_stay_within_a_few_pixels():
    Ht, ell = _scene(noise=1.0, seed=3)
    world = np.array([[0.0, 0.0], [0.0, R], [0.0, -R], [-30.0, 0.0]])
    img = CI._project(Ht, world)[0] + np.random.default_rng(1).normal(0, 1.0, (4, 2))
    out = CI.fit_points_ellipse(world, img, ell, R)
    assert out is not None and _grid_error(out["H"], Ht) < 6.0


def test_the_ellipse_helps_when_the_clicks_are_few_and_noisy():
    Ht, ell = _scene(noise=0.5, seed=5)
    world = np.array([[0.0, 0.0], [0.0, R], [-30.0, 0.0], [30.0, 0.0]])       # 4 clicks, poorly spread across the field
    img = CI._project(Ht, world)[0] + np.random.default_rng(2).normal(0, 3.0, (4, 2))
    import cv2
    Hd = cv2.findHomography(world.astype(np.float32), img.astype(np.float32), 0)[0]
    with_e = CI.fit_points_ellipse(world, img, ell, R)
    assert _grid_error(with_e["H"], Ht) < 0.6 * _grid_error(Hd / Hd[2, 2], Ht)
