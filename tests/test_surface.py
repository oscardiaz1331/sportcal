import cv2
import numpy as np
import pytest

from sportcal.core import surface as S


def _ice_with_red_line():
    img = np.full((240, 320, 3), (235, 235, 235), np.uint8)  # near-white ice (BGR)
    cv2.line(img, (20, 120), (300, 120), (60, 60, 220), 5)   # red painted line
    return img


def _grass_with_white_line():
    img = np.full((240, 320, 3), (60, 140, 50), np.uint8)    # green (BGR)
    cv2.line(img, (20, 120), (300, 120), (240, 240, 240), 5)
    return img


def test_ice_threshold_excludes_the_painted_line_and_play_region_refills_it():
    ice = S.hsv_threshold(_ice_with_red_line(), "ice")
    assert ice[60, 100] == 1 and ice[120, 100] == 0  # ice yes, line no
    assert S.play_region(ice)[120, 100] == 1         # the line is a hole -> filled


def test_grass_threshold_selects_green_only():
    g = S.hsv_threshold(_grass_with_white_line(), "grass")
    assert g[60, 100] == 1 and g[120, 100] == 0


def test_local_chroma_peaks_on_a_painted_line_and_local_l_on_a_white_line():
    da, _ = S.local_chroma(_ice_with_red_line(), win=41)
    assert da[120, 150] > 20 and abs(da[60, 150]) < 1
    dl = S.local_l(_grass_with_white_line(), win=41)
    assert dl[120, 150] > 30 and abs(dl[60, 150]) < 1


def test_robust_surface_covers_the_ice_but_not_the_line():
    rng = np.random.default_rng(0)
    img = np.clip(_ice_with_red_line().astype(int) + rng.normal(0, 3, (240, 320, 3)), 0, 255).astype(np.uint8)
    mask = S.robust_surface(img, surface="ice")
    assert mask[60:110].mean() > 0.9 and mask[118:123, 30:290].mean() < 0.2


def test_invalid_arguments_are_rejected():
    img = _ice_with_red_line()
    with pytest.raises(ValueError):
        S.hsv_threshold(img, "clay")
    with pytest.raises(ValueError):
        S.robust_surface(img, space="rgb")
    with pytest.raises(ValueError):
        S.robust_surface(img, space="lab", channels="xz")


def test_solidity_flags_a_broken_region():
    solid = np.zeros((100, 100), np.uint8)
    solid[20:80, 20:80] = 1
    notched = solid.copy()
    notched[20:50, 50:80] = 0  # L-shape: a quarter removed
    assert S.solidity(solid) > 0.99 and S.solidity(notched) < 0.9


def test_central_fraction_of_a_full_mask_is_one_and_of_an_empty_one_is_zero():
    h, w = 200, 300
    assert S.central_fraction(np.ones((h, w), np.uint8)) == 1.0
    assert S.central_fraction(np.zeros((h, w), np.uint8)) == 0.0


def test_central_fraction_sees_a_hole_in_the_middle_even_at_the_same_total_coverage():
    """The close-up case: surface present all around a central subject, at coverage similar to a wide shot, but
    empty right in the middle -- total coverage alone cannot tell the two apart, central_fraction can."""
    h, w = 300, 400
    wide = np.ones((h, w), np.uint8)
    wide[:60] = 0                                    # the stands band along the top: still ~80% coverage
    close_up = np.ones((h, w), np.uint8)
    close_up[80:260, 120:280] = 0                     # a big central subject, tuned to about the same coverage
    assert abs(wide.mean() - close_up.mean()) < 0.05   # same ballpark of total coverage...
    assert S.central_fraction(wide) > 0.9 and S.central_fraction(close_up) < 0.1   # ...but very different centres


def test_wide_shot_score_separates_a_surface_shot_from_a_close_up():
    h, w = 300, 400
    surf_wide = np.ones((h, w), np.uint8)
    surf_wide[:60] = 0
    region_wide = S.play_region(surf_wide)
    surf_close = np.ones((h, w), np.uint8)
    surf_close[70:260, 110:290] = 0                   # large central subject: breaks solidity of what remains too
    cov_w, cen_w, sol_w = S.wide_shot_score(surf_wide)
    cov_c, cen_c, sol_c = S.wide_shot_score(surf_close)
    assert cen_w > 0.85 and cen_c < 0.2
    assert sol_w > 0.95 and sol_c < sol_w


def test_integrate_and_normals():
    R = np.zeros((100, 100), np.float32)
    R[50, :] = 1.0
    line = np.stack([np.linspace(5, 95, 90), np.full(90, 50.0)], 1)
    assert S.integrate(R, line, 100, 100) == 1.0
    assert S.integrate(R, line[:10], 100, 100) is None  # too few in-frame points
    n = S.normals(line)
    assert np.allclose(np.abs(n[:, 1]), 1.0) and np.allclose(n[:, 0], 0.0)


def test_robust_surface_model_contract():
    """Consumers (the classical CV lab app) read these keys to draw the fitted Gaussian; renaming one
    silently broke the app once, and no other test noticed."""
    rng = np.random.default_rng(0)  # noise: EM cannot fit a GMM to two perfectly flat colours
    img = np.clip(_ice_with_red_line().astype(int) + rng.normal(0, 3, (240, 320, 3)), 0, 255).astype(np.uint8)
    mask, model = S.robust_surface(img, surface="ice", info=True)
    assert mask.shape == img.shape[:2]
    assert set(model) == {"space", "idx", "mu", "cov", "chi2"}
    assert model["space"] == "lab" and len(model["mu"]) == len(model["idx"]) == 3
    _, gmm_info = S.gmm_surface(img, surface="ice")
    assert {"L", "chroma", "w", "sel", "means", "var"} <= set(gmm_info)
