import numpy as np

from sportcal import sports
from sportcal.lab.basketball import deepsport_h as D
from sportcal.sports.basketball import court as C


def test_curves_lie_where_the_rules_put_them_and_the_keypoints_survive_the_mirrors():
    P = C.polylines()
    top = C.HALF_LENGTH - C.BASKET_X - C.ARC_R                      # the arc's point nearest midcourt: 5.675 m from the centre
    for s, n in ((1.0, "pos"), (-1.0, "neg")):
        basket = np.array([s * (C.HALF_LENGTH - C.BASKET_X), 0.0])
        arc = P["three_point_arc_" + n]
        assert np.allclose(np.linalg.norm(arc - basket, axis=1), C.ARC_R)
        assert (arc[:, 0] * s > 0).all() and np.isclose(np.abs(arc[:, 0]).min(), top, atol=0.01)   # on its own half, apex towards midcourt
        assert np.allclose(sorted(arc[[0, -1], 1]), [-C.CORNER_Y, C.CORNER_Y])              # it ends on the straight stretches
        assert np.allclose(arc[0], P["corner_three_%s_%s" % ("pos" if arc[0, 1] > 0 else "neg", n)][1])
        ft = P["free_throw_circle_" + n]
        centre = np.array([s * (C.HALF_LENGTH - C.FREE_THROW_X), 0.0])
        assert np.allclose(np.linalg.norm(ft - centre, axis=1), C.CIRCLE_R) and ((ft[:, 0] - centre[0]) * s <= 1e-9).all()
    K = sports.get("basketball-fiba").keypoints()
    assert len(K) == 26 and len(sports.get("basketball-fiba").straight_lines()) == 15
    for S in (np.diag([-1.0, 1.0]), np.diag([1.0, -1.0])):
        assert np.abs((K @ S)[:, None] - K[None]).sum(-1).min(1).max() < 1e-9


def test_calibration_gives_the_homography_of_the_court_plane_and_padding_shifts_it():
    """A pinhole camera in DeepSportRadar's units (cm, origin at a court corner): `calib_to_H` projects the corners and
    the centre of the court (in metres, centred) where K (R X + T) does."""
    a, b = 0.3, 2.0
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    R, T = Rz @ Rx, np.array([-200.0, -300.0, 3000.0])
    K = np.array([[1800.0, 0, 811.5], [0, 1800.0, 616.5], [0, 0, 1]])
    H = D.calib_to_H({"KK": K.ravel().tolist(), "R": R.ravel().tolist(), "T": T.tolist()})
    for cm, m in (([0, 0], [-14, -7.5]), ([2800, 1500], [14, 7.5]), ([1400, 750], [0, 0]), ([2800, 0], [14, -7.5])):
        p, q = K @ (R @ np.r_[cm, 0.0] + T), H @ np.r_[m, 1.0]
        assert np.allclose(p[:2] / p[2], q[:2] / q[2], atol=1e-6)
    img, Hp = D.pad_to_16_9(np.zeros((1234, 1624, 3), np.uint8), H)
    assert img.shape[1] == round(1234 * 16 / 9)
    p, q = H @ [0, 0, 1.0], Hp @ [0, 0, 1.0]
    assert np.allclose(q[:2] / q[2], p[:2] / p[2] + [(img.shape[1] - 1624) // 2, 0])


def test_splits_keep_the_challenge_test_arenas_and_never_share_an_arena():
    arenas = ["KS-FR-%s" % a for a in "ABCDEFGHIJKL"] + list(D.TEST_ARENAS)
    s = D.splits(arenas)
    assert all(s[a] == "test" for a in D.TEST_ARENAS) and list(s.values()).count("dev") == D.N_DEV_ARENAS
    assert D.splits(arenas) == s
