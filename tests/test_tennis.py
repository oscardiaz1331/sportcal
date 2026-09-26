import numpy as np

from sportcal import sports
from sportcal.lab.tennis.tennis_h import fit, split_of
from sportcal.sports.tennis import court as C


def test_keypoints_are_line_crossings_and_the_set_survives_the_court_mirrors():
    lines = np.vstack([pl for pl in C.polylines().values()])
    K = sports.get("tennis-itf").keypoints()
    assert len(K) == 14 and all(np.linalg.norm(lines - k, axis=1).min() < 1e-9 for k in K)   # every point ends a segment
    for S in (np.diag([-1.0, 1.0]), np.diag([1.0, -1.0])):
        assert np.abs((K @ S)[:, None] - K[None]).sum(-1).min(1).max() < 1e-9


def test_fit_recovers_the_homography_and_reports_the_miss():
    H = np.array([[-20.0, 55.0, 640], [30.0, 4.0, 380], [0.03, 0.002, 1.0]])
    q = np.c_[C.KEYPOINT_COORDS, np.ones(14)] @ H.T
    kps = q[:, :2] / q[:, 2:]
    He, med, mx = fit(kps)
    assert np.allclose(He, H / H[2, 2], atol=1e-4) and mx < 1e-3
    kps[3] += (4.0, 0.0)                     # one point 4 px off at 1280; the fit spreads it, 2.4 px left at 1280
    assert 3.0 < fit(kps)[2] < 4.2          # reported at 1920


def test_split_is_by_video_and_stable():
    assert split_of("abc") == split_of("abc")
    s = [split_of("v%d" % i) for i in range(2000)]
    assert 0.07 < s.count("test") / 2000 < 0.13 and 0.07 < s.count("dev") / 2000 < 0.13
