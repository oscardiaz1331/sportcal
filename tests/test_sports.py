import numpy as np
import pytest

from sportcal import sports
from sportcal.sports.hockey import rink


def test_registry_lists_both_rink_standards_and_rejects_unknown_names():
    assert {"hockey-nhl", "hockey-iihf"} <= set(sports.names())
    with pytest.raises(KeyError, match="registered"):
        sports.get("curling")


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError):
        sports.register(sports.get("hockey-nhl"))


@pytest.mark.parametrize("name", ["hockey-nhl", "hockey-iihf"])
def test_every_line_class_has_geometry_inside_the_rink(name):
    s = sports.get(name)
    assert s.line_classes[0] == "background" and len(s.line_classes) == 12
    seen = set()
    for cls, poly in s.polylines():
        seen.add(cls)
        assert poly.shape[1] == 2 and np.isfinite(poly).all()
        assert poly[:, 0].min() >= -1e-6 and poly[:, 0].max() <= s.length + 1e-6
        assert poly[:, 1].min() >= -1e-6 and poly[:, 1].max() <= s.width + 1e-6
    assert seen == set(range(1, 12))  # classes 1..11, background has no geometry
    assert len(s.class_polylines(1)) == 8  # boards: 4 sides + 4 corner arcs


def test_template_has_56_defined_keypoints():
    for p in (rink.RINK_NHL, rink.RINK_IIHF):
        T = rink.build_template(p)
        assert T.shape == (56, 2) and np.isfinite(T).all()


def test_mirror_is_an_involution_and_matches_the_template_geometry():
    """MIRROR pairs must be the x-mirror of each other. The i+46 / i+26 formulas it replaced
    were wrong for most of zone A (111 px vs 5.6 px error, docs/experiments/hockey.md)."""
    assert sorted(rink.MIRROR) == list(range(56))
    assert all(rink.MIRROR[rink.MIRROR[i]] == i for i in range(56))
    for p in (rink.RINK_NHL, rink.RINK_IIHF):
        T = rink.build_template(p)
        for i, j in enumerate(rink.MIRROR):
            assert np.allclose(T[i], (p["length"] - T[j, 0], T[j, 1]), atol=1e-9), (i, j)


def test_iihf_measurements_match_the_rulebook():
    """Sanity anchors deduced from the dataset (postes +-0.915 m, hash marks 1.70 m apart)."""
    T = rink.build_template(rink.RINK_IIHF)
    assert abs(T[51, 1] - T[50, 1] - 1.83) < 1e-9        # goal posts
    assert abs(T[37, 1] - T[36, 1]) > 0                   # hash mark pair straddles the dot
    hash_gap = T[42, 0] - T[36, 0]
    assert abs(hash_gap - 1.70) < 1e-9


def test_neutral_faceoff_dots_are_inside_the_neutral_zone():
    """Regression: the sign was inverted once and dots landed 3 m outside the zone."""
    p = rink.RINK_NHL
    T = rink.build_template(p)
    blue_lo, blue_hi = p["blue_from_end"], p["length"] - p["blue_from_end"]
    for k in (22, 23):
        assert blue_lo < T[k, 0] < p["length"] / 2
    for k in (32, 33):
        assert p["length"] / 2 < T[k, 0] < blue_hi


def test_nhl_and_iihf_are_different_rinks():
    nhl, iihf = sports.get("hockey-nhl"), sports.get("hockey-iihf")
    assert (nhl.length, nhl.width) != (iihf.length, iihf.width)
    assert abs(nhl.length - 60.96) < 0.01 and abs(nhl.width - 25.908) < 0.01
    assert not np.allclose(nhl.keypoints(), iihf.keypoints())


def test_derived_keypoints_lie_on_painted_curves_and_survive_the_rink_mirrors():
    """Every derived point sits on a circle or a board corner, and mirroring the rink maps the set onto itself - what
    lets `canonicalize` rename them after a mirrored frame instead of teaching the model the wrong names."""
    p = rink.RINK_NHL
    L, W = p["length"], p["width"]
    D = rink.derived_keypoints(p)
    a, b = zip(*[(pl[:-1], pl[1:]) for c, pl in rink.rink_polylines(p) if c in (1, 7, 8, 9, 10, 11)])
    a, b = np.vstack(a), np.vstack(b)
    for q in D:          # distance to the nearest segment of any curve
        t = np.clip(((q - a) * (b - a)).sum(1) / ((b - a) ** 2).sum(1), 0, 1)
        assert np.linalg.norm(a + t[:, None] * (b - a) - q, axis=1).min() < 0.02, q
    for S in (np.array([[-1.0, 0], [0, 1]]), np.array([[1.0, 0], [0, -1]])):
        M = D @ S.T + np.array([L if S[0, 0] < 0 else 0, W if S[1, 1] < 0 else 0])
        assert np.abs(M[:, None] - D[None]).sum(-1).min(1).max() < 1e-9
    assert len(D) == 34 and np.abs(D[:, None] - rink.build_template(p)[None]).sum(-1).min() > 0.5   # no duplicate
