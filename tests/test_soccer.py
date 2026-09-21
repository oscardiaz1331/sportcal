import numpy as np
import pytest

from sportcal import sports
from sportcal.core.camera import pose_to_H
from sportcal.sports.soccer import field as F


def _dist_to_polyline(p, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


def test_soccer_is_registered_with_fifa_dimensions():
    s = sports.get("soccer-fifa")
    assert (s.length, s.width, s.surface) == (105.0, 68.0, "grass")
    assert s.line_classes[0] == "background" and len(s.line_classes) == 1 + len(F.polylines())
    assert {c for c, _ in s.polylines()} == set(range(1, len(s.line_classes)))


def test_every_painted_line_stays_inside_the_field():
    for name, pl in F.polylines().items():
        assert np.abs(pl[:, 0]).max() <= F.HALF_LENGTH + 1e-9 and np.abs(pl[:, 1]).max() <= F.HALF_WIDTH + 1e-9, name


def test_template_is_symmetric_under_a_half_turn():
    """The 180-degree symmetry that canonicalize_H exists for: the set of painted points maps onto itself."""
    pts, _ = F.sample_template(0.5)
    for p in pts[::17]:
        assert min(np.linalg.norm(pts - (-p), axis=1)) < 0.25


def test_named_keypoints_and_corners_are_where_the_lines_cross():
    assert len(F.KEYPOINTS) == 31 and F.KEYPOINT_COORDS.shape == (31, 2) and len(F.CORNERS) == 22
    lines = list(F.polylines().values())
    for x, y in F.CORNERS:  # a corner is a crossing of an X=const and a Y=const template line, and is painted
        assert x in F.X_LINES and y in F.Y_LINES, (x, y)
        assert min(_dist_to_polyline(np.array([x, y]), pl) for pl in lines) < 1e-6, (x, y)
    for name, xy in F.KEYPOINTS:
        if "spot" in name or "field centre" in name:
            continue  # painted dots, not line crossings
        assert min(_dist_to_polyline(np.array(xy), pl) for pl in lines) < 0.02, name


def test_the_penalty_arc_meets_the_penalty_area_line_where_the_keypoint_says():
    arc = F.polylines()["penalty_arc_right"]
    for end in (arc[0], arc[-1]):
        assert abs(end[0] - F.PENALTY_AREA_X) < 1e-6 and abs(abs(end[1]) - F.ARC_CROSS_Y) < 1e-6
    assert arc[:, 0].min() < F.PENALTY_AREA_X - 3  # the arc bulges out of the area, towards midfield


def test_canonicalize_H_fixes_the_camera_side_and_is_idempotent():
    pose = np.array([[0.0, -60.0, 20.0, 0.0, np.radians(22), 1500.0]])
    H_ok = pose_to_H(pose, 960, 540)[0]
    H_flipped = H_ok @ np.diag([-1.0, -1.0, 1.0])
    assert not np.allclose(H_ok, H_flipped)
    assert np.allclose(F.canonicalize_H(H_ok), H_ok)                          # already canonical
    assert np.allclose(F.canonicalize_H(H_flipped), H_ok)                     # the half-turn twin maps back
    assert np.allclose(F.canonicalize_H(F.canonicalize_H(H_flipped)), H_ok)  # idempotent


def test_sample_template_never_bridges_two_lines():
    pts, cont = F.sample_template(1.0)
    assert not cont[0]
    jumps = np.linalg.norm(np.diff(pts, axis=0), axis=1)[cont[1:]]
    assert jumps.max() <= 1.0 + 1e-6      # a "continuing" step is never longer than the sampling step
    n_lines = len(F.polylines())
    assert (~cont).sum() == n_lines       # exactly one break per painted line


@pytest.mark.parametrize("bad", ["hockey-nhl", "hockey-iihf"])
def test_sports_stay_independent(bad):
    assert sports.get("soccer-fifa").keypoints().shape != sports.get(bad).keypoints().shape


def test_ellipse_labelling_gestures_and_fit():
    import numpy as np
    from sportcal.core import circle as CI
    from sportcal.core.camera import pose_to_H
    from sportcal.lab.soccer import labeler as LB
    w, h = 1920, 1080
    H = pose_to_H(np.array([[-3.0, -68.0, 13.5, np.radians(-10), np.radians(12), 1.9 * w]]), w, h)[0]
    H = H / H[2, 2]
    seed = LB.elipse_desde_plantilla(H, w, h, n=12)
    assert 5 <= len(seed) <= 12 and all(0 < x < w and 0 < y < h for x, y in seed)   # only the part of the circle inside the frame
    # gestures: click adds, click on a point removes it, dragging from a point moves it, dragging from empty does nothing
    pts = LB.procesa_gesto_elipse([], (100, 100, 100, 100), 10, 5)
    pts = LB.procesa_gesto_elipse(pts, (300, 300, 300, 300), 10, 5)
    assert pts == [(100, 100), (300, 300)]
    assert LB.procesa_gesto_elipse(pts, (102, 99, 102, 99), 10, 5) == [(300, 300)]
    assert LB.procesa_gesto_elipse(pts, (100, 100, 150, 160), 10, 5)[0] == (150, 160)
    assert LB.procesa_gesto_elipse(pts, (700, 700, 800, 800), 10, 5) == pts
    # fit: centre spot + centre circle x halfway line + the ellipse outline give the field
    ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    ell = CI._project(H, np.c_[9.15 * np.cos(ang), 9.15 * np.sin(ang)])[0]
    world = np.array([[0.0, 0.0], [0.0, 9.15]])
    Hc, res, giro, aviso, Hn = LB.ajusta_elipse(world, CI._project(H, world)[0], ell, w, h)
    assert Hc is not None and max(res) < 0.5
    P = np.array([[-30.0, -20.0], [30.0, 20.0], [0.0, 30.0]])
    assert np.abs(CI._project(Hn, P)[0] - CI._project(H, P)[0]).max() < 1.0
    assert LB.ajusta_elipse(world, CI._project(H, world)[0], ell[:3], w, h)[0] is None      # < 5 points on the outline
    assert LB.ajusta_elipse(world[:1], CI._project(H, world)[0][:1], ell, w, h)[0] is None   # < 2 clicked points
