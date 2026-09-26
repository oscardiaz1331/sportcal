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


def test_pnlcalib_camera_is_converted_to_our_world_convention():
    """SoccerNet's y points towards the main camera, ours towards the far touchline: a reflection that
    canonicalize_H cannot undo. A point off both axes must land where SoccerNet projects (X, -Y, 0)."""
    from sportcal.lab.soccer.pnlcalib_eval import to_H
    C = np.array([0.0, 60.0, -15.0])            # SoccerNet frame: near side, 15 m up (z points down)
    f = -C / np.linalg.norm(C)                   # optical axis towards the centre spot
    right = np.cross([0.0, 0.0, 1.0], f)
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(f, right), f])  # camera x right, y down, z forward
    P = np.array([[1500.0, 0, 960], [0, 1500, 540], [0, 0, 1]]) @ R @ np.c_[np.eye(3), -C]
    X, Y = 20.0, 10.0                            # far-right quadrant, ours
    q = P @ [X, -Y, 0.0, 1.0]
    p = to_H(P) @ [X, Y, 1.0]
    assert np.allclose(p[:2] / p[2], q[:2] / q[2], atol=1e-6)


def test_a_soccernet_style_annotation_gives_back_the_camera():
    """Points on the markings, in SoccerNet's format (normalised, its class names), made from a known camera: the line
    DLT must recover that camera, and a wrongly named line must not pass as a good fit."""
    from sportcal.core.camera import pose_to_H
    from sportcal.lab.soccer import soccernet_h as SN
    w, h = 960, 540
    H = pose_to_H(np.array([[-10.0, -60.0, 18.0, np.radians(-15), np.radians(15), 1.6 * w]]), w, h)[0]
    H /= H[2, 2]

    def pts(a, b, n=6):
        s = np.linspace(a, b, n)
        q = np.c_[s, np.ones(n)] @ H.T
        q = q[:, :2] / q[:, 2:]
        return [{"x": x / w, "y": y / h} for x, y in q if 0 < x < w and 0 < y < h]
    ann = {name: p for name, (a, b) in SN.LINES.items() if len(p := pts(a, b)) >= 2}
    assert len(ann) >= 4
    H2, resid, n, _ = SN.fit(ann, w, h)
    grid = np.array([[x, y, 1.0] for x in np.linspace(-40, 20, 7) for y in np.linspace(-30, 30, 7)])
    p1, p2 = grid @ H.T, grid @ (H2 / H2[2, 2]).T
    assert np.abs(p1[:, :2] / p1[:, 2:] - p2[:, :2] / p2[:, 2:]).max() < 0.5 and resid < 0.5
    swapped = dict(ann)
    keys = [k for k in ann if k.startswith("Big rect.") and k.endswith("top")] + ["Side line top"]
    swapped[keys[0]], swapped[keys[1]] = ann[keys[1]], ann[keys[0]]
    rng = np.random.default_rng(0)                 # 1 px of click noise shows up in the residual
    noisy = {k: [{"x": q["x"] + rng.normal(0, 1 / w), "y": q["y"] + rng.normal(0, 1 / h)} for q in v] for k, v in ann.items()}
    assert 0.3 < SN.fit(noisy, w, h)[1] < 2
    r = SN.fit(swapped, w, h)
    assert r is None or r[1] > 5                   # refused, or it no longer fits its own points


def test_a_centre_circle_view_with_two_lines_gives_back_the_camera():
    """A view with the centre circle, the halfway line and the far touchline only (too few lines for `fit`): the fit that
    also uses the circle points recovers the camera; with the halfway line alone it refuses (a circle and its diameter
    leave H one degree of freedom)."""
    from sportcal.lab.soccer import soccernet_h as SN
    w, h = 960, 540
    H = pose_to_H(np.array([[0.0, -60.0, 18.0, 0.0, np.radians(15), 3 * w]]), w, h)[0]
    H /= H[2, 2]

    def pts(world):
        q = np.c_[world, np.ones(len(world))] @ H.T
        q = q[:, :2] / q[:, 2:]
        return [{"x": x / w, "y": y / h} for x, y in q if 0 < x < w and 0 < y < h]
    ann = {"Middle line": pts(np.linspace((0, -34), (0, 34), 12)), "Side line top": pts(np.linspace((-52.5, 34), (52.5, 34), 30)),
           "Circle central": pts(F._arc(0, 0, F.CIRCLE_RADIUS, 0.3, 5.9, 14))}
    assert len(ann["Side line top"]) >= 2 and SN.fit(ann, w, h) is None
    H2, resid, n, resid_c = SN.fit_circle(ann, w, h)
    # these markings are symmetric about the halfway line: either twin fits them, the naming rule picks one
    H, H2 = (SN.canonical_mirror(M, (0.0, 0.0), y_down=False)[0] for M in (H, H2))
    grid = np.array([[x, y, 1.0] for x in np.linspace(-30, 30, 7) for y in np.linspace(-20, 34, 7)])
    p1, p2 = grid @ H.T, grid @ H2.T
    assert np.abs(p1[:, :2] / p1[:, 2:] - p2[:, :2] / p2[:, 2:]).max() < 0.5 and n == 2 and resid < 0.1 and resid_c < 0.1
    ann.pop("Side line top")
    assert SN.fit_circle(ann, w, h, min_lines=1) is None
