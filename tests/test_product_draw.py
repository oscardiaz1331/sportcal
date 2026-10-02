"""The product's drawing for any sport (ADR 0006): minimap orientation, field lines behind the camera."""
import numpy as np

from sportcal.product.draw import ball_trail, field_lines, minimap
from sportcal.sports import get


def test_the_ball_shows_on_ice():
    """A white puck on near-white ice: the dark ring around it is what makes it visible."""
    ice = np.full((60, 60, 3), 245, np.uint8)
    ball_trail(ice, [((30, 30), (0, 200, 0))], here=True)
    assert ice.min() < 100
    unseen = np.full((60, 60, 3), 245, np.uint8)
    ball_trail(unseen, [((30, 30), (0, 200, 0))], here=False)      # not detected in this frame: no ball drawn
    assert (unseen == 245).all()


def test_minimap_puts_the_field_corners_on_its_corners_the_way_the_main_camera_sees_them():
    hockey, soccer = get("hockey-nhl"), get("soccer-fifa")
    x0, x1, y0, y1 = hockey.box                      # corner origin; +y points down in the main camera
    cases = [(hockey, (x0, y0), (x1, y1))]
    x0, x1, y0, y1 = soccer.box                      # centred; +y (the far touchline) points up
    cases.append((soccer, (x0, y1), (x1, y0)))
    for sport, top_left, bottom_right in cases:
        img, to_px = minimap(sport, 400)
        h, w = img.shape[:2]
        assert w == 400, sport.name
        (xa, ya), (xb, yb) = to_px([top_left, bottom_right])
        assert xa == ya and 0 < xa < 40, sport.name            # the same margin on every side
        assert np.allclose((xb, yb), (w - xa, h - xa), atol=1), sport.name


def test_an_upright_minimap_with_an_apron_has_room_behind_the_baselines():
    """Tennis: the camera looks along the court - the far baseline (-x) at the top, +y on the left - and the players
    stand metres behind the baselines, which a minimap of the bare court would leave out."""
    court = get("tennis-itf")
    x0, x1, y0, y1 = court.box
    img, to_px = minimap(court, 300, 400, apron=5.0, upright=True)
    h, w = img.shape[:2]
    assert w < h <= 400 and w <= 300                                  # stands on end, inside the box it was given
    far_left, near_right, behind = to_px([(x0, y1), (x1, y0), (x1 + 4.0, 0.0)])
    assert abs(far_left[0] - far_left[1]) < 1e-6 and far_left[0] > 20  # the apron, the same on every side
    assert np.allclose(near_right, (w - far_left[0], h - far_left[1]), atol=1)
    assert near_right[1] < behind[1] < h and abs(behind[0] - w / 2) <= 1   # 4 m behind the near baseline: still on it
    flat, _ = minimap(court, 300, 400, apron=5.0)
    assert flat.shape[1] > flat.shape[0]                              # the same court lying down when not upright


def test_field_lines_keep_only_what_is_in_front_of_the_camera():
    """A camera that sees the rink up to x = 50 m: the last 11 m lie behind it and would come back mirrored, on the
    wrong side of the image (negative x), if they were drawn."""
    sport = get("hockey-nhl")
    H = np.array([[10.0, 0, 0], [0, 10.0, 0], [-1 / 50, 0, 1]])     # w = 1 - x/50: behind the camera past x = 50
    runs = field_lines(H, sport, 1920, 1080)
    pts = np.vstack(runs)
    assert len(runs) > 0 and np.isfinite(pts).all()
    assert (pts[:, 0] >= 0).all()                                     # nothing mirrored from behind the camera
    front = field_lines(np.diag([10.0, 10.0, 1.0]), sport, 1920, 1080)  # the whole rink in front
    assert sum(map(len, runs)) < sum(map(len, front))                  # the part behind was dropped, not kept
