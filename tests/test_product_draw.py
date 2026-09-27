"""The product's drawing for any sport (ADR 0006): minimap orientation, field lines behind the camera."""
import numpy as np

from sportcal.product.draw import field_lines, minimap
from sportcal.sports import get


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
