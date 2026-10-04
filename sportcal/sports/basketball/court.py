"""Basketball court template: FIBA 28 x 15 m, its painted lines and 26 keypoints.

World frame: metres, origin at the centre of the court, X along the length (baselines at +-14), Y across the width
(sidelines at +-7.5). Dimensions from the FIBA rules: lane (restricted area) 4.9 m wide and 5.80 m from the baseline to
the free-throw line, free-throw and centre circles of radius 1.80 m, three-point line at 6.75 m from the basket (centre
1.575 m from the baseline) with 0.90 m straight stretches along each sideline. The restricted-area arc and the dashed
half of the free-throw circle are left out: short or broken, they carry little evidence.
"""
import numpy as np

from sportcal.sports.base import Sport, register

HALF_LENGTH, HALF_WIDTH = 14.0, 7.5
LANE_HALF_WIDTH = 2.45          # lane 4.9 m wide
FREE_THROW_X = 5.80             # free-throw line, from the baseline
CIRCLE_R = 1.80                 # centre circle and free-throw circle
BASKET_X = 1.575                # basket centre, from the baseline
ARC_R = 6.75                    # three-point line radius
CORNER_Y = HALF_WIDTH - 0.90    # the straight three-point stretches sit 0.90 m inside the sideline
ARC_START_X = BASKET_X + np.sqrt(ARC_R ** 2 - CORNER_Y ** 2)    # where the arc leaves the straight stretch: 2.99 m


def _arc(cx, cy, r, a0, a1, n=40):
    t = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def polylines():
    """{name: (n, 2) polyline in metres} of every painted line; the straight ones have two points."""
    hx, hy, ly, fx, cy = HALF_LENGTH, HALF_WIDTH, LANE_HALF_WIDTH, FREE_THROW_X, CORNER_Y
    P = {"sideline_neg": [(-hx, -hy), (hx, -hy)], "sideline_pos": [(-hx, hy), (hx, hy)],
         "halfway": [(0.0, -hy), (0.0, hy)], "centre_circle": _arc(0.0, 0.0, CIRCLE_R, 0.0, 2 * np.pi)}
    for n, s in (("neg", -1.0), ("pos", 1.0)):
        end = hx * s
        P["baseline_" + n] = [(end, -hy), (end, hy)]
        P["lane_side_pos_" + n] = [(end, ly), (s * (hx - fx), ly)]
        P["lane_side_neg_" + n] = [(end, -ly), (s * (hx - fx), -ly)]
        P["free_throw_line_" + n] = [(s * (hx - fx), -ly), (s * (hx - fx), ly)]
        P["corner_three_pos_" + n] = [(end, cy), (s * (hx - ARC_START_X), cy)]
        P["corner_three_neg_" + n] = [(end, -cy), (s * (hx - ARC_START_X), -cy)]
        a = np.arcsin(cy / ARC_R)       # the arc, seen from the basket, spans pi -+ a around the direction to midcourt
        arc = _arc(0.0, 0.0, ARC_R, np.pi - a, np.pi + a)
        P["three_point_arc_" + n] = arc * [s, 1.0] + [s * (hx - BASKET_X), 0.0]
        h = _arc(0.0, 0.0, CIRCLE_R, np.pi / 2, 3 * np.pi / 2)  # the solid half of the free-throw circle, towards midcourt
        P["free_throw_circle_" + n] = h * [s, 1.0] + [s * (hx - fx), 0.0]
    return {k: np.asarray(v, float) for k, v in P.items()}


def _keypoints():
    hx, hy, ly, fx, cy = HALF_LENGTH, HALF_WIDTH, LANE_HALF_WIDTH, FREE_THROW_X, CORNER_Y
    pts = [(0.0, hy), (0.0, -hy), (0.0, CIRCLE_R), (0.0, -CIRCLE_R)]           # halfway line: ends, centre circle
    for s in (-1.0, 1.0):
        pts += [(s * hx, hy), (s * hx, -hy),                                         # court corners
                (s * hx, ly), (s * hx, -ly), (s * (hx - fx), ly), (s * (hx - fx), -ly),    # lane corners
                (s * hx, cy), (s * hx, -cy), (s * (hx - ARC_START_X), cy), (s * (hx - ARC_START_X), -cy),
                (s * (hx - BASKET_X - ARC_R), 0.0)]                                  # corner threes; top of the arc
    return np.asarray(pts, float)


KEYPOINT_COORDS = _keypoints()
_NAMES = list(polylines())
_STRAIGHT = [n for n, pl in polylines().items() if len(pl) == 2]
BASKETBALL = register(Sport(
    name="basketball-fiba", length=2 * HALF_LENGTH, width=2 * HALF_WIDTH, surface="court",
    line_classes=("background", *_NAMES),
    polylines=lambda: [(1 + _NAMES.index(n), pl) for n, pl in polylines().items()],
    keypoints=lambda: KEYPOINT_COORDS,
    straight_lines=lambda: [(tuple(polylines()[n][0]), tuple(polylines()[n][1])) for n in _STRAIGHT],
    centred=True,
))
