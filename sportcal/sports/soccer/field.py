"""Soccer field template: FIFA 105x68 m, the painted lines and the named labelling keypoints.

World frame: metres, origin at the field centre, X along the length (-52.5 .. 52.5), Y across the
width (-34 .. 34), Z up. The camera looks towards +Y from the Y<0 side, so "up" in the image is +Y
(far touchline) and +X is to the image right.

The field is symmetric under a 180-degree turn (X, Y -> -X, -Y): H and H @ Rot180 draw the SAME lines
and only swap which goal / touchline is which. Nothing in the painted lines breaks the tie, so by
convention the camera side is fixed (near touchline at the bottom) - see `canonicalize_H`.
"""
import numpy as np

from sportcal.core.camera import project
from sportcal.sports.base import Sport, register

HALF_LENGTH, HALF_WIDTH = 52.5, 34.0
CIRCLE_RADIUS = 9.15  # centre circle and penalty arcs
PENALTY_SPOT_X = 41.5
PENALTY_AREA_X, PENALTY_AREA_Y = 36.0, 20.16  # 16.5 m deep, 40.32 m wide
GOAL_AREA_X, GOAL_AREA_Y = 47.0, 9.16         # 5.5 m deep, 18.32 m wide

# Lines X=const and Y=const of the template (goal line, goal area, penalty area, halfway line / touchlines, areas)
X_LINES = (-52.5, -47.0, -36.0, 0.0, 36.0, 47.0, 52.5)
Y_LINES = (-34.0, -20.16, -9.16, 9.16, 20.16, 34.0)
# Centre circle and the two penalty arcs (same radius, centred on the spots)
CIRCLE_CENTERS = ((0.0, 0.0), (-PENALTY_SPOT_X, 0.0), (PENALTY_SPOT_X, 0.0))
# Crossings of two template lines (field corners 4, halfway x touchlines 2, and per side: penalty area x goal
# line 2, penalty-area corners 2, goal area x goal line 2, goal-area corners 2). No circle / spot points.
CORNERS = np.array(
    [(-HALF_LENGTH, -HALF_WIDTH), (HALF_LENGTH, -HALF_WIDTH), (-HALF_LENGTH, HALF_WIDTH),
     (HALF_LENGTH, HALF_WIDTH), (0.0, -HALF_WIDTH), (0.0, HALF_WIDTH)]
    + [(sg * x, sy * y) for sg in (-1, 1)
       for x, y in ((HALF_LENGTH, PENALTY_AREA_Y), (PENALTY_AREA_X, PENALTY_AREA_Y),
                    (HALF_LENGTH, GOAL_AREA_Y), (GOAL_AREA_X, GOAL_AREA_Y))
       for sy in (-1, 1)], float)

# Where the penalty arc meets the penalty-area line (the area reaches 5.5 m beyond the spot's circle)
ARC_CROSS_Y = float(np.sqrt(CIRCLE_RADIUS ** 2 - 5.5 ** 2))


def _arc(cx, cy, r, a0, a1, n=60):
    t = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def polylines():
    """{name: (N, 2) points in metres} of every painted line."""
    hx, hy = HALF_LENGTH, HALF_WIDTH
    P = {
        "near_touchline": [(-hx, -hy), (hx, -hy)], "far_touchline": [(-hx, hy), (hx, hy)],
        "left_goal_line": [(-hx, -hy), (-hx, hy)], "right_goal_line": [(hx, -hy), (hx, hy)],
        "halfway_line": [(0, -hy), (0, hy)],
        "center_circle": _arc(0, 0, CIRCLE_RADIUS, 0, 2 * np.pi, 90),
    }
    a = np.arccos(5.5 / CIRCLE_RADIUS)  # the arc is what lies outside the penalty area
    for s, side in ((-1, "left"), (1, "right")):
        py, gy = PENALTY_AREA_Y, GOAL_AREA_Y
        P["penalty_area_" + side] = [(s * hx, -py), (s * PENALTY_AREA_X, -py), (s * PENALTY_AREA_X, py), (s * hx, py)]
        P["goal_area_" + side] = [(s * hx, -gy), (s * GOAL_AREA_X, -gy), (s * GOAL_AREA_X, gy), (s * hx, gy)]
        centre = np.pi if s == 1 else 0  # arc centred on the spot, facing the middle of the field
        P["penalty_arc_" + side] = _arc(s * PENALTY_SPOT_X, 0, CIRCLE_RADIUS, centre - a, centre + a, 40)
    return {k: np.asarray(v, float) for k, v in P.items()}


def sample_template(step=0.5):
    """Points along the whole template every `step` m. Returns (pts (N, 2), cont (N,) bool):
    cont[j] means sample j continues the segment from sample j-1 (same polyline), so lengths
    are never measured across a jump between two lines."""
    pts, cont = [], []
    for pl in polylines().values():
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
        s = np.arange(0, d[-1] + 1e-9, step)
        p = np.stack([np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])], 1)
        pts.append(p)
        c = np.ones(len(p), bool)
        c[0] = False
        cont.append(c)
    return np.concatenate(pts), np.concatenate(cont)


def canonicalize_H(H):
    """Apply the 180-degree world turn if H leaves +Y (far touchline) pointing DOWN in the image,
    so every stored/estimated H follows the "near touchline at the bottom" convention."""
    xy, depth = project(np.asarray(H, float)[None], np.array([[0.0, 0.0], [0.0, 10.0]]))
    if (depth[0] > 0).all() and np.isfinite(xy).all() and xy[0, 1, 1] > xy[0, 0, 1]:
        return np.asarray(H, float) @ np.diag([-1.0, -1.0, 1.0])
    return np.asarray(H, float)


def _keypoints():
    hx, hy = HALF_LENGTH, HALF_WIDTH
    k = [("near-left corner", (-hx, -hy)), ("near-right corner", (hx, -hy)),
         ("far-left corner", (-hx, hy)), ("far-right corner", (hx, hy)),
         ("halfway line x near touchline", (0.0, -hy)), ("halfway line x far touchline", (0.0, hy)),
         ("field centre", (0.0, 0.0)),
         ("center circle x halfway line (near)", (0.0, -CIRCLE_RADIUS)),
         ("center circle x halfway line (far)", (0.0, CIRCLE_RADIUS))]
    for s, side in ((-1, "left"), (1, "right")):
        px, py, gx, gy = PENALTY_AREA_X, PENALTY_AREA_Y, GOAL_AREA_X, GOAL_AREA_Y
        k += [(side + ": penalty area x goal line (near)", (s * hx, -py)),
              (side + ": penalty area x goal line (far)", (s * hx, py)),
              (side + ": penalty area corner (near)", (s * px, -py)),
              (side + ": penalty area corner (far)", (s * px, py)),
              (side + ": goal area x goal line (near)", (s * hx, -gy)),
              (side + ": goal area x goal line (far)", (s * hx, gy)),
              (side + ": goal area corner (near)", (s * gx, -gy)),
              (side + ": goal area corner (far)", (s * gx, gy)),
              (side + ": penalty spot", (s * PENALTY_SPOT_X, 0.0)),
              (side + ": penalty arc x penalty-area line (near)", (s * px, -ARC_CROSS_Y)),
              (side + ": penalty arc x penalty-area line (far)", (s * px, ARC_CROSS_Y))]
    return k


def derived_keypoints():
    """Points on the circles that no two lines cross: the ends of the centre circle's diameter along X, its four points
    at 45 degrees, and the apex of each penalty arc. A centre-circle view otherwise has every keypoint on the halfway
    line, which leaves the homography undetermined. Closed under the field's mirrors in X and in Y."""
    r, c = CIRCLE_RADIUS, CIRCLE_RADIUS / np.sqrt(2)
    return np.array([(-r, 0.0), (r, 0.0), (-c, -c), (c, -c), (-c, c), (c, c),
                     (-(PENALTY_SPOT_X - r), 0.0), (PENALTY_SPOT_X - r, 0.0)], float)


KEYPOINTS = _keypoints()  # [(name, (X, Y)), ...] - the points a human clicks when labelling
KEYPOINT_NAMES = {i: name for i, (name, _) in enumerate(KEYPOINTS)}
KEYPOINT_COORDS = np.array([xy for _, xy in KEYPOINTS], float)

_NAMES = list(polylines())
SOCCER = register(Sport(
    name="soccer-fifa", length=2 * HALF_LENGTH, width=2 * HALF_WIDTH, surface="grass",
    line_classes=("background", *_NAMES),
    polylines=lambda: [(1 + _NAMES.index(n), pl) for n, pl in polylines().items()],
    keypoints=lambda: KEYPOINT_COORDS,
))
