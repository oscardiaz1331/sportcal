"""Tennis court template: ITF doubles court 23.77 x 10.97 m, its painted lines and 14 keypoints.

World frame: metres, origin at the centre of the net, X along the length (baselines at +-11.885), Y across the width
(doubles sidelines at +-5.485). Broadcast cameras sit behind a baseline, so the long axis runs up the image (an end view
in `core.camera.canonical_mirror`'s terms). The net and the centre marks are left out: the net is not on the ground and
the marks are 10 cm long.
"""
import numpy as np

from sportcal.sports.base import Sport, register

HALF_LENGTH, HALF_WIDTH = 11.885, 5.485
SINGLES_Y = 4.115           # singles sidelines
SERVICE_X = 6.40            # service lines, from the net


def polylines():
    """{name: (2, 2) segment in metres} of every painted line (all straight)."""
    hx, hy, sy, sx = HALF_LENGTH, HALF_WIDTH, SINGLES_Y, SERVICE_X
    P = {"baseline_neg": [(-hx, -hy), (-hx, hy)], "baseline_pos": [(hx, -hy), (hx, hy)],
         "doubles_side_neg": [(-hx, -hy), (hx, -hy)], "doubles_side_pos": [(-hx, hy), (hx, hy)],
         "singles_side_neg": [(-hx, -sy), (hx, -sy)], "singles_side_pos": [(-hx, sy), (hx, sy)],
         "service_line_neg": [(-sx, -sy), (-sx, sy)], "service_line_pos": [(sx, -sy), (sx, sy)],
         "centre_service_line": [(-sx, 0.0), (sx, 0.0)]}
    return {k: np.asarray(v, float) for k, v in P.items()}


# The 14 points of the TennisCourtDetector annotations, in their order: 0-3 doubles corners (far left, far right, near
# left, near right), 4-7 singles sideline x baseline (far left, near left, far right, near right), 8-11 singles sideline
# x service line (far left, far right, near left, near right), 12-13 centre service line x service line (far, near).
# "Far" is -X, "left" +Y: the naming `canonical_mirror` gives a camera behind the +X baseline.
_hx, _hy, _sy, _sx = HALF_LENGTH, HALF_WIDTH, SINGLES_Y, SERVICE_X
KEYPOINT_COORDS = np.array([(-_hx, _hy), (-_hx, -_hy), (_hx, _hy), (_hx, -_hy),
                            (-_hx, _sy), (_hx, _sy), (-_hx, -_sy), (_hx, -_sy),
                            (-_sx, _sy), (-_sx, -_sy), (_sx, _sy), (_sx, -_sy),
                            (-_sx, 0.0), (_sx, 0.0)], float)

_NAMES = list(polylines())
# ponytail: surface "court" has no colour model in core.surface (hard, clay and grass courts differ); the keypoint model
# does not need one. Add one there if a surface-based method is ever tried on tennis.
TENNIS = register(Sport(
    name="tennis-itf", length=2 * HALF_LENGTH, width=2 * HALF_WIDTH, surface="court",
    line_classes=("background", *_NAMES),
    polylines=lambda: [(1 + _NAMES.index(n), pl) for n, pl in polylines().items()],
    keypoints=lambda: KEYPOINT_COORDS,
))
