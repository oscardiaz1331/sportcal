"""Hockey: two rink standards, registered as separate sports because the same
keypoint index lands on different world coordinates (see rink.py)."""
from sportcal.sports.base import Sport, register
from sportcal.sports.hockey import rink


def _rink_sport(name, params):
    return register(Sport(
        name=name, length=params["length"], width=params["width"], surface="ice",
        line_classes=tuple(rink.CLASSES),
        polylines=lambda: rink.rink_polylines(params),
        keypoints=lambda: rink.build_template(params),
    ))


NHL = _rink_sport("hockey-nhl", rink.RINK_NHL)
IIHF = _rink_sport("hockey-iihf", rink.RINK_IIHF)
