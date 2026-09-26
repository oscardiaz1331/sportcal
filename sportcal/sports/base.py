"""What a sport must provide so that core/, lab/ and product/ can work with it.

A `Sport` is pure geometry + naming - no models, no data paths. To add a sport:
  1. create `sportcal/sports/<name>/` with its template (world metres, X along the
     length, Y across the width);
  2. build one or more `Sport` instances and `register()` them in its `__init__.py`;
  3. import that subpackage from `sportcal/sports/__init__.py`.
Nothing in core/ needs to change.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from sportcal.core.camera import canonical_mirror


@dataclass(frozen=True)
class Sport:
    name: str                      # registry key, e.g. "hockey-nhl"
    length: float                  # metres along X
    width: float                   # metres along Y
    surface: str                   # "ice" | "grass": key for core.surface
    line_classes: tuple            # class names; index 0 is background
    polylines: Callable[[], list]  # -> [(class index, (N, 2) world polyline in metres), ...]
    keypoints: Optional[Callable[[], np.ndarray]] = None  # -> (K, 2) world metres, NaN = undefined
    # what a keypoint + line model needs on top (models/kpline.py, lab/common/train_kpline.py):
    straight_lines: Optional[Callable[[], list]] = None      # -> [(a, b)]: the straight painted lines, world segments
    derived_keypoints: Optional[Callable[[], np.ndarray]] = None  # -> (D, 2): extra points on curves, set "derived"
    centred: bool = False          # world origin at the field centre (else at a corner)
    y_down: bool = True            # naming rule of `canonical_mirror`: in a side view +y points down (else up)

    def class_polylines(self, cls):
        """All polylines of one line class."""
        return [pl for c, pl in self.polylines() if c == cls]

    @property
    def box(self):
        """World extent (x0, x1, y0, y1) in metres."""
        if self.centred:
            return (-self.length / 2, self.length / 2, -self.width / 2, self.width / 2)
        return (0.0, self.length, 0.0, self.width)

    def keypoint_set(self, name):
        """(K, 2) world points of a named set: "base" (the labelling keypoints) or "derived" (base + derived)."""
        if name == "base":
            return self.keypoints()
        if name == "derived" and self.derived_keypoints is not None:
            return np.vstack([self.keypoints(), self.derived_keypoints()])
        raise KeyError("{} has no keypoint set {!r}".format(self.name, name))

    def canonicalize(self, H):
        """H renamed to the image naming rule of `core.camera.canonical_mirror` about the field centre."""
        x0, x1, y0, y1 = self.box
        return canonical_mirror(H, ((x0 + x1) / 2, (y0 + y1) / 2), y_down=self.y_down)[0]


REGISTRY = {}


def register(sport):
    if sport.name in REGISTRY:
        raise ValueError("sport {!r} already registered".format(sport.name))
    REGISTRY[sport.name] = sport
    return sport


def get(name):
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError("unknown sport {!r}; registered: {}".format(name, sorted(REGISTRY))) from None


def names():
    return sorted(REGISTRY)
