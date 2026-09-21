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


@dataclass(frozen=True)
class Sport:
    name: str                      # registry key, e.g. "hockey-nhl"
    length: float                  # metres along X
    width: float                   # metres along Y
    surface: str                   # "ice" | "grass": key for core.surface
    line_classes: tuple            # class names; index 0 is background
    polylines: Callable[[], list]  # -> [(class index, (N, 2) world polyline in metres), ...]
    keypoints: Optional[Callable[[], np.ndarray]] = None  # -> (K, 2) world metres, NaN = undefined

    def class_polylines(self, cls):
        """All polylines of one line class."""
        return [pl for c, pl in self.polylines() if c == cls]


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
