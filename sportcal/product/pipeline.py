"""Deployable homography pipeline: an ordered chain of estimators.

The product is the *composition* of methods that won in the labs, not a method of its
own. Each estimator either returns an `Estimate` or refuses (None); the pipeline returns
the first one that answers. Order = trust: put precise-but-rarely-available estimators
first and full-coverage fallbacks last. Which order and why: docs/decisions/0003-product-composition.md.

Adding a method = implementing `Estimator` and inserting it in the list; nothing else changes.
"""
from collections import Counter
from dataclasses import dataclass, replace
from typing import Optional, Protocol, Sequence

import numpy as np

from sportcal.core.geometry import image_to_world


@dataclass(frozen=True)
class Estimate:
    H: np.ndarray       # world metres -> image pixels
    confidence: float   # 0..1; only comparable between estimates of the SAME method
    method: str


class Estimator(Protocol):
    name: str

    def estimate(self, frame: np.ndarray) -> Optional[Estimate]:
        """H for this frame, or None when the estimator does not trust its own answer."""


class HomographyPipeline:
    """First-answer-wins chain with optional hold of the last H.

    hold_frames: when every estimator refuses, keep returning the last estimate for up to
    this many frames (a broadcast camera is nearly static within a shot). Call `reset()`
    at every shot cut, otherwise a stale H outlives its shot.
    """

    def __init__(self, estimators: Sequence[Estimator], hold_frames: int = 0):
        if not estimators:
            raise ValueError("at least one estimator is required")
        self.estimators = list(estimators)
        self.hold_frames = hold_frames
        self.stats = Counter()  # method name (or "hold" / "none") -> frames
        self._last = None
        self._age = 0

    def reset(self):
        self._last, self._age = None, 0

    def __call__(self, frame):
        for est in self.estimators:
            e = est.estimate(frame)
            if e is not None:
                self._last, self._age = e, 0
                self.stats[e.method] += 1
                return e
        if self._last is not None and self._age < self.hold_frames:
            self._age += 1
            self.stats["hold"] += 1
            return replace(self._last, method=self._last.method + "+hold")
        self.stats["none"] += 1
        return None


def project_to_field(estimate, img_xy, sport, margin=1.0):
    """Image pixels (e.g. players' feet) -> field metres, plus a mask of points that land
    inside the field (with `margin` metres of slack). Only feet/ground contact points are
    meaningful: H is a plane-to-plane map."""
    world = image_to_world(estimate.H, img_xy)
    inside = ((world[:, 0] >= -margin) & (world[:, 0] <= sport.length + margin)
              & (world[:, 1] >= -margin) & (world[:, 1] <= sport.width + margin))
    return world, inside
