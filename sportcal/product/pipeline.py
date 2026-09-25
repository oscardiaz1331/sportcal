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

import cv2
import numpy as np

from sportcal.core.geometry import REF_WIDTH, blend_homographies, image_to_world
from sportcal.core.motion import estimate_motion, to_native_motion, to_work


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
    """First-answer-wins chain, with optional hold of the last H and optional smoothing along the camera motion.

    hold_frames: when every estimator refuses, keep returning the last estimate for up to this many frames.
    smooth: None, or the weight (0-1) of each new answer against the last H carried to this frame by the camera motion
    (`core.motion`, KLT over the whole frame). Per-frame answers jitter by a few px; the motion between two frames is
    measured far more precisely, so the carried H is the prior and each answer only nudges it (a complementary filter):
    no lag when the camera pans, the jitter averaged over ~1/smooth frames. With smoothing a hold follows the camera
    instead of freezing, and an answer more than `max_jump_px` (at REF_WIDTH) away from the carried H counts as a refusal,
    unless it keeps disagreeing for more than `accept_after` frames (a missed shot cut, or a carried H that drifted): then
    it replaces the carried H.
    Call `reset()` at every shot cut, otherwise a stale H outlives its shot.
    """

    def __init__(self, estimators: Sequence[Estimator], hold_frames: int = 0, smooth: Optional[float] = None,
                 max_jump_px: float = 300.0, accept_after: int = 5):
        if not estimators:
            raise ValueError("at least one estimator is required")
        self.estimators = list(estimators)
        self.hold_frames, self.smooth = hold_frames, smooth
        self.max_jump_px, self.accept_after = max_jump_px, accept_after
        self.stats = Counter()  # method name (or "hold" / "none") -> frames
        self.reset()

    def reset(self):
        self._last, self._age, self._gray, self._jumps = None, 0, None, 0

    def _carry(self, frame):
        """The last output moved to this frame: along the camera motion when smoothing, as it was otherwise. None when
        there is none or the camera motion is lost (a shot cut, a close-up)."""
        if self.smooth is None:
            return self._last
        gray = cv2.cvtColor(to_work(frame), cv2.COLOR_BGR2GRAY)
        prev, self._gray = self._gray, gray
        if self._last is None or prev is None:
            return None
        # a tight RANSAC threshold: at the default, static overlays (score bug, channel logo) fit a slow pan within the
        # threshold, pull the motion towards zero, and the carried H lags (docs/experiments/hockey.md section 14i)
        M = estimate_motion(prev, gray, ransac_thresh=1.0)["M"]
        return None if M is None else replace(self._last, H=to_native_motion(M, frame.shape[1]) @ self._last.H)

    def __call__(self, frame):
        e = next((a for a in (est.estimate(frame) for est in self.estimators) if a is not None), None)
        prior = self._carry(frame)
        if e is not None and prior is not None and self.smooth is not None:
            h, w = frame.shape[:2]
            # every frame held without an answer is one missed correction: the carried H has drifted for that long
            H, gap = blend_homographies(prior.H, e.H, 1 - (1 - self.smooth) ** (1 + self._age), w, h)
            if gap * REF_WIDTH / w <= self.max_jump_px:
                e, self._jumps = replace(e, H=H, method=e.method + "+klt"), 0
            elif self._jumps < self.accept_after:
                e, self._jumps = None, self._jumps + 1
            else:
                self._jumps = 0
        if e is not None:
            self._last, self._age = e, 0
            self.stats[e.method] += 1
            return e
        if prior is not None and self._age < self.hold_frames:
            self._last, self._age = prior, self._age + 1
            self.stats["hold"] += 1
            return replace(prior, method=prior.method + "+hold")
        self._last = None
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
