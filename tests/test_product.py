import json

import numpy as np
import pytest

from sportcal.core.geometry import blend_homographies
from sportcal.product.pipeline import Estimate, HomographyPipeline, project_to_field
from sportcal.sports import get

FRAME = np.zeros((10, 10, 3), np.uint8)
H = np.array([[30.0, 0, 200], [0, 30.0, 150], [0, 0, 1.0]])


class Fake:
    """Scripted estimator: returns the queued answers in order (None = refuse)."""

    def __init__(self, name, answers):
        self.name, self.answers = name, list(answers)

    def estimate(self, frame):
        return self.answers.pop(0) if self.answers else None


def E(method, conf=1.0):
    return Estimate(H, conf, method)


def test_first_answer_wins_and_later_stages_are_not_consulted():
    precise, fallback = Fake("precise", [E("precise")]), Fake("fallback", [E("fallback")])
    pipe = HomographyPipeline([precise, fallback])
    assert pipe(FRAME).method == "precise"
    assert len(fallback.answers) == 1  # never asked
    assert pipe.stats["precise"] == 1


def test_falls_through_to_the_fallback_when_the_first_stage_refuses():
    pipe = HomographyPipeline([Fake("precise", [None]), Fake("fallback", [E("fallback")])])
    assert pipe(FRAME).method == "fallback"


def test_returns_none_when_everything_refuses_and_counts_it():
    pipe = HomographyPipeline([Fake("a", [None]), Fake("b", [None])])
    assert pipe(FRAME) is None and pipe.stats["none"] == 1


def test_hold_reuses_the_last_estimate_for_a_bounded_number_of_frames():
    pipe = HomographyPipeline([Fake("a", [E("a"), None, None, None])], hold_frames=2)
    assert pipe(FRAME).method == "a"
    assert pipe(FRAME).method == "a+hold"
    assert pipe(FRAME).method == "a+hold"
    assert pipe(FRAME) is None            # hold budget exhausted
    assert pipe.stats["hold"] == 2


def test_reset_drops_the_held_estimate_at_a_shot_cut():
    pipe = HomographyPipeline([Fake("a", [E("a"), None])], hold_frames=5)
    pipe(FRAME)
    pipe.reset()
    assert pipe(FRAME) is None


def test_default_is_no_hold():
    pipe = HomographyPipeline([Fake("a", [E("a"), None])])
    pipe(FRAME)
    assert pipe(FRAME) is None


def test_smoothing_follows_the_camera_averages_the_noise_and_rides_out_jumps():
    """A camera panning 3 px/frame over a textured scene; the estimator answers the true H plus +-6 px of noise, once
    300 px off, from frame 32 on 200 px off for good (a missed cut), then refuses for 5 frames and for longer than the
    hold. Smoothed along the KLT motion, the output stays within a few px, ignores the one-off jump, takes the persistent
    one after `accept_after` frames, trusts an answer more after a hold, and starts afresh once the hold has run out."""
    import cv2
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur((rng.random((600, 1200)) * 255).astype(np.uint8), (0, 0), 1.5)
    scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

    def shift(dx, dy=0.0):
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    frames = [cv2.warpPerspective(scene, shift(-3.0 * t), (960, 540)) for t in range(66)]
    truth = [shift(-3.0 * t) @ H for t in range(66)]
    noise = rng.uniform(-6, 6, (66, 2))
    noise[25] = (300, 0)
    noise[32:, 0] += 200
    noise[40:48] = (200, 0)
    noise[47] = (220, 0)                  # the first answer after the 5-frame refusal, 20 px off
    answers = [Estimate(shift(*n) @ Ht, 1.0, "noisy") for n, Ht in zip(noise, truth)]
    answers[42:47] = [None] * 5
    answers[50:64] = [None] * 14
    pipe = HomographyPipeline([Fake("noisy", answers)], hold_frames=10, smooth=0.1)
    out = [pipe(f) for f in frames]

    def err(Ha, Hb):
        return blend_homographies(Ha, Hb, 0.0, 960, 540)[1]
    raw = [err(a.H, t) for a, t in zip(answers[10:32], truth[10:32])]
    smoothed = [err(o.H, t) for o, t in zip(out[10:32], truth[10:32])]
    assert np.median(smoothed) < np.median(raw) / 2 and max(smoothed) < 10   # frame 25's jump never shows
    assert err(out[39].H, shift(200) @ truth[39]) < 10                         # the persistent offset is taken
    assert err(out[47].H, shift(200) @ truth[47]) > 5    # after 5 held frames the answer pulls ~half way, not 10%
    assert out[64].method == "noisy"                                           # no stale prior after the hold ran out


def test_pipeline_requires_an_estimator():
    with pytest.raises(ValueError):
        HomographyPipeline([])


def test_project_to_field_flags_points_outside_the_rink():
    sport = get("hockey-nhl")
    world_in, world_out = np.array([[30.0, 12.0]]), np.array([[-20.0, 12.0]])
    img = np.vstack([np.c_[world_in, [1]] @ H.T, np.c_[world_out, [1]] @ H.T])[:, :2]
    world, inside = project_to_field(E("a"), img, sport)
    assert np.allclose(world, np.vstack([world_in, world_out]), atol=1e-6)
    assert inside.tolist() == [True, False]


def test_project_to_field_uses_the_box_of_a_centred_field():
    """Soccer's origin is the centre spot: a player in the left half has x < 0 and is still on the pitch."""
    sport = get("soccer-fifa")
    world_in, world_out = np.array([[-40.0, 10.0]]), np.array([[-60.0, 10.0]])
    img = np.vstack([np.c_[world_in, [1]] @ H.T, np.c_[world_out, [1]] @ H.T])[:, :2]
    world, inside = project_to_field(E("a"), img, sport)
    assert np.allclose(world, np.vstack([world_in, world_out]), atol=1e-6)
    assert inside.tolist() == [True, False]


def test_project_to_field_with_no_points_returns_empty_arrays():
    """A close-up with no player in it: nothing to project, and no crash."""
    world, inside = project_to_field(E("a"), np.zeros((0, 2)), get("hockey-nhl"))
    assert world.shape == (0, 2) and inside.shape == (0,)


@pytest.mark.slow
def test_kpline_estimator_answers_a_new_arena_and_refuses_an_impossible_camera():
    """Real weights on two frames of arenas nothing was tuned on (split `fresh`): one answered close to the hand label,
    and the one whose DLT puts part of the rink behind the camera refused instead of answered hundreds of px off."""
    pytest.importorskip("torch")
    import cv2

    from sportcal.lab.common.train_kpline import error, index_of
    from sportcal.paths import ROOT, RUNS
    from sportcal.product.kpline import KplineEstimator

    weights, index = RUNS / "kpline" / "finetune" / "best_h.pt", index_of("hockey-nhl")
    if not (weights.exists() and index.exists()):
        pytest.skip("needs the kpline weights and the H index")
    rows = {r["id"]: r for r in map(json.loads, open(index, encoding="utf-8"))}
    good, bad = rows["hockeyrink_nhl_fresh/nhl14_004181"], rows["hockeyrink_nhl_fresh/nhl13_010575"]
    est = KplineEstimator(weights, device="cpu")
    e = est.estimate(cv2.imread(str(ROOT / good["image"])))
    assert error(e.H, good) < 10
    assert est.estimate(cv2.imread(str(ROOT / bad["image"]))) is None
