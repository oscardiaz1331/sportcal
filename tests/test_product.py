import numpy as np

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


def test_pipeline_requires_an_estimator():
    import pytest
    with pytest.raises(ValueError):
        HomographyPipeline([])


def test_project_to_field_flags_points_outside_the_rink():
    sport = get("hockey-nhl")
    world_in, world_out = np.array([[30.0, 12.0]]), np.array([[-20.0, 12.0]])
    img = np.vstack([np.c_[world_in, [1]] @ H.T, np.c_[world_out, [1]] @ H.T])[:, :2]
    world, inside = project_to_field(E("a"), img, sport)
    assert np.allclose(world, np.vstack([world_in, world_out]), atol=1e-6)
    assert inside.tolist() == [True, False]
