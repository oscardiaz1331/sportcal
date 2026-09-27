"""Swap signals of tracking step 0 (sportcal/lab/hockey/track_swaps.py, hockey.md section 15), on synthetic tracks."""
import numpy as np
import pytest

from sportcal.lab.hockey.track_swaps import A_MAX, V_MAX, h_jump_frames, jumps, merge, team_changes, turns

MARGIN = 0.3   # m of feet-point noise the jump test tolerates in these tests


def _track(v=(6.0, 0.0), start=(10.0, 5.0), fps=60.0, seconds=4.0, noise=0.03, seed=0):
    """(t, xy) of a skater at constant velocity whose feet point jitters."""
    t = np.arange(int(seconds * fps)) / fps
    rng = np.random.default_rng(seed)
    return t, np.array(start) + t[:, None] * np.array(v) + rng.normal(0, noise, (len(t), 2))


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_a_skater_at_record_speed_or_on_a_tight_curve_fires_nothing(fps):
    t, xy = _track(v=(11.0, 0.0), fps=fps)
    assert len(jumps(t, xy, V_MAX, MARGIN)) == 0 and len(turns(t, xy, A_MAX, fps)) == 0
    theta = 1.5 * t     # 6 m/s on a 4 m radius: 9 m/s^2 of turning, hard but real
    curve = np.c_[20 + 4 * np.cos(theta), 10 + 4 * np.sin(theta)] + np.random.default_rng(1).normal(0, 0.03, (len(t), 2))
    assert len(jumps(t, curve, V_MAX, MARGIN)) == 0 and len(turns(t, curve, A_MAX, fps)) == 0


def test_an_id_moving_to_a_player_one_metre_away_is_one_jump_at_the_swap():
    t, a = _track()
    _, b = _track(start=(10.0, 6.0), seed=1)
    xy = np.r_[a[:120], b[120:]]
    assert list(jumps(t, xy, V_MAX, MARGIN)) == [120]


def test_an_id_moving_between_two_crossing_players_is_an_impossible_turn_not_a_jump():
    t, a = _track(v=(6.0, 0.0), start=(10.0, 5.0))
    _, b = _track(v=(0.0, 6.0), start=(22.0, -7.0), seed=1)     # both pass (22, 5) at t = 2 s, frame 120
    xy = np.r_[a[:120], b[120:]]
    assert len(jumps(t, xy, V_MAX, MARGIN)) == 0
    flags = turns(t, xy, A_MAX, 60.0)
    assert len(flags) > 0 and np.abs(flags - 120).max() <= 8
    assert len(merge(np.zeros(len(t), int), t, flags)) == 1


def test_a_team_change_needs_a_second_of_each_team():
    t = np.arange(240) / 60.0
    team = np.r_[np.zeros(120, int), np.ones(120, int)]
    changes = team_changes(t, team)
    assert len(changes) == 1 and abs(changes[0] - 120) <= 1
    flicker = np.zeros(240, int)
    flicker[100:142] = 1                       # 0.7 s of misread shirts: the majority flips, but for less than 1 s
    assert len(team_changes(t, flicker)) == 0
    unsure = team.copy()
    unsure[::2] = -1                           # unconfident votes do not count
    assert len(team_changes(t, unsure)) == 1


def test_everyone_jumping_on_one_frame_is_the_homography():
    seen = np.repeat(np.arange(100), 4)        # 4 tracks on each of 100 frames
    assert list(h_jump_frames(np.array([50, 50, 50, 70]), seen)) == [50]


def test_flags_within_a_second_on_one_track_are_one_event():
    t = np.arange(240) / 60.0
    assert list(merge(np.zeros(240, int), t, np.array([40, 10, 12, 200]))) == [10, 200]
    two = np.r_[np.zeros(100, int), np.ones(140, int)]
    assert list(merge(two, t, np.array([95, 100]))) == [95, 100]


def test_observations_keep_measurable_tracked_feet_only_and_split_ids_by_shot():
    from sportcal import sports
    from sportcal.lab.hockey.track_swaps import observations

    H = np.array([[20.0, 0, 100], [0, 20.0, 50], [0, 0, 1]])      # world metres -> image px

    def box(x, y):                                                   # a box whose feet are at world (x, y)
        u, v = 100 + 20 * x, 50 + 20 * y
        return [u - 10, v - 60, u + 10, v]

    c = {"fps": np.float64(60), "shot": np.array([0, 0, 1]), "H": np.stack([H, np.full((3, 3), np.nan), H]),
         "det_frame": np.array([0, 0, 0, 1, 2]),
         "xyxy": np.array([box(10, 5), box(-5, 5), box(20, 8), box(10, 5), box(10, 5)], np.float32),
         "hist": np.full((5, 128), np.nan, np.float16),
         "id_bytetrack": np.array([3, 4, -1, 3, 3]), "id_botsort": np.zeros(5, int)}
    o = observations(c, "bytetrack", sports.get("hockey-nhl"))
    # kept: box 0 (frame 0) and box 4 (frame 2, next shot); dropped: off the rink (1), untracked (2), frame without H (3)
    assert list(o["frame"]) == [0, 2]
    np.testing.assert_allclose(o["xy"], [[10, 5], [10, 5]], atol=1e-4)
    assert o["key"][0] != o["key"][1]          # the same tracker ID in two shots is two tracks
    assert list(o["team"]) == [-1, -1]         # no jersey colour, no team


def test_labels_file_is_created_once_and_never_overwritten(tmp_path):
    from sportcal.lab.hockey.track_swaps import read_labels

    p = tmp_path / "labels.csv"
    assert read_labels(p) == {} and p.read_text() == "event_id,label\n"
    p.write_text("event_id,label\nnhl10:bytetrack:3:120,swap\n")
    assert read_labels(p) == {"nhl10:bytetrack:3:120": "swap"}
    assert "swap" in p.read_text()


def test_fresh_games_need_the_margin_frozen_on_development():
    from sportcal.lab.hockey.track_swaps import signals

    with pytest.raises(SystemExit, match="margin"):
        signals(["nhl10", "nhl12"])
