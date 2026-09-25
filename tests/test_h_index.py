import json

import cv2
import numpy as np
import pytest

from sportcal.lab.hockey.build_h_index import EXTRA, FRESH, HAND, HOLDOUT_VIDEOS, canonicalize, split_of, video_of
from sportcal.paths import ROOT
from sportcal.sports.hockey import rink


def test_held_out_videos_never_reach_train_or_dev():
    """The clean test set only means something if no frame of its videos is trained or selected on."""
    assert video_of("nhl10_001440") == ("nhl10", 1440) and video_of("nhl1_00007") == ("nhl1", 7)
    assert video_of("clip2_00510") == ("clip2", 510) and video_of("3f2a9c1e-uuid") == ("shl", None)
    for v in HOLDOUT_VIDEOS:
        for ds in ("hockeyrink_nhl", "hockeyrink"):
            for s in ("train", "val"):
                for hand in (False, True):
                    assert split_of(ds, s, v, hand) not in ("train", "dev"), (ds, s, v, hand)
        assert split_of(HAND, "val", v) == "test" and split_of("hockeyrink_nhl", "train", v, hand=True) == "test"
    assert split_of(HAND, "val", "nhl5") == "test_leaky"
    assert (split_of("hockeyrink_nhl", "train", "nhl5"), split_of("hockeyrink_nhl", "val", "nhl5")) == ("train", "dev")
    assert split_of("hockeyrink_nhl", "train", "nhl5", hand=True) == "train"
    assert [split_of(FRESH, "val", "nhl11", hand=True), split_of(FRESH, "val", "nhl11", propagated=True)] == ["fresh", "excluded"]
    for ds in (HAND, *EXTRA):                  # a label carried from another one by the labeller is never a test frame
        assert [split_of(ds, "val", v, propagated=True) for v in ("nhl4", "nhl5")] == ["excluded", "train"]
    assert split_of(EXTRA[0], "val", "nhl5") == "train" and split_of(EXTRA[0], "val", "nhl4") == "test"


def test_every_mirrored_label_comes_back_to_one_convention():
    """Zone A left and the y = 0 boards on top, whichever of the rink's mirrors the label was stored in."""
    p = rink.RINK_NHL
    L, W = p["length"], p["width"]
    world = np.float32([[0, 0], [L, 0], [L, W], [0, W]])
    H = cv2.getPerspectiveTransform(world, np.float32([[400, 300], [1500, 300], [1800, 900], [100, 900]])).astype(float)
    H /= H[2, 2]
    fx = np.array([[-1.0, 0, L], [0, 1, 0], [0, 0, 1]])
    fy = np.array([[1.0, 0, 0], [0, -1, W], [0, 0, 1]])
    for S, flips in ((np.eye(3), ""), (fx, "x"), (fy, "y"), (fx @ fy, "xy")):
        Hc, got = canonicalize(H @ S, p)
        assert got == flips and np.allclose(Hc, H, atol=1e-9), flips


def test_a_saved_proposal_is_not_read_as_a_hand_label(tmp_path, monkeypatch):
    """The labeller records where a saved H came from; the index must keep such frames out of the test sets."""
    import sportcal.lab.hockey.build_h_index as B
    monkeypatch.setattr(B, "DATASETS", tmp_path)
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "clicks.jsonl").write_text(json.dumps({"id": "nhl4_000030", "clics": {}, "H": np.eye(3).tolist()}) + "\n" +
                                                 json.dumps({"id": "nhl4_000060", "clics": {}, "H": np.eye(3).tolist(),
                                                             "origen": "propagada desde nhl4_000030"}) + "\n")
    assert {k: src for k, (_, src) in B._hand_labels("ds").items()} == {("nhl4", 30): "hand", ("nhl4", 60): "propagated"}


ENDVIEW = ROOT / "datasets" / "hockeyrink_nhl_endview"


@pytest.mark.slow
@pytest.mark.skipif(not (ENDVIEW / "clicks.jsonl").exists(), reason="needs the end-view hand labels and their videos")
def test_a_hand_label_carried_to_a_labelled_neighbour_lands_on_its_clicks(monkeypatch):
    """`click_labeler.propaga` end to end on real frames (the seek, the working-width scaling, the chaining direction):
    the proposal for a labelled frame, carried from ANOTHER label of its shot, must land near its own clicks. Scored at
    the clicks, whichever of the rink's mirrors the two labels were named in."""
    from sportcal.lab.hockey import click_labeler as CL
    monkeypatch.setattr(CL, "OUT", ENDVIEW)
    p = rink.RINK_NHL
    L, W = p["length"], p["width"]
    tpl = rink.build_template(p)
    mirrors = [np.diag([sx, sy, 1.0]) + np.array([[0, 0, L * (sx < 0)], [0, 0, W * (sy < 0)], [0, 0, 0]])
               for sx in (1, -1) for sy in (1, -1)]
    labels = {d["id"]: d for d in map(json.loads, open(ENDVIEW / "clicks.jsonl", encoding="utf-8"))}
    frame_of = {c: (c.rsplit("_", 1)[0], int(c.rsplit("_", 1)[1])) for c in labels}
    near = [c for c in sorted(labels) if any(o != c and frame_of[o][0] == frame_of[c][0] and
                                             abs(frame_of[o][1] - frame_of[c][1]) <= 30 for o in labels)]
    if not near:
        pytest.skip("no labelled frame has another label of its video within 30 frames")
    cid, (v, f) = near[0], frame_of[near[0]]
    d, frame = labels[cid], CL.lee_frame(v, f)
    cands = CL.propaga(v, f, frame)
    assert cands, cid                                # a wrong seek or a lost chain gives no proposal at all
    world = np.c_[tpl[[int(k) for k in d["clics"]]], np.ones(len(d["clics"]))]
    clicks = np.array(list(d["clics"].values()))
    err = min(np.median(np.linalg.norm(q[:, :2] / q[:, 2:] - clicks, axis=1))
              for q in (world @ S.T @ cands[0]["H"].T for S in mirrors)) * 1920.0 / frame.shape[1]
    assert err < 25, (cid, cands[0]["via"], err)      # the start label as is misses by several times this (section 14e)
