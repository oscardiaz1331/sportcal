import cv2
import numpy as np

from sportcal.lab.hockey.build_h_index import HAND, HOLDOUT_VIDEOS, canonicalize, split_of, video_of
from sportcal.sports.hockey import rink


def test_held_out_videos_never_reach_train_or_dev():
    """The clean test set only means something if no frame of its videos is trained or selected on."""
    assert video_of("nhl10_001440") == ("nhl10", 1440) and video_of("nhl1_00007") == ("nhl1", 7)
    assert video_of("clip2_00510") == ("clip2", 510) and video_of("3f2a9c1e-uuid") == ("shl", None)
    for v in HOLDOUT_VIDEOS:
        for ds in ("hockeyrink_nhl", "hockeyrink"):
            for s in ("train", "val"):
                assert split_of(ds, s, v) not in ("train", "dev"), (ds, s, v)
        assert split_of(HAND, "val", v) == "test"
    assert split_of(HAND, "val", "nhl5") == "test_leaky"
    assert (split_of("hockeyrink_nhl", "train", "nhl5"), split_of("hockeyrink_nhl", "val", "nhl5")) == ("train", "dev")


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
