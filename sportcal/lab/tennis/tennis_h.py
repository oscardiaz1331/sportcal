"""Per-frame H index for tennis from the TennisCourtDetector annotations (github.com/yastrebksv/TennisCourtDetector,
8841 broadcast frames at 1280x720, 14 court keypoints each), the tennis counterpart of `lab/soccer/soccernet_h`.

The annotations come from a classical court detector with bad results removed by hand, so the 14 points are already
consistent with one homography: the fit residual cannot reveal a wrong label, only a broken one. The published
train/val split shares videos; frames are re-split by video here (`split_of`). Results: docs/experiments/tennis.md.

    hf download Gholamreza/tennis_court_keypoints_dataset --repo-type dataset --local-dir datasets/tennis_court
    (extract tennis_court_det_dataset.zip there)
    python -m sportcal.lab.tennis.tennis_h          # -> datasets/tennis_h.jsonl
"""
import argparse
import json
import zlib

import cv2
import numpy as np

from sportcal.core.camera import canonical_mirror, is_plausible_view
from sportcal.paths import DATASETS
from sportcal.sports.tennis import court as C

ROOT = DATASETS / "tennis_court" / "data"
OUT = DATASETS / "tennis_h.jsonl"
W, H_IMG = 1280, 720
BOX = (-C.HALF_LENGTH, C.HALF_LENGTH, -C.HALF_WIDTH, C.HALF_WIDTH)


def fit(kps):
    """(H world -> px, median and max px distance of the 14 points to where H puts them, at 1920 px) - least squares."""
    p = np.asarray(kps, float)
    H, _ = cv2.findHomography(C.KEYPOINT_COORDS, p, 0)
    q = cv2.perspectiveTransform(C.KEYPOINT_COORDS[:, None], H)[:, 0]
    e = np.linalg.norm(q - p, axis=1) * 1920 / W
    return H / H[2, 2], float(np.median(e)), float(e.max())


def split_of(video, dev=0.1, test=0.1):
    """train / dev / test by a stable hash of the video id, so no video is in two splits."""
    u = zlib.crc32(video.encode()) / 2 ** 32
    return "test" if u < test else "dev" if u < test + dev else "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-px", type=float, default=3.0, help="drop frames whose points miss their H by more (at 1920)")
    args = ap.parse_args()
    rows, dropped = [], {"residual": 0, "implausible": 0, "no image": 0}
    for name in ("data_train.json", "data_val.json"):
        for a in json.load(open(ROOT / name)):
            H, med, mx = fit(a["kps"])
            if mx > args.max_px:
                dropped["residual"] += 1
                continue
            if not is_plausible_view(H, W, H_IMG, BOX):
                dropped["implausible"] += 1
                continue
            if not (ROOT / "images" / (a["id"] + ".png")).exists():
                dropped["no image"] += 1
                continue
            video, frame = a["id"].rsplit("_", 1)
            H = canonical_mirror(H, (0.0, 0.0))[0]
            rows.append({"id": "tennis/" + a["id"], "image": "datasets/tennis_court/data/images/{}.png".format(a["id"]),
                         "dataset": "tennis_court", "video": video, "frame": int(frame), "template": "tennis-itf",
                         "source": "tennisdet", "w": W, "h": H_IMG, "H": H.tolist(), "flips": "",
                         "fit": {"median_px": round(med, 2), "max_px": round(mx, 2)}, "split": split_of(video)})
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
    print("{} frames -> {}  {}  dropped {}".format(len(rows), OUT, counts, dropped))


if __name__ == "__main__":
    main()
