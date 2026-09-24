"""Find frames of a camera view the training set lacks, in unlabelled video, and queue them for hand labelling.

Written for the end-zone camera (behind the goal, looking down the rink): model A refuses or fails it, and the train
set holds 4 such NHL frames (hockey.md section 14b). Views are told apart by the labelled frames: each labelled frame
of the given videos is an "end" or a "side" reference by the angle of the rink's long axis in the image
(`core.camera.view_angle`), and every sampled video frame is scored by how much closer its thumbnail is to the nearest end
reference than to the nearest side one. Around every frame scoring above --threshold, frames every --step s within
--window s are queued (`click_labeler` labels a folder's queue.json in order).

    python -m sportcal.lab.hockey.mine_views                   # train videos -> datasets/hockeyrink_nhl_endview

ponytail: a 64x36 grey-thumbnail nearest neighbour, found good enough by eye on the train videos (the top candidates
were the two end-zone shots there are, plus one side-view shot); a learned view classifier is the upgrade once there
are more end-view references.
"""
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np

from sportcal.core.camera import view_angle
from sportcal.lab.hockey.build_h_index import HOLDOUT_VIDEOS
from sportcal.lab.hockey.build_h_index import OUT as INDEX
from sportcal.paths import DATASETS, ROOT


def _thumb(img):
    g = cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def _video(v):
    p = ROOT / (v + ".mp4")
    return p if p.exists() else ROOT / (v + ".mp4.webm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", nargs="*", help="videos to mine (default: every NHL video with train frames in the index);"
                    " the references are always the labelled frames of every video that is not held out")
    ap.add_argument("--out", default="hockeyrink_nhl_endview", help="dataset folder under datasets/ for queue.json")
    ap.add_argument("--min-angle", type=float, default=45.0, help="references above this view_angle are end views")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between scored frames")
    ap.add_argument("--threshold", type=float, default=0.04)
    ap.add_argument("--window", type=float, default=1.5, help="seconds queued either side of a hit")
    ap.add_argument("--step", type=float, default=0.5, help="seconds between queued frames")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(INDEX, encoding="utf-8")]
    train_videos = sorted({r["video"] for r in rows if r["split"] == "train" and r["template"] == "hockey-nhl"})
    videos = [v for v in (args.videos or train_videos) if v not in HOLDOUT_VIDEOS]
    refs = [r for r in rows if r["template"] == "hockey-nhl" and r["video"] in train_videos]
    F = np.array([_thumb(cv2.imread(str(ROOT / r["image"]))) for r in refs])
    end = np.array([view_angle(r["H"], r["w"], r["h"]) > args.min_angle for r in refs])
    print("references: {} end, {} side, from {}".format(end.sum(), (~end).sum(), " ".join(train_videos)))
    if not end.any():
        raise SystemExit("no end-view reference among the labelled frames of these videos")

    per = defaultdict(set)
    for v in videos:
        cap = cv2.VideoCapture(str(_video(v)))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        every, step, half = int(round(args.every * fps)), int(round(args.step * fps)), int(args.window / args.step)
        i, hits = 0, 0
        while cap.grab():
            if i % every == 0:
                ok, img = cap.retrieve()
                d = 1 - F @ _thumb(img) if ok else None
                if ok and d[~end].min() - d[end].min() > args.threshold:
                    hits += 1
                    per[v] |= {j for j in (i + k * step for k in range(-half, half + 1)) if 0 <= j < n}
            i += 1
        print("  {:<6} {} hits, {} frames queued".format(v, hits, len(per[v])), flush=True)

    lists = [sorted(per[v]) for v in sorted(per)]
    queue = [[v, lst[k]] for k in range(max(map(len, lists), default=0))
             for v, lst in zip(sorted(per), lists) if k < len(lst)]      # videos take turns
    out = DATASETS / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.json").write_text(json.dumps(queue))
    print("{} frames -> {}".format(len(queue), out / "queue.json"))


if __name__ == "__main__":
    main()
