"""The 56-keypoint YOLO rink model scored like the keypoint + line model (`lab/common/train_kpline --eval`), through the
product estimator, one frame at a time. Its H is canonicalized first: YOLO was trained on labels with mixed mirror
conventions (hockey.md section 0b), so the names of its points are not a fair part of the comparison.

    python -m sportcal.lab.hockey.eval_yolo runs/hockeyrink/yolo26m-18/weights/best_homography.pt --split test [--gate]
"""
import argparse
import json

import cv2
import numpy as np

from sportcal import sports
from sportcal.lab.common.train_kpline import error, index_of, report
from sportcal.paths import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("weights")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    from sportcal.product.hockey import YoloKeypointEstimator   # lab -> product: only here, for the comparison
    est = YoloKeypointEstimator(args.weights, device="cpu" if args.cpu else 0)
    sport = sports.get("hockey-nhl")
    rows = [r for r in map(json.loads, open(index_of(sport.name), encoding="utf-8"))
            if r["split"] == args.split and r["template"] == sport.name]
    errs = []
    for r in rows:
        e = est.estimate(cv2.imread(str(ROOT / r["image"])))
        errs.append(error(None if e is None else sport.canonicalize(e.H), r, args.gate))
    report(rows, np.array(errs), args.split)


if __name__ == "__main__":
    main()
