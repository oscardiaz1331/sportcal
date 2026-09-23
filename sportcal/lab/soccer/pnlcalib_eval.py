"""Score PnLCalib (public calibration trained on SoccerNet) against our hand labels: the reference number any
calibration we build ourselves has to beat. Write-up: docs/experiments/soccer.md section 18.

PnLCalib runs in its own environment (`pnlcalib_run.py`, see its docstring); this module only reads that JSON,
converts each camera to our world convention and measures it against `datasets/soccer_labels/clicks.jsonl`.

    python -m sportcal.lab.soccer.pnlcalib_eval --preds runs/pnlcalib/soccer_labels.json
"""
import argparse
import json

import numpy as np

from sportcal.core.geometry import geom_error
from sportcal.paths import DATASETS
from sportcal.sports.soccer.field import HALF_LENGTH, HALF_WIDTH, canonicalize_H

# SoccerNet world: origin at the centre spot, x along the length, y TOWARDS the main camera, z down.
# Ours: same origin and x, +Y towards the FAR touchline. On the ground plane the two differ by y -> -y: a
# reflection, which canonicalize_H (a 180-degree turn) cannot undo.
SN_TO_OURS = np.diag([1.0, -1.0, 1.0])

# Same measure as the hand-label checks of soccer.md section 12: median px (at 1920) over a pitch grid, on the
# points the LABEL puts inside the frame.
GRID = np.stack(np.meshgrid(np.arange(-HALF_LENGTH, HALF_LENGTH + 1e-9, 1.0),
                            np.arange(-HALF_WIDTH, HALF_WIDTH + 1e-9, 1.0)), -1).reshape(-1, 2)


def to_H(P):
    """Our world (X, Y) -> image homography of the ground plane, from a SoccerNet 3x4 projection matrix."""
    H = np.asarray(P, float)[:, [0, 1, 3]] @ SN_TO_OURS
    return canonicalize_H(H / H[2, 2])


def evaluate(preds, labels, w=1920, h=1080):
    """[(frame id, error px at 1920 or None if PnLCalib refused, PnLCalib's own rep_err)] in label order."""
    rows = []
    for fid, H_gt in labels.items():
        p = preds.get(fid)
        if p is None or p["P"] is None:
            rows.append((fid, None, None))
            continue
        rows.append((fid, geom_error(to_H(p["P"]), H_gt, GRID, w, h), p["rep_err"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description="PnLCalib vs the hand-labelled soccer frames")
    ap.add_argument("--preds", required=True, help="JSON written by pnlcalib_run.py")
    ap.add_argument("--labels", default=str(DATASETS / "soccer_labels" / "clicks.jsonl"))
    args = ap.parse_args()

    labels = {}
    for line in open(args.labels, encoding="utf-8"):
        d = json.loads(line)
        labels[d["id"]] = np.asarray(d["H"], float)
    rows = evaluate(json.load(open(args.preds, encoding="utf-8")), labels)

    print("{:<16} {:>10} {:>9}".format("frame", "err px", "rep_err"))
    for fid, e, r in rows:
        print("{:<16} {:>10} {:>9}".format(fid, "refused" if e is None else "{:.1f}".format(e),
                                           "" if r is None else "{:.2f}".format(r)))
    for clip in sorted({fid.rsplit("_", 1)[0] for fid in labels}):
        errs = np.array([e for fid, e, _ in rows if fid.rsplit("_", 1)[0] == clip and e is not None])
        n = sum(fid.rsplit("_", 1)[0] == clip for fid, _, _ in rows)
        if len(errs):
            print("{:<8} answered {}/{}  p50 {:.1f}  p90 {:.1f}  <10 px {:.0f}%  <25 px {:.0f}% (of all frames)".format(
                clip, len(errs), n, np.median(errs), np.percentile(errs, 90),
                100 * (errs < 10).sum() / n, 100 * (errs < 25).sum() / n))


if __name__ == "__main__":
    main()
