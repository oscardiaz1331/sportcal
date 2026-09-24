"""Calibrate the fixed camera centre of a broadcast from rich frames spread over a clip.

A broadcast camera pans, tilts and zooms about a fixed centre, so once that centre is known a frame has only three
unknowns (`FieldSolver.search_fixed_center`) and a couple of primitives are enough to solve it. The centre is estimated
here: solve frames spread over the clip with the line-intersection solver, split each accepted homography into a pinhole
pose (`core.camera.decompose_H`) and take the robust median of the centres (`core.camera.robust_centre`). Frames with
few primitives decompose to wild centres, so only well-scored, plausible, pinhole-consistent frames vote. Results:
docs/experiments/soccer.md.

    python -m sportcal.lab.soccer.camera_center --video soccer --n 16 --save
"""
import argparse
import json
import time

import cv2
import numpy as np

from sportcal.core import camera as CAM
from sportcal.lab.soccer import evaluation as EV
from sportcal.lab.soccer import field_solver as FS
from sportcal.lab.soccer import labeler as LB

MIN_VOTES = 8


def frame_centre(frame_bgr, min_score=0.6, max_mismatch=0.05):
    """Camera pose of one frame from the line-intersection solver, or None when it is not trustworthy.

    Accepted: solver score >= min_score, pinhole mismatch <= max_mismatch and a plausible broadcast position (on the
    near side of the pitch, outside the touchline, a few metres up). Returns {"C", "f_over_w", "pan_deg", "tilt_deg", "score"}."""
    mask = EV.etapas_mascara(frame_bgr)["lineas"]
    best = FS.FieldSolver(mask).search_lines(top=3)
    if not best or best[0]["score"] < min_score:
        return None
    h, w = mask.shape
    d = CAM.decompose_H(best[0]["H"], w, h)
    if d is None or d["mismatch"] > max_mismatch:
        return None
    C = d["C"]
    if not (C[1] < -30.0 and 3.0 < C[2] < 60.0):
        return None
    return {"C": C, "f_over_w": d["f"] / w, "pan_deg": float(np.degrees(d["pan"])), "tilt_deg": float(np.degrees(d["tilt"])),
            "score": best[0]["score"]}


def calibrate(video, n=16, min_score=0.6, margin=0.03):
    """Solve `n` frames spread over the clip; returns (per-frame results, robust centre or None)."""
    cap = cv2.VideoCapture(str(LB.ROOT / (video + ".mp4")))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frames = np.linspace(margin * total, (1 - margin) * total, n).astype(int)
    results = []
    for f in frames:
        img = LB.lee_frame(video, int(f))
        r = None if img is None else frame_centre(img, min_score)
        results.append({"frame": int(f), "result": r})
    votes = [r["result"]["C"] for r in results if r["result"] is not None]
    return results, (CAM.robust_centre(votes) if votes else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="soccer")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--save", action="store_true", help="write the centre where the labelling app looks for it")
    a = ap.parse_args()
    t0 = time.time()
    results, est = calibrate(a.video, a.n, a.min_score)
    print("frame  used   score   f/w   pan   tilt   centre (x, y, z)")
    for r in results:
        d = r["result"]
        print("{:>6}  {:>4}  {}".format(r["frame"], "yes" if d else "no", "" if d is None else "{:>6.2f} {:>5.2f} {:>5.1f} {:>5.1f}   ({:.1f}, {:.1f}, {:.1f})".format(
            d["score"], d["f_over_w"], d["pan_deg"], d["tilt_deg"], *d["C"])))
    if est is None:
        print("no frame was solved well enough to vote")
        return
    centre, spread, votes = est
    print("\ncentre = ({:.2f}, {:.2f}, {:.2f}) m, spread ({:.2f}, {:.2f}, {:.2f}) m, {} of {} frames voted ({:.0f} s)".format(
        *centre, *spread, votes, len(results), time.time() - t0))
    if votes < MIN_VOTES:
        print("warning: only {} votes (< {}), the centre is provisional; raise --n or lower --min-score".format(votes, MIN_VOTES))
    if a.save:
        LB.save_center(a.video, centre, {"spread": [float(x) for x in spread], "votes": int(votes), "frames": len(results)})
        print("saved to", LB.center_path(a.video))


if __name__ == "__main__":
    main()
