"""Calibrate the radial lens distortion of the broadcast camera, from many frames of the SAME clip.

Only frames that plausibly come from the fixed, central broadcast camera (the one `camera_center.py` calibrated:
same position, only pan/tilt/zoom changing) can share one distortion curve -- a different camera (behind the
goal, a handheld sideline shot, a replay from another angle) has a different lens and would corrupt the fit, so
`is_broadcast_frame` gates every candidate on it before its lines are used (docs/experiments/soccer.md section 15
asked for exactly this before trusting a k1).

A joint pose+distortion fit on a single frame does not separate a small pose error from real distortion (measured:
section 15). This module never fits pose and k1 together: it measures pure line STRAIGHTNESS (the same "bow"
diagnostic that did find a real, clean signal in section 15) on the longest lines of every accepted frame, pooled
across the whole clip, and fits one shared k1 to all of them -- decoupled from any pose. Only after k1 is known
should a frame's pose be solved again, with k1 held fixed (not built here yet: this module stops at reporting the
calibrated k1 and how well it explains the pooled lines. Results: docs/experiments/soccer.md).

    python -m sportcal.lab.soccer.lens_distortion --video soccer --n 60
"""
import argparse
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from sportcal.core.camera import decompose_H
from sportcal.core.fitting import MIN_LINE_PX, detect_lines
from sportcal.lab.soccer import evaluation as EV
from sportcal.lab.soccer import field_solver as FS
from sportcal.lab.soccer import labeler as LB

MIN_LINE_FRAC = 0.2    # a line shorter than this fraction of the mask width carries too little leverage (section 15)
N_BINS = 24


def is_broadcast_frame(frame_bgr, center, min_score=0.6, max_mismatch=0.15, max_center_dev=8.0):
    """The decomposed pose of this frame if it plausibly comes from the known, fixed broadcast camera, else None.

    Same score/mismatch/position gates as `camera_center.frame_centre`, plus the one that matters here: the
    decomposed camera centre must land within `max_center_dev` metres of the ALREADY calibrated `center` (a
    different physical camera -- another angle, a replay -- would rarely decompose anywhere near it). Returns
    {"pose": decompose_H dict, "mask": working-resolution line mask} or None."""
    mask = EV.etapas_mascara(frame_bgr)["lineas"]
    h, w = mask.shape
    best = FS.FieldSolver(mask).search_lines(top=1)
    if not best or best[0]["score"] < min_score:
        return None
    d = decompose_H(best[0]["H"], w, h)
    if d is None or d["mismatch"] > max_mismatch:
        return None
    if not (d["C"][1] < -30.0 and 3.0 < d["C"][2] < 60.0):
        return None
    if np.linalg.norm(np.asarray(d["C"]) - np.asarray(center)) > max_center_dev:
        return None
    return {"pose": d, "mask": mask}


def longest_line_points(mask, min_len_frac=MIN_LINE_FRAC, n_bins=N_BINS):
    """Centre-binned points (n_bins, 2) of the single longest straight line detected in `mask`, or None.

    Binning along the line's long axis (as `core.fitting.skeleton_points` does) keeps a stripe that is wider on
    the near side from tilting the sample; only the longest candidate is kept because a short, central line
    carries almost no distortion signal (section 15)."""
    lines = detect_lines(mask, n_max=10)
    if len(lines) == 0:
        return None
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    pts_all = np.stack([xs, ys], 1).astype(float)
    best = None
    for a, b, c in lines:
        d = np.abs(a * pts_all[:, 0] + b * pts_all[:, 1] + c)
        inl = pts_all[d < 4.0]
        if len(inl) < MIN_LINE_PX:
            continue
        horizontal = (inl[:, 0].max() - inl[:, 0].min()) >= (inl[:, 1].max() - inl[:, 1].min())
        along, across = (inl[:, 0], inl[:, 1]) if horizontal else (inl[:, 1], inl[:, 0])
        edges = np.linspace(along.min(), along.max(), n_bins + 1)
        idx = np.clip(np.digitize(along, edges) - 1, 0, n_bins - 1)
        centres = []
        for b_ in range(n_bins):
            sel = idx == b_
            if sel.sum() >= 3:
                centres.append((np.median(along[sel]), np.median(across[sel])))
        if len(centres) < n_bins // 2:
            continue
        centres = np.array(centres)
        pts = centres[:, [0, 1]] if horizontal else centres[:, [1, 0]]
        length = np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
        if length >= min_len_frac * w and (best is None or length > best[0]):
            best = (length, pts)
    return None if best is None else best[1]


def collect_lines(video, center, n=60, margin=0.03, **kw):
    """Walk `n` frames spread over the clip; for each accepted broadcast-camera frame (`is_broadcast_frame`),
    keep its longest line and the K matching that frame's own focal length (mask-resolution pixels: k1 is
    defined in normalized coordinates, so it does not matter which resolution K and the points share, as long
    as they agree with each other). Returns [{"frame", "pts", "K", "f_over_w"}], accepted frames only."""
    cap = cv2.VideoCapture(str(LB.ROOT / (video + ".mp4")))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    out = []
    for f in np.linspace(margin * total, (1 - margin) * total, n).astype(int):
        img = LB.lee_frame(video, int(f))
        if img is None:
            continue
        acc = is_broadcast_frame(img, center, **kw)
        if acc is None:
            continue
        pts = longest_line_points(acc["mask"])
        if pts is None:
            continue
        h, w = acc["mask"].shape
        fpx = acc["pose"]["f"]
        K = np.array([[fpx, 0, w / 2.0], [0, fpx, h / 2.0], [0, 0, 1.0]])
        out.append({"frame": int(f), "pts": pts, "K": K, "f_over_w": fpx / w})
    return out


def _residuals(k1, lines):
    """Perpendicular deviation (px) of every binned point from its line's own end-to-end chord, after
    undistorting with this k1 -- concatenated over every collected line. A straight world line gives ~0 at the
    right k1; the shared k1 minimising the sum of squares is the one all the pooled lines agree on."""
    out = []
    for ln in lines:
        und = cv2.undistortPoints(ln["pts"].reshape(-1, 1, 2).astype(np.float32), ln["K"],
                                  np.array([k1[0], 0, 0, 0]), P=ln["K"]).reshape(-1, 2)
        d = und[-1] - und[0]
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            out.append(np.zeros(len(und)))
            continue
        n = np.array([-d[1], d[0]]) / norm
        out.append((und - und[0]) @ n)
    return np.concatenate(out)


def fit_shared_k1(lines, k1_0=0.0):
    """Least-squares k1 shared by every collected line. Returns {"k1", "rms_before", "rms_after", "n_lines"}."""
    if not lines:
        return None
    r0 = _residuals(np.array([k1_0]), lines)
    r = least_squares(_residuals, np.array([k1_0]), args=(lines,), bounds=([-0.6], [0.6]), xtol=1e-8, ftol=1e-8)
    r1 = _residuals(r.x, lines)
    return {"k1": float(r.x[0]), "rms_before": float(np.sqrt(np.mean(r0 ** 2))),
            "rms_after": float(np.sqrt(np.mean(r1 ** 2))), "n_lines": len(lines)}


def calibrate(video, center=None, n=60, **kw):
    """(collected lines, fit result or None). `center` defaults to the saved calibration of this video."""
    if center is None:
        cal = LB.load_center(video)
        if cal is None:
            raise ValueError("no calibrated centre for '{}': run camera_center.py --save first".format(video))
        center = cal["center"]
    lines = collect_lines(video, center, n, **kw)
    return lines, fit_shared_k1(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="soccer")
    ap.add_argument("--n", type=int, default=60, help="frames spread over the clip to try")
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--max-center-dev", type=float, default=8.0, help="metres: how far the decomposed centre may be from the calibrated one")
    a = ap.parse_args()
    t0 = time.time()
    lines, fit = calibrate(a.video, n=a.n, min_score=a.min_score, max_center_dev=a.max_center_dev)
    print("{} of {} frames gave a usable broadcast-camera line ({:.0f}s)".format(len(lines), a.n, time.time() - t0))
    for ln in lines:
        print("  frame {:>5}  f/w {:.2f}  {} pts".format(ln["frame"], ln["f_over_w"], len(ln["pts"])))
    if fit is None:
        print("no lines collected -- nothing to fit")
        return
    print("\nshared k1 = {:.3f}  (rms bow {:.2f} -> {:.2f} px over {} lines)".format(
        fit["k1"], fit["rms_before"], fit["rms_after"], fit["n_lines"]))


if __name__ == "__main__":
    main()
