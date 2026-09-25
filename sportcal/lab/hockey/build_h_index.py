"""Per-frame homography index of every labelled hockey frame, with a per-video split.

The keypoint + line model renders all its training targets (keypoint heatmaps, line extremities, masks) from ONE
homography per frame plus the rink template, so this index is its single label source. The split decision and the
counts are in docs/experiments/hockey.md section 0b.

    python -m sportcal.lab.hockey.build_h_index        # writes datasets/hockey_h.jsonl and prints a summary

One JSON object per line: id ("<dataset>/<stem>"), image (relative to ROOT), dataset, video, frame (index in the
video, None for SHL), template ("hockey-nhl" | "hockey-iihf"), source ("hand" | "auto_seg" | "nhl_prior" |
"hockeyrink"), w, h, H (world metres -> image px, H[2,2] = 1), fit (None for hand labels, else {"inliers",
"resid_px"} with the residual at 1920 px), split ("train" | "dev" | "test" | "test_leaky" | "excluded").
A `hockeyrink_nhl` frame that was hand-clicked again (the audit, `audit_labels`) takes the hand H and keeps what it
replaced: `label_source` and `H_label`.
"""
import argparse
import json
import re
from collections import Counter

import cv2
import numpy as np

from sportcal.core.camera import is_plausible_view
from sportcal.core.labels import read_label
from sportcal.lab.hockey.make_line_masks import fit_from_label
from sportcal.paths import DATASETS, ROOT
from sportcal.sports.hockey import rink

# Videos that never reach train or dev. The clean test set is the part of the hand-labelled valh that comes from them.
HOLDOUT_VIDEOS = ("nhl4", "nhl10", "nhl9")
OUT = DATASETS / "hockey_h.jsonl"
PARAMS = {"hockey-nhl": rink.RINK_NHL, "hockey-iihf": rink.RINK_IIHF}
TEMPLATES = {k: rink.build_template(p) for k, p in PARAMS.items()}
FITTED = (("hockeyrink_nhl", "hockey-nhl"), ("hockeyrink", "hockey-iihf"))
HAND = "hockeyrink_nhl_valh"             # new frames, hand-labelled
EXTRA = ("hockeyrink_nhl_endview",)      # new frames hand-labelled for training (views the train set lacks)
RELABELLED = "hockeyrink_nhl_audit"      # hockeyrink_nhl frames hand-labelled again


def canonicalize(H, p):
    """(H, flips): H mirrored so that zone A (low x) sits on the image left and the y = 0 boards (the far side) above the
    y = W ones - the orientation of nearly every hand label. The rink is symmetric in x and in y, so two labels that differ
    by a mirror draw the same lines while every keypoint and line swaps its name: poison for a model that predicts named
    points. The click labeller only fixes x (click_labeler docstring); the index fixes both. The referee crease is the
    one marking that is not y-symmetric: the template puts it at y = W, which after this is always the camera side."""
    L, W = p["length"], p["width"]

    def img(H, x, y):
        q = H @ np.array([x, y, 1.0])
        return q[:2] / q[2]

    flips = ""
    if img(H, L / 2 + 5, W / 2)[0] < img(H, L / 2 - 5, W / 2)[0]:
        H, flips = H @ np.array([[-1.0, 0, L], [0, 1, 0], [0, 0, 1]]), flips + "x"
    if img(H, L / 2, W / 2 - 5)[1] > img(H, L / 2, W / 2 + 5)[1]:
        H, flips = H @ np.array([[1.0, 0, 0], [0, -1, W], [0, 0, 1]]), flips + "y"
    return H / H[2, 2], flips


def video_of(stem):
    """('nhl5', 1234) for 'nhl5_001234' or 'nhl5_01234'; SHL frames (uuid names) come from no known video: ('shl', None)."""
    m = re.match(r"(clip\d*|nhl\d+)_(\d+)$", stem)
    return (m.group(1), int(m.group(2))) if m else ("shl", None)


def split_of(dataset, orig_split, video, hand=False, propagated=False):
    """Per-video split: a held-out video only ever lands in the test sets (a hand label from it is a test frame). A label
    the click labeller carried from another one (`click_labeler.propaga`) is never a test frame."""
    if propagated:
        return "excluded" if video in HOLDOUT_VIDEOS else "train"
    if dataset == HAND:
        return "test" if video in HOLDOUT_VIDEOS else "test_leaky"
    if dataset in EXTRA:
        return "test" if video in HOLDOUT_VIDEOS else "train"
    if video in HOLDOUT_VIDEOS:
        return "test" if hand else "excluded"
    return "train" if orig_split == "train" else "dev"


def _size(img):
    im = cv2.imread(str(img)) if img.exists() else None
    return (im.shape[1], im.shape[0]) if im is not None else (None, None)


def _hand_labels(dataset):
    """{(video, frame): (H, source)} from a click-labelled dataset; a re-saved frame keeps its last H. source is
    "propagated" when the saved H was a proposal carried from another label (the labeller records its "origen")."""
    path = DATASETS / dataset / "clicks.jsonl"
    if not path.exists():
        return {}
    return {video_of(d["id"]): (np.asarray(d["H"], float), "propagated" if d.get("origen") else "hand")
            for d in map(json.loads, open(path, encoding="utf-8"))}


def _row(dataset, stem, img, w, h, template, source, H, fit, orig_split, **extra):
    video, frame = video_of(stem)
    H, flips = canonicalize(np.asarray(H, float), PARAMS[template])
    return {"id": "{}/{}".format(dataset, stem), "image": img.relative_to(ROOT).as_posix(), "dataset": dataset,
            "video": video, "frame": frame, "template": template, "source": source, "w": w, "h": h,
            "H": H.tolist(), "flips": flips, "fit": fit,
            "split": split_of(dataset, orig_split, video, source == "hand", source == "propagated"), **extra}


def build():
    """(rows, Counter of why frames were left out)."""
    rows, dropped = [], Counter()
    auto = {p.stem for p in (DATASETS / "hockeyrink_auto_seg" / "images" / "train").glob("*.jpg")}
    relabelled = _hand_labels(RELABELLED)
    for dataset, template in FITTED:
        tpl = TEMPLATES[template]
        for orig_split in ("train", "val"):
            for lbl in sorted((DATASETS / dataset / "labels" / orig_split).glob("*.txt")):
                img = DATASETS / dataset / "images" / orig_split / (lbl.stem + ".jpg")
                rec = read_label(lbl)
                w, h = _size(img)
                if rec is None or w is None:
                    dropped["{}: no label or image".format(dataset)] += 1
                    continue
                # the gate of relabel_reproject / gen_synthetic_lines, but from 6 visible points instead of 8 (the minimum
                # label_from_H writes): 6-7 point frames are the sparse views (few markings in frame) the model most needs,
                # and 6 points still leave 2 redundant ones for the residual check to mean something
                fit, info = fit_from_label(rec[2], tpl, w, h, 6, 6.0, 8.0)
                source = "hockeyrink" if dataset == "hockeyrink" else ("auto_seg" if lbl.stem in auto else "nhl_prior")
                H_hand = relabelled.get(video_of(lbl.stem), (None,))[0] if dataset == "hockeyrink_nhl" else None
                if H_hand is not None:
                    old = {"H_label": canonicalize(fit[0], PARAMS[template])[0].tolist()} if fit else {}
                    rows.append(_row(dataset, lbl.stem, img, w, h, template, "hand", H_hand, None, orig_split,
                                     label_source=source, **old))
                    continue
                if fit is None:
                    dropped["{}: {}".format(dataset, info)] += 1
                    continue
                if not is_plausible_view(fit[0], w, h, (0.0, PARAMS[template]["length"], 0.0, PARAMS[template]["width"])):
                    dropped["{}: no camera gives this H (hockey.md 14d)".format(dataset)] += 1
                    continue
                rows.append(_row(dataset, lbl.stem, img, w, h, template, source, fit[0],
                                 {"inliers": len(fit[1]), "resid_px": round(float(info), 2)}, orig_split))
    for dataset in (HAND, *EXTRA):
        for (video, frame), (H, source) in sorted(_hand_labels(dataset).items()):
            stem = "{}_{:06d}".format(video, frame)
            img = DATASETS / dataset / "images" / "val" / (stem + ".jpg")
            w, h = _size(img)
            if w is None:
                dropped["{}: no image".format(dataset)] += 1
                continue
            rows.append(_row(dataset, stem, img, w, h, "hockey-nhl", source, H, None, "val"))
    return rows, dropped


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    rows, dropped = build()
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print("{} frames -> {}".format(len(rows), OUT))
    for (split, source), n in sorted(Counter((r["split"], r["source"]) for r in rows).items()):
        vids = Counter(r["video"] for r in rows if r["split"] == split and r["source"] == source)
        print("  {:<11} {:<11} {:>4}   {}".format(split, source, n, " ".join("{}:{}".format(*kv) for kv in sorted(vids.items()))))
    for source in sorted({r["source"] for r in rows}):
        print("  mirrored to the convention, {:<11} {}".format(
            source, dict(Counter(r["flips"] or "-" for r in rows if r["source"] == source))))
    for why, n in sorted(dropped.items()):
        print("  left out  {:>4}  {}".format(n, why))
    res = np.array([r["fit"]["resid_px"] for r in rows if r["fit"]])
    print("  fit residual (px at 1920): p50 {:.2f}  p90 {:.2f}".format(np.median(res), np.percentile(res, 90)))


if __name__ == "__main__":
    main()
