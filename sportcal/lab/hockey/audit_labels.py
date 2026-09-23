"""Audit of the NHL training labels: hand-click a sample of labelled frames and measure how far the stored
homography is from the hand one. Mostly for the frames labelled by segmentation + DLT (hockey.md section 7), whose
accuracy was never measured; a few of the older `nhl_prior` labels are sampled too. Write-up: hockey.md section 0b.

    python -m sportcal.lab.hockey.audit_labels --sample      # writes datasets/hockeyrink_nhl_audit/queue.json
    set SPORTCAL_HOCKEY_LABELS=hockeyrink_nhl_audit          # PowerShell: $env:SPORTCAL_HOCKEY_LABELS="hockeyrink_nhl_audit"
    venv/Scripts/python.exe -m streamlit run sportcal/lab/common/annotate_val_app.py   # "hockey": clicks the queue in order
    python -m sportcal.lab.hockey.audit_labels               # stored H vs the clicks, per frame and per source

Needs datasets/hockey_h.jsonl (`build_h_index`).
"""
import argparse
import json

import numpy as np

from sportcal.core.geometry import geom_error
from sportcal.lab.hockey.build_h_index import OUT as INDEX
from sportcal.lab.hockey.build_h_index import canonicalize, video_of
from sportcal.paths import DATASETS
from sportcal.sports.hockey import rink

AUDIT = DATASETS / "hockeyrink_nhl_audit"
SAMPLE = {"auto_seg": 30, "nhl_prior": 10}
GRID = np.stack(np.meshgrid(np.arange(0.0, rink.RINK_NHL["length"] + 1e-9, 1.0),
                            np.arange(0.0, rink.RINK_NHL["width"] + 1e-9, 1.0)), -1).reshape(-1, 2)


def load_index():
    return [json.loads(line) for line in open(INDEX, encoding="utf-8")]


def sample(rows, n_per_source=SAMPLE):
    """[(video, frame, source)]: per source, frames evenly spread in time inside each video, videos taking turns so
    that stopping half-way still covers every video."""
    out = []
    for source, n in n_per_source.items():
        by_video = {}
        for r in rows:
            if r.get("label_source", r["source"]) == source and r["dataset"] == "hockeyrink_nhl":
                by_video.setdefault(r["video"], []).append(r["frame"])
        vids = sorted(by_video)
        picks = []
        for j, v in enumerate(vids):
            fr = sorted(by_video[v])
            m = n // len(vids) + (j < n % len(vids))
            picks.append([(v, fr[i], source) for i in dict.fromkeys(np.linspace(0, len(fr) - 1, m).round().astype(int))])
        for k in range(max(map(len, picks))):
            out += [p[k] for p in picks if k < len(p)]
    return out


def _proj(H, world):
    q = np.c_[world, np.ones(len(world))] @ H.T
    return q[:, :2] / q[:, 2:]


def evaluate(rows, hand):
    """[(video, frame, source, grid error, error at the clicked points, n clicks)], px at 1920, for every hand-labelled
    frame. Grid: median over a 1 m rink grid inside the frame. Clicks: how far the stored H puts the clicked points from
    the clicks themselves - the direct measure, no extrapolation."""
    p = rink.RINK_NHL
    tpl = rink.build_template(p)
    mx = np.array([[-1.0, 0, p["length"]], [0, 1, 0], [0, 0, 1]])
    my = np.array([[1.0, 0, 0], [0, -1, p["width"]], [0, 0, 1]])
    # once the audit labels are in the index, the label they replaced is kept as H_label / label_source
    stored = {(r["video"], r["frame"]): dict(r, H=r.get("H_label", r["H"]), source=r.get("label_source", r["source"]))
              for r in rows if r["dataset"] == "hockeyrink_nhl" and (r["source"] != "hand" or "H_label" in r)}
    out = []
    for (v, f), d in sorted(hand.items()):
        r = stored.get((v, f))
        if r is None:
            continue
        k = np.array([int(i) for i in d["clics"]])
        px = np.array(list(d["clics"].values()), float)
        # the app saves H after its own x mirror while the clicks keep the names they were clicked with
        H_hand = min((np.asarray(d["H"], float), np.asarray(d["H"], float) @ mx),
                     key=lambda H: np.linalg.norm(_proj(H, tpl[k]) - px, axis=1).mean())
        H_hand, flips = canonicalize(H_hand, p)
        world = _proj((mx if "x" in flips else np.eye(3)) @ (my if "y" in flips else np.eye(3)), tpl[k])
        H = np.asarray(r["H"], float)
        at_clicks = float(np.median(np.linalg.norm(_proj(H, world) - px, axis=1))) * 1920.0 / r["w"]
        out.append((v, f, r["source"], geom_error(H, H_hand, GRID, r["w"], r["h"]), at_clicks, len(k)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", action="store_true", help="write the queue of frames to hand-label")
    args = ap.parse_args()
    rows = load_index()

    if args.sample:
        q = sample(rows)
        AUDIT.mkdir(parents=True, exist_ok=True)
        (AUDIT / "queue.json").write_text(json.dumps([[v, f] for v, f, _ in q]))
        print("{} frames queued in {} ({})".format(len(q), AUDIT / "queue.json",
                                                    ", ".join("{} {}".format(n, s) for s, n in SAMPLE.items())))
        return

    hand = {}
    for line in open(AUDIT / "clicks.jsonl", encoding="utf-8"):
        d = json.loads(line)
        hand[video_of(d["id"])] = d          # a re-saved frame keeps its last label
    res = evaluate(rows, hand)
    print("{:<8} {:>7} {:<10} {:>8} {:>10} {:>7}".format("video", "frame", "source", "grid px", "clicks px", "clicks"))
    for v, f, s, g, c, n in res:
        print("{:<8} {:>7} {:<10} {:>8.1f} {:>10.1f} {:>7}".format(v, f, s, g, c, n))
    for s in SAMPLE:
        g = np.array([x[3] for x in res if x[2] == s])
        c = np.array([x[4] for x in res if x[2] == s])
        if len(c):
            print("{:<10} n={:<3} grid p50 {:.1f} p90 {:.1f} | clicks p50 {:.1f} p90 {:.1f}  <10 px {:.0f}%  >50 px {}".format(
                s, len(c), np.median(g), np.percentile(g, 90), np.median(c), np.percentile(c, 90),
                100 * (c < 10).mean(), int((c > 50).sum())))


if __name__ == "__main__":
    main()
