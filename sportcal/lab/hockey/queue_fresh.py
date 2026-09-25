"""Queue frames of new games for the "fresh" hand labels (`build_h_index.FRESH`): as few frames to click as possible
while staying a fair sample. One frame per camera shot (shots split where a grey thumbnail jumps), only shots where the
HockeyRink detector sees the rink (`fetch_clips` writes rink_preds_<video>.json: a box and >= 4 keypoints above 0.5,
which drops close-ups, crowd replays and graphics): the middle frame of each, plus one every --every s inside long shots
(the main camera fills most of a broadcast); then --per-video of them spread evenly over the clip. Shots are cached in
<dataset>/shots_<video>.json (a video takes about a minute to scan). None of our models picks them, so the set stays independent of what it will measure (hockey.md 14h).

    python -m sportcal.lab.hockey.queue_fresh --videos nhl11 nhl12 nhl13 nhl14

ponytail: a thumbnail-correlation cut detector, fine for broadcast hard cuts; dissolves and fast pans can split or merge
shots, which only changes which frames are offered, not the labels.
"""
import argparse
import json

import cv2
import numpy as np

from sportcal.paths import DATASETS, ROOT

OUT = DATASETS / "hockeyrink_nhl_fresh"


def _thumb(img):
    g = cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def shots(video, step=6, cut=0.35, min_len_s=1.0):
    """[(first, last)] frame ranges of the shots of ROOT/<video>.mp4, from thumbnails every `step` frames."""
    cap = cv2.VideoCapture(str(ROOT / (video + ".mp4")))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, start, prev, i = [], 0, None, 0
    while cap.grab():
        if i % step == 0:
            ok, img = cap.retrieve()
            t = _thumb(img)
            if prev is not None and 1.0 - float(t @ prev) > cut:
                out.append((start, i - step))
                start = i
            prev = t
        i += 1
    out.append((start, i - 1))
    return [(a, b) for a, b in out if b - a >= min_len_s * fps]


def sees_rink(preds, frame, every):
    """The HockeyRink prediction nearest to `frame` has a confident box and >= 4 confident keypoints."""
    p = preds.get(int(round(frame / every)) * every)
    return p is not None and p["box_conf"] > 0.5 and sum(c > 0.5 for c in p["conf"]) >= 4


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--per-video", type=int, default=10)
    ap.add_argument("--every", type=float, default=10.0, help="seconds between frames inside a long shot")
    args = ap.parse_args()
    lists = []
    for v in args.videos:
        rp = json.load(open(ROOT / "rink_preds_{}.json".format(v)))
        preds = {p["frame"]: p for p in rp["preds"]}
        cache = OUT / "shots_{}.json".format(v)
        if not cache.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(shots(v)))
        sh = json.loads(cache.read_text())
        step = int(args.every * rp.get("fps", 59.94))
        cand = [f for a, b in sh for f in sorted({(a + b) // 2, *range(a + step // 2, b, step)})]
        ok = [f for f in cand if sees_rink(preds, f, rp["every"])]
        pick = [ok[int(j)] for j in np.linspace(0, len(ok) - 1, min(args.per_video, len(ok))).round()] if ok else []
        print("{:<6} {} shots, {} candidate frames that see the rink, {} queued".format(v, len(sh), len(ok), len(pick)), flush=True)
        lists.append([[v, f] for f in pick])
    queue = [q for k in range(max(map(len, lists))) for q in (lst[k] for lst in lists if k < len(lst))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "queue.json").write_text(json.dumps(queue))
    print("{} frames -> {}".format(len(queue), OUT / "queue.json"))


if __name__ == "__main__":
    main()
