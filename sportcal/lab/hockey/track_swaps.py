"""Tracking step 0 (hockey.md section 15): do the stock trackers swap players' identities within a shot?

No hockey tracks with IDs exist here, so swaps are counted through indirect signals in rink metres - a jump no skater
makes, a change of course no skater makes, a change of jersey colour - and those signals are checked by eye on
contact sheets.

    python -m sportcal.lab.hockey.track_swaps cache nhl4.mp4 nhl7.mp4 nhl10.mp4      # GPU -> runs/tracking/<video>.npz
    python -m sportcal.lab.hockey.track_swaps signals nhl4 nhl7 nhl10                 # measures the noise margin
    python -m sportcal.lab.hockey.track_swaps signals nhl11 nhl12 nhl13 nhl14 --margin <frozen margin>

Hand labels go to runs/tracking/labels.csv, one `event_id,label` line per event read off the sheets in
runs/tracking/sheets/ (label: swap / no / unclear).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

from sportcal import sports
from sportcal.core.teams import fit_teams, jersey_histogram, team_of
from sportcal.paths import ROOT, RUNS

V_MAX = 12.0        # m/s, fastest skater on record rounded up (source: hockey.md section 15)
A_MAX = 15.0        # m/s^2, above the peak push-off acceleration of elite skaters (source: hockey.md section 15)
TEAM_RATIO = 0.8    # a box's team counts when its distance ratio to the two jersey centroids is below this
TRACKERS = ("bytetrack", "botsort")
SIGNALS = ("jump", "turn", "team", "any")
FRESH_VIDEOS = {"nhl11", "nhl12", "nhl13", "nhl14"}
KPLINE = RUNS / "kpline" / "finetune" / "best_h.pt"
OUT = RUNS / "tracking"


def jumps(t, xy, v_max, margin):
    """Indices i where the step from observation i-1 to i is longer than a skater covers in that time plus the
    feet-point noise `margin`: the ID moved to someone else, or the homography moved."""
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.flatnonzero(step > v_max * np.diff(t) + margin) + 1


def turns(t, xy, a_max, fps, window=0.5, min_seen=0.8):
    """Indices where the mean velocity over the `window` s before and over the `window` s after differ by more than
    a_max * window (the two window centres are `window` s apart): a change of course no skater makes. Each window
    needs `min_seen` of its frames observed, so gaps and the ends of a track do not fire it."""
    idx = np.arange(len(t))
    lo = np.searchsorted(t, t - window)
    hi = np.searchsorted(t, t + window, "right") - 1
    need = min_seen * window * fps
    ok = (idx - lo + 1 >= need) & (hi - idx + 1 >= need) & (t - t[lo] > 0) & (t[hi] - t > 0)
    v_before = (xy - xy[lo]) / np.maximum(t - t[lo], 1e-9)[:, None]
    v_after = (xy[hi] - xy) / np.maximum(t[hi] - t, 1e-9)[:, None]
    return np.flatnonzero(ok & (np.linalg.norm(v_after - v_before, axis=1) > a_max * window))


def team_changes(t, team, hold=1.0):
    """Indices where a track's confident team votes (team >= 0) switch: the majority over the `hold` s around each vote
    holds one team for >= `hold` s, then the other for >= `hold` s. Shorter flickers are misread shirts, not swaps."""
    v = np.flatnonzero(team >= 0)
    if len(v) < 2:
        return np.array([], int)
    tv = t[v]
    lo, hi = np.searchsorted(tv, tv - hold / 2), np.searchsorted(tv, tv + hold / 2, "right")
    ones = np.r_[0, np.cumsum(team[v])]
    major = ((ones[hi] - ones[lo]) / (hi - lo) > 0.5).astype(int)
    b = np.r_[0, np.flatnonzero(np.diff(major)) + 1, len(v)]
    dur = tv[b[1:] - 1] - tv[b[:-1]]
    return np.array([v[b[k]] for k in range(1, len(b) - 1) if dur[k - 1] >= hold and dur[k] >= hold], int)


def h_jump_frames(jump_frames, obs_frames, min_tracks=3):
    """Frames on which more than half of the tracks seen, and at least `min_tracks`, jump at once: the homography
    moved, not the players."""
    seen = np.bincount(obs_frames)
    jumped = np.bincount(jump_frames, minlength=len(seen))
    return np.flatnonzero((jumped >= min_tracks) & (jumped > seen / 2))


def merge(key, t, idx, gap=1.0):
    """The first of each run of flags on one track that follow each other within `gap` s: one event per run. `key` and
    `t` are sorted by (track, time), as `observations` returns them."""
    idx = np.unique(idx)
    if len(idx) == 0:
        return idx
    return idx[np.r_[True, (key[idx][1:] != key[idx][:-1]) | (np.diff(t[idx]) > gap)]]


def noise(xy, k):
    """Distance from each position to the median of the k consecutive positions around it (k odd): the jitter of the
    feet point, which the jump test must tolerate."""
    # ponytail: consecutive rows, ignores time gaps inside a track (a gap inflates the estimate slightly); use a
    # time-windowed median if the margin ever looks too loose
    return np.linalg.norm(xy - median_filter(xy, size=(k, 1), mode="nearest"), axis=1)


def _tracks(key):
    """(start, end) of each track's rows in arrays sorted by track."""
    if len(key) == 0:
        return []
    s = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return list(zip(s, np.r_[s[1:], len(key)]))


def observations(c, tracker, sport):
    """The boxes of one cached video that `tracker` tracked and that can be measured - on a frame with H, feet inside
    the field. Dict of arrays sorted by (track, frame): key (shot * 10**6 + tracker id, as IDs restart at every cut),
    frame, t (s), xy (m), team (0/1, -1 when unsure or without jersey colour), xyxy."""
    f = c["det_frame"]
    H = c["H"][f]
    has_h = ~np.isnan(H[:, 0, 0])
    feet = np.c_[(c["xyxy"][:, 0] + c["xyxy"][:, 2]) / 2, c["xyxy"][:, 3], np.ones(len(f))]
    xy = np.full((len(f), 2), np.nan)
    if has_h.any():
        w = np.einsum("nij,nj->ni", np.linalg.inv(H[has_h]), feet[has_h])
        xy[has_h] = w[:, :2] / w[:, 2:]
    x0, x1, y0, y1 = sport.box
    inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    team = np.full(len(f), -1)
    hist = c["hist"].astype(np.float32)
    coloured = inside & ~np.isnan(hist[:, 0])
    if coloured.sum() >= 2:
        cv2.setRNGSeed(0)
        teams, ratio = team_of(hist[coloured], fit_teams(hist[coloured]))
        team[coloured] = np.where(ratio < TEAM_RATIO, teams, -1)
    ids = c[f"id_{tracker}"]
    key = c["shot"][f].astype(np.int64) * 10**6 + ids
    sel = np.flatnonzero(inside & (ids >= 0))
    sel = sel[np.lexsort((f[sel], key[sel]))]
    return {"key": key[sel], "frame": f[sel], "t": f[sel] / float(c["fps"]), "xy": xy[sel], "team": team[sel],
            "xyxy": c["xyxy"][sel]}


def _hist_or_nan(frame, box):
    h = jersey_histogram(frame, box)
    return np.full(128, np.nan) if h is None else h


def cache(video, device="cuda:0", imgsz=1280, max_frames=0):
    """One GPU pass over `video`: shot cuts, the product's H, COCO person boxes with their jersey histogram, and the IDs
    the two stock trackers give them (both recreated at every cut). Saved to runs/tracking/<stem>[-<max_frames>f].npz."""
    import supervision as sv
    from scenedetect.common import FrameTimecode
    from scenedetect.detectors import ContentDetector
    from scenedetect.scene_manager import compute_downscale_factor
    from tqdm import tqdm
    from trackers import BoTSORTTracker, ByteTrackTracker
    from ultralytics import YOLO

    from sportcal.product.hockey import build_pipeline     # lab -> product: only here, to measure what the product sees

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = compute_downscale_factor(max(w, h))
    small = (round(w / k), round(h / k))
    yolo = YOLO(str(ROOT / "yolo26m.pt"))
    pipe = build_pipeline(None, kpline_weights=KPLINE, hold_frames=15, device=device, smooth=0.1)
    cuts = ContentDetector()

    def new_trackers():
        return {"bytetrack": ByteTrackTracker(frame_rate=fps), "botsort": BoTSORTTracker(frame_rate=fps)}

    trk, shot_no = new_trackers(), 0
    shot, Hs, method, det = [], [], [], defaultdict(list)
    for i in tqdm(range(max_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), unit="frame"):
        ok, frame = cap.read()
        if not ok:
            break
        if cuts.process_frame(FrameTimecode(i, fps=fps), cv2.resize(frame, small)):
            trk, shot_no = new_trackers(), shot_no + 1
            pipe.reset()
        shot.append(shot_no)
        est = pipe(frame)
        Hs.append(np.full((3, 3), np.nan) if est is None else est.H)
        method.append("none" if est is None else est.method)
        d = sv.Detections.from_ultralytics(
            yolo.predict(frame, classes=[0], conf=0.1, imgsz=imgsz, device=device, verbose=False)[0])
        d.data["i"] = np.arange(len(d))       # trackers reorder and drop rows; this maps them back
        det["det_frame"].append(np.full(len(d), i, np.int32))
        det["xyxy"].append(d.xyxy.astype(np.float32))
        det["conf"].append(d.confidence.astype(np.float32))
        det["hist"].append(np.array([_hist_or_nan(frame, b) for b in d.xyxy], np.float16).reshape(-1, 128))
        for name, tr in trk.items():
            out = tr.update(d, frame=frame) if name == "botsort" else tr.update(d)   # only BoT-SORT uses the frame
            ids = np.full(len(d), -1, np.int32)
            if len(out):
                ids[out.data["i"]] = out.tracker_id
            det[f"id_{name}"].append(ids)
    path = OUT / (Path(video).stem + (f"-{max_frames}f" if max_frames else "") + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    Hs = np.array(Hs)
    np.savez_compressed(path, video=str(Path(video).resolve()), fps=fps, shot=np.array(shot, np.int32), H=Hs,
                        method=np.array(method), **{k: np.concatenate(v) for k, v in det.items()})
    print(f"{path}: {len(shot)} frames, {np.isfinite(Hs[:, 0, 0]).mean():.0%} with H, "
          f"{sum(map(len, det['xyxy']))} boxes, {shot_no + 1} shots")


def find_events(o, fps, margin):
    """Where events start, as observation indices per signal ('any' = the union of jump, turn and team), flags merged
    per track within 1 s. Jump and turn flags within 0.5 s of an H-jump frame are dropped: the homography moved, not a
    player. Also returns those H-jump frames."""
    raw = defaultdict(list)
    for a, b in _tracks(o["key"]):
        t, xy = o["t"][a:b], o["xy"][a:b]
        raw["jump"].append(a + jumps(t, xy, V_MAX, margin))
        raw["turn"].append(a + turns(t, xy, A_MAX, fps))
        raw["team"].append(a + team_changes(t, o["team"][a:b]))
    flags = {s: np.concatenate(raw[s] or [np.array([], int)]).astype(int) for s in ("jump", "turn", "team")}
    hj = h_jump_frames(o["frame"][flags["jump"]], o["frame"])
    if len(hj):
        for s in ("jump", "turn"):
            f = o["frame"][flags[s]]
            flags[s] = flags[s][np.abs(f[:, None] - hj[None]).min(1) > 0.5 * fps]
    flags["any"] = np.concatenate([flags["jump"], flags["turn"], flags["team"]])
    return {s: merge(o["key"], o["t"], v) for s, v in flags.items()}, hj


def read_labels(path):
    """{event_id: label} typed by hand into `path` (swap / no / unclear). Created with its header when missing and never
    rewritten: it holds the only copy of the hand labels."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            f.write("event_id,label\n")
    with open(path, newline="") as f:
        return {r["event_id"]: r["label"].strip() for r in csv.DictReader(f)}


def _around(o, i, dt=0.5):
    """(frame, box) of the track of observation i at dt s before, at i, and dt s after (clipped to the track)."""
    a = np.searchsorted(o["key"], o["key"][i])
    b = np.searchsorted(o["key"], o["key"][i], "right")
    rows = [min(a + np.searchsorted(o["t"][a:b], o["t"][i] + d), b - 1) for d in (-dt, 0.0, dt)]
    return [(int(o["frame"][r]), o["xyxy"][r]) for r in rows]


def _crop(img, box, size=(300, 200)):
    """The region around a box, 3 box heights tall (at least 200 px), with the box drawn: enough to see who is in it."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    hh = max(1.5 * (y2 - y1), 100.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r0, r1 = int(max(0, cy - hh)), int(min(img.shape[0], cy + hh))
    c0, c1 = int(max(0, cx - 1.5 * hh)), int(min(img.shape[1], cx + 1.5 * hh))
    return cv2.resize(img[r0:r1, c0:c1], size)


def _sheets(folder, stem, items, per_png=10):
    """PNG contact sheets: per event its id, then three crops (0.5 s before, at, 0.5 s after) with the track's box."""
    folder.mkdir(parents=True, exist_ok=True)
    caps = {}
    for p in range(0, len(items), per_png):
        rows = []
        for eid, video, shots in items[p:p + per_png]:
            if video not in caps:
                caps[video] = cv2.VideoCapture(video)
            tiles = []
            for f, box in shots:
                caps[video].set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, img = caps[video].read()
                tiles.append(_crop(img, box) if ok else np.zeros((200, 300, 3), np.uint8))
            head = np.zeros((22, 900, 3), np.uint8)
            cv2.putText(head, eid, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            rows += [head, np.hstack(tiles)]
        cv2.imwrite(str(folder / f"{stem}_{p // per_png}.png"), np.vstack(rows))


def signals(names, margin=None, n_sample=20, seed=0):
    """Swap signals of the cached videos `names`: rates per video and tracker, contact sheets of `n_sample` random
    events per tracker and signal, and, for the events labelled in runs/tracking/labels.csv, each signal's precision
    and the real swaps per minute of play with H."""
    fresh = sorted(set(names) & FRESH_VIDEOS)
    if fresh and margin is None:
        raise SystemExit(f"{fresh} are `fresh` games: pass --margin, the value frozen on the development videos "
                         "(hockey.md section 15)")
    sport = sports.get("hockey-nhl")
    caches = {n: dict(np.load(OUT / f"{n}.npz")) for n in names}
    obs = {(n, tr): observations(caches[n], tr, sport) for n in names for tr in TRACKERS}
    if margin is None:
        margin = 2 * np.percentile(np.concatenate(
            [noise(o["xy"][a:b], max(3, round(0.1 * float(caches[n]["fps"])) | 1))
             for (n, _), o in obs.items() for a, b in _tracks(o["key"])]), 99)
        print(f"margin measured on these videos: {margin:.2f} m - freeze it in hockey.md 15, pass --margin on `fresh`")
    pool, minutes = defaultdict(list), defaultdict(float)
    print(f"{'video':8} {'tracker':9} {'min H':>6} {'IDs/player':>10} {'len s':>6} "
          + " ".join(f"{s + '/min':>9}" for s in SIGNALS) + f" {'Hjump/min':>9}")
    for (n, tr), o in obs.items():
        c = caches[n]
        fps = float(c["fps"])
        ev, hj = find_events(o, fps, margin)
        m = max(np.isfinite(c["H"][:, 0, 0]).sum() / fps / 60, 1e-9)
        minutes[tr] += m
        for s, idx in ev.items():
            pool[tr, s] += [(n, int(i)) for i in idx]
        shots = c["shot"][o["frame"]]
        ids = sum(len(np.unique(o["key"][shots == s])) for s in np.unique(shots))
        players = sum(np.median(np.unique(o["frame"][shots == s], return_counts=True)[1]) for s in np.unique(shots))
        spans = _tracks(o["key"])
        length = np.median([o["t"][b - 1] - o["t"][a] for a, b in spans]) if spans else 0.0
        print(f"{n:8} {tr:9} {m:6.1f} {ids / max(players, 1):10.2f} {length:6.1f} "
              + " ".join(f"{len(ev[s]) / m:9.2f}" for s in SIGNALS) + f" {len(hj) / m:9.2f}")
    labels = read_labels(OUT / "labels.csv")
    rng = np.random.default_rng(seed)
    folder = OUT / "sheets" / "-".join(names)
    for tr in TRACKERS:
        for s in SIGNALS:
            events = pool[tr, s]
            pick = [events[j] for j in rng.choice(len(events), min(n_sample, len(events)), replace=False)]
            eids = [f"{n}:{tr}:{obs[n, tr]['key'][i]}:{obs[n, tr]['frame'][i]}" for n, i in pick]
            _sheets(folder, f"{tr}_{s}",
                    [(e, str(caches[n]["video"]), _around(obs[n, tr], i)) for e, (n, i) in zip(eids, pick)])
            got = [labels[e] for e in eids if labels.get(e) in ("swap", "no")]
            line = f"{tr:9} {s:5} events {len(events):5}  labelled {len(got):2}/{len(eids)}"
            if got:
                p = got.count("swap") / len(got)
                line += f"  precision {p:.2f}"
                if s == "any":
                    line += f"  real swaps/min {len(events) / minutes[tr] * p:.2f}"
            print(line)
    print(f"sheets: {folder}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cache", help="one GPU pass per video -> runs/tracking/<video>.npz")
    a.add_argument("videos", nargs="+")
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--imgsz", type=int, default=1280)
    a.add_argument("--max-frames", type=int, default=0, help="smoke runs: stop here, cache named <video>-<n>f")
    s = sub.add_parser("signals", help="swap signals, rates, contact sheets and, with labels.csv, precision")
    s.add_argument("names", nargs="+", help="cache names under runs/tracking, e.g. nhl4 nhl7 nhl10")
    s.add_argument("--margin", type=float, help="feet-point noise margin (m): measured when omitted, required on fresh")
    args = ap.parse_args()
    if args.cmd == "cache":
        for v in args.videos:
            cache(v, args.device, args.imgsz, args.max_frames)
    else:
        signals(args.names, args.margin)


if __name__ == "__main__":
    main()
