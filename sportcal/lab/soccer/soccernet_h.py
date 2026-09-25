"""Per-frame H index for soccer from the SoccerNet camera-calibration annotations (calibration-2023), the soccer
counterpart of `lab/hockey/build_h_index`: one H per frame, fitted to the annotated pitch markings, so every target of the
keypoint + line model can be rendered from H and the template. Results: docs/experiments/soccer.md section 19.

Each annotation gives, per marking class, points on that marking in normalised image coordinates. Straight markings
become image lines (least squares through their points) matched to the template's world lines; with the points where
two of them meet on the pitch, one DLT (`core.geometry.solve_points_lines`) gives H; the circle points only check it.
A frame is kept when its straight-line points lie within --max-px of the lines its H draws, its circle points within
--max-circle-px of the circles (looser: arcs are clicked less precisely and a wide lens bends them), and a real camera
can give the H (`core.camera.is_plausible_view`). Thresholds read off a 1500-frame sample: soccer.md section 19.
World frame: ours (origin at the centre, +Y = far touchline); SoccerNet's y points the other way. Goal posts and
crossbars are off the ground plane and are not used.

    pip install SoccerNet     # then, once: SoccerNetDownloader(LocalDirectory=...).downloadDataTask("calibration-2023", ...)
    python -m sportcal.lab.soccer.soccernet_h --root D:/SoccerNet/calibration-2023      # -> datasets/soccer_h.jsonl

Our own hand labels of `soccer` / `soccer2` (datasets/soccer_labels) join the index as split "fresh", never trained on.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from sportcal.core.camera import canonical_mirror, is_plausible_view
from sportcal.core.geometry import line_through, solve_points_lines
from sportcal.paths import DATASETS
from sportcal.sports.soccer import field as F

OUT = DATASETS / "soccer_h.jsonl"
HX, HY = F.HALF_LENGTH, F.HALF_WIDTH
PX, PY, GX, GY = F.PENALTY_AREA_X, F.PENALTY_AREA_Y, F.GOAL_AREA_X, F.GOAL_AREA_Y
BOX = (-HX, HX, -HY, HY)
# SoccerNet class -> the world segment it lies on, in OUR frame ("top" is the far side, +Y here)
LINES = {"Side line top": ((-HX, HY), (HX, HY)), "Side line bottom": ((-HX, -HY), (HX, -HY)),
         "Side line left": ((-HX, -HY), (-HX, HY)), "Side line right": ((HX, -HY), (HX, HY)),
         "Middle line": ((0, -HY), (0, HY))}
for s, side in ((-1, "left"), (1, "right")):
    for name, x, y in (("Big rect.", PX, PY), ("Small rect.", GX, GY)):
        LINES["{} {} top".format(name, side)] = ((s * HX, y), (s * x, y))
        LINES["{} {} bottom".format(name, side)] = ((s * HX, -y), (s * x, -y))
        LINES["{} {} main".format(name, side)] = ((s * x, -y), (s * x, y))
CIRCLES = {"Circle central": (0.0, 0.0), "Circle left": (-F.PENALTY_SPOT_X, 0.0), "Circle right": (F.PENALTY_SPOT_X, 0.0)}


def _dist_to_segment_polyline(pts, poly):
    """Distance of each point (n, 2) to a polyline (m, 2)."""
    a, b = poly[:-1][None], poly[1:][None]
    ab = b - a
    t = np.clip(((pts[:, None] - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(pts[:, None] - (a + t[..., None] * ab), axis=-1).min(1)


def fit(ann, w, h, min_lines=4):
    """(H world -> px, median px distance of the annotated straight-line points to their lines as H draws them, number
    of lines, the same median for the circle points - nan without them) or None. The circles are not fitted, so they
    also catch what the lines cannot show (a wide lens bends them away from any H)."""
    wl, il, used = [], [], []
    for name, (a, b) in LINES.items():
        p = np.array([[q["x"] * w, q["y"] * h] for q in ann.get(name, [])], float)
        if len(p) < 2 or np.ptp(p, 0).max() < 5:
            continue
        vx, vy, x0, y0 = cv2.fitLine(p.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        wl.append(line_through(a, b))
        il.append(line_through((x0, y0), (x0 + vx, y0 + vy)))
        used.append((name, p))
    if len(wl) < min_lines:
        return None
    # where two annotated markings meet on the pitch (a box corner, a line end), the image lines meet too: those points
    # condition the DLT far better than lines alone (two families of parallel lines are close to degenerate)
    wp, ip = [], []
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            X, x = np.cross(wl[i], wl[j]), np.cross(il[i], il[j])
            if abs(X[2]) < 1e-9 or abs(x[2]) < 1e-9:
                continue
            X, x = X[:2] / X[2], x[:2] / x[2]
            if all(np.linalg.norm(np.subtract(X, a)) + np.linalg.norm(np.subtract(X, b)) < np.linalg.norm(np.subtract(a, b)) + 0.1
                   for a, b in (LINES[used[i][0]], LINES[used[j][0]])):
                wp.append(X)
                ip.append(x)
    H = solve_points_lines(wp, ip, wl, il, 2 * HX, w, 3.0)
    if H is None:
        return None
    H = H / H[2, 2]
    d, dc = [], []
    for name, p in used:
        seg = np.linspace(LINES[name][0], LINES[name][1], 200)
        q = np.c_[seg, np.ones(len(seg))] @ H.T
        d.append(_dist_to_segment_polyline(p, q[:, :2] / q[:, 2:]))
    for name, c in CIRCLES.items():
        p = np.array([[q["x"] * w, q["y"] * h] for q in ann.get(name, [])], float)
        if len(p):
            t = np.linspace(0, 2 * np.pi, 360)
            q = np.c_[c[0] + F.CIRCLE_RADIUS * np.cos(t), c[1] + F.CIRCLE_RADIUS * np.sin(t), np.ones(360)] @ H.T
            dc.append(_dist_to_segment_polyline(p, q[:, :2] / q[:, 2:]))
    return H, float(np.median(np.concatenate(d))), len(wl), float(np.median(np.concatenate(dc))) if dc else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="folder with train/ valid/ test/ (images + one json per image)")
    ap.add_argument("--max-px", type=float, default=3.0, help="median straight-line residual at 1920 px to keep a frame")
    ap.add_argument("--max-circle-px", type=float, default=10.0, help="the same for the circle points, when there are any")
    args = ap.parse_args()
    rows, dropped = [], Counter()
    for split_dir, split in (("train", "train"), ("valid", "dev"), ("test", "test")):
        for js in sorted((Path(args.root) / split_dir).glob("*.json")):
            img = js.with_suffix(".jpg")
            im = cv2.imread(str(img)) if img.exists() else None
            if im is None:
                dropped["no image"] += 1
                continue
            h, w = im.shape[:2]
            r = fit(json.load(open(js, encoding="utf-8")), w, h)
            if r is None:
                dropped["fewer than 4 straight markings"] += 1
                continue
            H, resid, n, resid_c = r
            if resid * 1920.0 / w > args.max_px:
                dropped["line residual > {} px".format(args.max_px)] += 1
                continue
            if resid_c * 1920.0 / w > args.max_circle_px:
                dropped["circle residual > {} px".format(args.max_circle_px)] += 1
                continue
            if not is_plausible_view(H, w, h, BOX):
                dropped["no camera gives this H"] += 1
                continue
            H, flips = canonical_mirror(H, (0.0, 0.0), y_down=False)
            rows.append({"id": "soccernet/{}/{}".format(split_dir, js.stem), "image": str(img), "template": "soccer-fifa",
                         "source": "soccernet", "split": split, "w": w, "h": h, "H": H.tolist(), "flips": flips,
                         "fit": {"lines": n, "resid_px": round(resid * 1920.0 / w, 2),
                                 "circle_resid_px": round(resid_c * 1920.0 / w, 2) if resid_c == resid_c else None}})
    # our own hand labels (labeler.py) of broadcasts SoccerNet never saw: the "fresh" split, never trained on
    hand = DATASETS / "soccer_labels"
    if (hand / "clicks.jsonl").exists():
        for d in {d["id"]: d for d in map(json.loads, open(hand / "clicks.jsonl", encoding="utf-8"))}.values():
            img = hand / "images" / (d["id"] + ".jpg")
            im = cv2.imread(str(img))
            if im is None:
                continue
            h, w = im.shape[:2]
            H, flips = canonical_mirror(np.asarray(d["H"], float), (0.0, 0.0), y_down=False)
            rows.append({"id": "soccer_labels/" + d["id"], "image": str(img), "template": "soccer-fifa", "source": "hand",
                         "split": "fresh", "w": w, "h": h, "H": H.tolist(), "flips": flips, "fit": None})
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print("{} frames -> {}".format(len(rows), OUT))
    for s, n in sorted(Counter(r["split"] for r in rows).items()):
        print("  {:<6} {}".format(s, n))
    for why, n in dropped.most_common():
        print("  left out {:>6}  {}".format(n, why))
    fitted = [r["fit"]["resid_px"] for r in rows if r["fit"]]
    if fitted:
        print("  line residual p50 {:.2f} p90 {:.2f} px at 1920".format(*np.percentile(fitted, [50, 90])))


if __name__ == "__main__":
    main()
