"""Per-frame H index for soccer from the SoccerNet camera-calibration annotations (calibration-2023), the soccer
counterpart of `lab/hockey/build_h_index`: one H per frame, fitted to the annotated pitch markings, so every target of the
keypoint + line model can be rendered from H and the template. Results: docs/experiments/soccer.md section 19.

Each annotation gives, per marking class, points on that marking in normalised image coordinates. Straight markings
become image lines (least squares through their points) matched to the template's world lines; with the points where
two of them meet on the pitch, one DLT (`core.geometry.solve_points_lines`) gives H; the circle points only check it.
Frames with fewer than 4 straight markings (most centre-circle views) are fitted to their lines AND circle points
together (`fit_circle`, soccer.md section 22) and pass the same gates; frames whose DLT misses the gates get their H
refined on the line points themselves (`refine`, section 23).
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
from scipy.optimize import least_squares

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
    return (H, *_check(H, used, ann, w, h, len(wl)))


def _check(H, used, ann, w, h, n):
    """(median px distance of the straight-line points to their segments as H draws them, n, the same for the circle
    points - nan without them)."""
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
    return float(np.median(np.concatenate(d))), n, float(np.median(np.concatenate(dc))) if dc else float("nan")


def _line_residuals(Hi, wl, Pn):
    """Signed distances of each line's points `Pn` to its world line `wl` drawn by the inverse homography `Hi`."""
    out = []
    for l, p in zip(wl, Pn):
        li = Hi.T @ l
        out.append((p @ li[:2] + li[2]) / np.hypot(li[0], li[1]))
    return out


def refine(ann, w, h, H):
    """H refined by least squares (soft L1) on the straight-line points themselves - each point's distance to its
    projected world line - in the format of `fit`, or None. For frames whose `fit` misses the gates: the DLT fits lines
    through the points and their crossings, not the points (soccer.md section 23). The circles stay out of the fit, so
    their gate still checks it.
    ponytail: no lens distortion - a k1 fitted with H recovers a few more frames, but the pinhole H stored for them
    would miss their points by more than the gate; store k1 once the targets can be rendered with it."""
    used = _lines(ann, w, h)
    if len(used) < 4:
        return None
    wl = [line_through(*LINES[n]) for n, _ in used]
    N = np.diag([1.0 / w, 1.0 / w, 1.0])             # optimise in image coordinates / w (conditioning)
    Pn = [(np.c_[p, np.ones(len(p))] @ N.T)[:, :2] for _, p in used]
    H0 = N @ H
    try:
        sol = least_squares(lambda h8: np.concatenate(_line_residuals(
            np.linalg.inv(np.append(h8, 1.0).reshape(3, 3)), wl, Pn)), (H0 / H0[2, 2]).ravel()[:8],
            loss="soft_l1", f_scale=2.0 / w)
    except (np.linalg.LinAlgError, ValueError):
        return None
    H = np.linalg.inv(N) @ np.append(sol.x, 1.0).reshape(3, 3)
    H = H / H[2, 2]
    return (H, *_check(H, used, ann, w, h, len(used)))


def _lines(ann, w, h):
    """[(name, (n, 2) px)] of the straight markings with at least 2 points that span 5 px."""
    out = []
    for name in LINES:
        p = np.array([[q["x"] * w, q["y"] * h] for q in ann.get(name, [])], float)
        if len(p) >= 2 and np.ptp(p, 0).max() >= 5:
            out.append((name, p))
    return out


def fit_circle(ann, w, h, min_lines=2, n_start=8):
    """For frames with too few straight markings for `fit` (a centre-circle view has 1-3): H fitted to the straight-line
    points and the circle points together - each circle point's distance to the projected circle, each line point's to
    its projected line - in the format of `fit`, or None. Start: the ellipse through the best-annotated circle
    (`cv2.fitEllipse`) read as an affine image of the world circle, at `n_start` rotations x 2 mirrorings; each start is
    refined briefly, and the lowest cost is refined to the end (least squares, soft L1). Refused when every line passes through the circle's
    centre: a circle and its diameters leave H one degree of freedom, whatever the fit returns."""
    used = _lines(ann, w, h)
    circ = [(CIRCLES[k], np.array([[q["x"] * w, q["y"] * h] for q in ann[k]], float)) for k in CIRCLES
            if len(ann.get(k, [])) >= 3]
    if len(used) < min_lines or not circ:
        return None
    (C0, P0) = max(circ, key=lambda c: len(c[1]))
    if len(P0) < 5:
        return None
    R = F.CIRCLE_RADIUS
    wl = [line_through(*LINES[n]) for n, _ in used]
    if all(abs(l @ (*C0, 1.0)) / np.hypot(*l[:2]) < 0.5 for l in wl):
        return None
    N = np.diag([1.0 / w, 1.0 / w, 1.0])            # optimise in image coordinates / w (conditioning)
    Pn = [(np.c_[p, np.ones(len(p))] @ N.T)[:, :2] for _, p in used]
    Cn = [(c, (np.c_[p, np.ones(len(p))] @ N.T)[:, :2]) for c, p in circ]

    def residuals(h8):
        Hn = np.append(h8, 1.0).reshape(3, 3)
        Hi = np.linalg.inv(Hn)
        r = _line_residuals(Hi, wl, Pn)
        for (cx, cy), p in Cn:
            q = np.c_[p, np.ones(len(p))] @ Hi.T
            q = q[:, :2] / q[:, 2:] - (cx, cy)
            q = (cx, cy) + R * q / np.linalg.norm(q, axis=1, keepdims=True)    # nearest point of the world circle
            b = np.c_[q, np.ones(len(q))] @ Hn.T
            r.append((b[:, :2] / b[:, 2:] - p).ravel())
        return np.concatenate(r)

    (cx, cy), (ax, bx), ang = cv2.fitEllipse(P0.astype(np.float32))
    t = np.radians(ang)
    E = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ np.diag([ax / 2, bx / 2])
    best = None
    for k in range(n_start):
        phi = 2 * np.pi * k / n_start
        Rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        for S in (np.eye(2), np.diag([1.0, -1.0])):
            A = E @ Rot @ S / R
            H0 = np.eye(3)
            H0[:2, :2], H0[:2, 2] = A, np.array([cx, cy]) - A @ np.asarray(C0)
            H0 = N @ H0
            try:     # a short plain least-squares run per start, the robust refinement only for the best one
                sol = least_squares(residuals, (H0 / H0[2, 2]).ravel()[:8], max_nfev=30)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
                best = sol
    if best is None:
        return None
    try:
        best = least_squares(residuals, best.x, loss="soft_l1", f_scale=2.0 / w)
    except (np.linalg.LinAlgError, ValueError):
        return None
    H = np.linalg.inv(N) @ np.append(best.x, 1.0).reshape(3, 3)
    H = H / H[2, 2]
    return (H, *_check(H, used, ann, w, h, len(used)))


def refused(r, w, h, max_px, max_circle_px):
    """Why a `fit` result is left out of the index, or None to keep it."""
    H, resid, _, resid_c = r
    if resid * 1920.0 / w > max_px:
        return "line residual > {} px".format(max_px)
    if resid_c * 1920.0 / w > max_circle_px:
        return "circle residual > {} px".format(max_circle_px)
    if not is_plausible_view(H, w, h, BOX):
        return "no camera gives this H"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="folder with train/ valid/ test/ (images + one json per image)")
    ap.add_argument("--max-px", type=float, default=3.0, help="median straight-line residual at 1920 px to keep a frame")
    ap.add_argument("--max-circle-px", type=float, default=10.0, help="the same for the circle points, when there are any")
    ap.add_argument("--lines-only", action="store_true", help="skip `fit_circle` and `refine` (the index of section 19)")
    ap.add_argument("--out", default=str(OUT), help="index to write (another file while a training reads this one)")
    args = ap.parse_args()
    if not all((Path(args.root) / d).is_dir() for d in ("train", "valid", "test")):
        # a wrong --root would otherwise rewrite the index with the hand labels only
        raise SystemExit("{} has no train/ valid/ test/ folders: nothing written".format(Path(args.root).resolve()))
    rows, dropped = [], Counter()
    for split_dir, split in (("train", "train"), ("valid", "dev"), ("test", "test")):
        for js in sorted((Path(args.root) / split_dir).glob("*.json")):
            img = js.with_suffix(".jpg")
            im = cv2.imread(str(img)) if img.exists() else None
            if im is None:
                dropped["no image"] += 1
                continue
            h, w = im.shape[:2]
            ann = json.load(open(js, encoding="utf-8"))
            gates = (w, h, args.max_px, args.max_circle_px)
            r, method = fit(ann, w, h), "lines"
            if r is not None and refused(r, *gates) and not args.lines_only:
                r2 = refine(ann, w, h, r[0])
                if r2 is not None and not refused(r2, *gates):
                    r, method = r2, "lines-ls"
            if r is None and not args.lines_only:
                r, method = fit_circle(ann, w, h), "lines+circle"
            if r is None:
                dropped["fewer than 4 straight markings" + ("" if args.lines_only else ", no usable circle")] += 1
                continue
            why = refused(r, *gates)
            if why:
                dropped[why] += 1
                continue
            H, resid, n, resid_c = r
            H, flips = canonical_mirror(H, (0.0, 0.0), y_down=False)
            rows.append({"id": "soccernet/{}/{}".format(split_dir, js.stem), "image": str(img), "template": "soccer-fifa",
                         "source": "soccernet", "split": split, "w": w, "h": h, "H": H.tolist(), "flips": flips,
                         "fit": {"method": method, "lines": n, "resid_px": round(resid * 1920.0 / w, 2),
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
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print("{} frames -> {}".format(len(rows), args.out))
    for s, n in sorted(Counter(r["split"] for r in rows).items()):
        print("  {:<6} {}".format(s, n))
    for m in ("lines+circle", "lines-ls"):
        print("  fitted {}: {}".format(m, sum(bool(r["fit"]) and r["fit"]["method"] == m for r in rows)))
    for why, n in dropped.most_common():
        print("  left out {:>6}  {}".format(n, why))
    fitted = [r["fit"]["resid_px"] for r in rows if r["fit"]]
    if fitted:
        print("  line residual p50 {:.2f} p90 {:.2f} px at 1920".format(*np.percentile(fitted, [50, 90])))


if __name__ == "__main__":
    main()
