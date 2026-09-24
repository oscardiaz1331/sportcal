"""Camera motion of the soccer broadcast from the stands and advertising, and how well it carries the field homography.

The tracked points come from the NON-grass region only (stands and boards), so the estimate never looks at the field
lines; that makes the field lines an independent check: warp the lines of frame 0 by the estimated motion and see how close
they land to the lines of frame 1. Results: docs/experiments/soccer.md. The tracking itself is sport-agnostic and lives in
`core.motion` (`track_pair`, `track_video`); this module adds the grass surface, the line refinement and the experiments.

    python -m sportcal.lab.soccer.camera_motion --video soccer --frame 900 --gaps 1,5,10,25
    python -m sportcal.lab.soccer.camera_motion --video soccer --frame 900 --chain-step 10 --chain-n 8
"""
import argparse
import time

import cv2
import numpy as np

from sportcal.core import motion as MOT
from sportcal.core.motion import REGIONS, W_WORK, to_native_motion
from sportcal.lab.soccer import evaluation as EV
from sportcal.lab.soccer import field_solver as FS
from sportcal.lab.soccer import labeler as LB


def track_pair(frame0, frame1, **kw):
    """`core.motion.track_pair` on a grass field."""
    return MOT.track_pair(frame0, frame1, "grass", **kw)


def line_alignment(lines0, lines1, M, max_points=4000, seed=0):
    """How well motion M carries the field-line mask of frame 0 onto the one of frame 1.

    Returns {"median": px, "within3": fraction of warped points within 3 px of a line pixel of frame 1} for M and for the
    identity (no motion), so the improvement over 'the camera did not move' is visible."""
    ys, xs = np.nonzero(lines0)
    if len(xs) == 0 or not lines1.any():
        return {"median": float("nan"), "within3": float("nan"), "median_identity": float("nan"), "within3_identity": float("nan")}
    if len(xs) > max_points:
        sel = np.random.default_rng(seed).choice(len(xs), max_points, replace=False)
        xs, ys = xs[sel], ys[sel]
    pts = np.stack([xs, ys], 1).astype(float)
    dist = cv2.distanceTransform((lines1 == 0).astype(np.uint8), cv2.DIST_L2, 3)
    h, w = lines1.shape

    def measure(P):
        ok = (P[:, 0] >= 0) & (P[:, 0] < w - 1) & (P[:, 1] >= 0) & (P[:, 1] < h - 1)
        d = dist[np.round(P[ok, 1]).astype(int), np.round(P[ok, 0]).astype(int)]
        return (float(np.median(d)), float((d <= 3).mean())) if len(d) else (float("nan"), float("nan"))

    m, f3 = measure(MOT.apply_motion(M, pts))
    mi, f3i = measure(pts)
    return {"median": m, "within3": f3, "median_identity": mi, "within3_identity": f3i}


def gap_experiment(video, frame, gaps, **kw):
    """One row per gap: tracking statistics and field-line alignment with and without the estimated motion."""
    f0 = LB.lee_frame(video, frame)
    lines0 = EV.etapas_mascara(f0)["lineas"]
    rows = []
    for g in gaps:
        f1 = LB.lee_frame(video, frame + g)
        if f1 is None:
            continue
        t0 = time.time()
        r = track_pair(f0, f1, **kw)
        dt = time.time() - t0
        row = {"gap": g, "selected": r["n_selected"], "tracked": r["n_tracked"], "inliers": r["n_inliers"],
               "residual": r["median_residual"], "seconds": dt}
        if r["M"] is not None:
            row.update(line_alignment(lines0, EV.etapas_mascara(f1)["lineas"], r["M"]))
            row["shift"] = MOT.motion_summary(r["M"], *r["work0"].shape[1::-1])
        rows.append(row)
    return rows


def chain_experiment(video, frame, step, n, min_solver_score=0.6, check_every=1, refine=None, **kw):
    """Carry the field homography solved at `frame` through `n` steps of `step` frames using only the background motion.

    At every checkpoint the propagated H is compared with the solver's own solution for that frame (no reference is used
    that the background tracker could have influenced): the score of both on that frame's line mask and the median
    distance between the two projected templates. The solver only runs every `check_every` steps (it is the expensive part;
    the motion is chained at every step regardless). Returns (rows, H0) or (None, None) if the start frame does not solve."""
    f0 = LB.lee_frame(video, frame)
    mask0 = EV.etapas_mascara(f0)["lineas"]
    start = FS.FieldSolver(mask0).search_lines()
    if not start or start[0]["score"] < min_solver_score:
        return None, None
    H = start[0]["H"]
    rows, prev, w_h = [], f0, mask0.shape[::-1]
    for k in range(1, n + 1):
        cur = LB.lee_frame(video, frame + k * step)
        if cur is None:
            break
        r = track_pair(prev, cur, **kw)
        if r["M"] is None:
            rows.append({"frame": frame + k * step, "lost": True})
            break
        H = MOT.propagate_homography(H, r["M"])
        prev = cur
        if refine is None and k % check_every:
            continue
        solver = FS.FieldSolver(EV.etapas_mascara(cur)["lineas"])
        if refine is not None:                       # pull the propagated H back onto the painted lines at every step
            H, _ = solver.refine(H, taus=refine)
        if k % check_every:
            continue
        solved = solver.search_lines()
        row = {"frame": frame + k * step, "inliers": r["n_inliers"], "propagated_score": float(solver.score(H[None], fine=True)[0])}
        if solved:
            row["solved_score"] = solved[0]["score"]
            row["h_distance"] = FS.reprojection_error(H, solved[0]["H"], *w_h) * 1920.0 / w_h[0]
        rows.append(row)
    return rows, start[0]["H"]


def refine_on_lines(taus):
    """`refine` for `core.motion.track_video`: pull H onto the painted lines of the frame (`FieldSolver.refine`, taus are
    fractions of the width)."""
    def refine(frame, H):
        S = np.diag([W_WORK / float(frame.shape[1]), W_WORK / float(frame.shape[1]), 1.0])
        Hw, score = FS.FieldSolver(EV.etapas_mascara(frame)["lineas"]).refine(S @ H, taus=tuple(taus))
        return np.linalg.inv(S) @ Hw, score
    return refine


def track_video(get_frame, frame0, H0, refine_taus=None, **kw):
    """`core.motion.track_video` on a grass field, refined on the painted lines when `refine_taus` is given."""
    return MOT.track_video(get_frame, frame0, H0, "grass", refine=None if refine_taus is None else refine_on_lines(refine_taus),
                           **kw)


def render_step(native, step, width=960):
    """RGB image (width px wide) of one `track_video` row: the frame with the carried template (yellow) and the background
    points that supported this step (green, drawn in the previous frame's positions). Cheap enough to call per step."""
    from sportcal.lab.soccer import labeler as LB
    h0, w0 = native.shape[:2]
    vis = native.copy()
    if step.get("H") is not None:
        LB.dibuja_plantilla(vis, step["H"], (0, 255, 255), max(2, w0 // 640))
    vis = cv2.resize(vis, (width, int(round(h0 * width / w0))), interpolation=cv2.INTER_AREA)
    sc = width / float(W_WORK)
    for x, y in step["points"][::3]:
        cv2.circle(vis, (int(x * sc), int(y * sc)), 2, (0, 255, 0), -1)
    txt = "frame {}   inliers {}/{}".format(step["frame"], step["n_inliers"], step["n_selected"])
    if step["lost"]:
        txt = "frame {}   BACKGROUND LOST (shot cut or close-up)".format(step["frame"])
    cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def panels(res, native0=None, native1=None, H=None, amplify=5.0):
    """[(title, rgb image, text)] to look at one `track_pair` result.

    1. Tracked points on frame 0 (grass dimmed: no feature comes from there): green = kept, orange = tracked but rejected
       by RANSAC, red = lost or failed the forward-backward check; the displacement is drawn `amplify` times longer.
    2. Absolute difference of the two frames, without and with the estimated motion: what is static in the world cancels.
    3. If `H` (native world -> frame-0 pixels) and both native frames are given: the template on frame 0 and carried to
       frame 1 with the background motion only."""
    from sportcal.lab.soccer import labeler as LB
    w0, w1, region = res["work0"], res["work1"], res["region"]
    rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    out = []

    v = w0.copy()
    v[region] = (v[region] * 0.45).astype(np.uint8)
    p0, p1 = res["p0"], res["p1"]
    for k in range(len(p0)):
        a = (int(p0[k, 0]), int(p0[k, 1]))
        if res["inliers"][k] or res["tracked"][k]:
            b = (int(p0[k, 0] + amplify * (p1[k, 0] - p0[k, 0])), int(p0[k, 1] + amplify * (p1[k, 1] - p0[k, 1])))
            col = (0, 255, 0) if res["inliers"][k] else (0, 140, 255)
            cv2.line(v, a, b, col, 1, cv2.LINE_AA)
            cv2.circle(v, a, 2, col, -1)
        else:
            cv2.circle(v, a, 2, (0, 0, 255), -1)
    n_sel, n_tr, n_in = res["n_selected"], res["n_tracked"], res["n_inliers"]
    text = "{} points selected in the non-grass region, {} survive the forward-backward check, {} are RANSAC inliers ({:.0f}%).".format(
        n_sel, n_tr, n_in, 100.0 * n_in / max(1, n_sel))
    if res["M"] is not None:
        s = MOT.motion_summary(res["M"], *w0.shape[1::-1])
        text += " Motion at the image centre: dx {:.1f} px, dy {:.1f} px, zoom x{:.4f}, rotation {:.2f} deg; median inlier residual {:.2f} px.".format(
            s["dx"], s["dy"], s["zoom"], s["rotation_deg"], res["median_residual"])
    else:
        text += " Not enough consistent tracks to fit a motion."
    out.append(("Tracked points (green = kept, orange = RANSAC outlier, red = lost); displacement x{:g}".format(amplify), rgb(v), text))

    g0, g1 = cv2.cvtColor(w0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY)
    raw = cv2.absdiff(g0, g1)
    if res["M"] is not None:
        comp = cv2.absdiff(cv2.warpPerspective(g0, res["M"], g0.shape[1::-1], borderMode=cv2.BORDER_REPLICATE), g1)
        bg = res["mask"] > 0
        text2 = "Mean absolute difference over the background: {:.1f} without motion, {:.1f} with it. What stays lit after the compensation is what moves by itself (players) or is glued to the screen (score bug, logos).".format(
            float(raw[bg].mean()), float(comp[bg].mean()))
        out.append(("Difference between the two frames, without motion compensation", rgb(cv2.cvtColor(np.clip(raw.astype(int) * 3, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)), "The whole scene lights up when the camera moves."))
        out.append(("Difference after compensating with the background motion", rgb(cv2.cvtColor(np.clip(comp.astype(int) * 3, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)), text2))
        if H is not None and native0 is not None and native1 is not None:
            Mn = to_native_motion(res["M"], native0.shape[1])
            H1 = MOT.propagate_homography(H, Mn)
            a, b = native0.copy(), native1.copy()
            LB.dibuja_plantilla(a, H, (0, 255, 255), max(2, a.shape[1] // 640))
            LB.dibuja_plantilla(b, H1, (255, 200, 0), max(2, b.shape[1] // 640))
            out.append(("Field template on frame 0 (yellow)", rgb(a), "The homography of the current frame (your clicks, or the chosen suggestion)."))
            out.append(("Same template carried to frame 1 with the background motion only (blue)", rgb(b),
                        "No field line was used to move it: if it lands on the painted lines, the stands and boards carry the camera motion. Errors add up when chaining several steps."))
    return out


def _fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else ("{:.%df}" % nd).format(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="soccer")
    ap.add_argument("--frame", type=int, default=900)
    ap.add_argument("--gaps", default="")
    ap.add_argument("--chain-step", type=int, default=0)
    ap.add_argument("--chain-n", type=int, default=8)
    ap.add_argument("--check-every", type=int, default=1, help="run the solver for comparison every N chain steps")
    ap.add_argument("--model", default="homography", choices=MOT.MODELS)
    ap.add_argument("--max-fb", type=float, default=1.0)
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--refine", default="", help="comma-separated tau fractions: refine the chained H on the line mask at every step")
    ap.add_argument("--region", default="background", choices=REGIONS, help="where the KLT features are taken from")
    a = ap.parse_args()
    kw = dict(model=a.model, max_fb_error=a.max_fb, ransac_thresh=a.ransac)
    if a.region != "background":
        kw["region_kind"] = a.region
    if a.gaps:
        rows = gap_experiment(a.video, a.frame, [int(x) for x in a.gaps.split(",")], **kw)
        print("gap  selected tracked inliers resid  line-dist(med px, KLT / no motion)  within 3 px (KLT / no motion)   zoom   dx    s")
        for r in rows:
            s = r.get("shift", {})
            print("{:>3}  {:>8} {:>7} {:>7} {:>5}  {:>8} / {:<8}                {:>5} / {:<5}             {:>6} {:>5} {:>5}".format(
                r["gap"], r["selected"], r["tracked"], r["inliers"], _fmt(r["residual"]), _fmt(r.get("median")), _fmt(r.get("median_identity")),
                _fmt(r.get("within3")), _fmt(r.get("within3_identity")), _fmt(s.get("zoom"), 3), _fmt(s.get("dx"), 1), _fmt(r["seconds"], 2)))
    if a.chain_step:
        refine = tuple(float(x) for x in a.refine.split(",")) if a.refine else None
        rows, _ = chain_experiment(a.video, a.frame, a.chain_step, a.chain_n, check_every=a.check_every, refine=refine, **kw)
        if rows is None:
            print("the start frame does not solve well enough to carry its homography")
        else:
            print("frame  inliers  propagated score  re-solved score  distance between templates (px at 1920)")
            for r in rows:
                print("{:>5}  {}".format(r["frame"], "LOST" if r.get("lost") else "{:>7}  {:>16}  {:>15}  {:>10}".format(
                    r["inliers"], _fmt(r["propagated_score"]), _fmt(r.get("solved_score")), _fmt(r.get("h_distance"), 1))))


if __name__ == "__main__":
    main()
