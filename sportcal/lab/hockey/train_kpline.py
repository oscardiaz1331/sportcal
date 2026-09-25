"""Per-sport keypoint + line model for the NHL rink ("model A" of the multi-sport plan, PnLCalib-style): one U-Net
predicts a heatmap per template keypoint and one per visible end of every straight line; the template and a DLT turn
them into H. Trained in two phases from the per-frame H index (`build_h_index`): pretrain on every train label (NHL +
IIHF, the stored NHL ones are 11-22 px off), then fine-tune on the hand labels only. Numbers: hockey.md section 14.
`--keypoints derived` adds the points on the circles and board corners (`rink.derived_keypoints`).

    python -m sportcal.lab.hockey.train_kpline --phase pretrain
    python -m sportcal.lab.hockey.train_kpline --phase finetune --init runs/kpline/pretrain/best_h.pt
    python -m sportcal.lab.hockey.train_kpline --eval runs/kpline/finetune/best_h.pt --split test [--gate]
    python -m sportcal.lab.hockey.train_kpline --eval runs/hockeyrink/yolo26m-18/weights/best_homography.pt --yolo
    python -m sportcal.lab.hockey.train_kpline --phase pretrain --keypoints derived     # + circle points, section 14c
    python -m sportcal.lab.hockey.train_kpline --sport soccer --phase pretrain          # FIFA pitch, soccer.md section 19

Targets are rendered from H every time, never stored: a flip or a camera turn (`core.camera.ptz_warp`) changes H,
`canonicalize` renames the points, and no remapping table exists to go wrong. Checkpoints are chosen by the median
homography error on `dev`, not by the loss (hockey.md section 2: a good pose loss can mean useless homographies). The
dev labels are the stored ones (~11-16 px off), so differences of a few px between checkpoints are noise. The GPU has
8 GB: nothing else may run on it during a training (`nvidia-smi`).
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sportcal.core.camera import canonical_mirror, is_plausible_view, ptz_warp
from sportcal.core.geometry import geom_error, line_through, solve_points_lines
from sportcal.core.labels import heatmap_peaks, render_heatmaps
from sportcal.lab.hockey.build_h_index import OUT as HOCKEY_INDEX
from sportcal.lab.hockey.build_h_index import PARAMS
from sportcal.lab.hockey.train_lines_seg import MEAN, STD, UNetResNet34
from sportcal.paths import DATASETS, ROOT, RUNS
from sportcal.sports.hockey import rink
from sportcal.sports.soccer import field as FIFA

SIZE = (960, 544)                            # network input: a 16:9 frame at 960 px, padded at the bottom to /32
OUT_SIZE = (SIZE[0] // 2, SIZE[1] // 2)      # heatmaps at half the input (HalfResUNet)
SIGMA = 1.5                                  # heatmap Gaussian, px at 480 (6 px at 1920)
# per template: world extent (x0, x1, y0, y1) and the straight painted lines as world segments (circles are not lines)
BOX = {**{k: (0.0, p["length"], 0.0, p["width"]) for k, p in PARAMS.items()},
       "soccer-fifa": (-FIFA.HALF_LENGTH, FIFA.HALF_LENGTH, -FIFA.HALF_WIDTH, FIFA.HALF_WIDTH)}
SEGMENTS = {**{k: [(a, b) for _, a, b in rink.straight_lines(p)] for k, p in PARAMS.items()},
            "soccer-fifa": [(tuple(a), tuple(b)) for name, pl in FIFA.polylines().items() if "circle" not in name
                            and "arc" not in name for a, b in zip(pl[:-1], pl[1:])]}
# sport -> (the template its model is trained and scored on, its per-frame H index, its runs folder)
SPORTS = {"hockey": ("hockey-nhl", HOCKEY_INDEX, "kpline"), "soccer": ("soccer-fifa", DATASETS / "soccer_h.jsonl", "kpline-soccer")}
# the 56 keypoints of the rink template, or those plus the points derived from the circles and board corners
KEYPOINT_SETS = {"base": rink.build_template,
                 "derived": lambda p: np.vstack([rink.build_template(p), rink.derived_keypoints(p)])}
GRID = {k: np.stack(np.meshgrid(np.arange(x0, x1 + 1e-9, 1.0), np.arange(y0, y1 + 1e-9, 1.0)), -1).reshape(-1, 2)
        for k, (x0, x1, y0, y1) in BOX.items()}
# hand labels of training videos; test_leaky gives its frames to fine-tuning (the clean test set is `test`)
FINETUNE_SPLITS = ("train", "test_leaky")


def keypoints(kp_set):
    """{template name: (K, 2) world keypoints} of one keypoint set. Passed around explicitly rather than set as a module
    global: DataLoader workers on Windows re-import this module and would see the default."""
    # ponytail: one soccer set (the 31 named points of sports/soccer/field.py) whatever kp_set says
    return {**{k: KEYPOINT_SETS[kp_set](p) for k, p in PARAMS.items()}, "soccer-fifa": FIFA.KEYPOINT_COORDS}


def n_channels(kp, template="hockey-nhl"):
    return len(kp[template]) + 2 * len(SEGMENTS[template])


def canonicalize(H, template):
    """H renamed to the image naming rule (`core.camera.canonical_mirror`): the hockey index's, and for soccer the one
    `lab/soccer/soccernet_h` stores (+Y, the far touchline, up in a side view)."""
    x0, x1, y0, y1 = BOX[template]
    return canonical_mirror(H, ((x0 + x1) / 2, (y0 + y1) / 2), y_down=template != "soccer-fifa")[0]


class HalfResUNet(UNetResNet34):
    """UNetResNet34 whose heatmaps stay at half the input resolution: 74+ channels at full resolution run out of memory
    (6 GB at batch 2 on the CPU, before the backward pass).
    ponytail: reading perfect half-resolution heatmaps back with `heatmap_peaks` already costs ~1 px at 1920; a
    quadratic peak fit or an offset head is the upgrade if that ever dominates."""

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.e1(self.pool(s0))
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        d = self.d3(self._up(self.d4(self._up(s4, s3)), s2))
        return self.final(self.d1(self._up(self.d2(self._up(d, s1)), s0)))


def select(rows, phase):
    if phase == "pretrain":
        return [r for r in rows if r["split"] == "train"]
    return [r for r in rows if r["source"] == "hand" and r["split"] in FINETUNE_SPLITS and r["template"] == "hockey-nhl"]


class Frames(Dataset):
    """(image tensor, target heatmaps, row index). With `augment`, a random camera turn / zoom and a mirror, applied to
    the real frame and to its H alike (exact, see ptz_warp), plus a brightness / contrast jitter."""

    def __init__(self, rows, augment, kp):
        self.rows, self.augment, self.kp = rows, augment, kp

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(ROOT / r["image"]))
        h0, w0 = img.shape[:2]
        H = np.asarray(r["H"], float)
        if self.augment:
            rng = np.random.default_rng()
            G, _ = ptz_warp(H, w0, h0, pan=rng.uniform(-0.08, 0.08), tilt=rng.uniform(-0.05, 0.05),
                            roll=rng.uniform(-0.03, 0.03), zoom=rng.uniform(0.9, 1.3))
            if rng.random() < 0.5:
                G = np.array([[-1.0, 0, w0 - 1], [0, 1, 0], [0, 0, 1]]) @ G
            img = cv2.warpPerspective(img, G, (w0, h0), flags=cv2.INTER_LINEAR)
            H = canonicalize(G @ H, r["template"])                 # a mirrored frame renames its points
            img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.25), beta=rng.uniform(-25, 25))
        x, s = to_input(img)
        s /= 2                                                     # heatmaps are at half the input
        tgt = render_heatmaps(np.diag([s, s, 1.0]) @ H, self.kp[r["template"]], SEGMENTS[r["template"]], *OUT_SIZE,
                              SIGMA)
        return x, torch.from_numpy(tgt), i


def to_input(img):
    """(1 x 3 x SIZE tensor, scale): the frame resized to SIZE's width and padded at the bottom."""
    s = SIZE[0] / img.shape[1]
    small = cv2.resize(img, (SIZE[0], int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    canvas[:small.shape[0]] = small[:SIZE[1]]
    x = (canvas[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), s


def focal_loss(logits, target):
    """CenterNet's penalty-reduced focal loss: one positive pixel per visible point among ~130k negatives. The targets
    peak off-centre (sub-pixel), so the positive is each channel's highest pixel rather than an exact 1."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    t = target.float()
    pos = (t.eq(t.amax(dim=(-2, -1), keepdim=True)) & (t > 0.5)).float()
    loss = (torch.log(p) * (1 - p) ** 2 * pos).sum() + (torch.log(1 - p) * p ** 2 * (1 - t) ** 4 * (1 - pos)).sum()
    return -loss / pos.sum().clamp(min=1)


def estimate_H(heat, template, w0, kp, thr=0.3):
    """H in original-frame px from one (channels, h, w) sigmoid stack, or None."""
    s = OUT_SIZE[0] / w0
    peaks = heatmap_peaks(heat, thr)
    tpl, segs = kp[template], SEGMENTS[template]
    nkp = len(tpl)
    wp = [tpl[k] for k in range(nkp) if peaks[k] is not None]
    ip = [np.asarray(peaks[k]) / s for k in range(nkp) if peaks[k] is not None]
    wl, il = [], []
    for j, (a, b) in enumerate(segs):
        e0, e1 = peaks[nkp + 2 * j], peaks[nkp + 2 * j + 1]
        if e0 is not None and e1 is not None and np.hypot(e0[0] - e1[0], e0[1] - e1[1]) > 5:
            wl.append(line_through(a, b))
            il.append(line_through(np.asarray(e0) / s, np.asarray(e1) / s))
    return solve_points_lines(wp, ip, wl, il, BOX[template][1] - BOX[template][0], w0, 8.0 * w0 / 1920)


@torch.no_grad()
def evaluate(model, rows, device, kp, batch=4, gate=False):
    """Error px at 1920 against the label H, one per row (inf when no H comes out, or with `gate` when no real camera
    gives it: `core.camera.is_plausible_view`)."""
    model.eval()
    errs = [np.inf] * len(rows)
    for x, _, idx in DataLoader(Frames(rows, augment=False, kp=kp), batch_size=batch, num_workers=2):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            heat = torch.sigmoid(model(x.to(device)).float()).cpu().numpy()
        for hm, i in zip(heat, idx.tolist()):
            r = rows[i]
            H = estimate_H(hm, r["template"], r["w"], kp)
            if H is not None and (not gate or is_plausible_view(H, r["w"], r["h"], BOX[r["template"]])):
                errs[i] = geom_error(H, np.asarray(r["H"], float), GRID[r["template"]], r["w"], r["h"])
    return np.array(errs)


def evaluate_yolo(weights, rows, device, gate=False):
    """The same errors for the 56-keypoint YOLO model, through the product estimator (one frame at a time). Its H is
    canonicalized first: YOLO was trained on labels with mixed mirror conventions (hockey.md section 0b), so the names
    of its points are not a fair part of the comparison."""
    from sportcal.product.hockey import YoloKeypointEstimator
    est = YoloKeypointEstimator(weights, device=0 if device.type == "cuda" else "cpu")
    errs = []
    for r in rows:
        e = est.estimate(cv2.imread(str(ROOT / r["image"])))
        H = None if e is None else canonicalize(e.H, r["template"])
        if H is not None and gate and not is_plausible_view(H, r["w"], r["h"], BOX[r["template"]]):
            H = None
        errs.append(np.inf if H is None else
                    geom_error(H, np.asarray(r["H"], float), GRID[r["template"]], r["w"], r["h"]))
    return np.array(errs)


def summary(errs):
    ok = np.isfinite(errs)
    if not ok.any():
        return "coverage 0%"
    return "coverage {:.0f}%  p50 {:.1f}  p90 {:.1f}  <10 px {:.0f}%  <25 px {:.0f}% (of all frames)".format(
        100 * ok.mean(), np.median(errs[ok]), np.percentile(errs[ok], 90), 100 * (errs < 10).mean(),
        100 * (errs < 25).mean())


def train(args, rows, device):
    tr = select(rows, args.phase)
    template, _, runs = SPORTS[args.sport]
    dev = [r for r in rows if r["split"] == "dev" and r["template"] == template]
    kp = keypoints(args.keypoints)
    out = RUNS / runs / (args.phase + ("" if args.keypoints == "base" else "-" + args.keypoints))
    out.mkdir(parents=True, exist_ok=True)
    print("{}: {} train frames, {} dev frames, {} channels -> {}".format(args.phase, len(tr), len(dev),
                                                                         n_channels(kp, template), out))
    model = HalfResUNet(ncls=n_channels(kp, template)).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    else:   # CenterNet's prior: start every pixel at p = 0.01, or the ~130k negatives swamp the first steps
        torch.nn.init.constant_(model.final[-1].bias, -4.6)
    epochs = args.epochs or (60 if args.phase == "pretrain" else 30)
    lr = args.lr or (3e-4 if args.phase == "pretrain" else 1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loader = DataLoader(Frames(tr, augment=True, kp=kp), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, persistent_workers=args.workers > 0)
    best, saved = np.inf, False
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, tgt, _ in loader:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x.to(device))
            loss = focal_loss(logits, tgt.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item(), n + 1
            if args.max_steps and n >= args.max_steps:
                break
        sched.step()
        line = "epoch {:3d}  loss {:.4f}  {:.0f}s".format(ep, tot / max(n, 1), time.time() - t0)
        torch.save(model.state_dict(), out / "last.pt")
        if (ep + 1) % args.eval_every == 0 or ep == epochs - 1:
            errs = evaluate(model, dev, device, kp)
            p50 = np.median(errs) if np.isfinite(errs).any() else np.inf
            line += "  dev " + summary(errs)
            if p50 < best or not saved:          # a run that never answers on dev still leaves its own file
                best, saved = p50, True
                torch.save(model.state_dict(), out / "best_h.pt")
                line += "  <- best_h.pt"
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sport", choices=tuple(SPORTS), default="hockey")
    ap.add_argument("--phase", choices=("pretrain", "finetune"), default="pretrain")
    ap.add_argument("--init", help="weights to start from (the pretrain best_h.pt for the fine-tune)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="steps per epoch, 0 = all (smoke tests)")
    ap.add_argument("--eval", help="weights to evaluate instead of training")
    ap.add_argument("--split", default="test", help="with --eval: test | test_leaky | dev")
    ap.add_argument("--yolo", action="store_true", help="with --eval: the weights are a YOLO pose model (product path)")
    ap.add_argument("--keypoints", choices=tuple(KEYPOINT_SETS), default="base",
                    help="keypoint set to train (runs go to <phase>-derived); --eval reads it from the weights")
    ap.add_argument("--gate", action="store_true", help="with --eval: refuse the H no real camera gives (section 14d)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    template, index, _ = SPORTS[args.sport]
    rows = [json.loads(line) for line in open(index, encoding="utf-8")]
    if not args.eval:
        train(args, rows, device)
        return
    rows = [r for r in rows if r["split"] == args.split and r["template"] == template]
    if args.yolo:
        errs = evaluate_yolo(args.eval, rows, device, args.gate)
    else:
        state = torch.load(args.eval, map_location=device)
        kp = next(keypoints(k) for k in KEYPOINT_SETS if n_channels(keypoints(k), template) == state["final.1.weight"].shape[0])
        model = HalfResUNet(ncls=n_channels(kp, template)).to(device)
        model.load_state_dict(state)
        errs = evaluate(model, rows, device, kp, gate=args.gate)
    for r, e in zip(rows, errs):
        print("{:<40} {:>8.1f}".format(r["id"], e))
    for v in sorted({r["video"] for r in rows}):
        print("  {:<6} {}".format(v, summary(errs[[r["video"] == v for r in rows]])))
    print("{} ({} frames): {}".format(args.split, len(rows), summary(errs)))


if __name__ == "__main__":
    main()
