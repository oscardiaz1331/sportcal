"""Training and scoring of the keypoint + line model (`models/kpline.py`) for any registered sport. A sport needs its
template in `sports/` (keypoints, straight lines) and its per-frame H index at `datasets/<family>_h.jsonl` (the part
of the sport name before "-": `lab/hockey/build_h_index`, `lab/soccer/soccernet_h`, `lab/tennis/tennis_h`). Two phases:
pretrain on every train label of the index, then (hockey) fine-tune on the hand labels only. Numbers: hockey.md section
14, soccer.md sections 19-20, tennis.md.

    python -m sportcal.lab.common.train_kpline --sport hockey-nhl --phase pretrain
    python -m sportcal.lab.common.train_kpline --sport hockey-nhl --phase finetune --init runs/kpline-hockey-nhl/pretrain/best_h.pt
    python -m sportcal.lab.common.train_kpline --sport soccer-fifa --phase pretrain --keypoints derived
    python -m sportcal.lab.common.train_kpline --sport tennis-itf --phase pretrain
    python -m sportcal.lab.common.train_kpline --sport hockey-nhl --eval runs/kpline/finetune/best_h.pt --split test [--gate]

Runs go to runs/kpline-<sport>/<phase>[-derived][tag]. An index may hold several templates (hockey: NHL and IIHF): all
of its train rows are used (`--templates` keeps some of them), dev and scoring use the rows of `--sport` only.
Targets are rendered from H every time, never stored: a flip or a camera turn (`core.camera.ptz_warp`) changes H,
`Sport.canonicalize` renames the points, and no remapping table exists to go wrong. Checkpoints are chosen by the median
homography error on `dev`, not by the loss (hockey.md section 2: a good pose loss can mean useless homographies). The
GPU has 8 GB: nothing else may run on it during a training (`nvidia-smi`).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sportcal import sports
from sportcal.core.camera import is_plausible_view, ptz_warp
from sportcal.core.geometry import geom_error
from sportcal.core.labels import render_heatmaps
from sportcal.models.kpline import KEYPOINT_SETS, OUT_SIZE, estimate_H, load, n_channels, to_input
from sportcal.models.unet import HalfResUNet
from sportcal.paths import DATASETS, ROOT, RUNS

SIGMA = 1.5                                  # heatmap Gaussian, px at 480 (6 px at 1920)
# hand labels of training videos; test_leaky gives its frames to fine-tuning (the clean test set is `test`)
FINETUNE_SPLITS = ("train", "test_leaky")


def index_of(sport_name):
    return DATASETS / "{}_h.jsonl".format(sport_name.split("-")[0])


def grid(sport):
    """World points every metre over the field: where `geom_error` compares two H's."""
    x0, x1, y0, y1 = sport.box
    return np.stack(np.meshgrid(np.arange(x0, x1 + 1e-9, 1.0), np.arange(y0, y1 + 1e-9, 1.0)), -1).reshape(-1, 2)


def keypoints(kp_set, rows):
    """{template name: (K, 2) world keypoints} for every template in `rows`. Passed around explicitly rather than kept
    as a module global: DataLoader workers on Windows re-import this module and would see the default."""
    return {t: sports.get(t).keypoint_set(kp_set) for t in {r["template"] for r in rows}}


def select(rows, phase, sport_name):
    if phase == "pretrain":
        return [r for r in rows if r["split"] == "train"]
    return [r for r in rows if r["source"] == "hand" and r["split"] in FINETUNE_SPLITS and r["template"] == sport_name]


class Frames(Dataset):
    """(image tensor, target heatmaps, row index). With `augment`, a random camera turn / zoom and a mirror, applied to
    the real frame and to its H alike (exact, see ptz_warp), plus a brightness / contrast jitter."""

    def __init__(self, rows, augment, kp):
        self.rows, self.augment, self.kp = rows, augment, kp

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        sport = sports.get(r["template"])
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
            H = sport.canonicalize(G @ H)                          # a mirrored frame renames its points
            img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.25), beta=rng.uniform(-25, 25))
        x, s = to_input(img)
        s /= 2                                                     # heatmaps are at half the input
        tgt = render_heatmaps(np.diag([s, s, 1.0]) @ H, self.kp[r["template"]], sport.straight_lines(), *OUT_SIZE, SIGMA)
        return x, torch.from_numpy(tgt), i


def focal_loss(logits, target):
    """CenterNet's penalty-reduced focal loss: one positive pixel per visible point among ~130k negatives. The targets
    peak off-centre (sub-pixel), so the positive is each channel's highest pixel rather than an exact 1."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    t = target.float()
    pos = (t.eq(t.amax(dim=(-2, -1), keepdim=True)) & (t > 0.5)).float()
    loss = (torch.log(p) * (1 - p) ** 2 * pos).sum() + (torch.log(1 - p) * p ** 2 * (1 - t) ** 4 * (1 - pos)).sum()
    return -loss / pos.sum().clamp(min=1)


def corrupted(x, tgt):
    """True for a batch no frame and no rendered target can give: an input outside the normalised image range, a target
    outside [0, 1], a NaN or an inf (NaN fails every comparison). Such batches appeared at random in a soccer run and one
    of them poisoned every BatchNorm statistic (soccer.md section 24).
    ponytail: skipped, not explained - the cause was not found; if the skipped count climbs, test the hardware."""
    return not bool(x.abs().amax() <= 3 and tgt.amin() >= 0 and tgt.amax() <= 1)


def error(H, r, gate=False):
    """Px at 1920 between H and the row's label, inf for no H (or, with `gate`, one no real camera gives)."""
    sport = sports.get(r["template"])
    if H is None or (gate and not is_plausible_view(H, r["w"], r["h"], sport.box)):
        return np.inf
    return geom_error(H, np.asarray(r["H"], float), grid(sport), r["w"], r["h"])


@torch.no_grad()
def evaluate(model, rows, device, kp, batch=4, gate=False):
    """`error` for every row."""
    model.eval()
    errs = [np.inf] * len(rows)
    for x, tgt, idx in DataLoader(Frames(rows, augment=False, kp=kp), batch_size=batch, num_workers=2):
        if corrupted(x, tgt):                     # scored as misses: NaN heatmaps would stop the solver
            print("corrupted batch, scored as misses:", [rows[i]["id"] for i in idx.tolist()], flush=True)
            continue
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            heat = torch.sigmoid(model(x.to(device)).float()).cpu().numpy()
        for hm, i in zip(heat, idx.tolist()):
            r = rows[i]
            errs[i] = error(estimate_H(hm, sports.get(r["template"]), r["w"], kp[r["template"]]), r, gate)
    return np.array(errs)


def summary(errs):
    ok = np.isfinite(errs)
    if not ok.any():
        return "coverage 0%"
    return "coverage {:.0f}%  p50 {:.1f}  p90 {:.1f}  <10 px {:.0f}%  <25 px {:.0f}% (of all frames)".format(
        100 * ok.mean(), np.median(errs[ok]), np.percentile(errs[ok], 90), 100 * (errs < 10).mean(),
        100 * (errs < 25).mean())


def report(rows, errs, split):
    """Per frame, per video and total, as printed by `--eval`."""
    for r, e in zip(rows, errs):
        print("{:<40} {:>8.1f}".format(r["id"], e))
    for v in sorted({r.get("video") for r in rows} - {None}):
        print("  {:<12} {}".format(v, summary(errs[[r.get("video") == v for r in rows]])))
    print("{} ({} frames): {}".format(split, len(rows), summary(errs)))


def train(args, rows, device):
    tr = select(rows, args.phase, args.sport)
    if args.templates:
        tr = [r for r in tr if r["template"] in args.templates]
    if args.limit:      # few labels: the first N of one shuffle (seed 0, so a smaller N is a subset of a larger one),
        # repeated so that an epoch keeps its number of steps
        few = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:args.limit]]
        tr = few * (len(tr) // args.limit)
    sport = sports.get(args.sport)
    dev = [r for r in rows if r["split"] == "dev" and r["template"] == args.sport]
    kp = keypoints(args.keypoints, tr + dev)
    out = RUNS / ("kpline-" + args.sport) / (args.phase + ("" if args.keypoints == "base" else "-" + args.keypoints)
                                             + args.tag)
    if args.init and Path(args.init).resolve().parent == out.resolve():
        # the first dev evaluation always writes best_h.pt: it would overwrite the run being continued
        raise SystemExit("--init is inside {}: continue into another folder with --tag".format(out))
    out.mkdir(parents=True, exist_ok=True)
    nch = n_channels(sport, kp[args.sport])
    print("{}: {} train frames ({} distinct), {} dev frames, {} channels -> {}".format(
        args.phase, len(tr), len({r["id"] for r in tr}), len(dev), nch, out))
    model = HalfResUNet(ncls=nch).to(device)
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
    steps, t_start = min(len(loader), args.max_steps or len(loader)), time.time()
    for ep in range(epochs):
        model.train()
        t0, tot, n, skipped = time.time(), 0.0, 0, 0
        for x, tgt, idx in loader:
            x, tgt = x.to(device), tgt.to(device)
            if corrupted(x, tgt):                 # before the forward pass, which updates the BatchNorm statistics
                skipped += 1
                print("skipped a corrupted batch:", [tr[i]["id"] for i in idx.tolist()], flush=True)
                continue
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = focal_loss(logits, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item(), n + 1
            if n % max(1, steps // 20) == 0:      # progress every 5% of an epoch; the ETA includes the dev evaluations
                done = (ep + n / steps) / epochs
                eta = (time.time() - t_start) * (1 - done) / done
                print("  epoch {}/{}  {:3.0f}%  | total {:3.0f}%  | eta {:.0f}h{:02.0f}m  | loss {:.4f}".format(
                    ep + 1, epochs, 100 * n / steps, 100 * done, eta // 3600, eta % 3600 // 60, tot / n), flush=True)
            if args.max_steps and n >= args.max_steps:
                break
        sched.step()
        line = "epoch {:3d}  loss {:.4f}  {:.0f}s".format(ep, tot / max(n, 1), time.time() - t0)
        line += "  skipped {} corrupted batches".format(skipped) if skipped else ""
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
    ap.add_argument("--sport", choices=sports.names(), default="hockey-nhl")
    ap.add_argument("--phase", choices=("pretrain", "finetune"), default="pretrain")
    ap.add_argument("--init", help="weights to start from (the pretrain best_h.pt for the fine-tune)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="steps per epoch, 0 = all (smoke tests)")
    ap.add_argument("--eval", help="weights to evaluate instead of training")
    ap.add_argument("--split", default="test", help="with --eval: test | test_leaky | dev | fresh")
    ap.add_argument("--keypoints", choices=KEYPOINT_SETS, default="base",
                    help="keypoint set to train (runs go to <phase>-derived); --eval reads it from the weights")
    ap.add_argument("--templates", nargs="+", choices=sports.names(),
                    help="train on the rows of these templates only (default: every template of the index)")
    ap.add_argument("--limit", type=int, default=0, help="train on this many random train rows only (few-label runs)")
    ap.add_argument("--tag", default="", help="suffix of the run folder, e.g. --tag=-cont (with =) to continue a run without overwriting it")
    ap.add_argument("--gate", action="store_true", help="with --eval: refuse the H no real camera gives (hockey.md 14d)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [json.loads(line) for line in open(index_of(args.sport), encoding="utf-8")]
    if not args.eval:
        train(args, rows, device)
        return
    rows = [r for r in rows if r["split"] == args.split and r["template"] == args.sport]
    model, kp = load(args.eval, sports.get(args.sport), device)
    report(rows, evaluate(model, rows, device, {args.sport: kp}, gate=args.gate), args.split)


if __name__ == "__main__":
    main()
