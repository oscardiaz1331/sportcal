"""Training and scoring of the template-conditioned model (`models/conditioned.py`, ADR 0005). One network is trained on
several sports at once; the sport held out (basketball) is only ever read for the experiment rows. Reuses the dataset,
loss, scoring and solver of `train_kpline.py` unchanged (that file is not edited, so a training of model A can run).

The five rows of ADR 0005 section 5, read on basketball:

    # zero-shot: hockey (NHL + IIHF), soccer and tennis only; the backbone starts from the hockey model A
    python -m sportcal.lab.common.train_conditioned --backbone runs/kpline/finetune/best_h.pt
    python -m sportcal.lab.common.train_conditioned --eval runs/kpline-cond/shared/best_h.pt --sport basketball-fiba --split test --gate
    # few-shot: the zero-shot weights fine-tuned on N basketball labels (N = 10, 50, 200)
    python -m sportcal.lab.common.train_conditioned --sports basketball-fiba --limit 50 --init runs/kpline-cond/shared/best_h.pt
    # few-shot control (the lazy alternative): a named model-A head on the same backbone, trained with the existing script
    python -m sportcal.lab.common.train_kpline --sport basketball-fiba --phase pretrain --limit 50 --init <hockey+soccer model A>

An epoch is a fixed number of steps; each batch draws every sport with equal probability, so a sport with 11k frames does
not drown one with 600. A checkpoint is chosen by the mean over the TRAINING sports of the median dev error: the held-out
sport's dev is never read when it is not being trained on. Runs go to runs/kpline-cond/<name><tag>.
"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sportcal import sports
from sportcal.lab.common.train_kpline import (Frames, corrupted, error, focal_loss, index_of, keypoints, report,
                                              summary)
from sportcal.models.conditioned import ConditionedUNet, query_inputs, raster
from sportcal.models.kpline import estimate_H, n_channels
from sportcal.paths import RUNS

TRAIN_SPORTS = ("hockey-nhl", "hockey-iihf", "soccer-fifa", "tennis-itf")
HELD_OUT = "basketball-fiba"


def family(template):
    """The sport of a template: hockey-nhl and hockey-iihf are both 'hockey'."""
    return template.split("-")[0]


class CondFrames(Frames):
    """`Frames` plus what the conditioned model needs for each frame: the drawing of its sport's template and the
    queries (see `query_inputs`). Sports have different numbers of elements, so every item is padded to the largest
    template in `rows`; `mask` marks the real queries."""

    def __init__(self, rows, augment, kp):
        super().__init__(rows, augment, kp)
        # per template: its drawing and its queries, computed once
        self.q = {t: (raster(sports.get(t)), *query_inputs(sports.get(t), kp[t])) for t in {r["template"] for r in rows}}
        self.qmax = max(len(v[2]) for v in self.q.values())

    def __getitem__(self, i):
        x, tgt, i = super().__getitem__(i)                  # frame, target heatmaps (n, h, w), row index
        ras, samples, extra = self.q[self.rows[i]["template"]]
        n = len(extra)
        pad = lambda a: torch.from_numpy(np.concatenate([a, np.zeros((self.qmax - n, *a.shape[1:]), a.dtype)]))
        mask = torch.arange(self.qmax) < n
        return x, pad(tgt.numpy()), pad(samples), pad(extra), mask, ras, i


def loader_of(rows, kp, args, train):
    ds = CondFrames(rows, augment=train, kp=kp)
    if not train:
        return DataLoader(ds, batch_size=4, num_workers=2)
    # equal chance per sport: weight = 1 / number of frames of the row's sport (ADR 0005, point 3). NHL and IIHF count
    # as one sport (hockey); each row still draws its own template
    count = Counter(family(r["template"]) for r in rows)
    weights = [1.0 / count[family(r["template"])] for r in rows]
    sampler = WeightedRandomSampler(weights, args.steps * args.batch, replacement=True)
    return DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True,
                      persistent_workers=args.workers > 0)


def run(model, batch, device):
    """Logits for a batch of CondFrames. Padded queries are forced to 'nothing here' so they cost the loss nothing."""
    x, tgt, samples, extra, mask, ras, idx = batch
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        logits = model(x.to(device), samples.to(device), extra.to(device), ras.to(device))
    return logits.masked_fill(~mask.to(device)[:, :, None, None], -20.0)


@torch.no_grad()
def evaluate(model, rows, device, kp, gate=False):
    """`error` (px at 1920, inf = no H) for every row, solved with the row's own template."""
    model.eval()
    errs = [np.inf] * len(rows)
    for batch in loader_of(rows, kp, None, train=False):
        x, tgt, idx = batch[0], batch[1], batch[-1]
        if corrupted(x, tgt):
            print("corrupted batch, scored as misses:", [rows[i]["id"] for i in idx.tolist()], flush=True)
            continue
        heat = torch.sigmoid(run(model, batch, device).float()).cpu().numpy()
        for hm, i in zip(heat, idx.tolist()):
            r = rows[i]
            s = sports.get(r["template"])
            errs[i] = error(estimate_H(hm[:n_channels(s, kp[r["template"]])], s, r["w"], kp[r["template"]]), r, gate)
    return np.array(errs)


def load_rows(names):
    """Every row of the indexes that hold the sports in `names` (the hockey index holds both of its templates)."""
    rows = []
    for path in {index_of(n) for n in names}:
        rows += [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows


def train(args, device):
    rows = load_rows(args.sports)
    if HELD_OUT in args.sports and not args.limit:
        raise SystemExit("{} is the held-out sport: train on it only with --limit N (the few-shot rows)".format(HELD_OUT))
    tr = [r for r in rows if r["split"] == "train" and r["template"] in args.sports]
    if args.limit:      # the first N of one shuffle (seed 0, as in train_kpline: a smaller N is a subset of a larger one)
        tr = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:args.limit]]
    rng = np.random.default_rng(0)
    dev = {s: [r for r in rows if r["split"] == "dev" and r["template"] == s] for s in args.sports}
    dev = {s: [d[i] for i in sorted(rng.permutation(len(d))[:args.dev_max])] for s, d in dev.items()}   # a fixed subset
    kp = keypoints("base", tr + sum(dev.values(), []))
    out = RUNS / "kpline-cond" / ((args.name or ("fewshot-{}".format(args.limit) if args.limit else "shared")) + args.tag)
    if (out / "best_h.pt").exists():
        raise SystemExit("{} already holds a run: use --tag=-other (the first dev evaluation writes best_h.pt)".format(out))
    out.mkdir(parents=True, exist_ok=True)
    print("train {} frames {} | dev {} -> {}".format(len(tr), dict(Counter(r["template"] for r in tr)),
                                                    {s: len(d) for s, d in dev.items()}, out), flush=True)

    model = ConditionedUNet().to(device)
    if args.init:           # continue from a conditioned checkpoint (zero-shot weights -> few-shot)
        model.load_state_dict(torch.load(args.init, map_location=device))
    elif args.backbone:     # frame part from a model-A checkpoint
        model.load_backbone(args.backbone)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loader = loader_of(tr, kp, args, train=True)
    best, saved, t_start = np.inf, False, time.time()
    for ep in range(args.epochs):
        model.train()
        t0, tot, n, skipped = time.time(), 0.0, 0, 0
        for batch in loader:
            if corrupted(batch[0], batch[1]):        # before the forward pass, which updates the BatchNorm statistics
                skipped += 1
                continue
            loss = focal_loss(run(model, batch, device), batch[1].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item(), n + 1
            if n % max(1, args.steps // 10) == 0:
                done = (ep + n / args.steps) / args.epochs
                eta = (time.time() - t_start) * (1 - done) / done
                print("  epoch {}/{}  {:3.0f}%  | total {:3.0f}%  | eta {:.0f}h{:02.0f}m  | loss {:.4f}".format(
                    ep + 1, args.epochs, 100 * n / args.steps, 100 * done, eta // 3600, eta % 3600 // 60, tot / n),
                    flush=True)
        sched.step()
        line = "epoch {:3d}  loss {:.4f}  {:.0f}s".format(ep, tot / max(n, 1), time.time() - t0)
        line += "  skipped {} corrupted batches".format(skipped) if skipped else ""
        torch.save(model.state_dict(), out / "last.pt")
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            errs = {s: evaluate(model, d, device, kp) for s, d in dev.items()}
            for s, e in errs.items():
                line += "\n    dev {:<12} {}".format(s, summary(e))
            # one median per sport, hockey pooling its two templates; inf when fewer than half of the frames get an H
            p50 = [np.median(np.concatenate([e for s, e in errs.items() if family(s) == f]))
                   for f in {family(s) for s in errs}]
            score = np.mean(p50)                               # mean over the sports being trained on
            if score < best or not saved:                      # a run that never answers still leaves its own file
                best, saved = score, True
                torch.save(model.state_dict(), out / "best_h.pt")
                line += "\n    mean p50 {:.1f}  <- best_h.pt".format(score)
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sports", nargs="+", choices=sports.names(), default=list(TRAIN_SPORTS),
                    help="sports to train on (basketball only together with --limit)")
    ap.add_argument("--backbone", help="model-A checkpoint for the frame part (e.g. runs/kpline/finetune/best_h.pt)")
    ap.add_argument("--init", help="conditioned checkpoint to continue from (zero-shot weights for the few-shot rows)")
    ap.add_argument("--limit", type=int, default=0, help="train on this many random train rows only (few-shot)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=500, help="steps per epoch (an epoch is fixed, not a pass over the data)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--dev-max", type=int, default=60, help="dev frames per sport read at each evaluation")
    ap.add_argument("--name", help="run folder name (default: shared, or fewshot-<limit>)")
    ap.add_argument("--tag", default="", help="suffix of the run folder, e.g. --tag=-cont (with =)")
    ap.add_argument("--eval", help="conditioned weights to evaluate instead of training")
    ap.add_argument("--sport", choices=sports.names(), default=HELD_OUT, help="with --eval: the sport to read")
    ap.add_argument("--split", default="test", help="with --eval: test | dev | fresh ...")
    ap.add_argument("--gate", action="store_true", help="with --eval: refuse the H no real camera gives")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.eval:
        train(args, device)
        return
    rows = [r for r in load_rows([args.sport]) if r["split"] == args.split and r["template"] == args.sport]
    model = ConditionedUNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.eval, map_location=device))
    report(rows, evaluate(model, rows, device, keypoints("base", rows), gate=args.gate), args.split)


if __name__ == "__main__":
    main()
