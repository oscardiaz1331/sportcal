"""Set aside part of datasets/hockeyrink_nhl as validation (images/val + labels/val).

The cut is by BLOCKS of consecutive frames (same shot), not single frames: consecutive
frames of one shot are nearly identical and would inflate the metrics if spread across
train and val. A block is frames of the same video less than --gap frames apart; each
block goes entirely to train or to val according to its hash, so the split is stable
across runs and independent of ordering.

    python -m sportcal.lab.hockey.split_val                  # ~20% to validation
    python -m sportcal.lab.hockey.split_val --fraction 0.3
    python -m sportcal.lab.hockey.split_val --undo           # everything back to train

CAUTION: this RE-DERIVES the whole partition and moves files. Frames merged into the
dataset after the last split (e.g. auto-labelled ones) can change sides, which silently
changes the validation set. Only run it when you mean to re-split.
"""
import argparse
import hashlib
import shutil

from sportcal.paths import DATASETS

DS = DATASETS / "hockeyrink_nhl"


def move(stem, src, dst):
    for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
        s, d = DS / sub / src / f"{stem}{ext}", DS / sub / dst / f"{stem}{ext}"
        if s.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(d))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.2, help="fraction of blocks sent to validation")
    ap.add_argument("--gap", type=int, default=300, help="frame gap that cuts a block")
    ap.add_argument("--undo", action="store_true", help="move everything back to train")
    args = ap.parse_args(argv)

    if not (DS / "images" / "train").exists():
        raise SystemExit(f"cannot find {DS}/images/train")

    # current names (in train or val) -> (video, frame)
    names = sorted(p.stem for sub in ("train", "val") for p in (DS / "images" / sub).glob("*.jpg"))
    if args.undo:
        for n in names:
            move(n, "val", "train")
        print(f"[val] everything to train: {len(names)} frames")
        return

    parsed = sorted((n.rsplit("_", 1)[0], int(n.rsplit("_", 1)[1]), n) for n in names)
    blocks = {}  # block = consecutive frames of the same video
    last = None
    for video, frame, name in parsed:
        if last is None or last[0] != video or frame - last[1] > args.gap:
            bid = f"{video}_{frame}"
        blocks.setdefault(bid, []).append(name)
        last = (video, frame)

    # per-video quota: whole blocks in (stable) hash order until that video's requested
    # fraction is reached, so every arena appears in both train and val
    val_names = set()
    for video in sorted({v for v, _, _ in parsed}):
        vblocks = sorted((b for b in blocks if b.rsplit("_", 1)[0] == video),
                         key=lambda b: hashlib.md5(b.encode()).hexdigest())
        quota = args.fraction * sum(len(blocks[b]) for b in vblocks)
        taken = 0
        for b in vblocks:
            if taken + len(blocks[b]) <= quota and len(vblocks) > 1:
                val_names.update(blocks[b])
                taken += len(blocks[b])
        if not taken and len(vblocks) > 1:  # no block fit the quota: send the smallest one
            smallest = min(vblocks, key=lambda b: (len(blocks[b]), b))
            val_names.update(blocks[smallest])

    for _, _, name in parsed:
        to_val = name in val_names
        move(name, "train" if to_val else "val", "val" if to_val else "train")

    per_video = {}
    for video, _, name in parsed:
        per_video.setdefault(video, [0, 0])[name in val_names] += 1
    print(f"[val] {len(blocks)} blocks, {len(val_names)} frames to val, {len(parsed) - len(val_names)} to train")
    for v, (tr, va) in sorted(per_video.items()):
        print(f"      {v}: train={tr} val={va}")


if __name__ == "__main__":
    main()
