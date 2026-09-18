"""Aparta una parte de datasets/hockeyrink_nhl como validacion (images/val + labels/val).

El corte es por BLOQUES de frames seguidos (mismo plano), no por frames sueltos: frames
consecutivos de un mismo plano son casi identicos y, repartidos entre train y val, inflarian
las metricas. Un bloque son frames del mismo video separados menos de --gap frames; cada
bloque va entero a train o a val segun el hash de su id, asi que el reparto es estable entre
ejecuciones y no depende del orden.

    python split_val.py                 # ~20% a validacion
    python split_val.py --fraction 0.3
    python split_val.py --undo          # devuelve todo a train

Idempotente: se puede lanzar las veces que haga falta.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DS = ROOT / "datasets" / "hockeyrink_nhl"

ap = argparse.ArgumentParser()
ap.add_argument("--fraction", type=float, default=0.2, help="proporcion de bloques a validacion")
ap.add_argument("--gap", type=int, default=300, help="frames de separacion que cortan un bloque")
ap.add_argument("--undo", action="store_true", help="devuelve todo a train")
args = ap.parse_args()


def move(stem: str, src: str, dst: str) -> None:
    for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
        s, d = DS / sub / src / f"{stem}{ext}", DS / sub / dst / f"{stem}{ext}"
        if s.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(d))


if not (DS / "images" / "train").exists():
    raise SystemExit(f"no encuentro {DS}/images/train")

# nombres actuales (esten en train o en val) -> (video, frame)
names = sorted(p.stem for sub in ("train", "val") for p in (DS / "images" / sub).glob("*.jpg"))
if args.undo:
    for n in names:
        move(n, "val", "train")
    print(f"[val] todo a train: {len(names)} frames")
    raise SystemExit

parsed = sorted((n.rsplit("_", 1)[0], int(n.rsplit("_", 1)[1]), n) for n in names)
blocks, block_of = {}, {}                       # bloque = frames seguidos del mismo video
last = None
for video, frame, name in parsed:
    if last is None or last[0] != video or frame - last[1] > args.gap:
        bid = f"{video}_{frame}"
    blocks.setdefault(bid, []).append(name)
    block_of[name] = bid
    last = (video, frame)

# cuota por video: se cogen bloques enteros, en orden de hash (estable), hasta llegar a la
# fraccion pedida DE ESE video, para que todos los pabellones aparezcan en train y en val
val_names = set()
for video in sorted({v for v, _, _ in parsed}):
    vblocks = sorted((b for b in blocks if b.rsplit("_", 1)[0] == video),
                     key=lambda b: hashlib.md5(b.encode()).hexdigest())
    total = sum(len(blocks[b]) for b in vblocks)
    quota, taken = args.fraction * total, 0
    for b in vblocks:                            # se anaden bloques mientras quepan en la cuota
        if taken + len(blocks[b]) <= quota and len(vblocks) > 1:
            val_names.update(blocks[b])
            taken += len(blocks[b])
    if not taken and len(vblocks) > 1:           # ningun bloque cabia: va el mas pequeno
        b = min(vblocks, key=lambda b: (len(blocks[b]), b))
        val_names.update(blocks[b])

for video, frame, name in parsed:
    to_val = name in val_names
    move(name, "train" if to_val else "val", "val" if to_val else "train")
n_val = len(val_names)

per_video = {}
for video, frame, name in parsed:
    per_video.setdefault(video, [0, 0])[name in val_names] += 1
print(f"[val] {len(blocks)} bloques, {n_val} frames a val, {len(parsed) - n_val} a train")
for v, (tr, va) in sorted(per_video.items()):
    print(f"      {v}: train={tr} val={va}")
