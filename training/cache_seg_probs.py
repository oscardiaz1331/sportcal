"""Precalcula y cachea, para cada frame etiquetado, el mapa de probabilidad de
la red de segmentacion (reescalado pequeno) + la H de verdad -- para poder
entrenar homography_head.py sin repetir el forward de la red de segmentacion
en cada epoca (a ~3 fps, barrer 1700 imagenes por epoca serian ~9 min solo en
esto).

    python training/cache_seg_probs.py
    python training/cache_seg_probs.py --size 128x72

Escribe en datasets/<nombre>/seg_probs/<split>/<stem>.npz con:
    probs  (12, Hs, Ws) float16 -- softmax de la red, ya reescalado
    H      (3, 3) float64       -- verdad, metros -> pixeles (convencion de
                                    MM.project_points: [X,Y,1] @ H.T)
    wh     (2,) int32           -- ancho, alto ORIGINALES del frame (para
                                    poder medir el error de reproyeccion en
                                    la escala real durante el entrenamiento)
"""
import os

_CUDNN_DIR = os.sep.join(("NVIDIA", "CUDNN"))

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if _CUDNN_DIR not in p
)

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import make_line_masks as MM  # noqa: E402
import rink  # noqa: E402
from diagnose_lines_seg import PESOS, load_model  # noqa: E402
from relabel_reproject import read_label  # noqa: E402
from train_lines_seg import MEAN, STD  # noqa: E402

JOBS = [
    ("hockeyrink", rink.RINK_IIHF),
    ("hockeyrink_nhl", rink.RINK_NHL),
]


@torch.no_grad()
def predict_probs_small(model, img, imgsz, device, out_size):
    """Como diagnose_lines_seg.predict_probs, pero reescala DIRECTO de imgsz a
    out_size (pequeno) en vez de pasar primero por la resolucion original --
    evita un resize caro que solo para volver a encoger despues."""
    small = cv2.resize(img, imgsz, interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), 1)[0].cpu().numpy()
    out = np.stack([cv2.resize(probs[c], out_size, interpolation=cv2.INTER_AREA)
                     for c in range(probs.shape[0])])
    return out.astype(np.float16)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, imgsz = load_model(device)
    out_size = tuple(int(v) for v in args.size.split("x"))   # (w, h) pequeno

    for ds, params in JOBS:
        tpl = rink.build_template(params)
        for split in ("train", "val"):
            lbl_dir = ROOT / "datasets" / ds / "labels" / split
            img_dir = ROOT / "datasets" / ds / "images" / split
            out_dir = ROOT / "datasets" / ds / "seg_probs" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            lbls = sorted(lbl_dir.glob("*.txt"))
            t0 = time.time()
            n_ok = n_skip = 0
            for i, lbl in enumerate(lbls):
                out_path = out_dir / (lbl.stem + ".npz")
                if out_path.exists() and not args.force:
                    n_ok += 1
                    continue
                ip = img_dir / (lbl.stem + ".jpg")
                if not ip.exists():
                    n_skip += 1
                    continue
                img = cv2.imread(str(ip))
                if img is None:
                    n_skip += 1
                    continue
                h, w = img.shape[:2]
                rec = read_label(lbl)
                if rec is None:
                    n_skip += 1
                    continue
                fq, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
                if fq is None:
                    n_skip += 1
                    continue
                H = fq[0]
                probs = predict_probs_small(model, img, imgsz, device, out_size)
                np.savez(out_path, probs=probs, H=H.astype(np.float64),
                          wh=np.array([w, h], np.int32))
                n_ok += 1
                if (i + 1) % 100 == 0:
                    fps = (i + 1) / (time.time() - t0 + 1e-9)
                    print("  {} {} {}/{}  ({:.1f} fps)".format(ds, split, i + 1, len(lbls), fps))
            print("{:20s} {:6s}  {} cacheados, {} sin H/etiqueta  -> {}".format(
                ds, split, n_ok, n_skip, out_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="128x72", help="ancho x alto del mapa cacheado")
    ap.add_argument("--force", action="store_true", help="recalcular aunque ya exista el cache")
    args = ap.parse_args()
    run(args)
