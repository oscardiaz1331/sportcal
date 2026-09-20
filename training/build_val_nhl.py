"""Construye un val NHL mas grande y VERIFICADO A OJO (docs: CLAUDE.md 7sexies --
el val actual son ~56 frames con 10-25 curvas por clase, un fallo = +-8pp).

Tres pasos:

  --generar     muestrea 1 frame/s de los videos NHL en zonas SIN vecinos de train
                (>= GAP frames de cualquier frame de hockeyrink_nhl/train), corre los
                DOS modelos de segmentacion (referencia y UDA -- la union evita que el
                val favorezca al modelo que genero los candidatos) con
                solve_from_probs (dof>=10, sin filtro de coste: decide la revision
                visual), y guarda cada candidato + hojas de contacto 2x2 con la
                plantilla reproyectada (amarillo = modelo A, cian = modelo B).
  (revision)    se lee cada hoja y se escribe datasets/nhl_val_cand/review.json:
                {"id": "A" | "B" | "no"}.
  --finalizar   escribe datasets/hockeyrink_nhl_valx/{images,labels}/val = val actual
                (copiado) + candidatos aceptados, en formato YOLO pose de 56 keypoints
                (label_from_H), para que diagnose_lines_seg.py / seg_to_homography.py
                lo lean con --dataset hockeyrink_nhl_valx sin tocar nada.

SESGO CONOCIDO, y por que se documenta: los candidatos salen de frames donde el DLT
resuelve (dof>=10), o sea frames con >=5 correspondencias en cuadro -- val de frames
"faciles". Sirve para medir LOCALIZACION de la segmentacion con mas muestra, no para
medir la cobertura del DLT (circular).
"""
import os

_CUDNN_DIR = os.sep.join(("NVIDIA", "CUDNN"))
os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if _CUDNN_DIR not in p
)

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import diagnose_lines_seg as DS  # noqa: E402
import relabel_reproject as RP  # noqa: E402
import rink  # noqa: E402
from auto_label import label_from_H  # noqa: E402
from seg_to_homography import solve_from_probs  # noqa: E402

CAND = ROOT / "datasets" / "nhl_val_cand"
VALX = ROOT / "datasets" / "hockeyrink_nhl_valx"
VIDEOS = ["nhl3", "nhl4", "nhl5", "nhl7", "nhl8", "nhl9", "nhl10"]
MODELOS = {"A": ROOT / "runs" / "lineas_seg" / "best.pt",
           "B": ROOT / "runs" / "lineas_seg_uda" / "last.pt"}
GAP = 120          # frames (a 60 fps) de separacion con cualquier frame de train


def train_indices():
    d = {}
    for p in (ROOT / "datasets" / "hockeyrink_nhl" / "images" / "train").glob("*.jpg"):
        m = re.match(r"(.+?)_(\d+)$", p.stem)
        if m:
            d.setdefault(m.group(1), []).append(int(m.group(2)))
    return d


def generar(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = rink.RINK_NHL
    tr = train_indices()
    (CAND / "frames").mkdir(parents=True, exist_ok=True)
    (CAND / "sheets").mkdir(parents=True, exist_ok=True)

    modelos = {}
    for k, path in MODELOS.items():
        DS.PESOS = path
        modelos[k] = DS.load_model(device)

    cands = []
    for v in VIDEOS:
        cap = cv2.VideoCapture(str(ROOT / (v + ".mp4")))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gap = int(GAP * fps / 60)
        bloq = np.zeros(n + 1, bool)
        for i in tr.get(v, []):
            bloq[max(0, i - gap):min(n, i + gap) + 1] = True
        step = int(round(fps))
        vistos = 0
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0 and not bloq[min(i, n)]:
                vistos += 1
                h, w = fr.shape[:2]
                sol = {}
                for k, (model, imgsz) in modelos.items():
                    probs = DS.predict_probs(model, fr, imgsz, device)
                    H, n_l, n_c, costo = solve_from_probs(probs, params, w, h)
                    if H is not None:
                        sol[k] = (H, float(costo))
                if sol:
                    cid = "{}_{:06d}".format(v, i)
                    cv2.imwrite(str(CAND / "frames" / (cid + ".jpg")), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    np.savez(CAND / "frames" / (cid + ".npz"), **{k: s[0] for k, s in sol.items()})
                    cands.append({"id": cid, "video": v, "frame": i, "modelos": {k: s[1] for k, s in sol.items()}})
            i += 1
        cap.release()
        print("{:6s} libres muestreados={:4d}  candidatos acumulados={}".format(v, vistos, len(cands)), flush=True)

    (CAND / "candidates.json").write_text(json.dumps(cands, indent=1))
    hojas(cands, params)


def dibuja(cid, params):
    fr = cv2.imread(str(CAND / "frames" / (cid + ".jpg")))
    z = np.load(CAND / "frames" / (cid + ".npz"))
    colores = {"A": (0, 255, 255), "B": (255, 255, 0)}       # BGR: A amarillo, B cian
    for k in z.files:
        rink.draw_rink(fr, z[k], params, colores[k], 2)
    fr = cv2.resize(fr, (1000, 562), interpolation=cv2.INTER_AREA)
    cv2.rectangle(fr, (0, 0), (330, 30), (0, 0, 0), -1)
    cv2.putText(fr, "{} [{}]".format(cid, "+".join(z.files)), (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return fr


def hojas(cands, params, por_hoja=4):
    for s in range(0, len(cands), por_hoja):
        tiles = [dibuja(c["id"], params) for c in cands[s:s + por_hoja]]
        while len(tiles) < por_hoja:
            tiles.append(np.zeros_like(tiles[0]))
        hoja = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        cv2.imwrite(str(CAND / "sheets" / "sheet_{:03d}.jpg".format(s // por_hoja)), hoja, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("hojas:", (len(cands) + por_hoja - 1) // por_hoja, "->", CAND / "sheets")


def finalizar(args):
    params = rink.RINK_NHL
    tpl = rink.build_template(params)
    rev = json.loads((CAND / "review.json").read_text())
    (VALX / "images" / "val").mkdir(parents=True, exist_ok=True)
    (VALX / "labels" / "val").mkdir(parents=True, exist_ok=True)
    src = ROOT / "datasets" / "hockeyrink_nhl"
    n_old = 0
    for p in (src / "images" / "val").glob("*.jpg"):
        shutil.copy2(p, VALX / "images" / "val" / p.name)
        shutil.copy2(src / "labels" / "val" / (p.stem + ".txt"), VALX / "labels" / "val" / (p.stem + ".txt"))
        n_old += 1
    n_new = 0
    for cid, dec in rev.items():
        if dec not in ("A", "B"):
            continue
        fr = cv2.imread(str(CAND / "frames" / (cid + ".jpg")))
        H = np.load(CAND / "frames" / (cid + ".npz"))[dec]
        h, w = fr.shape[:2]
        lab = label_from_H(H, tpl, w, h)
        if lab is None:
            continue
        kpts, box = lab
        cv2.imwrite(str(VALX / "images" / "val" / (cid + ".jpg")), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        RP.write_label(VALX / "labels" / "val" / (cid + ".txt"), 0, box, kpts)
        n_new += 1
    print("valx: {} del val original + {} verificados nuevos = {}  -> {}".format(n_old, n_new, n_old + n_new, VALX))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generar", action="store_true")
    ap.add_argument("--finalizar", action="store_true")
    args = ap.parse_args()
    if args.generar:
        generar(args)
    elif args.finalizar:
        finalizar(args)
    else:
        print(__doc__)
