"""Referencia visual del modelo HockeyRink (SimulaMet-HOST/HockeyRink) + semillas NHL.

Hace dos cosas, las dos con el modelo ORIGINAL de HuggingFace (no el yolo26 local):

1. hockeyrink_reference.mp4 -- corre HockeyRink sobre imagenes del train y dibuja
   cada keypoint con su id: relleno = prediccion, hueco = ground truth (etiqueta
   del dataset), linea gris = error entre ambos. Colores por bloque de ids:
   rojo 0-19, verde 20-35, azul 36-55. Sirve para ver que representa cada punto.

2. rink_preds_<video>.json -- corre HockeyRink cada --every frames de tus videos
   NHL y guarda los 56 (x, y, conf). Son las semillas para etiquetar esos frames
   (el indice de frame es el de una lectura secuencial con cv2, sin seeks).

    python rink_reference.py
    python rink_reference.py --n 200 --every 20 --videos clip.mp4.webm clip2.mp4
    python rink_reference.py --n 0 --every 15 --videos nhl3.mp4 nhl4.mp4   # solo semillas
"""
import argparse
import json
import os

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if "NVIDIA\\CUDNN" not in p
)

from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
COLORS = {0: (60, 60, 255), 20: (60, 220, 60), 36: (255, 150, 40)}  # BGR por bloque


def block_color(i):
    return COLORS[0] if i < 20 else COLORS[20] if i < 36 else COLORS[36]


def predict(model, img):
    r = model.predict(source=img, verbose=False, device=0)[0]
    if r.keypoints is None or r.keypoints.xy.shape[0] == 0:
        return None
    xy = r.keypoints.xy[0].cpu().numpy()
    conf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
    return {"box_conf": float(r.boxes.conf[0]), "xy": xy.round(1).tolist(), "conf": conf.round(3).tolist()}


def text(img, s, org, color, scale=0.55):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def reference_video(model, n, fps, conf_thr, out):
    imgs = sorted((ROOT / "datasets" / "hockeyrink" / "images" / "train").glob("*.jpg"))
    imgs = imgs[:: max(1, len(imgs) // n)][:n]
    writer = None
    for k, path in enumerate(imgs):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(ROOT / out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        gt = {}
        lbl = path.parent.parent.parent / "labels" / "train" / f"{path.stem}.txt"
        if lbl.exists():
            vals = np.array(lbl.read_text().split()[5:], float).reshape(-1, 3)
            gt = {i: (x * w, y * h) for i, (x, y, v) in enumerate(vals) if v > 0}

        pred = predict(model, img)
        vis = img.copy()
        for i, (x, y) in gt.items():
            cv2.circle(vis, (int(x), int(y)), 9, block_color(i), 2, cv2.LINE_AA)
        if pred:
            for i, ((x, y), c) in enumerate(zip(pred["xy"], pred["conf"])):
                if c < conf_thr:
                    continue
                p = (int(x), int(y))
                if i in gt:
                    cv2.line(vis, p, tuple(int(v) for v in gt[i]), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.circle(vis, p, 5, block_color(i), -1, cv2.LINE_AA)
                text(vis, str(i), (p[0] + 7, p[1] - 7), block_color(i))
        # ids solo-GT (el modelo no los ve) tambien etiquetados, en gris
        for i, (x, y) in gt.items():
            if not pred or pred["conf"][i] < conf_thr:
                text(vis, str(i), (int(x) + 10, int(y) + 16), (190, 190, 190))

        text(vis, f"{k + 1}/{len(imgs)}  {path.name}", (15, 30), (255, 255, 255), 0.7)
        text(vis, "relleno = HockeyRink | hueco = ground truth | rojo 0-19  verde 20-35  azul 36-55",
             (15, 60), (255, 255, 255), 0.6)
        writer.write(vis)
    writer.release()
    print(f"[ref] {len(imgs)} imagenes -> {out}")


def video_seeds(model, video, every):
    cap = cv2.VideoCapture(str(ROOT / video))
    if not cap.isOpened():
        print(f"[!] no se pudo abrir {video}")
        return
    preds, idx = [], 0
    while True:
        ok, frame = cap.read()  # lectura secuencial: el indice coincide al re-decodificar
        if not ok:
            break
        if idx % every == 0:
            p = predict(model, frame)
            if p:
                preds.append({"frame": idx, **p})
        idx += 1
    cap.release()
    out = ROOT / f"rink_preds_{Path(video).name.split('.')[0]}.json"
    out.write_text(json.dumps({"video": video, "n_frames": idx, "every": every, "preds": preds}))
    print(f"[seeds] {video}: {len(preds)} frames con prediccion -> {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SimulaMet-HOST/HockeyRink")
    ap.add_argument("--file", default="HockeyRink.pt")
    ap.add_argument("--n", type=int, default=150, help="imagenes del train en el video")
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--conf", type=float, default=0.5, help="umbral de conf por keypoint al dibujar")
    ap.add_argument("--videos", nargs="*", default=["clip.mp4.webm", "clip2.mp4"])
    ap.add_argument("--every", type=int, default=30, help="frames entre semillas en tus videos")
    args = ap.parse_args()

    model = YOLO(hf_hub_download(args.repo, args.file))
    if args.n > 0:  # --n 0: only compute seeds for --videos
        reference_video(model, args.n, args.fps, args.conf, "hockeyrink_reference.mp4")
    for v in args.videos:
        video_seeds(model, v, args.every)
