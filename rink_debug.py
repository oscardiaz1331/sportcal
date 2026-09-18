"""Visor rapido del detector de keypoints de HockeyRink, solo, con imshow.

Nada de tracking ni YOLO de jugadores ni escritura a disco: es para ver si el
modelo de rink detecta bien o mal antes de decidir si vale la pena reentrenar.

Controles:
  espacio  pausa / reanuda
  n        avanza un frame (estando en pausa)
  h        muestra/oculta la reproyeccion de las lineas de la pista
  +/-      sube/baja el umbral de confianza de los keypoints
  q / ESC  salir

Uso:
  python rink_debug.py
  python rink_debug.py --start 3300 --imgsz 320 --conf 0.4
"""
import argparse
import os

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if "NVIDIA\\CUDNN" not in p
)

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

import rink

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="clip.mp4.webm")
parser.add_argument("--start", type=int, default=0, help="frame inicial")
parser.add_argument("--imgsz", type=int, default=None, help="imgsz de predict; por defecto el del modelo (640)")
parser.add_argument("--conf", type=float, default=0.5, help="umbral inicial de confianza por keypoint")
parser.add_argument("--min-kpts", type=int, default=6, help="minimo de keypoints para calcular homografia")
args = parser.parse_args()

rink_model = YOLO(hf_hub_download("SimulaMet-HOST/HockeyRink", "HockeyRink.pt"))
RINK_TEMPLATE = rink.build_template(rink.RINK_NHL_FITTED)

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise IOError(f"no se pudo abrir {args.video}")
cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

conf_threshold = args.conf
show_homography = True
paused = False
frame_idx = args.start
predict_kwargs = dict(verbose=False, device=0)
if args.imgsz:
    predict_kwargs["imgsz"] = args.imgsz

WINDOW = "rink keypoints debug"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1280, 720)

frame = None
while True:
    if not paused or frame is None:
        ret, frame = cap.read()
        if not ret:
            print("fin del video")
            break
        frame_idx += 1

    vis = frame.copy()
    result = rink_model.predict(source=frame, **predict_kwargs)[0]

    n_ok = 0
    good_ids, good_xy = [], []
    if result.keypoints is not None and result.keypoints.xy.shape[0] > 0:
        kpts_xy = result.keypoints.xy[0].cpu().numpy()
        kpts_conf = result.keypoints.conf[0].cpu().numpy()
        for idx, ((x, y), c) in enumerate(zip(kpts_xy, kpts_conf)):
            # color por confianza: rojo (baja) -> verde (alta), asi se ve de
            # un vistazo que tan seguro esta el modelo sin leer numeros
            color = (0, int(255 * min(c, 1.0)), int(255 * (1 - min(c, 1.0))))
            radius = 3 if c < conf_threshold else 5
            cv2.circle(vis, (int(x), int(y)), radius, color, -1 if c >= conf_threshold else 1)
            cv2.putText(vis, f"{idx}:{c:.2f}", (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if c >= conf_threshold:
                n_ok += 1
                good_ids.append(idx)
                good_xy.append((x, y))

    homography_ok = False
    if show_homography and len(good_ids) >= args.min_kpts:
        Hm, mask = cv2.findHomography(
            RINK_TEMPLATE[good_ids].astype(np.float32),
            np.asarray(good_xy, np.float32),
            cv2.RANSAC, 8.0,
        )
        if Hm is not None:
            homography_ok = True
            rink.draw_rink(vis, Hm, rink.RINK_NHL_FITTED, (0, 255, 255), 2)
            inliers = int(mask.sum())

    hud = [
        f"frame {frame_idx}/{total_frames}  {'PAUSA' if paused else ''}",
        f"conf>={conf_threshold:.2f}: {n_ok}/56 keypoints  imgsz={predict_kwargs.get('imgsz', 'default')}",
        f"homografia: {'OK (' + str(inliers) + ' inliers)' if homography_ok else 'insuficientes puntos' if show_homography else 'oculta (h)'}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(vis, line, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, line, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow(WINDOW, vis)
    key = cv2.waitKey(1 if not paused else 0) & 0xFF
    if key in (ord("q"), 27):
        break
    elif key == ord(" "):
        paused = not paused
    elif key == ord("n"):
        paused = True
        frame = None  # fuerza leer el siguiente frame en la proxima vuelta
    elif key == ord("h"):
        show_homography = not show_homography
    elif key in (ord("+"), ord("=")):
        conf_threshold = min(1.0, conf_threshold + 0.05)
    elif key in (ord("-"), ord("_")):
        conf_threshold = max(0.0, conf_threshold - 0.05)

cap.release()
cv2.destroyAllWindows()
