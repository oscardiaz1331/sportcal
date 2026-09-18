"""Escaneo de HSV por frame (con tiempo de video) para calibrar los rangos de
camara principal en cualquier clip, igual que se hizo para clip.mp4.webm y
iihf_clip.mp4. Lee frames secuencialmente (sin cap.set por frame, que es
lento) y guarda el tiempo en segundos junto al frame, para ubicar facil los
ejemplos que se den por timestamp.

Uso:
  python hsv_scan.py --video clip2.mp4 --out clip2_hsv_scan.csv
"""
import argparse
import csv

import cv2
import numpy as np
from scenedetect.scene_manager import compute_downscale_factor

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--out", default=None, help="por defecto <video sin extension>_hsv_scan.csv")
args = parser.parse_args()

out_path = args.out or (args.video.rsplit(".", 1)[0] + "_hsv_scan.csv")

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise IOError(f"no se pudo abrir {args.video}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

downscale = compute_downscale_factor(max(frame_width, frame_height))
size = (round(frame_width / downscale), round(frame_height / downscale))

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_idx", "time_s", "h_median", "s_median", "v_median"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h_med, s_med, v_med = (int(np.median(hsv[:, :, c])) for c in range(3))
        w.writerow([frame_idx, round(frame_idx / fps, 3), h_med, s_med, v_med])

        if frame_idx % 1000 == 0:
            print(f"frame {frame_idx}/{total_frames}", flush=True)
        frame_idx += 1

cap.release()
print(f"listo: {frame_idx} frames -> {out_path}")
