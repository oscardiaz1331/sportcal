"""Herramienta de anotacion semi-asistida para los 56 keypoints de HockeyRink
sobre frames NHL de tu propio clip.

Por que semi-asistida: el modelo (entrenado en SHL/IIHF) coloca los 56 puntos
en cualquier frame, pero mal en NHL (el "campo inventado" que se vio en
rink_debug.py). Partir de esa prediccion y arrastrar cada punto a su sitio
real es mas rapido que hacer clic desde cero en los 56; los puntos que no
aparecen en el frame (fuera de camara, tapados) se ocultan en vez de forzar
una posicion inventada.

El resultado se guarda en un CSV (frame_idx, keypoint_id, x, y, visible) que
sirve luego como dataset de fine-tuning sobre NHL.

Controles:
  clic + arrastrar   mueve el punto mas cercano (radio de "pick" ~15px)
  0-9 luego Enter    selecciona un punto por su id (para los muy juntos:
                     hash marks, esquinas del area) -- lo resalta en magenta
  flechas            mueve el punto seleccionado (paso actual, ver f)
  f                  alterna el paso de las flechas entre 1px y 10px
  v                  oculta/muestra el punto seleccionado
  s                  guarda las anotaciones del frame actual
  r                  descarta ediciones y vuelve a la prediccion cruda del modelo
  n / p              siguiente/anterior frame candidato (avanza --step)
  g luego digitos+Enter   ir a un numero de frame concreto
  q / ESC            salir

Uso:
  python rink_annotate.py
  python rink_annotate.py --start 500 --step 60 --out nhl_annotations.csv

Nota: los codigos de flecha son los habituales de cv2.waitKeyEx en Windows;
si en tu maquina no responden, dimelo y los ajusto (varian algo por build).
"""
import argparse
import csv
import os

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

N_KEYPOINTS = 56
PICK_RADIUS = 15
ARROW_LEFT, ARROW_UP, ARROW_RIGHT, ARROW_DOWN = 2424832, 2490368, 2555904, 2621440


def load_annotations(path):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame_idx = int(row["frame_idx"])
            kid = int(row["keypoint_id"])
            data.setdefault(frame_idx, [[0.0, 0.0, 0] for _ in range(N_KEYPOINTS)])
            data[frame_idx][kid] = [float(row["x"]), float(row["y"]), int(row["visible"])]
    return data


def save_annotations(path, data):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "keypoint_id", "x", "y", "visible"])
        for frame_idx in sorted(data):
            for kid, (x, y, vis) in enumerate(data[frame_idx]):
                w.writerow([frame_idx, kid, round(x, 2), round(y, 2), vis])


def predict_seed(rink_model, frame):
    result = rink_model.predict(source=frame, verbose=False, device=0)[0]
    points = [[frame.shape[1] / 2, frame.shape[0] / 2, 1] for _ in range(N_KEYPOINTS)]
    if result.keypoints is not None and result.keypoints.xy.shape[0] > 0:
        xy = result.keypoints.xy[0].cpu().numpy()
        for i, (x, y) in enumerate(xy):
            points[i] = [float(x), float(y), 1]
    return points


parser = argparse.ArgumentParser()
parser.add_argument("--video", default="clip.mp4.webm")
parser.add_argument("--start", type=int, default=500)
parser.add_argument("--step", type=int, default=60, help="frames a avanzar con n/p")
parser.add_argument("--out", default="nhl_annotations.csv")
args = parser.parse_args()

rink_model = YOLO(hf_hub_download("SimulaMet-HOST/HockeyRink", "HockeyRink.pt"))
annotations = load_annotations(args.out)

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise IOError(f"no se pudo abrir {args.video}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def load_frame(idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, img = cap.read()
    return img if ret else None


def get_points(idx, img):
    if idx in annotations:
        return [p[:] for p in annotations[idx]]
    return predict_seed(rink_model, img)


frame_idx = args.start
frame_img = load_frame(frame_idx)
if frame_img is None:
    raise IOError(f"no se pudo leer el frame {frame_idx}")
points = get_points(frame_idx, frame_img)

dirty = False
active_id = None
nudge = 1
input_mode = None   # None, "select" o "goto"
input_buffer = ""
dragging_id = None


def nearest_point(x, y):
    best_i, best_d = None, PICK_RADIUS
    for i, (px, py, _vis) in enumerate(points):
        d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def on_mouse(event, x, y, _flags, _param):
    global dragging_id, active_id, dirty
    if event == cv2.EVENT_LBUTTONDOWN:
        i = nearest_point(x, y)
        if i is not None:
            dragging_id = i
            active_id = i
    elif event == cv2.EVENT_MOUSEMOVE and dragging_id is not None:
        points[dragging_id][0] = float(x)
        points[dragging_id][1] = float(y)
        dirty = True
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_id = None


WINDOW = "anotacion NHL rink"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1280, 720)
cv2.setMouseCallback(WINDOW, on_mouse)


def goto_frame(idx):
    global frame_idx, frame_img, points, active_id, dirty
    idx = max(0, min(total_frames - 1, idx))
    img = load_frame(idx)
    if img is None:
        print(f"[!] no se pudo leer el frame {idx}")
        return
    if dirty:
        print(f"[!] frame {frame_idx} tenia cambios sin guardar (se pierden)")
    frame_idx, frame_img = idx, img
    points = get_points(frame_idx, frame_img)
    active_id, dirty = None, False


while True:
    vis = frame_img.copy()
    for i, (x, y, kvis) in enumerate(points):
        p = (int(x), int(y))
        if not kvis:
            cv2.circle(vis, p, 3, (100, 100, 100), 1)
            continue
        color = (0, 255, 255)
        if i == active_id:
            cv2.circle(vis, p, 10, (255, 0, 255), 2)
            color = (255, 0, 255)
        cv2.circle(vis, p, 4, color, -1)
        cv2.putText(vis, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    n_visible = sum(1 for _, _, kvis in points if kvis)
    status = "[SIN GUARDAR]" if dirty else ("[guardado]" if frame_idx in annotations else "[nuevo]")
    hud = [
        f"frame {frame_idx}/{total_frames}  {status}",
        f"visibles: {n_visible}/{N_KEYPOINTS}   activo: {active_id if active_id is not None else '-'}   paso flechas: {nudge}px",
        f"modo: {input_mode or '-'}  {input_buffer}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(vis, line, (10, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(vis, line, (10, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    cv2.imshow(WINDOW, vis)
    raw_key = cv2.waitKeyEx(20)
    if raw_key == -1:
        continue
    key = raw_key & 0xFF

    if input_mode in ("select", "goto"):
        if key in (13, 10):  # Enter
            if input_buffer:
                n = int(input_buffer)
                if input_mode == "select" and 0 <= n < N_KEYPOINTS:
                    active_id = n
                elif input_mode == "goto":
                    goto_frame(n)
            input_mode, input_buffer = None, ""
        elif key == 27:
            input_mode, input_buffer = None, ""
        elif key < 256 and chr(key).isdigit():
            input_buffer += chr(key)
        continue

    if key in (ord("q"), 27):
        if dirty:
            print(f"[!] frame {frame_idx} tiene cambios sin guardar")
        break
    elif key == ord("s"):
        annotations[frame_idx] = [p[:] for p in points]
        save_annotations(args.out, annotations)
        dirty = False
        print(f"[guardado] frame {frame_idx} -> {args.out}")
    elif key == ord("r"):
        points = predict_seed(rink_model, frame_img)
        dirty = True
    elif key == ord("v") and active_id is not None:
        points[active_id][2] = 0 if points[active_id][2] else 1
        dirty = True
    elif key == ord("f"):
        nudge = 10 if nudge == 1 else 1
    elif key in tuple(ord(str(d)) for d in range(10)):
        input_mode, input_buffer = "select", chr(key)
    elif key == ord("g"):
        input_mode, input_buffer = "goto", ""
    elif key == ord("n"):
        goto_frame(frame_idx + args.step)
    elif key == ord("p"):
        goto_frame(frame_idx - args.step)
    elif active_id is not None and raw_key in (ARROW_LEFT, ARROW_RIGHT, ARROW_UP, ARROW_DOWN):
        dx = -nudge if raw_key == ARROW_LEFT else (nudge if raw_key == ARROW_RIGHT else 0)
        dy = -nudge if raw_key == ARROW_UP else (nudge if raw_key == ARROW_DOWN else 0)
        points[active_id][0] += dx
        points[active_id][1] += dy
        dirty = True

cap.release()
cv2.destroyAllWindows()
