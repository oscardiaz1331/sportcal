# ponytail: this prototype is module-level script code (loads models, opens the video, overwrites
# tracking_log.csv / tracked_out.mp4). Ceiling: it cannot be imported or tested. Upgrade: split into
# main(). Until then refuse to run by accident (an import once ran it). The rink calibration is the product
# pipeline (build_pipeline, ADR 0003).
if __name__ != "__main__":
    raise ImportError("hockey_demo is a script: run `python -m sportcal.product.hockey_demo`")

import argparse
import csv
import os

# Evita que Windows mezcle el cuDNN standalone del sistema con el que trae el
# wheel de torch (cu130); mezclarlos causa CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
import queue
import threading
from collections import deque

import cv2
import numpy as np
import supervision as sv
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ultralytics import YOLO
from trackers import ByteTrackTracker
from scenedetect.common import FrameTimecode
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import compute_downscale_factor

from sportcal.paths import RUNS
from sportcal.product.hockey import build_pipeline
from sportcal.sports.hockey import rink

ap = argparse.ArgumentParser(description="Detection + tracking + rink calibration (product pipeline) + minimap video.")
ap.add_argument("video", nargs="?", default="clip2.mp4")
ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0: whole video)")
args = ap.parse_args()

# rangos derivados de hsv_scan.csv (percentiles 2-98 de los frames de camara
# principal); ver analisis: h,s muy estables por la iluminacion fija del hielo
MAIN_CAMERA_H_RANGE = (120, 126)
MAIN_CAMERA_S_RANGE = (11, 12)
MAIN_CAMERA_V_RANGE = (150, 217)

# rink calibration model (ADR 0003), how many frames the last H is kept when it refuses, and the weight of each new
# answer against the H carried along the camera motion (the minimap stops shaking; hockey.md section 14i)
KPLINE_WEIGHTS = RUNS / "kpline" / "finetune" / "best_h.pt"
HOLD_FRAMES = 15
SMOOTH = 0.1

RINK = rink.RINK_NHL


def is_main_camera(h_med, s_med, v_med):
    return (
        MAIN_CAMERA_H_RANGE[0] <= h_med <= MAIN_CAMERA_H_RANGE[1]
        and MAIN_CAMERA_S_RANGE[0] <= s_med <= MAIN_CAMERA_S_RANGE[1]
        and MAIN_CAMERA_V_RANGE[0] <= v_med <= MAIN_CAMERA_V_RANGE[1]
    )


# colores fijos para lo que no es "jugador de un equipo"; los jugadores usan
# TEAM_COLORS (rojo/azul) segun el equipo que les asigne el histograma de
# camiseta. NO_TEAM_COLOR (verde) es el comodin: jugador aun sin clasificar,
# o tramo del recorrido del puck sin ningun jugador de equipo conocido cerca
GOALIE_COLOR = (180, 0, 180)
REFEREE_COLOR = (0, 215, 255)
PUCK_COLOR = (255, 255, 255)
TEAM_COLORS = [(0, 0, 255), (255, 0, 0)]  # rojo (equipo 0) / azul (equipo 1)
NO_TEAM_COLOR = (0, 200, 0)  # verde

PUCK_TRAIL_LEN = 25
# muestras de histograma de camiseta a acumular antes de correr el k-means;
# con ~10 jugadores visibles por frame en plano principal, se junta rapido
TEAM_CALIB_SAMPLES = 300
# distancia (m) dentro de la cual se considera que un jugador "tiene" el puck,
# para colorear su recorrido segun el equipo que lo controla
PUCK_POSSESSION_MAX_DIST_M = 3.0


# por debajo de esto un pixel se considera "sin color" (hielo, valla blanca,
# reflejos, patines) y no debe contar para el color de camiseta del equipo
JERSEY_SAT_THRESHOLD = 40
# si menos de esta fraccion del recorte tiene color de verdad, el recorte no
# es de fiar (jugador agachado/tapado: la banda 25%-65% cayo sobre todo en
# fondo, no en la camiseta) y se descarta en vez de ensuciar la calibracion
JERSEY_MIN_COLOR_FRACTION = 0.15


def player_jersey_histogram(frame, bbox):
    """Histograma HSV (H,S) de la camiseta de un jugador, para diferenciar equipos.

    Se recorta solo la banda vertical 25%-65% de la caja (torso), evitando el
    casco/cabeza arriba y los patines/hielo abajo. Eso no basta cuando el
    jugador esta agachado o tapado: se verifico a mano (ver conversacion) que
    en esos casos la banda cae sobre hielo/valla/patin en vez de la camiseta,
    con saturacion muy baja (~18) frente a un recorte limpio (~35-150) -- por
    eso ademas se enmascaran los pixeles de baja saturacion antes de construir
    el histograma, y si casi no queda pixel "con color" se descarta el recorte
    entero en vez de devolver un histograma que en realidad es de hielo.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    h = y2 - y1
    crop = frame[y1 + int(h * 0.25):y1 + int(h * 0.65), x1:x2]
    if crop.size == 0:
        return None
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_mask = (hsv_crop[:, :, 1] > JERSEY_SAT_THRESHOLD).astype(np.uint8)
    if color_mask.mean() < JERSEY_MIN_COLOR_FRACTION:
        return None  # recorte sin camiseta real visible (fondo, patin, oclusion)
    hist = cv2.calcHist([hsv_crop], [0, 1], color_mask, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def classify_team(hist, centroids):
    d0 = float(np.linalg.norm(hist - centroids[0]))
    d1 = float(np.linalg.norm(hist - centroids[1]))
    return 0 if d0 < d1 else 1


def load_model(local_relpath, hf_repo, hf_file):
    """Prefiere los pesos reentrenados en local (runs/); si no existen, HuggingFace."""
    local = os.path.join(RUNS, local_relpath)
    if os.path.exists(local):
        print(f"[modelo] pesos locales: {local}")
        return YOLO(local)
    print(f"[modelo] pesos de HuggingFace: {hf_repo}/{hf_file}")
    return YOLO(hf_hub_download(hf_repo, hf_file))


yolo = load_model(
    "hockeyai/yolo26s/weights/best.pt",
    "SimulaMet-HOST/HockeyAI", "HockeyAI_model_weight.pt",
)
CLASS_NAME_TO_ID = {name: cid for cid, name in yolo.names.items()}
CLASS_GOALIE = CLASS_NAME_TO_ID["goalie"]
CLASS_PLAYER = CLASS_NAME_TO_ID["player"]
CLASS_PUCK = CLASS_NAME_TO_ID["puck"]
CLASS_REFEREE = CLASS_NAME_TO_ID["referee"]
# unicas clases con un "rol" de partido; el resto (faceoff/goal/centroide son
# marcas de la pista, no entidades en juego) no se proyecta a la homografia
ROLE_CLASSES = frozenset({CLASS_PUCK, CLASS_GOALIE, CLASS_REFEREE, CLASS_PLAYER})
# no YOLO fallback: a refused frame keeps the last H (hold) rather than take an answer ~100 px off (ADR 0003)
rink_pipeline = build_pipeline(None, kpline_weights=KPLINE_WEIGHTS, hold_frames=HOLD_FRAMES, device=args.device,
                               smooth=SMOOTH)
tracker = ByteTrackTracker()
scene_detector = ContentDetector()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise IOError("no se pudo abrir el video")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    "tracked_out.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

write_queue: queue.Queue = queue.Queue(maxsize=8)


def _writer_worker():
    while True:
        item = write_queue.get()
        if item is None:
            write_queue.task_done()
            break
        writer.write(item)
        write_queue.task_done()


writer_thread = threading.Thread(target=_writer_worker, daemon=True)
writer_thread.start()

# el detector solo necesita un histograma agregado, no resolucion completa;
# igualamos el downscale que SceneManager aplicaria por defecto (~256px de ancho)
scene_downscale = compute_downscale_factor(max(frame_width, frame_height))
scene_detect_size = (round(frame_width / scene_downscale), round(frame_height / scene_downscale))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# el tipo de plano es una propiedad del plano completo, no de cada frame; se
# decide una sola vez con el promedio de los primeros CLASSIFY_WINDOW frames
# tras un corte, y esa etiqueta se mantiene fija hasta el siguiente corte
CLASSIFY_WINDOW = 30
shot_samples = []
shot_camera_label = None  # texto a mostrar; None mientras se acumulan muestras del plano actual

# minimapa cenital sobre el que se pintan los jugadores proyectados
minimap_base, minimap_to_img = rink.minimap_base(RINK)
mm_h, mm_w = minimap_base.shape[:2]
# imagen -> pista (m) del frame actual, o None si no hay calibracion
last_homography = None

# calibracion de equipos por color de camiseta: es global al video (los
# equipos no cambian de plano a plano), se hace una vez con las primeras
# TEAM_CALIB_SAMPLES camisetas vistas y no se reinicia en los cortes de plano
team_calib_histograms = []
team_centroids = None  # (2, dim) tras el k-means; None mientras se acumulan muestras

# ultimas posiciones del puck en el minimapa, para dibujar su recorrido; si se
# mantuviera entre planos distintos el trazo saltaria sin sentido, asi que se
# reinicia en cada corte de plano
puck_trail = deque(maxlen=PUCK_TRAIL_LEN)

# csv con un escalar (mediana) por canal HSV y por frame, para decidir a
# posteriori que canal(es) separan mejor pista de publico/banquillo
hsv_log_path = "hsv_scan.csv"
hsv_log_file = open(hsv_log_path, "w", newline="")
hsv_log = csv.writer(hsv_log_file)
hsv_log.writerow(["frame_idx", "h_median", "s_median", "v_median", "is_main_camera", "scene_cut", "rink_method"])

# tracking persistente en coordenadas de pista (metros): una fila por jugador y
# frame mientras haya homografia valida. shot_id identifica el plano continuo,
# porque tracker_id se reinicia en cada corte y por si solo no es unico en todo
# el video (el jugador #3 del plano 2 no tiene relacion con el #3 del plano 5)
tracking_log_path = "tracking_log.csv"
tracking_log_file = open(tracking_log_path, "w", newline="")
tracking_log = csv.writer(tracking_log_file)
tracking_log.writerow(["frame_idx", "shot_id", "tracker_id", "class_name", "pitch_x_m", "pitch_y_m"])
shot_id = 0

frame_idx = 0
try:
    with tqdm(total=total_frames, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or (args.max_frames and frame_idx >= args.max_frames):
                break  # se acabaron los frames

            small_frame = cv2.resize(frame, scene_detect_size, interpolation=cv2.INTER_LINEAR)

            cuts = scene_detector.process_frame(FrameTimecode(frame_idx, fps=fps), small_frame)
            if cuts:
                tqdm.write(f"[scene cut] frame {frame_idx}, se reinicia el tracker")
                tracker = ByteTrackTracker()
                shot_samples = []
                shot_camera_label = None
                rink_pipeline.reset()  # the held H belongs to the previous shot
                puck_trail.clear()
                shot_id += 1

            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            h_med, s_med, v_med = (int(np.median(hsv[:, :, c])) for c in range(3))
            frame_is_main = is_main_camera(h_med, s_med, v_med)

            if shot_camera_label is None:
                shot_samples.append(frame_is_main)
                if len(shot_samples) >= CLASSIFY_WINDOW:
                    shot_samples.sort()
                    majority_is_main = shot_samples[CLASSIFY_WINDOW // 2]
                    shot_camera_label = "CAMARA PRINCIPAL" if majority_is_main else "OTRO PLANO"

            camera_label = shot_camera_label or ("CAMARA PRINCIPAL" if frame_is_main else "OTRO PLANO")
            main_camera = camera_label == "CAMARA PRINCIPAL"

            # sin imgsz explicito, el checkpoint trae fijado 1280 (mas grande que
            # necesario a 1080p); medido: 800 da +50% fps y solo pierde detecciones
            # marginales (conf media 0.44, max 0.68 en la muestra probada) -- la
            # degradacion real empieza en 640 (ahi si se pierden detecciones de
            # hasta 0.89 de confianza)
            yolo_predictions = yolo.predict(source=frame, conf=0.25, imgsz=800, verbose=False, device=args.device)
            detections = sv.Detections.from_ultralytics(yolo_predictions[0])
            tracked = tracker.update(detections=detections)
            assert tracked.tracker_id is not None and tracked.class_id is not None

            # color por deteccion, calculado una sola vez y reusado tanto para la
            # caja en la imagen principal como para el punto en la homografia: el
            # mismo jugador se ve del mismo color en los dos sitios
            det_colors, det_teams = [], []
            for cls_id, bbox, tid in zip(tracked.class_id, tracked.xyxy, tracked.tracker_id):
                if cls_id == CLASS_PUCK:
                    det_colors.append(PUCK_COLOR)
                    det_teams.append(None)
                elif cls_id == CLASS_GOALIE:
                    det_colors.append(GOALIE_COLOR)
                    det_teams.append(None)
                elif cls_id == CLASS_REFEREE:
                    det_colors.append(REFEREE_COLOR)
                    det_teams.append(None)
                elif cls_id == CLASS_PLAYER:
                    hist = player_jersey_histogram(frame, bbox)
                    if hist is None:
                        det_colors.append(NO_TEAM_COLOR)
                        det_teams.append(None)
                    elif team_centroids is None:
                        team_calib_histograms.append(hist)
                        det_colors.append(NO_TEAM_COLOR)
                        det_teams.append(None)
                        if len(team_calib_histograms) >= TEAM_CALIB_SAMPLES:
                            calib_data = np.array(team_calib_histograms, dtype=np.float32)
                            best_labels = np.zeros((len(calib_data), 1), dtype=np.int32)
                            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
                            _, _, centers = cv2.kmeans(
                                calib_data, 2, best_labels, criteria, 10, cv2.KMEANS_PP_CENTERS
                            )
                            team_centroids = centers
                            tqdm.write(
                                f"[equipos] calibrados con {len(team_calib_histograms)} "
                                f"muestras en el frame {frame_idx}"
                            )
                    else:
                        team = classify_team(hist, team_centroids)
                        det_colors.append(TEAM_COLORS[team])
                        det_teams.append(team)
                else:
                    det_colors.append((int(37 * tid) % 255, int(97 * tid) % 255, int(157 * tid) % 255))
                    det_teams.append(None)

            annotated = frame.copy()
            for tid, bbox, color in zip(tracked.tracker_id, tracked.xyxy, det_colors):
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"#{tid}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            camera_color = (0, 200, 0) if main_camera else (0, 0, 200)
            cv2.putText(annotated, camera_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, camera_color, 2)

            # calibracion de la pista: H mundo -> imagen del producto; se dibujan las lineas de la
            # plantilla proyectadas para ver a ojo si encajan con las pintadas en el hielo
            est = rink_pipeline(frame)
            last_homography = None if est is None else np.linalg.inv(est.H)
            rink_method = "sin calibrar" if est is None else est.method
            if est is not None:
                rink.draw_rink(annotated, est.H, RINK, (0, 255, 255), 1)
            cv2.putText(annotated, f"pista: {rink_method}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if last_homography is not None:
                minimap = minimap_base.copy()
                # los pies del jugador (centro del borde inferior de la caja) son lo
                # unico que esta realmente sobre el hielo, que es el plano de la homografia
                if len(tracked) > 0:
                    feet = np.stack([
                        (tracked.xyxy[:, 0] + tracked.xyxy[:, 2]) / 2,
                        tracked.xyxy[:, 3],
                    ], axis=1).astype(np.float32)
                    on_rink = cv2.perspectiveTransform(
                        feet.reshape(-1, 1, 2), last_homography
                    ).reshape(-1, 2)
                    # posiciones (equipo, x, y) de jugadores ya clasificados, para
                    # saber que equipo tiene mas cerca el puck y colorear su recorrido
                    player_positions = []
                    for cls_id, team, (rx, ry) in zip(tracked.class_id, det_teams, on_rink):
                        if cls_id != CLASS_PLAYER or team is None:
                            continue
                        if 0 <= rx <= RINK["length"] and 0 <= ry <= RINK["width"]:
                            player_positions.append((team, rx, ry))

                    for tid, cls_id, (rx, ry), color in zip(
                        tracked.tracker_id, tracked.class_id, on_rink, det_colors
                    ):
                        if cls_id not in ROLE_CLASSES:
                            continue  # marca de la pista (faceoff/goal/centroide), no una entidad en juego
                        if not (0 <= rx <= RINK["length"] and 0 <= ry <= RINK["width"]):
                            continue  # fuera de la pista: proyeccion poco fiable
                        class_name = yolo.names[int(cls_id)]
                        tracking_log.writerow(
                            [frame_idx, shot_id, tid, class_name, round(float(rx), 3), round(float(ry), 3)]
                        )
                        px, py = minimap_to_img((rx, ry))[0].astype(int)

                        if cls_id == CLASS_PUCK:
                            nearest_team, nearest_dist = None, PUCK_POSSESSION_MAX_DIST_M
                            for team, px_m, py_m in player_positions:
                                d = ((px_m - rx) ** 2 + (py_m - ry) ** 2) ** 0.5
                                if d < nearest_dist:
                                    nearest_team, nearest_dist = team, d
                            trail_color = TEAM_COLORS[nearest_team] if nearest_team is not None else NO_TEAM_COLOR
                            puck_trail.append(((px, py), trail_color))
                            for i in range(1, len(puck_trail)):
                                (p1, _), (p2, seg_color) = puck_trail[i - 1], puck_trail[i]
                                fade = i / len(puck_trail)
                                faded = tuple(int(c * fade) for c in seg_color)
                                cv2.line(minimap, p1, p2, faded, 2)
                            cv2.circle(minimap, (px, py), 4, PUCK_COLOR, -1)
                            cv2.circle(minimap, (px, py), 5, (0, 0, 0), 1)
                            continue

                        cv2.circle(minimap, (px, py), 5, color, -1)
                        cv2.circle(minimap, (px, py), 6, (30, 30, 30), 1)
                        cv2.putText(minimap, str(tid), (px + 7, py - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1)
                annotated[frame_height - mm_h - 10:frame_height - 10, 10:10 + mm_w] = minimap

            hsv_log.writerow([frame_idx, h_med, s_med, v_med, int(main_camera), int(bool(cuts)), rink_method])

            write_queue.put(annotated)

            frame_idx += 1
            pbar.update(1)

except KeyboardInterrupt:
    tqdm.write(f"[Ctrl+C] deteniendo en el frame {frame_idx}; guardando lo grabado hasta ahora...")

cap.release()
write_queue.put(None)
writer_thread.join()
writer.release()
hsv_log_file.close()
tracking_log_file.close()
print(f"csv escrito en {hsv_log_path}")
print(f"csv escrito en {tracking_log_path}")