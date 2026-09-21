"""Visor rapido de EdgeDrawing (lineas + elipses) sobre el video, con imshow.

Alternativa clasica (no aprendida) al modelo de keypoints de HockeyRink: en vez
de una red entrenada en pistas SHL, usa geometria de bordes para encontrar
lineas rectas y circulos/elipses, que no tiene el sesgo de dominio SHL->NHL que
mostraba rink_debug.py. A cambio da primitivas crudas sin etiqueta (una elipse
grande puede ser el circulo central... o un anuncio ovalado en la valla), asi
que aqui se clasifican usando lo que sabemos de la geometria real de la pista:

  - circulo central: tiene un punto pequeno concentrico y la linea roja del
    centro del campo pasa cerca de su centro.
  - circulos laterales (de faceoff): tienen un punto pequeno concentrico y
    marcas en forma de L (aqui, segmentos cortos) cerca de su borde.

Todo esto solo tiene sentido asumiendo la vista de camara principal (fija,
gran angulo), asi que el clasificador solo corre cuando `is_main_camera` (el
mismo criterio HSV de test.py) da True; en otros planos la geometria no se
sostiene y no vale la pena ni intentarlo.

Requiere opencv-contrib-python (ya instalado); ximgproc/EdgeDrawing no esta en
el opencv-python base.

Ademas se estima por donde de la pista pasa la camara juntando los tramos de
la linea (muy discontinua: se corta con jugadores, publicidad, etc.) donde el
hielo se encuentra con la valla: es casi horizontal en toda la pista pero su
inclinacion cambia con el pan de la camara. Si el lado derecho de esa linea
queda mas arriba en la imagen (angulo negativo) es porque la camara mira hacia
el lado derecho de la pista; si es el izquierdo el que queda mas arriba, mira
al lado izquierdo; si esta practicamente plana, mira al centro.

Tambien se buscan, junto a cada circulo LATERAL, la linea de gol (mas cerca
del circulo) y la linea azul (mas lejos, hacia el circulo central) usando las
proporciones reales de rink.py: distancia punto->gol y punto->azul, ambas en
funcion del radio del circulo detectado. Combinando esto con el circulo
CENTRO/LATERAL y el lado de la linea hielo/valla se estima la zona de pista
que muestra la camara (neutral, o zona final si se ve la linea de gol).

Esta zona es inherentemente temporal: en un frame suelto casi nunca se ven
todas las senales a la vez (un jugador tapa el circulo, la linea de gol se
corta, etc.), pero dentro de un mismo plano la camara no salta de zona de
golpe. Por eso se guarda un estado (`ZoneState`) que acumula un voto por
frame dentro del plano actual y muestra la mayoria de una ventana movil, en
vez de depender del frame suelto; el estado se reinicia solo al detectar un
corte de plano real (mismo `ContentDetector` que test.py).

Controles:
  espacio  pausa / reanuda
  n        avanza un frame (estando en pausa)
  l        muestra/oculta las lineas
  e        muestra/oculta las elipses sin clasificar (ruido considerado)
  r        muestra/oculta los circulos grandes rechazados (ni centro ni lateral)
  b        muestra/oculta la estimacion de la linea hielo/valla
  z        muestra/oculta la busqueda de linea de gol / linea azul
  +/-      sube/baja el tamano minimo de elipse considerada "grande"
  [/]      baja/sube el umbral de "plano" (grados) para la linea hielo/valla
  q / ESC  salir

Uso:
  python rink_edges_debug.py
  python rink_edges_debug.py --start 3300 --scale 2
"""
import argparse
import time
from collections import Counter

import cv2
import numpy as np
from scenedetect.common import FrameTimecode
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import compute_downscale_factor

from sportcal.sports.hockey import rink

# mismos rangos y logica que test.py (derivados de hsv_scan.csv sobre este clip)
MAIN_CAMERA_H_RANGE = (120, 126)
MAIN_CAMERA_S_RANGE = (11, 12)
MAIN_CAMERA_V_RANGE = (150, 217)


def is_main_camera(h_med, s_med, v_med):
    return (
        MAIN_CAMERA_H_RANGE[0] <= h_med <= MAIN_CAMERA_H_RANGE[1]
        and MAIN_CAMERA_S_RANGE[0] <= s_med <= MAIN_CAMERA_S_RANGE[1]
        and MAIN_CAMERA_V_RANGE[0] <= v_med <= MAIN_CAMERA_V_RANGE[1]
    )


def point_segment_distance(p, a, b):
    ab = b - a
    t = float(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9))
    t = max(0.0, min(1.0, t))
    return np.linalg.norm(p - (a + t * ab))


def classify_circles(
    ellipses, lines, scale, min_axis_px,
    dot_ratio=(0.03, 0.22), dot_center_tol=0.20,
    hash_len_ratio=(0.04, 0.35), hash_radial_tol=0.15,
    center_line_tol=0.12, center_line_min_len_ratio=0.8,
):
    """Etiqueta cada elipse grande como 'CENTRO', 'LATERAL' o None (ruido).

    IMPORTANTE, verificado a mano sobre varios frames reales:
    - Las marcas ("hash marks") de los circulos de faceoff casi no se ven para
      EdgeDrawing en esta calidad de retransmision (contraste muy bajo contra
      el hielo); lo que el filtro por posicion/longitud capturaba en la
      practica era texto de la publicidad de la valla (Glidder/PPG), no
      marcas reales. hash_hits por tanto NO es una senal fiable todavia.
    - La linea roja central si es una senal real y fuerte cuando aparece (en
      frame 500 pasa a 16px del centro de un circulo de radio 352px), pero es
      intermitente: en una muestra de 7 frames solo se detecto en 2, porque
      jugadores/graficos rompen el segmento el resto del tiempo.
    - El punto pequeno concentrico casi nunca se detecta limpio: en el hielo
      suele estar tapado por el logo del equipo pintado en el centro.
    Con esto: CENTRO se decide solo por la linea que cruza (sin exigir el
    punto), y LATERAL queda como hipotesis debil (dot+hash) a falta de una
    senal mejor; en la practica raramente disparara.
    """
    if ellipses is None or len(ellipses) == 0:
        return []

    circles = []
    for cx, cy, r, a, b, angle in ellipses:
        ra, rb = (r, r) if r > 0 else (a, b)
        circles.append({
            "center": np.array([cx, cy]) * scale,
            "radius": max(ra, rb) * scale,
        })

    segments = []
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            p1 = np.array([x1, y1]) * scale
            p2 = np.array([x2, y2]) * scale
            segments.append((p1, p2, float(np.linalg.norm(p2 - p1))))

    results = []
    for c in circles:
        center, radius = c["center"], c["radius"]
        if radius * 2 < min_axis_px:
            continue  # muy chico para ser un circulo real de la pista

        has_dot = any(
            d is not c
            and np.linalg.norm(d["center"] - center) < radius * dot_center_tol
            and dot_ratio[0] <= d["radius"] / radius <= dot_ratio[1]
            for d in circles
        )

        hash_hits = sum(
            1 for p1, p2, length in segments
            if radius * hash_len_ratio[0] <= length <= radius * hash_len_ratio[1]
            and abs(np.linalg.norm((p1 + p2) / 2 - center) - radius) <= radius * hash_radial_tol
        )

        crossing_line = any(
            length >= radius * center_line_min_len_ratio
            and point_segment_distance(center, p1, p2) <= radius * center_line_tol
            for p1, p2, length in segments
        )

        if crossing_line:
            kind = "CENTRO"
        elif has_dot and hash_hits >= 2:
            kind = "LATERAL"
        else:
            kind = None
        results.append({"center": center, "radius": radius, "has_dot": has_dot,
                         "hash_hits": hash_hits, "crossing_line": crossing_line, "kind": kind})
    return results


def estimate_board_line(lines, scale, angle_max_deg=40.0, min_len_px=25.0):
    """Ajusta una recta por minimos cuadrados ponderados a los tramos casi
    horizontales detectados: es la union de la linea, muy discontinua, donde
    el hielo se encuentra con la valla/publico.

    Devuelve None si no hay suficientes tramos, o si no, (m, b, segs) donde
    y = m*x + b (en pixeles a resolucion completa) y segs son los tramos
    [(p1, p2, length), ...] que finalmente entraron en el ajuste (tras
    descartar una vez los que quedan lejos de la recta: texto de publicidad,
    reflejos, etc. que pasaron el filtro de angulo/longitud por casualidad).
    """
    if lines is None or len(lines) == 0:
        return None

    segs = []
    for x1, y1, x2, y2 in lines:
        p1 = np.array([x1, y1]) * scale
        p2 = np.array([x2, y2]) * scale
        dx, dy = p2 - p1
        length = float(np.hypot(dx, dy))
        if length < min_len_px:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        angle = ((angle + 90) % 180) - 90  # normaliza a [-90, 90)
        if abs(angle) > angle_max_deg:
            continue
        segs.append((p1, p2, length))

    if len(segs) < 3:
        return None

    def weighted_fit(segs):
        xs, ys, ws = [], [], []
        for p1, p2, length in segs:
            xs += [p1[0], p2[0]]
            ys += [p1[1], p2[1]]
            ws += [length, length]
        xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
        sw = ws.sum()
        sx, sy = (ws * xs).sum(), (ws * ys).sum()
        sxx, sxy = (ws * xs * xs).sum(), (ws * xs * ys).sum()
        denom = sw * sxx - sx * sx
        if abs(denom) < 1e-6:
            return None
        m = (sw * sxy - sx * sy) / denom
        b = (sy - m * sx) / sw
        return m, b

    fit = weighted_fit(segs)
    if fit is None:
        return None
    m, b = fit

    residuals = [abs((p1[1] + p2[1]) / 2 - (m * (p1[0] + p2[0]) / 2 + b)) for p1, p2, _ in segs]
    thresh = max(15.0, float(np.median(residuals)) * 2.5)
    inliers = [s for s, r in zip(segs, residuals) if r <= thresh]
    if len(inliers) >= 3:
        fit2 = weighted_fit(inliers)
        if fit2 is not None:
            m, b = fit2
            segs = inliers

    return m, b, segs


def classify_board_side(m, flat_threshold_deg=2.0):
    angle_deg = float(np.degrees(np.arctan(m)))
    if angle_deg < -flat_threshold_deg:
        return "LADO DERECHO", angle_deg
    if angle_deg > flat_threshold_deg:
        return "LADO IZQUIERDO", angle_deg
    return "CENTRO", angle_deg


def find_zone_lines(lines, scale, lateral_circles, rink_params,
                     angle_min_deg=55.0, min_len_px=20.0, match_tol=0.35):
    """Para cada circulo LATERAL, busca la linea de gol (mas cerca del
    circulo) y la linea azul (mas lejos, hacia el circulo central) entre los
    tramos casi verticales detectados.

    La distancia real punto de faceoff -> linea de gol y -> linea azul se
    conoce (rink.py), asi que se pasa a proporcion del radio del circulo
    detectado (que si conocemos en pixeles) y se busca un tramo casi vertical
    cuya posicion horizontal respecto al centro del circulo encaje con esa
    proporcion, dentro de `match_tol` (fraccion relativa de error admitida).

    Devuelve una lista, una entrada por circulo LATERAL:
    [{"circle": c, "goal_line": (p1,p2,len) o None, "blue_line": ... o None}]
    """
    if not lateral_circles:
        return []

    ratio_goal = rink_params["dot_from_goal_line"] / rink_params["circle_r"]
    ratio_blue = (
        rink_params["blue_from_end"] - rink_params["goal_line_from_end"] - rink_params["dot_from_goal_line"]
    ) / rink_params["circle_r"]

    segs = []
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            p1 = np.array([x1, y1]) * scale
            p2 = np.array([x2, y2]) * scale
            dx, dy = p2 - p1
            length = float(np.hypot(dx, dy))
            if length < min_len_px:
                continue
            angle_from_horiz = abs(((np.degrees(np.arctan2(dy, dx)) + 90) % 180) - 90)
            if angle_from_horiz < angle_min_deg:
                continue  # muy horizontal para ser gol/azul (vertical en la imagen)
            segs.append((p1, p2, length))

    results = []
    for circle in lateral_circles:
        center, radius = circle["center"], circle["radius"]
        best: dict = {"goal_line": None, "blue_line": None}
        best_err = {"goal_line": float("inf"), "blue_line": float("inf")}
        for p1, p2, length in segs:
            ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
            if ymax < center[1] - radius * 2 or ymin > center[1] + radius * 2:
                continue  # no pasa a la altura del circulo
            mid_x = (p1[0] + p2[0]) / 2
            dx_ratio = abs(mid_x - center[0]) / radius
            for kind, ratio in (("goal_line", ratio_goal), ("blue_line", ratio_blue)):
                err = abs(dx_ratio - ratio) / ratio
                if err < match_tol and err < best_err[kind]:
                    best_err[kind] = err
                    best[kind] = (p1, p2, length)
        results.append({"circle": circle, **best})
    return results


def determine_zone(circles, zone_lines, board_side):
    """Combina circulo CENTRO/LATERAL + linea de gol/azul + lado de la valla
    en una etiqueta de zona. Devuelve None si no hay senal suficiente en el
    frame (para que el estado temporal no se contamine con "sin dato").
    """
    if any(c["kind"] == "CENTRO" for c in circles):
        base = "ZONA NEUTRAL"
    elif any(zl["goal_line"] is not None for zl in zone_lines):
        base = "ZONA FINAL"
    elif any(zl["blue_line"] is not None for zl in zone_lines):
        base = "ZONA NEUTRAL/FINAL"
    else:
        return None
    return f"{base} - {board_side}" if board_side else base


class ZoneState:
    """Voto por mayoria de la zona en una ventana movil de frames del plano
    actual. La zona de camara es temporal por naturaleza (depende del frame
    anterior, no solo del actual): en un frame suelto casi nunca se ven todas
    las senales a la vez, pero dentro de un mismo plano no cambia de golpe.
    Se reinicia solo en corte de plano real, no cuando un frame no aporta dato.
    """

    def __init__(self, window=45):
        self.window = window
        self.votes = []

    def reset(self):
        self.votes = []

    def update(self, zone_label):
        if zone_label is None:
            return
        self.votes.append(zone_label)
        if len(self.votes) > self.window:
            self.votes.pop(0)

    def current(self):
        if not self.votes:
            return "SIN DATOS (aun)"
        return Counter(self.votes).most_common(1)[0][0]


parser = argparse.ArgumentParser()
parser.add_argument("--video", default="clip.mp4.webm")
parser.add_argument("--start", type=int, default=0, help="frame inicial")
parser.add_argument("--scale", type=int, default=2, help="downscale para EdgeDrawing (1, 2 o 4); mas rapido a mas escala")
parser.add_argument("--min-line-len", type=int, default=30)
parser.add_argument("--min-path-len", type=int, default=50)
parser.add_argument("--min-ellipse-axis", type=int, default=25, help="eje minimo (px, ya en resolucion completa) para no contar ruido pequeno")
parser.add_argument("--board-angle-max", type=float, default=40.0, help="grados max. respecto a horizontal para considerar un tramo parte de la linea hielo/valla")
parser.add_argument("--board-min-len", type=float, default=25.0, help="longitud minima (px) de un tramo para entrar en el ajuste de la linea hielo/valla")
parser.add_argument("--board-flat-deg", type=float, default=2.0, help="umbral (grados) por debajo del cual se considera CENTRO")
parser.add_argument("--zone-match-tol", type=float, default=0.35, help="error relativo max. al emparejar linea de gol/azul con un circulo LATERAL")
parser.add_argument("--zone-window", type=int, default=45, help="tamano de la ventana movil (frames) para el voto de zona")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise IOError(f"no se pudo abrir {args.video}")
cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# downscale para la clasificacion de camara, igual de agresivo que en test.py
camera_downscale = compute_downscale_factor(max(frame_width, frame_height))
camera_size = (round(frame_width / camera_downscale), round(frame_height / camera_downscale))

# corte de plano real (mismo detector que test.py): resetea el voto de zona,
# que de lo contrario arrastraria estado de un plano completamente distinto
scene_detector = ContentDetector()
zone_state = ZoneState(window=args.zone_window)

ed = cv2.ximgproc.createEdgeDrawing()
ed_params = cv2.ximgproc_EdgeDrawing_Params()
ed_params.MinLineLength = args.min_line_len
ed_params.MinPathLength = args.min_path_len
ed.setParams(ed_params)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

show_lines = True
show_noise_ellipses = True
show_rejected_big = False
show_board_line = True
show_zone_lines = True
min_axis = args.min_ellipse_axis
board_flat_deg = args.board_flat_deg
paused = False
frame_idx = args.start

WINDOW = "edgedrawing debug"
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

    small_cam = cv2.resize(frame, camera_size, interpolation=cv2.INTER_LINEAR)
    cuts = scene_detector.process_frame(FrameTimecode(frame_idx, fps=fps), small_cam)
    if cuts:
        zone_state.reset()

    hsv = cv2.cvtColor(small_cam, cv2.COLOR_BGR2HSV)
    h_med, s_med, v_med = (int(np.median(hsv[:, :, c])) for c in range(3))
    main_camera = is_main_camera(h_med, s_med, v_med)

    scale = args.scale
    small = cv2.resize(frame, (frame.shape[1] // scale, frame.shape[0] // scale)) if scale != 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    t0 = time.perf_counter()
    ed.detectEdges(gray)
    lines = ed.detectLines()
    ellipses = ed.detectEllipses()
    t1 = time.perf_counter()

    circles = classify_circles(ellipses, lines, scale, min_axis) if main_camera else []
    board_fit = (
        estimate_board_line(lines, scale, args.board_angle_max, args.board_min_len)
        if main_camera and show_board_line else None
    )
    board_side, board_angle_deg = (None, None)
    if board_fit is not None:
        board_side, board_angle_deg = classify_board_side(board_fit[0], board_flat_deg)

    lateral_circles = [c for c in circles if c["kind"] == "LATERAL"]
    zone_lines = (
        find_zone_lines(lines, scale, lateral_circles, rink.RINK_NHL_FITTED, match_tol=args.zone_match_tol)
        if main_camera and show_zone_lines else []
    )
    zone_label = determine_zone(circles, zone_lines, board_side) if main_camera else None
    zone_state.update(zone_label)

    vis = frame.copy()
    n_lines = 0 if lines is None else len(lines)

    if show_lines and lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(vis, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), (0, 255, 255), 1)

    if show_noise_ellipses and ellipses is not None:
        for cx, cy, r, a, b, angle in ellipses:
            axes = (r, r) if r > 0 else (a, b)
            if max(axes) * scale * 2 >= min_axis:
                continue  # esos se dibujan mas abajo, clasificados o rechazados
            cv2.ellipse(vis, (int(cx * scale), int(cy * scale)), (int(axes[0] * scale), int(axes[1] * scale)),
                        0.0 if r > 0 else angle, 0, 360, (120, 120, 120), 1)

    n_centro = n_lateral = n_rejected = 0
    for c in circles:
        center = tuple(c["center"].astype(int))
        radius = int(c["radius"])
        if c["kind"] == "CENTRO":
            n_centro += 1
            cv2.circle(vis, center, radius, (255, 0, 255), 4)
            cv2.putText(vis, "CENTRO", (center[0] - 40, center[1] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        elif c["kind"] == "LATERAL":
            n_lateral += 1
            cv2.circle(vis, center, radius, (0, 255, 0), 4)
            cv2.putText(vis, f"LATERAL h={c['hash_hits']}", (center[0] - 60, center[1] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            n_rejected += 1
            if show_rejected_big:
                cv2.circle(vis, center, radius, (0, 0, 255), 1)
                cv2.putText(vis, f"dot={c['has_dot']} h={c['hash_hits']} line={c['crossing_line']}",
                            (center[0] - 80, center[1] - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    board_label = "sin datos"
    if board_fit is not None:
        m, b, board_segs = board_fit
        for p1, p2, _ in board_segs:
            cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 128, 255), 2)
        x0, x1 = 0, frame_width - 1
        y0, y1 = int(m * x0 + b), int(m * x1 + b)
        cv2.line(vis, (x0, y0), (x1, y1), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{board_side} ({board_angle_deg:+.1f} grados)", (x1 - 340, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        board_label = f"{board_side} ({board_angle_deg:+.1f} grados, {len(board_segs)} tramos)"
    elif not (main_camera and show_board_line):
        board_label = "n/a" if not main_camera else "oculta (b)"

    for zl in zone_lines:
        center = tuple(zl["circle"]["center"].astype(int))
        if zl["goal_line"] is not None:
            p1, p2, _ = zl["goal_line"]
            cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 0, 255), 3)
            cv2.putText(vis, "GOL", (int(p1[0]) + 6, int(p1[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if zl["blue_line"] is not None:
            p1, p2, _ = zl["blue_line"]
            cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 120, 0), 3)
            cv2.putText(vis, "AZUL", (int(p1[0]) + 6, int(p1[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

    camera_label = "CAMARA PRINCIPAL" if main_camera else "OTRO PLANO"
    camera_color = (0, 200, 0) if main_camera else (0, 0, 200)
    hud = [
        f"frame {frame_idx}/{total_frames}  {'PAUSA' if paused else ''}  scale=1/{scale}",
        f"{camera_label}  (h={h_med} s={s_med} v={v_med})",
        f"deteccion: {(t1 - t0) * 1000:.1f} ms  |  lineas: {n_lines if show_lines else 'ocultas (l)'}",
        f"circulos -> centro:{n_centro}  lateral:{n_lateral}  rechazados:{n_rejected}"
        + ("" if show_rejected_big else " (ocultos, r)") + f"  min_eje>={min_axis}px (+/-)",
        f"linea hielo/valla: {board_label}  plano<{board_flat_deg:.1f} grados ([ ])",
        f"zona (voto x{len(zone_state.votes)}/{zone_state.window}): {zone_state.current()}"
        + ("" if show_zone_lines else "  (busqueda gol/azul oculta, z)"),
    ]
    for i, line in enumerate(hud):
        color = camera_color if i == 1 else (255, 255, 255)
        cv2.putText(vis, line, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, line, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    cv2.imshow(WINDOW, vis)
    key = cv2.waitKey(1 if not paused else 0) & 0xFF
    if key in (ord("q"), 27):
        break
    elif key == ord(" "):
        paused = not paused
    elif key == ord("n"):
        paused = True
        frame = None
    elif key == ord("l"):
        show_lines = not show_lines
    elif key == ord("e"):
        show_noise_ellipses = not show_noise_ellipses
    elif key == ord("r"):
        show_rejected_big = not show_rejected_big
    elif key == ord("b"):
        show_board_line = not show_board_line
    elif key == ord("z"):
        show_zone_lines = not show_zone_lines
    elif key in (ord("+"), ord("=")):
        min_axis += 5
    elif key in (ord("-"), ord("_")):
        min_axis = max(0, min_axis - 5)
    elif key == ord("]"):
        board_flat_deg += 0.5
    elif key == ord("["):
        board_flat_deg = max(0.0, board_flat_deg - 0.5)

cap.release()
cv2.destroyAllWindows()
