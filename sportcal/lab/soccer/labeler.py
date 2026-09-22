"""Logica del etiquetador por clics para FUTBOL (sportcal/lab/common/annotate_val_app.py, deporte
"futbol"), sin nada de Streamlit: de N puntos clicados (punto de la plantilla <->
pixel) a una H, mas el guardado. Es el equivalente de sportcal/lab/hockey/click_labeler.py.

Plantilla: los 31 puntos nombrados de soccer_field.py (esquinas, area, area pequeña,
punto de penalti, cruces del circulo y del arco), en metros con el origen en el centro.
Convencion de mundo (la misma que los solvers de soccer_field): +X hacia la derecha de
la imagen, +Y hacia ARRIBA (banda lejana), camara en el lado Y<0 (banda "cercana" abajo).

Simetria: el campo es igual bajo un giro de 180 grados (X,Y -> -X,-Y), asi que dos
etiquetados que difieren en ese giro dibujan las MISMAS lineas y solo cambia que area es
la izquierda y que banda es la cercana. Igual que en hockey se canonicaliza: si la H sale
con +Y hacia abajo (el usuario clico la banda cercana arriba) se aplica el giro, para que
todas las etiquetas guardadas cumplan la convencion de arriba. El umbral de aviso de
inestabilidad (sensibilidad > 30 px) esta HEREDADO de hockey, sin calibrar en futbol.

Salida (datasets/soccer_labels/, o $SOCCER_LABELS_DIR): images/<id>.jpg,
labels/<id>.txt (YOLO-pose de 31 keypoints, PROVISIONAL: la caja es la de los puntos
visibles, no hay pipeline de entrenamiento de futbol todavia), keypoints.csv
(frame_id,keypoint_id,x,y,visible) y clicks.jsonl (clics originales + H canonica).
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from sportcal.core import camera as CAM
from sportcal.core import circle as CIRC
from sportcal.core import fitting as FIT
from sportcal.lab.soccer import field_solver as SF
from sportcal.sports.soccer import field as FIELD

from sportcal.paths import ROOT
OUT = Path(os.environ.get("SOCCER_LABELS_DIR", ROOT / "datasets" / "soccer_labels"))
HX, HY = FIELD.HALF_LENGTH, FIELD.HALF_WIDTH
KEYPOINTS = FIELD.KEYPOINTS  # [(name, (X, Y)), ...]: what a human clicks (defined in sports/soccer/field.py)
N_KP = len(KEYPOINTS)
TPL = FIELD.KEYPOINT_COORDS
NOMBRES = FIELD.KEYPOINT_NAMES
_MUESTRAS, _ = FIELD.sample_template(1.0)      # puntos de las lineas, para comprobar que la plantilla cae en cuadro
_ROT180 = np.diag([-1.0, -1.0, 1.0])


# ---------------------------------------------------------------- minimapa y dibujo

def minimapa(escala=8, margen=30):
    """(imagen BGR, xy (N,2) de cada keypoint en esa imagen). +Y (banda lejana) arriba."""
    w, h = int(2 * HX * escala + 2 * margen), int(2 * HY * escala + 2 * margen)
    img = np.full((h, w, 3), (70, 130, 70), np.uint8)
    f = lambda p: np.array([margen + (p[0] + HX) * escala, margen + (HY - p[1]) * escala])
    for pl in FIELD.polylines().values():
        cv2.polylines(img, [np.round(np.array([f(p) for p in pl])).astype(np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
    return img, np.array([f(p) for p in TPL])


def dibuja_plantilla(vis, H, color, grosor):
    """Pinta las lineas del campo proyectadas por H (mundo -> pixel) sobre `vis` (BGR)."""
    for pl in FIELD.polylines().values():
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
        s = np.arange(0, d[-1], 0.25)
        p = np.stack([np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])], 1)
        xy, den = CAM.project(np.asarray(H, float)[None], p)
        xy, den = xy[0], den[0]
        ok = (den > 1e-3) & np.isfinite(xy).all(1) & (np.abs(xy) < 1e5).all(1)
        for run in np.split(np.arange(len(p)), np.where(~ok)[0]):
            run = run[ok[run]]
            if len(run) > 1:
                cv2.polylines(vis, [np.round(xy[run]).astype(np.int32)], False, color, grosor, cv2.LINE_AA)


# ---------------------------------------------------------------- ajuste

def _proj(H, pts):
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def _en_cuadro(Hn, w, h):
    """Indices de las muestras de linea que Hn deja dentro del frame y delante de la camara."""
    xy, den = CAM.project(Hn[None], _MUESTRAS)
    xy, den = xy[0], den[0]
    ok = (den > 0) & np.isfinite(xy).all(1) & (xy[:, 0] > 0) & (xy[:, 0] < w) & (xy[:, 1] > 0) & (xy[:, 1] < h)
    return np.where(ok)[0]


def sensibilidad(world, img_pts, Hn, w, h, sigma=2.0, n=24):
    """Cuanto se mueve la plantilla DENTRO DEL FRAME (px a 1920, mediana) si cada clic
    se desvia ~sigma px: se reajusta la H sobre clics perturbados y se compara la
    proyeccion de las muestras de linea que caen en cuadro."""
    idx = _en_cuadro(Hn, w, h)
    if len(idx) < 5:
        return float("inf")
    P = _MUESTRAS[idx]
    ref = _proj(Hn, P)
    rng = np.random.default_rng(0)
    sc = w / 1920.0
    d = []
    for _ in range(n):
        noisy = np.asarray(img_pts, np.float64) + rng.normal(0, sigma * sc, np.shape(img_pts))
        Hi, _ = cv2.findHomography(np.asarray(world, np.float32), noisy.astype(np.float32), 0)
        if Hi is None or not np.all(np.isfinite(Hi)):
            return float("inf")
        d.append(np.linalg.norm(_proj(Hi / Hi[2, 2], P) - ref, axis=1) / sc)
    d = np.concatenate(d)
    return float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else float("inf")


def canonicaliza(H):
    """Giro de 180 grados del mundo si la H deja +Y hacia abajo en la imagen."""
    a = _proj(H, np.array([[0.0, 0.0], [0.0, 10.0]]))
    if a[1, 1] > a[0, 1]:
        return H @ _ROT180, True
    return H, False


def ajusta(world, img_pts, w, h=None):
    """(H, residuos_px_a_1920, giro_aplicado, aviso, H_sin_giro) o (None, [], False, motivo, None).

    H es la canonica (la que se guarda y se dibuja); H_sin_giro es la que corresponde al
    etiquetado del usuario: proyectar el punto k con ELLA da donde el usuario espera ver
    el punto k. Minimos cuadrados sobre todos los clics (sin RANSAC: cada clic es una
    decision humana; el residuo por punto se ensena para corregir el que falle).
    world: (n,2) metros; img_pts: (n,2) pixeles de la imagen ORIGINAL."""
    world = np.asarray(world, np.float64)
    img_pts = np.asarray(img_pts, np.float64)
    n = len(world)
    if n < 4:
        return None, [], False, "hacen falta al menos 4 puntos (hay {})".format(n), None
    sv = np.linalg.svd(world - world.mean(0), compute_uv=False)
    if sv[1] < 0.01 * sv[0]:
        return None, [], False, "puntos casi alineados en el mundo: elige alguno fuera de esa recta", None
    H, _ = cv2.findHomography(world.astype(np.float32), img_pts.astype(np.float32), 0)
    if H is None or not np.all(np.isfinite(H)):
        return None, [], False, "no se pudo ajustar la homografia", None
    Hn = H / H[2, 2]
    if np.abs(Hn).max() > 1e6:
        return None, [], False, "homografia degenerada (puntos mal repartidos)", None
    if h is not None and len(_en_cuadro(Hn, w, h)) < 10:
        return None, [], False, "con esos puntos casi todo el campo cae fuera del frame: revisa que no haya un punto mal asignado", None
    res = np.linalg.norm(_proj(Hn, world) - img_pts, axis=1) * 1920.0 / w
    Hc, giro = canonicaliza(Hn)
    aviso = None
    sens = sensibilidad(world, img_pts, Hn, w, h if h else w * 9 / 16)
    esperado = 0.42 * sens
    if sens > 30.0:
        aviso = ("Ajuste poco estable: con clics a ±2px la plantilla puede desviarse ~{:.0f}px dentro del "
                 "frame. No bloquea; se corrige añadiendo o arrastrando mas puntos (mejor si no estan "
                 "todos pegados).".format(esperado if np.isfinite(esperado) else 999))
    elif n == 4:
        aviso = ("con 4 puntos el ajuste es exacto (residuo 0); error esperado ~{:.0f}px: "
                 "revisa el dibujo o añade un 5o punto".format(esperado))
    return Hc, [float(r) for r in res], giro, aviso, Hn


def ajusta_elipse(world, img_pts, elipse_pts, w, h=None):
    """Como `ajusta`, pero ademas el contorno clicado del CIRCULO CENTRAL (>= 5 puntos sobre la elipse).

    La elipse fija 5 de los 8 grados de libertad de H, asi que bastan 2 puntos clicados (p. ej. el centro de campo y un
    cruce circulo x linea central); con mas puntos la elipse solo refina. Devuelve lo mismo que `ajusta`; el aviso dice el
    error de la elipse y si el ajuste es ambiguo. Sin minimos cuadrados de puntos solos: la elipse pesa como un punto mas."""
    world = np.asarray(world, np.float64)
    img_pts = np.asarray(img_pts, np.float64)
    if len(elipse_pts) < 5:
        return None, [], False, "hacen falta al menos 5 puntos sobre el circulo central (hay {})".format(len(elipse_pts)), None
    if len(world) < 2:
        return None, [], False, "con la elipse hacen falta al menos 2 puntos de la plantilla (p. ej. el centro de campo y un cruce circulo x linea central)", None
    out = CIRC.fit_points_ellipse(world, img_pts, elipse_pts, FIELD.CIRCLE_RADIUS)
    if out is None:
        return None, [], False, "no se pudo ajustar: la elipse no es una elipse valida o los puntos no encajan con ella", None
    Hn = out["H"]
    if h is not None and len(_en_cuadro(Hn, w, h)) < 10:
        return None, [], False, "con esa elipse y esos puntos casi todo el campo cae fuera del frame: revisa los puntos", None
    res = np.linalg.norm(_proj(Hn, world) - img_pts, axis=1) * 1920.0 / w
    Hc, giro = canonicaliza(Hn)
    aviso = "elipse: error medio {:.1f}px a 1920".format(out["rms_ellipse"] * 1920.0 / w)
    if out["n_solutions"] > 1:
        aviso = ("AMBIGUO: {} homografias distintas encajan igual con estos clics; anade un punto que no este sobre la "
                 "linea central (una esquina o el punto de penalti). ".format(out["n_solutions"])) + aviso
    return Hc, [float(r) for r in res], giro, aviso, Hn


def procesa_gesto_elipse(pts, g, radio, tol_click):
    """Gesto sobre el frame (px originales, g = (x1, y1, x2, y2)) -> nueva lista de puntos del contorno de la elipse.

    Clic sobre un punto existente (a <= radio): lo quita. Clic en vacio: anade un punto. Arrastre que EMPIEZA sobre un
    punto: lo mueve a donde se suelta. Arrastre desde un sitio vacio: no hace nada."""
    x1, y1, x2, y2 = g
    pts = [tuple(p) for p in pts]
    d = [float(np.hypot(p[0] - x1, p[1] - y1)) for p in pts]
    near = int(np.argmin(d)) if d and min(d) <= radio else None
    arrastre = float(np.hypot(x2 - x1, y2 - y1)) > tol_click
    if arrastre:
        if near is not None:
            pts[near] = (x2, y2)
        return pts
    if near is not None:
        pts.pop(near)
    else:
        pts.append((x2, y2))
    return pts


def elipse_desde_plantilla(H, w, h, n=12):
    """n puntos sobre el circulo central proyectado por H (mundo -> pixel) que caen en cuadro: la semilla que el usuario
    arrastra a la elipse real. Lista vacia si H no ve el circulo."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.c_[FIELD.CIRCLE_RADIUS * np.cos(a), FIELD.CIRCLE_RADIUS * np.sin(a)]
    xy, den = CAM.project(np.asarray(H, float)[None], P)
    xy, den = xy[0], den[0]
    ok = (den > 1e-3) & np.isfinite(xy).all(1) & (xy[:, 0] > 0) & (xy[:, 0] < w) & (xy[:, 1] > 0) & (xy[:, 1] < h)
    return [(float(x), float(y)) for (x, y), o in zip(xy, ok) if o]


def dibuja_elipse(vis, pts, color=(255, 0, 255), grosor=2):
    """Pinta los puntos del contorno y, con >= 5, la elipse ajustada (cv2.fitEllipse) sobre `vis` (BGR)."""
    r = max(5, vis.shape[1] // 260)
    for k, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), r, color, grosor)
    if len(pts) >= 5:
        try:
            cv2.ellipse(vis, cv2.fitEllipse(np.asarray(pts, np.float32).reshape(-1, 1, 2)), color, grosor, cv2.LINE_AA)
        except cv2.error:
            pass


def proyecta_pendientes(H_sin_giro, pts, w, h, margen=4):
    """{k: (x,y)} de los keypoints AUN sin clicar que H situa dentro del frame y delante de
    la camara -- los que el usuario puede arrastrar a su sitio real."""
    xy, den = CAM.project(np.asarray(H_sin_giro, float)[None], TPL)
    xy, den = xy[0], den[0]
    out = {}
    for k in range(N_KP):
        if k in pts or den[k] <= 0 or not np.isfinite(xy[k]).all():
            continue
        x, y = float(xy[k, 0]), float(xy[k, 1])
        if margen <= x <= w - margen and margen <= y <= h - margen:
            out[k] = (x, y)
    return out


# ---------------------------------------------------------------- sugerencia automatica

def sugiere(frame_bgr, modelo=None, det_elipse="auto", center=None):
    """Candidatas de H del solver de soccer_field para este frame, en pixeles del frame ORIGINAL:
    [{"via","score","H","nota"}] mejor primero. Vias: intersecciones (rectas de la mascara), chamfer
    (rejilla de poses) y, si se pasa `modelo` (YOLO de personas, en CPU), la ELIPSE del circulo detectada
    por gradiente dentro del campo (hipotesis por pose pinhole) y las ESQUINAS como correspondencias
    de punto; la elipse y las esquinas entran ademas en la puntuacion (score = Dice + 0.6*elipse +
    0.4*esquinas, asi que con evidencia la escala llega a ~2). H sale canonica: banda cercana abajo.
    det_elipse: que detector de elipses alimenta al solver ("auto", "edgedrawing", "find" o "fit", ver
    soccer_grad.procesa). Tarda ~15 s sin modelo, ~40 s con el. Un plano cercano da puntuaciones bajas: no fiarse.
    center: centro fijo de la camara (x, y, z) en metros, calibrado con `camera_center`; si se da, se anade la via
    "centro fijo" (solo paneo, inclinacion y zoom: 3 incognitas en vez de 8).
    OJO: la puntuacion de una candidata con evidencia incluye el termino de elipse, asi que NO es comparable con la
    de otra que se ajusta peor a una elipse imprecisa; mirar el dibujo, no solo el numero."""
    from sportcal.lab.soccer import evaluation as EV
    from sportcal.lab.soccer import gradient as G
    h, w = frame_bgr.shape[:2]
    lineas = EV.etapas_mascara(frame_bgr)["lineas"]
    campo = SF.FieldSolver(lineas)
    E = Q = None
    nota = "solo mascara"
    if modelo is not None:
        res = G.procesa(frame_bgr, None, resp=G.respuesta(frame_bgr), cajas=G.personas(frame_bgr, modelo), det_elipse=det_elipse)
        E = res["elipse"]["pts"] if res["elipse"] is not None else None
        Q = res["esquinas"] if len(res["esquinas"]) else None
        campo.set_evidence(E, Q)
        nota = "mascara + {} + {}".format("elipse ({})".format(res["elipse"]["origen"]) if E is not None else "sin elipse",
                                          "{} esquinas".format(len(Q)) if Q is not None else "sin esquinas")
    S = np.diag([w / campo.w, w / campo.w, 1.0])          # pixel de trabajo -> pixel original
    vias = [("intersecciones", campo.search_lines()), ("chamfer", campo.search_pose())]
    if E is not None:
        vias.append(("elipse", campo.search_ellipse(E)))
        if center is not None:
            vias.append(("elipse+centro fijo", campo.search_ellipse_fixed_center(center, E)))
    if center is not None:
        vias.append(("centro fijo", campo.search_fixed_center(center)))
    out = []
    for via, r in vias:
        if not r:
            continue
        H0, sc = r[0]["H"], r[0]["score"]
        if Q is not None:                                    # esquinas: solo se acepta el reajuste si no empeora
            Hs, npar = campo.snap_corners(H0, Q)
            if npar and campo.score(Hs[None], True)[0] >= sc:
                H0, sc, via = Hs, campo.score(Hs[None], True)[0], via + "+esquinas"
        out.append({"via": via, "score": float(sc), "H": S @ FIELD.canonicalize_H(H0), "nota": nota})
    return sorted(out, key=lambda c: -c["score"])


# ---------------------------------------------------------------- procesado visible

def etapas(frame_bgr, elong_max=8.0):
    """Etapas de la mascara de lineas de este frame (ver soccer_eval.etapas_mascara),
    a 960 px de ancho. ~1-2 s. elong_max=None: regla antigua de jugadores."""
    from sportcal.lab.soccer import evaluation as EV
    return EV.etapas_mascara(frame_bgr, elong_max=elong_max)


def _tinte(base, mask, bgr, alpha=0.55):
    out = base.copy()
    out[mask > 0] = (base[mask > 0] * (1 - alpha) + np.array(bgr) * alpha).astype(np.uint8)
    return out


def _dibuja_recta(vis, r, color, etiqueta):
    """Recta a*x+b*y+c=0 recortada al frame, con su numero en el punto medio."""
    h, w = vis.shape[:2]
    a, b, c = r
    p0 = np.array([-a * c, -b * c])            # punto de la recta mas cercano al origen
    d = np.array([-b, a])
    ok, q1, q2 = cv2.clipLine((0, 0, w, h), tuple(int(v) for v in p0 - d * 5000), tuple(int(v) for v in p0 + d * 5000))
    if ok:
        cv2.line(vis, q1, q2, color, 2, cv2.LINE_AA)
        cv2.putText(vis, etiqueta, ((q1[0] + q2[0]) // 2 + 6, (q1[1] + q2[1]) // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _gradientes(mask, sigma_g=1.5, sigma_t=3.0):
    """(|dI/dx|, |dI/dy|, desviacion_h, energia) de la mascara suavizada. desviacion_h: angulo
    (0-90 grados) entre la DIRECCION DE LA LINEA y la horizontal, por tensor de estructura
    (la linea es perpendicular al gradiente): 0 = horizontal, 90 = vertical."""
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma_g)
    ix = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), sigma_t) for v in (ix * ix, iy * iy, ix * iy))
    grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)                 # direccion del gradiente (mod pi)
    phi = (grad + np.pi / 2) % np.pi                            # direccion de la linea
    return np.abs(ix), np.abs(iy), np.degrees(np.minimum(phi, np.pi - phi)), jxx + jyy


def paneles(et, H_nativa=None, etiqueta_H="plantilla", w_nativo=None, tol_deg=30.0):
    """[(titulo, imagen_rgb, texto)] con cada paso del procesado, para ensenarlo bajo el
    resultado. H_nativa: una H en pixeles del frame ORIGINAL (la de clics o una sugerencia)
    para dibujar la plantilla sobre la mascara final. tol_deg: tolerancia angular de las
    familias horizontal/vertical (el resto es oblicuo: con la perspectiva muchas lineas
    del campo salen a 20-60 grados)."""
    im = et["im"]
    h, w = im.shape[:2]
    pct = lambda m: "{:.2f}% del frame".format(100.0 * float(np.mean(m > 0)))
    rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    negro = lambda: np.zeros_like(im)
    out = []

    out.append(("Region de cesped (gaussiana robusta)", rgb(_tinte(im, et["region"], (120, 255, 0))),
                pct(et["region"]) + ". Todo lo de fuera se ignora."))

    f = negro().astype(np.int32)
    f[et["a_pos"] > 0] += (60, 60, 255)      # a+ rojo
    f[et["b_neg"] > 0] += (255, 100, 60)     # b- azul
    f[et["l_pos"] > 0] += (255, 255, 255)    # L+ blanco
    out.append(("Filtros de color (a+ rojo · b- azul · L+ blanco)", rgb(np.clip(f, 0, 255).astype(np.uint8)),
                "a+ {} · b- {} · L+ {} (donde se solapan se suman los colores)".format(
                    pct(et["a_pos"]), pct(et["b_neg"]), pct(et["l_pos"]))))

    out.append(("Mascara binaria tras los filtros (OR)", rgb(cv2.cvtColor(et["todo"] * 255, cv2.COLOR_GRAY2BGR)),
                pct(et["todo"]) + ". Lineas + jugadores + ruido."))

    ab = cv2.cvtColor(et["abiertos"] * 255, cv2.COLOR_GRAY2BGR)
    ab[et["vertical"] > 0] = (60, 60, 255)
    ab[et["alargados"] > 0] = (0, 165, 255)
    out.append(("Tras la apertura por grosor (rojo = jugador · naranja = grueso y muy alargado, se conserva)", rgb(ab),
                "Sobrevive lo mas ancho que el kernel; las lineas finas desaparecen. Rojo: compacto y vertical -> jugador. "
                "Naranja: tambien vertical pero muy alargado (linea central, fondos gruesos por perspectiva) -> se queda. "
                "Blanco: ancho y no vertical, se queda."))

    quitado = (et["todo"] > 0) & (et["jugadores"] > 0)
    v5 = _tinte(im, quitado, (60, 60, 255), 0.75)
    v5 = _tinte(v5, et["lineas"], (0, 255, 255), 0.75)
    out.append(("Descartado como jugador (rojo) y lo que se queda como linea (amarillo)", rgb(v5),
                "Se quitan {} px de mascara ({:.0f}% de la mascara previa). Si una linea sale roja, la regla se la comio.".format(
                    int(quitado.sum()), 100.0 * quitado.sum() / max(1, int((et["todo"] > 0).sum())))))

    rectas = FIT.detect_lines(et["lineas"], 7)
    v6 = cv2.cvtColor(et["lineas"] * 255, cv2.COLOR_GRAY2BGR)
    paleta = [(0, 200, 255), (255, 160, 0), (0, 255, 0), (255, 0, 255), (60, 60, 255), (255, 255, 0), (200, 120, 255)]
    for n, r in enumerate(rectas):
        _dibuja_recta(v6, r, paleta[n % len(paleta)], str(n + 1))
    out.append(("Mascara final de lineas + rectas detectadas (Hough)", rgb(v6),
                "{}. {} rectas (numeradas por longitud): son las que el solver de intersecciones asigna a lineas del campo.".format(
                    pct(et["lineas"]), len(rectas))))

    gx, gy, dev_h, energia = _gradientes(et["lineas"])

    def esc(g):
        top = float(np.percentile(g[g > 0], 99)) if (g > 0).any() else 1.0
        return np.clip(g / max(top, 1e-6), 0, 1)[..., None]

    out.append(("Gradiente vertical |dI/dy|: bordes de lineas horizontales (bandas, lados del area)",
                rgb((esc(gy) * np.array([255, 255, 0])).astype(np.uint8)),
                "Sobre la mascara final suavizada. Se enciende donde hay un borde horizontal-ish; una linea horizontal da dos "
                "bordes paralelos, uno a cada lado."))
    out.append(("Gradiente horizontal |dI/dx|: bordes de lineas verticales (central, fondos)",
                rgb((esc(gx) * np.array([255, 0, 255])).astype(np.uint8)),
                "Igual, en la otra direccion. Un jugador (mancha vertical) tambien enciende este panel en sus costados: "
                "por eso el gradiente solo NO separa jugadores de lineas."))
    en_mask = (cv2.dilate(et["lineas"], np.ones((3, 3), np.uint8)) > 0) & (energia > 0.05)
    hor, ver = en_mask & (dev_h <= tol_deg), en_mask & (dev_h >= 90 - tol_deg)
    obl = en_mask & ~hor & ~ver
    fam = negro()
    fam[hor], fam[ver], fam[obl] = (255, 255, 0), (255, 0, 255), (0, 255, 255)
    tot = max(1, int(en_mask.sum()))
    out.append(("Familias por orientacion (±{:.0f}°): horizontales cian · verticales magenta · oblicuas amarillo".format(tol_deg),
                rgb(fam),
                "{:.0f}% horizontales · {:.0f}% verticales · {:.0f}% oblicuas. Con la perspectiva muchas lineas salen oblicuas: "
                "sube la tolerancia para agrupar mas.".format(100.0 * hor.sum() / tot, 100.0 * ver.sum() / tot, 100.0 * obl.sum() / tot)))

    if H_nativa is not None and w_nativo:
        s = w / float(w_nativo)
        v7 = cv2.cvtColor(et["lineas"] * 255, cv2.COLOR_GRAY2BGR)
        dibuja_plantilla(v7, np.diag([s, s, 1.0]) @ np.asarray(H_nativa, float), (255, 160, 0), 1)
        out.append(("{} sobre la mascara (azul fino)".format(etiqueta_H), rgb(v7),
                    "Linea azul con blanco a los dos lados = encaja. Azul sin blanco = la mascara no tiene esa linea; "
                    "blanco suelto lejos del azul = mascara que la plantilla no explica (ruido, jugadores, o linea mal colocada)."))
    return [("{} · {}".format(i + 1, t), img_p, tx) for i, (t, img_p, tx) in enumerate(out)]


# ---------------------------------------------------------------- guardado y frames

def _yolo_linea(H, w, h):
    xy, den = CAM.project(np.asarray(H, float)[None], TPL)
    xy, den = xy[0], den[0]
    vis = (den > 0) & np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if vis.sum() < 4:
        return None, xy, vis
    x0, y0 = xy[vis].min(0)
    x1, y1 = xy[vis].max(0)
    x0, y0, x1, y1 = max(0, x0 - 0.02 * w), max(0, y0 - 0.02 * h), min(w, x1 + 0.02 * w), min(h, y1 + 0.02 * h)
    campos = ["0", "{:.6f}".format((x0 + x1) / 2 / w), "{:.6f}".format((y0 + y1) / 2 / h),
              "{:.6f}".format((x1 - x0) / w), "{:.6f}".format((y1 - y0) / h)]
    for k in range(N_KP):
        campos += ["{:.6f}".format(xy[k, 0] / w), "{:.6f}".format(xy[k, 1] / h), "2"] if vis[k] else ["0", "0", "0"]
    return " ".join(campos), xy, vis


def guarda(frame_bgr, cid, H, clics):
    """Escribe imagen + etiqueta YOLO-pose + filas de keypoints.csv + clics en OUT."""
    h, w = frame_bgr.shape[:2]
    linea, xy, vis = _yolo_linea(H, w, h)
    if linea is None:
        return False, "muy pocos keypoints del campo caen en cuadro con esa H"
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "images" / (cid + ".jpg")), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (OUT / "labels" / (cid + ".txt")).write_text(linea + "\n", encoding="utf-8")
    csv_p = OUT / "keypoints.csv"
    nuevo = not csv_p.exists()
    with open(csv_p, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if nuevo:
            wr.writerow(["frame_id", "keypoint_id", "x", "y", "visible"])
        for k in range(N_KP):
            wr.writerow([cid, k, round(float(xy[k, 0]), 2) if vis[k] else 0, round(float(xy[k, 1]), 2) if vis[k] else 0, 2 if vis[k] else 0])
    with open(OUT / "clicks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": cid, "clics": clics, "H": np.asarray(H).tolist()}) + "\n")
    return True, "guardado {}".format(cid)


def anotados():
    d = OUT / "images"
    return {p.stem for p in d.glob("*.jpg")} if d.exists() else set()


def saltados():
    p = OUT / "skipped.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def marca_saltado(cid):
    s = saltados()
    s.add(cid)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "skipped.json").write_text(json.dumps(sorted(s)))


def videos():
    """Videos de futbol de la raiz del proyecto (soccer*.mp4, sin extension)."""
    return sorted(p.stem for p in ROOT.glob("soccer*.mp4"))


def frames_candidatos(paso_s=2.0):
    """[(video, frame)] cada `paso_s` segundos, en orden, sin los ya anotados/saltados.
    Es SECUENCIAL (en hockey se barajaba para un val independiente; aqui interesa
    recorrer el partido)."""
    hecho = anotados() | saltados()
    out = []
    for v in videos():
        cap = cv2.VideoCapture(str(ROOT / (v + ".mp4")))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        out += [(v, i) for i in range(0, n, int(round(fps * paso_s))) if "{}_{:06d}".format(v, i) not in hecho]
    return out


def center_path(video):
    """Where the calibrated camera centre of `video` is stored (see lab/soccer/camera_center.py)."""
    return OUT / "camera_center_{}.json".format(video)


def save_center(video, center, info=None):
    OUT.mkdir(parents=True, exist_ok=True)
    center_path(video).write_text(json.dumps({"center": [float(x) for x in center], **(info or {})}), encoding="utf-8")


def load_center(video):
    """{"center": [x, y, z], ...} calibrated for `video`, or None."""
    p = center_path(video)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def lee_frame(v, i):
    cap = cv2.VideoCapture(str(ROOT / (v + ".mp4")))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None
