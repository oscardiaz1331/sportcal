"""Resuelve la homografia directamente de la segmentacion, sin pasar por keypoints.

Hasta ahora solo habiamos medido que la red de segmentacion LOCALIZA bien
(diagnose_lines_seg.py, z de cientos en IIHF) -- pero eso no es una homografia,
es un diagnostico. Este modulo es el paso que falta para que la segmentacion sea
una alternativa real a YOLO pose: convierte las curvas identificadas por clase en
correspondencias (mundo <-> imagen) y resuelve H por DLT.

Ventaja sobre la via clasica de color: alli el cuello de botella era la
CORRESPONDENCIA (que linea del mundo es este pixel rojo) -- por eso el ajuste
conjunto (fit_homography_lines.fit()) fallaba, tenia que adivinar. Aqui la
correspondencia viene GRATIS: la red ya dice "esto es linea_azul_A", no solo
"esto es una linea azul".

Geometria: las lineas del template (goles, azules, central) son verticales en
mundo (x=cte para todo y), asi que son correspondencias de RECTA, no de punto --
sabemos que cada pixel de esa clase esta en algun punto de esa recta, pero no en
que punto exacto. La forma correcta de usarlo es el DLT dual de rectas
(l_mundo ~ H^T l_imagen), no muestrear puntos arbitrarios a lo largo de la recta
ajustada (eso inventaria una correspondencia mundo<->pixel que no existe). Los
circulos (central y faceoff) SI dan una correspondencia de punto razonable (centro
del circulo ajustado <-> centro del circulo del template), con el sesgo conocido
de que el centro de una elipse proyectada no es exactamente la proyeccion del
centro del circulo real bajo perspectiva -- aceptable como ancla aproximada.

    python training/seg_to_homography.py --selftest          # verifica el DLT antes de nada
    python training/seg_to_homography.py --dataset hockeyrink_nhl --n 70
"""
import os

_CUDNN_DIR = os.sep.join(("NVIDIA", "CUDNN"))

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if _CUDNN_DIR not in p
)

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import make_line_masks as MM  # noqa: E402
import rink  # noqa: E402
from diagnose_lines_seg import load_model, predict_probs  # noqa: E402
from fit_homography_lines import geom_error  # noqa: E402
from relabel_reproject import read_label  # noqa: E402
from robust_fit import (  # noqa: E402
    BORDE_MARGEN_PX, MAX_ANGLE_GAP_DEG, MIN_CIRCLE_PX, MIN_LINE_PX,
    fit_circle_center, fit_line_px, fit_line_ransac, skeleton_points,
)

LINE_CLASSES = {2, 3, 4, 5, 6}     # verticales en mundo (x = cte)
CIRCLE_CLASSES = {7, 8, 9, 10, 11}


# ------------------------------------------------------------------ DLT

def dlt_point_rows(Xw, Yw, xi, yi):
    """x_i ~ H [Xw,Yw,1]^T -- las dos filas estandar del DLT de puntos."""
    return np.array([
        [Xw, Yw, 1, 0, 0, 0, -xi * Xw, -xi * Yw, -xi],
        [0, 0, 0, Xw, Yw, 1, -yi * Xw, -yi * Yw, -yi],
    ])


def dlt_line_rows(a, b, c, p, q, r):
    """Recta mundo (a,b,c) <-> recta imagen (p,q,r), via l_w ~ H^T l_i.

    Derivado del mismo truco de producto vectorial que el DLT de puntos
    (l_w x (M l_i) = 0 con M = H^T), pero sin asumir componente homogenea = 1
    (una recta no tiene esa normalizacion natural como un punto). Expresado en
    los coeficientes de H (no de H^T) para poder apilarlo con las filas de
    puntos y resolver un unico h. Verificado en self_test() antes de usarse.
    """
    return np.array([
        [0, -c * p, b * p, 0, -c * q, b * q, 0, -c * r, b * r],
        [c * p, 0, -a * p, c * q, 0, -a * q, c * r, 0, -a * r],
    ])


def solve_dlt(rows, return_cond=False):
    A = np.vstack(rows).astype(np.float64)
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / np.where(norm > 1e-12, norm, 1.0)
    _, S, VT = np.linalg.svd(A)
    H = VT[-1].reshape(3, 3)
    H = H / H[2, 2]
    if not return_cond:
        return H
    # un sistema bien planteado tiene UN valor singular casi nulo, bien separado
    # del resto -- si los dos ultimos son parecidos, el "nulo" no esta aislado
    # (tipico de 3 rectas paralelas + 1 punto: 8 dof en el recuento, pero la
    # direccion paralela solo aporta info redundante) y un poco de ruido en las
    # correspondencias se amplifica en un H completamente distinto. Confirmado
    # en un caso real: 3 rectas (todas x=cte, paralelas) + 1 circulo, cada
    # correspondencia a <16px de su verdad, pero el H resultante daba 2189px.
    cond = S[-1] / (S[-2] or 1e-12)
    return H, cond


def scale_transform(ref):
    """T tal que X' = T[:2,:2]@X + T[:2,2] manda [0,ref] a [-1,1] (isotropica).

    Sin esto el DLT mezcla filas con magnitudes de ~1000 (px de imagen) y ~10-60
    (metros de mundo) y el SVD encuentra un nulo numericamente inutil -- es EL
    motivo clasico por el que un DLT "matematicamente correcto" (ver self_test)
    puede dar basura sobre datos reales. Hartley (2004) 4.4.4.
    """
    s = 2.0 / ref
    return np.array([[s, 0, -1], [0, s, -1], [0, 0, 1]])


def transform_point(T, X, Y):
    x, y, w = T @ [X, Y, 1.0]
    return x / w, y / w


def transform_line(T, a, b, c):
    return np.linalg.inv(T).T @ [a, b, c]


def solve_dlt_with_residual(points, lines, T_world, T_img):
    """Como solve_dlt_correspondences pero devuelve tambien el residuo (S[-1]
    normalizado) -- se usa para elegir, entre varias asignaciones candidatas
    de una correspondencia ambigua (p.ej. el signo de la recta de vallas), la
    que mejor encaja con el RESTO de correspondencias."""
    rows = []
    for Xw, Yw, xi, yi in points:
        Xn, Yn = transform_point(T_world, Xw, Yw)
        xn, yn = transform_point(T_img, xi, yi)
        rows += list(dlt_point_rows(Xn, Yn, xn, yn))
    for a, b, c, p, q, r in lines:
        an, bn, cn = transform_line(T_world, a, b, c)
        pn, qn, rn = transform_line(T_img, p, q, r)
        rows += list(dlt_line_rows(an, bn, cn, pn, qn, rn))
    if len(rows) < 8:
        return None, np.inf
    A = np.vstack(rows).astype(np.float64)
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / np.where(norm > 1e-12, norm, 1.0)
    _, S, VT = np.linalg.svd(A)
    Hn = VT[-1].reshape(3, 3)
    Hn = Hn / Hn[2, 2]
    H = np.linalg.inv(T_img) @ Hn @ T_world
    return H / H[2, 2], float(S[-1])


# ------------------------------------------------------------------ refinamiento no lineal

def sample_world_line(a, b, c, L, W, n=5):
    """Puntos sobre la recta mundo (a,b,c), cubriendo su extension REAL en la
    pista (no una extrapolacion arbitraria) -- X=cte para y en [0,W], o Y=cte
    para x en [0,L]."""
    if abs(a) >= abs(b):
        x0 = -c / a
        return [(x0, t) for t in np.linspace(0, W, n)]
    y0 = -c / b
    return [(t, y0) for t in np.linspace(0, L, n)]


def refine_residuals(h8, points_n, line_samples_n):
    H = np.array(list(h8) + [1.0]).reshape(3, 3)
    res = []
    for Xn, Yn, xn, yn in points_n:
        proj = H @ [Xn, Yn, 1.0]
        px, py = proj[:2] / proj[2]
        res.append(px - xn)
        res.append(py - yn)
    for (pn, qn, rn), pts in line_samples_n:
        norm_pq = np.hypot(pn, qn) or 1.0
        for Xn, Yn in pts:
            proj = H @ [Xn, Yn, 1.0]
            px, py = proj[:2] / proj[2]
            res.append((pn * px + qn * py + rn) / norm_pq)
    return np.array(res)


def refine_nonlinear(H0, points, lines, T_world, T_img, L, W):
    """Parte de la H del DLT (una semilla, casi siempre mala) y minimiza la
    distancia de reproyeccion REAL -- no el residuo algebraico del DLT, que ya
    hemos visto que puede ser pequeño y aun asi dar una H inutil (ver docstring
    de arriba: caso con todas las correspondencias a <16px pero H a 2189px).
    Todo en el mismo espacio normalizado que el DLT, por la misma razon
    (Hartley). Perdida 'soft_l1': una sola correspondencia mala no debe poder
    arrastrar el ajuste como hace un min-cuadrados puro.
    """
    from scipy.optimize import least_squares

    points_n = [(*transform_point(T_world, Xw, Yw), *transform_point(T_img, xi, yi))
                for Xw, Yw, xi, yi in points]
    line_samples_n = []
    for a, b, c, p, q, r in lines:
        pn, qn, rn = transform_line(T_img, p, q, r)
        pts_n = [transform_point(T_world, Xs, Ys) for Xs, Ys in sample_world_line(a, b, c, L, W)]
        line_samples_n.append(((pn, qn, rn), pts_n))

    Hn0 = T_img @ H0 @ np.linalg.inv(T_world)
    Hn0 = Hn0 / Hn0[2, 2]
    h8_0 = Hn0.flatten()[:8]
    # 1a pasada: minimos cuadrados puros (loss lineal) -- con soft_l1 y un
    # f_scale mal calibrado el gradiente se satura desde el arranque y el
    # optimizador "converge" (xtol) sin moverse casi nada: comprobado en un
    # caso real, costo baja de 0.72 a 0.12 pero el residuo maximo NO CAMBIA
    # (sigue en 0.59 = ~570px), porque f_scale=0.05 esta muy por debajo del
    # error inicial tipico. Sin robustez primero para tener gradiente de
    # verdad en todo el rango.
    sol = least_squares(refine_residuals, h8_0, args=(points_n, line_samples_n),
                        loss="linear", method="trf", max_nfev=3000, xtol=1e-12, ftol=1e-12)
    # 2a pasada: ahora si, robusta, para no dejar que una correspondencia mala
    # arrastre el resultado -- f_scale calibrado al residuo YA CERCA del optimo
    # (equivalente a ~15px reales), no al residuo inicial (que podia ser de
    # cientos de px).
    f_scale = max(0.01, 1.5 * np.median(np.abs(refine_residuals(sol.x, points_n, line_samples_n))))
    sol = least_squares(refine_residuals, sol.x, args=(points_n, line_samples_n),
                        loss="soft_l1", f_scale=f_scale, method="trf", max_nfev=2000)
    Hn = np.array(list(sol.x) + [1.0]).reshape(3, 3)
    H = np.linalg.inv(T_img) @ Hn @ T_world
    # mediana del residuo final (espacio normalizado): mide consistencia
    # geometrica REAL, a diferencia del residuo algebraico del DLT que
    # confirmamos NO discrimina bien (ver mas abajo, desambiguacion de vallas).
    r_final = np.median(np.abs(refine_residuals(sol.x, points_n, line_samples_n)))
    return H / H[2, 2], float(r_final)


def solve_dlt_correspondences(points, lines, T_world, T_img, max_cond=0.05):
    """points: [(Xw,Yw,xi,yi)...]   lines: [(a,b,c,p,q,r)...]  (coords SIN normalizar).

    Normaliza, resuelve, des-normaliza -- ver scale_transform(). Devuelve None si
    la configuracion de correspondencias es demasiado degenerada (ver solve_dlt).
    """
    rows = []
    for Xw, Yw, xi, yi in points:
        Xn, Yn = transform_point(T_world, Xw, Yw)
        xn, yn = transform_point(T_img, xi, yi)
        rows += list(dlt_point_rows(Xn, Yn, xn, yn))
    for a, b, c, p, q, r in lines:
        an, bn, cn = transform_line(T_world, a, b, c)
        pn, qn, rn = transform_line(T_img, p, q, r)
        rows += list(dlt_line_rows(an, bn, cn, pn, qn, rn))
    Hn, cond = solve_dlt(rows, return_cond=True)
    if cond > max_cond:
        return None
    H = np.linalg.inv(T_img) @ Hn @ T_world
    return H / H[2, 2]


def self_test(seed=0, n_point=4, n_line=4, noise=0.0):
    """H conocida -> genera correspondencias -> resuelve -> compara. Sin esto no
    hay forma de confiar en los signos/indices del DLT de rectas."""
    rng = np.random.default_rng(seed)
    while True:
        Htrue = np.eye(3) + rng.normal(0, 0.15, (3, 3))
        Htrue[2, 2] = 1.0
        if abs(np.linalg.det(Htrue)) > 0.2:
            break
    rows = []
    for _ in range(n_point):
        Xw, Yw = rng.uniform(-8, 8, 2)
        p = Htrue @ [Xw, Yw, 1]
        xi, yi = p[:2] / p[2] + rng.normal(0, noise, 2)
        rows += list(dlt_point_rows(Xw, Yw, xi, yi))
    for _ in range(n_line):
        a, b, c = rng.uniform(-1, 1, 3)
        l_i = np.linalg.inv(Htrue).T @ [a, b, c]
        if noise:
            l_i = l_i + rng.normal(0, noise * 0.01, 3)
        rows += list(dlt_line_rows(a, b, c, *l_i))
    Hest = solve_dlt(rows)
    err = np.abs(Hest - Htrue / Htrue[2, 2]).max()
    print("self_test (escala unidad)      max|H_est-H_true| = {:.2e}  ({})".format(
        err, "OK" if err < 1e-6 else "FALLO"))
    ok1 = err < 1e-6

    # mismo test pero con las magnitudes REALES del problema (mundo en metros
    # 0..60, imagen en px 0..1920) -- esto es lo que se le olvido comprobar la
    # primera vez: el DLT sin normalizar da basura numerica aqui aunque los
    # signos/indices sean correctos (visto en produccion: p50=1858px). Hartley
    # normalization (scale_transform) es lo que lo arregla.
    world_pts = rng.uniform([5, 5], [55, 25], (4, 2))
    img_pts = rng.uniform([100, 100], [1800, 950], (4, 2)).astype(np.float32)
    Htrue2 = cv2.getPerspectiveTransform(world_pts.astype(np.float32), img_pts)
    Htrue2 = Htrue2 / Htrue2[2, 2]
    T_world, T_img = scale_transform(60.0), scale_transform(1920.0)
    points, lines = [], []
    for _ in range(4):
        Xw, Yw = rng.uniform([5, 5], [55, 25])
        proj = Htrue2 @ [Xw, Yw, 1]
        xi, yi = proj[:2] / proj[2]
        points.append((Xw, Yw, xi, yi))
    for _ in range(4):
        a, b, c = rng.uniform(-1, 1, 3) * [1, 1, 30]
        p, q, r = np.linalg.inv(Htrue2).T @ [a, b, c]
        lines.append((a, b, c, p, q, r))
    Hest2 = solve_dlt_correspondences(points, lines, T_world, T_img)
    err2 = np.abs(Hest2 - Htrue2).max()
    print("self_test (escala real, normalizado)  max|H_est-H_true| = {:.2e}  ({})".format(
        err2, "OK" if err2 < 1e-3 else "FALLO"))
    return ok1 and err2 < 1e-3


# ------------------------------------------------------------------ extraccion desde la red
# skeleton_points / fit_line_px / fit_line_ransac / fit_circle_center viven en
# robust_fit.py (importadas arriba) -- son puro cv2/numpy, sin torch, para que
# testing/classical_cv_lab.py pueda reutilizarlas sin cargar la red.

def world_boards_line(sign, p):
    """Recta mundo Y=0 (sign=0) o Y=W (sign=1) -- los dos tramos rectos de la valla."""
    W = p["width"]
    return np.array([0.0, 1.0, 0.0 if sign == 0 else -W])


def world_vertical_line(x0):
    """Recta mundo X = x0  ->  (a,b,c) con a*X+b*Y+c=0."""
    return np.array([1.0, 0.0, -x0])


def world_circle_center(cls, p):
    """Centro en metros de cada clase de circulo, EXACTAMENTE como rink_polylines."""
    L, W = p["length"], p["width"]
    gl = [p["goal_line_from_end"], L - p["goal_line_from_end"]]
    dot = p["dot_from_goal_line"]
    dy = p["dot_from_axis"]
    cy = W / 2
    return {
        7: (L / 2, cy),
        8: (gl[0] + dot, cy - dy), 9: (gl[0] + dot, cy + dy),
        10: (gl[1] - dot, cy - dy), 11: (gl[1] - dot, cy + dy),
    }[cls]


def class_x0(cls, p):
    L = p["length"]
    gl = [p["goal_line_from_end"], L - p["goal_line_from_end"]]
    bl = [p["blue_from_end"], L - p["blue_from_end"]]
    return {2: gl[0], 3: gl[1], 4: bl[0], 5: bl[1], 6: L / 2}[cls]


def solve_from_probs(probs, params_rink, w, h, debug=False):
    """probs: (NCLS,h,w) de la red. Devuelve (H, n_lineas, n_circulos, costo) o
    (None, n_lineas, n_circulos, inf) si no hay suficiente redundancia. `costo`
    es la mediana del residuo del refinamiento (espacio normalizado, ~0 es
    perfecto) -- util como segunda señal de confianza ademas del dof (ver
    CLAUDE.md sección 7: casos justo en el limite de dof siguen fallando a
    veces, este numero deberia distinguirlos).

    Con debug=True devuelve un 5o elemento: dict con las correspondencias
    realmente usadas (para pintar -- ver dibuja_debug()).
    """
    seg = np.argmax(probs, axis=0)
    points, lines, n_l, n_c = [], [], 0, 0
    cls_por_linea, cls_por_punto = [], []
    for cls in LINE_CLASSES:
        li = fit_line_px(seg == cls)
        if li is None:
            continue
        lw = world_vertical_line(class_x0(cls, params_rink))
        lines.append((*lw, *li))
        cls_por_linea.append(cls)
        n_l += 1
    for cls in CIRCLE_CLASSES:
        c = fit_circle_center(seg == cls, w, h)
        if c is None:
            continue
        Xw, Yw = world_circle_center(cls, params_rink)
        points.append((Xw, Yw, *c))
        cls_por_punto.append(cls)
        n_c += 1

    # goles/azules/central son TODAS paralelas en el mundo: por muchas que
    # haya, nunca fijan por si solas la direccion transversal de H. La valla SI
    # corre en esa otra direccion (Y=0 / Y=W), pero no sabemos de que lado esta
    # sin conocer H todavia.
    #
    # Elegir el signo por el residuo algebraico del DLT (ultimo valor singular)
    # se probo y FALLA sistematicamente: comprobado contra la verdad en 27
    # frames reales, acierta 0/27 -- no es ruido, esta ANTICORRELACIONADO (ya
    # sabiamos que ese residuo puede ser pequeño con una H inutil, ver el caso
    # de 2189px mas abajo; aqui ademas resulta ser peor que al azar). En vez de
    # eso: se resuelve DLT+refinamiento COMPLETO para cada signo candidato y se
    # elige por el residuo del refinamiento (mediana del error de reproyeccion
    # real, no el algebraico) -- ese si se ha visto que mide lo que dice medir.
    lb = fit_line_ransac(seg == 1)
    valla_lines = [None]
    if lb is not None:
        valla_lines = [(*world_boards_line(0, params_rink), *lb),
                       (*world_boards_line(1, params_rink), *lb)]

    dof_base = 2 * (n_l + n_c)
    T_world = scale_transform(params_rink["length"])
    T_img = scale_transform(w)
    H_final, mejor_costo, mejor_lines, mejor_valla_signo, mejor_H_dlt = None, np.inf, None, None, None
    for i_extra, extra in enumerate(valla_lines):
        cand_lines = lines + ([extra] if extra else [])
        dof = dof_base + (2 if extra else 0)
        n_dir_transversal = n_c + (1 if extra else 0)
        # dof>=8 es el minimo matematico, pero con EXACTAMENTE 8 (4 correspon-
        # dencias, cero redundancia) no hay margen para que nada delate una
        # configuracion mala -- confirmado en un caso real (2 circulos+2
        # lineas, cada uno a <25px de su verdad, en direcciones NO paralelas)
        # que aun asi colapso (columna Y de H practicamente a cero). Exigir
        # margen de sobra (>=10, o sea 5+ correspondencias) cuesta cobertura
        # pero es la unica manera de que quede alguna redundancia real.
        if dof < 10 or n_dir_transversal < 1:
            continue
        H_dlt, _ = solve_dlt_with_residual(points, cand_lines, T_world, T_img)
        if H_dlt is None:
            continue
        # sanidad numerica ANTES de refinar: una H normalizada con coeficientes
        # de ~1e14 (visto en un caso real) es basicamente singular -- a esa
        # escala un paso de optimizador normal no cambia nada detectable en
        # float64 y "converge" sin moverse (Jacobiano con 6 de 8 columnas
        # exactamente a cero, confirmado). Mejor descartar la semilla que
        # fingir refinarla.
        Hn_check = T_img @ H_dlt @ np.linalg.inv(T_world)
        if not np.isfinite(Hn_check).all() or np.abs(Hn_check).max() > 50:
            continue
        try:
            H_ref, costo = refine_nonlinear(H_dlt, points, cand_lines, T_world, T_img,
                                            params_rink["length"], params_rink["width"])
            if not np.isfinite(H_ref).all():
                continue
        except Exception:
            continue
        if costo < mejor_costo:
            H_final, mejor_costo, mejor_lines = H_ref, costo, cand_lines
            mejor_valla_signo = None if extra is None else i_extra
            mejor_H_dlt = H_dlt

    if H_final is None:
        if debug:
            return None, n_l, n_c, np.inf, None
        return None, n_l, n_c, np.inf

    if debug:
        info = dict(points=points, lines=mejor_lines, cls_por_linea=cls_por_linea,
                    cls_por_punto=cls_por_punto, valla_recta_img=lb, valla_signo=mejor_valla_signo,
                    H_dlt=mejor_H_dlt)
        return H_final, n_l, n_c, mejor_costo, info
    return H_final, n_l, n_c, mejor_costo


def dibuja_debug(img, Hgt, Hest, info, params_rink, err_txt):
    """Pinta: recta/circulo EXTRAIDOS de la red (lo que usa el solver), la
    plantilla proyectada por la H real (verde) y por la H estimada (magenta).
    La separacion entre verde y magenta ES el error -- donde coincidan, bien;
    donde no, ahi esta fallando el solver.
    """
    vis = img.copy()
    h, w = img.shape[:2]
    rink.draw_rink(vis, Hgt.astype(np.float32), params_rink, (60, 200, 60), 2)
    rink.draw_rink(vis, Hest.astype(np.float32), params_rink, (230, 60, 230), 2)

    for cls, (Xw, Yw, xi, yi) in zip(info["cls_por_punto"], info["points"]):
        cv2.circle(vis, (int(xi), int(yi)), 9, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(vis, "c{}".format(cls), (int(xi) + 10, int(yi)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    def dibuja_recta(p, q, r, color, txt):
        pts = []
        for x in (0, w - 1):
            if abs(q) > 1e-9:
                pts.append((x, int(-(p * x + r) / q)))
        for y in (0, h - 1):
            if abs(p) > 1e-9:
                pts.append((int(-(q * y + r) / p), y))
        pts = [pt for pt in pts if -w <= pt[0] <= 2 * w and -h <= pt[1] <= 2 * h]
        if len(pts) >= 2:
            cv2.line(vis, pts[0], pts[-1], color, 2, cv2.LINE_AA)
            cv2.putText(vis, txt, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    for cls, (a, b, c, p, q, r) in zip(info["cls_por_linea"], info["lines"][:len(info["cls_por_linea"])]):
        dibuja_recta(p, q, r, (255, 180, 0), "L{}".format(cls))
    if info["valla_recta_img"] is not None:
        p, q, r = info["valla_recta_img"]
        signo = info["valla_signo"]
        dibuja_recta(p, q, r, (0, 140, 255),
                     "vallas (Y={})".format("0" if signo == 0 else "W" if signo == 1 else "?"))

    cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(vis, err_txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "verde=H real  magenta=H estimada  amarillo=circulos  naranja=lineas/vallas usadas",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ------------------------------------------------------------------ evaluacion

def run(args):
    ok = self_test()
    if args.selftest:
        return
    if not ok:
        raise SystemExit("el DLT no pasa el self-test, no sigo")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, imgsz = load_model(device)
    params_rink = rink.RINK_NHL if "nhl" in args.dataset else rink.RINK_IIHF
    tpl = rink.build_template(params_rink)

    lbls = sorted((ROOT / "datasets" / args.dataset / "labels" / "val").glob("*.txt"))
    print("frames disponibles en val: {}".format(len(lbls)))

    errores, cobertura, costes, hechos = [], [], [], 0
    frames_info = []
    for lbl in lbls:
        if hechos >= args.n:
            break
        ip = ROOT / "datasets" / args.dataset / "images" / "val" / (lbl.stem + ".jpg")
        if not ip.exists():
            continue
        rec = read_label(lbl)
        img = cv2.imread(str(ip))
        if rec is None or img is None:
            continue
        h, w = img.shape[:2]
        fq, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
        if fq is None:
            continue
        Hgt = fq[0]
        hechos += 1
        probs = predict_probs(model, img, imgsz, device)
        Hest, n_l, n_c, costo, info = solve_from_probs(probs, params_rink, w, h, debug=True)
        cobertura.append(1 if Hest is not None else 0)
        if Hest is None:
            continue
        e = geom_error(Hest, Hgt, tpl, w, h)
        errores.append(e)
        costes.append(costo)
        if args.figuras:
            frames_info.append((e, lbl.stem, ip, Hgt, Hest, info))

    cobertura = np.array(cobertura)
    print("\n=== {} val: homografia directa desde segmentacion (sin keypoints) ===".format(args.dataset))
    print("  cobertura (>=8 dof disponibles): {:.0f} % de {} frames".format(
        100 * cobertura.mean() if len(cobertura) else 0.0, len(cobertura)))
    if errores:
        E = np.array(errores)
        print("  error de reproyeccion (px a 1920), sobre los frames con H:")
        print("    p50 {:.1f}   p90 {:.1f}   <10px {:.0f}%   <25px {:.0f}%   <60px {:.0f}%".format(
            np.median(E), np.percentile(E, 90),
            100 * np.mean(E < 10), 100 * np.mean(E < 25), 100 * np.mean(E < 60)))
        C = np.array(costes)
        r = np.corrcoef(C, np.log1p(np.clip(E, 0, 1e6)))[0, 1] if len(C) > 2 else float("nan")
        print("  coste de refinamiento vs error real: r={:.2f} (log)   mediana costo={:.4f}".format(r, np.median(C)))
    else:
        print("  ningun frame con suficientes correspondencias")

    if args.figuras and frames_info:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_info.sort(key=lambda t: t[0])
        peores = frames_info[-args.figuras:]
        mejores = frames_info[:args.figuras]
        for etiqueta, grupo in (("mal", peores), ("bien", mejores)):
            for e, stem, ip, Hgt, Hest, info in grupo:
                img = cv2.imread(str(ip))
                txt = "{}  {}  error={:.0f}px".format(etiqueta.upper(), stem[:24], e)
                vis = dibuja_debug(img, Hgt, Hest, info, params_rink, txt)
                cv2.imwrite(str(out_dir / "{}_{}_{}.jpg".format(args.dataset, etiqueta, stem)), vis)
        print("\n  figuras (verde=H real, magenta=H estimada) en {}".format(out_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hockeyrink_nhl")
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--figuras", type=int, default=0, help="cuantos peores/mejores frames dibujar")
    ap.add_argument("--out", default=str(ROOT / "scratch_frames" / "seg_to_h"))
    run(ap.parse_args())
