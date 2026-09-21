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

    python -m sportcal.lab.hockey.seg_to_homography --selftest          # verifica el DLT antes de nada
    python -m sportcal.lab.hockey.seg_to_homography --dataset hockeyrink_nhl --n 70
"""
import os

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from sportcal.paths import ROOT

from sportcal.lab.hockey import make_line_masks as MM
from sportcal.sports.hockey import rink
from sportcal.lab.hockey.diagnose_lines_seg import load_model, predict_probs  # noqa: E402
from sportcal.core.geometry import geom_error
from sportcal.lab.hockey.relabel_reproject import read_label  # noqa: E402
from sportcal.core.fitting import (  # noqa: E402
    BORDER_MARGIN_PX, MAX_ANGLE_GAP_DEG, MIN_CIRCLE_PX, MIN_LINE_PX,
    fit_circle_center, fit_line_px, fit_line_ransac, skeleton_points,
)

from sportcal.core.geometry import (  # noqa: E402  (DLT + refinement live in core)
    dlt_line_rows, dlt_point_rows, refine_nonlinear, scale_transform, solve_dlt, solve_dlt_correspondences, solve_dlt_with_residual,
)

LINE_CLASSES = {2, 3, 4, 5, 6}     # verticales en mundo (x = cte)
CIRCLE_CLASSES = {7, 8, 9, 10, 11}


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
# sportcal/lab/common/classical_cv_lab.py pueda reutilizarlas sin cargar la red.

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
