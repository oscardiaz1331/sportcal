"""Evalua soccer_field.py: (1) sintetico con pose conocida, donde el error es real;
(2) frames de un video real, donde solo se puede juzgar a ojo (montaje en imagen).

    python testing/soccer_eval.py --sintetico --n 40
    python testing/soccer_eval.py --video soccer.mp4 --frames 900,3000,5100 --salida montaje.jpg
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soccer_field as SF  # noqa: E402

W, H_IMG = 960, 540   # tamaño de trabajo


# ------------------------------------------------------------ mascara de un frame real

def alargamiento(lab, n):
    """(n,) raiz del cociente de los autovalores de la covarianza de los pixeles de cada
    componente (1 = redondo, mucho mayor = alargado). Vectorizado con bincount."""
    ys, xs = np.indices(lab.shape)
    l = lab.ravel()
    cnt = np.maximum(np.bincount(l, minlength=n).astype(float), 1.0)
    mom = lambda v: np.bincount(l, weights=v.ravel().astype(float), minlength=n) / cnt
    mx, my = mom(xs), mom(ys)
    cxx, cyy, cxy = mom(xs * xs) - mx ** 2, mom(ys * ys) - my ** 2, mom(xs * ys) - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0.0))
    return np.sqrt(np.maximum(tr / 2 + disc, 1e-9) / np.maximum(tr / 2 - disc, 0.05))


def etapas_mascara(img, w_trab=W, chi2=10.17, umbrales=(5.0, 5.0, 8.5), elong_max=8.0):
    """Todas las etapas de la mascara de lineas del baseline de futbol del laboratorio
    (region por gaussiana robusta con semilla de cesped + a+, b-, L+ locales en OR,
    recortado a la region) y jugadores quitados por grosor: apertura -> los trozos
    compactos y VERTICALES que sobreviven son jugadores; se restan (dilatados).
    Devuelve dict de arrays uint8 0/1 al tamaño de trabajo: im (BGR), region, a_pos,
    b_neg, l_pos (cada filtro ya recortado a la region), todo (su OR), abiertos (lo que
    sobrevive a la apertura), vertical (componentes de `abiertos` tomados por jugador),
    jugadores (esos mismos dilatados: lo que se resta), alargados (componentes gruesos y
    verticales que NO se toman por jugador por ser muy alargados), lineas (todo - jugadores).

    elong_max: un componente de la apertura es jugador si es mas alto que ancho (h>=1.3w) Y su
    alargamiento (ver `alargamiento`) es <= elong_max. Sin el segundo criterio (elong_max=None,
    la regla original) la linea CENTRAL, gruesa y vertical, se descartaba entera como jugador:
    en los frames 1500 y 2400 de soccer.mp4 su componente tiene alargamiento 32 y 22 frente a
    mediana 2.2-2.6 y maximo 4.4 en los jugadores; 8 cae en ese hueco."""
    import ice_lines_probe as P
    h0, w0 = img.shape[:2]
    im = cv2.resize(img, (w_trab, int(round(h0 * w_trab / w0))), interpolation=cv2.INTER_AREA)
    sig = w_trab / 1920.0
    region = P.rink_region(P.ice_robust(im, chi2=chi2, deporte="futbol")) > 0
    win = max(11, int(61 * sig) | 1)
    da, db = P.local_chroma(im, win=win)
    dl = P.local_l(im, win=win)
    a_min, b_min, l_min = umbrales
    a_pos = (np.maximum(da, 0) >= a_min) & region
    b_neg = (np.maximum(-db, 0) >= b_min) & region
    l_pos = (np.maximum(dl, 0) >= l_min) & region
    todo = (a_pos | b_neg | l_pos).astype(np.uint8)
    k = max(3, int(9 * sig) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    abiertos = cv2.morphologyEx(todo, cv2.MORPH_OPEN, ker)
    n, lab, st, _ = cv2.connectedComponentsWithStats(abiertos, connectivity=8)
    vert = np.zeros(n, bool)
    vert[1:] = (st[1:, cv2.CC_STAT_HEIGHT] >= 1.3 * st[1:, cv2.CC_STAT_WIDTH]) & (st[1:, cv2.CC_STAT_AREA] >= 6)
    alargados = np.zeros(n, bool)
    if elong_max is not None:
        el = alargamiento(lab, n)
        alargados = vert & (el > elong_max)
        vert = vert & (el <= elong_max)
    vertical = vert[lab].astype(np.uint8)
    jug = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 4, k + 4)))
    return {"im": im, "region": region.astype(np.uint8), "a_pos": a_pos.astype(np.uint8),
            "b_neg": b_neg.astype(np.uint8), "l_pos": l_pos.astype(np.uint8), "todo": todo,
            "abiertos": abiertos, "vertical": vertical, "jugadores": jug,
            "alargados": alargados[lab].astype(np.uint8), "lineas": todo & (1 - jug)}


def mascara_lineas(img, w_trab=W, chi2=10.17, umbrales=(5.0, 5.0, 8.5)):
    """(lineas, todo): la mascara final de lineas (sin jugadores) y la previa. Ver etapas_mascara."""
    e = etapas_mascara(img, w_trab, chi2, umbrales)
    return e["lineas"], e["todo"]


# ------------------------------------------------------------ sintetico

def pose_aleatoria(rng, w=W, h=H_IMG):
    """Pose plausible de camara de retransmision cuyo encuadre ve al menos 1.2 anchos de linea."""
    pts, cont = SF.muestrea(0.5)
    while True:
        p = np.array([rng.uniform(-25, 25), rng.uniform(-75, -42), rng.uniform(10, 28),
                      np.radians(rng.uniform(-45, 45)), np.radians(rng.uniform(8, 38)),
                      w * np.exp(rng.uniform(np.log(0.7), np.log(4.0)))])
        H = SF.pose_a_H(p, w, h)[0]
        xy, den = SF.proyecta(H[None], pts)
        ok = (den[0] > 0) & (xy[0, :, 0] >= 0) & (xy[0, :, 0] < w) & (xy[0, :, 1] >= 0) & (xy[0, :, 1] < h)
        seg = ok[1:] & ok[:-1] & cont[1:]
        L = np.hypot(*(xy[0, 1:] - xy[0, :-1]).T)
        if (L * seg).sum() >= 1.2 * w:
            return p, H


def dibuja(H, rng, nivel, w=W, h=H_IMG):
    """Mascara sintetica: la plantilla proyectada + suciedad segun `nivel` (0 limpio, 1, 2)."""
    m = np.zeros((h, w), np.uint8)
    for pl in SF.polilineas().values():
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
        s = np.arange(0, d[-1], 0.1)
        p = np.stack([np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])], 1)
        xy, den = SF.proyecta(H[None], p)
        xy, den = xy[0], den[0]
        ok = (den > 1e-3) & np.isfinite(xy).all(1) & (np.abs(xy) < 5 * w).all(1)
        for run in np.split(np.arange(len(p)), np.where(~ok)[0]):
            run = run[ok[run]]
            if len(run) > 1:
                cv2.polylines(m, [np.round(xy[run]).astype(np.int32)], False, 1, 2)
    if nivel == 0:
        return m
    # (huecos por jugadores, residuos compactos, rectas espurias, dropout, ruido, manchas SIN quitar)
    n_o, n_r, n_s, drop, ruido, n_b = {1: (12, 4, 3, 0.10, 0.002, 0), 2: (20, 10, 6, 0.20, 0.005, 0),
                                       3: (20, 10, 6, 0.20, 0.005, 12)}[nivel]
    m[rng.random(m.shape) < drop * m] = 0                    # huecos aleatorios en la LINEA (antes de nada mas)
    def jugador():
        return ((int(rng.uniform(0, w)), int(rng.uniform(0.25 * h, 0.95 * h))),
                (int(w * rng.uniform(0.008, 0.02)), int(w * rng.uniform(0.02, 0.045))))
    for _ in range(n_o):      # jugador tapando la linea (la apertura del pipeline real lo quita: queda el hueco)
        c, ax = jugador()
        cv2.ellipse(m, c, (ax[0] + 3, ax[1] + 3), 0, 0, 360, 0, -1)
    for _ in range(n_r):      # residuos que la apertura no quita: manchas pequeñas
        c, _ = jugador()
        cv2.ellipse(m, c, (int(rng.uniform(2, 5)), int(rng.uniform(4, 9))), 0, 0, 360, 1, -1)
    for _ in range(n_b):      # manchas de jugador llenas, sin quitar
        c, ax = jugador()
        cv2.ellipse(m, c, ax, 0, 0, 360, 1, -1)
    for _ in range(n_s):      # rectas espurias (vallas, publicidad)
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        a, l = rng.uniform(0, np.pi), rng.uniform(0.05, 0.2) * w
        cv2.line(m, (int(x0), int(y0)), (int(x0 + l * np.cos(a)), int(y0 + l * np.sin(a))), 1, 2)
    m[rng.random(m.shape) < ruido] = 1
    return m


def evalua_sintetico(n, seed):
    rng = np.random.default_rng(seed)
    k1920 = 1920.0 / W
    for nivel in (0, 1, 2, 3):
        filas = []
        for i in range(n):
            p, Hgt = pose_aleatoria(rng)
            mask = dibuja(Hgt, rng, nivel)
            campo = SF.Campo(mask)
            t0 = time.time()
            rB = campo.busca_pose()
            tB = time.time() - t0
            t0 = time.time()
            rA = campo.busca_lineas()
            tA = time.time() - t0
            def err(r, k=0):
                return SF.error_reproy(r[k]["H"], Hgt, W, H_IMG) * k1920 if len(r) > k else 1e4
            def mejor(r):
                return min((SF.error_reproy(c["H"], Hgt, W, H_IMG) * k1920 for c in r), default=1e4)
            todos = sorted(rB + rA, key=lambda r: -r["score"])
            filas.append([err(rB), err(rA), err(todos), mejor(rB), mejor(rA), tB, tA,
                          rB[0]["score"] if rB else 0, rA[0]["score"] if rA else 0,
                          SF.error_reproy(SF.Campo(mask).refina(Hgt)[0], Hgt, W, H_IMG) * k1920])
        f = np.array(filas)
        print("\n== nivel de suciedad {} ({} frames) ==".format(nivel, n))
        print("{:<38s} {:>8s} {:>8s} {:>8s}".format("(error a 1920 px de ancho)", "p50", "p90", "<10px"))
        for j, nom in ((0, "chamfer (rejilla de poses) top-1"), (1, "intersecciones (rectas) top-1"),
                       (2, "mejor puntuacion de las dos"), (3, "chamfer: mejor de 8 (oraculo)"),
                       (4, "intersecciones: mejor de 8 (oraculo)"), (9, "refinar DESDE la verdad (techo del refino)")):
            print("{:<38s} {:>7.1f}  {:>7.1f}  {:>6.0f}%".format(
                nom, np.median(f[:, j]), np.percentile(f[:, j], 90), 100 * (f[:, j] < 10).mean()))
        print("tiempo por frame: chamfer {:.1f}s   intersecciones {:.1f}s".format(f[:, 5].mean(), f[:, 6].mean()))


# ------------------------------------------------------------ real

def dibuja_plantilla(img, H, color, esc):
    for pl in SF.polilineas().values():
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
        s = np.arange(0, d[-1], 0.25)
        p = np.stack([np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])], 1)
        xy, den = SF.proyecta(H[None], p)
        xy, den = xy[0] * esc, den[0]
        ok = (den > 1e-3) & np.isfinite(xy).all(1) & (np.abs(xy) < 1e5).all(1)
        for run in np.split(np.arange(len(p)), np.where(~ok)[0]):
            run = run[ok[run]]
            if len(run) > 1:
                cv2.polylines(img, [np.round(xy[run]).astype(np.int32)], False, color, 2, cv2.LINE_AA)


def evalua_real(video, frames, salida):
    cap = cv2.VideoCapture(video)
    filas = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        lineas, _ = mascara_lineas(img)
        campo = SF.Campo(lineas)
        t0 = time.time()
        rB, rA = campo.busca_pose(), campo.busca_lineas()
        print("frame {:>5d}: mascara {:.2f}%  chamfer {}  intersecciones {}  ({:.1f}s)".format(
            f, 100 * lineas.mean(),
            "{:.3f}".format(rB[0]["score"]) if rB else "-", "{:.3f}".format(rA[0]["score"]) if rA else "-",
            time.time() - t0))
        esc = 1.0
        base = cv2.resize(img, (W, H_IMG))
        cols = [cv2.cvtColor(lineas * 255, cv2.COLOR_GRAY2BGR)]
        for nom, r, col in (("chamfer", rB, (0, 0, 255)), ("intersecciones", rA, (255, 0, 255))):
            v = base.copy()
            if r:
                dibuja_plantilla(v, r[0]["H"], col, esc)
            cv2.putText(v, "{} {}".format(nom, "{:.3f}".format(r[0]["score"]) if r else "sin hipotesis"),
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cols.append(v)
        filas.append(np.hstack(cols))
    if filas:
        cv2.imwrite(salida, cv2.resize(np.vstack(filas), None, fx=0.5, fy=0.5))
        print("montaje:", salida)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--frames", default="900,3000")
    ap.add_argument("--salida", default="montaje.jpg")
    a = ap.parse_args()
    if a.sintetico:
        evalua_sintetico(a.n, a.seed)
    if a.video:
        evalua_real(a.video, [int(x) for x in a.frames.split(",")], a.salida)
