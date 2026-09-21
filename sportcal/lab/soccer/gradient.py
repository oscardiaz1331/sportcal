"""Lineas del campo de futbol por GRADIENTE sobre la imagen COMPLETA, sin region de cesped
ni quitajugadores (que descartaban lineas utiles, ver soccer_eval.etapas_mascara).

Cadena, cada paso visible en los paneles del etiquetador (sportcal/lab/common/annotate_val_app.py):
  respuesta()      R = max(L+/8.5, a+/5, b-/5) continua sobre TODO el frame (R>=1 equivale a
                   pasar los filtros del baseline, pero sin umbral, region ni apertura) y su
                   gradiente vertical |dR/dy| y horizontal |dR/dx|.
  personas()       YOLO (CPU: hay un entrenamiento usando la GPU) -> cajas; sus bordes se ignoran.
  bordes()         Canny sobre R: bordes finos + orientacion del gradiente en cada uno.
  familias         borde de "linea horizontal-ish" si su gradiente esta a <= tol_h de la
                   vertical; de "vertical-ish" si esta a <= tol_v de la horizontal. Con la
                   perspectiva las lineas no respetan esas direcciones, por eso hay tolerancia
                   y las familias se solapan un poco (los duplicados se funden).
  ransac_lineas()  rectas INDIVIDUALES: RANSAC secuencial donde un pixel de borde solo apoya una
                   recta si su gradiente es perpendicular a ella (ademas de estar a <= tol_px).
  hough_lineas()   alternativa: Hough (soccer_field.detecta_lineas) + el mismo apoyo por gradiente.
  empareja()       una linea pintada da DOS bordes paralelos con gradientes opuestos; su centro es
                   la recta media. Bordes sueltos (vallas, publicidad) se quedan como estan.
  elipse_circulo() fitEllipse robusto sobre lo que las rectas no explican, con lo que SABEMOS del
                   campo: eje mayor casi horizontal, relacion de ejes plausible, tamaño acotado.
  esquinas()       cruces de rectas horizontal-ish x vertical-ish: candidatas a esquinas de area.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from sportcal.paths import ROOT
from sportcal.core import camera as CAM
from sportcal.core import fitting as FIT
from sportcal.sports.soccer import field as TPL

W_TRAB = 960
UMBRALES = (5.0, 5.0, 8.5)      # a+, b-, L+ del baseline de futbol del laboratorio
R_MAX = 6.0


def _angdiff(a, b):
    """Diferencia entre dos angulos de EJE (modulo pi), en [0, pi/2]."""
    return np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2)


# ---------------------------------------------------------------- respuesta y personas

def _respuesta_de(im, region, sigma):
    """R (max de L+/a+/b- normalizados por sus umbrales) de `im` y su gradiente. Con `region` (bool), R vale 0
    fuera: la mediana local de local_chroma/local_l (ventana grande) se contamina con la grada si no se llena
    antes lo de fuera, asi que aqui se rellena con el cesped MEDIANO, que apenas crea borde en el limite."""
    from sportcal.core import surface as P
    if region is not None and region.sum() > 100:
        im = im.copy()
        im[~region] = np.median(im[region], axis=0).astype(np.uint8)
    win = max(11, int(61 * im.shape[1] / 1920.0) | 1)
    da, db = P.local_chroma(im, win=win)
    dl = P.local_l(im, win=win)
    a_min, b_min, l_min = UMBRALES
    R = np.maximum.reduce([np.maximum(da, 0) / a_min, np.maximum(-db, 0) / b_min,
                           np.maximum(dl, 0) / l_min]).astype(np.float32)
    R = np.minimum(R, R_MAX)
    if region is not None:
        R = R * region
    Rs = cv2.GaussianBlur(R, (0, 0), sigma)
    gx = cv2.Sobel(Rs, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(Rs, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return {"R": R, "Rs": Rs, "gx": gx, "gy": gy}


def respuesta(img, w_trab=W_TRAB, sigma=1.0, chi2=10.17):
    """{im, R, Rs, gx, gy, verde, region, sin_mascara}: frame reducido a w_trab y respuesta de linea continua con
    su gradiente. La MASCARA DE CESPED (gaussiana robusta con semilla de cesped, la del baseline) se aplica
    ANTES de calcular R y el gradiente: no tiene sentido calcular nada fuera del terreno de juego (grada,
    anuncios). `sin_mascara` guarda la version sobre la imagen entera, solo para comparar (ver `procesa`)."""
    from sportcal.core import surface as P
    h0, w0 = img.shape[:2]
    im = cv2.resize(img, (w_trab, int(round(h0 * w_trab / w0))), interpolation=cv2.INTER_AREA)
    region = P.play_region(P.robust_surface(im, chi2=chi2, surface="grass")) > 0
    out = _respuesta_de(im, region, sigma)
    out.update({"im": im, "verde": P.grass_hsv(im).astype(np.float32), "region": region.astype(np.uint8),
                "sin_mascara": _respuesta_de(im, None, sigma)})
    return out


def _resp_sel(resp, usar_region):
    """La respuesta con la mascara antes de R (usar_region) o la de la imagen entera."""
    return resp if (usar_region or "sin_mascara" not in resp) else {**resp, **resp["sin_mascara"]}


def modelo_personas(nombre="yolo26s.pt"):
    from ultralytics import YOLO
    return YOLO(str(ROOT / nombre))


def personas(img, modelo, w_trab=W_TRAB, imgsz=1280, conf=0.25):
    """(n,5) x1,y1,x2,y2,conf de las personas, en pixeles de trabajo. SIEMPRE en CPU:
    la GPU la ocupa un entrenamiento (CLAUDE.md: no cargar un segundo modelo)."""
    r = modelo.predict(img, imgsz=imgsz, conf=conf, classes=[0], device="cpu", verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 5))
    s = w_trab / float(img.shape[1])
    return np.c_[r.boxes.xyxy.cpu().numpy() * s, r.boxes.conf.cpu().numpy()]


# ---------------------------------------------------------------- bordes y familias

def bordes(resp, cajas=None, canny=(20, 60), margen_caja=0.06, usar_region=True, margen_region=2):
    """Bordes finos (Canny sobre R) como puntos: pts (n,2) x,y; ax (n,) eje del gradiente
    [0,pi); g (n,2) vector gradiente. Se descartan los bordes dentro de las cajas de personas y,
    con usar_region, los que caen fuera de la region de cesped (erosionada margen_region px)."""
    u8 = np.clip(resp["Rs"] / R_MAX * 255.0, 0, 255).astype(np.uint8)
    e = cv2.Canny(u8, canny[0], canny[1], L2gradient=True) > 0
    if usar_region and "region" in resp:
        k = 2 * int(margen_region) + 1
        e &= cv2.erode(resp["region"], np.ones((k, k), np.uint8)) > 0
    if cajas is not None and len(cajas):
        h, w = e.shape
        for x1, y1, x2, y2, _ in cajas:
            mx, my = margen_caja * (x2 - x1), margen_caja * (y2 - y1)
            e[max(0, int(y1 - my)):min(h, int(y2 + my) + 1), max(0, int(x1 - mx)):min(w, int(x2 + mx) + 1)] = False
    ys, xs = np.nonzero(e)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    g = np.stack([resp["gx"][ys, xs], resp["gy"][ys, xs]], 1).astype(np.float64)
    ax = np.arctan2(g[:, 1], g[:, 0]) % np.pi
    return {"pts": pts, "ax": ax, "g": g, "mapa": e}


def familias(ed, tol_h_deg=50.0, tol_v_deg=50.0):
    """(idx_horizontal, idx_vertical): borde de linea horizontal-ish si su gradiente esta a
    <= tol_h de la vertical; vertical-ish si esta a <= tol_v de la horizontal. Solapan."""
    ax = ed["ax"]
    return (np.flatnonzero(_angdiff(ax, np.pi / 2) <= np.radians(tol_h_deg)),
            np.flatnonzero(_angdiff(ax, 0.0) <= np.radians(tol_v_deg)))


# ---------------------------------------------------------------- rectas individuales

def _linea(pts_in, g_in):
    """Recta por minimos cuadrados sobre los puntos que la apoyan: dict con abc (a*x+b*y+c=0,
    a^2+b^2=1), n (normal), p0 (un punto), d (direccion), ext (t_min,t_max a lo largo de d),
    n_in, signo (lado, respecto a la normal, hacia el que crece R: +1 / -1) y los puntos."""
    vx, vy, x0, y0 = cv2.fitLine(pts_in.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    d = np.array([vx, vy], float)
    n = np.array([-vy, vx], float)
    p0 = np.array([x0, y0], float)
    t = (pts_in - p0) @ d
    signo = 1.0 if float(np.mean(g_in @ n)) >= 0 else -1.0
    return {"abc": np.array([n[0], n[1], -float(n @ p0)]), "n": n, "p0": p0, "d": d,
            "ext": (float(t.min()), float(t.max())), "n_in": len(pts_in), "signo": signo, "pts": pts_in}


def _apoyo(abc, pts, ax, tol_px, tol_ang):
    """Indices de los puntos que apoyan una recta: cerca de ella Y con gradiente perpendicular."""
    d = np.abs(pts @ abc[:2] + abc[2])
    nang = np.arctan2(abc[1], abc[0]) % np.pi
    return np.flatnonzero((d < tol_px) & (_angdiff(ax, nang) < tol_ang))


def ransac_lineas(ed, idx, tol_px=2.0, tol_ang_deg=15.0, n_iter=250, min_in=35, min_len=40.0, n_max=10, seed=0,
                  max_sub=4000):
    """RANSAC secuencial sobre los bordes `idx`: en cada vuelta se busca la recta con mas
    apoyo (cerca + gradiente perpendicular), se ajusta por minimos cuadrados, se retiran sus
    puntos y se repite. El segundo punto de cada muestra se elige entre los alineados con la
    direccion de linea que marca el gradiente del primero: casi toda muestra es una recta
    plausible, asi que bastan pocas iteraciones (con pares uniformes seria hopeless)."""
    rng = np.random.default_rng(seed)
    tol_ang = np.radians(tol_ang_deg)
    pts, ax, g = ed["pts"][idx], ed["ax"][idx], ed["g"][idx]
    resto = np.arange(len(pts))
    lineas = []
    for _ in range(n_max):
        if len(resto) < min_in:
            break
        P, A = pts[resto], ax[resto]
        # las hipotesis se puntuan sobre una submuestra (el coste por iteracion es O(n) y con 20k bordes
        # eran 15-19 s por frame); el apoyo final y el ajuste usan TODOS los puntos
        sub = np.arange(len(resto)) if len(resto) <= max_sub else rng.choice(len(resto), max_sub, replace=False)
        Ps, As = P[sub], A[sub]
        dirl = (As + np.pi / 2) % np.pi
        mejor, mejor_n = None, 0
        for _ in range(n_iter):
            i = int(rng.integers(len(sub)))
            v = Ps - Ps[i]
            dist = np.hypot(v[:, 0], v[:, 1])
            ok = (dist > 15) & (dist < 500) & (_angdiff(np.arctan2(v[:, 1], v[:, 0]) % np.pi, dirl[i]) < tol_ang)
            js = np.flatnonzero(ok)
            if len(js) == 0:
                continue
            j = js[int(rng.integers(len(js)))]
            dv = Ps[j] - Ps[i]
            nrm = np.array([-dv[1], dv[0]]) / np.hypot(*dv)
            abc = np.array([nrm[0], nrm[1], -float(nrm @ Ps[i])])
            n_in = len(_apoyo(abc, Ps, As, tol_px, tol_ang)) * len(resto) / len(sub)     # apoyo estimado en todos
            if n_in > mejor_n:
                mejor, mejor_n = abc, n_in
        if mejor is None or mejor_n < min_in:
            break
        ap = _apoyo(mejor, P, A, tol_px, tol_ang)
        ln = _linea(P[ap], g[resto][ap])
        ap = _apoyo(ln["abc"], P, A, tol_px, tol_ang)              # reajustada: reevaluar el apoyo
        if len(ap) >= min_in:
            ln = _linea(P[ap], g[resto][ap])
            if ln["ext"][1] - ln["ext"][0] >= min_len:
                lineas.append(ln)
        resto = np.delete(resto, ap)
    return lineas


def hough_lineas(ed, idx, shape, tol_px=2.0, tol_ang_deg=15.0, min_in=35, min_len=40.0, n_max=10):
    """Alternativa: Hough sobre el mapa de bordes de la familia (FIT.detect_lines) y, para
    cada recta candidata, el mismo apoyo por gradiente que en RANSAC."""
    m = np.zeros(shape, np.uint8)
    p = ed["pts"][idx].astype(int)
    m[p[:, 1], p[:, 0]] = 1
    cand = FIT.detect_lines(m, n_max)
    tol_ang = np.radians(tol_ang_deg)
    pts, ax, g = ed["pts"][idx], ed["ax"][idx], ed["g"][idx]
    lineas = []
    for abc in cand:
        ap = _apoyo(abc, pts, ax, tol_px, tol_ang)
        if len(ap) >= min_in:
            ln = _linea(pts[ap], g[ap])
            if ln["ext"][1] - ln["ext"][0] >= min_len:
                lineas.append(ln)
    return lineas


def fld_lineas(resp, u8, length=20, merge=True, tol_frac=0.012, ang_deg=4.0, n_max=14):
    """Lines from cv2.ximgproc.FastLineDetector run on the 8-bit line response `u8`.

    The detector returns many short segments; they are grouped by angle and position (all segment ends within
    `tol_frac` of the image width of the group line, angle within `ang_deg`) and each group is fitted like the
    other methods do, using points sampled along the segments and the gradient of R there, so the result can be
    paired, annotated and filtered exactly like RANSAC or Hough lines. The `n_max` longest groups are kept."""
    det = cv2.ximgproc.createFastLineDetector(int(length), 1.41421356, 30, 60, 3, bool(merge))
    raw = det.detect(u8)
    if raw is None:
        return []
    segs = np.asarray(raw, float).reshape(-1, 4)
    h, w = resp["R"].shape
    seg_len = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    seg_ang = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]) % np.pi
    free = np.ones(len(segs), bool)
    groups = []
    for i in np.argsort(-seg_len):
        if not free[i] or seg_len[i] < 8:
            continue
        n = np.array([-np.sin(seg_ang[i]), np.cos(seg_ang[i])])
        c = -(n @ segs[i, :2])
        near = (np.abs(segs[:, :2] @ n + c) < tol_frac * w) & (np.abs(segs[:, 2:] @ n + c) < tol_frac * w)
        member = free & near & (_angdiff(seg_ang, seg_ang[i]) < np.radians(ang_deg))
        free &= ~member
        pts = []
        for x1, y1, x2, y2 in segs[member]:
            t = np.linspace(0.0, 1.0, max(2, int(np.hypot(x2 - x1, y2 - y1) / 2)))
            pts.append(np.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], 1))
        pts = np.concatenate(pts)
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        grad = np.stack([resp["gx"][yi, xi], resp["gy"][yi, xi]], 1).astype(float)
        groups.append((seg_len[member].sum(), _linea(pts, grad)))
    groups.sort(key=lambda t: -t[0])
    return [ln for _, ln in groups[:n_max]]


def _dedup(lineas, ang_deg=2.0, dist_px=3.0):
    """Funde las rectas casi identicas (las familias solapan: la misma recta sale dos veces)."""
    out = []
    for ln in sorted(lineas, key=lambda l: -l["n_in"]):
        dup = False
        for o in out:
            da = _angdiff(np.arctan2(ln["abc"][1], ln["abc"][0]) % np.pi, np.arctan2(o["abc"][1], o["abc"][0]) % np.pi)
            if da < np.radians(ang_deg) and abs(o["abc"][:2] @ ln["p0"] + o["abc"][2]) < dist_px:
                dup = True
                break
        if not dup:
            out.append(ln)
    return out


def empareja(lineas, sep_min=1.5, sep_max=14.0, ang_deg=3.0):
    """Una linea PINTADA da dos bordes paralelos con gradientes de signo opuesto (R sube hacia
    la linea desde cada lado). Se funden en su recta media; los que no tienen pareja se quedan.
    Devuelve las rectas con "par": True/False."""
    usadas, out = set(), []
    cand = []
    for i in range(len(lineas)):
        for j in range(i + 1, len(lineas)):
            a, b = lineas[i], lineas[j]
            # los dos bordes de una linea BRILLANTE apuntan el uno hacia el otro: en la normal de a,
            # si b esta a su "derecha" (off>0), el gradiente de a sube hacia +n y el de b hacia -n
            # (al reves si b esta a la izquierda). Si no, es una franja oscura entre dos cosas claras.
            sa = a["signo"]
            sb = b["signo"] * (1 if a["n"] @ b["n"] > 0 else -1)
            off = a["n"] @ (b["p0"] - a["p0"])
            if not ((off > 0 and sa > 0 and sb < 0) or (off < 0 and sa < 0 and sb > 0)):
                continue
            da = _angdiff(np.arctan2(a["abc"][1], a["abc"][0]) % np.pi, np.arctan2(b["abc"][1], b["abc"][0]) % np.pi)
            if da > np.radians(ang_deg):
                continue
            sep = abs(a["abc"][:2] @ b["p0"] + a["abc"][2])
            if not sep_min <= sep <= sep_max:
                continue
            ta = (b["p0"] - a["p0"]) @ a["d"]               # solapamiento a lo largo de la recta
            if min(a["ext"][1], ta + b["ext"][1]) - max(a["ext"][0], ta + b["ext"][0]) < 20:
                continue
            cand.append((-(a["n_in"] + b["n_in"]), i, j))
    for _, i, j in sorted(cand):
        if i in usadas or j in usadas:
            continue
        usadas |= {i, j}
        pts = np.concatenate([lineas[i]["pts"], lineas[j]["pts"]])
        ln = _linea(pts, np.zeros_like(pts))
        ln["signo"], ln["par"] = 0.0, True
        out.append(ln)
    for i, l in enumerate(lineas):
        if i not in usadas:
            l["par"] = False
            out.append(l)
    return sorted(out, key=lambda l: -l["n_in"])


# ---------------------------------------------------------------- perfil de linea pintada

def perfil_linea(R, ln, paso=3.0, o_max=16):
    """(offsets, perfil): R muestreada a lo largo de la recta (cada `paso` px) y a +-o_max px de ella
    en la normal, con la MEDIANA a lo largo de la recta (robusta a jugadores tapando tramos)."""
    from sportcal.core import camera as _SF
    t = np.arange(ln["ext"][0], ln["ext"][1] + 1e-6, paso)
    o = np.arange(-o_max, o_max + 1)
    xy = ln["p0"][None, None, :] + t[:, None, None] * ln["d"][None, None, :] + o[None, :, None] * ln["n"][None, None, :]
    h, w = R.shape
    dentro = (xy[..., 0] >= 0) & (xy[..., 0] <= w - 1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h - 1)
    v = np.where(dentro, _SF.bilinear(R, xy), np.nan)
    if not dentro.any():
        return o, np.zeros(len(o))
    return o, np.nan_to_num(np.nanmedian(v, axis=0))


def clasifica_pintada(o, prof, pico_min=0.8, flanco_max=0.45, c=3, f0=9, f1=15):
    """Una linea PINTADA es un pico brillante y estrecho: pico central >= pico_min (R>=1 = pasa los filtros
    del baseline) y los dos flancos por debajo de flanco_max x pico. Una arista de valla o de grada es un
    escalon (un flanco alto) y una franja ancha no tiene flancos bajos a esa distancia. Devuelve (bool, pico, flanco_max)."""
    pico = float(prof[np.abs(o) <= c].max())
    izq = float(prof[(o <= -f0) & (o >= -f1)].mean())
    der = float(prof[(o >= f0) & (o <= f1)].mean())
    fl = max(izq, der)
    return bool(pico >= pico_min and fl <= flanco_max * pico), pico, fl


def cesped_lados(ln, verde, o0=9, o1=15, paso=3.0):
    """(izq, der): fraccion de cesped a cada lado de la recta (offsets o0..o1 px en la normal,
    media a lo largo del tramo con apoyo)."""
    t = np.arange(ln["ext"][0], ln["ext"][1] + 1e-6, paso)
    h, w = verde.shape
    res = []
    for sgn in (-1, 1):
        vals = []
        for off in range(o0, o1 + 1):
            xy = ln["p0"][None] + t[:, None] * ln["d"][None] + sgn * off * ln["n"][None]
            xi, yi = np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if ok.any():
                vals.append(float(verde[yi[ok], xi[ok]].mean()))
        res.append(float(np.mean(vals)) if vals else 0.0)
    return res[0], res[1]


def anota_lineas(lineas, resp, cesped_min=0.3, pico_min=0.5):
    """Anota cada recta con su perfil y con el CESPED A LOS DOS LADOS, y decide si es una recta de
    campo: hay cesped (>= cesped_min) a ambos lados Y pico central >= pico_min (R; 1 = pasa los
    filtros del baseline). Una arista de grada o de valla no tiene cesped a los dos lados, y es lo que
    mas separa (mediana 0.63 en las rectas que coinciden con la plantilla frente a 0.05 en el resto;
    con pico>=0.5 quedan 8 de 9 correctas y 2 de 25 espurias en f900+f3000 de soccer.mp4 -- umbrales
    tomados de esos MISMOS 34 rectas, sin validar). El perfil de linea pintada estricto (pico>=0.8,
    flancos<=0.45 pico) tiraba las lineas lejanas, mas tenues: se conserva como dato ("pintada")."""
    for ln in lineas:
        o, prof = perfil_linea(resp["R"], ln)
        ln["pintada"], ln["pico"], ln["flanco"] = clasifica_pintada(o, prof)
        ln["perfil"] = prof
        ln["cesped_izq"], ln["cesped_der"] = cesped_lados(ln, resp["verde"])
        ln["campo"] = bool(min(ln["cesped_izq"], ln["cesped_der"]) >= cesped_min and ln["pico"] >= pico_min)
    return lineas


# ---------------------------------------------------------------- detectores de elipses (findEllipses / EdgeDrawing)

PRIOR_ELIPSE = {"a_min": 0.05, "a_max": 0.65, "ratio": (0.08, 0.95), "theta_deg": 25.0}


def entrada_u8(resp, cajas, techo, margen_caja=0.06):
    """Imagen de 8 bits que reciben los detectores: R suavizada (ya con la mascara de cesped aplicada) con R=techo
    a 255, y las cajas de personas puestas a 0 para que no vean siluetas. `techo` cambia el contraste: los
    detectores traen umbrales de gradiente internos y una linea tenue queda por debajo si la escala es grande."""
    u8 = np.clip(resp["Rs"] / float(techo) * 255.0, 0, 255).astype(np.uint8)
    h, w = u8.shape
    for x1, y1, x2, y2, _ in (cajas if cajas is not None else []):
        mx, my = margen_caja * (x2 - x1), margen_caja * (y2 - y1)
        u8[max(0, int(y1 - my)):min(h, int(y2 + my) + 1), max(0, int(x1 - mx)):min(w, int(x2 + mx) + 1)] = 0
    return u8


def _dict_elipse(cx, cy, a, b, ang_deg, origen, **extra):
    """(cx, cy, a, b, theta): semiejes con a >= b y theta en radianes (mod pi), como _elipse_de."""
    if b > a:
        a, b, ang_deg = b, a, ang_deg + 90.0
    return {"e": np.array([cx, cy, a, b, np.radians(ang_deg) % np.pi]), "origen": origen, **extra}


def elipses_edgedrawing(u8, grad=36, ancla=8, camino_min=10):
    """EdgeDrawing.detectEllipses (EDCircles, cv2.ximgproc). Devuelve una lista de dicts. Las filas que da son
    (x, y, 0, a, b, angulo_grados) para elipses y (x, y, r, 0, 0, 0) para circulos."""
    ed = cv2.ximgproc.createEdgeDrawing()
    prm = cv2.ximgproc_EdgeDrawing_Params()
    prm.GradientThresholdValue, prm.AnchorThresholdValue, prm.MinPathLength = int(grad), int(ancla), int(camino_min)
    ed.setParams(prm)
    ed.detectEdges(u8)
    filas = ed.detectEllipses()
    out = []
    for r in (np.zeros((0, 6)) if filas is None else np.asarray(filas).reshape(-1, 6)):
        if r[3] > 0 and r[4] > 0:
            out.append(_dict_elipse(r[0], r[1], r[3], r[4], r[5], "edgedrawing"))
        elif r[2] > 0:
            out.append(_dict_elipse(r[0], r[1], r[2], r[2], 0.0, "edgedrawing", circulo=True))
    return out


def elipses_find(u8, score=0.3, fiabilidad=0.3):
    """cv2.ximgproc.findEllipses. Filas (x, y, a, b, score, fiabilidad): NO devuelve la orientacion, se asume
    horizontal (theta=0), razonable para una retransmision sin roll. Devuelve una lista de dicts."""
    e = cv2.ximgproc.findEllipses(u8, scoreThreshold=float(score), reliabilityThreshold=float(fiabilidad))
    e = e[0] if isinstance(e, tuple) else e
    return [_dict_elipse(r[0], r[1], r[2], r[3], 0.0, "findEllipses", score=float(r[4]), fiabilidad=float(r[5]))
            for r in (np.zeros((0, 6)) if e is None else np.asarray(e).reshape(-1, 6))]


def puntos_elipse(e, n=80):
    """n puntos sobre la elipse (cx, cy, a, b, theta): sirven como evidencia para el solver."""
    cx, cy, a, b, th = e
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = a * np.cos(t), b * np.sin(t)
    return np.stack([cx + x * np.cos(th) - y * np.sin(th), cy + x * np.sin(th) + y * np.cos(th)], 1)


def elige_elipse(elipses, shape, det="auto"):
    """La elipse que se pasa al solver como evidencia: de entre las que cumplen el prior, la de mayor area.
    det: "edgedrawing", "find", "fit" (el fitEllipse antiguo) o "auto" (edgedrawing, si no hay, find)."""
    h, w = shape
    orden = {"auto": ("edgedrawing", "find"), "edgedrawing": ("edgedrawing",), "find": ("find",), "fit": ("fit",)}[det]
    for nombre in orden:
        ok = [d for d in elipses.get(nombre, []) if _elipse_ok(d["e"], w, h, PRIOR_ELIPSE)]
        if ok:
            d = max(ok, key=lambda d: d["e"][2] * d["e"][3])
            pts = d["pts"] if "pts" in d else puntos_elipse(d["e"])
            return {"e": d["e"], "pts": pts, "n_in": len(pts), "origen": nombre}
    return None


# ---------------------------------------------------------------- elipse (fitEllipse robusto: el ANTIGUO, solo para comparar)

def _elipse_de(pts):
    """cv2.fitEllipse -> (cx, cy, a, b, theta_rad) con a >= b semiejes; None si falla."""
    if len(pts) < 5:
        return None
    try:
        (cx, cy), (d1, d2), ang = cv2.fitEllipse(pts.astype(np.float32))
    except cv2.error:
        return None
    a, b = max(d1, d2) / 2.0, min(d1, d2) / 2.0
    th = np.radians(ang if d1 >= d2 else ang + 90.0)
    return np.array([cx, cy, a, b, th % np.pi])


def _elipse_ok(e, w, h, prior):
    """Lo que sabemos del circulo central visto por una camara de retransmision: la elipse es
    ancha (eje mayor casi horizontal), no un circulo perfecto ni una raya, y no ocupa media imagen."""
    cx, cy, a, b, th = e
    if not np.isfinite(e).all() or b < 3:
        return False
    return (prior["a_min"] * w <= 2 * a <= prior["a_max"] * w and prior["ratio"][0] <= b / a <= prior["ratio"][1]
            and _angdiff(th, 0.0) <= np.radians(prior["theta_deg"]) and -0.2 * w <= cx <= 1.2 * w and -0.2 * h <= cy <= 1.2 * h)


def _distancia_orient(e, P, A):
    """(d, dang): distancia geometrica aproximada de cada punto a la elipse (|f|/|grad f|) y
    diferencia entre su gradiente y la normal de la elipse en ese punto."""
    cx, cy, a, b, th = e
    c, s = np.cos(th), np.sin(th)
    u = (P[:, 0] - cx) * c + (P[:, 1] - cy) * s
    v = -(P[:, 0] - cx) * s + (P[:, 1] - cy) * c
    f = (u / a) ** 2 + (v / b) ** 2 - 1.0
    gu, gv = 2 * u / a ** 2, 2 * v / b ** 2
    gn = np.maximum(np.hypot(gu, gv), 1e-9)
    nx, ny = gu * c - gv * s, gu * s + gv * c                   # normal en coordenadas de imagen
    return np.abs(f) / gn, _angdiff(np.arctan2(ny, nx) % np.pi, A)


def elipse_circulo(ed, excluir, shape, tol_px=3.5, tol_ang_deg=30.0, prior=None):
    """Elipse del circulo central sobre los bordes que las rectas NO explican. Semillas: cada
    componente conexa grande de esos bordes (fitEllipse); se refina 3 veces con todos los bordes
    que la apoyan (cerca + gradiente normal) y gana la de mas apoyo que cumpla `prior`.
    Devuelve dict (e=(cx,cy,a,b,theta), n_in, pts) o None."""
    h, w = shape
    prior = prior or {"a_min": 0.05, "a_max": 0.65, "ratio": (0.08, 0.95), "theta_deg": 25.0}
    libre = np.ones(len(ed["pts"]), bool)
    libre[excluir] = False
    P, A = ed["pts"][libre], ed["ax"][libre]
    if len(P) < 30:
        return None
    m = np.zeros((h, w), np.uint8)
    ip = P.astype(int)
    m[ip[:, 1], ip[:, 0]] = 1
    _, lab = cv2.connectedComponents(cv2.dilate(m, np.ones((3, 3), np.uint8)), connectivity=8)
    lp = lab[ip[:, 1], ip[:, 0]]
    tol_ang = np.radians(tol_ang_deg)
    mejor = None
    for k in np.unique(lp):
        sel = lp == k
        if k == 0 or sel.sum() < 25 or np.ptp(P[sel], axis=0).max() < 0.04 * w:
            continue
        e = _elipse_de(P[sel])
        for _ in range(3):
            if e is None or not _elipse_ok(e, w, h, prior):
                e = None
                break
            d, da = _distancia_orient(e, P, A)
            ap = (d < tol_px) & (da < tol_ang)
            if ap.sum() < 12:
                e = None
                break
            e = _elipse_de(P[ap])
        if e is None or not _elipse_ok(e, w, h, prior):
            continue
        d, da = _distancia_orient(e, P, A)
        ap = (d < tol_px) & (da < tol_ang)
        if mejor is None or ap.sum() > mejor["n_in"]:
            mejor = {"e": e, "n_in": int(ap.sum()), "pts": P[ap]}
    return mejor


# ---------------------------------------------------------------- esquinas y orquestacion

def esquinas(lin_h, lin_v, shape, margen=0.05, holgura=40.0):
    """Cruces recta horizontal-ish x recta vertical-ish que caen en el frame (+margen) y cerca
    del tramo con apoyo de las DOS rectas: candidatas a esquinas de area, cruces con la central..."""
    h, w = shape
    out = []
    for a in lin_h:
        for b in lin_v:
            q = np.cross(a["abc"], b["abc"])
            if abs(q[2]) < 1e-9:
                continue
            p = q[:2] / q[2]
            if not (-margen * w <= p[0] <= (1 + margen) * w and -margen * h <= p[1] <= (1 + margen) * h):
                continue
            ok = True
            for ln in (a, b):
                t = (p - ln["p0"]) @ ln["d"]
                ok &= ln["ext"][0] - holgura <= t <= ln["ext"][1] + holgura
            if ok:
                out.append(p)
    return np.array(out) if out else np.zeros((0, 2))


def procesa(img, modelo=None, metodo="ransac", canny=(20, 60), tol_h=50.0, tol_v=50.0, usar_personas=True,
            tol_px=2.0, min_in=35, n_max=10, resp=None, cajas=None, cesped_min=0.3, pico_min=0.5,
            usar_region=True, margen_region=2, seed=0, det_elipse="auto", ed_params=(36, 8, 10), ed_techo=6.0,
            fe_params=(0.3, 0.3), fe_techo=2.0, fld_params=(20, True), fld_techo=3.0):
    """Todo el proceso sobre un frame BGR. `resp` y `cajas` se pueden pasar ya calculados
    (son lo caro: respuesta ~0.5 s, YOLO en CPU ~1-3 s) para reajustar solo los parametros.

    metodo: "ransac" (RANSAC sobre los bordes, por familia), "hough" (Hough sobre los bordes, por familia) o "fld"
    (FastLineDetector sobre la respuesta de linea, `fld_params` = (longitud minima, unir segmentos), `fld_techo` = valor de R
    que se manda a 255); con "fld" las rectas se reparten en las dos familias por su inclinacion con las mismas tolerancias."""
    resp = _resp_sel(resp if resp is not None else respuesta(img), usar_region)
    if cajas is None:
        cajas = personas(img, modelo) if (usar_personas and modelo is not None) else np.zeros((0, 5))
    elif not usar_personas:
        cajas = np.zeros((0, 5))
    ed = bordes(resp, cajas, canny, usar_region=usar_region, margen_region=margen_region)
    ih, iv = familias(ed, tol_h, tol_v)
    shape = resp["R"].shape
    if metodo == "fld":
        fld = fld_lineas(resp, entrada_u8(resp, cajas, fld_techo), *fld_params)
        direction = lambda ln: np.arctan2(ln["abc"][0], -ln["abc"][1]) % np.pi          # angle of the line itself
        lh = _dedup([ln for ln in fld if _angdiff(direction(ln), 0.0) <= np.radians(tol_h)])
        lv = _dedup([ln for ln in fld if _angdiff(direction(ln), np.pi / 2) <= np.radians(tol_v)])
    else:
        f = (lambda idx: ransac_lineas(ed, idx, tol_px=tol_px, min_in=min_in, n_max=n_max, seed=seed)) if metodo == "ransac" else \
            (lambda idx: hough_lineas(ed, idx, shape, tol_px=tol_px, min_in=min_in, n_max=n_max))
        lh, lv = _dedup(f(ih)), _dedup(f(iv))
    todas = _dedup(lh + lv)
    exp = np.zeros(len(ed["pts"]), bool)
    for ln in todas:
        exp[_apoyo(ln["abc"], ed["pts"], ed["ax"], tol_px * 1.5, np.radians(20))] = True
    lh_c = anota_lineas(empareja(lh), resp, cesped_min, pico_min)
    lv_c = anota_lineas(empareja(lv), resp, cesped_min, pico_min)
    ell_fit = elipse_circulo(ed, np.flatnonzero(exp), shape)
    u_ed, u_fe = entrada_u8(resp, cajas, ed_techo), entrada_u8(resp, cajas, fe_techo)
    elipses = {"edgedrawing": elipses_edgedrawing(u_ed, *ed_params), "find": elipses_find(u_fe, *fe_params),
               "fit": [] if ell_fit is None else [{"e": ell_fit["e"], "origen": "fit", "pts": ell_fit["pts"]}]}
    ell = elige_elipse(elipses, shape, det_elipse)
    return {"resp": resp, "cajas": cajas, "ed": ed, "usar_region": usar_region, "lin_h": lh_c, "lin_v": lv_c, "elipse": ell,
            "elipses": elipses, "entradas": {"edgedrawing": u_ed, "find": u_fe},
            "params_elipse": {"ed": tuple(ed_params), "ed_techo": ed_techo, "fe": tuple(fe_params), "fe_techo": fe_techo, "det": det_elipse},
            "esquinas": esquinas([l for l in lh_c if l["campo"]], [l for l in lv_c if l["campo"]], shape),
            "idx_h": ih, "idx_v": iv}


# ---------------------------------------------------------------- evaluacion contra una H

def evalua_lineas(lineas, H, shape, tol_px=4.0, tol_ang_deg=3.0, min_vis=0.08):
    """Contra una H (mundo->pixel de trabajo): (recall, precision, n_visibles, n_detectadas).
    recall: fraccion de las rectas de la plantilla visibles (>= min_vis del ancho) para las que
    hay una detectada casi coincidente (angulo y distancia de sus extremos visibles).
    precision: fraccion de las detectadas que coinciden con alguna recta de la plantilla."""
    h, w = shape
    vis = []
    for nombre, pl in TPL.polylines().items():
        if "circulo" in nombre or "arco" in nombre:
            continue
        for k in range(len(pl) - 1):
            s = np.linspace(0, 1, 60)[:, None]
            seg = pl[k] + s * (pl[k + 1] - pl[k])
            xy, den = CAM.project(np.asarray(H, float)[None], seg)
            xy, den = xy[0], den[0]
            ok = (den > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            if ok.sum() >= 2 and np.hypot(*(xy[ok][-1] - xy[ok][0])) >= min_vis * w:
                vis.append((xy[ok][0], xy[ok][-1]))
    if not vis or not lineas:
        return 0.0, 0.0, len(vis), len(lineas)

    def coincide(ln, p, q):
        v = q - p
        ang_t = np.arctan2(v[1], v[0]) % np.pi
        ang_l = np.arctan2(ln["abc"][0], -ln["abc"][1]) % np.pi
        return (_angdiff(ang_t, ang_l) < np.radians(tol_ang_deg)
                and max(abs(ln["abc"][:2] @ p + ln["abc"][2]), abs(ln["abc"][:2] @ q + ln["abc"][2])) < tol_px)

    rec = np.mean([any(coincide(ln, p, q) for ln in lineas) for p, q in vis])
    pre = np.mean([any(coincide(ln, p, q) for p, q in vis) for ln in lineas])
    return float(rec), float(pre), len(vis), len(lineas)


# ---------------------------------------------------------------- paneles para el etiquetador

def _seg(v, ln, color, grosor):
    p1, p2 = ln["p0"] + ln["d"] * ln["ext"][0], ln["p0"] + ln["d"] * ln["ext"][1]
    cv2.line(v, tuple(int(a) for a in p1), tuple(int(a) for a in p2), color, grosor, cv2.LINE_AA)


def tabla_lineas(res):
    """Una fila por recta detectada, para st.dataframe."""
    filas = []
    for fam, lst in (("horizontal-ish", res["lin_h"]), ("vertical-ish", res["lin_v"])):
        for k, ln in enumerate(lst):
            filas.append({"familia": fam, "n": k + 1, "de campo": "si" if ln["campo"] else "no",
                          "longitud px": round(ln["ext"][1] - ln["ext"][0]), "apoyo (bordes)": ln["n_in"],
                          "pico R": round(ln["pico"], 2), "cesped izq": round(ln["cesped_izq"], 2),
                          "cesped der": round(ln["cesped_der"], 2), "par de bordes": "si" if ln["par"] else ""})
    return filas


def tabla_elipses(res):
    """Una fila por elipse detectada (todos los detectores), para st.dataframe."""
    h, w = res["resp"]["R"].shape
    elegida = res["elipse"]["e"] if res["elipse"] is not None else None
    filas = []
    for det, lst in res["elipses"].items():
        for d in lst:
            cx, cy, a, b, th = d["e"]
            filas.append({"detector": {"edgedrawing": "EdgeDrawing", "find": "findEllipses", "fit": "fitEllipse (antiguo)"}[det],
                          "centro x": round(cx), "centro y": round(cy), "semieje a": round(a), "semieje b": round(b),
                          "angulo °": round(float(np.degrees(th)) % 180, 1) if det != "find" else float("nan"),   # findEllipses no lo da
                          "score / fiabilidad": ("{:.2f} / {:.2f}".format(d["score"], d["fiabilidad"]) if "score" in d else ""),
                          "cumple el prior": "si" if _elipse_ok(d["e"], w, h, PRIOR_ELIPSE) else "no",
                          "elegida": "SI" if (elegida is not None and np.allclose(d["e"], elegida)) else ""})
    return filas


def paneles_grad(res, mostrar_descartadas=True, mostrar_fit=False):
    """[(titulo, imagen_rgb, texto)] de cada paso: R, |dR/dy|, |dR/dx| (imagen completa), personas y
    bordes por familia, rectas individuales, elipse y esquinas."""
    rs, im = res["resp"], res["resp"]["im"]
    rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    out = []
    # con region, lo de fuera (grada, publicidad) se ve muy atenuado: sigue calculado pero no se usa
    fuera = np.where((res.get("usar_region") and "region" in rs and cv2.erode(rs["region"], np.ones((5, 5), np.uint8)) == 0)[..., None]
                     if res.get("usar_region") and "region" in rs else np.zeros(im.shape[:2] + (1,), bool), 0.25, 1.0)

    def esc(g):
        top = float(np.percentile(g[g > 0], 99)) if (g > 0).any() else 1.0
        return np.clip(g / max(top, 1e-6), 0, 1)[..., None]

    out.append(("Respuesta de linea R calculada DESPUES de aplicar la mascara de cesped (fuera del campo vale 0)"
                if res.get("usar_region") else "Respuesta de linea R sobre la imagen entera (sin mascara de cesped)",
                rgb((np.clip(rs["R"] / R_MAX, 0, 1)[..., None] * fuera * np.array([255, 255, 255])).astype(np.uint8)),
                "R = max(L+/8.5, a+/5, b-/5), continua: R>=1 equivale a pasar los filtros del baseline. "
                "{:.1f}% del frame tiene R>=1.".format(100.0 * float((rs["R"] >= 1).mean()))))
    out.append(("Gradiente vertical |dR/dy|: bordes de lineas horizontal-ish",
                rgb((esc(np.abs(rs["gy"])) * fuera * np.array([255, 255, 0])).astype(np.uint8)),
                "Una linea pintada horizontal-ish da DOS bordes paralelos; las de la grada y las vallas tambien salen."))
    out.append(("Gradiente horizontal |dR/dx|: bordes de lineas vertical-ish",
                rgb((esc(np.abs(rs["gx"])) * fuera * np.array([255, 0, 255])).astype(np.uint8)),
                "Igual en la otra direccion. Los costados de un jugador tambien encienden este panel: por eso los jugadores "
                "se quitan con el detector de personas y no con el gradiente."))

    ed = res["ed"]
    v4 = (im * 0.45 * fuera).astype(np.uint8)
    ph, pv = ed["pts"][res["idx_h"]].astype(int), ed["pts"][res["idx_v"]].astype(int)
    v4[pv[:, 1], pv[:, 0]] = (255, 0, 255)
    v4[ph[:, 1], ph[:, 0]] = np.where((v4[ph[:, 1], ph[:, 0]] == (255, 0, 255)).all(1)[:, None], (255, 255, 255), (255, 255, 0))
    for x1, y1, x2, y2, c in res["cajas"]:
        cv2.rectangle(v4, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 1)
    out.append(("Personas (YOLO en CPU) y bordes finos por familia (cian = horizontal-ish, magenta = vertical-ish, blanco = ambas)",
                rgb(v4),
                "{} personas: sus bordes se ignoran. {} bordes finos (Canny sobre R); las familias solapan a proposito "
                "porque la perspectiva inclina las lineas.".format(len(res["cajas"]), len(ed["pts"]))))

    v5 = (im * 0.6).astype(np.uint8)
    n_campo = 0
    for (lst, col) in ((res["lin_h"], (255, 200, 0)), (res["lin_v"], (255, 0, 255))):
        for k, ln in enumerate(lst):
            if ln["campo"]:
                _seg(v5, ln, col, 3)
                n_campo += 1
                mid = ln["p0"] + ln["d"] * 0.5 * (ln["ext"][0] + ln["ext"][1])
                cv2.putText(v5, str(k + 1), (int(mid[0]) + 5, int(mid[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            elif mostrar_descartadas:
                _seg(v5, ln, (150, 150, 150), 1)
    out.append(("Rectas individuales: de campo = gruesas de color, descartadas = finas grises", rgb(v5),
                "{} de campo de {} detectadas. De campo = hay cesped a ambos lados Y pico central alto: una arista de grada o de "
                "valla no tiene cesped a los dos lados. La tabla de abajo da los numeros de cada una.".format(
                    n_campo, len(res["lin_h"]) + len(res["lin_v"]))))

    pe = res["params_elipse"]
    colores = {"edgedrawing": (0, 0, 255), "find": (0, 255, 0), "fit": (0, 255, 255)}

    def dibuja_elipses(v, lst, color, grosor=2):
        for d in lst:
            cx, cy, a, b, th = d["e"]
            cv2.ellipse(v, (int(cx), int(cy)), (int(max(a, 1)), int(max(b, 1))), float(np.degrees(th)), 0, 360, color, grosor, cv2.LINE_AA)

    for clave, nombre, txt in (
            ("edgedrawing", "EdgeDrawing.detectEllipses (rojo) sobre su entrada",
             "Entrada: R con techo {:g} y personas a 0. Umbrales: gradiente {}, ancla {}, camino minimo {}. {} elipses. "
             "Detecta elipses/circulos COMPLETOS; los recortados por el borde del frame suelen escaparse."),
            ("find", "findEllipses (verde) sobre su entrada",
             "Entrada: R con techo {:g} y personas a 0. Umbrales: score >= {}, fiabilidad >= {}. {} elipses. No devuelve orientacion (se "
             "dibuja horizontal). Suele dar duplicados: uno por cada borde de la linea pintada.")):
        u = res["entradas"][clave]
        v = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
        dibuja_elipses(v, res["elipses"][clave], colores[clave])
        techo, prm = (pe["ed_techo"], pe["ed"]) if clave == "edgedrawing" else (pe["fe_techo"], pe["fe"])
        out.append((nombre, rgb(v), txt.format(techo, *prm, len(res["elipses"][clave]))))

    v8 = (im * 0.6).astype(np.uint8)
    for clave in ("edgedrawing", "find") + (("fit",) if mostrar_fit else ()):
        dibuja_elipses(v8, res["elipses"][clave], colores[clave], 1)
    txt8 = "Ninguna elipse cumple el prior (eje mayor casi horizontal, relacion 0.08-0.95, tamaño acotado): no hay evidencia de elipse para el solver."
    if res["elipse"] is not None:
        cx, cy, a, b, th = res["elipse"]["e"]
        cv2.ellipse(v8, (int(cx), int(cy)), (int(a), int(b)), float(np.degrees(th)), 0, 360, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.drawMarker(v8, (int(cx), int(cy)), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        txt8 = ("Elegida para el solver (cian, detector '{}' = {}): {:.0f}x{:.0f} px, centro ({:.0f},{:.0f}). Es el circulo central O el arco del "
                "area: con lo que sabemos no se distinguen solos, lo decide la puntuacion del solver.".format(
                    res["elipse"]["origen"], pe["det"], 2 * a, 2 * b, cx, cy))
    for p in res["esquinas"]:
        cv2.circle(v8, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
    out.append(("Todas las elipses sobre el frame: EdgeDrawing rojo · findEllipses verde" + (" · fitEllipse antiguo amarillo" if mostrar_fit else "")
                + " · elegida cian · esquinas amarillas", rgb(v8), txt8 + " {} esquinas candidatas.".format(len(res["esquinas"]))))
    return out
