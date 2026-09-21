"""Prueba: segmentacion adaptativa del hielo (GMM) + realce de lineas por croma local.

Por que no basta el umbral fijo HSV (S<60, V>180):

  - El umbral esta calibrado para UN pabellon. Cambia la temperatura de color del
    hielo, el grading de la emision o la iluminacion y se descuadra.
  - Y sobre todo: las lineas pintadas son MAS saturadas que el hielo, asi que el
    propio umbral que aisla el hielo expulsa justo lo que queremos encontrar.
    Por eso hay que separar dos cosas distintas: el HIELO (color) y la REGION DE
    PISTA (el area conexa que ocupa, lineas y jugadores incluidos).

Que hace esta prueba, por frame:

  1. hielo por GMM: ajusta un modelo de mezclas en Lab con cv2.ml.EM y se queda con
     los componentes claros y poco cromaticos. Sin umbrales absolutos.
  2. region de pista: cierre morfologico + relleno de huecos sobre el hielo, y se
     queda con la componente conexa mayor. Recupera lineas y jugadores.
  3. realce de lineas: Lab respecto al color LOCAL del hielo (mediana de ventana
     grande). El tono absoluto no sirve -- una linea roja bajo el hielo es un rosa
     casi neutro y su H es puro ruido -- pero su desviacion respecto al hielo de al
     lado es estable.
  4. filtro de cresta multiescala: las lineas son estructuras finas y alargadas; los
     jugadores, manchas compactas. Esto separa unas de otras.

Y lo mide, en vez de solo pintarlo: con la homografia del frame se proyecta el
contorno real de la pista y se compara por IoU contra cada metodo de segmentacion,
y la respuesta de linea contra la mascara de sportcal/lab/hockey/make_line_masks.py.

    python -m sportcal.lab.hockey.ice_lines_probe
    python -m sportcal.lab.hockey.ice_lines_probe --n 12 --dataset hockeyrink_nhl
"""
import os

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from sportcal.paths import ROOT

from sportcal.lab.hockey import make_line_masks as MM
from sportcal.sports.hockey import rink
from sportcal.lab.hockey.relabel_reproject import read_label  # noqa: E402
from sportcal.core.surface import (  # noqa: F401  (also re-exported: P.best_region, ...)
    best_region, gmm_surface, ice_hsv, integrate, iou, local_chroma, normals, play_region,
    ridge, robust_surface, solidity,
)


# ---------------------------------------------------------------- segmentacion


def region_kickplate_global(img, thr=12.0, k_frac=0.02, mode="hull"):
    """INTENTO DESCARTADO: usar el zocalo de las vallas para segmentar la region
    de pista entera, en vez del color del hielo. La idea es razonable -- el zocalo
    es 5-16x mas discriminativo que las lineas pintadas (ver fit_homography_lines.py,
    z=79.9/48.5 contra z=5-10) -- pero medido a nivel de FRAME COMPLETO falla peor
    que el umbral de color, y por mucho:

                              hockeyrink              hockeyrink_nhl
                          p50    p10   peor        p50    p10   peor
        best_region      0.914  0.831 0.344       0.885  0.792 0.711   <- lo que hay
        flood-fill anillo0.920  0.007 0.003       0.015  0.002 0.000   <- catastrofico
        casco convexo    0.435  0.263 0.122       0.565  0.208 0.062
        mayor contorno   0.036  0.013 0.003       0.019  0.005 0.002

    La causa, medida directamente: de los pixeles con dB > 8 en un frame tipico,
    solo el 17-38 % estan cerca del contorno real de la pista. El resto (43-69 %)
    esta en la grada, en anuncios de madera con tonos calidos, en publicidad de las
    vallas lejanas o en la piel/luces del graderio -- cualquier cosa amarillenta
    del FRAME ENTERO, no solo el zocalo. El umbral no tiene forma de distinguir
    "zocalo de la pista" de "cualquier otra cosa calida en la imagen" sin saber
    antes donde esta la pista, que es precisamente lo que se le pide que averigue.

    Por eso el flood-fill (que exige un anillo topologicamente cerrado) colapsa en
    cuanto hay una fuga hacia la grada, y el casco convexo se infla hasta cubrir
    cualquier mancha calida lejana por pequeña que sea.

    La leccion, y por que SI funciona en fit_homography_lines.py: la misma senal
    integrada a lo largo de una curva HIPOTETICA conocida (el contorno que predice
    una H candidata) es extremadamente selectiva, porque solo mira esos pixeles
    concretos y descarta todo el resto del frame. Perdiendo esa restriccion de
    forma -- pidiendole que decida sola, sobre la imagen entera, donde esta el
    borde -- la misma senal dej de servir. Funcion conservada solo como referencia;
    no la uses, no esta enganchada a best_region().
    """
    da, db = local_chroma(img, win=int(61 * img.shape[1] / 1920) | 1)
    ring = (db > thr).astype(np.uint8)
    k = max(3, int(round(img.shape[1] * k_frac)) | 1)
    ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    if mode == "hull":
        pts = cv2.findNonZero(ring)
        if pts is None or len(pts) < 500:
            return np.zeros_like(ring)
        out = np.zeros_like(ring)
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
        return out
    # mode == "flood": requiere que el anillo cierre topologicamente
    h, w = ring.shape
    free = (ring == 0).astype(np.uint8)
    seed = next(((x, y) for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))
                if free[y, x] == 1), None)
    if seed is None:
        return np.zeros_like(ring)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flooded = free.copy()
    cv2.floodFill(flooded, mask, seed, 2)
    return (flooded != 2).astype(np.uint8)


# ---------------------------------------------------------------- lineas


def localization_z(R, H, polys, w, h, offsets=(-30, -24, -18, -12, 12, 18, 24, 30)):
    """Cuanto destaca la curva en su sitio frente a la misma curva desplazada.

    Es la metrica que decide si esto sirve: no mide contraste por pixel -- que es
    inutil, porque el rival no es el ruido del sensor sino los jugadores y los
    logos del hielo -- sino si una busqueda sabria distinguir la posicion correcta.
    Se desplaza en la NORMAL de la curva, que es la unica direccion en la que una
    linea se puede confundir consigo misma.
    """
    out = []
    for cls, world in polys:
        if cls not in (2, 3, 6, 7, 8, 9, 10, 11):     # curvas rojas
            continue
        xy, ok = MM.project_points(H, world, w, h)
        if ok.sum() < 60:
            continue
        base = xy[ok]
        v0 = integrate(R, base, w, h)
        if v0 is None:
            continue
        nz = normals(base)
        alt = [integrate(R, base + nz * d, w, h) for d in offsets]
        alt = [a for a in alt if a is not None]
        if len(alt) < 5:
            continue
        out.append((v0 - np.mean(alt)) / (np.std(alt) or 1e-6))
    return out


# ---------------------------------------------------------------- verdad

def rink_outline(p, n=160):
    """Contorno cerrado de las vallas en metros (rectas + arcos de esquina)."""
    L, W, r = p["length"], p["width"], p["corner_r"]
    return np.vstack([
        MM.seg((r, 0), (L - r, 0), n), MM.arc(L - r, r, r, -90, 0, n),
        MM.seg((L, r), (L, W - r), n), MM.arc(L - r, W - r, r, 0, 90, n),
        MM.seg((L - r, W), (r, W), n), MM.arc(r, W - r, r, 90, 180, n),
        MM.seg((0, W - r), (0, r), n), MM.arc(r, r, r, 180, 270, n),
    ])


def clip_halfplane(poly, a, b, c, eps=0.0):
    """Sutherland-Hodgman contra el semiplano a*x + b*y + c > eps.

    El contorno de la pista es convexo, asi que recortarlo con semiplanos lo deja
    convexo y el resultado se puede rellenar directamente.
    """
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    d = poly @ np.array([a, b]) + c - eps
    for i in range(n):
        j = (i + 1) % n
        di, dj = d[i], d[j]
        if di > 0:
            out.append(poly[i])
        if (di > 0) != (dj > 0):
            t = di / (di - dj)
            out.append(poly[i] + t * (poly[j] - poly[i]))
    return np.array(out) if out else np.zeros((0, 2))


def true_region(H, p, w, h, margin=3.0):
    """Region real de la pista proyectando el contorno, recortando por el horizonte.

    Antes esto devolvia None en cuanto un punto del contorno caia detras de la
    camara, y eso descartaba la mayoria de los frames: en un plano de television la
    valla lejana casi siempre cruza el horizonte. Lo correcto es recortar el
    poligono EN EL MUNDO contra la recta preimagen del horizonte (la de w = 0, que
    en coordenadas de pista es H[2,0]*x + H[2,1]*y + H[2,2] = 0) y proyectar lo que
    queda. Asi el frame sigue siendo utilizable.
    """
    poly = rink_outline(p)
    # 1) recorte en el mundo: solo lo que queda delante de la camara
    a, b, c = H[2, 0], H[2, 1], H[2, 2]
    if a * poly[:, 0].mean() + b * poly[:, 1].mean() + c < 0:
        a, b, c = -a, -b, -c
    escala = max(abs(a), abs(b), abs(c), 1e-9)
    poly = clip_halfplane(poly, a, b, c, eps=1e-3 * escala)
    if len(poly) < 3:
        return None
    xy = cv2.perspectiveTransform(poly.reshape(-1, 1, 2).astype(np.float32),
                                  H.astype(np.float32)).reshape(-1, 2)
    if not np.isfinite(xy).all():
        return None
    # 2) recorte en imagen contra un rectangulo amplio, para no desbordar int32
    lo_x, hi_x = -margin * w, (1 + margin) * w
    lo_y, hi_y = -margin * h, (1 + margin) * h
    for a_, b_, c_ in ((1, 0, -lo_x), (-1, 0, hi_x), (0, 1, -lo_y), (0, -1, hi_y)):
        xy = clip_halfplane(xy, a_, b_, c_)
        if len(xy) < 3:
            return None
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [xy.astype(np.int32)], 1)
    return m if m.any() else None


# ---------------------------------------------------------------- visual

def panel(title, img):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def gray3(x, lo=None, hi=None):
    lo = np.percentile(x, 1) if lo is None else lo
    hi = np.percentile(x, 99.5) if hi is None else hi
    v = np.clip((x - lo) / max(1e-6, hi - lo), 0, 1)
    return cv2.cvtColor((v * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def tint(mask, color):
    out = np.zeros(mask.shape + (3,), np.uint8)
    out[mask > 0] = color
    return out


def build_figure(img, ice_f, ice_g, reg_f, reg_g, da, db, cres, gt):
    h, w = img.shape[:2]
    over_f = cv2.addWeighted(img, 0.6, tint(reg_f, (0, 180, 255)), 0.4, 0)
    over_g = cv2.addWeighted(img, 0.6, tint(reg_g, (0, 255, 120)), 0.4, 0)
    if gt is not None:
        cnt, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over_f, cnt, -1, (255, 255, 255), 3)
        cv2.drawContours(over_g, cnt, -1, (255, 255, 255), 3)

    croma = np.zeros_like(img)
    croma[:, :, 2] = np.clip(da / 8.0, 0, 1).astype(np.float32).__mul__(255).astype(np.uint8)
    croma[:, :, 0] = np.clip(-db / 12.0, 0, 1).astype(np.float32).__mul__(255).astype(np.uint8)
    croma[reg_g == 0] //= 6

    cres_v = gray3(cres * (reg_g > 0))
    tiles = [
        panel("1) original", img),
        panel("2) HSV fijo S<60 V>180 (blanco = pista real)", over_f),
        panel("3) region combinada (elegida por solidez)", over_g),
        panel("4) croma local: rojo = +a, azul = -b", croma),
        panel("5) cresta multiescala sobre +a", cres_v),
        panel("6) candidatos de linea", cv2.addWeighted(img, 0.5, tint(
            (cres * (reg_g > 0) > np.percentile(cres[reg_g > 0], 99.2)).astype(np.uint8),
            (0, 0, 255)), 0.9, 0)),
    ]
    sc = 640.0 / w
    tiles = [cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) for t in tiles]
    return np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6])])


# ---------------------------------------------------------------- main

def run(args):
    params = rink.RINK_NHL if "nhl" in args.dataset else rink.RINK_IIHF
    tpl = rink.build_template(params)
    polys = MM.rink_polylines(params)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lbls = sorted((ROOT / "datasets" / args.dataset / "labels" / "train").glob("*.txt"))
    res = {"hsv": [], "gmm": [], "robusta": [], "solidez": []}
    snr = {"hsv": [], "gmm": [], "robusta": [], "solidez": []}
    hechos = 0
    for lbl in lbls[:: max(1, len(lbls) // (args.n * 4))]:
        if hechos >= args.n:
            break
        ip = ROOT / "datasets" / args.dataset / "images" / "train" / (lbl.stem + ".jpg")
        if not ip.exists():
            continue
        rec = read_label(lbl)
        if rec is None:
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        fit, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
        if fit is None:
            continue
        H = fit[0]
        gt = true_region(H, params, w, h)
        if gt is None:
            continue

        ice_f = ice_hsv(img)
        ice_g, info = gmm_surface(img)
        ice_r = robust_surface(img)
        reg_f, reg_g, reg_r = play_region(ice_f), play_region(ice_g), play_region(ice_r)
        res["hsv"].append(iou(reg_f > 0, gt > 0))
        res["gmm"].append(iou(reg_g > 0, gt > 0))
        res["robusta"].append(iou(reg_r > 0, gt > 0))
        reg_s = reg_g if solidity(reg_g) >= solidity(reg_f) else reg_f
        res["solidez"].append(iou(reg_s > 0, gt > 0))

        da, db = local_chroma(img, win=int(args.median_win * w / 1920) | 1)
        cres = ridge(da)

        # localizacion: se distingue la curva en su sitio de la misma desplazada?
        for nombre, reg in (("hsv", reg_f), ("gmm", reg_g), ("robusta", reg_r), ("solidez", reg_s)):
            snr[nombre] += localization_z(da * (reg > 0), H, polys, w, h)

        cv2.imwrite(str(out_dir / (args.dataset + "_" + lbl.stem + ".jpg")),
                    build_figure(img, ice_f, ice_g, reg_f, reg_s, da, db, cres, gt))
        hechos += 1

    print("\n=== {} ({} frames) ===".format(args.dataset, hechos))
    print("  IoU de la region de pista contra el contorno real proyectado:")
    for k in ("hsv", "gmm", "robusta", "solidez"):
        if res[k]:
            v = np.array(res[k])
            print("    {:8s} p50 {:.3f}   p10 {:.3f}   peor {:.3f}".format(
                k, np.median(v), np.percentile(v, 10), v.min()))
    print("  localizacion de curva (z de la posicion correcta vs desplazada +-12..30 px):")
    for k in ("hsv", "gmm", "robusta", "solidez"):
        if snr[k]:
            v = np.array(snr[k])
            print("    {:8s} p50 {:5.1f}   >2z {:3.0f} %   >3z {:3.0f} %   (n={})".format(
                k, np.median(v), 100 * np.mean(v > 2), 100 * np.mean(v > 3), len(v)))
    print("\n  figuras en {}".format(out_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hockeyrink")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--median-win", type=float, default=61, help="ventana del color local, en px a 1920")
    ap.add_argument("--out", default=str(ROOT / "scratch_frames" / "ice_probe"))
    run(ap.parse_args())
