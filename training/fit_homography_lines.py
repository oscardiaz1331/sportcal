"""Ajuste conjunto de la homografia maximizando la respuesta sobre TODAS las curvas.

Una curva suelta se localiza sola solo el ~55% de las veces (medido en
training/ice_lines_probe.py: z de la posicion correcta frente a la misma curva
desplazada). Pero un frame tiene 6-10 curvas y comparten una unica homografia, asi
que ajustarlas a la vez convierte varias señales mediocres en una buena.

Objetivo que se maximiza:

    J(H) = media sobre curvas de la respuesta cromatica integrada a lo largo de
           la curva proyectada por H, mas un termino de solape entre el contorno
           de la pista proyectado y la region detectada

La respuesta es la desviacion de color respecto al hielo LOCAL (+a para las lineas
rojas, -b para las azules), no el tono absoluto: una linea pintada bajo el hielo
llega a camara como un rosa casi neutro y su H es ruido, pero su desviacion
respecto al hielo de al lado es estable.

Parametrizacion: en vez de los 9 numeros de H se optimizan las posiciones EN
IMAGEN de 4 puntos de anclaje del mundo (las esquinas de la pista). Son 8 numeros
en pixeles, todos del mismo orden de magnitud, que es lo que necesita un
optimizador sin gradientes para portarse bien.

Lo que mide este script es el RADIO DE CAPTURA: se parte de la homografia real,
se perturba con ruido creciente y se mira desde que error inicial el ajuste vuelve
a converger. Ese numero decide si la via sirve como auto-etiquetador: con
inicializacion del frame anterior de un video, basta con que el radio cubra el
movimiento de camara entre frames.

    python training/fit_homography_lines.py
    python training/fit_homography_lines.py --n 30 --dataset hockeyrink_nhl
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
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import ice_lines_probe as P  # noqa: E402
import make_line_masks as MM  # noqa: E402
import rink  # noqa: E402
from relabel_reproject import read_label  # noqa: E402

ROJAS = (2, 3, 6, 7, 8, 9, 10, 11)
AZULES = (4, 5)


def anchors_world(p):
    """4 puntos de anclaje bien repartidos: los centros de los arcos de esquina."""
    L, W, r = p["length"], p["width"], p["corner_r"]
    return np.array([[r, r], [L - r, r], [L - r, W - r], [r, W - r]], np.float32)


def H_from_anchors(world4, img4):
    try:
        return cv2.getPerspectiveTransform(world4, img4.astype(np.float32))
    except cv2.error:
        return None


def anchors_from_H(H, world4):
    return cv2.perspectiveTransform(world4.reshape(-1, 1, 2), H).reshape(-1, 2)


def responses(img, region):
    """(+a, -b, +b) normalizadas.

    Las dos primeras se recortan a la region de pista, pero la tercera NO: el
    zocalo de las vallas cae por definicion FUERA del hielo, asi que recortarlo
    borraria justo la senal mas util del frame.
    """
    da, db = P.local_chroma(img, win=int(61 * img.shape[1] / 1920) | 1)
    m = region > 0
    rojo = da * m
    azul = -db * m
    for r in (rojo, azul):
        sd = r[m].std() if m.any() else 1.0
        r /= (sd or 1.0)
    zocalo = db / (db.std() or 1.0)
    return rojo, azul, zocalo


# El zocalo de las vallas: la senal mas fuerte del frame con diferencia.
#
# Las vallas llevan en su base una franja (amarilla en casi todas las pistas) que
# recorre TODO el contorno de la pista. Medida como desviacion de b respecto al
# hielo local da +59 en hockeyrink y +34 en hockeyrink_nhl; para comparar, la mejor
# linea pintada daba dB = -11 (azul) y dA = +13 (central).
#
# Deslizando el contorno proyectado por su normal, el pico sale a z = 79.9 de
# mediana en hockeyrink (97 % de frames por encima de 5z) y 48.5 en hockeyrink_nhl.
# Las curvas de linea individuales daban z = 5-10. Es entre 8 y 16 veces mas
# discriminativa, esta en practicamente todos los frames -- frente a una mediana de
# solo 3 curvas de linea en cuadro -- y al ser cerrada restringe la homografia por
# todo el perimetro, que es justo donde no llegan los keypoints.
#
# El pico no cae en el contorno sino unos 8-10 px FUERA: el zocalo es una
# superficie vertical que arranca en la linea del hielo, asi que en imagen aparece
# desplazado hacia el publico.
#
# Hay que usar un desplazamiento FIJO, no el maximo sobre un abanico. Con el maximo
# el objetivo se vuelve plano: si el zocalo esta a +8 y se muestrea +4/+8/+12,
# desplazar la homografia 4 px hacia fuera tambien acierta. Medido, el abanico mete
# un sesgo de -4 px y hunde la agudeza:
#
#     formulacion            sesgo p50   |sesgo|<=2px   z p50
#     max(+4,+8,+12)          -4.0 px        26 %        4.3
#     fijo +8                 +0.0 px        60 %       20.2
#     dif +8 menos -8         -2.0 px        57 %        3.6
#
# El offset depende de la altura de camara (mediana +8 px en hockeyrink, +10 en
# hockeyrink_nhl), asi que lo suyo acabaria siendo optimizarlo como un parametro
# mas compartido por todo el contorno; +8 fijo cubre bien los dos datasets.
OFFSET_ZOCALO = 8.0


def outline_normals(H, params_rink, w, h):
    """Contorno de la pista proyectado, con la normal apuntando hacia FUERA."""
    xy, ok = MM.project_points(H, P.rink_outline(params_rink), w, h)
    if ok.sum() < 80:
        return None, None
    base = xy[ok]
    nz = P.normals(base)
    c = base.mean(0)
    signo = np.sign(np.einsum("ij,ij->i", base - c, nz))[:, None]
    return base, nz * np.where(signo == 0, 1.0, signo)


def kickplate_score(db_norm, H, params_rink, w, h):
    """Respuesta del zocalo: media de dB sobre el contorno, al mejor desplazamiento."""
    base, nz = outline_normals(H, params_rink, w, h)
    if base is None:
        return None
    q = base + nz * (OFFSET_ZOCALO * w / 1920.0)
    inb = (q[:, 0] > 1) & (q[:, 0] < w - 2) & (q[:, 1] > 1) & (q[:, 1] < h - 2)
    if inb.sum() < 60:
        return None
    qq = q[inb].astype(int)
    return float(db_norm[qq[:, 1], qq[:, 0]].mean())


def curve_score(R, H, world, w, h, min_pts=40):
    xy, ok = MM.project_points(H, world, w, h)
    if ok.sum() < min_pts:
        return None
    q = xy[ok]
    inb = (q[:, 0] > 1) & (q[:, 0] < w - 2) & (q[:, 1] > 1) & (q[:, 1] < h - 2)
    if inb.sum() < min_pts:
        return None
    q = q[inb]
    return float(R[q[:, 1].astype(int), q[:, 0].astype(int)].mean())


def objective(params, ctx):
    """-J. Powell minimiza, asi que se devuelve el negativo."""
    img4 = params.reshape(4, 2)
    H = H_from_anchors(ctx["world4"], img4)
    if H is None or not np.isfinite(H).all():
        return 10.0
    w, h = ctx["w"], ctx["h"]

    vals = []
    for cls, world in ctx["polys"]:
        R = ctx["rojo"] if cls in ROJAS else (ctx["azul"] if cls in AZULES else None)
        if R is None:
            continue
        v = curve_score(R, H, world, w, h)
        if v is not None:
            vals.append(v)
    j = float(np.mean(vals)) if vals else 0.0

    # el zocalo domina a proposito: mide z ~50-80 frente a z ~5-10 de una linea
    if ctx["w_zocalo"] > 0:
        z = kickplate_score(ctx["zocalo"], H, ctx["params_rink"], w, h)
        if z is None:
            return 10.0
        j += ctx["w_zocalo"] * z
    elif not vals:
        return 10.0

    # termino de region: el contorno proyectado debe solapar con la pista detectada
    if ctx["w_region"] > 0:
        pred = P.true_region(H, ctx["params_rink"], w, h)
        if pred is None:
            return 10.0
        j += ctx["w_region"] * P.iou(pred > 0, ctx["region"] > 0)
    return -j


def fit(img, region, H0, params_rink, polys, w_region=2.0, w_zocalo=4.0, maxiter=2000,
        blurs=(14.0, 7.0, 3.0, 0.0), region_off_from=2):
    """Refina H0 maximizando la respuesta conjunta, de grueso a fino.

    Sin el desenfoque progresivo esto no converge: medido sobre 6 frames, el
    objetivo tiene su maximo en la homografia correcta de media (J cae de forma
    monotona al perturbar), pero en 5 de 6 frames alguna perturbacion aleatoria lo
    SUPERA. Son maximos espurios estrechos -- una linea que cae encima de otra, o
    sobre una camiseta roja -- y un optimizador local se queda en el primero que
    pilla, incluso partiendo de una inicializacion buena.

    Desenfocar la respuesta ensancha el pico verdadero y aplasta los espurios,
    porque el verdadero esta respaldado por varias curvas a la vez y los otros no.
    Se optimiza en cada escala partiendo del resultado de la anterior, igual que
    una piramide de Lucas-Kanade.

    El termino de region se APAGA en las etapas finas (region_off_from): sirve para
    encuadrar en grueso, pero su optimo esta sesgado -- la region detectada solo
    coincide en un IoU de ~0.91 con la real -- asi que si sigue pesando al afinar
    tira de la solucion fuera del optimo de las lineas.
    """
    world4 = anchors_world(params_rink)
    rojo, azul, zocalo = responses(img, region)
    w, h = img.shape[1], img.shape[0]
    x = anchors_from_H(H0, world4).ravel()
    for etapa, b in enumerate(blurs):
        if b > 0:
            sig = b * w / 1920.0
            r = cv2.GaussianBlur(rojo, (0, 0), sig)
            a = cv2.GaussianBlur(azul, (0, 0), sig)
        else:
            r, a = rojo, azul
        z = cv2.GaussianBlur(zocalo, (0, 0), b * w / 1920.0) if b > 0 else zocalo
        ctx = {"world4": world4, "polys": polys, "rojo": r, "azul": a, "zocalo": z,
               "region": region, "params_rink": params_rink,
               "w": w, "h": h, "w_zocalo": w_zocalo,
               "w_region": w_region if etapa < region_off_from else 0.0}
        res = minimize(objective, x, args=(ctx,), method="Powell",
                       options={"maxiter": maxiter, "xtol": max(0.3, b / 4), "ftol": 1e-3})
        if np.isfinite(res.x).all():
            x = res.x
    H = H_from_anchors(world4, x.reshape(4, 2))
    return H if H is not None and np.isfinite(H).all() else H0


def geom_error(Ha, Hb, tpl, w, h):
    """Error mediano en px entre dos homografias, sobre los puntos del template en cuadro."""
    xa, oka = MM.project_points(Ha, tpl, w, h)
    xb, okb = MM.project_points(Hb, tpl, w, h)
    ok = oka & okb
    inb = ok & (xb[:, 0] > 0) & (xb[:, 0] < w) & (xb[:, 1] > 0) & (xb[:, 1] < h)
    if inb.sum() < 4:
        return np.inf
    return float(np.median(np.linalg.norm(xa[inb] - xb[inb], axis=1)) * 1920.0 / w)


def perturb(H, world4, sigma_px, w, rng):
    """Perturba H moviendo sus anclas en imagen (sigma en px a 1920)."""
    a = anchors_from_H(H, world4)
    a = a + rng.normal(0, sigma_px * w / 1920.0, a.shape)
    return H_from_anchors(world4, a)


def run(args):
    params_rink = rink.RINK_NHL if "nhl" in args.dataset else rink.RINK_IIHF
    tpl = rink.build_template(params_rink)
    polys = MM.rink_polylines(params_rink)
    world4 = anchors_world(params_rink)
    rng = np.random.default_rng(0)

    lbls = sorted((ROOT / "datasets" / args.dataset / "labels" / "train").glob("*.txt"))
    paso = max(1, len(lbls) // (args.n * 2))
    sigmas = [float(s) for s in args.sigmas.split(",")]
    antes = {s: [] for s in sigmas}
    despues = {s: [] for s in sigmas}
    hechos = 0

    for lbl in lbls[::paso]:
        if hechos >= args.n:
            break
        ip = ROOT / "datasets" / args.dataset / "images" / "train" / (lbl.stem + ".jpg")
        if not ip.exists():
            continue
        rec = read_label(lbl)
        img = cv2.imread(str(ip))
        if rec is None or img is None:
            continue
        h, w = img.shape[:2]
        fitq, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
        if fitq is None:
            continue
        Hgt = fitq[0]
        region = P.best_region(img)
        if region.mean() < 0.15:
            continue
        hechos += 1
        for s in sigmas:
            H0 = perturb(Hgt, world4, s, w, rng)
            if H0 is None:
                continue
            e0 = geom_error(H0, Hgt, tpl, w, h)
            H1 = fit(img, region, H0, params_rink, polys, w_region=args.w_region,
                     w_zocalo=args.w_zocalo)
            e1 = geom_error(H1, Hgt, tpl, w, h)
            antes[s].append(e0)
            despues[s].append(e1)

    print("\n=== {}  ({} frames) ===".format(args.dataset, hechos))
    print("  perturbacion    error inicial      error tras ajustar     recuperados")
    print("  (sigma px)       p50      p50        p50      p90       <10px    <25px")
    for s in sigmas:
        if not antes[s]:
            continue
        a = np.array(antes[s])
        d = np.array(despues[s])
        fin = np.isfinite(d)
        print("  {:6.0f}        {:7.1f}            {:7.1f}  {:7.1f}     {:5.0f} %  {:5.0f} %".format(
            s, np.median(a), np.median(d[fin]), np.percentile(d[fin], 90),
            100 * np.mean(d[fin] < 10), 100 * np.mean(d[fin] < 25)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hockeyrink")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sigmas", default="5,10,20,40")
    ap.add_argument("--w-region", type=float, default=2.0)
    ap.add_argument("--w-zocalo", type=float, default=4.0)
    run(ap.parse_args())
