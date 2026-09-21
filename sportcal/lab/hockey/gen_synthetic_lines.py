"""Generador de datos sinteticos para la segmentacion de lineas, por domain randomization.

Por que: diagnose_lines_seg.py demostro que la red de segmentacion (train_lines_seg.py)
localiza muchisimo mejor que la via clasica DONDE hay datos homogeneos suficientes
(IIHF, z de cientos/miles), pero se cae en NHL (z=3-9 en varias clases, practicamente
ruido) porque solo hay 344 imagenes de train repartidas entre muchas retransmisiones
y pistas distintas -- poca redundancia de apariencia por escena. La solucion, siguiendo
a Rink-Agnostic (Waterloo, arXiv:2401.01003), es generar muchas mas apariencias sin
depender de mas etiquetado manual.

Geometria vs apariencia, por separado a proposito:

  - GEOMETRIA: se reutiliza el pool de ~955 homografias YA AJUSTADAS sobre los
    keypoints reales de train (hockeyrink + hockeyrink_nhl). Construir un modelo de
    camara 3D desde cero para inventar vistas nuevas es mas trabajo y mas riesgo de
    generar encuadres irreales; el pool real ya cubre la diversidad de angulos de las
    retransmisiones reales, gratis y sin inventar nada.
  - APARIENCIA: aqui es donde esta el hueco (colores de hielo/lineas/vallas, publicidad,
    graderio) que SI se randomiza fuerte, porque es lo que le falta a NHL.

Render: un "lienzo" cenital en metros (hielo + anillo de zocalo + banda de publicidad +
graderio de relleno), con las lineas dibujadas por clase reutilizando EXACTAMENTE
make_line_masks.rink_polylines (mismo esquema de 12 clases, cero riesgo de que la
imagen y la mascara se desincronicen). Un unico cv2.warpPerspective con
H_total = H_real @ A (A = pixel-de-lienzo -> metros) proyecta el lienzo entero a la
imagen; otro warpPerspective con INTER_NEAREST hace lo mismo con el lienzo de clases
para sacar la mascara. El fondo de la imagen (mas alla de lo que cubre el lienzo) se
rellena por separado con textura procedural, porque el graderio/publicidad tiene
altura real (no esta en el plano del hielo) y proyectarlo como si estuviera en el
suelo da una geometria incorrecta cerca del horizonte -- para el objetivo (que la red
aprenda contraste linea/fondo, no reconstruir el estadio) basta con que tenga textura
y color plausibles, no con que sea fisicamente exacto.

Grosor de linea en METROS (0.15 m), no en px de referencia como en make_line_masks.py:
al proyectar por la H real, la escala en pixeles sale sola segun lo lejos que este esa
parte de la pista en cada vista, que es lo que pasa de verdad con una linea pintada.

    python -m sportcal.lab.hockey.gen_synthetic_lines --n 1500
    python -m sportcal.lab.hockey.gen_synthetic_lines --n 20 --overlay   # revisar antes de comprometerse
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

TPX = 15.0            # texeles del lienzo por metro
MARGIN = 8.0           # metros de lienzo mas alla de la valla (zocalo + publicidad)
KICKPLATE_W = 0.40     # ancho del zocalo, metros
ADS_W = 2.2            # banda de publicidad tras el zocalo, metros
LINE_W = 0.15          # grosor de lineas/circulos pintados, metros
OUT_IMG_DS = "hockeyrink_synth"
OUT_MASK_DS = "hockeyrink_synth_lines"

ANCHOR_JOBS = [("hockeyrink", rink.RINK_IIHF), ("hockeyrink_nhl", rink.RINK_NHL)]


def boundary_loop(p, offset=0.0):
    """Contorno cerrado y CONTINUO de la valla (rectangulo redondeado), en metros.

    make_line_masks.rink_polylines trae la valla en 8 trozos sueltos (4 rectas + 4
    arcos) sin orden de recorrido -- sirve para dibujar pero no para fillPoly. Aqui
    se reconstruye en orden conectado, con `offset` metros hacia fuera (para el
    zocalo/publicidad, engordando el radio de esquina otro tanto).
    """
    L, W = p["length"], p["width"]
    r = p["corner_r"] + offset
    x0, x1 = -offset, L + offset
    y0, y1 = -offset, W + offset

    def arc(cx_, cy_, rad, a0, a1, n=30):
        t = np.radians(np.linspace(a0, a1, n))
        return np.stack([cx_ + rad * np.cos(t), cy_ + rad * np.sin(t)], 1)

    # top: de (corner_r, y0) a (L-corner_r, y0)
    top = np.array([[p["corner_r"], y0], [L - p["corner_r"], y0]])
    tr = arc(L - p["corner_r"], p["corner_r"], r, 270, 360)
    right = np.array([[x1, p["corner_r"]], [x1, W - p["corner_r"]]])
    br = arc(L - p["corner_r"], W - p["corner_r"], r, 0, 90)
    bottom = np.array([[L - p["corner_r"], y1], [p["corner_r"], y1]])
    bl = arc(p["corner_r"], W - p["corner_r"], r, 90, 180)
    left = np.array([[x0, W - p["corner_r"]], [x0, p["corner_r"]]])
    tl = arc(p["corner_r"], p["corner_r"], r, 180, 270)
    return np.concatenate([top, tr, right, br, bottom, bl, left, tl], axis=0)


def rng_color(rng, base, jitter=25, vmin=0, vmax=255):
    c = np.array(base, float) + rng.uniform(-jitter, jitter, 3)
    return tuple(int(v) for v in np.clip(c, vmin, vmax))


def sample_palette(rng):
    """Un juego de colores por imagen -- esto es lo que aporta diversidad de apariencia."""
    return dict(
        ice=rng_color(rng, (200, 205, 208), jitter=30),          # hielo: blanco azulado variable
        ice_noise=rng.uniform(2, 10),
        rojo=rng_color(rng, (40, 30, 195), jitter=35),            # BGR: rojo pintura
        azul=rng_color(rng, (170, 70, 20), jitter=35),            # BGR: azul pintura
        kickplate=rng_color(rng, (30, 200, 225), jitter=45) if rng.random() < 0.7
                  else rng_color(rng, (200, 200, 200), jitter=40),  # normalmente amarillo, a veces claro
        ads=[rng_color(rng, (rng.integers(20, 235),) * 3, jitter=60) for _ in range(rng.integers(3, 7))],
        stands=rng_color(rng, (60, 55, 50), jitter=25),
        logo=rng_color(rng, (rng.integers(30, 220),) * 3, jitter=50),
    )


def front_of_camera_mask(H, p, x0, y0, cw, ch):
    """True donde el texel esta delante de la camara (no al otro lado del horizonte).

    warpPerspective invierte la homografia globalmente sin mirar el signo de la
    coordenada homogenea: un texel "detras" de la camara (mismo problema que
    make_line_masks.project_points resuelve punto a punto) se proyecta igualmente
    y aparece como un duplicado fantasma al otro lado del horizonte en la imagen.
    Aqui se enmascara ANTES de deformar, con el mismo criterio de signo.
    """
    xs = x0 + (np.arange(cw) + 0.5) / TPX
    ys = y0 + (np.arange(ch) + 0.5) / TPX
    WX, WY = np.meshgrid(xs, ys)
    s = H[2, 0] * WX + H[2, 1] * WY + H[2, 2]
    s_ref = np.sign(H[2, 0] * p["length"] / 2 + H[2, 1] * p["width"] / 2 + H[2, 2]) or 1.0
    return (s * s_ref) > 1e-9


def build_canvas(p, palette, rng, H):
    """Lienzo cenital (color BGR uint8, clases uint8), origen y escala del mundo."""
    L, W = p["length"], p["width"]
    x0, y0 = -MARGIN, -MARGIN
    cw = int(round((L + 2 * MARGIN) * TPX))
    ch = int(round((W + 2 * MARGIN) * TPX))

    def to_tex(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        return np.stack([(pts[:, 0] - x0) * TPX, (pts[:, 1] - y0) * TPX], 1)

    color = np.zeros((ch, cw, 3), np.uint8)
    color[:] = palette["stands"]
    # ruido de graderio: bandas horizontales de contraste + grano
    bandas = rng.integers(6, 14)
    for _ in range(bandas):
        yb = rng.integers(0, ch)
        hb = rng.integers(3, 14)
        c = rng_color(rng, palette["stands"], jitter=40)
        color[max(0, yb - hb):yb + hb] = c
    noise = rng.normal(0, 14, color.shape).astype(np.int16)
    color = np.clip(color.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # publicidad: rectangulos de color en la banda justo fuera del zocalo
    ads_mask = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(ads_mask, [to_tex(boundary_loop(p, KICKPLATE_W + ADS_W)).astype(np.int32)], 255)
    inner_ads = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(inner_ads, [to_tex(boundary_loop(p, KICKPLATE_W)).astype(np.int32)], 255)
    ads_ring = (ads_mask > 0) & (inner_ads == 0)
    n_ads = int(rng.integers(10, 22))
    for _ in range(n_ads):
        ys, xs = np.where(ads_ring)
        if len(xs) == 0:
            break
        i = rng.integers(len(xs))
        cx, cy = int(xs[i]), int(ys[i])
        w_, h_ = int(rng.integers(20, 90)), int(rng.integers(15, 50))
        c = palette["ads"][rng.integers(len(palette["ads"]))]
        cv2.rectangle(color, (cx - w_ // 2, cy - h_ // 2), (cx + w_ // 2, cy + h_ // 2), c, -1)

    # zocalo
    kick_mask = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(kick_mask, [to_tex(boundary_loop(p, KICKPLATE_W)).astype(np.int32)], 255)
    ice_mask = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(ice_mask, [to_tex(boundary_loop(p, 0.0)).astype(np.int32)], 255)
    kick_ring = (kick_mask > 0) & (ice_mask == 0)
    color[kick_ring] = palette["kickplate"]

    # hielo
    color[ice_mask > 0] = palette["ice"]
    ice_noise = rng.normal(0, palette["ice_noise"], color.shape).astype(np.int16)
    m3 = np.repeat(ice_mask[:, :, None] > 0, 3, axis=2)
    color = np.where(m3, np.clip(color.astype(np.int16) + ice_noise, 0, 255).astype(np.uint8), color)

    # logos difusos bajo el hielo (circulo central y puntos de faceoff): solo apariencia,
    # no son una clase -- igual que en video real, la red debe ignorarlos.
    if rng.random() < 0.8:
        for cls, world in MM.rink_polylines(p):
            if cls not in (7, 8, 9, 10, 11):
                continue
            if rng.random() < 0.5:
                continue
            c = to_tex(world.mean(0, keepdims=True))[0].astype(int)
            rad = int(p["circle_r"] * TPX * rng.uniform(0.5, 0.9))
            overlay = color.copy()
            cv2.circle(overlay, tuple(c), rad, palette["logo"], -1, cv2.LINE_AA)
            alpha = rng.uniform(0.15, 0.4)
            color = cv2.addWeighted(overlay, alpha, color, 1 - alpha, 0)

    # lineas, por clase -- MISMA geometria que las mascaras reales (rink_polylines)
    cls_canvas = np.zeros((ch, cw), np.uint8)
    thick = max(1, int(round(LINE_W * TPX)))
    for cls, world in MM.rink_polylines(p):
        pts = to_tex(world).astype(np.int32)
        col = palette["kickplate"] if cls == 1 else (palette["azul"] if cls in (4, 5) else palette["rojo"])
        cv2.polylines(color, [pts], False, col, thick, cv2.LINE_AA)
        cv2.polylines(cls_canvas, [pts], False, int(cls), thick, cv2.LINE_8)

    front = front_of_camera_mask(H, p, x0, y0, cw, ch)
    alpha = np.where(front, 255, 0).astype(np.uint8)
    cls_canvas[~front] = 0
    return color, cls_canvas, alpha, x0, y0


def add_players(img, valid, rng):
    """Manchas de color en la zona de pista visible, igual que jugadores tapando lineas.

    No se toca la mascara -- exactamente como en los datos reales (make_line_masks
    dibuja la geometria pase lo que pase encima en la foto), asi que esto ensena a la
    red a predecir la clase de linea aunque algo la tape visualmente.
    """
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return img
    n = rng.integers(0, 6)
    out = img.copy()
    for _ in range(n):
        i = rng.integers(len(xs))
        cx, cy = int(xs[i]), int(ys[i])
        h_ = int(rng.integers(18, 55))
        w_ = int(rng.integers(10, 28))
        color = tuple(int(v) for v in rng.integers(10, 240, 3))
        cv2.ellipse(out, (cx, cy), (w_, h_), 0, 0, 360, color, -1, cv2.LINE_AA)
        # "casco/cabeza"
        cv2.circle(out, (cx, cy - h_), max(4, w_ // 2), (230, 220, 210), -1, cv2.LINE_AA)
    return out


def photometric_augment(img, rng):
    img = img.astype(np.float32)
    img *= rng.uniform(0.75, 1.3)
    img += rng.uniform(-20, 20)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.6, 1.3)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-8, 8)) % 180
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.6:
        q = int(rng.integers(40, 90))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    noise = rng.normal(0, rng.uniform(2, 9), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def load_anchor_pool():
    pool = []
    for ds_name, params in ANCHOR_JOBS:
        tpl = rink.build_template(params)
        for lbl in sorted((ROOT / "datasets" / ds_name / "labels" / "train").glob("*.txt")):
            pool.append((lbl, tpl, params))
    return pool


def sample_anchor(rng, pool):
    for _ in range(20):
        lbl, tpl, params = pool[rng.integers(len(pool))]
        rec = read_label(lbl)
        if rec is None:
            continue
        ds_dir = lbl.parents[2]
        img_path = ds_dir / "images" / "train" / (lbl.stem + ".jpg")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        fit, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
        if fit is None:
            continue
        H, _, _ = fit
        return H, w, h, params
    return None


def generate_one(rng, pool):
    anc = sample_anchor(rng, pool)
    if anc is None:
        return None
    H, w, h, params = anc
    palette = sample_palette(rng)
    canvas, cls_canvas, alpha_src, x0, y0 = build_canvas(params, palette, rng, H)
    A = np.array([[1.0 / TPX, 0, x0], [0, 1.0 / TPX, y0], [0, 0, 1]])
    H_total = (H @ A).astype(np.float32)

    img = cv2.warpPerspective(canvas, H_total, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(alpha_src, H_total, (w, h),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0) > 0
    mask = cv2.warpPerspective(cls_canvas, H_total, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    bg = np.zeros((h, w, 3), np.uint8)
    bg[:] = palette["stands"]
    bg += rng.normal(0, 12, bg.shape).astype(np.uint8)
    out = np.where(valid[:, :, None], img, bg)
    out = add_players(out, valid, rng)
    out = photometric_augment(out, rng)
    return out, mask


def run(args):
    rng = np.random.default_rng(args.seed)
    pool = load_anchor_pool()
    print("pool de homografias ancla (train real): {}".format(len(pool)))

    if args.overlay:
        out_dir = Path(args.out_overlay)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            r = generate_one(rng, pool)
            if r is None:
                continue
            img, mask = r
            cv2.imwrite(str(out_dir / "synth_{:04d}.jpg".format(i)), img)
            cv2.imwrite(str(out_dir / "synth_{:04d}_overlay.jpg".format(i)), MM.overlay(img, mask))
        print("revisar en {}".format(out_dir))
        return

    n_val = max(1, int(round(args.n * args.val_frac)))
    hecho = {"train": 0, "val": 0}
    for i in range(args.n):
        r = generate_one(rng, pool)
        if r is None:
            continue
        img, mask = r
        split = "val" if i < n_val else "train"
        stem = "synth_{:06d}".format(i)
        ip = ROOT / "datasets" / OUT_IMG_DS / "images" / split / (stem + ".jpg")
        mp = ROOT / "datasets" / OUT_MASK_DS / split / (stem + ".png")
        ip.parent.mkdir(parents=True, exist_ok=True)
        mp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ip), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(mp), mask)
        hecho[split] += 1
        if (i + 1) % 200 == 0:
            print("  {}/{}".format(i + 1, args.n))
    print("generadas: train {}  val {}  en datasets/{}, datasets/{}".format(
        hecho["train"], hecho["val"], OUT_IMG_DS, OUT_MASK_DS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlay", action="store_true", help="solo genera unas pocas y pinta overlay, sin guardar dataset")
    ap.add_argument("--out-overlay", default=str(ROOT / "scratch_frames" / "synth_preview"))
    run(ap.parse_args())
