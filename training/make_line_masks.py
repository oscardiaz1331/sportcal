"""Genera mascaras de segmentacion de las lineas de la pista, sin anotar nada a mano.

Por que lineas y no solo keypoints: de los 56 keypoints solo ~14 caen dentro del
frame (mediana medida sobre las 1093 imagenes etiquetadas), asi que la homografia
se estima con 14 observaciones para 8 grados de libertad. Las lineas en cambio
cruzan el frame casi siempre, aunque sus intersecciones concretas queden fuera, y
una correspondencia de linea aporta 2 ecuaciones al DLT igual que un punto. Es lo
que usan los ganadores del SoccerNet Camera Calibration: heatmaps de keypoint MAS
segmentacion de lineas, resueltos juntos.

Y no cuesta anotacion: rink.draw_rink() ya proyecta las polilineas del template
por una homografia, y la H de cada frame se ajusta con sus propios keypoints
ground truth. Las etiquetas salen gratis de lo que ya hay.

Salida (mascara de indices uint8, un PNG por imagen, mismo tamano que la imagen):

    datasets/hockeyrink_lines/{train,val}/*.png
    datasets/hockeyrink_nhl_lines/{train,val}/*.png

    python training/make_line_masks.py
    python training/make_line_masks.py --overlay 12      # + visualizacion para revisar
    python training/make_line_masks.py --thickness 6 --dry-run
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import rink  # noqa: E402
from relabel_reproject import read_label  # noqa: E402

REF_WIDTH = 1920.0
IGNORE = 255   # pixeles que la loss debe ignorar (H poco fiable ahi)

# Clases. La zona A es la mitad de x bajo y la B la de x alto, igual que en
# build_template. Se distinguen izquierda/derecha a proposito: una linea azul sin
# saber cual de las dos es no sirve como correspondencia para la homografia.
CLASSES = [
    "fondo",            # 0
    "vallas",           # 1  perimetro completo (rectas + arcos de esquina)
    "linea_gol_A",      # 2
    "linea_gol_B",      # 3
    "linea_azul_A",     # 4
    "linea_azul_B",     # 5
    "linea_central",    # 6
    "circulo_central",  # 7
    "faceoff_A_lo",     # 8
    "faceoff_A_hi",     # 9
    "faceoff_B_lo",     # 10
    "faceoff_B_hi",     # 11
]

# BGR solo para la visualizacion
COLORS = {
    1: (200, 200, 200), 2: (60, 60, 255), 3: (60, 60, 255),
    4: (255, 140, 40), 5: (255, 140, 40), 6: (60, 60, 255),
    7: (60, 220, 60), 8: (0, 220, 220), 9: (0, 220, 220),
    10: (220, 0, 220), 11: (220, 0, 220),
}

JOBS = [
    ("hockeyrink", "hockeyrink_lines", rink.RINK_IIHF),
    ("hockeyrink_nhl", "hockeyrink_nhl_lines", rink.RINK_NHL),
]


def seg(a, b, n=80):
    t = np.linspace(0, 1, n)[:, None]
    return np.array(a, float) * (1 - t) + np.array(b, float) * t


def arc(cx, cy, rad, a0, a1, n=80):
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rad * np.cos(t), cy + rad * np.sin(t)], 1)


def rink_polylines(p):
    """Devuelve [(clase, polilinea en metros), ...] con la geometria de rink.draw_rink."""
    L, W = p["length"], p["width"]
    r, cy = p["corner_r"], W / 2
    gl = [p["goal_line_from_end"], L - p["goal_line_from_end"]]
    bl = [p["blue_from_end"], L - p["blue_from_end"]]
    dot = p["dot_from_goal_line"]
    dy = p["dot_from_axis"]

    out = [
        (1, seg((r, 0), (L - r, 0))), (1, seg((r, W), (L - r, W))),
        (1, seg((0, r), (0, W - r))), (1, seg((L, r), (L, W - r))),
        (1, arc(r, r, r, 180, 270)), (1, arc(L - r, r, r, 270, 360)),
        (1, arc(L - r, W - r, r, 0, 90)), (1, arc(r, W - r, r, 90, 180)),
        (2, seg((gl[0], 0), (gl[0], W))), (3, seg((gl[1], 0), (gl[1], W))),
        (4, seg((bl[0], 0), (bl[0], W))), (5, seg((bl[1], 0), (bl[1], W))),
        (6, seg((L / 2, 0), (L / 2, W))),
        (7, arc(L / 2, cy, p["circle_r"], 0, 360)),
        (8, arc(gl[0] + dot, cy - dy, p["circle_r"], 0, 360)),
        (9, arc(gl[0] + dot, cy + dy, p["circle_r"], 0, 360)),
        (10, arc(gl[1] - dot, cy - dy, p["circle_r"], 0, 360)),
        (11, arc(gl[1] - dot, cy + dy, p["circle_r"], 0, 360)),
    ]
    return out


def project_points(H, world, w, h):
    """Proyecta puntos del mundo y marca cuales son dibujables.

    Una homografia proyecta tambien lo que queda detras de la camara: la tercera
    coordenada cambia de signo y el punto reaparece al otro lado del plano imagen.
    Hay que descartarlos, o una polilinea que cruza el horizonte se dibuja como una
    recta falsa de lado a lado del frame.
    """
    P = np.c_[world, np.ones(len(world))] @ H.T
    z = P[:, 2]
    finite = np.isfinite(z) & (np.abs(z) > 1e-9)
    xy = np.full((len(world), 2), np.nan)
    if not finite.any():
        return xy, np.zeros(len(world), bool)
    sign = np.sign(np.median(z[finite])) or 1.0
    ok = finite & (z * sign > 1e-9)
    xy[ok] = P[ok, :2] / z[ok, None]
    # fuera de un margen razonable no aporta y desborda el int32 de polylines
    ok &= np.isfinite(xy).all(1) & (np.abs(xy[:, 0]) < 4 * w) & (np.abs(xy[:, 1]) < 4 * h)
    return xy, ok


def runs_of(flags):
    """Tramos contiguos de indices donde flags es True."""
    runs, cur = [], []
    for i, good in enumerate(flags):
        if good:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) > 1]


def bootstrap_homographies(world_pts, img_pts, n=12, frac=0.8, seed=0):
    """Reajusta H con subconjuntos de los inliers, para medir cuanto se mueve."""
    m = len(world_pts)
    k = max(4, int(round(frac * m)))
    if m < 5:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.choice(m, k, replace=False)
        Hb, _ = cv2.findHomography(world_pts[idx].astype(np.float32),
                                   img_pts[idx].astype(np.float32), 0)
        if Hb is not None and np.isfinite(Hb).all():
            out.append(Hb)
    return out


def point_spread(Hs, world, w, h):
    """Dispersion en px de cada punto entre las homografias del bootstrap.

    Es la incertidumbre real de la etiqueta: donde los keypoints marcados restringen
    bien la H la dispersion es de ~1 px; en las zonas a las que la H solo llega
    extrapolando (las vallas, los extremos de las lineas azules) se dispara. Esos
    pixeles se marcan IGNORE en vez de inventarles una clase.
    """
    if not Hs:
        return np.full(len(world), np.inf)
    acc = []
    for H in Hs:
        xy, ok = project_points(H, world, w, h)
        xy[~ok] = np.nan
        acc.append(xy)
    A = np.stack(acc)                                  # (n_boot, n_pts, 2)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(A, axis=0)
        d = np.linalg.norm(A - med, axis=2)
        spread = np.nanpercentile(d, 75, axis=0)
    return np.where(np.isfinite(spread), spread, np.inf)


def render_mask(H, polys, w, h, thickness, Hs=None, max_spread=None):
    """Mascara de indices (h, w) uint8; IGNORE donde la H no es de fiar."""
    mask = np.zeros((h, w), np.uint8)
    for cls, world in polys:
        xy, ok = project_points(H, world, w, h)
        if not ok.any():
            continue
        if Hs and max_spread:
            dudoso = point_spread(Hs, world, w, h) > (max_spread * w / REF_WIDTH)
        else:
            dudoso = np.zeros(len(world), bool)
        for value, flags in ((int(cls), ok & ~dudoso), (IGNORE, ok & dudoso)):
            for run in runs_of(flags):
                cv2.polylines(mask, [xy[run].astype(np.int32)], False,
                              value, thickness, cv2.LINE_8)
    return mask


def fit_from_label(kpts, tpl, w, h, min_fit, max_resid, ransac_px):
    """Misma puerta de calidad que relabel_reproject.

    Devuelve (H, info) donde info es el residuo si hay H, o el motivo si no. Los
    inliers quedan en H.inliers_ para poder hacer bootstrap sin reajustar."""
    vis = kpts[:, 2] > 0
    if vis.sum() < min_fit:
        return None, "pocos puntos"
    px = kpts[:, :2] * np.array([w, h])
    scale = REF_WIDTH / w
    H, mask = cv2.findHomography(tpl[vis].astype(np.float32), px[vis].astype(np.float32),
                                 cv2.RANSAC, ransac_px / scale)
    if H is None:
        return None, "sin homografia"
    inl = mask.ravel().astype(bool)
    if inl.sum() < 4:
        return None, "menos de 4 inliers"
    proj = cv2.perspectiveTransform(tpl[vis][inl].reshape(-1, 1, 2).astype(np.float32),
                                    H.astype(np.float32)).reshape(-1, 2)
    resid = np.median(np.linalg.norm(proj - px[vis][inl], axis=1) * scale)
    if resid > max_resid:
        return None, "residuo alto"
    return (H, tpl[vis][inl], px[vis][inl]), resid


def overlay(img, mask, alpha=0.55):
    vis = img.copy()
    for cls, color in COLORS.items():
        vis[mask == cls] = color
    vis[mask == IGNORE] = (40, 40, 40)      # gris oscuro = zona ignorada
    return cv2.addWeighted(vis, alpha, img, 1 - alpha, 0)


def run(args):
    over_dir = Path(args.overlay_dir)
    over_left = args.overlay
    if over_left:
        over_dir.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name, params in JOBS:
        tpl = rink.build_template(params)
        polys = rink_polylines(params)
        src = ROOT / "datasets" / src_name
        dst = ROOT / "datasets" / dst_name
        ok = 0
        motivos = {}
        cobertura = []
        ignorado = []
        for split in ("train", "val"):
            for img_path in sorted((src / "images" / split).glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl = src / "labels" / split / (img_path.stem + ".txt")
                if not lbl.exists():
                    continue
                rec = read_label(lbl)
                if rec is None:
                    continue
                _, _, kpts = rec
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                thickness = max(2, int(round(args.thickness * w / REF_WIDTH)))
                fit, info = fit_from_label(kpts, tpl, w, h, args.min_fit,
                                           args.max_resid, args.ransac_px)
                if fit is None:
                    motivos[info] = motivos.get(info, 0) + 1
                    continue
                H, inl_world, inl_img = fit
                Hs = (bootstrap_homographies(inl_world, inl_img, args.bootstrap)
                      if args.bootstrap else [])
                mask = render_mask(H, polys, w, h, thickness, Hs, args.max_spread)
                ok += 1
                linea = np.count_nonzero((mask > 0) & (mask != IGNORE))
                ign = np.count_nonzero(mask == IGNORE)
                cobertura.append(100.0 * linea / mask.size)
                ignorado.append(100.0 * ign / max(1, linea + ign))
                if not args.dry_run:
                    out = dst / split / (img_path.stem + ".png")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), mask)
                if over_left and ok % max(1, args.overlay_every) == 0:
                    cv2.imwrite(str(over_dir / (src_name + "_" + img_path.stem + ".jpg")),
                                overlay(img, mask))
                    over_left -= 1
        print("  {}: {} mascaras".format(dst_name, ok))
        if cobertura:
            print("     pixeles de linea: {:.2f} % del frame (mediana)".format(np.median(cobertura)))
            print("     marcados IGNORE:  {:.1f} % de la linea (mediana)".format(np.median(ignorado)))
        if motivos:
            print("     sin mascara: " + ", ".join(
                "{} ({})".format(k, v) for k, v in sorted(motivos.items(), key=lambda x: -x[1])))

    print("\n  clases: " + ", ".join(
        "{}={}".format(i, n) for i, n in enumerate(CLASSES)))
    if args.dry_run:
        print("  --dry-run: no se ha escrito nada")
    elif args.overlay:
        print("  visualizaciones en {}".format(over_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thickness", type=float, default=5.0,
                    help="grosor de linea en px a 1920 de ancho; se escala por imagen")
    ap.add_argument("--min-fit", type=int, default=6)
    ap.add_argument("--max-resid", type=float, default=8.0)
    ap.add_argument("--ransac-px", type=float, default=8.0)
    ap.add_argument("--bootstrap", type=int, default=12,
                    help="reajustes de H para medir incertidumbre; 0 desactiva el IGNORE")
    ap.add_argument("--max-spread", type=float, default=6.0,
                    help="dispersion maxima en px a 1920; por encima se marca IGNORE")
    ap.add_argument("--overlay", type=int, default=0, help="cuantas visualizaciones escribir")
    ap.add_argument("--overlay-every", type=int, default=40)
    ap.add_argument("--overlay-dir", default=str(ROOT / "scratch_frames" / "line_masks"))
    ap.add_argument("--dry-run", action="store_true")
    run(ap.parse_args())
