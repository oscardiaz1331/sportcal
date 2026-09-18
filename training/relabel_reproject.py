"""Densifica las etiquetas de keypoints reproyectando el template por la homografia.

El anotador solo marco los puntos que reconocio: mediana de 13 de 56 por imagen, y
algunos indices aparecen en 3-16 imagenes de 955 enteras. Con tan pocas muestras esos
keypoints no se aprenden nunca, y el 76% de la salida del modelo no recibe gradiente.

Pero los 56 puntos no son independientes: estan todos en el plano del hielo, asi que
cada vista es una homografia del mismo template. Con los puntos que SI estan marcados
se ajusta esa H y se reproyectan los 56; los que caen dentro del frame pasan a ser
etiqueta. No se anota nada nuevo a mano y la supervision por imagen se multiplica.

Que sale de aqui:

    datasets/hockeyrink_rp/       imagenes en hardlink + etiquetas densificadas
    datasets/hockeyrink_nhl_rp/
    training/hockeyrink_pose_rp.yaml

Los originales no se tocan. Convenciones de visibilidad en la salida:

    v=2  punto marcado a mano que el RANSAC acepta (se respeta su posicion)
    v=1  punto reproyectado: o bien no estaba marcado, o estaba marcado pero
         el RANSAC lo descarto y se corrige a su sitio geometrico

Ultralytics trata v=1 y v=2 igual en la OKS loss (la mascara es v != 0), asi que la
distincion es solo para poder auditar despues cuanta etiqueta es humana.

    python training/relabel_reproject.py
    python training/relabel_reproject.py --min-fit 8 --max-resid 6 --dry-run
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rink  # noqa: E402

NKPT = 56
REF_WIDTH = 1920.0

# (dataset origen, dataset destino, parametros de pista)
JOBS = [
    ("hockeyrink", "hockeyrink_rp", rink.RINK_IIHF),
    ("hockeyrink_nhl", "hockeyrink_nhl_rp", rink.RINK_NHL),
]


def read_label(path):
    """Devuelve (cls, box_xywh, kpts (56,3)) normalizados, o None."""
    txt = path.read_text().split()
    if len(txt) < 5 + NKPT * 3:
        return None
    return int(float(txt[0])), np.asarray(txt[1:5], float), \
        np.asarray(txt[5:5 + NKPT * 3], float).reshape(NKPT, 3)


def write_label(path, cls, box, kpts):
    parts = [str(cls)] + ["{:.6f}".format(v) for v in box]
    for x, y, v in kpts:
        parts += ["{:.6f}".format(x), "{:.6f}".format(y), "{:g}".format(v)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(parts) + "\n")


def link(src, dst):
    """Hardlink como hace prepare_data.py, con copia de respaldo si falla."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def in_front(H, world):
    """Mascara de puntos que caen delante de la camara.

    Una homografia proyecta tambien los puntos que quedan "detras": su tercera
    coordenada cambia de signo y acaban en cualquier parte del plano imagen. Sin
    este filtro se etiquetan puntos fantasma al otro lado de la linea del horizonte.
    """
    w = world @ H[2, :2] + H[2, 2]
    if not len(w):
        return np.zeros(0, bool)
    sign = np.sign(np.median(w[np.abs(w) > 1e-9])) or 1.0
    return w * sign > 1e-9


def densify(kpts, tpl, w, h, min_fit, max_resid, ransac_px, margin):
    """Devuelve (kpts_nuevos, stats) o (None, motivo) si la imagen no se puede ajustar."""
    vis = kpts[:, 2] > 0
    if vis.sum() < min_fit:
        return None, "pocos puntos marcados"

    px = kpts[:, :2] * np.array([w, h])
    scale = REF_WIDTH / w
    H, mask = cv2.findHomography(tpl[vis].astype(np.float32), px[vis].astype(np.float32),
                                 cv2.RANSAC, ransac_px / scale)
    if H is None:
        return None, "sin homografia"
    inl = mask.ravel().astype(bool)
    if inl.sum() < 4:
        return None, "menos de 4 inliers"

    # residuo sobre los inliers: si ni ellos cuadran, la H no es de fiar
    proj_in = cv2.perspectiveTransform(tpl[vis][inl].reshape(-1, 1, 2).astype(np.float32),
                                       H.astype(np.float32)).reshape(-1, 2)
    resid = np.median(np.linalg.norm(proj_in - px[vis][inl], axis=1) * scale)
    if resid > max_resid:
        return None, "residuo {:.1f} px".format(resid)

    proj = cv2.perspectiveTransform(tpl.reshape(-1, 1, 2).astype(np.float32),
                                    H.astype(np.float32)).reshape(-1, 2)
    keep = in_front(H, tpl)
    norm = proj / np.array([w, h])
    keep &= (norm[:, 0] >= -margin) & (norm[:, 0] <= 1 + margin)
    keep &= (norm[:, 1] >= -margin) & (norm[:, 1] <= 1 + margin)

    out = np.zeros_like(kpts)
    # 1) los marcados a mano que el RANSAC acepta se quedan tal cual
    idx_vis = np.where(vis)[0]
    good = idx_vis[inl]
    out[good, :2] = kpts[good, :2]
    out[good, 2] = 2
    # 2) todo lo demas que caiga en cuadro, reproyectado
    added = corrected = 0
    for i in np.where(keep)[0]:
        if out[i, 2] > 0:
            continue
        out[i, :2] = np.clip(norm[i], 0.0, 1.0)
        out[i, 2] = 1
        if vis[i]:
            corrected += 1      # estaba marcado pero era outlier: se corrige
        else:
            added += 1
    return out, {"antes": int(vis.sum()), "despues": int((out[:, 2] > 0).sum()),
                 "anadidos": added, "corregidos": corrected, "resid": float(resid),
                 "inliers": float(inl.mean())}


def box_union(box, kpts):
    """Caja original ampliada para cubrir los keypoints nuevos, recortada al frame."""
    cx, cy, bw, bh = box
    x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    k = kpts[kpts[:, 2] > 0, :2]
    if len(k):
        x0, y0 = min(x0, k[:, 0].min()), min(y0, k[:, 1].min())
        x1, y1 = max(x1, k[:, 0].max()), max(y1, k[:, 1].max())
    x0, y0 = max(x0, 0.0), max(y0, 0.0)
    x1, y1 = min(x1, 1.0), min(y1, 1.0)
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])


def run(args):
    total = {}
    for src_name, dst_name, params in JOBS:
        tpl = rink.build_template(params)
        src = ROOT / "datasets" / src_name
        dst = ROOT / "datasets" / dst_name
        stats = {"ok": 0, "antes": 0, "despues": 0, "anadidos": 0, "corregidos": 0}
        motivos = {}
        for split in ("train", "val"):
            for img in sorted((src / "images" / split).glob("*")):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl = src / "labels" / split / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                rec = read_label(lbl)
                if rec is None:
                    continue
                cls, box, kpts = rec
                w, h = Image.open(img).size
                new, info = densify(kpts, tpl, w, h, args.min_fit, args.max_resid,
                                    args.ransac_px, args.margin)
                if new is None:
                    motivos[info] = motivos.get(info, 0) + 1
                    new, out_box = kpts, box           # se copia la etiqueta original
                else:
                    stats["ok"] += 1
                    for k in ("antes", "despues", "anadidos", "corregidos"):
                        stats[k] += info[k]
                    out_box = box_union(box, new)
                if not args.dry_run:
                    link(img, dst / "images" / split / img.name)
                    write_label(dst / "labels" / split / (img.stem + ".txt"), cls, out_box, new)
        total[src_name] = (stats, motivos)

    print("\n=== re-etiquetado por reproyeccion ===")
    for name, (s, motivos) in total.items():
        if not s["ok"]:
            print("  {}: ninguna imagen ajustable".format(name))
            continue
        print("  {}: {} imgs densificadas".format(name, s["ok"]))
        print("     keypoints/img: {:.1f} -> {:.1f}   (+{} anadidos, {} outliers corregidos)".format(
            s["antes"] / s["ok"], s["despues"] / s["ok"], s["anadidos"], s["corregidos"]))
        if motivos:
            print("     sin densificar: " + ", ".join(
                "{} ({})".format(k, v) for k, v in sorted(motivos.items(), key=lambda x: -x[1])))
    if args.dry_run:
        print("\n  --dry-run: no se ha escrito nada")


def write_yaml():
    src = (ROOT / "training" / "hockeyrink_pose.yaml").read_text(encoding="utf-8")
    out = (src
           .replace("datasets/hockeyrink", "datasets/hockeyrink_rp")
           .replace("../hockeyrink_nhl/", "../hockeyrink_nhl_rp/"))
    header = ("# GENERADO por training/relabel_reproject.py -- no editar a mano.\n"
              "# Mismas imagenes que hockeyrink_pose.yaml, con los keypoints densificados\n"
              "# reproyectando el template del rink por la homografia de cada frame.\n")
    path = ROOT / "training" / "hockeyrink_pose_rp.yaml"
    path.write_text(header + out, encoding="utf-8")
    print("\n  escrito {}".format(path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-fit", type=int, default=6,
                    help="minimo de puntos marcados para intentar ajustar H")
    ap.add_argument("--max-resid", type=float, default=8.0,
                    help="residuo mediano maximo sobre inliers, en px a 1920")
    ap.add_argument("--ransac-px", type=float, default=8.0)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="margen fuera del frame que se admite, en fraccion del lado")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args)
    if not args.dry_run:
        write_yaml()
