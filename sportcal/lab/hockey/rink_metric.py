"""Metrica de homografia: lo que de verdad importa, en vez de pose mAP.

Pose mAP mide si los keypoints caen cerca de su sitio ponderando por OKS, pero no
dice si la homografia que sale de ellos sirve. Un modelo con mAP alto y dos puntos
sistematicamente sesgados produce una H inutil; uno con mAP mediocre pero errores
sin sesgo produce una H buena, porque el RANSAC se come el ruido.

Lo que se mide aqui, por frame de val:

  1. se predicen los 56 keypoints y se filtran por confianza
  2. se ajusta H (mundo -> imagen) con RANSAC contra el template de rink.py
  3. se reproyecta el template por esa H y se compara con los keypoints
     ground truth VISIBLES de la etiqueta

El error va normalizado a un frame de 1920 de ancho para poder comparar entre
resoluciones. Las tres cifras que interesan:

  cobertura   % de frames donde se pudo estimar una H (si esto baja, da igual
              lo preciso que sea el resto: no hay homografia que usar)
  err p50     error mediano de reproyeccion; el numero de "como de buena es"
  usables     % de frames con error mediano < 10 px, el umbral practico para
              que el minimapa no baile

OJO con box_conf: el rink es un unico objeto que ocupa casi todo el frame y la
cabeza de deteccion es insegura (mediana 0.59 pero p25 = 0.03). A conf 0.25 se
pierde el 42% de los frames de val sin que los keypoints tengan la culpa, asi que
por defecto se coge siempre la mejor caja y se filtra solo por confianza de kpt.

    python -m sportcal.lab.hockey.rink_metric runs/hockeyrink/yolo26m-8/weights/best.pt
    python -m sportcal.lab.hockey.rink_metric best.pt --conf 0.3 --split train
"""
import os

import sys
from pathlib import Path

import cv2
import numpy as np

from sportcal.paths import ROOT

from sportcal.core.geometry import fit_homography_ransac as fit_homography  # noqa: F401
from sportcal.sports.hockey import rink

# el dataset sueco es IIHF (60x30 m) y los frames propios son de pistas NHL
# (60.96x25.91 m): el mismo indice de keypoint cae en coordenadas distintas.
TEMPLATES = {
    "hockeyrink_nhl": rink.build_template(rink.RINK_NHL),
    "hockeyrink": rink.build_template(rink.RINK_IIHF),
}

REF_WIDTH = 1920.0   # los errores se reescalan a este ancho
USABLE_PX = 10.0     # umbral de "homografia usable"


def template_for(path):
    """El template depende de la liga; hockeyrink_nhl primero porque contiene a hockeyrink."""
    parts = Path(path).parts
    for key, tpl in TEMPLATES.items():
        # exacto o con sufijo (hockeyrink_nhl_valh, hockeyrink_nhl_valx...): las variantes
        # de evaluacion llevan el nombre de su liga como prefijo
        if any(p == key or p.startswith(key + "_") for p in parts):
            return tpl
    raise ValueError("no se de que liga es " + str(path))


def label_path(img_path):
    """datasets/X/images/val/f.jpg -> datasets/X/labels/val/f.txt (sin tocar separadores)."""
    parts = list(Path(img_path).parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_gt(lbl_path, w, h):
    """Devuelve (xy_px, visible) de la etiqueta YOLO-pose, o None si no hay objeto."""
    if not Path(lbl_path).exists():
        return None
    txt = Path(lbl_path).read_text().split()
    if len(txt) < 5 + 56 * 3:
        return None
    kp = np.asarray(txt[5:5 + 56 * 3], dtype=float).reshape(56, 3)
    xy = kp[:, :2] * np.array([w, h])
    return xy, kp[:, 2] > 0


def evaluate(weights, images_dirs, conf=0.3, min_kpts=6, ransac_px=8.0, imgsz=1024,
             device=0, box_conf=0.001, verbose=True):
    """Corre el modelo sobre las imagenes y devuelve el resumen de homografia."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    paths = []
    for d in images_dirs:
        paths += sorted(p for p in Path(d).glob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SystemExit("sin imagenes en " + str(images_dirs))

    per_frame, n_no_h, n_pocos_kpts, inlier_frac = [], 0, 0, []
    for k in range(0, len(paths), 16):
        chunk = paths[k:k + 16]
        results = model.predict(source=[str(p) for p in chunk], imgsz=imgsz, device=device,
                                verbose=False, conf=box_conf)
        for path, r in zip(chunk, results):
            h, w = r.orig_shape
            gt = load_gt(label_path(path), w, h)
            if gt is None:
                continue
            gt_xy, gt_vis = gt
            tpl = template_for(path)

            if r.keypoints is None or len(r.keypoints.xy) == 0:
                n_no_h += 1
                continue
            pred_xy = r.keypoints.xy[0].cpu().numpy()
            pred_cf = (r.keypoints.conf[0].cpu().numpy()
                       if r.keypoints.conf is not None else np.ones(56))
            sel = pred_cf >= conf
            if sel.sum() < min_kpts:
                n_pocos_kpts += 1
                continue

            scale = REF_WIDTH / w   # el umbral de RANSAC se define a 1920
            H, inl = fit_homography(tpl[sel], pred_xy[sel], ransac_px / scale)
            if H is None:
                n_no_h += 1
                continue
            inlier_frac.append(inl.mean())

            # el error se mide contra el ground truth, no contra la propia prediccion
            if gt_vis.sum() < 4:
                continue
            proj = cv2.perspectiveTransform(
                tpl[gt_vis].reshape(-1, 1, 2).astype(np.float32), H.astype(np.float32)
            ).reshape(-1, 2)
            err = np.linalg.norm(proj - gt_xy[gt_vis], axis=1) * scale
            per_frame.append(float(np.median(err)))

    n = len(paths)
    e = np.asarray(per_frame)
    out = {
        "frames": n,
        "cobertura": 100.0 * len(e) / n,
        "sin_homografia": n_no_h,
        "pocos_keypoints": n_pocos_kpts,
        "err_p50": float(np.median(e)) if len(e) else float("nan"),
        "err_p90": float(np.percentile(e, 90)) if len(e) else float("nan"),
        "usables": 100.0 * float(np.mean(e < USABLE_PX)) if len(e) else 0.0,
        "inliers": 100.0 * float(np.mean(inlier_frac)) if inlier_frac else 0.0,
        # utiles = sobre TODOS los frames, no solo los cubiertos: junta cobertura y
        # precision en un unico numero, que es lo que hace falta para comparar
        # checkpoints (uno con 100% de cobertura y 40 px es peor que otro con 70% y 8).
        "utiles10": 100.0 * float(np.sum(e < 10.0)) / n if len(e) else 0.0,
        "utiles25": 100.0 * float(np.sum(e < 25.0)) / n if len(e) else 0.0,
    }
    if verbose:
        report(out, str(weights))
    return out


def report(m, tag=""):
    print("\n  homografia sobre val " + ("(" + tag + ")" if tag else ""))
    print("    cobertura      {:5.1f} %   ({} frames; {} con <min kpts, {} sin H)".format(
        m["cobertura"], m["frames"], m["pocos_keypoints"], m["sin_homografia"]))
    print("    error p50      {:5.1f} px   (normalizado a 1920 de ancho)".format(m["err_p50"]))
    print("    error p90      {:5.1f} px".format(m["err_p90"]))
    print("    usables <10px  {:5.1f} %   (de los cubiertos)".format(m["usables"]))
    print("    utiles <10px   {:5.1f} %   <25px {:5.1f} %   (de TODOS los frames)".format(
        m["utiles10"], m["utiles25"]))
    print("    inliers RANSAC {:5.1f} %\n".format(m["inliers"]))


def val_dirs(split="val", root=None):
    root = Path(root) if root else ROOT
    return [root / "datasets" / "hockeyrink" / "images" / split,
            root / "datasets" / "hockeyrink_nhl" / "images" / split]


def sweep(weights_dir, split="val", imgsz=1024, device=0, conf=0.3, copiar=True):
    """Evalua todos los checkpoints de un run y se queda con el mejor por homografia.

    Existe porque el fitness de Ultralytics pondera el mAP de CAJA, y en este
    problema la caja se degrada mientras los keypoints mejoran: medido en
    yolo26m-13, best.pt da 85.2 px de error de keypoint y last.pt 63.4 px. Elegir
    por fitness te deja el checkpoint equivocado.

    Se hace DESPUES de entrenar, sobre los checkpoints que deja save_period, en vez
    de en un callback: evaluar carga un segundo modelo en la GPU y durante el
    entrenamiento no hay hueco en 8 GB (el entrenamiento ya ocupa ~7.8).
    """
    import shutil

    d = Path(weights_dir)
    ckpts = sorted(d.glob("epoch*.pt"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    for extra in ("last.pt", "best.pt"):
        if (d / extra).exists():
            ckpts.append(d / extra)
    if not ckpts:
        print("  sin checkpoints en {}".format(d))
        return None

    dirs = val_dirs(split)
    filas = []
    for c in ckpts:
        try:
            m = evaluate(c, dirs, imgsz=imgsz, device=device, conf=conf, verbose=False)
        except Exception as exc:
            print("  {:14s} fallo: {}".format(c.name, exc))
            continue
        filas.append((m["utiles25"], m["utiles10"], -m["err_p50"], c, m))
        print("  {:14s} cobertura {:5.1f} %   p50 {:7.1f} px   utiles<25px {:5.1f} %".format(
            c.name, m["cobertura"], m["err_p50"], m["utiles25"]))
    if not filas:
        return None
    filas.sort(reverse=True)
    mejor = filas[0]
    print("")
    print("  MEJOR por homografia: {}".format(mejor[3].name))
    report(mejor[4], mejor[3].name)
    if copiar:
        dst = d / "best_homography.pt"
        shutil.copy2(mejor[3], dst)
        print("  copiado a {}".format(dst))
    return mejor[3]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="fichero .pt, o carpeta weights/ con --sweep")
    ap.add_argument("--sweep", action="store_true",
                    help="evalua todos los checkpoints de la carpeta y elige el mejor")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.3,
                    help="umbral de confianza por keypoint")
    ap.add_argument("--box-conf", type=float, default=0.001,
                    help="umbral de la caja; solo hay un rink, asi que se coge la mejor")
    ap.add_argument("--min-kpts", type=int, default=6)
    ap.add_argument("--ransac-px", type=float, default=8.0)
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()
    if args.sweep:
        sweep(args.weights, split=args.split, imgsz=args.imgsz, conf=args.conf)
    else:
        evaluate(args.weights, val_dirs(args.split), conf=args.conf, min_kpts=args.min_kpts,
                 ransac_px=args.ransac_px, imgsz=args.imgsz, box_conf=args.box_conf)
