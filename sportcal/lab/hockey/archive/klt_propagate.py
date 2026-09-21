"""Prototipo: propagar la homografia por tracking KLT en vez de resolverla cada frame.

La razon de esto: el ajuste por lineas falla porque un frame suelto tiene solo
~3 curvas (mediana) para 8 grados de libertad, y ~30% de las veces alguna esta
contaminada por publicidad pintada bajo el hielo. Pero los planos de camara duran
cientos o miles de frames sin cortar (medido: nhl3.mp4 va sin cortes desde el
frame 0 hasta el 9523, 159 s a 60 fps). En vez de resolver el problema mal
condicionado en cada frame, se resuelve UNA VEZ en un frame ancla (aqui, con los
keypoints ground truth, como sustituto de "el mejor ajuste clasico disponible") y
se seguye el resto del plano por tracking.

Metodo, frame a frame:

  1. en el ancla se siembran esquinas fuertes (goodFeaturesToTrack) dentro de la
     region de pista, SIN exigir que esten sobre una linea concreta: cualquier
     punto de la superficie del hielo tiene una posicion en el mundo conocida via
     H_ancla^-1, asi que no hay que resolver la correspondencia "que keypoint es
     este", que es el cuello de botella de toda la via clasica.
  2. cada frame se trackean con Lucas-Kanade piramidal (calcOpticalFlowPyrLK).
  3. con los puntos trackeados (pixel) y sus coordenadas de mundo (fijas, del
     ancla) se reajusta H por RANSAC -- esto descarta solo los puntos que se
     desvian de una homografia comun, tipicamente jugadores que se han llevado
     por delante el punto que se les puso encima.
  4. cada REPLENISH frames se siembran puntos nuevos donde se han perdido, para
     no quedarse sin puntos segun avanza el plano.

Se mide el DRIFT real: en los frames que tienen keypoints ground truth (no todos,
pero hay bastantes en nhl3 entre 2610 y 9060) se compara la H propagada con la H
que sale de esos keypoints, con la misma metrica de siempre (error de reproyeccion
en px a 1920). Esto es honesto -- no es autoconsistencia, es contraste con verdad
independiente en cada punto de control.

    python -m sportcal.lab.hockey.archive.klt_propagate
    python -m sportcal.lab.hockey.archive.klt_propagate --video nhl3.mp4 --frame0 2610 --hasta 9060
"""
import os

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sportcal.paths import ROOT

from sportcal.lab.hockey import fit_homography_lines as F
from sportcal.lab.hockey import ice_lines_probe as P
from sportcal.lab.hockey import make_line_masks as MM
from sportcal.sports.hockey import rink
from sportcal.lab.hockey.relabel_reproject import read_label  # noqa: E402

LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def seed_points(gray, region, H, exclude=None, max_new=250, min_dist=14):
    """Esquinas fuertes dentro de la region, con su coordenada de mundo via H^-1.

    `exclude` es una mascara de puntos ya trackeados (para no resembrar encima).
    """
    mask = (region > 0).astype(np.uint8) * 255
    if exclude is not None:
        mask[exclude] = 0
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_new, qualityLevel=0.015,
                                  minDistance=min_dist, mask=mask, blockSize=7)
    if pts is None or len(pts) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    img_pts = pts.reshape(-1, 2).astype(np.float32)
    world = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2),
                                     np.linalg.inv(H).astype(np.float32)).reshape(-1, 2)
    return img_pts, world


def exclusion_mask(shape, img_pts, radius=10):
    m = np.zeros(shape, np.uint8)
    for x, y in img_pts.astype(int):
        cv2.circle(m, (x, y), radius, 255, -1)
    return m > 0


def track(video_path, frame0, hasta, H0, params_rink, tpl, region0, replenish=25,
          max_pts=500, ransac_px=4.0, min_pts=8, verbose_every=300):
    """Genera (frame_idx, H, n_inliers) para cada frame de frame0 a hasta."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame0)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("no se pudo leer el frame {}".format(frame0))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_pts, world_pts = seed_points(prev_gray, region0, H0, max_new=max_pts)
    print("  ancla: {} puntos sembrados".format(len(img_pts)))

    H = H0
    t0 = time.time()
    for i in range(frame0, hasta):
        ok, frame = cap.read()
        if not ok:
            print("  video terminado en el frame {}".format(i))
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(img_pts) >= min_pts:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, img_pts.reshape(-1, 1, 2), None, **LK_PARAMS)
            status = status.ravel() == 1
            new_pts = new_pts.reshape(-1, 2)
            img_pts, world_pts = new_pts[status], world_pts[status]

        h, w = gray.shape
        if len(img_pts) >= min_pts:
            Ht, inl = cv2.findHomography(world_pts.astype(np.float32), img_pts.astype(np.float32),
                                         cv2.RANSAC, ransac_px * w / 1920.0)
            if Ht is not None and np.isfinite(Ht).all():
                inl = inl.ravel().astype(bool)
                H = Ht
                img_pts, world_pts = img_pts[inl], world_pts[inl]

        yield i + 1, H, len(img_pts)

        # resembrar donde hagan falta puntos
        if (i - frame0) % replenish == 0 or len(img_pts) < max_pts // 3:
            excl = exclusion_mask(gray.shape, img_pts) if len(img_pts) else None
            reg = P.best_region(frame) if (i - frame0) % (replenish * 4) == 0 else \
                (region0 if len(img_pts) == 0 else None)
            if reg is None:
                # region barata: reproyectar el contorno con la H actual
                out = P.true_region(H, params_rink, w, h)
                reg = out if out is not None else region0
            new_i, new_w = seed_points(gray, reg, H, exclude=excl,
                                       max_new=max_pts - len(img_pts))
            if len(new_i):
                img_pts = np.vstack([img_pts, new_i]) if len(img_pts) else new_i
                world_pts = np.vstack([world_pts, new_w]) if len(world_pts) else new_w

        prev_gray = gray
        if verbose_every and (i - frame0) % verbose_every == 0:
            fps = (i - frame0 + 1) / (time.time() - t0 + 1e-9)
            print("  frame {:6d}   puntos activos {:4d}   {:.0f} fps de proceso".format(
                i, len(img_pts), fps))
    cap.release()


def run(args):
    params_rink = rink.RINK_NHL if "nhl" in args.video else rink.RINK_IIHF
    tpl = rink.build_template(params_rink)
    base = ROOT / "datasets" / "hockeyrink_nhl"

    stem0 = "{}_{:05d}".format(Path(args.video).stem, args.frame0)
    lbl0 = None
    for split in ("train", "val"):
        p = base / "labels" / split / (stem0 + ".txt")
        if p.exists():
            lbl0 = p
            break
    if lbl0 is None:
        raise SystemExit("no hay ground truth en el frame ancla {}".format(stem0))

    rec = read_label(lbl0)
    img0 = cv2.imread(str(base / "images" / ("train" if (base / "images" / "train" /
                                                          (stem0 + ".jpg")).exists() else "val") /
                          (stem0 + ".jpg")))
    h, w = img0.shape[:2]
    fq, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
    if fq is None:
        raise SystemExit("el frame ancla no tiene homografia de verdad fiable")
    H0 = fq[0]
    region0 = P.best_region(img0)

    # todos los frames con ground truth de este video, dentro del rango, para medir drift
    checkpoints = {}
    for split in ("train", "val"):
        for p in (base / "labels" / split).glob("{}_*.txt".format(Path(args.video).stem)):
            idx = int(p.stem.split("_")[-1])
            if args.frame0 <= idx <= args.hasta:
                checkpoints[idx] = p

    print("video {}   ancla en frame {}   hasta {}   ({} puntos de control con GT)".format(
        args.video, args.frame0, args.hasta, len(checkpoints)))

    errores = []
    for idx, H, n_inl in track(ROOT / args.video, args.frame0, args.hasta, H0,
                               params_rink, tpl, region0, replenish=args.replenish):
        if idx not in checkpoints:
            continue
        rec_c = read_label(checkpoints[idx])
        fq_c, _ = MM.fit_from_label(rec_c[2], tpl, w, h, 8, 6.0, 8.0)
        if fq_c is None:
            continue
        Hgt = fq_c[0]
        e = F.geom_error(H, Hgt, tpl, w, h)
        t = (idx - args.frame0) / 59.94
        errores.append((idx, t, e, n_inl))
        print("  frame {:6d}  (t=+{:5.1f}s)   error vs GT = {:6.1f} px   puntos = {:4d}".format(
            idx, t, e, n_inl))

    if not errores:
        print("\n  sin puntos de control en el rango")
        return

    E = np.array([e for _, _, e, _ in errores])
    print("\n=== drift de la homografia propagada por KLT ===")
    print("  {} puntos de control entre t=0 y t=+{:.0f}s".format(len(E), errores[-1][1]))
    print("  error p50 {:.1f} px   p90 {:.1f} px   <10px {:.0f} %   <25px {:.0f} %".format(
        np.median(E), np.percentile(E, 90), 100 * np.mean(E < 10), 100 * np.mean(E < 25)))
    # a partir de que instante se degrada de forma sostenida (mirando cuando supera 25px
    # y no vuelve a bajar de ahi)
    malos = np.where(E > 25)[0]
    if len(malos):
        primero = malos[0]
        print("  primer punto de control por encima de 25 px: t=+{:.0f}s (frame {})".format(
            errores[primero][1], errores[primero][0]))
    else:
        print("  nunca supera 25 px en todo el rango medido")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="nhl3.mp4")
    ap.add_argument("--frame0", type=int, default=2610)
    ap.add_argument("--hasta", type=int, default=9060)
    ap.add_argument("--replenish", type=int, default=25)
    run(ap.parse_args())
