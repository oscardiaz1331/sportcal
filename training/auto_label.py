"""Auto-etiquetado masivo: YOLO ancla + refinamiento clasico + puerta de aceptacion.

Cambio de objetivo respecto a todo lo probado hasta ahora en la via clasica: no
hace falta una homografia fiable en CADA frame (eso es lo que llevaba todo el dia
fallando -- ajuste conjunto, KLT, crecimiento de circulos, bootstrap geometrico).
Para generar datos de entrenamiento basta con saber decir "de este me fio" y
descartar sin piedad el resto. Con horas de video, aceptar solo el 10-20% de los
frames sigue dando miles de etiquetas nuevas gratis.

Pipeline por frame:

  1. ANCLA: YOLO predice 56 keypoints (imprecisos, ~60 px, pero nunca fallan del
     todo) -> se ajusta una H inicial por RANSAC, igual que hace rink_metric.
  2. REFINO: fit_homography_lines.fit() ajusta esa H con el zocalo de las vallas
     (señal fuerte, z~50-80) y las curvas de color, coarse-to-fine.
  3. PUERTA DE ACEPTACION: se piden CUATRO señales de acuerdo independientes;
     si falla una sola, se descarta el frame entero. No se guarda nada "a medias".

        - iou_contorno:  el contorno de pista que predice H debe solapar mucho
          con la region de hielo detectada por color (best_region)
        - zocalo_z:      el zocalo debe seguir localizandose con margen alto en
          la posicion final (si esto es bajo, la H no esta anclada a nada real)
        - curvas_ok:     cuantas curvas de color (lineas, circulos) caen donde
          la H dice que deben caer
        - inliers_yolo:  fraccion de keypoints YOLO que sobrevivieron el RANSAC
          inicial (si casi ninguno encajaba, la ancla ya era mala)

  4. Si se acepta: se reproyectan los 56 keypoints del template (como
     relabel_reproject.py) y se escribe la etiqueta.

USAR PRIMERO --calibrar: corre esto sobre datasets YA etiquetados (con verdad
disponible) para medir que error real tienen los frames que la puerta acepta,
ANTES de soltarlo sobre video crudo sin ninguna forma de verificar.

    python training/auto_label.py --calibrar --pesos <ruta a best_homography.pt>
    python training/auto_label.py --video nhl5.mp4 --cada 5 --salida hockeyrink_auto
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

import diagnose_lines_fit as D  # noqa: E402
import fit_homography_lines as F  # noqa: E402
import ice_lines_probe as P  # noqa: E402
import make_line_masks as MM  # noqa: E402
import relabel_reproject as RP  # noqa: E402
import rink  # noqa: E402
import rink_metric as RM  # noqa: E402

NKPT = 56
UMBRALES = dict(iou=0.85, zocalo_z=15.0, curvas_ok=2, inliers_yolo=0.5)


def kickplate_z(img, region, H, params_rink, w, h, offsets=(-30, -24, -18, -12, 12, 18, 24, 30)):
    """Igual que localization_z pero para el contorno cerrado del zocalo."""
    base, nz = F.outline_normals(H, params_rink, w, h)
    if base is None:
        return 0.0
    da, db = P.local_chroma(img, win=int(61 * w / 1920) | 1)
    sig = w / 1920.0
    v0 = P.integrate(db, base + nz * F.OFFSET_ZOCALO * sig, w, h)
    if v0 is None:
        return 0.0
    alt = [P.integrate(db, base + nz * o * sig, w, h) for o in offsets]
    alt = [a for a in alt if a is not None]
    if len(alt) < 5:
        return 0.0
    return float((v0 - np.mean(alt)) / (np.std(alt) or 1e-6))


def anchor_H(model, img, tpl, conf=0.25, min_kpts=8, ransac_px=8.0):
    """YOLO -> H inicial por RANSAC. Devuelve (H, inlier_frac) o (None, 0)."""
    h, w = img.shape[:2]
    r = model.predict(source=img, imgsz=1024, device=0, verbose=False, conf=0.001)[0]
    if r.keypoints is None or len(r.keypoints.xy) == 0:
        return None, 0.0
    xy = r.keypoints.xy[0].cpu().numpy()
    cf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(NKPT)
    sel = cf >= conf
    if sel.sum() < min_kpts:
        return None, 0.0
    scale = RM.REF_WIDTH / w
    H, inl = RM.fit_homography(tpl[sel], xy[sel], ransac_px / scale)
    if H is None:
        return None, 0.0
    return H, float(inl.mean())


def evaluate_frame(model, img, params_rink, tpl, polys, umbrales=UMBRALES):
    """Corre el pipeline completo sobre un frame. Devuelve dict con todo, incluida
    la decision final (`aceptado`)."""
    h, w = img.shape[:2]
    out = {"aceptado": False, "motivo": None}

    H0, inl_yolo = anchor_H(model, img, tpl)
    if H0 is None:
        out["motivo"] = "sin ancla YOLO"
        return out
    out["inliers_yolo"] = inl_yolo
    if inl_yolo < umbrales["inliers_yolo"]:
        out["motivo"] = "pocos inliers en el ancla"
        return out

    region = P.best_region(img)
    if region.mean() < 0.15:
        out["motivo"] = "sin region de hielo"
        return out

    H1 = F.fit(img, region, H0, params_rink, polys, w_region=2.0, w_zocalo=4.0)

    pred_region = P.true_region(H1, params_rink, w, h)
    iou_val = P.iou(pred_region > 0, region > 0) if pred_region is not None else 0.0
    out["iou"] = iou_val
    if iou_val < umbrales["iou"]:
        out["motivo"] = "contorno no coincide con el hielo"
        out["H"] = H1
        return out

    zz = kickplate_z(img, region, H1, params_rink, w, h)
    out["zocalo_z"] = zz
    if zz < umbrales["zocalo_z"]:
        out["motivo"] = "zocalo no se localiza"
        out["H"] = H1
        return out

    rep = D.curve_report(img, region, H1, params_rink, polys)
    n_ok = sum(1 for r in rep if abs(r["d"]) <= D.ACIERTO_PX)
    out["curvas_ok"] = n_ok
    out["curvas_total"] = len(rep)
    if n_ok < umbrales["curvas_ok"]:
        out["motivo"] = "pocas curvas de acuerdo"
        out["H"] = H1
        return out

    out["aceptado"] = True
    out["H"] = H1
    return out


def label_from_H(H, tpl, w, h, margin=0.0):
    """56 keypoints reproyectados por H, formato v=1 (generado), o None si nada visible."""
    xy, ok = MM.project_points(H, tpl, w, h)
    norm = xy / np.array([w, h])
    keep = ok & (norm[:, 0] >= -margin) & (norm[:, 0] <= 1 + margin) & \
        (norm[:, 1] >= -margin) & (norm[:, 1] <= 1 + margin)
    if keep.sum() < 6:
        return None
    kpts = np.zeros((NKPT, 3))
    kpts[keep, :2] = np.clip(norm[keep], 0.0, 1.0)
    kpts[keep, 2] = 1
    k = kpts[keep, :2]
    x0, y0 = k[:, 0].min(), k[:, 1].min()
    x1, y1 = k[:, 0].max(), k[:, 1].max()
    box = np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
    return kpts, box


def calibrar(pesos, n=60, split="train"):
    """Corre la puerta sobre frames YA etiquetados y mide el error REAL de lo
    que se acepta, contra lo que se rechaza. Esto es lo que decide si los
    umbrales de UMBRALES sirven antes de soltarlo sobre video sin verdad."""
    from ultralytics import YOLO

    model = YOLO(str(pesos))
    filas = []
    for ds, params in (("hockeyrink", rink.RINK_IIHF), ("hockeyrink_nhl", rink.RINK_NHL)):
        tpl = rink.build_template(params)
        polys = MM.rink_polylines(params)
        lbls = sorted((ROOT / "datasets" / ds / "labels" / split).glob("*.txt"))
        paso = max(1, len(lbls) // n)
        for lbl in lbls[::paso][:n]:
            ip = ROOT / "datasets" / ds / "images" / split / (lbl.stem + ".jpg")
            if not ip.exists():
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue
            h, w = img.shape[:2]
            rec = RP.read_label(lbl)
            if rec is None:
                continue
            fq, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
            if fq is None:
                continue
            Hgt = fq[0]
            res = evaluate_frame(model, img, params, tpl, polys)
            if "H" not in res:
                filas.append((res["aceptado"], np.inf, res.get("motivo")))
                continue
            e = F.geom_error(res["H"], Hgt, tpl, w, h)
            filas.append((res["aceptado"], e, res.get("motivo")))
            print("  {:20s} {} err={:7.1f}px  {}".format(
                lbl.stem[:20], "ACEPTADO" if res["aceptado"] else "rechazado ", e,
                "" if res["aceptado"] else "(" + str(res.get("motivo")) + ")"))

    ac = np.array([e for ok, e, _ in filas if ok and np.isfinite(e)])
    re = np.array([e for ok, e, _ in filas if not ok and np.isfinite(e)])
    n_ac = sum(1 for ok, _, _ in filas if ok)
    print("\n=== calibracion ({} frames) ===".format(len(filas)))
    print("  aceptados: {} ({:.0f} %)".format(n_ac, 100 * n_ac / max(len(filas), 1)))
    if len(ac):
        print("  error de los ACEPTADOS:  p50 {:.1f} px   p90 {:.1f} px   <10px {:.0f}%   <25px {:.0f}%".format(
            np.median(ac), np.percentile(ac, 90), 100 * np.mean(ac < 10), 100 * np.mean(ac < 25)))
    if len(re):
        print("  error de los rechazados: p50 {:.1f} px   (para contraste)".format(np.median(re)))


def run_video(video_path, pesos, cada, salida, max_frames=None):
    from ultralytics import YOLO

    video_path = Path(video_path)
    stem = video_path.stem
    params = rink.RINK_NHL   # los videos propios son pistas NHL
    tpl = rink.build_template(params)
    polys = MM.rink_polylines(params)
    model = YOLO(str(pesos))

    out_img = ROOT / "datasets" / salida / "images" / "train"
    out_lbl = ROOT / "datasets" / salida / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(ROOT / video_path))
    i = 0
    vistos = aceptados = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % cada == 0:
            if max_frames and vistos >= max_frames:
                break
            vistos += 1
            res = evaluate_frame(model, frame, params, tpl, polys)
            if res["aceptado"]:
                h, w = frame.shape[:2]
                lab = label_from_H(res["H"], tpl, w, h)
                if lab is not None:
                    kpts, box = lab
                    name = "{}_{:06d}".format(stem, i)
                    cv2.imwrite(str(out_img / (name + ".jpg")), frame)
                    RP.write_label(out_lbl / (name + ".txt"), 0, box, kpts)
                    aceptados += 1
            if vistos % 50 == 0:
                print("  {} frames vistos, {} aceptados ({:.0f} %)".format(
                    vistos, aceptados, 100 * aceptados / vistos))
        i += 1
    cap.release()
    print("\n{}: {} vistos, {} aceptados ({:.0f} %) -> {}".format(
        video_path.name, vistos, aceptados, 100 * aceptados / max(vistos, 1), out_img.parent))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(ROOT / "runs/hockeyrink/yolo26m-17/weights/best_homography.pt"))
    ap.add_argument("--calibrar", action="store_true")
    ap.add_argument("--n", type=int, default=60, help="frames a calibrar")
    ap.add_argument("--video", default=None)
    ap.add_argument("--cada", type=int, default=5, help="1 de cada N frames del video")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--salida", default="hockeyrink_auto")
    args = ap.parse_args()

    if args.calibrar:
        calibrar(args.pesos, n=args.n)
    elif args.video:
        run_video(args.video, args.pesos, args.cada, args.salida, args.max_frames)
    else:
        print("pasa --calibrar o --video <fichero>")
