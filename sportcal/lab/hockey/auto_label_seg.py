"""Auto-etiquetado masivo via segmentacion -> homografia directa (sin YOLO).

Reemplaza el enfoque de auto_label.py (YOLO ancla + refinamiento clasico +
puerta de 4 señales), que se quedo esperando a que YOLO bajara de 20-25 px de
error para que la puerta empezara a discriminar (calibrado, nunca paso de
38% de precision -- ver CLAUDE.md sección 5).

sportcal/lab/hockey/seg_to_homography.py ya resuelve H directamente de la segmentacion,
sin ancla YOLO, con precision comparable o mejor que YOLO EN LOS FRAMES QUE
CUBRE (IIHF p50=16.4px p90=20.7px 100%<25px; NHL p50=37.8px -- ver CLAUDE.md
sección 7). La puerta de aceptacion ya esta dentro de `solve_from_probs()`
(exige dof>=10, 5+ correspondencias con direcciones no paralelas): si devuelve
H, es de fiar; si devuelve None, se descarta. No hace falta calibrar umbrales
de color aparte como en auto_label.py.

Coste: cobertura baja (12% IIHF / 7% NHL de los frames individuales), pero
con horas de video eso sigue siendo miles de etiquetas nuevas gratis, y con
MUCHA mas confianza por etiqueta que la via anterior.

    python -m sportcal.lab.hockey.auto_label_seg --calibrar --n 60
    python -m sportcal.lab.hockey.auto_label_seg --video nhl5.mp4 --cada 5 --salida hockeyrink_auto_seg
"""
import os

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sportcal.paths import ROOT

from sportcal.lab.hockey import make_line_masks as MM
from sportcal.lab.hockey import relabel_reproject as RP
from sportcal.sports.hockey import rink
from sportcal.core.labels import NKPT, label_from_H  # noqa: E402
from sportcal.lab.hockey.diagnose_lines_seg import load_model, predict_probs  # noqa: E402
from sportcal.core.geometry import geom_error
from sportcal.lab.hockey.seg_to_homography import solve_from_probs  # noqa: E402


def calibrar(n=60, split="train"):
    """Igual que auto_label.calibrar pero con este pipeline -- corre sobre
    frames YA etiquetados y mide el error real de lo aceptado."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, imgsz = load_model(device)
    filas = []
    for ds, params in (("hockeyrink", rink.RINK_IIHF), ("hockeyrink_nhl", rink.RINK_NHL)):
        tpl = rink.build_template(params)
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
            probs = predict_probs(model, img, imgsz, device)
            Hest, n_l, n_c, costo = solve_from_probs(probs, params, w, h)
            aceptado = Hest is not None
            e = geom_error(Hest, Hgt, tpl, w, h) if aceptado else np.inf
            filas.append((ds, aceptado, e, n_l, n_c, costo))
            print("  {:10s} {:20s} {}  n_l={} n_c={} costo={:.4f}  err={:7.1f}px".format(
                ds, lbl.stem[:20], "ACEPTADO " if aceptado else "rechazado", n_l, n_c, costo, e))

    for ds in ("hockeyrink", "hockeyrink_nhl"):
        sub = [f for f in filas if f[0] == ds]
        ac = np.array([e for _, ok, e, _, _, _ in sub if ok and np.isfinite(e)])
        n_ac = sum(1 for _, ok, _, _, _, _ in sub if ok)
        print("\n=== {} ({} frames) ===".format(ds, len(sub)))
        print("  aceptados: {} ({:.0f} %)".format(n_ac, 100 * n_ac / max(len(sub), 1)))
        if len(ac):
            print("  error de los ACEPTADOS:  p50 {:.1f} px   p90 {:.1f} px   <25px {:.0f} %   <60px {:.0f} %".format(
                np.median(ac), np.percentile(ac, 90), 100 * np.mean(ac < 25), 100 * np.mean(ac < 60)))
        ac_costo = np.array([(c, e) for _, ok, e, _, _, c in sub if ok and np.isfinite(e)])
        if len(ac_costo) > 3:
            r = np.corrcoef(ac_costo[:, 0], np.log1p(np.clip(ac_costo[:, 1], 0, 1e6)))[0, 1]
            print("  coste de refinamiento vs error (dentro de los aceptados): r={:.2f} (log)".format(r))
            for q in (0.9, 0.75, 0.5):
                umbral = np.quantile(ac_costo[:, 0], q)
                sel = ac_costo[:, 0] <= umbral
                if sel.sum() >= 3:
                    print("    costo<=p{:.0f} ({:.4f}): quedan {}/{}  p50={:.1f}px  p90={:.1f}px".format(
                        100 * q, umbral, int(sel.sum()), len(ac_costo),
                        np.median(ac_costo[sel, 1]), np.percentile(ac_costo[sel, 1], 90)))


def run_video(video_path, cada, salida, max_frames=None, max_costo=1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, imgsz = load_model(device)

    video_path = Path(video_path)
    stem = video_path.stem
    params = rink.RINK_NHL   # los videos propios son pistas NHL
    tpl = rink.build_template(params)

    out_img = ROOT / "datasets" / salida / "images" / "train"
    out_lbl = ROOT / "datasets" / salida / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(ROOT / video_path))
    i = 0
    vistos = aceptados = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % cada == 0:
            if max_frames and vistos >= max_frames:
                break
            vistos += 1
            h, w = frame.shape[:2]
            probs = predict_probs(model, frame, imgsz, device)
            Hest, n_l, n_c, costo = solve_from_probs(probs, params, w, h)
            if Hest is not None and costo > max_costo:
                Hest = None
            if Hest is not None:
                lab = label_from_H(Hest, tpl, w, h)
                if lab is not None:
                    kpts, box = lab
                    name = "{}_{:06d}".format(stem, i)
                    cv2.imwrite(str(out_img / (name + ".jpg")), frame)
                    RP.write_label(out_lbl / (name + ".txt"), 0, box, kpts)
                    aceptados += 1
            if vistos % 50 == 0:
                fps = vistos / (time.time() - t0 + 1e-9)
                print("  {} frames vistos, {} aceptados ({:.0f} %)  {:.1f} fps".format(
                    vistos, aceptados, 100 * aceptados / vistos, fps))
        i += 1
    cap.release()
    print("\n{}: {} vistos, {} aceptados ({:.0f} %) -> {}".format(
        video_path.name, vistos, aceptados, 100 * aceptados / max(vistos, 1), out_img.parent))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrar", action="store_true")
    ap.add_argument("--n", type=int, default=60, help="frames a calibrar")
    ap.add_argument("--video", default=None)
    ap.add_argument("--cada", type=int, default=5, help="1 de cada N frames del video")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-costo", type=float, default=0.001,
                    help="segunda señal de confianza -- calibrado en IIHF (r=0.97 con el error "
                         "real): costo<=0.0009 tira los casos catastroficos del limite de dof "
                         "por solo 2/9 frames. En NHL la muestra (n=6) es aun insuficiente para "
                         "fiarse de este numero -- correr --calibrar de nuevo segun crezca el "
                         "dataset NHL de referencia.")
    ap.add_argument("--salida", default="hockeyrink_auto_seg")
    args = ap.parse_args()

    if args.calibrar:
        calibrar(n=args.n)
    elif args.video:
        run_video(args.video, args.cada, args.salida, args.max_frames, args.max_costo)
    else:
        print("pasa --calibrar o --video <fichero>")
