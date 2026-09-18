"""Diagnostico de la red de segmentacion: localiza tan bien como la via clasica?

Mismo criterio que diagnose_lines_fit.py (desliza cada curva del template por su
normal y mide donde pica la respuesta), pero la respuesta ya no es color clasico
(F.responses) sino el mapa de probabilidad por clase que saca la red entrenada en
train_lines_seg.py. Comparacion directa, mismo z-score, misma tabla -- para poder
poner la fila de la red al lado de las filas clasicas ya documentadas en CLAUDE.md
(zocalo z=79.9, respuesta cruda de linea z=3.3).

Evalua SOLO sobre el split val de cada dataset de lineas (la red nunca vio esas
imagenes en entrenamiento) -- si se usara train el numero saldria falsamente alto
por sobreajuste, que ya sabemos que existe (loss de train seguia bajando con
IoU de val plano).

    python training/diagnose_lines_seg.py
    python training/diagnose_lines_seg.py --dataset hockeyrink_nhl --n 40
"""
import os

_CUDNN_DIR = os.sep.join(("NVIDIA", "CUDNN"))

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if _CUDNN_DIR not in p
)

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import make_line_masks as MM  # noqa: E402
import rink  # noqa: E402
from diagnose_lines_fit import ACIERTO_PX, RANGO_PX, dibuja, slide  # noqa: E402
from ice_lines_probe import normals  # noqa: E402
from relabel_reproject import read_label  # noqa: E402
from train_lines_seg import MEAN, NCLS, STD, UNetResNet34  # noqa: E402

PESOS = ROOT / "runs" / "lineas_seg" / "best.pt"


def load_model(device):
    ckpt = torch.load(PESOS, map_location=device)
    model = UNetResNet34(NCLS).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("cargado {} (epoch {}, imgsz {})".format(PESOS, ckpt["epoch"], ckpt["imgsz"]))
    return model, tuple(ckpt["imgsz"])


@torch.no_grad()
def predict_probs(model, img, imgsz, device):
    """Devuelve (NCLS, h, w) de probabilidades, reescalado al tamano ORIGINAL de img."""
    h, w = img.shape[:2]
    small = cv2.resize(img, imgsz, interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, 1)[0].cpu().numpy()
    out = np.stack([cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
                     for c in range(probs.shape[0])])
    return out


def curve_report_seg(probs, H, polys, Hs=None, max_spread=8.0):
    """Igual que diagnose_lines_fit.curve_report pero R = mapa de prob. de la red.

    `Hs` (bootstrap de la homografia ground truth, ver make_line_masks) filtra
    curvas cuya "verdad" no es de fiar -- la misma logica que usa render_mask()
    para marcar IGNORE al generar las mascaras de entrenamiento, que este
    diagnostico nunca heredo. Sin esto, en un plano cerrado (p.ej. camara pegada
    a la porteria, sin ninguna linea azul en cuadro) la H ground truth se ajusta
    solo con puntos cercanos (crease, circulo, vallas) y EXTRAPOLA una posicion
    para la linea azul que puede estar a decenas de px de cualquier sitio real;
    la red "falla" contra una referencia que en realidad no significa nada.
    Confirmado visualmente: los frames clip_00015/00075 (camara de porteria, sin
    ninguna linea azul visible) daban azul_A a d=+-40px -- el limite del rango de
    busqueda, o sea ruido -- y eran la mayoria de los "fallos" de esa clase.
    """
    h, w = probs.shape[1:]
    sig = max(1e-6, w / 1920.0)
    out = []
    for cls, world in polys:
        if cls == 0:
            continue
        if Hs:
            spread = MM.point_spread(Hs, world, w, h)
            fiable = np.isfinite(spread) & (spread < max_spread * w / MM.REF_WIDTH)
            if fiable.mean() < 0.5:
                continue
        R = probs[cls]
        xy, ok = MM.project_points(H, world, w, h)
        if ok.sum() < 60:
            continue
        base = xy[ok]
        inb = (base[:, 0] > 1) & (base[:, 0] < w - 2) & (base[:, 1] > 1) & (base[:, 1] < h - 2)
        if inb.sum() < 60:
            continue
        base = base[inb]
        nz = normals(base)
        ds, vals = slide(R, base, nz, w, h, rango=RANGO_PX * sig, paso=max(1.0, sig))
        fin = np.isfinite(vals)
        if fin.sum() < 20:
            continue
        ds, vals = ds[fin], vals[fin]
        i = int(np.argmax(vals))
        d_best = float(ds[i] / sig)
        lejos = np.abs(ds) > 12 * sig
        margen = float(vals[i] - np.median(vals[lejos])) if lejos.any() else 0.0
        ruido = float(np.std(vals[lejos])) if lejos.sum() > 3 else 1e-3
        out.append({"cls": cls, "d": d_best, "z": margen / (ruido or 1e-6),
                    "n": int(len(base)), "base": base, "nz": nz})
    return out


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, imgsz = load_model(device)

    params_rink = rink.RINK_NHL if "nhl" in args.dataset else rink.RINK_IIHF
    tpl = rink.build_template(params_rink)
    polys = MM.rink_polylines(params_rink)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    por_clase = defaultdict(lambda: {"n": 0, "ok": 0, "d": [], "z": []})
    por_frame = []
    lbls = sorted((ROOT / "datasets" / args.dataset / "labels" / "val").glob("*.txt"))
    print("frames disponibles en val: {}".format(len(lbls)))
    hechos = 0

    for lbl in lbls:
        if hechos >= args.n:
            break
        ip = ROOT / "datasets" / args.dataset / "images" / "val" / (lbl.stem + ".jpg")
        if not ip.exists():
            continue
        rec = read_label(lbl)
        img = cv2.imread(str(ip))
        if rec is None or img is None:
            continue
        h, w = img.shape[:2]
        fq, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
        if fq is None:
            continue
        Hgt, inl_world, inl_img = fq
        Hs = MM.bootstrap_homographies(inl_world, inl_img)
        probs = predict_probs(model, img, imgsz, device)
        rep = curve_report_seg(probs, Hgt, polys, Hs=Hs)
        if len(rep) < 2:
            continue
        hechos += 1

        aciertos = sum(1 for r in rep if abs(r["d"]) <= ACIERTO_PX)
        for r in rep:
            c = por_clase[r["cls"]]
            c["n"] += 1
            c["ok"] += int(abs(r["d"]) <= ACIERTO_PX)
            c["d"].append(abs(r["d"]))
            c["z"].append(r["z"])
        por_frame.append({"stem": lbl.stem, "img": ip, "rep": rep,
                          "n": len(rep), "ok": aciertos,
                          "frac": aciertos / len(rep)})

    print("\n=== red de segmentacion sobre {} val  ({} frames) ===".format(args.dataset, hechos))
    print("\n  QUE CLASES son de fiar (desplazamiento del maximo de cada curva):")
    print("  clase                veces  acierta  |d| p50   margen z")
    filas = [(v["ok"] / v["n"], k, v) for k, v in por_clase.items() if v["n"] >= 3]
    for frac, k, v in sorted(filas, reverse=True):
        print("  {:2d} {:18s} {:4d}   {:4.0f} %   {:6.1f}px   {:6.1f}".format(
            k, MM.CLASSES[k], v["n"], 100 * frac, np.median(v["d"]), np.median(v["z"])))

    if por_frame:
        fr = np.array([f["frac"] for f in por_frame])
        print("\n  QUE FRAMES son recuperables (fraccion de curvas que aciertan):")
        print("    todas aciertan        : {:4.0f} %".format(100 * np.mean(fr >= 0.999)))
        print("    mayoria (>=2/3)       : {:4.0f} %".format(100 * np.mean(fr >= 0.667)))
        print("    ninguna acierta       : {:4.0f} %".format(100 * np.mean(fr <= 0.001)))
        print("    curvas en cuadro: mediana {:.0f}".format(np.median([f["n"] for f in por_frame])))

        por_frame.sort(key=lambda f: f["frac"])
        peores = por_frame[:args.figuras]
        mejores = por_frame[-args.figuras:]
        for etiqueta, grupo in (("MAL", peores), ("BIEN", mejores)):
            for f in grupo:
                img = cv2.imread(str(f["img"]))
                txt = "{}  {}  curvas que aciertan: {}/{}  (red seg)".format(
                    etiqueta, f["stem"][:20], f["ok"], f["n"])
                cv2.imwrite(str(out_dir / "{}_seg_{}_{}.jpg".format(args.dataset, etiqueta.lower(), f["stem"])),
                            dibuja(img, f["rep"], txt))
        print("\n  figuras en {}".format(out_dir))
    else:
        print("\n  ningun frame con >=2 curvas en cuadro -- revisar dataset/split")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hockeyrink")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--figuras", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "scratch_frames" / "diag_lineas_seg"))
    run(ap.parse_args())
