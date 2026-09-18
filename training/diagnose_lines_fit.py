"""Diagnostico del ajuste por lineas: que curvas sostienen la homografia y cuales la sabotean.

El ajuste conjunto de training/fit_homography_lines.py falla, pero "falla" no es
accionable. Esto lo desglosa: para CADA curva del template se busca, deslizandola
por su propia normal, donde la respuesta cromatica es maxima. Si ese maximo cae en
el sitio correcto (desplazamiento ~0) la curva esta sosteniendo el ajuste; si cae
lejos, esa curva esta tirando de la homografia hacia otro lado.

Con eso se responden tres cosas distintas:

  1. QUE CLASES son de fiar. Tabla agregada por clase: cuantas veces esta en cuadro
     y con que frecuencia su maximo cae donde debe. Sirve para decidir a cual hacer
     caso y a cual no.
  2. QUE FRAMES son recuperables. Si la mayoria de las curvas de un frame apuntan
     al sitio correcto, ese frame es ajustable; si se contradicen, no.
  3. POR QUE falla uno concreto. Las figuras pintan cada curva en verde o rojo
     segun acierte, con su desplazamiento anotado.

    python training/diagnose_lines_fit.py
    python training/diagnose_lines_fit.py --n 40 --dataset hockeyrink_nhl
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import fit_homography_lines as F  # noqa: E402
import ice_lines_probe as P  # noqa: E402
import make_line_masks as MM  # noqa: E402
import rink  # noqa: E402
from relabel_reproject import read_label  # noqa: E402

ACIERTO_PX = 8.0        # |desplazamiento| por debajo del cual la curva "acierta"
                        # (era 4.0: demasiado estricto, marcaba como fallo curvas
                        #  que a ojo estan perfectamente encima de su linea)
RANGO_PX = 40.0         # hasta donde se desliza la curva buscando su maximo


def slide(R, base, normal, w, h, rango=RANGO_PX, paso=1.0):
    """Desliza la curva por su normal y devuelve (desplazamientos, respuesta)."""
    ds = np.arange(-rango, rango + paso, paso)
    vals = []
    for d in ds:
        q = base + normal * d
        inb = (q[:, 0] > 1) & (q[:, 0] < w - 2) & (q[:, 1] > 1) & (q[:, 1] < h - 2)
        if inb.sum() < 40:
            vals.append(np.nan)
            continue
        qq = q[inb]
        vals.append(float(R[qq[:, 1].astype(int), qq[:, 0].astype(int)].mean()))
    return ds, np.array(vals)


def curve_report(img, region, H, params_rink, polys):
    """Por curva en cuadro: (clase, desplazamiento del maximo, margen, puntos)."""
    h, w = img.shape[:2]
    rojo, azul, _ = F.responses(img, region)
    sig = max(1e-6, w / 1920.0)
    out = []
    for cls, world in polys:
        R = rojo if cls in F.ROJAS else (azul if cls in F.AZULES else None)
        if R is None:
            continue
        xy, ok = MM.project_points(H, world, w, h)
        if ok.sum() < 60:
            continue
        base = xy[ok]
        inb = (base[:, 0] > 1) & (base[:, 0] < w - 2) & (base[:, 1] > 1) & (base[:, 1] < h - 2)
        if inb.sum() < 60:
            continue
        base = base[inb]
        nz = P.normals(base)
        ds, vals = slide(R, base, nz, w, h, rango=RANGO_PX * sig, paso=max(1.0, sig))
        fin = np.isfinite(vals)
        if fin.sum() < 20:
            continue
        ds, vals = ds[fin], vals[fin]
        i = int(np.argmax(vals))
        d_best = float(ds[i] / sig)                    # normalizado a 1920
        lejos = np.abs(ds) > 12 * sig
        margen = float(vals[i] - np.median(vals[lejos])) if lejos.any() else 0.0
        ruido = float(np.std(vals[lejos])) if lejos.sum() > 3 else 1.0
        out.append({"cls": cls, "d": d_best, "z": margen / (ruido or 1e-6),
                    "n": int(len(base)), "base": base, "nz": nz})
    return out


def dibuja(img, rep, err_txt):
    """Pinta la posicion REAL de cada curva y donde cree ella que esta.

    Antes solo se pintaba la posicion real coloreada de rojo cuando fallaba, y eso
    confunde: la curva se ve perfectamente encima de su linea y aun asi sale roja,
    porque lo que falla es el maximo de su respuesta, que no se dibujaba. Ahora la
    continua es la verdad y la discontinua donde pica la respuesta; si fallan,
    se ve la separacion.
    """
    vis = img.copy()
    sig = img.shape[1] / 1920.0
    for r in rep:
        acierta = abs(r["d"]) <= ACIERTO_PX
        color = (60, 220, 60) if acierta else (60, 160, 255)
        cv2.polylines(vis, [r["base"].astype(np.int32)], False, color, 3, cv2.LINE_AA)
        if not acierta:
            desp = (r["base"] + r["nz"] * r["d"] * sig).astype(np.int32)
            for k in range(0, len(desp) - 6, 12):      # discontinua
                cv2.polylines(vis, [desp[k:k + 6]], False, (60, 60, 255), 3, cv2.LINE_AA)
        i = len(r["base"]) // 2
        x, y = r["base"][i].astype(int)
        txt = "{} {:+.0f}px".format(MM.CLASSES[r["cls"]][:12], r["d"])
        cv2.putText(vis, txt, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(vis, err_txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def run(args):
    params_rink = rink.RINK_NHL if "nhl" in args.dataset else rink.RINK_IIHF
    tpl = rink.build_template(params_rink)
    polys = MM.rink_polylines(params_rink)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    por_clase = defaultdict(lambda: {"n": 0, "ok": 0, "d": [], "z": []})
    por_frame = []
    lbls = sorted((ROOT / "datasets" / args.dataset / "labels" / "train").glob("*.txt"))
    paso = max(1, len(lbls) // (args.n * 2))
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
        fq, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
        if fq is None:
            continue
        region = P.best_region(img)
        if region.mean() < 0.15:
            continue
        Hgt = fq[0]
        rep = curve_report(img, region, Hgt, params_rink, polys)
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
                          "frac": aciertos / len(rep),
                          "region": float(region.mean())})

    print("\n=== {}  ({} frames) ===".format(args.dataset, hechos))
    print("\n  QUE CLASES son de fiar (desplazamiento del maximo de cada curva):")
    print("  clase                veces  acierta  |d| p50   margen z")
    filas = [(v["ok"] / v["n"], k, v) for k, v in por_clase.items() if v["n"] >= 3]
    for frac, k, v in sorted(filas, reverse=True):
        print("  {:2d} {:18s} {:4d}   {:4.0f} %   {:6.1f}px   {:6.1f}".format(
            k, MM.CLASSES[k], v["n"], 100 * frac, np.median(v["d"]), np.median(v["z"])))

    fr = np.array([f["frac"] for f in por_frame])
    print("\n  QUE FRAMES son recuperables (fraccion de curvas que aciertan):")
    print("    todas aciertan        : {:4.0f} %".format(100 * np.mean(fr >= 0.999)))
    print("    mayoria (>=2/3)       : {:4.0f} %".format(100 * np.mean(fr >= 0.667)))
    print("    empate o peor (<1/2)  : {:4.0f} %".format(100 * np.mean(fr < 0.5)))
    print("    ninguna acierta       : {:4.0f} %".format(100 * np.mean(fr <= 0.001)))
    print("    curvas en cuadro: mediana {:.0f}".format(np.median([f["n"] for f in por_frame])))

    por_frame.sort(key=lambda f: f["frac"])
    peores = por_frame[:args.figuras]
    mejores = por_frame[-args.figuras:]
    for etiqueta, grupo in (("MAL", peores), ("BIEN", mejores)):
        for f in grupo:
            img = cv2.imread(str(f["img"]))
            txt = "{}  {}  curvas que aciertan: {}/{}  region {:.0f}% del frame".format(
                etiqueta, f["stem"][:20], f["ok"], f["n"], 100 * f["region"])
            cv2.imwrite(str(out_dir / "{}_{}_{}.jpg".format(args.dataset, etiqueta.lower(), f["stem"])),
                        dibuja(img, f["rep"], txt))
    print("\n  figuras (verde = la curva acierta, rojo = apunta a otro sitio) en {}".format(out_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hockeyrink")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--figuras", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "scratch_frames" / "diag_lineas"))
    run(ap.parse_args())
