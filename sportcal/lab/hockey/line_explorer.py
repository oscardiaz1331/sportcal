"""Explorador interactivo del procesado clasico de lineas, paso a paso.

Sirve para mirar cada etapa de la tuberia sobre frames reales y ver donde se
rompe. Los frames vienen ordenados de PEOR a mejor (por fraccion de curvas que
localizan bien su linea), asi que lo primero que ves son los casos problematicos.

    python -m sportcal.lab.hockey.line_explorer                     # IIHF, el peor dataset
    python -m sportcal.lab.hockey.line_explorer --dataset hockeyrink_nhl --video nhl4
    python -m sportcal.lab.hockey.line_explorer --n 40 --solo-con-circulo

Teclas:

    izquierda / derecha   frame anterior / siguiente
    arriba / abajo o 1-9  etapa del procesado
    c                     siguiente curva (cambia el perfil de abajo)
    r                     recalcular el frame actual (si tocas el codigo)
    q                     salir

El panel de abajo es el mas util para entender los fallos: muestra la respuesta
cromatica integrada segun se desliza la curva por su normal. Si el maximo no esta
en 0, esa curva esta tirando de la homografia hacia otro sitio, y ahi se ve a
cuanta distancia y con cuanto margen sobre el resto.
"""
import os

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from sportcal.paths import ROOT

from sportcal.lab.hockey import diagnose_lines_fit as D
from sportcal.lab.hockey import fit_homography_lines as F
from sportcal.lab.hockey import ice_lines_probe as P
from sportcal.lab.hockey import make_line_masks as MM
from sportcal.sports.hockey import rink
from sportcal.lab.hockey.relabel_reproject import read_label  # noqa: E402

ETAPAS = [
    "1 original",
    "2 hielo: umbral HSV fijo (S<60 V>180)",
    "3 hielo: GMM en Lab",
    "4 region de pista (elegida por solidez) + contorno real",
    "5 croma local +a  (lineas rojas)",
    "6 croma local -b  (lineas azules)",
    "7 +a recortado a la region  <- lo que ve el ajuste",
    "8 cresta multiescala sobre +a  (lineas rojas)",
    "9 cresta multiescala sobre -b  (lineas azules)",
    "a diagnostico por curva (continua = verdad, discontinua = donde pica)",
    "b zocalo de las vallas: +b  (la senal mas fuerte del frame)",
]


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def norm_gray(x, lo=1, hi=99.5):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / max(1e-6, b - a), 0, 1)


def tinte(base, mask, color, alpha=0.45):
    over = base.copy()
    over[mask > 0] = color
    return cv2.addWeighted(over, alpha, base, 1 - alpha, 0)


class Frame:
    """Calcula y guarda todas las etapas de un frame."""

    def __init__(self, img_path, lbl_path, params_rink, tpl, polys):
        self.path = img_path
        self.img = cv2.imread(str(img_path))
        self.params = params_rink
        self.polys = polys
        h, w = self.img.shape[:2]
        rec = read_label(lbl_path)
        fq, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
        self.H = fq[0] if fq else None

        self.ice_hsv = P.ice_hsv(self.img)
        self.ice_gmm = P.gmm_surface(self.img)[0]
        rh, rg = P.play_region(self.ice_hsv), P.play_region(self.ice_gmm)
        self.elegido = "GMM" if P.solidity(rg) >= P.solidity(rh) else "HSV"
        self.region = rg if self.elegido == "GMM" else rh
        self.da, self.db = P.local_chroma(self.img, win=int(61 * w / 1920) | 1)
        self.cresta_a = P.ridge(self.da * (self.region > 0))
        self.cresta_b = P.ridge(-self.db * (self.region > 0))
        self.verdad = P.true_region(self.H, params_rink, w, h) if self.H is not None else None
        self.rep = (D.curve_report(self.img, self.region, self.H, params_rink, polys)
                    if self.H is not None else [])
        ok = sum(1 for r in self.rep if abs(r["d"]) <= D.ACIERTO_PX)
        self.frac = ok / len(self.rep) if self.rep else 0.0
        self.resumen = "curvas que aciertan {}/{}   region {} ({:.0f} % del frame)".format(
            ok, len(self.rep), self.elegido, 100 * self.region.mean())

    def etapa(self, i):
        img = self.img
        if i == 0:
            return to_rgb(img)
        if i == 1:
            return to_rgb(tinte(img, self.ice_hsv, (0, 180, 255)))
        if i == 2:
            return to_rgb(tinte(img, self.ice_gmm, (0, 255, 120)))
        if i == 3:
            vis = tinte(img, self.region, (0, 255, 120))
            if self.verdad is not None:
                cnt, _ = cv2.findContours(self.verdad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnt, -1, (255, 255, 255), 3)
            return to_rgb(vis)
        if i == 4:
            return norm_gray(self.da)
        if i == 5:
            return norm_gray(-self.db)
        if i == 6:
            return norm_gray(self.da * (self.region > 0))
        if i == 7:
            return norm_gray(self.cresta_a)
        if i == 8:
            return norm_gray(self.cresta_b)
        if i == 9:
            return to_rgb(D.dibuja(img, self.rep, ""))
        # zocalo: +b sin recortar a la region, con el contorno real encima
        vis = norm_gray(self.db)
        vis = cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if self.H is not None:
            base, nz = F.outline_normals(self.H, self.params, img.shape[1], img.shape[0])
            if base is not None:
                sig = img.shape[1] / 1920.0
                cv2.polylines(vis, [base.astype(np.int32)], True, (60, 220, 60), 2, cv2.LINE_AA)
                q = base + nz * F.OFFSET_ZOCALO * sig
                cv2.polylines(vis, [q.astype(np.int32)], True, (60, 160, 255), 2, cv2.LINE_AA)
        return to_rgb(vis)

    def perfil(self, k):
        """(desplazamientos, respuesta) de la curva k deslizada por su normal."""
        if not self.rep:
            return None, None, None
        r = self.rep[k % len(self.rep)]
        h, w = self.img.shape[:2]
        rojo, azul, _ = F.responses(self.img, self.region)
        R = rojo if r["cls"] in F.ROJAS else azul
        sig = max(1e-6, w / 1920.0)
        ds, vals = D.slide(R, r["base"], r["nz"], w, h, rango=D.RANGO_PX * sig, paso=max(1.0, sig))
        return ds / sig, vals, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hockeyrink")
    ap.add_argument("--video", default=None, help="prefijo del frame, p.ej. nhl4")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--solo-con-circulo", action="store_true",
                    help="solo frames donde el circulo central esta en cuadro")
    args = ap.parse_args()

    params_rink = rink.RINK_NHL if "nhl" in args.dataset else rink.RINK_IIHF
    tpl = rink.build_template(params_rink)
    polys = MM.rink_polylines(params_rink)

    base = ROOT / "datasets" / args.dataset
    lbls = sorted((base / "labels" / "train").glob("*.txt"))
    if args.video:
        lbls = [l for l in lbls if l.stem.startswith(args.video)]
    if not lbls:
        raise SystemExit("sin frames para ese filtro")
    paso = max(1, len(lbls) // (args.n * 2))

    print("preparando frames (el GMM tarda ~1 s por frame)...")
    frames = []
    for lbl in lbls[::paso]:
        if len(frames) >= args.n:
            break
        ip = base / "images" / "train" / (lbl.stem + ".jpg")
        if not ip.exists():
            continue
        try:
            f = Frame(ip, lbl, params_rink, tpl, polys)
        except Exception as exc:
            print("  {} : {}".format(lbl.stem, exc))
            continue
        if f.H is None or not f.rep:
            continue
        if args.solo_con_circulo and not any(r["cls"] == 7 for r in f.rep):
            continue
        frames.append(f)
        print("  {:3d}/{}  {}  {}".format(len(frames), args.n, lbl.stem[:24], f.resumen))
    if not frames:
        raise SystemExit("ningun frame utilizable")

    frames.sort(key=lambda f: f.frac)          # peores primero
    estado = {"i": 0, "e": 0, "c": 0}

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.22)
    ax = fig.add_subplot(gs[0])
    axp = fig.add_subplot(gs[1])

    def pinta():
        f = frames[estado["i"]]
        ax.clear()
        dat = f.etapa(estado["e"])
        ax.imshow(dat, cmap=None if dat.ndim == 3 else "inferno")
        ax.set_axis_off()
        ax.set_title("[{}/{}] {}   |   {}\n{}".format(
            estado["i"] + 1, len(frames), f.path.stem[:28],
            ETAPAS[estado["e"]], f.resumen), fontsize=10)

        axp.clear()
        ds, vals, r = f.perfil(estado["c"])
        if ds is not None:
            axp.plot(ds, vals, lw=1.6)
            i = int(np.nanargmax(vals))
            axp.axvline(0, color="g", ls="--", lw=1.2, label="posicion correcta")
            axp.axvline(ds[i], color="r", ls="-", lw=1.2,
                        label="maximo ({:+.0f} px)".format(ds[i]))
            axp.set_title("curva {}/{}: {}   desplazamiento del maximo {:+.0f} px   margen z {:.1f}".format(
                estado["c"] % len(f.rep) + 1, len(f.rep),
                MM.CLASSES[r["cls"]], r["d"], r["z"]), fontsize=9)
            axp.set_xlabel("desplazamiento por la normal (px a 1920)")
            axp.legend(fontsize=8, loc="upper right")
            axp.grid(alpha=0.3)
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key == "right":
            estado["i"] = (estado["i"] + 1) % len(frames); estado["c"] = 0
        elif ev.key == "left":
            estado["i"] = (estado["i"] - 1) % len(frames); estado["c"] = 0
        elif ev.key in ("up", "down"):
            estado["e"] = (estado["e"] + (1 if ev.key == "up" else -1)) % len(ETAPAS)
        elif ev.key and ev.key.isdigit() and ev.key != "0":
            k = int(ev.key)
            if 1 <= k <= len(ETAPAS):
                estado["e"] = k - 1
        elif ev.key == "c":
            estado["c"] += 1
        elif ev.key == "r":
            f = frames[estado["i"]]
            frames[estado["i"]] = Frame(f.path, (base / "labels" / "train" /
                                                 (f.path.stem + ".txt")),
                                        params_rink, tpl, polys)
        elif ev.key == "q":
            plt.close(fig); return
        else:
            return
        pinta()

    fig.canvas.mpl_connect("key_press_event", on_key)
    pinta()
    print("\nteclas:  <- ->  frame   |   arriba/abajo o 1-9  etapa   |   c  curva   |   q  salir")
    plt.show()


if __name__ == "__main__":
    main()
