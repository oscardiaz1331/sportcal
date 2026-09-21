"""Explorador interactivo del mismo procesado de line_explorer.py, pero sobre un
video crudo en vez de frames etiquetados: sin ground truth no hay homografia, asi
que solo se ven las etapas que no la necesitan (segmentacion de hielo, region,
croma local, cresta). Sirve para mirar de cerca como se comporta esa senal a lo
largo de MUCHOS frames seguidos del mismo video, con un slider para moverse.

    python -m sportcal.lab.hockey.archive.video_line_explorer nhl3.mp4
    python -m sportcal.lab.hockey.archive.video_line_explorer nhl3.mp4 --start 4000
    python -m sportcal.lab.hockey.archive.video_line_explorer clip2.mp4 --step 5 --cache-size 80

Teclas (ademas de arrastrar el slider de abajo):

    izquierda / derecha   -step / +step frames (--step, por defecto 1)
    av pag / re pag       -200 / +200 frames
    arriba / abajo o 1-9  etapa del procesado
    q                     salir

El GMM (etapa 3) tarda ~0.3-1 s por frame; las demas son casi instantaneas. Los
frames visitados se cachean (--cache-size) para que volver atras no recalcule.
"""
import os

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from sportcal.paths import ROOT

from sportcal.lab.hockey import ice_lines_probe as P

ETAPAS = [
    "1 original",
    "2 hielo: umbral HSV fijo (S<60 V>180)",
    "3 hielo: GMM en Lab",
    "4 region de pista (elegida por solidez)",
    "5 croma local +a  (lineas rojas)",
    "6 croma local -b  (lineas azules)",
    "7 +a recortado a la region  <- lo que ve el ajuste",
    "8 cresta multiescala sobre +a  (lineas rojas)",
    "9 cresta multiescala sobre -b  (lineas azules)",
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
    """Calcula cada etapa solo la primera vez que se pide (la GMM es cara)."""

    def __init__(self, img):
        self.img = img
        self._hsv = self._gmm = self._region = None
        self._chroma = None
        self._cresta = None

    @property
    def hsv(self):
        if self._hsv is None:
            self._hsv = P.ice_hsv(self.img)
        return self._hsv

    @property
    def gmm(self):
        if self._gmm is None:
            self._gmm = P.gmm_surface(self.img)[0]
        return self._gmm

    @property
    def region(self):
        if self._region is None:
            rh, rg = P.play_region(self.hsv), P.play_region(self.gmm)
            self._region = rg if P.solidity(rg) >= P.solidity(rh) else rh
        return self._region

    @property
    def chroma(self):
        if self._chroma is None:
            w = self.img.shape[1]
            self._chroma = P.local_chroma(self.img, win=int(61 * w / 1920) | 1)
        return self._chroma

    @property
    def cresta(self):
        if self._cresta is None:
            da, db = self.chroma
            reg = self.region > 0
            self._cresta = (P.ridge(da * reg), P.ridge(-db * reg))
        return self._cresta

    def etapa(self, i):
        if i == 0:
            return to_rgb(self.img)
        if i == 1:
            return to_rgb(tinte(self.img, self.hsv, (0, 180, 255)))
        if i == 2:
            return to_rgb(tinte(self.img, self.gmm, (0, 255, 120)))
        if i == 3:
            return to_rgb(tinte(self.img, self.region, (0, 255, 120)))
        da, db = self.chroma
        if i == 4:
            return norm_gray(da)
        if i == 5:
            return norm_gray(-db)
        if i == 6:
            return norm_gray(da * (self.region > 0))
        ca, cb = self.cresta
        if i == 7:
            return norm_gray(ca)
        return norm_gray(cb)


class VideoCache:
    """Lee frames por indice (seek + read) y cachea los Frame ya procesados."""

    def __init__(self, path, maxlen=48):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit("no se puede abrir el video: {}".format(path))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.maxlen = maxlen
        self.cache = OrderedDict()

    def get(self, idx):
        idx = max(0, min(self.n - 1, idx))
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return idx, self.cache[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = self.cap.read()
        if not ok:
            return idx, None
        f = Frame(img)
        self.cache[idx] = f
        if len(self.cache) > self.maxlen:
            self.cache.popitem(last=False)
        return idx, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="nhl3.mp4",
                    help="video a explorar, relativo a la raiz del repo si no es ruta absoluta")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1, help="frames que avanzan las flechas <- ->")
    ap.add_argument("--cache-size", type=int, default=48)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute() and not video_path.exists():
        video_path = ROOT / args.video

    vc = VideoCache(video_path, args.cache_size)
    print("{}: {} frames a {:.1f} fps".format(video_path.name, vc.n, vc.fps))

    estado = {"i": max(0, min(vc.n - 1, args.start)), "e": 0}

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0.03, 0.10, 0.94, 0.85])
    ax_slider = fig.add_axes([0.12, 0.02, 0.76, 0.03])
    slider = Slider(ax_slider, "frame", 0, max(1, vc.n - 1),
                     valinit=estado["i"], valstep=1)

    def pinta():
        idx, f = vc.get(estado["i"])
        estado["i"] = idx
        if f is None:
            ax.set_title("no se pudo leer el frame {}".format(idx))
            fig.canvas.draw_idle()
            return
        dat = f.etapa(estado["e"])
        ax.clear()
        ax.imshow(dat, cmap=None if dat.ndim == 3 else "inferno")
        ax.set_axis_off()
        ax.set_title("frame {}/{}  t={:.1f}s   |   {}".format(
            idx, vc.n - 1, idx / vc.fps, ETAPAS[estado["e"]]), fontsize=11)
        if abs(slider.val - idx) >= 1:
            slider.eventson = False
            slider.set_val(idx)
            slider.eventson = True
        fig.canvas.draw_idle()

    def on_slider(val):
        idx = int(round(val))
        if idx == estado["i"]:
            return
        estado["i"] = idx
        pinta()

    def on_key(ev):
        if ev.key == "right":
            estado["i"] += args.step
        elif ev.key == "left":
            estado["i"] -= args.step
        elif ev.key == "pageup":
            estado["i"] += 200
        elif ev.key == "pagedown":
            estado["i"] -= 200
        elif ev.key in ("up", "down"):
            estado["e"] = (estado["e"] + (1 if ev.key == "up" else -1)) % len(ETAPAS)
        elif ev.key and ev.key.isdigit() and ev.key != "0":
            k = int(ev.key)
            if 1 <= k <= len(ETAPAS):
                estado["e"] = k - 1
        elif ev.key == "q":
            plt.close(fig)
            return
        else:
            return
        estado["i"] = max(0, min(vc.n - 1, estado["i"]))
        pinta()

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("key_press_event", on_key)
    pinta()
    print("\nteclas:  <- ->  +/-{} frames   |   av/re pag  +/-200   |   arriba/abajo o 1-9  etapa"
          "   |   q  salir".format(args.step))
    plt.show()


if __name__ == "__main__":
    main()
