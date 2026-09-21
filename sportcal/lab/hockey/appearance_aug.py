"""Aumentacion de apariencia para segmentacion de lineas (estudio: Rink-Agnostic,
docs/hockey_research.md seccion C). Dos piezas, ambas sobre la IMAGEN sin tocar
las etiquetas de linea:

- copy-paste de jugadores REALES (cajas de datasets/hockeyai, clases goalie /
  player / referee) sobre el frame: simula oclusion de las lineas. Las etiquetas
  de linea se dejan como estan (la mascara es geometria reproyectada, no
  visibilidad) -- asi la red aprende a inferir la linea por debajo de un jugador,
  que es justo lo que pasa en video real.
- logo augmentation: rectangulos / elipses / texto de colores aleatorios sobre
  pixeles de FONDO (mascara==0) -- ensena a ignorar publicidad pintada en el hielo,
  la fuente tipica de falsos positivos en NHL. No se pinta encima de pixeles de
  linea (asi no se destruye la senal que se quiere segmentar).

    from sportcal.lab.hockey.appearance_aug import AppearanceAug
    aug = AppearanceAug(); img = aug(img, msk)      # img BGR uint8 (h,w,3), msk uint8 (h,w)
"""
import random
from pathlib import Path

import cv2
import numpy as np

from sportcal.paths import ROOT
PLAYER_CLASSES = {3, 4, 6}          # goalie, player, referee (hockeyai_det.yaml)
IGNORE = 255


class AppearanceAug:
    def __init__(self, p_players=0.6, max_players=6, p_logos=0.5, max_logos=5, max_crops=1500, seed=0):
        self.p_players, self.max_players = p_players, max_players
        self.p_logos, self.max_logos = p_logos, max_logos
        self.rng = random.Random(seed)
        self.crops = []
        base = ROOT / "datasets" / "hockeyai"
        lbls = sorted((base / "labels" / "train").glob("*.txt"))
        self.rng.shuffle(lbls)
        for lp in lbls:
            if len(self.crops) >= max_crops:
                break
            ip = base / "images" / "train" / (lp.stem + ".jpg")
            if not ip.exists():
                continue
            rows = [r.split() for r in lp.read_text().splitlines() if r.strip()]
            rows = [r for r in rows if int(r[0]) in PLAYER_CLASSES]
            if not rows:
                continue
            r = self.rng.choice(rows)
            self.crops.append((ip, [float(v) for v in r[1:5]]))
        self._cache = {}
        self._seeded = False

    def _reseed(self):
        # cada worker de DataLoader recibe una copia con el MISMO estado del rng:
        # sin re-sembrar, los 4 workers pegarian exactamente los mismos jugadores
        import os, time
        import torch
        wi = torch.utils.data.get_worker_info()
        self.rng.seed((wi.id if wi else 0) * 1000003 + os.getpid() + int(time.time() * 1000) % 100000)
        self._seeded = True

    def _crop(self, k):
        if k in self._cache:
            return self._cache[k]
        ip, (cx, cy, bw, bh) = self.crops[k]
        im = cv2.imread(str(ip))
        if im is None:
            return None
        h, w = im.shape[:2]
        x0, x1 = int(max(0, (cx - bw / 2) * w)), int(min(w, (cx + bw / 2) * w))
        y0, y1 = int(max(0, (cy - bh / 2) * h)), int(min(h, (cy + bh / 2) * h))
        if x1 - x0 < 8 or y1 - y0 < 16:
            return None
        c = im[y0:y1, x0:x1].copy()
        if len(self._cache) < 400:
            self._cache[k] = c
        return c

    def paste_players(self, img):
        if not self.crops:
            return img
        h, w = img.shape[:2]
        out = img.copy()
        for _ in range(self.rng.randint(1, self.max_players)):
            c = self._crop(self.rng.randrange(len(self.crops)))
            if c is None:
                continue
            # tamano en imagen: un jugador cercano ocupa ~15-35% del alto, uno lejano ~5-12%
            th = int(h * self.rng.uniform(0.06, 0.35))
            tw = max(6, int(c.shape[1] * th / c.shape[0]))
            c = cv2.resize(c, (tw, th), interpolation=cv2.INTER_AREA)
            if self.rng.random() < 0.5:
                c = c[:, ::-1]
            x = self.rng.randint(0, max(0, w - tw))
            y = self.rng.randint(int(h * 0.25), max(int(h * 0.25), h - th))
            # borde difuminado: sin esto el rectangulo del crop se convierte en una
            # pista trivial ("recuadro con bordes duros = jugador") que la red aprende
            a = np.ones((th, tw), np.float32)
            f = max(2, min(th, tw) // 8)
            a[:f, :] *= np.linspace(0, 1, f)[:, None]
            a[-f:, :] *= np.linspace(1, 0, f)[:, None]
            a[:, :f] *= np.linspace(0, 1, f)[None, :]
            a[:, -f:] *= np.linspace(1, 0, f)[None, :]
            roi = out[y:y + th, x:x + tw].astype(np.float32)
            out[y:y + th, x:x + tw] = (roi * (1 - a[..., None]) + c.astype(np.float32) * a[..., None]).astype(np.uint8)
        return out

    def paint_logos(self, img, msk):
        h, w = img.shape[:2]
        out = img.copy()
        bg = (msk == 0)
        layer = np.zeros_like(out)
        drawn = np.zeros((h, w), np.uint8)
        for _ in range(self.rng.randint(1, self.max_logos)):
            col = tuple(int(v) for v in np.random.randint(0, 256, 3))
            kind = self.rng.choice(("rect", "ell", "text"))
            x, y = self.rng.randint(0, w - 1), self.rng.randint(int(h * 0.2), h - 1)
            sw, sh = self.rng.randint(w // 30, w // 5), self.rng.randint(h // 40, h // 6)
            if kind == "rect":
                cv2.rectangle(layer, (x, y), (x + sw, y + sh), col, -1)
                cv2.rectangle(drawn, (x, y), (x + sw, y + sh), 1, -1)
            elif kind == "ell":
                ang = self.rng.randint(0, 180)          # el mismo angulo en capa y mascara
                cv2.ellipse(layer, (x, y), (sw // 2, sh // 2), ang, 0, 360, col, -1)
                cv2.ellipse(drawn, (x, y), (sw // 2, sh // 2), ang, 0, 360, 1, -1)
            else:
                txt = "".join(self.rng.choice("ABCDEFGHIJKLMNOPRSTUWXYZ0123456789") for _ in range(self.rng.randint(3, 8)))
                sc, th = self.rng.uniform(0.8, 3.0), self.rng.randint(2, 6)
                cv2.putText(layer, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
                cv2.putText(drawn, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, 1, th, cv2.LINE_AA)
        alpha = self.rng.uniform(0.35, 0.9)
        sel = (drawn > 0) & bg
        out[sel] = (out[sel].astype(np.float32) * (1 - alpha) + layer[sel].astype(np.float32) * alpha).astype(np.uint8)
        return out

    def __call__(self, img, msk):
        if not self._seeded:
            self._reseed()
        if self.rng.random() < self.p_players:
            img = self.paste_players(img)
        if self.rng.random() < self.p_logos:
            img = self.paint_logos(img, msk)
        return img
