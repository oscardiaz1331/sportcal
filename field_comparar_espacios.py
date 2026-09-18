"""Comparador de espacios de color en 3 ventanas separadas (antes era una
sola figura muy apretada).

  Ventana 1 "principal": frame original con el hielo (S<=s_ice_max &
    V>=v_ice_min de HSV) teñido de blanco, sliders de frame/hielo, e
    inspector de clic (imprime H/S/V, L/a/b, Y/Cr/Cb, L/u/v del pixel).
    Es el hub de control: los sliders y el clic viven aqui.

  Ventana 2 "color real": la misma imagen, pero cada pixel que NO es hielo
    se pinta con su hue real a saturacion y valor maximos (S=255,V=255) -
    para ver de un vistazo el color "puro" de cada zona sin que el brillo o
    lo desaturado que salga por compresion lo disimule. Los pixeles que SI
    son hielo se fuerzan a blanco puro.

  Ventana 3 "espacios de color": los 4 mapas de densidad (HSV, Lab, YCbCr,
    Luv) de los pixeles sin hielo + el espectro de Hue coloreado, todo con
    mas sitio para respirar.

Las 3 ventanas comparten el mismo estado (frame, umbrales de hielo, pixel
clicado) y se redibujan juntas.

No calibra mascaras de rojo/azul (para eso estan field_hsv_zones.py,
field_lab_zones.py, field_ycbcr_zones.py y field_luv_zones.py).

Uso:
  python field_comparar_espacios.py --video clip2.mp4
"""
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# esquinas de referencia por espacio, calculadas con cv2.cvtColor sobre
# primarios puros sRGB (8-bit). HSV incluye "rojo" en H=0 Y H=179 porque el
# rojo envuelve en los dos extremos de la rueda de hue.
CORNERS = {
    "hsv": {
        "rojo (H=0)": (0, 255),
        "rojo (H=179)": (179, 255),
        "verde": (60, 255),
        "azul": (120, 255),
    },
    "lab": {
        "rojo": (208, 195),
        "verde": (42, 211),
        "azul": (207, 20),
    },
    "ycc": {
        "rojo": (255, 85),
        "verde": (21, 43),
        "azul": (107, 255),
    },
    "luv": {
        "rojo": (222, 173),
        "verde": (37, 241),
        "azul": (90, 10),
    },
}
NEUTRO = {
    "lab": (128, 128),
    "ycc": (128, 128),
    "luv": (96, 136),
}
COLOR_PUNTO = {"rojo": "red", "verde": "green", "azul": "blue"}

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="clip2.mp4")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise IOError(f"no se pudo abrir el video {args.video}")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

estado = {
    "frame": 0,
    "s_ice_max": 50,
    "v_ice_min": 140,
    "click_xy": None,   # (x, y) en pixeles del frame, o None si no se ha clicado
}

_cache = {"idx": None, "bgr": None}


def get_frame_bgr(idx):
    if _cache["idx"] == idx:
        return _cache["bgr"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    _cache["idx"], _cache["bgr"] = idx, frame
    return frame


def valores_pixel(frame, x, y):
    h, s, v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y, x]
    l_lab, a, b = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)[y, x]
    yy, cr, cb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[y, x]
    l_luv, u, vv = cv2.cvtColor(frame, cv2.COLOR_BGR2Luv)[y, x]
    return {
        "hsv": (int(h), int(s), int(v)),
        "lab": (int(l_lab), int(a), int(b)),
        "ycc": (int(yy), int(cr), int(cb)),
        "luv": (int(l_luv), int(u), int(vv)),
    }


def imagen_color_real(h, hielo):
    """Cada pixel a su hue real con S=255,V=255 (color puro); el hielo, blanco."""
    hsv_tope = np.dstack([h, np.full_like(h, 255), np.full_like(h, 255)])
    rgb = cv2.cvtColor(hsv_tope, cv2.COLOR_HSV2RGB)
    rgb[hielo] = (255, 255, 255)
    return rgb


# tira de 180 colores puros (S=255, V=255), uno por cada H posible, para
# poder "leer" el color real de cada hue de un vistazo.
_TIRA_HUE = cv2.cvtColor(
    np.dstack([np.arange(180, dtype=np.uint8),
               np.full(180, 255, np.uint8),
               np.full(180, 255, np.uint8)]),
    cv2.COLOR_HSV2RGB,
)[0].astype(np.float32) / 255.0


def dibujar_espacio(ax, xx, yy, xlim, ylim, xlabel, ylabel, titulo, corners, neutro, punto):
    ax.clear()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(titulo, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)

    hist, _, _ = np.histogram2d(xx, yy, bins=64, range=[list(xlim), list(ylim)])
    ax.imshow(np.log1p(hist).T, origin="lower",
              extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
              cmap="gray_r", aspect="auto")

    if neutro is not None:
        ax.plot(*neutro, "+", color="black", markersize=10, markeredgewidth=1.5)
        ax.annotate("neutro", neutro, fontsize=8, textcoords="offset points", xytext=(4, -12))
    else:
        ax.axhline(0, color="black", linewidth=1, linestyle=":")
        ax.annotate("neutro (S=0, cualquier H)", (xlim[0], 3), fontsize=7.5)

    for nombre, (cx, cy) in corners.items():
        base = nombre.split(" ")[0]
        color = COLOR_PUNTO.get(base, "black")
        ax.plot(cx, cy, "o", color=color, markersize=6)
        ax.annotate(nombre, (cx, cy), fontsize=7.5, color=color,
                    textcoords="offset points", xytext=(5, 5))

    if punto is not None:
        ax.plot(*punto, "*", color="gold", markersize=18,
                markeredgecolor="black", markeredgewidth=0.9, zorder=5)


def dibujar_espectro_hue(ax, h_no_hielo, h_click):
    ax.clear()
    ax.set_xlim(0, 179)
    ax.set_title("Hue de los pixeles SIN hielo, a S=255,V=255 (color real de cada H)", fontsize=10)
    ax.set_xlabel("H", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)

    conteos = np.bincount(h_no_hielo, minlength=180)[:180].astype(np.float64)
    ax.bar(np.arange(180), conteos, width=1.0, color=_TIRA_HUE, edgecolor="none")
    ax.set_ylim(0, max(conteos.max(), 1) * 1.08)

    if h_click is not None:
        ax.axvline(h_click, color="black", linewidth=1.5)
        ax.plot(h_click, ax.get_ylim()[1] * 0.97, "v", color="black", markersize=9)


# ======================= VENTANA 1: principal =======================
fig1 = plt.figure(figsize=(8, 8))
try:
    fig1.canvas.manager.set_window_title("1) Principal - segmentacion de hielo")
except Exception:
    pass

ax1_img = fig1.add_axes((0.06, 0.34, 0.88, 0.62))
ax1_img.axis("off")
ax1_img.set_title("original (hielo teñido de blanco) - clic para inspeccionar", fontsize=10)
im1 = ax1_img.imshow(np.zeros((2, 2, 3), np.uint8))

ax1_slider_frame = fig1.add_axes((0.10, 0.24, 0.80, 0.03))
slider_frame = Slider(ax1_slider_frame, "frame", 0, max(n_frames - 1, 1), valinit=0, valstep=1)
ax1_slider_s = fig1.add_axes((0.10, 0.19, 0.80, 0.03))
slider_s = Slider(ax1_slider_s, "hielo S<=", 0, 255, valinit=estado["s_ice_max"], valstep=1)
ax1_slider_v = fig1.add_axes((0.10, 0.14, 0.80, 0.03))
slider_v = Slider(ax1_slider_v, "hielo V>=", 0, 255, valinit=estado["v_ice_min"], valstep=1)

info_text = fig1.text(0.06, 0.09, "clic sobre la imagen para inspeccionar un pixel",
                       fontsize=9, family="monospace", va="top")

# ======================= VENTANA 2: color real =======================
fig2 = plt.figure(figsize=(9, 6))
try:
    fig2.canvas.manager.set_window_title("2) Color real (S,V a tope; hielo = blanco)")
except Exception:
    pass
ax2_img = fig2.add_axes((0.03, 0.03, 0.94, 0.90))
ax2_img.axis("off")
im2 = ax2_img.imshow(np.zeros((2, 2, 3), np.uint8))

# ======================= VENTANA 3: espacios de color =======================
fig3 = plt.figure(figsize=(13, 9))
try:
    fig3.canvas.manager.set_window_title("3) Espacios de color (HSV / Lab / YCbCr / Luv)")
except Exception:
    pass
ax3_hsv = fig3.add_axes((0.06, 0.55, 0.42, 0.38))
ax3_lab = fig3.add_axes((0.55, 0.55, 0.42, 0.38))
ax3_ycc = fig3.add_axes((0.06, 0.14, 0.42, 0.35))
ax3_luv = fig3.add_axes((0.55, 0.14, 0.42, 0.35))
ax3_hue = fig3.add_axes((0.06, 0.02, 0.91, 0.09))


def redibujar():
    frame = get_frame_bgr(estado["frame"])
    if frame is None:
        return

    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    _, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2Lab))
    _, cr, cb = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb))
    _, u, vv = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2Luv))

    hielo = (s <= estado["s_ice_max"]) & (v >= estado["v_ice_min"])

    # --- ventana 1: original con hielo teñido ---
    rgb1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb1[hielo] = rgb1[hielo] * 0.4 + np.array([255.0, 255.0, 255.0]) * 0.6
    im1.set_data(np.clip(rgb1, 0, 255).astype(np.uint8))
    im1.set_extent((0, frame.shape[1], frame.shape[0], 0))

    # --- ventana 2: color real (S,V a tope; hielo = blanco) ---
    rgb2 = imagen_color_real(h, hielo)
    im2.set_data(rgb2)
    im2.set_extent((0, frame.shape[1], frame.shape[0], 0))

    # --- ventana 3: espacios de color ---
    paso = 4   # submuestreo solo para las nubes de densidad, no para el clic
    no_hielo_sub = ~hielo[::paso, ::paso].ravel()
    hh, ss = h[::paso, ::paso].ravel()[no_hielo_sub], s[::paso, ::paso].ravel()[no_hielo_sub]
    aa, bbf = a[::paso, ::paso].ravel()[no_hielo_sub], b[::paso, ::paso].ravel()[no_hielo_sub]
    crf, cbf = cr[::paso, ::paso].ravel()[no_hielo_sub], cb[::paso, ::paso].ravel()[no_hielo_sub]
    uf, vf = u[::paso, ::paso].ravel()[no_hielo_sub], vv[::paso, ::paso].ravel()[no_hielo_sub]

    punto = None
    vals = None
    if estado["click_xy"] is not None:
        x, y = estado["click_xy"]
        alto, ancho = frame.shape[:2]
        x = min(max(x, 0), ancho - 1)
        y = min(max(y, 0), alto - 1)
        vals = valores_pixel(frame, x, y)

    dibujar_espacio(ax3_hsv, hh, ss, (0, 179), (0, 255), "H", "S", "HSV (H,S) - sin hielo",
                     CORNERS["hsv"], None, vals["hsv"][:2] if vals else None)
    dibujar_espacio(ax3_lab, aa, bbf, (0, 255), (0, 255), "a (verde<->rojo)", "b (azul<->amarillo)", "Lab (a,b) - sin hielo",
                     CORNERS["lab"], NEUTRO["lab"], vals["lab"][1:] if vals else None)
    dibujar_espacio(ax3_ycc, crf, cbf, (0, 255), (0, 255), "Cr (rojez)", "Cb (azulez)", "YCbCr (Cr,Cb) - sin hielo",
                     CORNERS["ycc"], NEUTRO["ycc"], vals["ycc"][1:] if vals else None)
    dibujar_espacio(ax3_luv, uf, vf, (0, 255), (0, 255), "u (verde<->rojo)", "v (azul<->amarillo)", "Luv (u,v) - sin hielo",
                     CORNERS["luv"], NEUTRO["luv"], vals["luv"][1:] if vals else None)
    dibujar_espectro_hue(ax3_hue, h[~hielo], vals["hsv"][0] if vals else None)

    total = hielo.size
    pct_hielo = 100.0 * np.count_nonzero(hielo) / total
    titulo = f'frame {estado["frame"]} / {n_frames - 1}   hielo {pct_hielo:5.1f}%'
    if vals is not None:
        x, y = estado["click_xy"]
        titulo += f'   px=({x},{y})'
    fig1.suptitle(titulo)
    fig2.suptitle(titulo)
    fig3.suptitle(titulo)

    if vals is not None:
        x, y = estado["click_xy"]
        info_text.set_text(
            f"px ({x},{y})\n"
            f"HSV    H={vals['hsv'][0]:3d}  S={vals['hsv'][1]:3d}  V={vals['hsv'][2]:3d}\n"
            f"Lab    L={vals['lab'][0]:3d}  a={vals['lab'][1]:3d}  b={vals['lab'][2]:3d}\n"
            f"YCbCr  Y={vals['ycc'][0]:3d}  Cr={vals['ycc'][1]:3d}  Cb={vals['ycc'][2]:3d}\n"
            f"Luv    L={vals['luv'][0]:3d}  u={vals['luv'][1]:3d}  v={vals['luv'][2]:3d}\n"
            f"es hielo (S<={estado['s_ice_max']} & V>={estado['v_ice_min']}): "
            f"{'si' if bool(hielo[y, x]) else 'no'}"
        )
    else:
        info_text.set_text("clic sobre la imagen para inspeccionar un pixel")

    fig1.canvas.draw_idle()
    fig2.canvas.draw_idle()
    fig3.canvas.draw_idle()


def on_slider_frame(v):
    estado["frame"] = int(v)
    redibujar()


def on_slider_s(v):
    estado["s_ice_max"] = int(v)
    redibujar()


def on_slider_v(v):
    estado["v_ice_min"] = int(v)
    redibujar()


slider_frame.on_changed(on_slider_frame)
slider_s.on_changed(on_slider_s)
slider_v.on_changed(on_slider_v)


def on_key(event):
    if event.key in ("left", "right"):
        paso = -1 if event.key == "left" else 1
        nuevo = min(max(estado["frame"] + paso, 0), n_frames - 1)
        slider_frame.set_val(nuevo)


def on_click(event):
    if event.inaxes is not ax1_img:
        return
    if event.xdata is None or event.ydata is None:
        return
    frame = get_frame_bgr(estado["frame"])
    if frame is None:
        return
    x, y = int(round(event.xdata)), int(round(event.ydata))
    alto, ancho = frame.shape[:2]
    if not (0 <= x < ancho and 0 <= y < alto):
        return
    estado["click_xy"] = (x, y)
    redibujar()


# navegacion por teclado funciona con cualquiera de las 3 ventanas enfocada
for f in (fig1, fig2, fig3):
    f.canvas.mpl_connect("key_press_event", on_key)
fig1.canvas.mpl_connect("button_press_event", on_click)

redibujar()
plt.show()
cap.release()
