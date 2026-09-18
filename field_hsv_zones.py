"""Deteccion de zonas del campo por color HSV, sin tocar test_colors.py.

Idea:
  - Hielo/pista: saturacion baja (S <= s_ice_max) y valor alto (V >= v_ice_min).
    El hue no importa: el hielo es basicamente blanco/gris.
  - Estructuras rojas (linea central, circulos, etc.): hue en torno a 0/179
    (el rojo "envuelve" en el circulo de hue de OpenCV), SIN filtrar por
    saturacion, solo acotando el value para descartar ruido muy oscuro.
  - Estructuras azules (lineas azules): hue en torno a ~110, tambien sin
    filtrar por saturacion, solo con un rango de value.

Resolucion real: no se reduce el frame antes de calcular las mascaras (antes
se hacia un resize a 0.5x con INTER_AREA, que promediaba y diluia el color
de lineas finas). Se trabaja sobre el frame nativo del video.

CLAHE en V (toggle): aplica contraste local adaptativo solo sobre el canal
de luminancia (V de HSV), nunca sobre H ni S, asi que no distorsiona el
color -realza el hielo/sombras/lineas en brillo sin tocar la crominancia
que deciden las mascaras de rojo/azul.

Herramienta interactiva (sliders) para calibrar los umbrales viendo el
resultado en vivo sobre el video.

Uso:
  python field_hsv_zones.py --video clip2.mp4
"""
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="clip2.mp4")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise IOError(f"no se pudo abrir el video {args.video}")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

estado = {
    "frame": 0,
    "s_ice_max": 50,       # hielo: saturacion <= esto
    "v_ice_min": 140,      # hielo: value >= esto
    "red_bajo": 12,         # rojo: H <= esto (banda cerca de 0)
    "red_alto": 167,        # rojo: H >= esto (banda cerca de 179)
    "blue_lo": 95,         # azul: rango de H
    "blue_hi": 130,
    "v_struct_min": 70,    # rojo/azul: value dentro de [min, max]
    "v_struct_max": 255,
    "limpiar": False,      # apertura+cierre morfologico para quitar ruido suelto
    "clahe_v": False,      # contraste adaptativo (CLAHE) solo sobre V, antes de umbralizar
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


_kernel = np.ones((3, 3), np.uint8)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def limpiar_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)
    return mask


def obtener_hsv(bgr):
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    if estado["clahe_v"]:
        v = _clahe.apply(v)
    return h, s, v


def calcular_mascaras(bgr):
    h, s, v = obtener_hsv(bgr)

    hielo = (s <= estado["s_ice_max"]) & (v >= estado["v_ice_min"])
    # opuesto literal a la mascara de hielo (De Morgan): S alta O V baja.
    # Si solo mirasemos S, un rojo oscuro (V baja) mide S baja por ruido del
    # sensor cerca de negro y se colaria como "hielo" sin serlo.
    no_hielo = ~hielo

    rojo = ((h <= estado["red_bajo"]) | (h >= estado["red_alto"])) & \
           (v >= estado["v_struct_min"]) & (v <= estado["v_struct_max"]) & no_hielo

    azul = (h >= estado["blue_lo"]) & (h <= estado["blue_hi"]) & \
           (v >= estado["v_struct_min"]) & (v <= estado["v_struct_max"]) & no_hielo

    hielo = (hielo.astype(np.uint8)) * 255
    rojo = (rojo.astype(np.uint8)) * 255
    azul = (azul.astype(np.uint8)) * 255

    if estado["limpiar"]:
        hielo = limpiar_mask(hielo)
        rojo = limpiar_mask(rojo)
        azul = limpiar_mask(azul)

    return hielo, rojo, azul


def overlay_rgb(bgr, hielo, rojo, azul):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    out = rgb.copy()
    out[hielo > 0] = out[hielo > 0] * 0.5 + np.array([255.0, 255.0, 255.0]) * 0.5
    out[rojo > 0] = (255.0, 0.0, 255.0)   # magenta
    out[azul > 0] = (0.0, 255.0, 255.0)   # cian
    return np.clip(out, 0, 255).astype(np.uint8)


# ======================= FIGURA =======================
fig = plt.figure(figsize=(15, 9))
try:
    fig.canvas.manager.set_window_title("Zonas del campo por HSV (hielo / rojo / azul)")
except Exception:
    pass

ax_orig = fig.add_axes((0.03, 0.68, 0.28, 0.29))
ax_overlay = fig.add_axes((0.36, 0.68, 0.28, 0.29))
ax_hielo = fig.add_axes((0.69, 0.68, 0.28, 0.29))
ax_rojo = fig.add_axes((0.03, 0.37, 0.28, 0.29))
ax_azul = fig.add_axes((0.36, 0.37, 0.28, 0.29))
ax_comb = fig.add_axes((0.69, 0.37, 0.28, 0.29))

for a, titulo in ((ax_orig, "original"), (ax_overlay, "overlay (hielo+rojo+azul)"),
                  (ax_hielo, "mascara hielo"), (ax_rojo, "mascara rojo"),
                  (ax_azul, "mascara azul"), (ax_comb, "rojo+azul (sin hielo)")):
    a.axis("off")
    a.set_title(titulo, fontsize=9)

im_orig = ax_orig.imshow(np.zeros((2, 2, 3), np.uint8))
im_overlay = ax_overlay.imshow(np.zeros((2, 2, 3), np.uint8))
im_hielo = ax_hielo.imshow(np.zeros((2, 2), np.uint8), cmap="gray", vmin=0, vmax=255)
im_rojo = ax_rojo.imshow(np.zeros((2, 2), np.uint8), cmap="gray", vmin=0, vmax=255)
im_azul = ax_azul.imshow(np.zeros((2, 2), np.uint8), cmap="gray", vmin=0, vmax=255)
im_comb = ax_comb.imshow(np.zeros((2, 2, 3), np.uint8))

# --- sliders ---
slider_specs = [
    ("frame", "frame", 0, max(n_frames - 1, 1), 1),
    ("s_ice_max", "hielo S<=", 0, 255, 1),
    ("v_ice_min", "hielo V>=", 0, 255, 1),
    ("red_bajo", "rojo H<= (abajo)", 0, 90, 1),
    ("red_alto", "rojo H>= (arriba)", 90, 179, 1),
    ("blue_lo", "azul H lo", 0, 179, 1),
    ("blue_hi", "azul H hi", 0, 179, 1),
    ("v_struct_min", "rojo/azul V>=", 0, 255, 1),
    ("v_struct_max", "rojo/azul V<=", 0, 255, 1),
]
sliders = {}
y0 = 0.30
for i, (clave, label, vmin, vmax, step) in enumerate(slider_specs):
    ax_s = fig.add_axes((0.08, y0 - i * 0.032, 0.55, 0.02))
    sliders[clave] = Slider(ax_s, label, vmin, vmax, valinit=estado[clave], valstep=step)

ax_check = fig.add_axes((0.72, 0.02, 0.20, 0.09))
CHECKS = [
    ("limpiar (morf.)", "limpiar"),
    ("CLAHE en V", "clahe_v"),
]
check = CheckButtons(ax_check, [lbl for lbl, _ in CHECKS], [estado[k] for _, k in CHECKS])

legend_text = fig.text(0.65, 0.36, "", fontsize=7.5, family="monospace", va="top")
info_text = fig.text(0.65, 0.20, "clic sobre una imagen para inspeccionar H/S/V",
                      fontsize=7.5, family="monospace", va="top")


def _ok(b):
    return "OK" if b else "no"


def texto_condiciones():
    return (
        "condiciones actuales:\n"
        f"hielo: S<={estado['s_ice_max']}  &  V>={estado['v_ice_min']}\n"
        f"rojo : (H<={estado['red_bajo']} o H>={estado['red_alto']})  &  V en [{estado['v_struct_min']},{estado['v_struct_max']}]"
        f"  &  NOT-hielo (S>{estado['s_ice_max']} o V<{estado['v_ice_min']})\n"
        f"azul : H en [{estado['blue_lo']},{estado['blue_hi']}]  &  V en [{estado['v_struct_min']},{estado['v_struct_max']}]"
        f"  &  NOT-hielo (S>{estado['s_ice_max']} o V<{estado['v_ice_min']})\n"
        f"CLAHE en V: {'ON' if estado['clahe_v'] else 'off'}"
    )


def redibujar():
    frame = get_frame_bgr(estado["frame"])
    if frame is None:
        return
    hielo, rojo, azul = calcular_mascaras(frame)
    overlay = overlay_rgb(frame, hielo, rojo, azul)
    combinado = overlay_rgb(frame, np.zeros_like(hielo), rojo, azul)

    for im, img, ax in (
        (im_orig, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ax_orig),
        (im_overlay, overlay, ax_overlay),
        (im_hielo, hielo, ax_hielo),
        (im_rojo, rojo, ax_rojo),
        (im_azul, azul, ax_azul),
        (im_comb, combinado, ax_comb),
    ):
        im.set_data(img)
        im.set_extent((0, img.shape[1], img.shape[0], 0))

    total = hielo.size
    pct_hielo = 100.0 * np.count_nonzero(hielo) / total
    pct_rojo = 100.0 * np.count_nonzero(rojo) / total
    pct_azul = 100.0 * np.count_nonzero(azul) / total
    fig.suptitle(
        f'frame {estado["frame"]} / {n_frames - 1}   '
        f'hielo {pct_hielo:5.1f}%   rojo {pct_rojo:5.1f}%   azul {pct_azul:5.1f}%'
    )
    legend_text.set_text(texto_condiciones())
    fig.canvas.draw_idle()


def on_slider(clave):
    def _cb(v):
        estado[clave] = int(v)
        redibujar()
    return _cb


for clave, slider in sliders.items():
    slider.on_changed(on_slider(clave))


def on_check(label):
    for lbl, clave in CHECKS:
        if lbl == label:
            estado[clave] = not estado[clave]
            break
    redibujar()


check.on_clicked(on_check)


def on_key(event):
    if event.key in ("left", "right"):
        paso = -1 if event.key == "left" else 1
        nuevo = min(max(estado["frame"] + paso, 0), n_frames - 1)
        sliders["frame"].set_val(nuevo)


def on_click(event):
    if event.inaxes not in (ax_orig, ax_overlay, ax_hielo, ax_rojo, ax_azul, ax_comb):
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
    h_full, s_full, v_full = obtener_hsv(frame)
    h, s, v = int(h_full[y, x]), int(s_full[y, x]), int(v_full[y, x])
    hielo_m, rojo_m, azul_m = calcular_mascaras(frame)

    c_hielo_s = s <= estado["s_ice_max"]
    c_hielo_v = v >= estado["v_ice_min"]
    c_no_hielo = not (c_hielo_s and c_hielo_v)
    c_v_rango = estado["v_struct_min"] <= v <= estado["v_struct_max"]
    c_rojo_hue = h <= estado["red_bajo"] or h >= estado["red_alto"]
    c_azul_hue = estado["blue_lo"] <= h <= estado["blue_hi"]

    msg = (
        f"px ({x},{y})  H={h}  S={s}  V={v}\n"
        f"hielo: S<={estado['s_ice_max']} {_ok(c_hielo_s)}  "
        f"V>={estado['v_ice_min']} {_ok(c_hielo_v)}  "
        f"=> {_ok(bool(hielo_m[y, x]))}\n"
        f"rojo : hue {_ok(c_rojo_hue)}  "
        f"V-rango {_ok(c_v_rango)}  "
        f"no-hielo(S alta o V baja) {_ok(c_no_hielo)}  "
        f"=> {_ok(bool(rojo_m[y, x]))}\n"
        f"azul : hue {_ok(c_azul_hue)}  "
        f"V-rango {_ok(c_v_rango)}  "
        f"no-hielo(S alta o V baja) {_ok(c_no_hielo)}  "
        f"=> {_ok(bool(azul_m[y, x]))}"
    )
    info_text.set_text(msg)
    print(msg)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", on_click)

redibujar()
plt.show()
cap.release()
