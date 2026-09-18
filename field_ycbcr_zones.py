"""Deteccion de zonas del campo, hibrido HSV + YCbCr (variante de field_hsv_zones.py).

  - Hielo: igual que en la version HSV (S<=s_ice_max & V>=v_ice_min), porque
    en la practica segmenta mucho mejor el hielo que el criterio de
    crominancia neutra de YCbCr.
  - Rojo/azul: con crominancia YCbCr en vez de hue.
    Cr = "rojez" del pixel (resta lineal fija, sin dividir por nada -> no se
    dispara de ruido en zonas oscuras como pasaba con S de HSV) y Cb =
    "azulez". Como no hay "rueda" de hue, no hace falta banda de
    arriba/abajo: basta un umbral simple "Cr >= umbral" / "Cb >= umbral".
  - Estructuras rojas/azules: ademas de superar su umbral de crominancia,
    deben caer FUERA de la mascara de hielo (S alta o V baja), igual que en
    la version HSV.

Uso:
  python field_ycbcr_zones.py --video clip2.mp4
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
    "s_ice_max": 50,        # hielo: saturacion (HSV) <= esto
    "v_ice_min": 140,       # hielo: value (HSV) >= esto
    "cr_min": 150,          # rojo: Cr >= esto
    "cb_min": 150,          # azul: Cb >= esto
    "y_struct_min": 60,     # rojo/azul: Y dentro de [min, max]
    "y_struct_max": 255,
    "limpiar": False,       # apertura+cierre morfologico para quitar ruido suelto
}

_cache = {"idx": None, "bgr": None}


def get_frame_bgr(idx):
    if _cache["idx"] == idx:
        return _cache["bgr"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _cache["idx"], _cache["bgr"] = idx, frame
    return frame


_kernel = np.ones((3, 3), np.uint8)


def limpiar_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)
    return mask


def calcular_mascaras(bgr):
    _, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    y, cr, cb = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))

    hielo = (s <= estado["s_ice_max"]) & (v >= estado["v_ice_min"])
    no_hielo = ~hielo   # opuesto literal a la mascara de hielo

    y_rango = (y >= estado["y_struct_min"]) & (y <= estado["y_struct_max"])
    rojo = (cr >= estado["cr_min"]) & y_rango & no_hielo
    azul = (cb >= estado["cb_min"]) & y_rango & no_hielo

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
    fig.canvas.manager.set_window_title("Zonas del campo por YCbCr (hielo / rojo / azul)")
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
    ("cr_min", "rojo Cr>=", 128, 255, 1),
    ("cb_min", "azul Cb>=", 128, 255, 1),
    ("y_struct_min", "rojo/azul Y>=", 0, 255, 1),
    ("y_struct_max", "rojo/azul Y<=", 0, 255, 1),
]
sliders = {}
y0 = 0.30
for i, (clave, label, vmin, vmax, step) in enumerate(slider_specs):
    ax_s = fig.add_axes((0.08, y0 - i * 0.032, 0.55, 0.02))
    sliders[clave] = Slider(ax_s, label, vmin, vmax, valinit=estado[clave], valstep=step)

ax_check = fig.add_axes((0.72, 0.02, 0.15, 0.06))
check = CheckButtons(ax_check, ["limpiar (morf.)"], [estado["limpiar"]])

legend_text = fig.text(0.65, 0.36, "", fontsize=7.5, family="monospace", va="top")
info_text = fig.text(0.65, 0.20, "clic sobre una imagen para inspeccionar S/V/Y/Cr/Cb",
                      fontsize=7.5, family="monospace", va="top")


def _ok(b):
    return "OK" if b else "no"


def texto_condiciones():
    return (
        "condiciones actuales:\n"
        f"hielo: S<={estado['s_ice_max']}  &  V>={estado['v_ice_min']}\n"
        f"rojo : Cr>={estado['cr_min']}  &  Y en [{estado['y_struct_min']},{estado['y_struct_max']}]"
        f"  &  NOT-hielo (S>{estado['s_ice_max']} o V<{estado['v_ice_min']})\n"
        f"azul : Cb>={estado['cb_min']}  &  Y en [{estado['y_struct_min']},{estado['y_struct_max']}]"
        f"  &  NOT-hielo (S>{estado['s_ice_max']} o V<{estado['v_ice_min']})"
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


def on_check(_label):
    estado["limpiar"] = not estado["limpiar"]
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
    x, y_px = int(round(event.xdata)), int(round(event.ydata))
    alto, ancho = frame.shape[:2]
    if not (0 <= x < ancho and 0 <= y_px < alto):
        return
    _, s, v = (int(c) for c in cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y_px, x])
    y, cr, cb = (int(c) for c in cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[y_px, x])
    hielo_m, rojo_m, azul_m = calcular_mascaras(frame)

    c_hielo_s = s <= estado["s_ice_max"]
    c_hielo_v = v >= estado["v_ice_min"]
    c_no_hielo = not (c_hielo_s and c_hielo_v)
    c_y_rango = estado["y_struct_min"] <= y <= estado["y_struct_max"]
    c_rojo_cr = cr >= estado["cr_min"]
    c_azul_cb = cb >= estado["cb_min"]

    msg = (
        f"px ({x},{y_px})  S={s}  V={v}  Y={y}  Cr={cr}  Cb={cb}\n"
        f"hielo: S<={estado['s_ice_max']} {_ok(c_hielo_s)}  "
        f"V>={estado['v_ice_min']} {_ok(c_hielo_v)}  => {_ok(bool(hielo_m[y_px, x]))}\n"
        f"rojo : Cr>={estado['cr_min']} {_ok(c_rojo_cr)}  "
        f"Y-rango {_ok(c_y_rango)}  no-hielo(S alta o V baja) {_ok(c_no_hielo)}  "
        f"=> {_ok(bool(rojo_m[y_px, x]))}\n"
        f"azul : Cb>={estado['cb_min']} {_ok(c_azul_cb)}  "
        f"Y-rango {_ok(c_y_rango)}  no-hielo(S alta o V baja) {_ok(c_no_hielo)}  "
        f"=> {_ok(bool(azul_m[y_px, x]))}"
    )
    info_text.set_text(msg)
    print(msg)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", on_click)

redibujar()
plt.show()
cap.release()
