"""Deteccion de zonas del campo, hibrido HSV + Luv (variante de field_hsv_zones.py).

  - Hielo: igual que en las otras versiones (S<=s_ice_max & V>=v_ice_min, HSV).
  - Rojo/azul: con los canales de crominancia de CIE Luv (u, v) en vez de
    hue o Lab. En el diagrama de cromaticidad de Luv, rojo/verde/azul caen
    en tres esquinas bien separadas -de ahi el "triangulo"-: rojo tiene u
    muy alto, verde tiene u muy bajo y v alto, azul tiene v muy bajo. Por
    eso, a diferencia de Lab (donde hubo que medir distancia a un centro),
    aqui basta un corte de UN SOLO LADO por canal, igual de simple que
    Cr/Cb en YCbCr:
        rojo = u >= u_min
        azul = v <= v_luv_max
  - OJO con el neutro: en Lab, el gris neutro (a=b=0 en la teoria) cae en
    128/128 al pasar a 8-bit porque OpenCV solo suma +128 a ambos canales.
    En Luv, el escalado de OpenCV es distinto por canal
    (u<-255/354*(u+134), v<-255/262*(v+140)), asi que el neutro real NO
    cae en 128/128 sino en, aproximadamente, u~=96, v~=136. Los sliders de
    aqui son valores absolutos (no "distancia a 128"), asi que esto no
    afecta al codigo, pero conviene saberlo al razonar sobre los numeros
    que salen en el inspector de clic.
  - Resolucion real: sin reducir el frame (ver field_hsv_zones.py).
  - CLAHE en V (HSV): igual que en field_hsv_zones.py, contraste adaptativo
    solo sobre el value de HSV (afecta a la mascara de hielo), nunca sobre
    H/S ni sobre L/u/v de Luv.
  - Estructuras rojas/azules: ademas de superar su umbral de crominancia,
    deben caer FUERA de la mascara de hielo (S alta o V baja).

Uso:
  python field_luv_zones.py --video clip2.mp4
"""
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

# esquinas del triangulo de gamut sRGB en Luv 8-bit, calculadas con
# cv2.cvtColor sobre primarios puros (ver conversacion / verificacion).
ESQUINAS_LUV = {
    "rojo": (222, 173),
    "verde": (37, 241),
    "azul": (90, 10),
}
NEUTRO_LUV = (96, 136)   # blanco/gris puro cae aqui, no en (128,128)

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
    "u_min": 150,            # rojo: u (Luv) >= esto (neutro ~96)
    "v_luv_max": 90,         # azul: v (Luv) <= esto (neutro ~136)
    "l_struct_min": 40,      # rojo/azul: L (Luv) dentro de [min, max]
    "l_struct_max": 255,
    "limpiar": False,       # apertura+cierre morfologico para quitar ruido suelto
    "clahe_v": False,       # CLAHE solo sobre V de HSV, antes de calcular hielo
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
    _, s, v = obtener_hsv(bgr)
    l_luv, u_luv, v_luv = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2Luv))

    hielo = (s <= estado["s_ice_max"]) & (v >= estado["v_ice_min"])
    no_hielo = ~hielo   # opuesto literal a la mascara de hielo

    l_rango = (l_luv >= estado["l_struct_min"]) & (l_luv <= estado["l_struct_max"])
    rojo = (u_luv >= estado["u_min"]) & l_rango & no_hielo
    azul = (v_luv <= estado["v_luv_max"]) & l_rango & no_hielo

    hielo = (hielo.astype(np.uint8)) * 255
    rojo = (rojo.astype(np.uint8)) * 255
    azul = (azul.astype(np.uint8)) * 255

    if estado["limpiar"]:
        hielo = limpiar_mask(hielo)
        rojo = limpiar_mask(rojo)
        azul = limpiar_mask(azul)

    return hielo, rojo, azul, u_luv, v_luv


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
    fig.canvas.manager.set_window_title("Zonas del campo por Luv (hielo / rojo / azul)")
except Exception:
    pass

ax_orig = fig.add_axes((0.02, 0.68, 0.23, 0.29))
ax_overlay = fig.add_axes((0.27, 0.68, 0.23, 0.29))
ax_hielo = fig.add_axes((0.52, 0.68, 0.23, 0.29))
ax_rojo = fig.add_axes((0.02, 0.37, 0.23, 0.29))
ax_azul = fig.add_axes((0.27, 0.37, 0.23, 0.29))
ax_comb = fig.add_axes((0.52, 0.37, 0.23, 0.29))
ax_tri = fig.add_axes((0.78, 0.37, 0.20, 0.60))

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
    ("u_min", "rojo u(Luv)>=", 0, 255, 1),
    ("v_luv_max", "azul v(Luv)<=", 0, 255, 1),
    ("l_struct_min", "rojo/azul L(Luv)>=", 0, 255, 1),
    ("l_struct_max", "rojo/azul L(Luv)<=", 0, 255, 1),
]
sliders = {}
y0 = 0.30
for i, (clave, label, vmin, vmax, step) in enumerate(slider_specs):
    ax_s = fig.add_axes((0.06, y0 - i * 0.032, 0.47, 0.02))
    sliders[clave] = Slider(ax_s, label, vmin, vmax, valinit=estado[clave], valstep=step)

ax_check = fig.add_axes((0.56, 0.02, 0.20, 0.09))
CHECKS = [
    ("limpiar (morf.)", "limpiar"),
    ("CLAHE en V", "clahe_v"),
]
check = CheckButtons(ax_check, [lbl for lbl, _ in CHECKS], [estado[k] for _, k in CHECKS])

legend_text = fig.text(0.56, 0.34, "", fontsize=7, family="monospace", va="top")
info_text = fig.text(0.56, 0.19, "clic sobre una imagen para inspeccionar H/S/V/L/u/v",
                      fontsize=7, family="monospace", va="top")


def _ok(b):
    return "OK" if b else "no"


def dibujar_triangulo(u_luv, v_luv):
    ax_tri.clear()
    ax_tri.set_xlim(0, 255)
    ax_tri.set_ylim(0, 255)
    ax_tri.set_aspect("equal", adjustable="box")
    ax_tri.set_title("Luv (u,v) 8-bit", fontsize=9)
    ax_tri.set_xlabel("u  (verde <-> rojo)", fontsize=7)
    ax_tri.set_ylabel("v  (azul <-> amarillo)", fontsize=7)
    ax_tri.tick_params(labelsize=7)

    # densidad de pixeles del frame actual (submuestreado, solo para pintar)
    uu = u_luv[::4, ::4].ravel()
    vv = v_luv[::4, ::4].ravel()
    hist, _, _ = np.histogram2d(uu, vv, bins=64, range=[[0, 255], [0, 255]])
    ax_tri.imshow(np.log1p(hist).T, origin="lower", extent=(0, 255, 0, 255),
                  cmap="gray_r", aspect="auto")

    # umbrales actuales: zona roja (u alto) y zona azul (v bajo)
    ax_tri.axvspan(estado["u_min"], 255, color="red", alpha=0.10)
    ax_tri.axhspan(0, estado["v_luv_max"], color="blue", alpha=0.10)
    ax_tri.axvline(estado["u_min"], color="red", linestyle="--", linewidth=1.2)
    ax_tri.axhline(estado["v_luv_max"], color="blue", linestyle="--", linewidth=1.2)

    # triangulo de gamut sRGB (esquinas R/G/B) y punto neutro
    corners = list(ESQUINAS_LUV.values())
    tx, ty = zip(*(corners + [corners[0]]))
    ax_tri.plot(tx, ty, color="0.4", linewidth=1, linestyle=":")
    colores = {"rojo": "red", "verde": "green", "azul": "blue"}
    for nombre, (cu, cv_) in ESQUINAS_LUV.items():
        ax_tri.plot(cu, cv_, "o", color=colores[nombre], markersize=5)
        ax_tri.annotate(nombre, (cu, cv_), fontsize=7, color=colores[nombre],
                         textcoords="offset points", xytext=(4, 4))
    ax_tri.plot(*NEUTRO_LUV, "+", color="black", markersize=9, markeredgewidth=1.5)
    ax_tri.annotate("neutro", NEUTRO_LUV, fontsize=7, color="black",
                     textcoords="offset points", xytext=(4, -10))


def texto_condiciones():
    return (
        "condiciones actuales:\n"
        f"hielo: S<={estado['s_ice_max']}  &  V>={estado['v_ice_min']}\n"
        f"rojo : u(Luv)>={estado['u_min']}  &  L(Luv) en [{estado['l_struct_min']},{estado['l_struct_max']}]"
        f"  &  NOT-hielo (S>{estado['s_ice_max']} o V<{estado['v_ice_min']})\n"
        f"azul : v(Luv)<={estado['v_luv_max']}  &  L(Luv) en [{estado['l_struct_min']},{estado['l_struct_max']}]"
        f"  &  NOT-hielo (S>{estado['s_ice_max']} o V<{estado['v_ice_min']})\n"
        f"neutro teorico Luv (8-bit): u~=96  v~=136   CLAHE en V: {'ON' if estado['clahe_v'] else 'off'}"
    )


def redibujar():
    frame = get_frame_bgr(estado["frame"])
    if frame is None:
        return
    hielo, rojo, azul, u_luv, v_luv = calcular_mascaras(frame)
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
    dibujar_triangulo(u_luv, v_luv)
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
    x, y_px = int(round(event.xdata)), int(round(event.ydata))
    alto, ancho = frame.shape[:2]
    if not (0 <= x < ancho and 0 <= y_px < alto):
        return
    h_full, s_full, v_full = obtener_hsv(frame)
    h, s, v = int(h_full[y_px, x]), int(s_full[y_px, x]), int(v_full[y_px, x])
    l_luv, u_luv, v_luv = (int(c) for c in cv2.cvtColor(frame, cv2.COLOR_BGR2Luv)[y_px, x])
    hielo_m, rojo_m, azul_m, _, _ = calcular_mascaras(frame)

    c_hielo_s = s <= estado["s_ice_max"]
    c_hielo_v = v >= estado["v_ice_min"]
    c_no_hielo = not (c_hielo_s and c_hielo_v)
    c_l_rango = estado["l_struct_min"] <= l_luv <= estado["l_struct_max"]
    c_rojo_u = u_luv >= estado["u_min"]
    c_azul_v = v_luv <= estado["v_luv_max"]

    msg = (
        f"px ({x},{y_px})  H={h}  S={s}  V={v}  L={l_luv}  u={u_luv}  v={v_luv}\n"
        f"hielo: S<={estado['s_ice_max']} {_ok(c_hielo_s)}  "
        f"V>={estado['v_ice_min']} {_ok(c_hielo_v)}  => {_ok(bool(hielo_m[y_px, x]))}\n"
        f"rojo : u>={estado['u_min']} {_ok(c_rojo_u)}  "
        f"L-rango {_ok(c_l_rango)}  no-hielo(S alta o V baja) {_ok(c_no_hielo)}  "
        f"=> {_ok(bool(rojo_m[y_px, x]))}\n"
        f"azul : v<={estado['v_luv_max']} {_ok(c_azul_v)}  "
        f"L-rango {_ok(c_l_rango)}  no-hielo(S alta o V baja) {_ok(c_no_hielo)}  "
        f"=> {_ok(bool(azul_m[y_px, x]))}"
    )
    info_text.set_text(msg)
    print(msg)
    ax_tri.plot(u_luv, v_luv, "*", color="gold", markersize=14,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", on_click)

redibujar()
plt.show()
cap.release()
