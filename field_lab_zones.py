"""Deteccion de zonas del campo, hibrido HSV + Lab (con extras experimentales).

  - Hielo: igual que en las versiones HSV/YCbCr (S<=s_ice_max & V>=v_ice_min),
    porque en la practica segmenta mucho mejor el hielo que un criterio de
    cromaticidad neutra.
  - Rojo/azul: con los canales de crominancia de Lab (a = verde<->rojo,
    b = azul<->amarillo). Estructuras rojas/azules ademas deben caer FUERA
    de la mascara de hielo (S alta o V baja).
  - Resolucion real: a diferencia de las versiones anteriores, aqui NO se
    reduce el frame a 0.5x antes de calcular las mascaras -ese resize
    promediaba (INTER_AREA) lineas finas con el hielo de alrededor y
    diluia su color antes de convertir a Lab-. Se trabaja al tamano nativo
    del video; solo se reescala para pintar en pantalla si hace falta
    (matplotlib lo hace solo, sin perder precision de calculo).
  - Neutro medido ("busqueda del neutral"): en vez de comparar a/b contra
    el 128 fijo de la teoria, se puede calibrar contra la MEDIANA de a/b
    de los pixeles que ya caen dentro de la mascara de hielo de ESE frame.
    Si la iluminacion de pista le da un tinte al hielo, el neutro real no
    es 128 y perdias margen de separacion; con esto se autoajusta por
    frame. Toggle "neutro = mediana hielo".
  - Ganancia de crominancia: separa a/b de su centro (128 o el medido)
    multiplicando la distancia. Solo toca a/b, nunca L ni R/G/B por
    separado, asi que no distorsiona el hue.
  - Mezcla con H (HSV): toggle "combinar con H (HSV)" que exige ADEMAS que
    el hue este en un rango razonable de rojo/azul (umbrales fijos, no
    expuestos como slider para no saturar la UI). La idea es que el ruido
    de compresion en 'a'/'b' y el ruido en 'H' no estan perfectamente
    correlados, asi que exigir que ambas señales coincidan debería tirar
    menos falsos positivos que cualquiera de las dos por separado.
  - Selector de vista: en vez de 6 paneles pequenos a la vez, un unico
    panel grande + botones de radio para elegir cual ver (como en
    test_colors.py). Mas facil de inspeccionar detalle con el frame a
    resolucion completa.

Uso:
  python field_lab_zones.py --video clip2.mp4
"""
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons

# umbrales de hue fijos para el modo "combinar con H (HSV)" - no se exponen
# como slider porque son un filtro secundario grueso, no el criterio principal.
HUE_ROJO_MARGEN = 15   # rojo: H<=margen o H>=179-margen
HUE_AZUL_LO = 95       # azul: H en [lo, hi]
HUE_AZUL_HI = 135

MIN_PIX_HIELO_PARA_NEUTRO = 500  # si hay menos pixeles de hielo, usa 128 fijo

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
    "dist_rojo": 22,        # rojo: (a con ganancia) - centro >= esto
    "dist_azul": 28,        # azul: centro - (b con ganancia) >= esto
    "l_struct_min": 60,     # rojo/azul: L dentro de [min, max]
    "l_struct_max": 255,
    "ganancia_ab_x10": 15,  # ganancia sobre (a-centro) y (b-centro), en decimas (10 = x1.0)
    "limpiar": False,       # apertura+cierre morfologico para quitar ruido suelto
    "neutro_medido": False, # centro de a/b = mediana del hielo del frame, en vez de 128 fijo
    "usar_hue": False,      # exigir tambien que el hue (HSV) encaje con rojo/azul
    "vista": "overlay",
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


def limpiar_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)
    return mask


def centro_ab(hielo_bool, a, b):
    """Centro (neutro) para a y b: 128 fijo, o la mediana del hielo del frame
    si el toggle esta activo y hay pixeles de hielo suficientes."""
    if not estado["neutro_medido"]:
        return 128.0, 128.0
    n = int(np.count_nonzero(hielo_bool))
    if n < MIN_PIX_HIELO_PARA_NEUTRO:
        return 128.0, 128.0
    return float(np.median(a[hielo_bool])), float(np.median(b[hielo_bool]))


def calcular_mascaras(bgr):
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab))

    hielo_bool = (s <= estado["s_ice_max"]) & (v >= estado["v_ice_min"])
    no_hielo = ~hielo_bool   # opuesto literal a la mascara de hielo

    ca, cb = centro_ab(hielo_bool, a, b)
    ganancia = estado["ganancia_ab_x10"] / 10.0
    a_g = ca + (a.astype(np.float32) - ca) * ganancia
    b_g = cb + (b.astype(np.float32) - cb) * ganancia

    l_rango = (l >= estado["l_struct_min"]) & (l <= estado["l_struct_max"])
    rojo = (a_g - ca >= estado["dist_rojo"]) & l_rango & no_hielo
    azul = (cb - b_g >= estado["dist_azul"]) & l_rango & no_hielo

    if estado["usar_hue"]:
        hue_rojo = (h <= HUE_ROJO_MARGEN) | (h >= 179 - HUE_ROJO_MARGEN)
        hue_azul = (h >= HUE_AZUL_LO) & (h <= HUE_AZUL_HI)
        rojo = rojo & hue_rojo
        azul = azul & hue_azul

    hielo = (hielo_bool.astype(np.uint8)) * 255
    rojo = (rojo.astype(np.uint8)) * 255
    azul = (azul.astype(np.uint8)) * 255

    if estado["limpiar"]:
        hielo = limpiar_mask(hielo)
        rojo = limpiar_mask(rojo)
        azul = limpiar_mask(azul)

    return hielo, rojo, azul, ca, cb


def overlay_rgb(bgr, hielo, rojo, azul):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    out = rgb.copy()
    out[hielo > 0] = out[hielo > 0] * 0.5 + np.array([255.0, 255.0, 255.0]) * 0.5
    out[rojo > 0] = (255.0, 0.0, 255.0)   # magenta
    out[azul > 0] = (0.0, 255.0, 255.0)   # cian
    return np.clip(out, 0, 255).astype(np.uint8)


VISTAS = ["original", "overlay", "hielo", "rojo", "azul", "combinado"]


def imagen_vista(nombre, frame, hielo, rojo, azul):
    if nombre == "original":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if nombre == "overlay":
        return overlay_rgb(frame, hielo, rojo, azul)
    if nombre == "hielo":
        return hielo
    if nombre == "rojo":
        return rojo
    if nombre == "azul":
        return azul
    return overlay_rgb(frame, np.zeros_like(hielo), rojo, azul)   # combinado


# ======================= FIGURA =======================
fig = plt.figure(figsize=(15, 9))
try:
    fig.canvas.manager.set_window_title("Zonas del campo por Lab (resolucion real)")
except Exception:
    pass

ax_big = fig.add_axes((0.22, 0.35, 0.75, 0.60))
ax_big.axis("off")
im_big = ax_big.imshow(np.zeros((2, 2, 3), np.uint8))

ax_radio_ver = fig.add_axes((0.02, 0.55, 0.17, 0.40))
ax_radio_ver.set_title("ver", fontsize=9)
radio_ver = RadioButtons(ax_radio_ver, VISTAS, active=VISTAS.index(estado["vista"]))
for t in radio_ver.labels:
    t.set_fontsize(8)

# --- sliders ---
slider_specs = [
    ("frame", "frame", 0, max(n_frames - 1, 1), 1),
    ("s_ice_max", "hielo S<=", 0, 255, 1),
    ("v_ice_min", "hielo V>=", 0, 255, 1),
    ("dist_rojo", "rojo dist>=", 0, 127, 1),
    ("dist_azul", "azul dist>=", 0, 127, 1),
    ("l_struct_min", "rojo/azul L>=", 0, 255, 1),
    ("l_struct_max", "rojo/azul L<=", 0, 255, 1),
    ("ganancia_ab_x10", "ganancia a/b (x0.1)", 10, 80, 1),
]
sliders = {}
y0 = 0.30
for i, (clave, label, vmin, vmax, step) in enumerate(slider_specs):
    ax_s = fig.add_axes((0.08, y0 - i * 0.032, 0.48, 0.02))
    sliders[clave] = Slider(ax_s, label, vmin, vmax, valinit=estado[clave], valstep=step)

ax_check = fig.add_axes((0.60, 0.02, 0.20, 0.13))
CHECKS = [
    ("limpiar (morf.)", "limpiar"),
    ("neutro = mediana hielo", "neutro_medido"),
    ("combinar con H (HSV)", "usar_hue"),
]
check = CheckButtons(ax_check, [lbl for lbl, _ in CHECKS], [estado[k] for _, k in CHECKS])
for t in check.labels:
    t.set_fontsize(7.5)

legend_text = fig.text(0.82, 0.32, "", fontsize=7, family="monospace", va="top")
info_text = fig.text(0.82, 0.16, "clic sobre la imagen para inspeccionar S/V/L/a/b",
                      fontsize=7, family="monospace", va="top")


def _ok(b):
    return "OK" if b else "no"


def texto_condiciones(ca, cb):
    origen = "medido" if estado["neutro_medido"] else "fijo"
    txt = (
        "condiciones actuales:\n"
        f"hielo: S<={estado['s_ice_max']}  &  V>={estado['v_ice_min']}\n"
        f"centro a/b ({origen}): ca={ca:.0f}  cb={cb:.0f}\n"
        f"rojo : (a_g-ca)>={estado['dist_rojo']}  &  L en [{estado['l_struct_min']},{estado['l_struct_max']}]"
        f"  &  NOT-hielo\n"
        f"azul : (cb-b_g)>={estado['dist_azul']}  &  L en [{estado['l_struct_min']},{estado['l_struct_max']}]"
        f"  &  NOT-hielo\n"
        f"ganancia a/b: x{estado['ganancia_ab_x10'] / 10.0:.1f}"
    )
    if estado["usar_hue"]:
        txt += (f"\n+ H rojo<={HUE_ROJO_MARGEN} o >={179 - HUE_ROJO_MARGEN}"
                f"  |  H azul en [{HUE_AZUL_LO},{HUE_AZUL_HI}]")
    return txt


def redibujar():
    frame = get_frame_bgr(estado["frame"])
    if frame is None:
        return
    hielo, rojo, azul, ca, cb = calcular_mascaras(frame)
    img = imagen_vista(estado["vista"], frame, hielo, rojo, azul)
    im_big.set_data(img)
    im_big.set_extent((0, img.shape[1], img.shape[0], 0))
    ax_big.set_title(f'vista: {estado["vista"]}', fontsize=10)

    total = hielo.size
    pct_hielo = 100.0 * np.count_nonzero(hielo) / total
    pct_rojo = 100.0 * np.count_nonzero(rojo) / total
    pct_azul = 100.0 * np.count_nonzero(azul) / total
    fig.suptitle(
        f'frame {estado["frame"]} / {n_frames - 1}   '
        f'hielo {pct_hielo:5.1f}%   rojo {pct_rojo:5.1f}%   azul {pct_azul:5.1f}%'
    )
    legend_text.set_text(texto_condiciones(ca, cb))
    fig.canvas.draw_idle()


def on_slider(clave):
    def _cb(v):
        estado[clave] = int(v)
        redibujar()
    return _cb


for clave, slider in sliders.items():
    slider.on_changed(on_slider(clave))


def on_radio_ver(label):
    estado["vista"] = label
    redibujar()


radio_ver.on_clicked(on_radio_ver)


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
    if event.inaxes is not ax_big:
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
    h, s, v = (int(c) for c in cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y_px, x])
    l, a_crudo, b_crudo = (int(c) for c in cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)[y_px, x])
    hielo_m, rojo_m, azul_m, ca, cb = calcular_mascaras(frame)

    ganancia = estado["ganancia_ab_x10"] / 10.0
    a_g = ca + (a_crudo - ca) * ganancia
    b_g = cb + (b_crudo - cb) * ganancia

    c_hielo_s = s <= estado["s_ice_max"]
    c_hielo_v = v >= estado["v_ice_min"]
    c_no_hielo = not (c_hielo_s and c_hielo_v)
    c_l_rango = estado["l_struct_min"] <= l <= estado["l_struct_max"]
    c_rojo_dist = (a_g - ca) >= estado["dist_rojo"]
    c_azul_dist = (cb - b_g) >= estado["dist_azul"]
    c_hue_rojo = (h <= HUE_ROJO_MARGEN) or (h >= 179 - HUE_ROJO_MARGEN)
    c_hue_azul = HUE_AZUL_LO <= h <= HUE_AZUL_HI

    msg = (
        f"px ({x},{y_px})  H={h}  S={s}  V={v}  L={l}\n"
        f"a={a_crudo}->{a_g:.0f} (centro={ca:.0f}, x{ganancia:.1f})  "
        f"b={b_crudo}->{b_g:.0f} (centro={cb:.0f}, x{ganancia:.1f})\n"
        f"hielo: S<={estado['s_ice_max']} {_ok(c_hielo_s)}  "
        f"V>={estado['v_ice_min']} {_ok(c_hielo_v)}  => {_ok(bool(hielo_m[y_px, x]))}\n"
        f"rojo : dist {_ok(c_rojo_dist)}  L-rango {_ok(c_l_rango)}  no-hielo {_ok(c_no_hielo)}"
        + (f"  H-rojo {_ok(c_hue_rojo)}" if estado["usar_hue"] else "")
        + f"  => {_ok(bool(rojo_m[y_px, x]))}\n"
        f"azul : dist {_ok(c_azul_dist)}  L-rango {_ok(c_l_rango)}  no-hielo {_ok(c_no_hielo)}"
        + (f"  H-azul {_ok(c_hue_azul)}" if estado["usar_hue"] else "")
        + f"  => {_ok(bool(azul_m[y_px, x]))}"
    )
    info_text.set_text(msg)
    print(msg)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", on_click)

redibujar()
plt.show()
cap.release()
