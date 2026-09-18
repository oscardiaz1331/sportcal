"""Visor por pasos de la extraccion de lineas rojas/azules dentro de la pista.

Idea: la mascara de hielo (Y > thr, como en test_rick.py) da el poligono de la
pista. Las lineas pintadas son justo lo que NO es hielo pero esta *rodeado* de
hielo, y ademas conserva la luminancia alta (pintura bajo hielo -> rojo/azul
palido) frente a las camisetas, que son oscuras y compactas.

Cadena:
  1. hielo = claro (Y>thr) Y neutro (|a|,|b| < cmax) -> pista (fill holes)
  2. recorte al techo del hielo, para dejar fuera la valla y su publicidad
  3. banda de croma: mas rojo/azul que el hielo a AMBOS lados, en 4 direcciones
     y varias distancias (la misma linea vale a*=18 cerca y a*=4 al fondo)
  4. rodeo: fraccion de hielo alrededor + guarda alrededor de lo oscuro
  5. candidatos = banda & rodeo & L>Lmin & dentro de la pista
  6. limpieza: close + area minima + elongacion (minAreaRect)

Lo que descarta cada filtro: el croma normalizado quita el tinte de camara, el
test de ambos lados quita los degradados de iluminacion, L>Lmin quita las
camisetas rojas (L~60-80 frente a L~130-155 de la linea roja), la guarda quita
sus bordes antialiasing, y el techo quita la publicidad de la valla.

Uso:
  python test_lineas.py                 # visor interactivo
  python test_lineas.py --dump 300      # guarda la rejilla del frame 300 a PNG
"""
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, CheckButtons

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="clip2.mp4")
parser.add_argument("--dump", type=int, default=None, help="guarda el frame N a PNG y sale")
parser.add_argument("--out", default="scratch_frames/lineas.png")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise IOError(f"no se pudo abrir el video {args.video}")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

estado = {
    "frame": 0, "thr": 185, "k_open": 5, "k_close": 35,   # pista
    "c_rojo": 25, "c_azul": 25, "l_min": 110, "k_top": 64, "c_max": 8,   # color (c_* en decimas de Lab)
    "radio": 81, "pct": 50, "k_guard": 9,                 # rodeo de hielo / guarda de oscuros
    "k_techo": 61, "m_techo": 14,                          # recorte del borde superior del hielo
    "k_lin": 9, "area_min": 150, "elong": 20,             # limpieza (elong en decimas)
    "rodeo": True, "filtro": True,
}
_cache = {"idx": None, "bgr": None}


def get_frame(idx):
    if _cache["idx"] != idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        _cache["idx"], _cache["bgr"] = (idx, fr) if ok else (None, None)
    return _cache["bgr"]


def _el(k):
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _impar(k):
    k = max(3, int(k))
    return k + 1 - k % 2


def u8_signed(x, escala=6.0):
    """canal con signo -> gris centrado en 0/128, para visualizar."""
    return np.clip(128.0 + x.astype(np.float32) * escala, 0, 255).astype(np.uint8)


# --------------------------- etapas ---------------------------
def preparar(frame):
    """Lab normalizado + mascara de hielo (claro Y NEUTRO).

    Pedir croma bajo ademas de Y alto es lo que deja fuera el kickplate
    amarillo de la valla, que si no entra como hielo y sube el techo.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    y = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]

    claro = y > int(estado["thr"])
    if claro.sum() > 500:                    # el hielo es gris -> offset de camara
        a = a - float(np.median(a[claro]))
        b = b - float(np.median(b[claro]))

    neutro = (np.abs(a) < estado["c_max"]) & (np.abs(b) < estado["c_max"])
    return L, a, b, np.uint8(claro & neutro) * 255


def mascara_pista(hielo):
    """Componente mayor del hielo con los huecos rellenos = poligono de pista."""
    op = cv2.morphologyEx(hielo, cv2.MORPH_OPEN, _el(estado["k_open"]))
    cl = cv2.morphologyEx(op, cv2.MORPH_CLOSE, _el(estado["k_close"]))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cl, connectivity=8)
    cc = np.uint8(lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255 if n > 1 else cl.copy()

    ff = cc.copy()
    h, w = cc.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cc | cv2.bitwise_not(ff)


def techo_hielo(hielo, pista):
    """Mascara de lo que queda por debajo del borde superior del hielo.

    El fill-holes de la pista se traga la valla (publicidad roja) cuando el
    hielo la toca; el primer pixel de hielo de cada columna, suavizado con una
    mediana movil, da el techo real y deja la valla fuera.
    """
    h, w = hielo.shape
    ice = (hielo > 0) & (pista > 0)
    col = np.where(ice.any(0), ice.argmax(0), h).astype(np.float32)

    k = _impar(estado["k_techo"])
    pad = np.pad(col, k // 2, mode="edge")
    vent = np.lib.stride_tricks.sliding_window_view(pad, k)
    col = np.median(vent, axis=1) + estado["m_techo"]
    return np.arange(h)[:, None] >= col[None, :]


DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _shift(c, dx, dy):
    """Traslacion con borde replicado (sin el wrap de np.roll)."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(c, M, (c.shape[1], c.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def realce_banda(c):
    """Cuanto mas rojo/azul es el pixel que el hielo a AMBOS lados.

    Un umbral absoluto no sirve (la misma linea vale a*=18 cerca y a*=4 al
    fondo) y un top-hat tampoco: un degradado de iluminacion lo pasa entero.
    Exigir contraste por los dos lados a la vez es justo "rodeado de hielo", y
    un tinte suave del hielo no lo cumple en ninguna direccion.

    Se prueba en 4 direcciones (la perpendicular a la linea es la que responde)
    y a varias distancias, porque en perspectiva la misma linea mide 4 px al
    fondo y 40 en primer plano.
    """
    k = max(8, int(estado["k_top"]))
    resp = np.zeros_like(c)
    for d in (k // 8, k // 4, k // 2):
        for dx, dy in DIRS:
            lado_a = c - _shift(c, d * dx, d * dy)
            lado_b = c - _shift(c, -d * dx, -d * dy)
            resp = np.maximum(resp, np.minimum(lado_a, lado_b))
    return resp


def bandas_croma(a, b):
    """Respuesta de banda para rojo (a+) y azul (b-)."""
    return realce_banda(a), realce_banda(-b)


def frac_hielo(hielo, pista):
    """Fraccion de hielo alrededor de cada pixel (disco ~ box de radio R)."""
    ice = ((hielo > 0) & (pista > 0)).astype(np.float32)
    r = _impar(estado["radio"])
    return cv2.blur(ice, (r, r))


def guarda_oscuros(L, pista):
    """Zona de exclusion alrededor de lo oscuro (jugadores, palos, sombras).

    Sin esto, el borde antialiasing camiseta-hielo da pixeles claros y rojos
    que pasan todos los filtros.
    """
    oscuro = np.uint8((L < estado["l_min"]) & (pista > 0)) * 255
    return cv2.dilate(oscuro, _el(estado["k_guard"])) > 0


def limpiar(mask):
    """close + descarte por area y por elongacion (minAreaRect). Devuelve (mask, ejes)."""
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _el(estado["k_lin"]))
    if not estado["filtro"]:
        return m, []
    out = np.zeros_like(m)
    ejes = []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < estado["area_min"]:
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        largo, corto = max(rw, rh), max(min(rw, rh), 1e-3)
        if largo / corto < estado["elong"] / 10.0:
            continue
        cv2.drawContours(out, [c], -1, 255, -1)
        t = np.deg2rad(ang if rw >= rh else ang + 90)
        dx, dy = np.cos(t) * largo / 2, np.sin(t) * largo / 2
        ejes.append(((int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy))))
    return out, ejes


def overlay(frame, rojo, azul, alpha=0.55):
    vis = frame.copy()
    capa = np.zeros_like(frame)
    capa[rojo > 0] = (0, 0, 255)
    capa[azul > 0] = (255, 40, 0)
    m = ((rojo > 0) | (azul > 0))[..., None]
    return np.where(m, cv2.addWeighted(vis, 1 - alpha, capa, alpha, 0), vis)


def pipeline(frame):
    L, a, b, hielo = preparar(frame)
    pista = mascara_pista(hielo)
    top_r, top_b = bandas_croma(a, b)
    frac = frac_hielo(hielo, pista)

    dentro = (pista > 0) & techo_hielo(hielo, pista)
    rodeado = frac >= estado["pct"] / 100.0 if estado["rodeo"] else np.ones_like(frac, bool)
    claro = L > estado["l_min"]                       # pintura bajo hielo, no camiseta
    cerca_oscuro = guarda_oscuros(L, pista)

    base = dentro & rodeado & claro & ~cerca_oscuro
    rojo_raw = np.uint8(base & (top_r > estado["c_rojo"] / 10.0)) * 255
    azul_raw = np.uint8(base & (top_b > estado["c_azul"] / 10.0)) * 255

    rojo, ejes_r = limpiar(rojo_raw)
    azul, ejes_a = limpiar(azul_raw)

    # --- visualizaciones ---
    vis_pista = frame.copy()
    cnts, _ = cv2.findContours(pista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(vis_pista, [max(cnts, key=cv2.contourArea)], -1, (0, 255, 0), 2)
    vis_pista[~dentro] = vis_pista[~dentro] // 2 + np.uint8((0, 60, 0))

    vis_frac = cv2.applyColorMap(np.uint8(frac * 255), cv2.COLORMAP_VIRIDIS)
    vis_frac[~dentro] //= 4
    vis_frac[rodeado & dentro] = (255, 255, 255)
    vis_frac[cerca_oscuro] = (0, 0, 160)

    vis_croma = cv2.merge([u8_signed(top_b, 12), np.zeros_like(hielo), u8_signed(top_r, 12)])
    vis_croma[~dentro] //= 4

    vis_raw = overlay(frame, rojo_raw, azul_raw)
    vis_fin = overlay(frame, rojo, azul, 0.7)
    for p, q in ejes_r + ejes_a:
        cv2.line(vis_fin, p, q, (0, 255, 255), 2)

    n_r, n_a = len(ejes_r), len(ejes_a)
    return [
        ("1. pista (Y>%d) recortada al techo del hielo" % estado["thr"], vis_pista),
        ("2. banda de croma d<=%d  R=rojo B=azul" % (estado["k_top"] // 2), vis_croma),
        ("3. rodeo r=%d >=%d%%  (rojo = guarda oscuros)" % (_impar(estado["radio"]), estado["pct"]), vis_frac),
        ("4. candidatos  top>%.1f/%.1f, L>%d" % (estado["c_rojo"] / 10, estado["c_azul"] / 10, estado["l_min"]), vis_raw),
        ("5. rojo limpio  (%d)" % n_r, cv2.cvtColor(rojo, cv2.COLOR_GRAY2BGR)),
        ("6. lineas  rojo=%d azul=%d" % (n_r, n_a), vis_fin),
    ]


# ======================= FIGURA =======================
fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
axes = axes.ravel()
fig.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.16, wspace=0.05, hspace=0.15)


def _tb(x, y, key, label, w=0.035):
    ax = fig.add_axes((x, y, w, 0.04))
    tb = TextBox(ax, label, initial=str(estado[key]))
    tb.label.set_fontsize(8)
    tb.on_submit(lambda s, k=key: _set(k, s))
    return tb


def _set(key, text):
    try:
        estado[key] = int(float(text))
    except ValueError:
        return
    redibujar()


CAJAS = [
    # fila 1: pista
    (0.055, 0.095, "thr", "Y> "), (0.125, 0.095, "k_open", "open "),
    (0.195, 0.095, "k_close", "close "), (0.285, 0.095, "l_min", "Lmin "),
    (0.355, 0.095, "radio", "radio "), (0.425, 0.095, "pct", "%hielo "),
    # fila 2: color y limpieza
    (0.055, 0.025, "c_rojo", "rojo "), (0.125, 0.025, "c_azul", "azul "),
    (0.195, 0.025, "k_lin", "close2 "), (0.285, 0.025, "area_min", "area "),
    (0.355, 0.025, "elong", "elong "),
]
_boxes = [_tb(x, y, k, lab) for x, y, k, lab in CAJAS]

ax_slider = fig.add_axes((0.60, 0.10, 0.35, 0.03))
slider = Slider(ax_slider, "frame", 0, max(n_frames - 1, 1), valinit=0, valstep=1)

ax_chk = fig.add_axes((0.60, 0.005, 0.10, 0.075))
chk = CheckButtons(ax_chk, ["rodeo", "filtro"], [estado["rodeo"], estado["filtro"]])
for t in chk.labels:
    t.set_fontsize(8)


def on_chk(label):
    estado[label] = not estado[label]
    redibujar()


def redibujar(_=None):
    frame = get_frame(estado["frame"])
    if frame is None:
        return
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    for ax, (titulo, img) in zip(axes, pipeline(frame)):
        ax.clear()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(titulo, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"frame {estado['frame']} / {n_frames - 1}   "
                 f"(elong = decimas, {estado['elong']/10:g})", fontsize=10)
    fig.canvas.draw_idle()


def on_slider(v):
    estado["frame"] = int(v)
    redibujar()


def on_key(event):
    if event.key in ("left", "right"):
        step = -1 if event.key == "left" else 1
        slider.set_val(min(max(estado["frame"] + step, 0), n_frames - 1))


slider.on_changed(on_slider)
chk.on_clicked(on_chk)
fig.canvas.mpl_connect("key_press_event", on_key)

if args.dump is not None:
    estado["frame"] = args.dump
    slider.set_val(args.dump)
    redibujar()
    fig.savefig(args.out, dpi=110)
    print("guardado", args.out)
else:
    redibujar()
    plt.show()
cap.release()
