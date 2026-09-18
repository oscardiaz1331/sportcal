import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, TextBox

VIDEO = "clip2.mp4"

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise IOError(f"no se pudo abrir el video {VIDEO}")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

_HAS_ED = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createEdgeDrawing")
print(f"EdgeDrawing {'disponible' if _HAS_ED else 'NO disponible'} (cv2.ximgproc)")
# ---- estado del miniprograma ----
estado = {
    "frame": 0,
    "espacio": "RGB",
    "canal": -1,        # -1 = ver todos los canales; 0/1/2 = ver uno solo en grande
    "normalize": False,
    "clahe": False,
    "otsu": False,
    "threshold": False,
    "canny": False,
    "edgedraw": False,
    "lineas": False,
    "elipses": False,
    "extremos": False,  # deja solo <=ext_margen (negro) y >=255-ext_margen (blanco), el resto a gris
    "thr_valor": 127,
    "s_umbral": 40,      # para "Difs": H se enmascara donde S <= s_umbral
    "canny_lo": 50,
    "canny_hi": 150,
    "ext_margen": 40,
}

# cache del frame BGR para no releer el video al cambiar de opcion
_cache = {"idx": None, "bgr": None}


def get_frame_bgr(idx):
    if _cache["idx"] == idx:
        return _cache["bgr"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    _cache["idx"], _cache["bgr"] = idx, frame
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return frame


# ---------- helpers de visualizacion float -> uint8 ----------
def u8_unit(x):
    """x en [0, 1] -> [0, 255]."""
    return np.clip(x.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)


def u8_signed(x, escala):
    """x con signo -> centrado en 128."""
    return np.clip(128.0 + x.astype(np.float32) * escala, 0, 255).astype(np.uint8)


# ---------- definicion de "espacios" (cada uno devuelve lista de (nombre, canal u8)) ----------
def esp_cvt(code, nombres):
    def f(bgr):
        return list(zip(nombres, cv2.split(cv2.cvtColor(bgr, code))))
    return f


def esp_cromaticidad(bgr):
    """r = R/(R+G+B), g = G/(R+G+B), b = B/(R+G+B). Quita sombras/iluminacion."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    s = b + g + r + 1e-6
    return [("r", u8_unit(r / s)), ("g", u8_unit(g / s)), ("b", u8_unit(b / s))]


def esp_excess(bgr):
    """Excess-Green (2G-R-B) y Excess-Red (2R-G-B). Resaltan camisetas."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    return [("ExG  2G-R-B", u8_signed(2 * g - r - b, 0.25)),
            ("ExR  2R-G-B", u8_signed(2 * r - g - b, 0.25))]


def esp_diferencias(bgr):
    """a-b (Lab), Cr-Cb (YCrCb) y H enmascarado por S > umbral."""
    _, a, bb = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32))
    _, Cr, Cb = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32))
    H, S, _ = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    h_mask = np.where(S > estado["s_umbral"], (H.astype(np.float32) * 255.0 / 179.0), 0)
    return [("a-b (Lab)", u8_signed(a - bb, 0.5)),
            ("Cr-Cb", u8_signed(Cr - Cb, 0.5)),
            (f'H | S>{estado["s_umbral"]}', h_mask.astype(np.uint8))]


ESPACIOS = {
    "RGB":     esp_cvt(cv2.COLOR_BGR2RGB,   ("R", "G", "B")),
    "HSV":     esp_cvt(cv2.COLOR_BGR2HSV,   ("H", "S", "V")),
    "HLS":     esp_cvt(cv2.COLOR_BGR2HLS,   ("H", "L", "S")),
    "Lab":     esp_cvt(cv2.COLOR_BGR2Lab,   ("L", "a", "b")),
    "Luv":     esp_cvt(cv2.COLOR_BGR2Luv,   ("L", "u", "v")),
    "YCrCb":   esp_cvt(cv2.COLOR_BGR2YCrCb, ("Y", "Cr", "Cb")),
    "YUV":     esp_cvt(cv2.COLOR_BGR2YUV,   ("Y", "U", "V")),
    "XYZ":     esp_cvt(cv2.COLOR_BGR2XYZ,   ("X", "Y", "Z")),
    "Cromat":  esp_cromaticidad,
    "ExG/ExR": esp_excess,
    "Difs":    esp_diferencias,
}
NOMBRES_ESPACIOS = list(ESPACIOS.keys())
NOMBRES_CANAL = {
    "RGB": ("R", "G", "B"), "HSV": ("H", "S", "V"), "HLS": ("H", "L", "S"),
    "Lab": ("L", "a", "b"), "Luv": ("L", "u", "v"), "YCrCb": ("Y", "Cr", "Cb"),
    "YUV": ("Y", "U", "V"), "XYZ": ("X", "Y", "Z"),
    "Cromat": ("r", "g", "b"), "ExG/ExR": ("ExG", "ExR"), "Difs": ("a-b", "Cr-Cb", "H|S"),
}


def procesar_canal(canal):
    """Cadena de realce/umbral sobre un canal (2D uint8)."""
    out = canal
    if estado["normalize"]:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    if estado["clahe"]:
        out = clahe.apply(out)
    if estado["otsu"]:
        _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if estado["threshold"]:
        _, out = cv2.threshold(out, int(estado["thr_valor"]), 255, cv2.THRESH_BINARY)
    if estado["extremos"]:
        # quita los valores medios: solo sobreviven los muy oscuros y los muy claros
        m = int(estado["ext_margen"])
        out = np.where(out <= m, 0, np.where(out >= 255 - m, 255, 128)).astype(np.uint8)
    return out


def _run_edgedrawing(img):
    ed = cv2.ximgproc.createEdgeDrawing()
    params = ed.Params()
    params.EdgeDetectionOperator = cv2.ximgproc.EDGE_DRAWING_SOBEL
    params.PFmode = False
    ed.setParams(params)
    ed.detectEdges(img)
    return ed


def postproceso(img):
    """Canny / EdgeDrawing + overlay de lineas y elipses. Devuelve imagen lista para imshow.

    - Lineas/elipses salen de EdgeDrawing si esta disponible.
    - Si NO hay EdgeDrawing, se usa el fallback (Hough / fitEllipse) SOLO cuando
      'canny' esta activo, sobre la imagen de bordes de Canny.
    """
    ed = None
    base = img

    if _HAS_ED and (estado["edgedraw"] or estado["lineas"] or estado["elipses"]):
        ed = _run_edgedrawing(img)
        if estado["edgedraw"]:
            base = ed.getEdgeImage()

    canny_edges = None
    if estado["canny"]:
        canny_edges = cv2.Canny(img, int(estado["canny_lo"]), int(estado["canny_hi"]))
        if not estado["edgedraw"]:
            base = canny_edges

    if not (estado["lineas"] or estado["elipses"]):
        return base   # 2D -> se pinta en gris

    vis = cv2.cvtColor(base if base.ndim == 2 else img, cv2.COLOR_GRAY2BGR)

    if estado["lineas"]:
        segs = []
        if ed is not None:                       # EdgeDrawing -> EDLines
            lines = ed.detectLines()
            segs = [] if lines is None else lines.reshape(-1, 4)
        elif canny_edges is not None:            # fallback: Hough sobre Canny
            hl = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, 80,
                                 minLineLength=40, maxLineGap=10)
            segs = [] if hl is None else hl.reshape(-1, 4)
        for x1, y1, x2, y2 in segs:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

    if estado["elipses"]:
        if ed is not None:                       # EdgeDrawing -> EDCircles
            ells = ed.detectEllipses()
            if ells is not None:
                for e in ells.reshape(-1, 6):
                    cx, cy, a, b, c, theta = e
                    cv2.ellipse(vis, (int(cx), int(cy)),
                                (int(a + b), int(a + c)), theta, 0, 360, (0, 255, 0), 1)
        elif canny_edges is not None:            # fallback: fitEllipse sobre contornos
            cnts, _ = cv2.findContours(canny_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cnt in cnts:
                if len(cnt) >= 20 and cv2.contourArea(cnt) > 150:
                    cv2.ellipse(vis, cv2.fitEllipse(cnt), (0, 255, 0), 1)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)   # 3D -> color


def render_canal(canal):
    return postproceso(procesar_canal(canal))


def canales_actuales():
    frame = get_frame_bgr(estado["frame"])
    if frame is None:
        return None
    return ESPACIOS[estado["espacio"]](frame)   # list[(nombre, canal)]


# ======================= FIGURA =======================
fig = plt.figure(figsize=(14, 8))
try:
    fig.canvas.manager.set_window_title("Explorador de espacios de color")
except Exception:
    pass

ax_imgs = [fig.add_axes((0.24 + i * 0.255, 0.30, 0.24, 0.62)) for i in range(3)]
ax_big = fig.add_axes((0.24, 0.30, 0.74, 0.62))
ims = [a.imshow(np.zeros((2, 2), np.uint8), cmap="gray", vmin=0, vmax=255) for a in ax_imgs]
im_big = ax_big.imshow(np.zeros((2, 2), np.uint8), cmap="gray", vmin=0, vmax=255)
for a in (*ax_imgs, ax_big):
    a.axis("off")

# --- slider de frame ---
ax_slider = fig.add_axes((0.24, 0.15, 0.52, 0.03))
slider = Slider(ax_slider, "frame", 0, max(n_frames - 1, 1), valinit=0, valstep=1)

# --- radio: espacio de color ---
ax_radio_esp = fig.add_axes((0.015, 0.42, 0.165, 0.55))
ax_radio_esp.set_title("espacio", fontsize=9)
radio_esp = RadioButtons(ax_radio_esp, NOMBRES_ESPACIOS, active=0)
for t in radio_esp.labels:
    t.set_fontsize(8)

# --- radio: que canal ver ---
ax_radio_ch = fig.add_axes((0.015, 0.28, 0.165, 0.12))
ax_radio_ch.set_title("ver", fontsize=9)
radio_ch = RadioButtons(ax_radio_ch, ("todos", "R", "G", "B"), active=0)
for t in radio_ch.labels:
    t.set_fontsize(8)

# --- checkbuttons: procesado ---
ax_check = fig.add_axes((0.015, 0.02, 0.165, 0.24))
ax_check.set_title("procesado", fontsize=9)
CHECKS = ["normalize", "clahe", "otsu", "threshold", "extremos",
          "canny", "edgedraw", "lineas", "elipses"]
check = CheckButtons(ax_check, CHECKS, [False] * len(CHECKS))
for t in check.labels:
    t.set_fontsize(8)

# --- textboxes ---
ax_txt = fig.add_axes((0.80, 0.15, 0.045, 0.035))
txt = TextBox(ax_txt, "thr ", initial="127")
ax_txt_s = fig.add_axes((0.90, 0.15, 0.045, 0.035))
txt_s = TextBox(ax_txt_s, "S> ", initial=str(estado["s_umbral"]))
ax_txt_clo = fig.add_axes((0.80, 0.09, 0.045, 0.035))
txt_clo = TextBox(ax_txt_clo, "C lo ", initial=str(estado["canny_lo"]))
ax_txt_chi = fig.add_axes((0.90, 0.09, 0.045, 0.035))
txt_chi = TextBox(ax_txt_chi, "C hi ", initial=str(estado["canny_hi"]))
ax_txt_ext = fig.add_axes((0.80, 0.03, 0.045, 0.035))
txt_ext = TextBox(ax_txt_ext, "ext +- ", initial=str(estado["ext_margen"]))


# ======================= DIBUJO =======================
def redibujar():
    items = canales_actuales()
    if not items:
        return

    n = min(len(items), 3)
    modo_big = 0 <= estado["canal"] < len(items)

    for i, a in enumerate(ax_imgs):
        a.set_visible(not modo_big and i < n)
    ax_big.set_visible(modo_big)

    if modo_big:
        nombre, canal = items[estado["canal"]]
        img = render_canal(canal)
        im_big.set_data(img)
        im_big.set_extent((0, img.shape[1], img.shape[0], 0))
        ax_big.set_title(f'{estado["espacio"]} · {nombre}')
    else:
        for i in range(n):
            nombre, canal = items[i]
            img = render_canal(canal)
            ims[i].set_data(img)
            ims[i].set_extent((0, img.shape[1], img.shape[0], 0))
            ax_imgs[i].set_title(f'{estado["espacio"]} · {nombre}')

    aviso = ""
    if estado["edgedraw"] and not _HAS_ED:
        aviso = "   [edgedraw necesita cv2.ximgproc]"
    elif (estado["lineas"] or estado["elipses"]) and not _HAS_ED and not estado["canny"]:
        aviso = "   [sin EdgeDrawing: activa 'canny' para lineas/elipses]"
    fig.suptitle(f'frame {estado["frame"]} / {n_frames - 1}{aviso}')
    fig.canvas.draw_idle()


# ======================= CALLBACKS =======================
def on_slider(v):
    estado["frame"] = int(v)
    redibujar()


def on_espacio(label):
    estado["espacio"] = label
    nombres = NOMBRES_CANAL[label]
    etiquetas = ["todos", *nombres] + ["-"] * (3 - len(nombres))
    for t, nuevo in zip(radio_ch.labels, etiquetas):
        t.set_text(nuevo)
    estado["canal"] = -1
    radio_ch.set_active(0)   # dispara on_canal -> redibuja
    redibujar()


def on_canal(label):
    nombres = list(NOMBRES_CANAL[estado["espacio"]])
    estado["canal"] = nombres.index(label) if label in nombres else -1
    redibujar()


def on_check(label):
    estado[label] = not estado[label]
    redibujar()


def _set_int(clave, text, solo_si=None):
    try:
        estado[clave] = int(float(text))
    except ValueError:
        return
    if solo_si is None or solo_si():
        redibujar()


slider.on_changed(on_slider)
radio_esp.on_clicked(on_espacio)
radio_ch.on_clicked(on_canal)
check.on_clicked(on_check)
txt.on_submit(lambda s: _set_int("thr_valor", s, lambda: estado["threshold"]))
txt_s.on_submit(lambda s: _set_int("s_umbral", s, lambda: estado["espacio"] == "Difs"))
txt_clo.on_submit(lambda s: _set_int("canny_lo", s, lambda: estado["canny"]))
txt_chi.on_submit(lambda s: _set_int("canny_hi", s, lambda: estado["canny"]))
txt_ext.on_submit(lambda s: _set_int("ext_margen", s, lambda: estado["extremos"]))


def on_key(event):
    if event.key in ("left", "right"):
        paso = -1 if event.key == "left" else 1
        slider.set_val(min(max(estado["frame"] + paso, 0), n_frames - 1))


fig.canvas.mpl_connect("key_press_event", on_key)

redibujar()
plt.show()
cap.release()
