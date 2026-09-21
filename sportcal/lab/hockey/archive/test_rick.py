"""Visor por pasos del pipeline de extraccion de la pista.

Muestra en una rejilla cada etapa sobre el frame elegido con el slider:
binaria -> open -> close -> componente mayor -> fill holes -> caracteristicas.
Sobre cada etapa se pintan los corners (Shi-Tomasi / Harris).
Textboxes para umbral, kernels y parametros de corners.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, CheckButtons

VIDEO = "clip2.mp4"

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise IOError(f"no se pudo abrir el video {VIDEO}")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

estado = {
    "frame": 0, "thr": 185, "k_open": 5, "k_close": 35,
    "corners": True, "harris": False,
    "max_corners": 150, "quality": 10, "min_dist": 10,   # quality en milesimas -> /1000
}
_cache = {"idx": None, "bgr": None}


def detectar_corners(gray):
    pts = cv2.goodFeaturesToTrack(
        gray, max(estado["max_corners"], 1), max(estado["quality"], 1) / 1000.0,
        max(estado["min_dist"], 1), useHarrisDetector=estado["harris"], k=0.04,
    )
    return np.empty((0, 2)) if pts is None else pts.reshape(-1, 2)


def con_corners(img):
    """img (gris o BGR) -> BGR con los corners pintados y su conteo."""
    if img.ndim == 2:
        gray, vis = img, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        gray, vis = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img.copy()
    if estado["corners"]:
        pts = detectar_corners(gray)
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 90, 255), -1)
        cv2.putText(vis, f"{len(pts)} corners", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 90, 255), 2)
    return vis


def get_frame(idx):
    if _cache["idx"] != idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        _cache["idx"], _cache["bgr"] = (idx, fr) if ok else (None, None)
    return _cache["bgr"]


def _el(k):
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def pipeline(frame):
    """Devuelve [(titulo, imagen), ...] con cada etapa del procesado."""
    y = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
    binaria = cv2.threshold(y, int(estado["thr"]), 255, cv2.THRESH_BINARY)[1]

    op = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _el(estado["k_open"]))
    cl = cv2.morphologyEx(op, cv2.MORPH_CLOSE, _el(estado["k_close"]))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cl, connectivity=8)
    cc = np.uint8(lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255 if n > 1 else cl.copy()

    ff = cc.copy()
    h, w = cc.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    lleno = cc | cv2.bitwise_not(ff)

    # ---- caracteristicas: corners sobre la imagen real + geometria del campo ----
    vis = con_corners(frame.copy())
    cnts, _ = cv2.findContours(lleno, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        field = max(cnts, key=cv2.contourArea)
        cv2.drawContours(vis, [field], -1, (0, 255, 0), 3)

        M = cv2.moments(field)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(vis, (cx, cy), 12, (0, 0, 255), -1)

        cv2.drawContours(vis, [np.int32(cv2.boxPoints(cv2.minAreaRect(field)))],
                         0, (0, 255, 255), 2)

        col_has = lleno.max(axis=0) > 0
        if col_has.sum() > 10:
            xs = np.arange(w, dtype=np.float32)[col_has]
            ys = np.argmax(lleno > 0, axis=0).astype(np.float32)[col_has]
            vx, vy, x0, y0 = cv2.fitLine(np.c_[xs, ys], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            cv2.line(vis, (int(x0 - vx * w), int(y0 - vy * w)),
                          (int(x0 + vx * w), int(y0 + vy * w)), (255, 0, 255), 2)

    area_pct = 100 * lleno.mean() / 255
    etiqueta = "Harris" if estado["harris"] else "Shi-Tomasi"
    return [
        (f"1. binaria  Y>{estado['thr']}", con_corners(binaria)),
        (f"2. open  k={estado['k_open']}", con_corners(op)),
        (f"3. close  k={estado['k_close']}", con_corners(cl)),
        ("4. componente mayor", con_corners(cc)),
        (f"5. fill holes  ({area_pct:.0f}%)", con_corners(lleno)),
        (f"6. caracteristicas + {etiqueta}", vis),
    ]


# ======================= FIGURA =======================
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()
fig.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.13, wspace=0.05, hspace=0.15)

ax_slider = fig.add_axes((0.07, 0.065, 0.40, 0.03))
slider = Slider(ax_slider, "frame", 0, max(n_frames - 1, 1), valinit=0, valstep=1)

ax_chk = fig.add_axes((0.50, 0.02, 0.09, 0.08))
chk = CheckButtons(ax_chk, ["corners", "harris"], [estado["corners"], estado["harris"]])
for t in chk.labels:
    t.set_fontsize(8)


def _tb(x, key, label):
    ax = fig.add_axes((x, 0.05, 0.04, 0.04))
    tb = TextBox(ax, label, initial=str(estado[key]))
    tb.on_submit(lambda s, k=key: _set(k, s))
    return tb


def _set(key, text):
    try:
        estado[key] = int(float(text))
    except ValueError:
        return
    redibujar()


tb_thr = _tb(0.655, "thr", "thr ")
tb_ko = _tb(0.715, "k_open", "open ")
tb_kc = _tb(0.775, "k_close", "close ")
tb_mc = _tb(0.855, "max_corners", "maxC ")
tb_q = _tb(0.915, "quality", "qual ")
tb_md = _tb(0.965, "min_dist", "dist ")


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
                 f"(qual = milesimas, {estado['quality']/1000:g})", fontsize=10)
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

redibujar()
plt.show()
cap.release()
