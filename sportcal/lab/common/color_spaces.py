"""Espacios de color para el laboratorio (classical_cv_lab.py): conversion,
valores de un pixel, mascara "similar a este pixel" y nubes de densidad.

Es la parte reutilizable de field_comparar_espacios.py (ese script es una app
de matplotlib con ventanas propias y argparse a nivel de modulo, no se puede
importar). Aqui las figuras son matplotlib.figure.Figure sueltas, para pintarlas
con st.pyplot sin tocar el estado global de pyplot.
"""
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle

# esquinas de referencia por espacio, calculadas con cv2.cvtColor sobre
# primarios puros sRGB (8-bit). HSV incluye "rojo" en H=0 Y H=179 porque el
# rojo envuelve en los dos extremos de la rueda de hue.
_CORNERS = {
    "hsv": {"rojo (H=0)": (0, 255), "rojo (H=179)": (179, 255),
            "verde": (60, 255), "azul": (120, 255)},
    "lab": {"rojo": (208, 195), "verde": (42, 211), "azul": (207, 20)},
    "ycc": {"rojo": (255, 85), "verde": (21, 43), "azul": (107, 255)},
    "luv": {"rojo": (222, 173), "verde": (37, 241), "azul": (90, 10)},
}
_COLOR_PUNTO = {"rojo": "red", "verde": "green", "azul": "blue"}

# ejes: indices de los 2 canales que forman el plano de la nube; lum: canal de
# luminosidad, que se queda fuera del plano; circ: el eje es circular (H).
ESPACIOS = {
    "hsv": dict(nombre="HSV (H,S)", code=cv2.COLOR_BGR2HSV, ejes=(0, 1), lum=2,
                lum_nombre="V", etiquetas=("H", "S"), lims=((0, 179), (0, 255)),
                circ=(True, False), neutro=None),
    "lab": dict(nombre="Lab (a,b)", code=cv2.COLOR_BGR2Lab, ejes=(1, 2), lum=0,
                lum_nombre="L", etiquetas=("a (verde<->rojo)", "b (azul<->amarillo)"),
                lims=((0, 255), (0, 255)), circ=(False, False), neutro=(128, 128)),
    "ycc": dict(nombre="YCbCr (Cr,Cb)", code=cv2.COLOR_BGR2YCrCb, ejes=(1, 2), lum=0,
                lum_nombre="Y", etiquetas=("Cr (rojez)", "Cb (azulez)"),
                lims=((0, 255), (0, 255)), circ=(False, False), neutro=(128, 128)),
    "luv": dict(nombre="Luv (u,v)", code=cv2.COLOR_BGR2Luv, ejes=(1, 2), lum=0,
                lum_nombre="L", etiquetas=("u (verde<->rojo)", "v (azul<->amarillo)"),
                lims=((0, 255), (0, 255)), circ=(False, False), neutro=(96, 136)),
}

# tira de 180 colores puros (S=255, V=255), uno por cada H posible
_TIRA_HUE = cv2.cvtColor(
    np.dstack([np.arange(180, dtype=np.uint8),
               np.full(180, 255, np.uint8),
               np.full(180, 255, np.uint8)]),
    cv2.COLOR_HSV2RGB)[0].astype(np.float32) / 255.0


def convertir_todo(bgr):
    """{clave de ESPACIOS: imagen HxWx3 uint8 en ese espacio}."""
    return {k: cv2.cvtColor(bgr, s["code"]) for k, s in ESPACIOS.items()}


def valores_pixel(conv, x, y, r=0):
    """Valor (3 canales) del pixel en cada espacio; mediana de un parche de
    (2r+1)^2 si r>0 (para no depender del ruido de un unico pixel)."""
    out = {}
    for k, c in conv.items():
        h, w = c.shape[:2]
        parche = c[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
        out[k] = tuple(int(v) for v in np.median(parche.reshape(-1, 3), axis=0))
    return out


def mascara_similar(conv, clave, ref, tol1, tol2, tol_lum=None):
    """Pixeles cuyo plano (eje1, eje2) esta a <= tol1/tol2 del pixel de
    referencia (caja, no elipse, para poder ajustar cada eje por separado).
    tol_lum: si no es None, restringe tambien el canal de luminosidad.
    El eje H de HSV se compara circularmente (0 y 179 son vecinos)."""
    s = ESPACIOS[clave]
    c = conv[clave]
    m = np.ones(c.shape[:2], bool)
    for eje, tol, circ in zip(s["ejes"], (tol1, tol2), s["circ"]):
        d = np.abs(c[..., eje].astype(np.int16) - ref[eje])
        if circ:
            d = np.minimum(d, 180 - d)
        m &= d <= tol
    if tol_lum is not None:
        m &= np.abs(c[..., s["lum"]].astype(np.int16) - ref[s["lum"]]) <= tol_lum
    return m.astype(np.uint8)


def _tramos(centro, tol, lim, circular):
    """Intervalos del eje cubiertos por centro+-tol (parte en 2 si envuelve)."""
    lo, hi = centro - tol, centro + tol
    if circular and lo < lim[0]:
        return [(lim[0], hi), (lim[1] + 1 + lo, lim[1])]
    if circular and hi > lim[1]:
        return [(lo, lim[1]), (lim[0], hi - lim[1] - 1)]
    return [(lo, hi)]


def fig_espacio(clave, xs, ys, punto=None, tols=None):
    """Nube de densidad (log) del plano del espacio. punto=(v1,v2) marca el
    pixel elegido; tols=(t1,t2) dibuja la caja de tolerancia alrededor de el."""
    s = ESPACIOS[clave]
    (x0, x1), (y0, y1) = s["lims"]
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title(s["nombre"], fontsize=10)
    ax.set_xlabel(s["etiquetas"][0], fontsize=8)
    ax.set_ylabel(s["etiquetas"][1], fontsize=8)
    ax.tick_params(labelsize=8)

    hist, _, _ = np.histogram2d(xs, ys, bins=64, range=[[x0, x1], [y0, y1]])
    ax.imshow(np.log1p(hist).T, origin="lower", extent=(x0, x1, y0, y1),
              cmap="gray_r", aspect="auto")

    if s["neutro"] is not None:
        ax.plot(*s["neutro"], "+", color="black", markersize=10, markeredgewidth=1.5)
        ax.annotate("neutro", s["neutro"], fontsize=8, textcoords="offset points", xytext=(4, -12))
    else:
        ax.axhline(0, color="black", linewidth=1, linestyle=":")
        ax.annotate("neutro (S=0, cualquier H)", (x0, 3), fontsize=7.5)

    for nombre, (cx, cy) in _CORNERS[clave].items():
        color = _COLOR_PUNTO.get(nombre.split(" ")[0], "black")
        ax.plot(cx, cy, "o", color=color, markersize=6)
        ax.annotate(nombre, (cx, cy), fontsize=7.5, color=color,
                    textcoords="offset points", xytext=(5, 5))

    if punto is not None:
        if tols is not None:
            for a, b in _tramos(punto[0], tols[0], (x0, x1), s["circ"][0]):
                ax.add_patch(Rectangle((a, punto[1] - tols[1]), b - a, 2 * tols[1],
                                       fill=False, edgecolor="crimson", linewidth=1.5))
        ax.plot(*punto, "*", color="gold", markersize=16,
                markeredgecolor="black", markeredgewidth=0.9, zorder=5)
    fig.tight_layout()
    return fig


def fig_espectro_hue(h_vals, h_click=None):
    """Histograma de H coloreado con el color real de cada H."""
    fig = Figure(figsize=(10, 1.8))
    ax = fig.subplots()
    ax.set_xlim(0, 179)
    ax.set_title("Hue de los pixeles del universo, a S=255,V=255 (color real de cada H)", fontsize=10)
    ax.set_xlabel("H", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)
    conteos = np.bincount(h_vals, minlength=180)[:180].astype(np.float64)
    ax.bar(np.arange(180), conteos, width=1.0, color=_TIRA_HUE, edgecolor="none")
    ax.set_ylim(0, max(conteos.max(), 1) * 1.08)
    if h_click is not None:
        ax.axvline(h_click, color="black", linewidth=1.5)
        ax.plot(h_click, ax.get_ylim()[1] * 0.97, "v", color="black", markersize=9)
    fig.tight_layout()
    return fig


def imagen_color_real(bgr, excluir=None):
    """Cada pixel a su hue real con S=V=255 (color puro); `excluir` (bool) a blanco."""
    h = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 0]
    tope = np.dstack([h, np.full_like(h, 255), np.full_like(h, 255)])
    rgb = cv2.cvtColor(tope, cv2.COLOR_HSV2RGB)
    if excluir is not None:
        rgb[excluir] = 255
    return rgb


def fig_mapa_mascara(canales, mask, par, letras, lims, elipses=(), rects=(), bins=128):
    """Mapa de color de una MASCARA: nube 2D de los pixeles de `canales` (Nx3, ya
    en el espacio elegido) en el plano `par` (indices de 2 canales), en verde los
    que la mascara acepta y en rojo los que rechaza (densidad log, oscuro = mas
    pixeles; donde se mezclan sale marron). Encima, el modelo que decidio:
    elipses=[(centro, cov2x2, radio, color, etiqueta)] (el borde es la distancia
    de Mahalanobis == radio) y rects=[(x0, x1, y0, y1, color, etiqueta)].
    """
    (x0, x1), (y0, y1) = lims
    xs, ys = canales[:, par[0]], canales[:, par[1]]
    dentro = mask.astype(bool)
    rango = [[x0, x1], [y0, y1]]
    # ~1 unidad por bin como minimo: los canales son 8-bit y con zoom fuerte mas
    # bins que valores posibles dejan rayas vacias en la nube
    bins = [int(np.clip(round(x1 - x0), 16, bins)), int(np.clip(round(y1 - y0), 16, bins))]
    h_in = np.histogram2d(xs[dentro], ys[dentro], bins=bins, range=rango)[0]
    h_out = np.histogram2d(xs[~dentro], ys[~dentro], bins=bins, range=rango)[0]
    tope = np.log1p(max(h_in.max(), h_out.max(), 1.0))
    n_in, n_out = np.log1p(h_in) / tope, np.log1p(h_out) / tope
    rgb = 1.0 - n_in[..., None] * np.array([0.90, 0.35, 0.75])               - n_out[..., None] * np.array([0.15, 0.85, 0.85])
    rgb = np.clip(rgb, 0.0, 1.0)

    fig = Figure(figsize=(5.2, 4.6))
    ax = fig.subplots()
    ax.imshow(rgb.transpose(1, 0, 2), origin="lower", extent=(x0, x1, y0, y1), aspect="auto")
    ax.set_xlabel(letras[par[0]], fontsize=9)
    ax.set_ylabel(letras[par[1]], fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("verde = aceptado ({:.1f}%)   rojo = rechazado".format(100 * dentro.mean()),
                 fontsize=9)
    for centro, cov, radio, color, etiqueta in elipses:
        val, vec = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))          # eje mayor = autovector grande
        ax.add_patch(Ellipse(centro, 2 * radio * np.sqrt(val[1]), 2 * radio * np.sqrt(val[0]),
                             angle=ang, fill=False, edgecolor=color, linewidth=1.8, label=etiqueta))
        ax.plot(*centro, "+", color=color, markersize=9, markeredgewidth=1.6)
    for rx0, rx1, ry0, ry1, color, etiqueta in rects:
        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False,
                               edgecolor=color, linewidth=1.8, linestyle="--", label=etiqueta))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if elipses or rects:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    return fig
