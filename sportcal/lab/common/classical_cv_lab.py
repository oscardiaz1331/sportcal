"""Laboratorio interactivo de vision clasica -- sport-agnostic.

Junta en una sola UI todas las piezas de vision clasica que se probaron para
hockey en sportcal/core/surface.py, sportcal/lab/hockey/fit_homography_lines.py y
training/line_explorer.py (ver CLAUDE.md seccion 4), pero SIN nada especifico
de hockey (ni rink.py, ni una H, ni un dataset etiquetado) -- solo toma un
video o imagen cualquiera. El objetivo es poder probar en minutos, sobre
video de OTRO deporte, si la misma estrategia (color/Lab -> region -> croma
local -> lineas/circulos por Hough o RANSAC -> restriccion geometrica) tiene
alguna oportunidad antes de invertir en construir una geometria + clases +
segmentacion para ese deporte.

Uso:
    uv pip install --python venv/Scripts/python.exe streamlit streamlit-image-coordinates   (una vez)
    venv/Scripts/python.exe -m streamlit run sportcal/lab/common/classical_cv_lab.py

Las pestañas 1, 2, 6 y 7 producen mascaras que se registran por nombre (MASKS);
las pestañas 3, 4 y 5 eligen cual usar como entrada (por defecto "2. binarizada").

Pon clips de otros deportes en sportcal/lab/common/samples/ (se detectan solos), o
escribe cualquier ruta de video/imagen a mano en la barra lateral.
"""
import importlib
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from sportcal.paths import ROOT
SAMPLES = Path(__file__).resolve().parent / "samples"

from sportcal.lab.common import color_spaces as CS
from sportcal.core import surface as P
from sportcal.core import fitting as RF

# Streamlit re-ejecuta este script pero NO recarga los modulos ya importados: sin
# esto, editar training/*.py exige reiniciar el servidor (AttributeError con lo
# recien anadido). Recargar solo re-ejecuta estos modulos, no sus dependencias.
for _m in (CS, P, RF):
    importlib.reload(_m)
fit_circle_center, fit_line_px, fit_line_ransac = (
    RF.fit_circle_center, RF.fit_line_px, RF.fit_line_ransac)

VIDEO_EXT = (".mp4", ".webm", ".mov", ".avi", ".mkv")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

st.set_page_config(page_title="CV lab (sport-agnostic)", layout="wide")


# ---------------------------------------------------------------- fuentes

def list_sources():
    SAMPLES.mkdir(exist_ok=True)
    out = []
    for base, label in ((SAMPLES, "samples/"), (ROOT, "raiz del proyecto/")):
        for ext in VIDEO_EXT + IMG_EXT:
            out += [(label + p.name, p) for p in sorted(base.glob("*" + ext))]
    return out


@st.cache_resource(show_spinner=False)
def open_video(path_str):
    cap = cv2.VideoCapture(path_str)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, n


def read_frame(path: Path, idx: int):
    if path.suffix.lower() in IMG_EXT:
        return cv2.imread(str(path))
    cap, n = open_video(str(path))
    idx = min(idx, max(0, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def norm01(x, lo=1, hi=99.5):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x.astype(np.float32) - a) / max(1e-6, b - a), 0, 1)


def overlay(base_bgr, mask, color=(0, 255, 120), alpha=0.45):
    over = base_bgr.copy()
    over[mask > 0] = color
    return cv2.addWeighted(over, alpha, base_bgr, 1 - alpha, 0)


# Registro de mascaras: cada pestaña que produce una mascara la registra aqui con
# nombre, y las que la consumen (pipeline, lineas, circulos, corredor) la eligen
# de la lista -- asi los filtros se encadenan en vez de estar cableados a la
# pestaña anterior.
MASKS = {}


def registra(nombre, m):
    MASKS[nombre] = (np.asarray(m) > 0).astype(np.uint8)


def elegir_mascara(etiqueta, key, defecto="2. binarizada"):
    nombres = list(MASKS)
    nombre = st.selectbox(etiqueta, nombres, key=key,
                          index=nombres.index(defecto) if defecto in nombres else 0)
    return nombre, MASKS[nombre]


# Mapa de color de la pestaña 1: nombres de canal por espacio y modelo -> primitivas
NOMBRES_CANALES = {"lab": ("L", "a", "b"), "luv": ("L", "u", "v"),
                   "yuv": ("Y", "U", "V"), "hsv": ("H", "S", "V")}


def zoom_lims(canales_px, par, lims):
    """Limites de los ejes ajustados a donde estan los pixeles (percentiles
    0.2-99.8 + 10% de margen, dentro del rango del canal): en el rango completo
    0-255 casi todo el hielo queda apelotonado en unas pocas unidades."""
    out = []
    for i, (lo_c, hi_c) in zip(par, lims):
        lo, hi = np.percentile(canales_px[:, i], [0.2, 99.8])
        hi = max(hi, lo + 8.0)
        pad = 0.10 * (hi - lo)
        out.append((max(lo_c, float(lo - pad)), min(hi_c, float(hi + pad))))
    return tuple(out)


def primitivas_modelo(over, esp_v, par):
    """(elipses, rects, nota) del modelo `over` proyectado en el plano `par` del
    espacio `esp_v`, en el formato de CS.fig_mapa_mascara. Solo se dibuja si el
    modelo vive en ese espacio (el umbral HSV no tiene forma en Lab, etc.)."""
    if over is None:
        return [], [], "el metodo cayo al umbral HSV (semilla insuficiente): sin modelo que dibujar"
    if over["espacio"] != esp_v:
        return [], [], "el modelo vive en {}: cambia el espacio del mapa para verlo".format(
            over["espacio"])
    rango = lambda i: (0, 179) if (esp_v == "hsv" and i == 0) else (0, 255)
    el, rc, nota = [], [], None
    if over["kind"] == "hsv":
        (x0, x1), (y0, y1) = (over["cons"].get(i, rango(i)) for i in par)
        rc.append((x0, x1, y0, y1, "crimson", "umbral HSV"))
        fuera = [NOMBRES_CANALES["hsv"][i] for i in over["cons"] if i not in par]
        if fuera:
            nota = "el umbral HSV tambien restringe {}: no se ve en este plano".format(
                ", ".join(fuera))
    elif over["kind"] == "gmm":
        p = list(par)
        for c in range(len(over["means"])):
            hielo = c in over["sel"]
            el.append((over["means"][c, p], np.diag(over["var"][c, p]), 2.0,
                       "limegreen" if hielo else "gray",
                       "comp {} w={:.2f}{}".format(c, over["w"][c], " (hielo)" if hielo else "")))
        nota = "GMM: elipses a 2 sigma; verde = componentes elegidos como hielo"
    else:
        idx, mu, cov, r = over["idx"], over["mu"], over["cov"], float(np.sqrt(over["chi2"]))
        en = [i for i in par if i in idx]
        if len(en) == 2:
            a, b = idx.index(par[0]), idx.index(par[1])
            el.append((mu[[a, b]], cov[np.ix_([a, b], [a, b])], r, "crimson",
                       "gaussiana (chi2={:g})".format(over["chi2"])))
        elif len(en) == 1:
            j = idx.index(en[0])
            lo, hi = mu[j] - r * np.sqrt(cov[j, j]), mu[j] + r * np.sqrt(cov[j, j])
            if par[0] == en[0]:
                rc.append((lo, hi, *rango(par[1]), "crimson", "banda de la gaussiana"))
            else:
                rc.append((*rango(par[0]), lo, hi, "crimson", "banda de la gaussiana"))
            nota = "la gaussiana solo usa un canal de este plano: se ve como banda"
        else:
            nota = "la gaussiana no usa ninguno de estos canales: no restringe este plano"
    return el, rc, nota


# ---------------------------------------------------------------- barra lateral

st.sidebar.title("Fuente")
sources = list_sources()
opciones = [lbl for lbl, _ in sources] + ["(ruta manual)"]
sel = st.sidebar.selectbox("Video / imagen", opciones, index=0 if opciones else None)
if sel == "(ruta manual)":
    ruta = st.sidebar.text_input("Ruta completa", "")
    path = Path(ruta) if ruta else None
else:
    path = dict(sources).get(sel)

if path is None or not path.exists():
    st.info("Pon un video o imagen en `sportcal/lab/common/samples/`, o escribe una ruta manual "
            "en la barra lateral, para empezar.")
    st.stop()

es_video = path.suffix.lower() in VIDEO_EXT
idx = 0
if es_video:
    _, n_frames = open_video(str(path))
    st.sidebar.caption("{} frames".format(n_frames))
    idx = st.sidebar.number_input("Frame", 0, max(0, n_frames - 1), 0, step=1)
    c1, c2 = st.sidebar.columns(2)
    if c1.button("<< -10"):
        idx = max(0, idx - 10)
    if c2.button(">> +10"):
        idx = min(n_frames - 1, idx + 10)

img = read_frame(path, int(idx))
if img is None:
    st.error("No se pudo leer ese frame/imagen.")
    st.stop()
h, w = img.shape[:2]
sig = w / 1920.0   # mismo factor de escala que el resto del repo, para que los
                    # valores por defecto (pensados a 1920px) se adapten al video real
st.sidebar.caption("{}x{}  (sig={:.2f})".format(w, h, sig))

tabs = st.tabs(["1. Color / region", "2. Croma y cresta", "3. Lineas",
                "4. Circulos / elipses", "5. Corredor geometrico",
                "6. Pixel y espacios de color", "7. Pipeline de mascaras"])
# El codigo va en orden de dependencia (1, 2, 6, 7, 3, 4, 5) para que 3-5 puedan
# consumir lo que producen 6 y 7; el orden de las pestañas en pantalla no cambia.

# ------------------------------------------------------------- 1. color/region

with tabs[0]:
    st.caption("Segmentacion de la superficie de juego, sin geometria. "
               "El deporte decide que es 'la superficie': hockey = hielo (claro y "
               "neutro), futbol = cesped (VERDE, zona verde del mapa de color). "
               "Otras superficies: usa HSV fijo y ajusta a mano.")
    surface = st.radio("Superficie", list(P.SURFACES), horizontal=True, key="surface",
                       format_func=lambda d: {"ice": "hockey (hielo)",
                                              "grass": "futbol (cesped)"}[d])
    metodo = st.radio("Metodo", ["HSV fijo", "GMM en Lab", "Gaussiana robusta",
                                  "Mejor por solidez (auto)"], horizontal=True)
    col1, col2 = st.columns(2)
    # `over` describe el modelo que decidio la mascara, para dibujarlo en el mapa
    # de color de abajo (espacio donde vive + parametros).
    if metodo == "HSV fijo" and surface == "grass":
        h_rng = col1.slider("H (tono verde)", 0, 179, P.GRASS_HUE, key="cesped_h")
        s_min = col2.slider("S min", 0, 255, P.GRASS_S_MIN, key="cesped_s")
        v_min = col2.slider("V min", 0, 255, P.GRASS_V_MIN, key="cesped_v")
        ice = P.grass_hsv(img, h_rng[0], h_rng[1], s_min, v_min)
        over = {"kind": "hsv", "espacio": "hsv", "idx": (0, 1, 2),
                "cons": {0: tuple(h_rng), 1: (s_min, 255), 2: (v_min, 255)}}
    elif metodo == "HSV fijo":
        s_max = col1.slider("S max", 0, 255, 60)
        v_min = col2.slider("V min", 0, 255, 180)
        ice = P.ice_hsv(img, s_max=s_max, v_min=v_min)
        over = {"kind": "hsv", "espacio": "hsv", "idx": (1, 2),
                "cons": {1: (0, s_max), 2: (v_min, 255)}}
    elif metodo == "GMM en Lab":
        k = col1.slider("componentes (k)", 2, 10, 5)
        with st.spinner("ajustando GMM..."):
            ice, info = P.gmm_surface(img, k=k, surface=surface)
        if info:
            st.caption("L={}  chroma={}  peso={}".format(info["L"], info["chroma"], info["w"]))
        over = {"kind": "gmm", "espacio": "lab", "idx": (0, 1, 2), **info} if info else None
    elif metodo == "Gaussiana robusta":
        chi2 = col1.slider("chi2 (tolerancia)", 2.0, 20.0, 9.0)
        espacio = col2.radio("espacio de color", list(P.COLOR_SPACES), horizontal=True)
        letras = P.COLOR_SPACES[espacio][1]
        sel_canales = st.multiselect("canales de la gaussiana", list(letras),
                                     default=list(letras), key="robust_ch_" + espacio)
        if sel_canales:
            ice, modelo = P.robust_surface(img, chi2=chi2, space=espacio,
                                          channels=sel_canales, info=True, surface=surface)
            if len(sel_canales) < 3:
                st.caption("con {} canal(es) el mismo chi2 es mas generoso "
                           "(grados de libertad = canales)".format(len(sel_canales)))
        else:
            st.warning("elige al menos un canal")
            ice, modelo = P.robust_surface(img, chi2=chi2, info=True, surface=surface)
        over = {"kind": "robust", "espacio": modelo["space"], **modelo} if modelo else None
    else:
        ice_h = P.hsv_threshold(img, surface)
        ice_g, info = P.gmm_surface(img, surface=surface)
        sh, sg = P.solidity(P.play_region(ice_h)), P.solidity(P.play_region(ice_g))
        st.caption("solidez HSV={:.3f}   GMM={:.3f}   -> elegido {}".format(
            sh, sg, "GMM" if sg >= sh else "HSV"))
        if sg >= sh and info:
            ice = ice_g
            over = {"kind": "gmm", "espacio": "lab", "idx": (0, 1, 2), **info}
        else:
            ice = ice_h
            if surface == "grass":
                over = {"kind": "hsv", "espacio": "hsv", "idx": (0, 1, 2),
                        "cons": {0: P.GRASS_HUE, 1: (P.GRASS_S_MIN, 255), 2: (P.GRASS_V_MIN, 255)}}
            else:
                over = {"kind": "hsv", "espacio": "hsv", "idx": (1, 2),
                        "cons": {1: (0, P.ICE_S_MAX), 2: (P.ICE_V_MIN, 255)}}

    region = P.play_region(ice)
    registra("1. hielo (color crudo)", ice)
    registra("1. region", region)
    st.caption("region tras cerrar+rellenar: {:.1f}% del frame, solidez {:.3f}".format(
        100 * region.mean(), P.solidity(region)))
    c1, c2, c3 = st.columns(3)
    c1.image(to_rgb(overlay(img, ice, (0, 180, 255))), caption="mascara de color cruda")
    c2.image(to_rgb(overlay(img, region, (0, 255, 120))), caption="region (cerrada + rellena)")

    # ---- mapa de color: como el modelo del metodo parte los pixeles de ESTE frame
    with c3:
        espacios_mapa = list(P.COLOR_SPACES)
        def_esp = over["espacio"] if over else "lab"
        esp_v = st.selectbox("espacio del mapa", espacios_mapa,
                             index=espacios_mapa.index(def_esp),
                             key="mapa_esp_{}_{}".format(metodo, def_esp))
        nombres_c = NOMBRES_CANALES[esp_v]
        pares = list(itertools.combinations(range(3), 2))
        usados = set(over["idx"]) if over else {1, 2}
        def_par = max(pares, key=lambda p: (len(set(p) & usados), pares.index(p)))
        par = st.selectbox("plano", pares, index=pares.index(def_par),
                           format_func=lambda p: "{} vs {}".format(nombres_c[p[0]], nombres_c[p[1]]),
                           key="mapa_par_{}_{}_{}".format(metodo, def_esp, esp_v))
        f = min(1.0, (150000 / (img.shape[0] * img.shape[1])) ** 0.5)
        chico = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        m_chico = cv2.resize(ice, (chico.shape[1], chico.shape[0]), interpolation=cv2.INTER_NEAREST)
        canales_px = cv2.cvtColor(chico, P.COLOR_SPACES[esp_v][0]).reshape(-1, 3).astype(np.float32)
        lims = tuple((0, 179) if (esp_v == "hsv" and i == 0) else (0, 255) for i in par)
        if st.checkbox("zoom a los datos", value=True, key="mapa_zoom"):
            lims = zoom_lims(canales_px, par, lims)
        elipses, rects, nota = primitivas_modelo(over, esp_v, par)
        st.pyplot(CS.fig_mapa_mascara(canales_px, m_chico.ravel(), par, nombres_c, lims,
                                      elipses, rects), clear_figure=True)
        if nota:
            st.caption(nota)

# ------------------------------------------------------------- 2. croma/cresta

with tabs[1]:
    st.caption("Desviacion de color/luz respecto al entorno LOCAL (no absoluto) -- "
               "esto es lo que de verdad localiza lineas pintadas, sea cual sea "
               "el balance de blancos o la iluminacion del pabellon/estadio.")
    win = st.slider("ventana de mediana local (px a 1920)", 15, 151, 61, step=2)
    LAB_SEP = "Lab separado (a+ a- b+ b- L+ L-)"
    canal = st.radio("canal", ["a (verde<->rojo)", "b (azul<->amarillo)",
                                "L (oscuro<->claro)", "magnitud combinada (a,b)", LAB_SEP],
                      horizontal=True)
    win_px = max(11, int(win * sig) | 1)
    da, db = P.local_chroma(img, win=win_px)
    dl = P.local_l(img, win=win_px)

    recortar = st.checkbox("recortar a la region de la pestaña 1", value=True)
    usar_cresta = st.checkbox("aplicar cresta multiescala (ridge)")
    escalas = (1.5, 3.0, 6.0, 10.0)
    if usar_cresta:
        escalas_str = st.text_input("escalas", "1.5,3.0,6.0,10.0")
        escalas = tuple(float(s) for s in escalas_str.split(",") if s.strip())

    def recorta_y_cresta(r):
        """Recorte a la region (pestaña 1) y cresta opcional; devuelve |r|."""
        r = r * (region > 0) if recortar else r
        return P.ridge(np.abs(r).astype(np.float32), scales=escalas) if usar_cresta else np.abs(r)

    if canal != LAB_SEP:
        resp = {"a (verde<->rojo)": da, "b (azul<->amarillo)": db,
                "L (oscuro<->claro)": dl, "magnitud combinada (a,b)": np.hypot(da, db)}[canal]
        resp_final = recorta_y_cresta(resp)

        thr_pct = st.slider("umbral (percentil) para binarizar", 50.0, 99.9, 97.0)
        thr = np.percentile(resp_final, thr_pct)
        bin_mask = (resp_final > thr).astype(np.uint8)
        registra("2. binarizada", bin_mask)

        c1, c2 = st.columns(2)
        c1.image(norm01(resp_final), caption="respuesta ({})".format(canal), clamp=True)
        c2.image(to_rgb(overlay(img, bin_mask, (255, 60, 60))), caption="binarizada (p{:.0f})".format(thr_pct))
    else:
        st.caption("Cada semieje de la desviacion local por separado (solo la parte positiva de "
                   "cada uno): a+ rojizo, a- verdoso, b+ amarillento, b- azulado, L+ mas claro "
                   "que el entorno, L- mas oscuro. Marca 'filtrar' en los que quieras y fija su "
                   "rango de valor; los filtrados activos se combinan abajo y el resultado "
                   "pasa a ser '2. binarizada' (y cada filtro suelto queda como '2. a+', '2. b-'... "
                   "para el pipeline de la pestaña 7).")
        k1, k2, k3 = st.columns(3)
        modo_color = k1.radio("colorear", ["color del eje", "pixeles de la imagen"], horizontal=True)
        # a/b rondan 10 (p99.9) y L llega a 150-200 en el mismo frame: una escala unica
        # dejaba a/b casi negros y L saturado.
        esc_ab = k2.slider("escala visual a/b (valor = brillo maximo)", 2.0, 100.0, 15.0)
        esc_l = k2.slider("escala visual L", 5.0, 255.0, 60.0)
        comb = k3.radio("combinar filtros activos", ["AND", "OR"], horizontal=True)

        CANALES = [("a+", "rojo", da, 1, (255, 60, 60)), ("a-", "verde", da, -1, (60, 220, 60)),
                   ("b+", "amarillo", db, 1, (255, 220, 0)), ("b-", "azul", db, -1, (60, 100, 255)),
                   ("L+", "mas claro", dl, 1, (255, 255, 255)), ("L-", "mas oscuro", dl, -1, (170, 170, 170))]
        activos = []
        for fila in range(2):
            cols = st.columns(3)
            for j in range(3):
                i = 3 * fila + j
                nombre, desc, d, signo, col_rgb = CANALES[i]
                v = recorta_y_cresta(np.maximum(signo * d, 0)).astype(np.float32)
                with cols[j]:
                    ph = st.empty()   # la imagen se pinta tras leer el filtro, para atenuar lo descartado
                    usar = st.checkbox("filtrar " + nombre, key="lab_use%d" % i)
                    lo, hi = st.slider("valor min-max " + nombre, 0.0, 255.0, (5.0, 255.0), 0.5,
                                       key="lab_rng%d" % i, disabled=not usar, label_visibility="collapsed")
                    pos = v[v > 0]
                    p97, p999 = np.percentile(pos, [97, 99.9]) if pos.size else (0.0, 0.0)
                    st.caption("sobre los pixeles >0:  p97={:.1f}  p99.9={:.1f}".format(p97, p999))

                    a = np.clip(v / (esc_l if nombre[0] == "L" else esc_ab), 0, 1)[..., None]
                    if modo_color.startswith("color"):
                        vis = a * np.array(col_rgb, np.float32)
                    else:
                        vis = a * to_rgb(img).astype(np.float32)
                    cap = "{} ({})".format(nombre, desc)
                    if usar:
                        m = ((v >= lo) & (v <= hi) & (v > 0)).astype(np.uint8)
                        activos.append((nombre, m))
                        registra("2. " + nombre, m)
                        vis = vis * np.where(m[..., None] > 0, 1.0, 0.3)
                        cap += "  -- pasa {:.2f}% del frame".format(100 * m.mean())
                    ph.image(vis.astype(np.uint8), caption=cap)

        if activos:
            ms = [m for _, m in activos]
            bin_mask = np.bitwise_and.reduce(ms) if comb == "AND" else np.bitwise_or.reduce(ms)
            cap = (" {} ".format(comb)).join(n for n, _ in activos)
        else:
            bin_mask = np.zeros((h, w), np.uint8)
            cap = "ningun filtro activo -> mascara vacia"
        registra("2. binarizada", bin_mask)
        st.image(to_rgb(overlay(img, bin_mask, (255, 60, 60))),
                 caption="binarizada: {}  ({:.2f}% del frame)".format(cap, 100 * bin_mask.mean()))

# ------------------------------------------------------------- 6. pixel y espacios

with tabs[5]:
    st.caption("Haz clic en la imagen para elegir un pixel: se ve su valor en HSV / Lab / "
               "YCbCr / Luv y donde cae (estrella dorada) en la nube de densidad de cada "
               "espacio. Con las tolerancias se genera una mascara 'similar al pixel' "
               "(caja roja en la nube) que se puede encadenar en las demas pestañas.")
    conv = CS.convertir_todo(img)
    if "px" not in st.session_state:
        st.session_state["px"] = None
    px = st.session_state["px"]
    if px is not None:   # el frame/fuente pudo cambiar de tamaño desde el ultimo clic
        px = (min(px[0], w - 1), min(px[1], h - 1))

    cc1, cc2, cc3 = st.columns(3)
    fondo = cc1.radio("fondo", ["original", "color real (S,V a tope)"], horizontal=True)
    over_nombre = cc2.selectbox("superponer mascara", ["(ninguna)"] + list(MASKS))
    r_parche = cc3.slider("parche (radio px, mediana)", 0, 15, 0)

    ref_uni = st.columns(2)
    uni_nombre = ref_uni[0].selectbox("universo de las nubes", ["(todo el frame)"] + list(MASKS))
    uni_modo = ref_uni[1].radio("universo: dentro/fuera", ["dentro de la mascara", "fuera de la mascara"], horizontal=True,
                                 disabled=uni_nombre == "(todo el frame)", label_visibility="collapsed")
    universo = np.ones((h, w), bool)
    if uni_nombre != "(todo el frame)":
        universo = MASKS[uni_nombre] > 0
        if uni_modo.startswith("fuera"):
            universo = ~universo

    st.markdown("**Mascara similar al pixel**")
    ce1, ce2, ce3, ce4 = st.columns(4)
    esp_clave = ce1.selectbox("espacio", list(CS.ESPACIOS), format_func=lambda k: CS.ESPACIOS[k]["nombre"])
    esp = CS.ESPACIOS[esp_clave]
    tol1 = ce2.slider("tol. " + esp["etiquetas"][0].split(" ")[0], 1, 90 if esp["circ"][0] else 128, 10)
    tol2 = ce3.slider("tol. " + esp["etiquetas"][1].split(" ")[0], 1, 128, 20)
    usar_lum = ce4.checkbox("limitar tambien " + esp["lum_nombre"])
    tol_lum = ce4.slider("tol. " + esp["lum_nombre"], 1, 128, 40, disabled=not usar_lum)

    base = img if fondo == "original" else np.ascontiguousarray(CS.imagen_color_real(img)[..., ::-1])
    vis6 = base.copy()
    mask_px = None
    vals = None
    if px is not None:
        vals = CS.valores_pixel(conv, px[0], px[1], r_parche)
        ref = vals[esp_clave]
        mask_px = CS.mascara_similar(conv, esp_clave, ref, tol1, tol2, tol_lum if usar_lum else None)
        registra("6. similar al pixel", mask_px)
    if over_nombre != "(ninguna)":
        vis6 = overlay(vis6, MASKS[over_nombre], (255, 60, 60))
    if px is not None:
        cv2.drawMarker(vis6, px, (0, 255, 255), cv2.MARKER_CROSS, int(40 * sig), max(1, int(2 * sig)))
        cv2.circle(vis6, px, max(r_parche, 3), (0, 0, 0), max(1, int(2 * sig)))

    st.caption("Clic = elegir pixel (la imagen se muestra reducida; el clic se reescala al frame real).")
    click = streamlit_image_coordinates(to_rgb(vis6), key="click_px", width=min(w, 1000))
    if click is not None:
        nuevo = (int(click["x"] * w / click["width"]), int(click["y"] * h / click["height"]))
        nuevo = (min(max(nuevo[0], 0), w - 1), min(max(nuevo[1], 0), h - 1))
        if nuevo != st.session_state["px"]:
            st.session_state["px"] = nuevo
            st.rerun()

    if px is None:
        st.info("Sin pixel elegido todavia.")
    else:
        lineas = ["px ({},{}){}".format(px[0], px[1], "  (mediana de parche {0}x{0})".format(2 * r_parche + 1) if r_parche else "")]
        for k, s_ in CS.ESPACIOS.items():
            a_, b_, c_ = vals[k]
            nombres = {"hsv": "HSV    H={:3d}  S={:3d}  V={:3d}", "lab": "Lab    L={:3d}  a={:3d}  b={:3d}",
                       "ycc": "YCbCr  Y={:3d}  Cr={:3d}  Cb={:3d}", "luv": "Luv    L={:3d}  u={:3d}  v={:3d}"}
            lineas.append(nombres[k].format(a_, b_, c_))
        lineas.append("universo: {:.1f}% del frame   mascara similar: {:.2f}% del frame".format(
            100 * universo.mean(), 100 * mask_px.mean()))
        st.code("\n".join(lineas))

    paso = 4   # submuestreo solo para las nubes de densidad, no para el pixel
    uni_sub = universo[::paso, ::paso].ravel()
    cols_nube = st.columns(2)
    for i, (k, s_) in enumerate(CS.ESPACIOS.items()):
        e1, e2 = s_["ejes"]
        c_ = conv[k][::paso, ::paso].reshape(-1, 3)[uni_sub]
        punto = (vals[k][e1], vals[k][e2]) if vals else None
        tols = (tol1, tol2) if (vals and k == esp_clave) else None
        cols_nube[i % 2].pyplot(CS.fig_espacio(k, c_[:, e1], c_[:, e2], punto, tols), clear_figure=True)
    h_uni = conv["hsv"][::paso, ::paso, 0].ravel()[uni_sub]
    st.pyplot(CS.fig_espectro_hue(h_uni, vals["hsv"][0] if vals else None), clear_figure=True)

# ------------------------------------------------------------- 7. pipeline de mascaras

with tabs[6]:
    st.caption("Encadena mascaras: parte de una, y en cada etapa la combina con otra "
               "(AND / OR / AND NOT / XOR). Despues, morfologia y limpieza de manchas "
               "pequeñas. El resultado se registra como '7. pipeline' y se puede "
               "elegir como entrada en las pestañas 3, 4 y 5.")
    fuentes = ["(frame completo)"] + list(MASKS)
    defectos = ["1. region", "2. binarizada", "6. similar al pixel", "(frame completo)", "(frame completo)"]
    n_et = st.slider("etapas", 1, 5, 2)
    acc = None
    pasos = []
    for i in range(n_et):
        cols = st.columns([2, 4, 1])
        if i == 0:
            cols[0].markdown("**entrada**")
            op = None
        else:
            op = cols[0].selectbox("operacion", ["AND", "OR", "AND NOT", "XOR"], key="pipe_op%d" % i)
        d = defectos[i] if defectos[i] in fuentes else fuentes[0]
        src = cols[1].selectbox("mascara", fuentes, index=fuentes.index(d), key="pipe_src%d" % i)
        inv = cols[2].checkbox("NOT", key="pipe_inv%d" % i)
        m = np.ones((h, w), np.uint8) if src == "(frame completo)" else MASKS[src]
        if inv:
            m = 1 - m
        if acc is None:
            acc = m.copy()
        elif op == "AND":
            acc = acc & m
        elif op == "OR":
            acc = acc | m
        elif op == "AND NOT":
            acc = acc & (1 - m)
        else:
            acc = acc ^ m
        pasos.append(("{}{}{}".format("" if op is None else op + " ", "NOT " if inv else "", src), acc.copy()))

    cm1, cm2, cm3 = st.columns(3)
    morf = cm1.selectbox("morfologia", ["ninguna", "apertura", "cierre", "dilatar", "erosionar"])
    k_px = cm2.slider("kernel (px a 1920)", 1, 41, 5, disabled=morf == "ninguna")
    area_min = cm3.slider("area minima de mancha (px a 1920)", 0, 5000, 0)
    out = acc.copy()
    if morf != "ninguna":
        ks = max(1, int(k_px * sig))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        op_cv = {"apertura": cv2.MORPH_OPEN, "cierre": cv2.MORPH_CLOSE,
                 "dilatar": cv2.MORPH_DILATE, "erosionar": cv2.MORPH_ERODE}[morf]
        out = cv2.morphologyEx(out, op_cv, ker)
        pasos.append(("{} k={}".format(morf, ks), out.copy()))
    if area_min > 0:
        n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        ok = np.zeros(n_cc, bool)
        ok[1:] = stats[1:, cv2.CC_STAT_AREA] >= area_min * sig * sig
        out = ok[lab].astype(np.uint8)
        pasos.append(("area >= {}".format(area_min), out.copy()))
    registra("7. pipeline", out)

    st.markdown("**Etapas**")
    per_row = 3
    for j in range(0, len(pasos), per_row):
        cols = st.columns(per_row)
        for col, (nombre, m) in zip(cols, pasos[j:j + per_row]):
            col.image(to_rgb(overlay(img, m, (255, 60, 60))),
                      caption="{}  ({:.2f}%)".format(nombre, 100 * m.mean()))
    st.image(to_rgb(overlay(img, out, (0, 255, 120))),
             caption="resultado: 7. pipeline ({:.2f}% del frame)".format(100 * out.mean()))
    st.image(out * 255, caption="mascara binaria final (blanco = 1)", clamp=True)

# ------------------------------------------------------------- 3. lineas

with tabs[2]:
    st.caption("Sobre la mascara de entrada (por defecto la binaria de la pestaña 2): "
               "Hough clasico (sin restriccion, se ahoga en candidatos) frente a ajuste "
               "robusto dado que ya sabes que es una mascara de una sola clase/tira.")
    nom3, ent3 = elegir_mascara("Entrada", "in3")
    m8 = (ent3 * 255).astype(np.uint8)

    st.subheader("Hough clasico (sin restriccion)")
    c1, c2, c3 = st.columns(3)
    h_thr = c1.slider("threshold", 5, 300, 60)
    h_minlen = c2.slider("minLineLength", 5, 400, int(40 * sig))
    h_gap = c3.slider("maxLineGap", 1, 100, int(15 * sig))
    lineas = cv2.HoughLinesP(m8, 1, np.pi / 360, h_thr, minLineLength=h_minlen, maxLineGap=h_gap)
    vis = img.copy()
    n_h = 0 if lineas is None else len(lineas)
    if lineas is not None:
        for x1, y1, x2, y2 in lineas.reshape(-1, 4):
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    st.image(to_rgb(vis), caption="{} segmentos detectados".format(n_h))

    st.subheader("Ajuste robusto (sabiendo que la mascara es 1 recta / 1 recta dominante)")
    c1, c2 = st.columns(2)
    modo = c1.radio("variante", ["fit_line_px (recta limpia unica)",
                                  "fit_line_ransac (mascara mixta, p.ej. perimetro)"])
    if modo.startswith("fit_line_ransac"):
        tol = c2.slider("tolerancia RANSAC (px a 1920)", 1.0, 15.0, 4.0)
        recta = fit_line_ransac(ent3.astype(bool), tol_px=tol * sig)
    else:
        recta = fit_line_px(ent3.astype(bool))
    vis2 = img.copy()
    if recta is not None:
        p, q, r = recta
        # dibuja la recta p*x+q*y+r=0 cruzando el frame
        pts = []
        for x in (0, w):
            if abs(q) > 1e-6:
                pts.append((x, int(-(p * x + r) / q)))
        for y in (0, h):
            if abs(p) > 1e-6:
                pts.append((int(-(q * y + r) / p), y))
        pts = [pt for pt in pts if -w <= pt[0] <= 2 * w and -h <= pt[1] <= 2 * h]
        if len(pts) >= 2:
            cv2.line(vis2, pts[0], pts[-1], (0, 255, 255), 3, cv2.LINE_AA)
        st.image(to_rgb(vis2), caption="recta ajustada: {:.4f}x + {:.4f}y + {:.1f} = 0".format(p, q, r))
    else:
        st.image(to_rgb(vis2), caption="sin ajuste valido (poca masa o sin inliers suficientes)")

# ------------------------------------------------------------- 4. circulos/elipses

with tabs[3]:
    st.caption("Circulos/elipses (centro de saque, aros, etc.) por Hough sobre la "
               "imagen y por ajuste de elipse sobre la mascara binaria, con el "
               "filtro robusto de 'arco parcial recortado por el borde'.")

    st.subheader("Hough circles (sobre gris)")
    c1, c2, c3 = st.columns(3)
    dp = c1.slider("dp", 1.0, 3.0, 1.2)
    min_dist = c2.slider("minDist", 10, 500, int(100 * sig))
    p2 = c3.slider("param2 (sensibilidad)", 10, 150, 40)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circ = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                             param1=100, param2=p2,
                             minRadius=int(20 * sig), maxRadius=int(300 * sig))
    vis3 = img.copy()
    n_c = 0 if circ is None else circ.shape[1]
    if circ is not None:
        for x, y, rad in circ[0]:
            cv2.circle(vis3, (int(x), int(y)), int(rad), (255, 0, 255), 2)
    st.image(to_rgb(vis3), caption="{} circulos detectados".format(n_c))

    st.subheader("Ajuste de elipse sobre la mascara de entrada")
    nom4, ent4 = elegir_mascara("Entrada", "in4")
    vis4 = img.copy()
    cnts, _ = cv2.findContours(ent4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    grande = max(cnts, key=cv2.contourArea) if cnts else None
    if grande is not None and len(grande) >= 5:
        elipse = cv2.fitEllipse(grande)
        cv2.ellipse(vis4, elipse, (0, 255, 0), 2)
    centro = fit_circle_center(ent4.astype(bool), w, h)
    if centro is not None:
        cv2.drawMarker(vis4, tuple(centro.astype(int)), (0, 0, 255),
                        cv2.MARKER_CROSS, 24, 2)
    st.image(to_rgb(vis4), caption="verde: fitEllipse sobre el contorno mayor.  "
             "cruz roja: centro robusto (None si el arco visible es un gajo estrecho "
             "o toca el borde -- {})".format("aceptado" if centro is not None else "RECHAZADO"))

# ------------------------------------------------------------- 5. corredor geometrico

with tabs[4]:
    st.caption("Sin plantilla todavia para este deporte: define a mano un segmento "
               "aproximado (como fracción del frame) y compara Hough libre vs. Hough "
               "restringido a un corredor alrededor de ese segmento -- la misma idea "
               "que el 'corredor geometrico' de CLAUDE.md (seccion 4), pero con la "
               "geometria puesta a mano en vez de proyectada por una H.")
    c1, c2, c3, c4 = st.columns(4)
    fx1 = c1.slider("x1 (frac.)", 0.0, 1.0, 0.1)
    fy1 = c2.slider("y1 (frac.)", 0.0, 1.0, 0.5)
    fx2 = c3.slider("x2 (frac.)", 0.0, 1.0, 0.9)
    fy2 = c4.slider("y2 (frac.)", 0.0, 1.0, 0.5)
    ancho = st.slider("ancho del corredor (px a 1920)", 5, 200, 40)
    nom5, ent5 = elegir_mascara("Entrada", "in5")

    p1, p2pt = (int(fx1 * w), int(fy1 * h)), (int(fx2 * w), int(fy2 * h))
    corredor = np.zeros((h, w), np.uint8)
    cv2.line(corredor, p1, p2pt, 1, thickness=int(ancho * sig))
    m_libre = (ent5 * 255).astype(np.uint8)
    m_restr = (ent5 & corredor) * 255

    l_libre = cv2.HoughLinesP(m_libre, 1, np.pi / 360, 40, minLineLength=int(30 * sig), maxLineGap=int(15 * sig))
    l_restr = cv2.HoughLinesP(m_restr.astype(np.uint8), 1, np.pi / 360, 40, minLineLength=int(30 * sig), maxLineGap=int(15 * sig))
    n_libre = 0 if l_libre is None else len(l_libre)
    n_restr = 0 if l_restr is None else len(l_restr)

    c1, c2 = st.columns(2)
    vis5 = img.copy()
    if l_libre is not None:
        for x1, y1, x2, y2 in l_libre.reshape(-1, 4):
            cv2.line(vis5, (x1, y1), (x2, y2), (0, 0, 255), 1)
    c1.image(to_rgb(vis5), caption="Hough libre: {} candidatos".format(n_libre))

    vis6 = img.copy()
    cv2.line(vis6, p1, p2pt, (0, 255, 255), 1, cv2.LINE_AA)
    if l_restr is not None:
        for x1, y1, x2, y2 in l_restr.reshape(-1, 4):
            cv2.line(vis6, (x1, y1), (x2, y2), (0, 0, 255), 2)
    c2.image(to_rgb(vis6), caption="Hough en el corredor: {} candidatos ({}x menos)".format(
        n_restr, round(n_libre / max(1, n_restr), 1) if n_restr else "inf"))
