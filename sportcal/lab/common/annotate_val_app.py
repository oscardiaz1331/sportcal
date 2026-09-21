"""Etiquetador por clics -- hockey (val NHL INDEPENDIENTE del DLT) y futbol.

Por cada frame: eliges un punto de la plantilla en el minimapa (arriba), haces clic
donde esta ese mismo punto en el frame (abajo), y repites con 4-8 puntos. Con >=4
puntos se ajusta la homografia, se dibuja la plantilla sobre el frame y el resto de
puntos aparece como marcadores naranja que arrastras a su sitio real (mucho mas rapido
que clicar cada uno desde cero):
si las lineas caen encima de las del hielo/cesped, guardas; si no, mira la tabla de
residuos y corrige el punto que falle. Entran tambien los frames DIFICILES (no se
filtra por si el DLT resuelve) -- por eso rompe la circularidad de valx.

    venv/Scripts/python.exe -m streamlit run sportcal/lab/common/annotate_val_app.py

Deporte (barra lateral):
  hockey  salida en datasets/hockeyrink_nhl_valh/{images,labels}/val (misma estructura
          que valx, se usa con --dataset hockeyrink_nhl_valh) + clicks.jsonl.
  futbol  plantilla FIFA 105x68 con 31 puntos (sportcal/lab/soccer/labeler.py); salida en
          datasets/soccer_labels/. Ademas del flujo de clics hay "Sugerir plantilla":
          el solver de sportcal/lab/soccer/field_solver.py (intersecciones y chamfer, ~15 s) propone
          una H y aparece como plantilla azul + marcadores naranja para arrastrar. En
          planos generales suele estar cerca; en planos cercanos es basura (puntuacion
          baja): usa Saltar.

Consejos de precision: prefiere puntos bien repartidos (esquinas opuestas, no todos
en la misma recta); en hockey la interseccion linea azul x valla y los puntos de faceoff
son los mas nitidos, con 6 puntos el error de las lineas medido con verdad conocida es
~2 px. Si el frame no es de pista/campo (primer plano, replay, publicidad): "Saltar".
La convencion de simetria (hockey: zona A a la izquierda; futbol: banda cercana abajo)
la aplica la app sola: la pista y el campo son simetricos.
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from sportcal.paths import ROOT

from sportcal.lab.hockey import click_labeler as CL
from sportcal.lab.soccer import camera_motion as SCM
from sportcal.lab.soccer import evaluation as SEV
from sportcal.lab.soccer import field_solver as SFD
from sportcal.lab.soccer import gradient as SGR
from sportcal.lab.soccer import labeler as SLB
from sportcal.sports.hockey import rink

# Streamlit re-ejecuta este script pero NO recarga los modulos ya importados: sin
# esto, editar sportcal/lab/hockey/click_labeler.py o cualquiera de los soccer_*.py da AttributeError
# en un servidor abierto hasta reiniciarlo. Recargarlos aqui cuesta milisegundos. El orden
# importa (cada uno depende del anterior): soccer_labeler usa soccer_eval, que usa soccer_field;
# soccer_eval se importa aqui aunque la app no lo llame, porque soccer_labeler lo carga perezoso
# y sin recargarlo se quedaria con la version vieja.
import importlib  # noqa: E402
for _m in (CL, SFD, SEV, SGR, SLB, SCM):
    importlib.reload(_m)

PARAMS = rink.RINK_NHL
TPL = rink.build_template(PARAMS)
W_FRAME = 1500
W_MINI = 860
SCALE_MINI = 13

st.set_page_config(page_title="Etiquetador por clics", layout="wide")


@st.cache_data(max_entries=4, show_spinner=False)
def get_frame(dep, v, i):
    return (SLB if dep == "futbol" else CL).lee_frame(v, i)


@st.cache_data(show_spinner=False)
def minimapa_hockey():
    img, to_img = rink.minimap_base(PARAMS, scale=SCALE_MINI, margin=20)
    return img, np.asarray(to_img(TPL))


@st.cache_data(show_spinner=False)
def minimapa_futbol():
    return SLB.minimapa()


@st.cache_data(max_entries=3, show_spinner="calculando el procesado de la mascara...")
def etapas_futbol(_fr, cid, elong_max):
    """Etapas de la mascara de este frame, cacheadas por id y regla (cada clic re-ejecuta el script)."""
    return SLB.etapas(_fr, elong_max)


@st.cache_resource(show_spinner="cargando el detector de personas (CPU)...")
def detector_personas():
    return SGR.modelo_personas()


@st.cache_data(max_entries=3, show_spinner="calculando respuesta y personas...")
def grad_base(_fr, cid):
    """Respuesta de linea y cajas de personas de este frame (lo caro, cacheado por id)."""
    return SGR.respuesta(_fr), SGR.personas(_fr, detector_personas())


@st.cache_data(max_entries=6, show_spinner="extrayendo rectas y elipse (unos segundos)...")
def grad_resultado(_fr, cid, metodo, canny_lo, canny_hi, tol_h, tol_v, usar_personas, cesped_min, pico_min, usar_region,
                   det_elipse, ed_params, ed_techo, fe_params, fe_techo, fld_params, fld_techo):
    resp, cajas = grad_base(_fr, cid)
    return SGR.procesa(_fr, None, metodo=metodo, canny=(canny_lo, canny_hi), tol_h=tol_h, tol_v=tol_v,
                       usar_personas=usar_personas, resp=resp, cajas=cajas, cesped_min=cesped_min, pico_min=pico_min,
                       usar_region=usar_region, det_elipse=det_elipse, ed_params=ed_params, ed_techo=ed_techo,
                       fe_params=fe_params, fe_techo=fe_techo, fld_params=fld_params, fld_techo=fld_techo)


@st.cache_data(max_entries=6, show_spinner="siguiendo puntos de la grada (KLT adelante y atras + RANSAC)...")
def klt_resultado(_fr0, _fr1, cid, gap, model, max_corners, quality, min_distance, win, levels, max_fb, ransac, erode_px, region_kind):
    """Motion between two frames of the same clip; cached per frame id and parameters."""
    return SCM.track_pair(_fr0, _fr1, erode_px=erode_px, region_kind=region_kind, model=model, max_corners=max_corners,
                          quality=quality, min_distance=min_distance, win=win, levels=levels, max_fb_error=max_fb,
                          ransac_thresh=ransac)


@st.cache_data(max_entries=3, show_spinner="siguiendo el vídeo paso a paso (KLT adelante-atrás + RANSAC; ~1 s por paso con líneas)...")
def klt_video(v, i0, H_key, step, n, refine, region_kind, model, max_corners, quality, min_distance, win, levels, max_fb, ransac, erode_px):
    """Every step of `SCM.track_video` from frame i0, already drawn and JPEG-encoded (a native frame is ~6 MB: never keep them)."""
    H0 = None if H_key is None else np.array(H_key, float).reshape(3, 3)
    out = []
    for r in SCM.track_video(lambda j: SLB.lee_frame(v, j), i0, H0, step=step, n=n, refine_taus=(0.015, 0.006) if refine else None,
                             region_kind=region_kind, erode_px=erode_px, model=model, max_corners=max_corners, quality=quality,
                             min_distance=min_distance, win=win, levels=levels, max_fb_error=max_fb, ransac_thresh=ransac):
        native = SLB.lee_frame(v, r["frame"])
        if native is None:
            break
        img = SCM.render_step(native, r)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        m = SCM.MOT.motion_summary(r["cum"], SCM.W_WORK, int(round(SCM.W_WORK * native.shape[0] / native.shape[1])))
        out.append({"jpg": jpg.tobytes(), "frame": r["frame"], "n_inliers": r["n_inliers"], "n_selected": r["n_selected"],
                    "residual": r["median_residual"], "score": r["score"], "lost": r["lost"], **m})
    return out


# Todo lo que cambia con el deporte. El flujo de hockey es exactamente el de siempre.
DEPORTES = {
    "hockey": SimpleNamespace(
        tpl=TPL, kp_nombre=lambda k: rink.KEYPOINT_NAMES.get(k, "punto"), minimapa=minimapa_hockey,
        frames=CL.frames_candidatos, videos=lambda: list(CL.VIDEOS),
        cid=lambda v, i: "{}_{:06d}".format(v, i), n_anotados=lambda: len(CL.anotados()),
        ajusta=lambda world, pts, w, h: CL.ajusta(world, pts, PARAMS, w, h),
        pendientes=lambda Hraw, pts, w, h: CL.proyecta_pendientes(Hraw, TPL, pts, w, h),
        dibuja=lambda vis, H, col, grosor: rink.draw_rink(vis, H, PARAMS, col, grosor),
        guarda=lambda fr, cid, H, clics: CL.guarda(fr, cid, H, PARAMS, clics),
        salta=CL.marca_saltado, sugiere=None, msg_giro="espejo en x aplicado (zona A a la izquierda)",
        ayuda="**Como se usa**\n\n1. Clic en un punto del **minimapa** (se pone en rojo).\n"
              "2. Clic en ese mismo punto en el **frame**.\n3. Repite con 4-8 puntos bien repartidos.\n"
              "4. Con >=4 puntos aparece la plantilla en amarillo: si encaja, **Guardar**.\n\n"
              "Si el frame no es de pista, **Saltar**."),
    "futbol": SimpleNamespace(
        tpl=SLB.TPL, kp_nombre=lambda k: SLB.NOMBRES.get(k, "punto"), minimapa=minimapa_futbol,
        frames=SLB.frames_candidatos, videos=SLB.videos,
        cid=lambda v, i: "{}_{:06d}".format(v, i), n_anotados=lambda: len(SLB.anotados()),
        ajusta=lambda world, pts, w, h: SLB.ajusta(world, pts, w, h),
        pendientes=lambda Hraw, pts, w, h: SLB.proyecta_pendientes(Hraw, pts, w, h),
        dibuja=lambda vis, H, col, grosor: SLB.dibuja_plantilla(vis, H, col, grosor),
        guarda=lambda fr, cid, H, clics: SLB.guarda(fr, cid, H, clics),
        salta=SLB.marca_saltado, sugiere=SLB.sugiere, msg_giro="giro de 180° aplicado (banda cercana abajo)",
        ayuda="**Como se usa**\n\n1. (Opcional) **Sugerir plantilla**: el solver propone una H; arrastra "
              "los marcadores naranja a su sitio.\n2. O clic en un punto del **minimapa** y luego en "
              "ese punto del **frame**.\n3. Con >=4 puntos aparece la plantilla en amarillo: si encaja, "
              "**Guardar**.\n\nArriba del minimapa = banda LEJANA; la camara esta abajo. "
              "Si el frame no es un plano general del campo, **Saltar**."),
}

ss = st.session_state

# ---------------------------------------------------------------- barra lateral
st.sidebar.title("Etiquetador por clics")
deporte = st.sidebar.radio("Deporte", list(DEPORTES), horizontal=True)
D = DEPORTES[deporte]
if "frames" in ss and "deporte" not in ss:      # sesion abierta antes de que existiera el selector: era hockey
    ss.deporte = "hockey"


def nuevo_frame():
    ss.pts, ss.sel, ss.msg, ss.cands = {}, None, None, []
    ss.ell = []            # points clicked on the outline of the centre circle (soccer)
    ss.ver += 1


if "frames" not in ss or ss.deporte != deporte:
    ss.deporte = deporte
    ss.frames = D.frames()
    ss.pos = 0
    ss.ver = ss.get("ver", 0)
    ss.pts, ss.sel, ss.msg, ss.cands = {}, None, None, []
    ss.ell = []
    ss.ver += 1
if "ell" not in ss:
    ss.ell = []
if "cands" not in ss:
    ss.cands = []

st.sidebar.metric("Anotados", D.n_anotados())
st.sidebar.caption("{} frames candidatos por ver".format(max(0, len(ss.frames) - ss.pos)))
st.sidebar.markdown(D.ayuda)
with st.sidebar.expander("Ir a un frame concreto"):
    vid_ir = st.selectbox("Video", D.videos(), key="ir_v_" + deporte)
    fr_ir = st.number_input("Frame", 0, 10_000_000, 0, step=1, key="ir_f_" + deporte)
    if st.button("Ir"):
        ss.frames.insert(ss.pos, (vid_ir, int(fr_ir)))
        nuevo_frame()
        st.rerun()

if ss.pos >= len(ss.frames):
    st.success("No quedan frames candidatos.")
    st.stop()

v, i = ss.frames[ss.pos]
cid = D.cid(v, i)
fr = get_frame(deporte, v, i)
if fr is None:
    st.warning("No se pudo leer {}; se salta.".format(cid))
    D.salta(cid)
    ss.pos += 1
    st.rerun()
h, w = fr.shape[:2]

pts = ss.pts
ids = sorted(pts)
H = None
res, espejo, aviso, H_raw = [], False, None, None
proy = {}
semilla = None     # H propuesta por el solver (solo futbol), mientras no haya >=4 clics
usa_elipse = deporte == "futbol" and len(ss.ell) >= 5
n_min = 2 if usa_elipse else 4
if usa_elipse and len(ids) >= 2:      # the outline of the centre circle fixes 5 of the 8 degrees of freedom
    H, res, espejo, aviso, H_raw = SLB.ajusta_elipse(D.tpl[ids], [pts[k] for k in ids], ss.ell, w, h)
elif len(ids) >= 4:
    H, res, espejo, aviso, H_raw = D.ajusta(D.tpl[ids], [pts[k] for k in ids], w, h)
if H is not None:
    proy = D.pendientes(H_raw, pts, w, h)

# ---------------------------------------------------------------- sugerencia del solver
centro_fijo = None
if deporte == "futbol":       # fixed camera centre calibrated for this clip (python -m sportcal.lab.soccer.camera_center --save)
    cal = SLB.load_center(v)
    if cal is not None:
        if st.checkbox("Usar el centro de cámara fijo de este vídeo ({:.1f}, {:.1f}, {:.1f}) m, calibrado con {} frames".format(
                *cal["center"], cal.get("votes", "?")), value=True, key="usar_centro"):
            centro_fijo = cal["center"]
    else:
        st.caption("Sin centro de cámara calibrado para '{}': `python -m sportcal.lab.soccer.camera_center --video {} --save` "
                   "añade la sugerencia «centro fijo» (paneo, inclinación y zoom: 3 incógnitas).".format(v, v))
if D.sugiere is not None:
    cs1, cs2 = st.columns([1, 3])
    with cs1:
        if st.button("Sugerir plantilla (~15 s)"):
            with st.spinner("buscando H (intersecciones + chamfer + elipse; ~40 s la primera vez)..."):
                ss.cands = D.sugiere(fr, detector_personas(), ss.get("g_det", "auto"), centro_fijo) if deporte == "futbol" else D.sugiere(fr)
            ss.ver += 1
            st.rerun()
    with cs2:
        if ss.cands:
            opc = ["ninguna"] + ["{} (puntuacion {:.2f})".format(c["via"], c["score"]) for c in ss.cands]
            st.caption("Evidencia usada: {}. La puntuacion incluye elipse y esquinas cuando las hay, por eso puede pasar de 1.".format(
                ss.cands[0].get("nota", "-")))
            elegida = st.radio("Sugerencia", opc, horizontal=True, key="cand_" + cid)
            if elegida != "ninguna":
                semilla = ss.cands[opc.index(elegida) - 1]["H"]
                if ss.cands[opc.index(elegida) - 1]["score"] < 0.3:
                    st.caption("Puntuacion baja: en planos cercanos/sin lineas suele ser basura; "
                               "si el frame no es un plano general, mejor Saltar.")
    if semilla is not None and H is None:
        proy = D.pendientes(semilla, pts, w, h)

# ---------------------------------------------------------------- minimapa
def dibuja_minimapa(pts, sel):
    base, xy = D.minimapa()
    img = base.copy()
    for k, (x, y) in enumerate(xy):
        col, r = (140, 140, 140), 4
        if k in pts:
            col, r = (0, 170, 0), 6
        if k == sel:
            col, r = (0, 0, 255), 9
        cv2.circle(img, (int(x), int(y)), r, col, -1)
        cv2.putText(img, str(k), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 0) if k not in pts else (0, 110, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), xy


st.subheader("{}   ·   frame {}/{}".format(cid, ss.pos + 1, len(ss.frames)))
modo_elipse = False
if deporte == "futbol":
    modo_elipse = st.radio("Qué clicas en el frame", ["Puntos de la plantilla (elige antes el punto en el minimapa)", "Contorno del círculo central (elipse)"],
                           horizontal=True, key="herr").startswith("Contorno")
    if modo_elipse:
        st.info("Clica **5 o más puntos sobre el borde del círculo central** (mejor 8-12 repartidos por el contorno visible; no hace falta que se vea entero). "
                "Clic sobre un punto = lo quita · arrastrar un punto = lo corrige. Después vuelve a «Puntos de la plantilla» y marca **2 puntos**: "
                "el centro de campo (6) y un cruce círculo × línea central (7 u 8); con la elipse basta para ajustar el campo. "
                "Con un tercer punto que no esté en la línea central el ajuste deja de ser ambiguo.")
        e1, e2, _ = st.columns([1, 1, 2])
        H_sem = H if H is not None else semilla
        if e1.button("Sembrar con el círculo de la plantilla actual", disabled=H_sem is None,
                     help="Pone hasta 12 puntos sobre el círculo que dibuja la H actual o la sugerida; arrástralos al círculo real."):
            ss.ell = SLB.elipse_desde_plantilla(H_sem, w, h)
            ss.ver += 1
            st.rerun()
        if e2.button("Borrar la elipse", disabled=not ss.ell):
            ss.ell = []
            ss.ver += 1
            st.rerun()
st.caption("1) Elige el punto en el minimapa   ->   2) haz clic en ese punto del frame  (con >=4 puntos: arrastra los marcadores naranja)")
mini, xy_mini = dibuja_minimapa(pts, ss.sel)
clic_m = streamlit_image_coordinates(mini, key="mini_{}".format(ss.ver), width=W_MINI)
if clic_m:
    cx, cy = clic_m["x"] * (mini.shape[1] / clic_m["width"]), clic_m["y"] * (mini.shape[0] / clic_m["height"])
    d = np.linalg.norm(xy_mini - np.array([cx, cy]), axis=1)
    k = int(np.argmin(d))
    if d[k] < 16:
        ss.sel = k
        ss.ver += 1
        st.rerun()

if ss.sel is not None:
    st.info("Punto seleccionado: **{}** ({}). Ahora haz clic en el frame donde esta.".format(ss.sel, D.kp_nombre(ss.sel)))
else:
    st.caption("Ningun punto seleccionado.")

# ---------------------------------------------------------------- frame
vis = fr.copy()
if H is not None:
    D.dibuja(vis, H, (0, 255, 255), max(2, w // 640))
elif semilla is not None:
    D.dibuja(vis, semilla, (255, 200, 0), max(2, w // 640))    # azul: propuesta, aun sin confirmar con clics
for k in ids:
    x, y = int(pts[k][0]), int(pts[k][1])
    cv2.circle(vis, (x, y), max(6, w // 200), (0, 0, 255), 2)
    cv2.putText(vis, str(k), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1920 * 0.9), (0, 0, 255), 2, cv2.LINE_AA)
# marcadores NARANJA = donde la H actual cree que estan los demas puntos: arrastra cada
# uno a su sitio real (o ignoralo si ya esta bien). Cada arrastre reajusta la H.
for k, (x, y) in ({} if modo_elipse else proy).items():
    cv2.circle(vis, (int(x), int(y)), max(7, w // 180), (0, 140, 255), 2)
    cv2.putText(vis, str(k), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 1920 * 0.7), (0, 140, 255), 2, cv2.LINE_AA)
if deporte == "futbol":
    SLB.dibuja_elipse(vis, ss.ell, (255, 0, 255), max(2, w // 640))
vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
clic_f = streamlit_image_coordinates(vis_rgb, key="frame_{}".format(ss.ver), width=W_FRAME, click_and_drag=True)
if clic_f:
    sx, sy = w / clic_f["width"], h / clic_f["height"]
    g = (clic_f["x1"] * sx, clic_f["y1"] * sy, clic_f["x2"] * sx, clic_f["y2"] * sy)
    if modo_elipse:
        ss.ell = SLB.procesa_gesto_elipse(ss.ell, g, radio=14 * sx, tol_click=5 * sx)
    else:
        ss.pts, ss.sel, ss.msg = CL.procesa_gesto(pts, ss.sel, proy, g, radio=18 * sx, tol_click=5 * sx)
    ss.ver += 1
    st.rerun()
if ss.msg:
    st.warning(ss.msg)

# ---------------------------------------------------------------- estado del ajuste
c1, c2 = st.columns([2, 1])
with c1:
    if len(ids) < n_min:
        st.write("Puntos: {} (hacen falta al menos {} para ver la plantilla{}).".format(
            len(ids), n_min, "; con la elipse de {} puntos bastan 2".format(len(ss.ell)) if usa_elipse else ""))
    elif H is None:
        st.error(aviso or "No se pudo ajustar.")
    else:
        peor = max(res)
        (st.success if peor < 6 else st.warning)(
            "{} puntos · residuo medio {:.1f}px · peor {:.1f}px (a 1920 de ancho){}".format(
                len(ids), float(np.mean(res)), peor, "  ·  " + D.msg_giro if espejo else ""))
        if aviso:
            (st.warning if aviso.startswith("Ajuste poco estable") else st.caption)(aviso)
    if ids:
        filas = [{"punto": k, "que es": D.kp_nombre(k), "x": round(pts[k][0]), "y": round(pts[k][1]),
                  "residuo px": (round(res[j], 1) if len(res) == len(ids) else "")} for j, k in enumerate(ids)]
        st.dataframe(filas, hide_index=True, width="stretch")
with c2:
    if ids and st.button("Deshacer último punto"):
        pts.pop(ids[-1])
        ss.ver += 1
        st.rerun()
    if ids:
        quitar = st.selectbox("Quitar un punto concreto", ["-"] + ids)
        if quitar != "-" and st.button("Quitar {}".format(quitar)):
            pts.pop(quitar)
            ss.ver += 1
            st.rerun()
    if ids and st.button("Limpiar todos"):
        nuevo_frame()
        st.rerun()
    guardable = H is not None
    if st.button("Guardar y siguiente", type="primary", disabled=not guardable):
        clics = {str(k): [float(pts[k][0]), float(pts[k][1])] for k in ids}
        if deporte == "futbol" and ss.ell:
            clics["elipse"] = [[float(x), float(y)] for x, y in ss.ell]
        ok, msg = D.guarda(fr, cid, H, clics)
        if ok:
            ss.pos += 1
            nuevo_frame()
            st.rerun()
        else:
            st.error(msg)
    if st.button("Saltar (no es pista / no se puede)"):
        D.salta(cid)
        ss.pos += 1
        nuevo_frame()
        st.rerun()

# ---------------------------------------------------------------- procesado de la mascara (futbol)
if deporte == "futbol":
    st.divider()
    if st.checkbox("Ver el procesado de la máscara (región, filtros, jugadores descartados, rectas…)", value=True, key="ver_proc"):
        cr1, cr2 = st.columns(2)
        regla = cr1.radio("Regla de jugadores", ["nueva: no descartar componentes muy alargados", "antigua (descarta la linea central)"],
                          key="regla_jug", horizontal=True)
        tol_ang = cr2.slider("Tolerancia angular horizontales/verticales (°)", 5, 60, 30, key="tol_ang")
        et = etapas_futbol(fr, cid, 8.0 if regla.startswith("nueva") else None)
        H_ver, etq = (H, "plantilla ajustada con tus clics") if H is not None else \
                     ((semilla, "plantilla sugerida") if semilla is not None else (None, ""))
        st.caption("Es la máscara que usa **Sugerir plantilla**, calculada a 960 px de ancho. Las etapas 2-5 dicen qué se "
                   "queda y qué se tira antes de que el solver vea nada; si falta una línea aquí, ninguna sugerencia "
                   "podrá encajarla.")
        paneles = SLB.paneles(et, H_ver, etq, w, tol_ang)
        for j in range(0, len(paneles), 2):
            cols = st.columns(2)
            for col, (titulo, img_p, texto) in zip(cols, paneles[j:j + 2]):
                col.markdown("**{}**".format(titulo))
                col.image(img_p, width="stretch")
                col.caption(texto)


# ---------------------------------------------------------------- extraccion por gradiente sobre toda la imagen (futbol)
if deporte == "futbol":
    st.divider()
    if st.checkbox("Ver extracción por gradiente (rectas Hough/RANSAC · elipse · personas)", value=False, key="ver_grad"):
        st.caption("La máscara de césped (gaussiana) se aplica ANTES de calcular la respuesta de línea R y su gradiente, sin apertura ni "
                   "quitajugadores; las rectas se ajustan una a una y los jugadores se descartan con un detector de personas (YOLO en "
                   "CPU). Además una recta de campo debe tener césped a los dos lados. Las elipses las buscan `EdgeDrawing.detectEllipses` "
                   "y `findEllipses` (cv2.ximgproc), cada una con su escala de R y sus umbrales; el `fitEllipse` antiguo es solo comparación.")
        g1, g2, g3, g4 = st.columns(4)
        metodo = g1.radio("Ajuste de rectas", ["ransac", "hough", "fld"], key="g_metodo", horizontal=True,
                          help="fld = FastLineDetector (cv2.ximgproc) sobre la respuesta de línea: mucho más rápido que RANSAC.")
        usar_pers = g2.checkbox("Ignorar bordes de personas", value=True, key="g_pers")
        usar_reg = g2.checkbox("Limitar al campo (gaussiana de césped)", value=True, key="g_region")
        tol_h = g3.slider("Tolerancia horizontal-ish (°)", 20, 70, 50, key="g_tolh")
        tol_v = g4.slider("Tolerancia vertical-ish (°)", 20, 70, 50, key="g_tolv")
        f1, f2, f3, _ = st.columns(4)
        fld_len = f1.slider("FLD: longitud mínima (px)", 8, 80, 20, key="g_fldlen", help="Solo se usa con el método fld.")
        fld_merge = f2.checkbox("FLD: unir segmentos cercanos", value=True, key="g_fldmerge")
        fld_techo = f3.slider("FLD: techo de R", 1.0, 8.0, 3.0, 0.5, key="g_fldtecho")
        g5, g6, g7, g8 = st.columns(4)
        c_lo = g5.slider("Canny bajo", 5, 60, 20, key="g_clo")
        c_hi = g6.slider("Canny alto", 20, 150, 60, key="g_chi")
        cesp = g7.slider("Césped mínimo a cada lado", 0.0, 1.0, 0.3, 0.05, key="g_cesp")
        pico = g8.slider("Pico central mínimo (R)", 0.0, 2.0, 0.5, 0.1, key="g_pico")
        st.markdown("**Detectores de elipses** (entrada = R con la máscara de césped, personas a 0; el techo es el valor de R que se "
                    "manda a 255: un techo alto deja las líneas tenues por debajo de los umbrales internos de gradiente)")
        e1, e2, e3, e4 = st.columns(4)
        det_el = e1.radio("Elipse que alimenta al solver", ["auto", "edgedrawing", "find", "fit"], key="g_det", horizontal=True,
                          help="auto = EdgeDrawing y, si no hay, findEllipses. fit = el fitEllipse robusto antiguo.")
        mostrar_fit = e1.checkbox("Comparar con el fitEllipse antiguo", value=False, key="g_fit")
        ed_techo = e2.slider("EdgeDrawing: techo de R", 1.0, 8.0, 6.0, 0.5, key="g_edtecho")
        ed_grad = e2.slider("EdgeDrawing: gradiente mínimo", 5, 100, 36, key="g_edgrad")
        ed_anc = e3.slider("EdgeDrawing: umbral de ancla", 2, 40, 8, key="g_edanc")
        ed_cam = e3.slider("EdgeDrawing: camino mínimo (px)", 5, 100, 10, key="g_edcam")
        fe_techo = e4.slider("findEllipses: techo de R", 1.0, 8.0, 2.0, 0.5, key="g_fetecho")
        fe_sc = e4.slider("findEllipses: score mínimo", 0.05, 0.95, 0.3, 0.05, key="g_fesc")
        fe_fi = e4.slider("findEllipses: fiabilidad mínima", 0.05, 0.95, 0.3, 0.05, key="g_fefi")
        res_g = grad_resultado(fr, cid, metodo, c_lo, max(c_hi, c_lo + 1), tol_h, tol_v, usar_pers, cesp, pico, usar_reg,
                               det_el, (ed_grad, ed_anc, ed_cam), ed_techo, (fe_sc, fe_fi), fe_techo, (fld_len, fld_merge), fld_techo)
        solo_campo = st.checkbox("Ocultar las rectas descartadas", value=False, key="g_solo")
        pg = SGR.paneles_grad(res_g, mostrar_descartadas=not solo_campo, mostrar_fit=mostrar_fit)
        for j in range(0, len(pg), 2):
            cols = st.columns(2)
            for col, (titulo, img_p, texto) in zip(cols, pg[j:j + 2]):
                col.markdown("**{}**".format(titulo))
                col.image(img_p, width="stretch")
                col.caption(texto)
        st.markdown("**Elipses detectadas** (todos los detectores; la marcada SI es la que recibe el solver)")
        tab_el = SGR.tabla_elipses(res_g)
        (st.dataframe(tab_el, hide_index=True, width="stretch") if tab_el else st.caption("Ningún detector ha encontrado elipses con estos umbrales."))
        st.markdown("**Rectas detectadas**")
        st.dataframe(SGR.tabla_lineas(res_g), hide_index=True, width="stretch")

# ---------------------------------------------------------------- camera motion from stands and boards (soccer)
# Separate zone on purpose: it never looks at the field. Points come from the NON-grass region, are tracked forward and
# backward, and RANSAC keeps the ones that move together (the whole camera moves the whole background the same way).
if deporte == "futbol":
    st.divider()
    if st.checkbox("Ver movimiento de cámara por KLT sobre la grada y la publicidad (sin mirar el campo)", value=False, key="ver_klt"):
        st.caption("Se eligen esquinas SOLO fuera del césped (grada y vallas), se siguen del frame t al t+Δ y de vuelta (KLT adelante-atrás: "
                   "las pistas que no vuelven a su sitio se descartan) y RANSAC se queda con las que se mueven de forma coherente. "
                   "Jugadores, marcador y logos se mueven distinto y salen como atípicos. Con una cámara que solo gira y hace zoom, el "
                   "fondo lejano da el movimiento imagen a imagen aunque el campo casi no tenga líneas.")
        k1, k2, k3, k4 = st.columns(4)
        gap = k1.slider("Separación Δ (frames)", 1, 100, 10, key="k_gap")
        modelo_k = k1.radio("Modelo de movimiento", list(SCM.MOT.MODELS), key="k_model", horizontal=True)
        max_c = k2.slider("Puntos (máx.)", 100, 2000, 800, 50, key="k_n")
        calidad = k2.slider("Calidad de esquina", 0.001, 0.1, 0.01, 0.001, key="k_q", format="%.3f")
        dmin = k3.slider("Distancia mínima entre puntos (px)", 3, 30, 8, key="k_d")
        ventana = k3.slider("Ventana del KLT (px)", 7, 51, 21, 2, key="k_win")
        niveles = k4.slider("Niveles de pirámide", 1, 6, 3, key="k_lv")
        fb = k4.slider("Error adelante-atrás máx. (px)", 0.1, 5.0, 1.0, 0.1, key="k_fb")
        regiones = {"grada y publicidad (recomendado)": "background", "campo (césped y líneas)": "field", "toda la imagen": "all"}
        region_k = regiones[k1.radio("Dónde tomar las esquinas", list(regiones), key="k_region",
                                     help="«campo» prueba el KLT sobre el césped y las líneas pintadas: hay poca textura (unas 100 esquinas, casi todas "
                                          "en las líneas, que solo fijan la posición en la dirección perpendicular) y suele fallar con Δ ≥ 10.")]
        k5, k6, k7 = st.columns(3)
        ransac = k5.slider("Umbral de RANSAC (px)", 0.5, 10.0, 3.0, 0.5, key="k_ransac")
        borde = k6.slider("Erosión del borde del césped (px)", 0, 30, 6, key="k_ero")
        amp = k7.slider("Ampliar el desplazamiento dibujado", 1, 20, 5, key="k_amp")
        fr1 = get_frame(deporte, v, i + gap)
        if fr1 is None:
            st.warning("No hay frame {} en este vídeo (Δ demasiado grande).".format(i + gap))
        else:
            rk = klt_resultado(fr, fr1, cid, gap, modelo_k, max_c, calidad, dmin, ventana, niveles, fb, ransac, borde, region_k)
            H_klt = H if H is not None else semilla
            if H_klt is None:
                st.caption("Para ver la plantilla arrastrada al frame t+Δ, ajusta una H con 4+ clics o elige una sugerencia arriba.")
            pk = SCM.panels(rk, fr, fr1, H_klt, amplify=amp)
            for j in range(0, len(pk), 2):
                cols = st.columns(2)
                for col, (titulo, img_p, texto) in zip(cols, pk[j:j + 2]):
                    col.markdown("**{}**".format(titulo))
                    col.image(img_p, width="stretch")
                    col.caption(texto)
            if rk["M"] is not None:
                st.markdown("**Homografía imagen a imagen (píxeles de trabajo, frame t → t+Δ)**")
                st.dataframe(np.round(rk["M"], 5), hide_index=True, width="stretch")

        # ---- play the tracking over the clip: step by step from this frame, carrying the template
        st.markdown("**Reproducir el seguimiento sobre el vídeo**")
        st.caption("Parte de este frame con la plantilla actual (tus clics o la sugerencia elegida) y la arrastra frame a frame con el movimiento del "
                   "fondo; con la casilla marcada (experimental, ver ayuda), en cada paso la vuelve a pegar a las líneas pintadas de ese frame (cuesta ~1 s por paso). "
                   "Los puntos verdes son las esquinas de fondo que sostienen el paso. Se para solo si pierde el fondo (corte de plano o primer plano).")
        v1, v2, v3, v4 = st.columns(4)
        v_paso = v1.slider("Frames entre pasos", 1, 10, 5, key="kv_step", help="Con pasos de 25 frames el KLT pierde el seguimiento en los paneos rápidos.")
        v_n = v2.slider("Número de pasos", 5, 100, 30, key="kv_n")
        v_ref = v3.checkbox("Corregir con las líneas del campo en cada paso", value=False, key="kv_ref",
                           help="Medido contra etiquetas a mano: empeora la mitad de los tramos de 50 frames; el KLT solo suele ir mejor.")
        v_fps = v4.slider("Velocidad (imágenes por segundo)", 1, 15, 5, key="kv_fps")
        H_play = H if H is not None else semilla
        if H_play is None:
            st.caption("Sin plantilla (ni clics ni sugerencia) solo se ven los puntos seguidos, sin campo dibujado.")
        if st.button("Calcular el seguimiento", key="kv_go"):
            ss.kv = klt_video(v, i, None if H_play is None else tuple(float(x) for x in np.asarray(H_play).ravel()), v_paso, v_n, v_ref,
                              region_k, modelo_k, max_c, calidad, dmin, ventana, niveles, fb, ransac, borde)
            ss.kv_id = cid
        pasos = ss.get("kv") if ss.get("kv_id") == cid else None
        if pasos:
            pos = st.slider("Paso mostrado", 0, len(pasos) - 1, 0, key="kv_pos")
            hueco = st.empty()
            info = st.empty()

            def muestra(j):
                q = pasos[j]
                hueco.image(q["jpg"], width="stretch")
                txt = "frame {} · inliers {}/{} · desplazamiento acumulado dx {:.0f}, dy {:.0f} px (960 de ancho), zoom ×{:.3f}".format(
                    q["frame"], q["n_inliers"], q["n_selected"], q["dx"], q["dy"], q["zoom"])
                if q["score"] is not None:
                    txt += " · puntuación de líneas {:.2f}".format(q["score"])
                info.caption(txt + ("  ·  FONDO PERDIDO: aquí se detiene" if q["lost"] else ""))

            muestra(pos)
            if st.button("▶ Reproducir desde el paso mostrado", key="kv_play"):
                for j in range(pos, len(pasos)):
                    muestra(j)
                    time.sleep(1.0 / v_fps)
