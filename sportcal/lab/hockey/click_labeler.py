"""Logica del etiquetador por clics (sportcal/lab/common/annotate_val_app.py): de N puntos
clicados (punto de plantilla <-> pixel) a una H y una etiqueta YOLO-pose de 56
keypoints, sin nada de Streamlit -- para poder testearla con verdad conocida.

Convencion de clases (medida sobre las H de train: +x -> derecha de la imagen en
el 99% de IIHF y NHL, o sea la zona A queda a la izquierda): la pista es simetrica
(el template es exactamente espejo en x y en y), asi que dos etiquetados que
difieren en una simetria dibujan las MISMAS lineas y solo cambian que clase es A/B
o lo/hi. Para que un frame nuevo respete la convencion de los datos de
entrenamiento se aplica el espejo en x cuando la H sale con +x hacia la izquierda.
El eje y NO se fuerza: en NHL los datos tienen las dos orientaciones (76/24) y
faceoff lo/hi es inherentemente ambiguo.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np

from sportcal.paths import ROOT

from sportcal.lab.hockey import make_line_masks as MM
from sportcal.lab.hockey import relabel_reproject as RP
from sportcal.sports.hockey import rink
from sportcal.core.labels import label_from_H  # noqa: E402

OUT = ROOT / "datasets" / "hockeyrink_nhl_valh"
VIDEOS = ["nhl3", "nhl4", "nhl5", "nhl7", "nhl8", "nhl9", "nhl10"]     # sin nhl6 (pista amateur)
GAP = 120


def _proj(H, pts):
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def canonicalizar(H, params):
    """Espejo en x si la H deja +x hacia la izquierda (convencion A-izquierda)."""
    L, W = params["length"], params["width"]
    a = _proj(H, np.array([[L / 2 - 5, W / 2], [L / 2 + 5, W / 2]]))
    if a[1, 0] < a[0, 0]:
        M = np.array([[-1.0, 0, L], [0, 1.0, 0], [0, 0, 1.0]])
        return H @ M, True
    return H, False


def sensibilidad(world, img_pts, Hn, w, h, params=None, sigma=2.0, n=24):
    """Cuanto se mueve la plantilla DENTRO DEL FRAME (px a 1920, mediana) si cada clic
    se desvia ~sigma px: rejilla de puntos de la imagen -> mundo con la H ajustada ->
    imagen con H reajustadas sobre clics perturbados. Es la medida que importa: no
    cuanta pista abarcan los puntos (un plano cerrado solo ve un trozo pequeno y no se
    le puede pedir esquinas opuestas) sino si el ajuste es estable en lo que se ve.

    Solo cuentan los puntos de la rejilla que caen SOBRE LA PISTA (mundo dentro del
    rectangulo +-2 m): la grada y el horizonte de la imagen se disparan con cualquier
    ajuste y no son donde se dibujan las lineas (primera version, promediando toda la
    imagen: avisaba en el 91-100% de los casos aunque el error real fuera de 7px)."""
    params = params or rink.RINK_NHL
    L, Wd = params["length"], params["width"]
    rng = np.random.default_rng(0)
    sc = w / 1920.0
    gx, gy = np.meshgrid(np.linspace(0.05 * w, 0.95 * w, 14), np.linspace(0.05 * h, 0.95 * h, 9))
    G = np.c_[gx.ravel(), gy.ravel()]
    try:
        with np.errstate(all="ignore"):
            Wg = _proj(np.linalg.inv(Hn), G)
    except np.linalg.LinAlgError:
        return float("inf")
    en_pista = (np.isfinite(Wg).all(1) & (Wg[:, 0] > -2) & (Wg[:, 0] < L + 2) & (Wg[:, 1] > -2) & (Wg[:, 1] < Wd + 2))
    if en_pista.sum() < 4:
        return float("inf")
    G, Wg = G[en_pista], Wg[en_pista]
    wf = np.asarray(world, np.float32)
    d = []
    for _ in range(n):
        noisy = np.asarray(img_pts, np.float64) + rng.normal(0, sigma * sc, np.shape(img_pts))
        Hi, _ = cv2.findHomography(wf, noisy.astype(np.float32), 0)
        if Hi is None or not np.all(np.isfinite(Hi)):
            return float("inf")
        with np.errstate(all="ignore"):
            d.append(np.linalg.norm(_proj(Hi, Wg) - G, axis=1) / sc)
    d = np.concatenate(d)
    return float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else float("inf")


def ajusta(world, img_pts, params, w, h=None):
    """(H, residuos_px_a_1920, espejo_aplicado, aviso, H_sin_espejo) o
    (None, [], False, motivo, None).

    H es la canonica (la que se guarda); H_sin_espejo es la que corresponde al
    etiquetado del usuario: proyectar el punto k con ELLA da donde el usuario espera
    ver el punto k (con la canonica saldria su simetrico).

    world: (n,2) metros;  img_pts: (n,2) pixeles de la imagen ORIGINAL.  Minimos
    cuadrados sobre todos los clics (sin RANSAC: cada clic es una decision humana y
    el residuo por punto se ensena para que el humano corrija el que falle)."""
    world = np.asarray(world, np.float64)
    img_pts = np.asarray(img_pts, np.float64)
    n = len(world)
    if n < 4:
        return None, [], False, "hacen falta al menos 4 puntos (hay {})".format(n), None
    sv = np.linalg.svd(world - world.mean(0), compute_uv=False)
    if sv[1] < 0.01 * sv[0]:
        return None, [], False, "puntos casi alineados en el mundo: elige alguno fuera de esa recta", None
    H, _ = cv2.findHomography(world.astype(np.float32), img_pts.astype(np.float32), 0)
    if H is None or not np.all(np.isfinite(H)):
        return None, [], False, "no se pudo ajustar la homografia", None
    Hn = H / H[2, 2]
    if np.abs(Hn).max() > 1e6:
        return None, [], False, "homografia degenerada (puntos mal repartidos)", None
    if h is not None:
        _, ok = MM.project_points(Hn, rink.build_template(params), w, h)
        xy = _proj(Hn, rink.build_template(params))
        dentro = ok & (xy[:, 0] > 0) & (xy[:, 0] < w) & (xy[:, 1] > 0) & (xy[:, 1] < h)
        if dentro.sum() < 4:
            return None, [], False, "con esos puntos casi toda la plantilla cae fuera del frame: revisa que no haya un punto mal asignado", None
    res = np.linalg.norm(_proj(Hn, world) - img_pts, axis=1) * 1920.0 / w
    Hc, espejo = canonicalizar(Hn, params)
    aviso = None
    sens = sensibilidad(world, img_pts, Hn, w, h if h else w * 9 / 16, params)
    # calibrado con verdad conocida (352 configuraciones de 4 puntos aleatorios sobre 19
    # frames de val): error real ~ 0.42 x sens (mediana; p90 ~1.2x); umbral 30 avisa al 56%
    # de esas configuraciones y de las que NO avisa solo el 6% tiene error real > 15px.
    esperado = 0.42 * sens
    if sens > 30.0:
        aviso = ("Ajuste poco estable: con clics a ±2px la plantilla puede desviarse ~{:.0f}px dentro del "
                 "frame. No bloquea; se corrige añadiendo o arrastrando mas puntos (mejor si no estan "
                 "todos pegados).".format(esperado if np.isfinite(esperado) else 999))
    elif n == 4:
        aviso = ("con 4 puntos el ajuste es exacto (residuo 0); error esperado ~{:.0f}px: "
                 "revisa el dibujo o añade un 5o punto".format(esperado))
    return Hc, [float(r) for r in res], espejo, aviso, Hn


def proyecta_pendientes(H_sin_espejo, tpl, pts, w, h, margen=4):
    """{k: (x,y)} de los puntos de la plantilla AUN sin clicar que la H situa dentro
    del frame y delante de la camara -- los que el usuario puede arrastrar a su sitio."""
    xy, ok = MM.project_points(H_sin_espejo, tpl, w, h)
    out = {}
    for k in range(len(tpl)):
        if k in pts or not ok[k]:
            continue
        x, y = float(xy[k, 0]), float(xy[k, 1])
        if margen <= x <= w - margen and margen <= y <= h - margen:
            out[k] = (x, y)
    return out


def procesa_gesto(pts, sel, proy, g, radio, tol_click):
    """Gesto sobre el frame (px de la imagen ORIGINAL) -> (pts, sel, mensaje).

    g = (x1, y1, x2, y2): pulsar y soltar. Arrastre = distancia > tol_click.
      - arrastre que EMPIEZA sobre un marcador (proyectado o ya clicado, a <= radio):
        ese punto pasa a estar en (x2, y2) -- "arrastrar a su sitio".
      - clic (o arrastre desde un sitio vacio) con un punto elegido en el minimapa:
        se coloca en (x2, y2).
      - clic sobre un marcador sin punto elegido: lo elige (el siguiente clic lo coloca).
    """
    x1, y1, x2, y2 = g
    pts = dict(pts)
    cand = {**proy, **pts}
    k_near, d_min = None, radio
    for k, (x, y) in cand.items():
        d = float(np.hypot(x - x1, y - y1))
        if d <= d_min:
            k_near, d_min = k, d
    arrastre = float(np.hypot(x2 - x1, y2 - y1)) > tol_click
    if arrastre and k_near is not None:
        pts[k_near] = (x2, y2)
        return pts, None, None
    if sel is not None:
        pts[sel] = (x2, y2)
        return pts, None, None
    if k_near is not None and not arrastre:
        return pts, k_near, None
    return pts, sel, "Elige antes un punto en el minimapa (o arrastra un marcador naranja)."


def guarda(frame_bgr, cid, H, params, clics):
    """Escribe imagen + etiqueta de 56 keypoints en datasets/hockeyrink_nhl_valh/."""
    tpl = rink.build_template(params)
    h, w = frame_bgr.shape[:2]
    lab = label_from_H(H, tpl, w, h)
    if lab is None:
        return False, "muy pocos keypoints de la plantilla caen en cuadro con esa H"
    kpts, box = lab
    (OUT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "val").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "images" / "val" / (cid + ".jpg")), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    RP.write_label(OUT / "labels" / "val" / (cid + ".txt"), 0, box, kpts)
    with open(OUT / "clicks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": cid, "clics": clics, "H": np.asarray(H).tolist()}) + "\n")
    return True, "guardado {}".format(cid)


def anotados():
    d = OUT / "images" / "val"
    return {p.stem for p in d.glob("*.jpg")} if d.exists() else set()


def saltados():
    p = OUT / "skipped.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def marca_saltado(cid):
    s = saltados()
    s.add(cid)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "skipped.json").write_text(json.dumps(sorted(s)))


def frames_candidatos(seed=0, paso_s=2.0):
    """[(video, frame)] en zonas a >=GAP frames de cualquier frame de train, sin
    filtrar por si el DLT resuelve (para que entren tambien los frames DIFICILES),
    barajados con semilla fija; se excluyen los ya anotados/saltados."""
    tr = {}
    for p in (ROOT / "datasets" / "hockeyrink_nhl" / "images" / "train").glob("*.jpg"):
        m = re.match(r"(.+?)_(\d+)$", p.stem)
        if m:
            tr.setdefault(m.group(1), []).append(int(m.group(2)))
    out = []
    for v in VIDEOS:
        cap = cv2.VideoCapture(str(ROOT / (v + ".mp4")))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        gap = int(GAP * fps / 60)
        bloq = np.zeros(n + 1, bool)
        for i in tr.get(v, []):
            bloq[max(0, i - gap):min(n, i + gap) + 1] = True
        step = int(round(fps * paso_s))
        for i in range(0, n, step):
            if not bloq[i]:
                out.append((v, i))
    # se baraja la lista COMPLETA con semilla fija y se filtran los ya hechos DESPUES:
    # el orden de los que quedan es estable, asi que al reabrir la app sale exactamente
    # el siguiente frame pendiente (barajar tras filtrar lo cambiaba con cada guardado)
    rng = np.random.default_rng(seed)
    rng.shuffle(out)
    hecho = anotados() | saltados()
    return [(v, i) for v, i in out if "{}_{:06d}".format(v, i) not in hecho]


def lee_frame(v, i):
    cap = cv2.VideoCapture(str(ROOT / (v + ".mp4")))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None
