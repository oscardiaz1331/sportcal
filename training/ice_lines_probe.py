"""Prueba: segmentacion adaptativa del hielo (GMM) + realce de lineas por croma local.

Por que no basta el umbral fijo HSV (S<60, V>180):

  - El umbral esta calibrado para UN pabellon. Cambia la temperatura de color del
    hielo, el grading de la emision o la iluminacion y se descuadra.
  - Y sobre todo: las lineas pintadas son MAS saturadas que el hielo, asi que el
    propio umbral que aisla el hielo expulsa justo lo que queremos encontrar.
    Por eso hay que separar dos cosas distintas: el HIELO (color) y la REGION DE
    PISTA (el area conexa que ocupa, lineas y jugadores incluidos).

Que hace esta prueba, por frame:

  1. hielo por GMM: ajusta un modelo de mezclas en Lab con cv2.ml.EM y se queda con
     los componentes claros y poco cromaticos. Sin umbrales absolutos.
  2. region de pista: cierre morfologico + relleno de huecos sobre el hielo, y se
     queda con la componente conexa mayor. Recupera lineas y jugadores.
  3. realce de lineas: Lab respecto al color LOCAL del hielo (mediana de ventana
     grande). El tono absoluto no sirve -- una linea roja bajo el hielo es un rosa
     casi neutro y su H es puro ruido -- pero su desviacion respecto al hielo de al
     lado es estable.
  4. filtro de cresta multiescala: las lineas son estructuras finas y alargadas; los
     jugadores, manchas compactas. Esto separa unas de otras.

Y lo mide, en vez de solo pintarlo: con la homografia del frame se proyecta el
contorno real de la pista y se compara por IoU contra cada metodo de segmentacion,
y la respuesta de linea contra la mascara de training/make_line_masks.py.

    python training/ice_lines_probe.py
    python training/ice_lines_probe.py --n 12 --dataset hockeyrink_nhl
"""
import os

_CUDNN_DIR = os.sep.join(("NVIDIA", "CUDNN"))

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if _CUDNN_DIR not in p
)

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import make_line_masks as MM  # noqa: E402
import rink  # noqa: E402
from relabel_reproject import read_label  # noqa: E402

# El umbral que se usa ahora, como linea base a batir
HSV_S_MAX = 60
HSV_V_MIN = 180


# ---------------------------------------------------------------- segmentacion

def ice_hsv(img):
    """Linea base: umbral fijo en HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] < HSV_S_MAX) & (hsv[:, :, 2] > HSV_V_MIN)).astype(np.uint8)


def ice_gmm(img, k=5, sample=30000, seed=0):
    """Hielo por mezcla de gaussianas en Lab, sin umbrales absolutos.

    Se ajusta el GMM sobre una version reducida (basta para estimar los modos de
    color) y se marcan como hielo TODOS los componentes claros y poco cromaticos,
    no solo el mayor: el hielo suele partirse en dos o tres modos por la
    iluminacion desigual del pabellon y por las zonas pisadas.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    small = cv2.resize(lab, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    X = small.reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)

    em = cv2.ml.EM_create()
    em.setClustersNumber(k)
    em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
    em.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 40, 0.1))
    if not em.trainEM(X[idx])[0]:
        return ice_hsv(img), None

    means = em.getMeans()
    weights = em.getWeights().ravel()
    L = means[:, 0]
    chroma = np.hypot(means[:, 1] - 128.0, means[:, 2] - 128.0)

    # el componente mas "hielo" que exista: claro, neutro y con peso
    score = weights * (L / 255.0) ** 2 * np.exp(-chroma / 8.0)
    ancla = int(np.argmax(score))
    # y con el todos los que estan cerca en claridad y siguen siendo neutros:
    # asi se recupera el hielo en sombra sin volver a fijar un umbral absoluto
    sel = np.where((L > L[ancla] - 30) & (chroma < max(12.0, chroma[ancla] + 4)))[0]
    if ancla not in sel:
        sel = np.append(sel, ancla)

    # posterior pixel a pixel sobre la imagen completa, a media resolucion
    half = cv2.resize(lab, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    Y = half.reshape(-1, 3).astype(np.float64)
    _, labels = em.predict(Y)
    lab_idx = labels.ravel().astype(np.int32) if labels.ndim > 1 else labels.astype(np.int32)
    if lab_idx.size != len(Y):                       # predict devuelve log-likelihoods
        lab_idx = np.argmax(labels, axis=1).astype(np.int32)
    mask_half = np.isin(lab_idx, sel).reshape(half.shape[:2]).astype(np.uint8)
    mask = cv2.resize(mask_half, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    info = {"L": L.round(0), "chroma": chroma.round(1), "w": weights.round(3), "sel": sel}
    return mask, info


def ice_robust(img, chi2=9.0, iters=4):
    """Hielo por gaussiana robusta en Lab, resembrada por percentiles.

    Es lo que el GMM de arriba deberia hacer y no hace de forma fiable: ice_gmm
    elige COMPONENTES, y cuando el hielo se parte en varios modos (un logo grande
    en el centro, iluminacion desigual) la regla de seleccion deja fuera medio
    hielo -- medido: IoU 0.97 de mediana en hockeyrink pero 0.38 en el peor caso, y
    0.74 de mediana en hockeyrink_nhl, por debajo del umbral fijo.

    Aqui se ajusta UNA gaussiana a todo el hielo y se itera (equivale a EM de un
    componente con asignacion dura). La semilla usa percentiles de la propia imagen,
    no valores absolutos: claro respecto a esta imagen y neutro respecto a esta
    imagen. Eso es lo que da la adaptacion entre pabellones sin recalibrar nada.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    X = lab.reshape(-1, 3)
    L = X[:, 0]
    chroma = np.hypot(X[:, 1] - 128.0, X[:, 2] - 128.0)
    m = (L > np.percentile(L, 55)) & (chroma < np.percentile(chroma, 45))
    if m.sum() < 1000:
        return ice_hsv(img)
    for _ in range(iters):
        mu = X[m].mean(0)
        cov = np.cov(X[m].T) + np.eye(3) * 2.0        # regularizado: el hielo es casi degenerado
        d = X - mu
        maha = np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
        nuevo = maha < chi2
        if nuevo.sum() < 0.02 * len(X):
            break
        m = nuevo
    return m.reshape(lab.shape[:2]).astype(np.uint8)


def rink_region(ice, k_frac=0.012, frac_keep=0.20):
    """De 'pixeles de hielo' a 'region de pista': cierra, agrupa y rellena.

    Las lineas y los jugadores son agujeros dentro del hielo, no fondo. Rellenarlos
    convierte una mascara de COLOR en una mascara de AREA, que es la que sirve para
    acotar la busqueda de lineas y cuyo contorno son las vallas.

    No se coge solo la componente mayor: un logo grande en el centro del hielo parte
    la pista en dos trozos y quedarse con uno pierde media pista. Se conservan todas
    las componentes con al menos frac_keep del area de la mayor (medido: p10 de IoU
    0.822 frente a 0.777 quedandose solo con la mayor).

    El casco convexo, que parecia prometedor, resulto ser PEOR que rellenar en todas
    las combinaciones medidas; queda descartado.
    """
    h, w = ice.shape
    k = max(3, int(round(w * k_frac)) | 1)
    m = cv2.morphologyEx(ice, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(ice)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.where(areas >= frac_keep * areas.max())[0]
    out = np.isin(lbl, keep).astype(np.uint8)
    # rellenar huecos: inundar desde fuera y quedarse con lo no alcanzado
    ff = out.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (out | (1 - ff)).astype(np.uint8)


def region_kickplate_global(img, thr=12.0, k_frac=0.02, mode="hull"):
    """INTENTO DESCARTADO: usar el zocalo de las vallas para segmentar la region
    de pista entera, en vez del color del hielo. La idea es razonable -- el zocalo
    es 5-16x mas discriminativo que las lineas pintadas (ver fit_homography_lines.py,
    z=79.9/48.5 contra z=5-10) -- pero medido a nivel de FRAME COMPLETO falla peor
    que el umbral de color, y por mucho:

                              hockeyrink              hockeyrink_nhl
                          p50    p10   peor        p50    p10   peor
        best_region      0.914  0.831 0.344       0.885  0.792 0.711   <- lo que hay
        flood-fill anillo0.920  0.007 0.003       0.015  0.002 0.000   <- catastrofico
        casco convexo    0.435  0.263 0.122       0.565  0.208 0.062
        mayor contorno   0.036  0.013 0.003       0.019  0.005 0.002

    La causa, medida directamente: de los pixeles con dB > 8 en un frame tipico,
    solo el 17-38 % estan cerca del contorno real de la pista. El resto (43-69 %)
    esta en la grada, en anuncios de madera con tonos calidos, en publicidad de las
    vallas lejanas o en la piel/luces del graderio -- cualquier cosa amarillenta
    del FRAME ENTERO, no solo el zocalo. El umbral no tiene forma de distinguir
    "zocalo de la pista" de "cualquier otra cosa calida en la imagen" sin saber
    antes donde esta la pista, que es precisamente lo que se le pide que averigue.

    Por eso el flood-fill (que exige un anillo topologicamente cerrado) colapsa en
    cuanto hay una fuga hacia la grada, y el casco convexo se infla hasta cubrir
    cualquier mancha calida lejana por pequeña que sea.

    La leccion, y por que SI funciona en fit_homography_lines.py: la misma senal
    integrada a lo largo de una curva HIPOTETICA conocida (el contorno que predice
    una H candidata) es extremadamente selectiva, porque solo mira esos pixeles
    concretos y descarta todo el resto del frame. Perdiendo esa restriccion de
    forma -- pidiendole que decida sola, sobre la imagen entera, donde esta el
    borde -- la misma senal dej de servir. Funcion conservada solo como referencia;
    no la uses, no esta enganchada a best_region().
    """
    da, db = local_chroma(img, win=int(61 * img.shape[1] / 1920) | 1)
    ring = (db > thr).astype(np.uint8)
    k = max(3, int(round(img.shape[1] * k_frac)) | 1)
    ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    if mode == "hull":
        pts = cv2.findNonZero(ring)
        if pts is None or len(pts) < 500:
            return np.zeros_like(ring)
        out = np.zeros_like(ring)
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
        return out
    # mode == "flood": requiere que el anillo cierre topologicamente
    h, w = ring.shape
    free = (ring == 0).astype(np.uint8)
    seed = next(((x, y) for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))
                if free[y, x] == 1), None)
    if seed is None:
        return np.zeros_like(ring)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flooded = free.copy()
    cv2.floodFill(flooded, mask, seed, 2)
    return (flooded != 2).astype(np.uint8)


def solidity(m):
    """Area / area del casco convexo. La pista es convexa: si esto baja, esta rota."""
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt:
        return 0.0
    ha = cv2.contourArea(cv2.convexHull(np.vstack(cnt)))
    return float(np.count_nonzero(m) / ha) if ha > 0 else 0.0


def best_region(img):
    """Region de pista combinando umbral fijo y GMM, eligiendo por solidez.

    Ninguno de los dos gana solo. Sobre 88 + 96 frames, con la verdad proyectada:

                       hockeyrink            hockeyrink_nhl
                   p50    p10   <0.7      p50    p10   <0.7
        HSV fijo  0.897  0.816  5.7 %    0.870  0.760  2.1 %
        GMM       0.931  0.822  6.8 %    0.905  0.724  7.3 %
        solidez   0.915  0.829  3.4 %    0.886  0.760  1.0 %

    El GMM tiene mejor mediana pero peor cola; el umbral fijo al reves. Elegir por
    solidez se queda cerca de la mejor mediana y CASI HALVES la tasa de fallos en
    los dos, porque el modo de fallo tipico -- quedarse con media pista cuando un
    logo la parte -- produce una region no convexa y se detecta sin verdad.
    """
    rh = rink_region(ice_hsv(img))
    rg = rink_region(ice_gmm(img)[0])
    return rg if solidity(rg) >= solidity(rh) else rh


# ---------------------------------------------------------------- lineas

def local_chroma(img, win=61):
    """Desviacion de color respecto al hielo LOCAL: (da, db) en Lab.

    La mediana de ventana grande estima el color del hielo porque la linea es
    minoria dentro de la ventana. Restarla cancela balance de blancos, grading e
    iluminacion, que es lo que hace inservible al tono absoluto.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    win = max(11, int(win) | 1)
    a_loc = cv2.medianBlur(lab[:, :, 1], win).astype(np.float32)
    b_loc = cv2.medianBlur(lab[:, :, 2], win).astype(np.float32)
    return lab[:, :, 1].astype(np.float32) - a_loc, lab[:, :, 2].astype(np.float32) - b_loc


def ridge(resp, scales=(1.5, 3.0, 6.0, 10.0)):
    """Cresta multiescala (laplaciano normalizado), solo para visualizar.

    Medido contra la metrica de localizacion de abajo, NO mejora al da crudo:
    z 3.1 frente a 3.3, y 52% de curvas sobre 3z frente a 57%. La anchura de linea
    varia tanto con la perspectiva que el maximo entre escalas acaba recogiendo
    ruido tan a menudo como senal. Se deja porque ayuda a ver la estructura, pero
    la deteccion va sobre da sin filtrar.
    """
    out = np.zeros_like(resp)
    for s in scales:
        g = cv2.GaussianBlur(resp, (0, 0), s)
        out = np.maximum(out, -cv2.Laplacian(g, cv2.CV_32F, ksize=3) * (s ** 2))
    return out


def normals(pts):
    """Normal unitaria en cada punto de una polilinea."""
    t = np.gradient(pts, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    L = np.linalg.norm(n, axis=1, keepdims=True)
    L[L < 1e-6] = 1.0
    return n / L


def integrate(R, pts, w, h, min_pts=50):
    """Media de la respuesta a lo largo de una curva, o None si sale del frame."""
    inb = (pts[:, 0] > 1) & (pts[:, 0] < w - 2) & (pts[:, 1] > 1) & (pts[:, 1] < h - 2)
    if inb.sum() < min_pts:
        return None
    q = pts[inb]
    return float(R[q[:, 1].astype(int), q[:, 0].astype(int)].mean())


def localization_z(R, H, polys, w, h, offsets=(-30, -24, -18, -12, 12, 18, 24, 30)):
    """Cuanto destaca la curva en su sitio frente a la misma curva desplazada.

    Es la metrica que decide si esto sirve: no mide contraste por pixel -- que es
    inutil, porque el rival no es el ruido del sensor sino los jugadores y los
    logos del hielo -- sino si una busqueda sabria distinguir la posicion correcta.
    Se desplaza en la NORMAL de la curva, que es la unica direccion en la que una
    linea se puede confundir consigo misma.
    """
    out = []
    for cls, world in polys:
        if cls not in (2, 3, 6, 7, 8, 9, 10, 11):     # curvas rojas
            continue
        xy, ok = MM.project_points(H, world, w, h)
        if ok.sum() < 60:
            continue
        base = xy[ok]
        v0 = integrate(R, base, w, h)
        if v0 is None:
            continue
        nz = normals(base)
        alt = [integrate(R, base + nz * d, w, h) for d in offsets]
        alt = [a for a in alt if a is not None]
        if len(alt) < 5:
            continue
        out.append((v0 - np.mean(alt)) / (np.std(alt) or 1e-6))
    return out


# ---------------------------------------------------------------- verdad

def rink_outline(p, n=160):
    """Contorno cerrado de las vallas en metros (rectas + arcos de esquina)."""
    L, W, r = p["length"], p["width"], p["corner_r"]
    return np.vstack([
        MM.seg((r, 0), (L - r, 0), n), MM.arc(L - r, r, r, -90, 0, n),
        MM.seg((L, r), (L, W - r), n), MM.arc(L - r, W - r, r, 0, 90, n),
        MM.seg((L - r, W), (r, W), n), MM.arc(r, W - r, r, 90, 180, n),
        MM.seg((0, W - r), (0, r), n), MM.arc(r, r, r, 180, 270, n),
    ])


def clip_halfplane(poly, a, b, c, eps=0.0):
    """Sutherland-Hodgman contra el semiplano a*x + b*y + c > eps.

    El contorno de la pista es convexo, asi que recortarlo con semiplanos lo deja
    convexo y el resultado se puede rellenar directamente.
    """
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    d = poly @ np.array([a, b]) + c - eps
    for i in range(n):
        j = (i + 1) % n
        di, dj = d[i], d[j]
        if di > 0:
            out.append(poly[i])
        if (di > 0) != (dj > 0):
            t = di / (di - dj)
            out.append(poly[i] + t * (poly[j] - poly[i]))
    return np.array(out) if out else np.zeros((0, 2))


def true_region(H, p, w, h, margin=3.0):
    """Region real de la pista proyectando el contorno, recortando por el horizonte.

    Antes esto devolvia None en cuanto un punto del contorno caia detras de la
    camara, y eso descartaba la mayoria de los frames: en un plano de television la
    valla lejana casi siempre cruza el horizonte. Lo correcto es recortar el
    poligono EN EL MUNDO contra la recta preimagen del horizonte (la de w = 0, que
    en coordenadas de pista es H[2,0]*x + H[2,1]*y + H[2,2] = 0) y proyectar lo que
    queda. Asi el frame sigue siendo utilizable.
    """
    poly = rink_outline(p)
    # 1) recorte en el mundo: solo lo que queda delante de la camara
    a, b, c = H[2, 0], H[2, 1], H[2, 2]
    if a * poly[:, 0].mean() + b * poly[:, 1].mean() + c < 0:
        a, b, c = -a, -b, -c
    escala = max(abs(a), abs(b), abs(c), 1e-9)
    poly = clip_halfplane(poly, a, b, c, eps=1e-3 * escala)
    if len(poly) < 3:
        return None
    xy = cv2.perspectiveTransform(poly.reshape(-1, 1, 2).astype(np.float32),
                                  H.astype(np.float32)).reshape(-1, 2)
    if not np.isfinite(xy).all():
        return None
    # 2) recorte en imagen contra un rectangulo amplio, para no desbordar int32
    lo_x, hi_x = -margin * w, (1 + margin) * w
    lo_y, hi_y = -margin * h, (1 + margin) * h
    for a_, b_, c_ in ((1, 0, -lo_x), (-1, 0, hi_x), (0, 1, -lo_y), (0, -1, hi_y)):
        xy = clip_halfplane(xy, a_, b_, c_)
        if len(xy) < 3:
            return None
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [xy.astype(np.int32)], 1)
    return m if m.any() else None


def iou(a, b):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------- visual

def panel(title, img):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def gray3(x, lo=None, hi=None):
    lo = np.percentile(x, 1) if lo is None else lo
    hi = np.percentile(x, 99.5) if hi is None else hi
    v = np.clip((x - lo) / max(1e-6, hi - lo), 0, 1)
    return cv2.cvtColor((v * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def tint(mask, color):
    out = np.zeros(mask.shape + (3,), np.uint8)
    out[mask > 0] = color
    return out


def build_figure(img, ice_f, ice_g, reg_f, reg_g, da, db, cres, gt):
    h, w = img.shape[:2]
    over_f = cv2.addWeighted(img, 0.6, tint(reg_f, (0, 180, 255)), 0.4, 0)
    over_g = cv2.addWeighted(img, 0.6, tint(reg_g, (0, 255, 120)), 0.4, 0)
    if gt is not None:
        cnt, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over_f, cnt, -1, (255, 255, 255), 3)
        cv2.drawContours(over_g, cnt, -1, (255, 255, 255), 3)

    croma = np.zeros_like(img)
    croma[:, :, 2] = np.clip(da / 8.0, 0, 1).astype(np.float32).__mul__(255).astype(np.uint8)
    croma[:, :, 0] = np.clip(-db / 12.0, 0, 1).astype(np.float32).__mul__(255).astype(np.uint8)
    croma[reg_g == 0] //= 6

    cres_v = gray3(cres * (reg_g > 0))
    tiles = [
        panel("1) original", img),
        panel("2) HSV fijo S<60 V>180 (blanco = pista real)", over_f),
        panel("3) region combinada (elegida por solidez)", over_g),
        panel("4) croma local: rojo = +a, azul = -b", croma),
        panel("5) cresta multiescala sobre +a", cres_v),
        panel("6) candidatos de linea", cv2.addWeighted(img, 0.5, tint(
            (cres * (reg_g > 0) > np.percentile(cres[reg_g > 0], 99.2)).astype(np.uint8),
            (0, 0, 255)), 0.9, 0)),
    ]
    sc = 640.0 / w
    tiles = [cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) for t in tiles]
    return np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6])])


# ---------------------------------------------------------------- main

def run(args):
    params = rink.RINK_NHL if "nhl" in args.dataset else rink.RINK_IIHF
    tpl = rink.build_template(params)
    polys = MM.rink_polylines(params)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lbls = sorted((ROOT / "datasets" / args.dataset / "labels" / "train").glob("*.txt"))
    res = {"hsv": [], "gmm": [], "robusta": [], "solidez": []}
    snr = {"hsv": [], "gmm": [], "robusta": [], "solidez": []}
    hechos = 0
    for lbl in lbls[:: max(1, len(lbls) // (args.n * 4))]:
        if hechos >= args.n:
            break
        ip = ROOT / "datasets" / args.dataset / "images" / "train" / (lbl.stem + ".jpg")
        if not ip.exists():
            continue
        rec = read_label(lbl)
        if rec is None:
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        fit, _ = MM.fit_from_label(rec[2], tpl, w, h, 8, 6.0, 8.0)
        if fit is None:
            continue
        H = fit[0]
        gt = true_region(H, params, w, h)
        if gt is None:
            continue

        ice_f = ice_hsv(img)
        ice_g, info = ice_gmm(img)
        ice_r = ice_robust(img)
        reg_f, reg_g, reg_r = rink_region(ice_f), rink_region(ice_g), rink_region(ice_r)
        res["hsv"].append(iou(reg_f > 0, gt > 0))
        res["gmm"].append(iou(reg_g > 0, gt > 0))
        res["robusta"].append(iou(reg_r > 0, gt > 0))
        reg_s = reg_g if solidity(reg_g) >= solidity(reg_f) else reg_f
        res["solidez"].append(iou(reg_s > 0, gt > 0))

        da, db = local_chroma(img, win=int(args.median_win * w / 1920) | 1)
        cres = ridge(da)

        # localizacion: se distingue la curva en su sitio de la misma desplazada?
        for nombre, reg in (("hsv", reg_f), ("gmm", reg_g), ("robusta", reg_r), ("solidez", reg_s)):
            snr[nombre] += localization_z(da * (reg > 0), H, polys, w, h)

        cv2.imwrite(str(out_dir / (args.dataset + "_" + lbl.stem + ".jpg")),
                    build_figure(img, ice_f, ice_g, reg_f, reg_s, da, db, cres, gt))
        hechos += 1

    print("\n=== {} ({} frames) ===".format(args.dataset, hechos))
    print("  IoU de la region de pista contra el contorno real proyectado:")
    for k in ("hsv", "gmm", "robusta", "solidez"):
        if res[k]:
            v = np.array(res[k])
            print("    {:8s} p50 {:.3f}   p10 {:.3f}   peor {:.3f}".format(
                k, np.median(v), np.percentile(v, 10), v.min()))
    print("  localizacion de curva (z de la posicion correcta vs desplazada +-12..30 px):")
    for k in ("hsv", "gmm", "robusta", "solidez"):
        if snr[k]:
            v = np.array(snr[k])
            print("    {:8s} p50 {:5.1f}   >2z {:3.0f} %   >3z {:3.0f} %   (n={})".format(
                k, np.median(v), 100 * np.mean(v > 2), 100 * np.mean(v > 3), len(v)))
    print("\n  figuras en {}".format(out_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hockeyrink")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--median-win", type=float, default=61, help="ventana del color local, en px a 1920")
    ap.add_argument("--out", default=str(ROOT / "scratch_frames" / "ice_probe"))
    run(ap.parse_args())
