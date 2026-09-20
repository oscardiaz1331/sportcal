"""Ajuste robusto de rectas y circulos a una mascara binaria -- puro cv2/numpy,
sin torch ni dependencia de ningun modelo. Extraido de seg_to_homography.py
para poder reutilizarlo desde herramientas que no quieren cargar la red de
segmentacion (p.ej. testing/classical_cv_lab.py, que aplica esto mismo sobre
mascaras de color clasicas en vez de mascaras de una red neuronal).

Las funciones no saben de donde viene la mascara -- de una red de segmentacion
o de un umbral de color clasico da igual, solo necesitan un array 2D booleano
o 0/1.
"""
import cv2
import numpy as np

MIN_LINE_PX = 150
MIN_CIRCLE_PX = 80
MAX_ANGLE_GAP_DEG = 150.0   # arco visible debe cubrir al menos 360-esto grados
BORDE_MARGEN_PX = 3


def skeleton_points(mask, n_bins=24):
    """Puntos centrales de la mascara, uniformes a lo largo de su eje largo.

    NO ajustar la recta a los pixeles en bruto: una franja pintada bajo
    perspectiva es mucho mas ancha (mas pixeles) en el lado cercano a camara
    que en el lejano, escorzado. cv2.fitLine sin ponderar deja que el lado
    cercano domine el angulo, y ese pequeño error de pendiente se amplifica
    muchisimo al extrapolar hacia el lado lejano -- confirmado midiendo contra
    el ground truth: error de 2-8px en el extremo con mas masa, 20-100px en el
    otro extremo de la MISMA recta. Aqui se bineable por eje largo y se toma
    la mediana del eje corto en cada bin, así cada tramo de la curva pesa por
    LONGITUD, no por area.
    """
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return None
    horiz = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    eje, otro = (xs, ys) if horiz else (ys, xs)
    edges = np.linspace(eje.min(), eje.max(), n_bins + 1)
    idx = np.clip(np.digitize(eje, edges) - 1, 0, n_bins - 1)
    pts = []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 3:
            continue
        e_med = np.median(eje[sel])
        o_med = np.median(otro[sel])
        pts.append((e_med, o_med) if horiz else (o_med, e_med))
    return np.array(pts, np.float32) if len(pts) >= 4 else None


def fit_line_px(mask):
    """Recta imagen (p,q,r) ajustada a los puntos esqueleto de una clase, o None."""
    if int(mask.sum()) < MIN_LINE_PX:
        return None
    pts = skeleton_points(mask)
    if pts is None:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    # recta por (x0,y0) con direccion (vx,vy) -> forma implicita p*x+q*y+r=0
    p, q = vy, -vx
    r = -(p * x0 + q * y0)
    return np.array([p, q, r]) / (np.hypot(p, q) or 1.0)


def fit_line_ransac(mask, tol_px=4.0, n_iter=400, min_inlier_frac=0.25, seed=0):
    """Recta dominante en una mascara con contenido MIXTO (vallas: 2 rectas +
    4 arcos de esquina, todo en una sola clase) -- RANSAC clasico, no el ajuste
    directo de fit_line_px (ese asume que la mascara es una unica recta limpia,
    aqui no lo es: un fitLine/skeleton sobre todo el blob de vallas mezclaria
    las dos rectas y las esquinas en un resultado sin sentido).
    """
    ys, xs = np.where(mask)
    n = len(xs)
    if n < MIN_LINE_PX:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float64)
    rng = np.random.default_rng(seed)
    muestra = pts if n <= 4000 else pts[rng.choice(n, 4000, replace=False)]
    mejor_in, mejor_line = 0, None
    for _ in range(n_iter):
        i, j = rng.choice(len(muestra), 2, replace=False)
        (x1, y1), (x2, y2) = muestra[i], muestra[j]
        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)
        if norm < 5:
            continue
        p, q = dy / norm, -dx / norm
        r = -(p * x1 + q * y1)
        d = np.abs(p * pts[:, 0] + q * pts[:, 1] + r)
        n_in = int((d < tol_px).sum())
        if n_in > mejor_in:
            mejor_in, mejor_line = n_in, (p, q, r)
    if mejor_line is None or mejor_in < max(30, min_inlier_frac * n):
        return None
    p, q, r = mejor_line
    d = np.abs(p * pts[:, 0] + q * pts[:, 1] + r)
    inl = pts[d < tol_px].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(inl, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    p, q = vy, -vx
    r = -(p * x0 + q * y0)
    return np.array([p, q, r]) / (np.hypot(p, q) or 1.0)


def fit_circle_center(mask, w=None, h=None):
    """Centro de elipse ajustada, o None si el arco visible no es de fiar.

    Un circulo cuyo centro real cae fuera del frame solo deja ver un arco
    parcial, a menudo recortado justo por el borde de la imagen -- ajustar una
    elipse a eso es un problema mal condicionado clasico y puede dar centros a
    cientos de px de distancia (confirmado: 476px en un caso real). Dos filtros
    baratos lo detectan sin necesitar saber la H: (1) el arco visible debe
    rodear el centro ajustado en un rango angular amplio, no un gajo estrecho;
    (2) si la mascara toca el borde de la imagen, sospechoso de recorte.
    """
    ys, xs = np.where(mask)
    if len(xs) < MIN_CIRCLE_PX:
        return None
    if w is not None and h is not None:
        if xs.min() <= BORDE_MARGEN_PX or xs.max() >= w - 1 - BORDE_MARGEN_PX:
            return None
        if ys.min() <= BORDE_MARGEN_PX or ys.max() >= h - 1 - BORDE_MARGEN_PX:
            return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    if len(pts) < 5:
        return None
    (cx, cy), _, _ = cv2.fitEllipse(pts)
    ang = np.sort(np.degrees(np.arctan2(ys - cy, xs - cx)))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 360]))
    if gaps.max() > MAX_ANGLE_GAP_DEG:
        return None
    return np.array([cx, cy])
