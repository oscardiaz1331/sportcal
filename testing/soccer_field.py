"""Campo de futbol: plantilla FIFA 105x68 y dos formas de encontrar la homografia
H (plano del campo -> imagen) a partir de una MASCARA DE LINEAS binaria.

  * busca_pose(): "chamfer sin emparejar lineas". Rejilla de poses de camara
    pinhole (pan, tilt, focal, y unas pocas posiciones tipicas de camara), cada
    una puntuada por lo bien que la plantilla proyectada cae sobre la mascara;
    las mejores se refinan con una H libre de 8 parametros.
  * busca_lineas(): "intersecciones". Se detectan las rectas de la mascara, y
    cada cuadruple de rectas (2 paralelas al eje X del campo + 2 al eje Y) contra
    cada par de lineas de la plantilla da 4 correspondencias de PUNTO (sus 4
    intersecciones) -> una H. Se puntua igual que en la otra via y se refina igual.

Convencion de mundo: metros, origen en el centro del campo, X a lo largo (-52.5 a
52.5), Y a lo ancho (-34 a 34), Z arriba. La camara mira hacia +Y desde el lado
Y<0 (asi el "arriba" de la imagen es +Y). El campo tiene una simetria de 180 grados
(X,Y -> -X,-Y) que ninguna via distingue por las lineas: fijar el lado de la camara
es lo que la rompe, por convencion.

Puntuacion (Dice sobre LONGITUD, en px): 2*M / (V + Mlen), con V la longitud
visible de la plantilla proyectada, M la parte de esa longitud que cae sobre la
mascara (ponderada por distancia a ella, truncada a tau) y Mlen la longitud de
linea de la mascara. Ponderar por longitud en imagen (no por numero de muestras
del mundo) evita que una plantilla diminuta aplastada sobre una zona densa puntue
alto solo por tener muchos puntos juntos.
"""
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

HX, HY = 52.5, 34.0                    # semilargo y semiancho del campo
X_LINEAS = (-52.5, -47.0, -36.0, 0.0, 36.0, 47.0, 52.5)   # rectas X=cte (fondo, area pequeña, area, central)
Y_LINEAS = (-34.0, -20.16, -9.16, 9.16, 20.16, 34.0)      # rectas Y=cte (bandas, lados de areas)


# ---------------------------------------------------------------- plantilla

def _arco(cx, cy, r, a0, a1, n=60):
    t = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def polilineas():
    """{nombre: (N,2) puntos en metros} de todas las lineas pintadas."""
    P = {
        "banda_cercana": [(-HX, -HY), (HX, -HY)], "banda_lejana": [(-HX, HY), (HX, HY)],
        "fondo_izq": [(-HX, -HY), (-HX, HY)], "fondo_der": [(HX, -HY), (HX, HY)],
        "central": [(0, -HY), (0, HY)],
        "circulo_central": _arco(0, 0, 9.15, 0, 2 * np.pi, 90),
    }
    a = np.arccos(5.5 / 9.15)   # el area llega a 5.5 m del punto de penalti: el arco es lo que queda fuera
    for s, nombre in ((-1, "izq"), (1, "der")):
        P["area_" + nombre] = [(s * HX, -20.16), (s * 36.0, -20.16), (s * 36.0, 20.16), (s * HX, 20.16)]
        P["area_peq_" + nombre] = [(s * HX, -9.16), (s * 47.0, -9.16), (s * 47.0, 9.16), (s * HX, 9.16)]
        # arco: centro en el punto de penalti (s*41.5, 0); queda fuera del area hacia el centro del campo
        P["arco_area_" + nombre] = _arco(s * 41.5, 0, 9.15, (np.pi if s == 1 else 0) - a, (np.pi if s == 1 else 0) + a, 40)
    return {k: np.asarray(v, float) for k, v in P.items()}


def muestrea(paso=0.5):
    """Puntos a lo largo de toda la plantilla cada `paso` m. Devuelve pts (N,2) y
    cont (N,) bool: cont[j] = la muestra j continua el segmento de la j-1 (misma
    polilinea), para no medir longitudes que saltan de una linea a otra."""
    pts, cont = [], []
    for pl in polilineas().values():
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
        s = np.arange(0, d[-1] + 1e-9, paso)
        p = np.stack([np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])], 1)
        pts.append(p)
        c = np.ones(len(p), bool)
        c[0] = False
        cont.append(c)
    return np.concatenate(pts), np.concatenate(cont)


# ---------------------------------------------------------------- camara

def pose_a_H(p, w, h):
    """p: (n,6) = cx, cy, cz, pan, tilt, f (rad, px). H (n,3,3): (X,Y,1) -> pixel,
    con den>0 delante de la camara. Camara sin roll, punto principal en el centro."""
    p = np.atleast_2d(p).astype(float)
    cx, cy, cz, pan, tilt, f = p.T
    sp, cp, st, ct = np.sin(pan), np.cos(pan), np.sin(tilt), np.cos(tilt)
    z = np.zeros_like(pan)
    r = np.stack([cp, -sp, z], 1)                         # derecha
    d = np.stack([sp * ct, cp * ct, -st], 1)              # adelante
    b = np.stack([-st * sp, -st * cp, -ct], 1)            # abajo
    R = np.stack([r, b, d], 1)                            # (n,3,3), filas r, abajo, adelante
    C = np.stack([cx, cy, cz], 1)
    t = -np.einsum("nij,nj->ni", R, C)
    M = np.stack([R[:, :, 0], R[:, :, 1], t], 2)
    K = np.zeros((len(p), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2.0, h / 2.0, 1.0
    return K @ M


def proyecta(H, pts):
    """H (n,3,3), pts (m,2) -> xy (n,m,2), den (n,m)."""
    P = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    q = np.einsum("nij,mj->nmi", H, P)
    den = q[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[..., :2] / den[..., None]
    return xy, den


def _bilineal(img, xy):
    x = np.clip(xy[..., 0], 0, img.shape[1] - 1.001)
    y = np.clip(xy[..., 1], 0, img.shape[0] - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - x0, y - y0
    a, b, c, d = img[y0, x0], img[y0, x0 + 1], img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# ---------------------------------------------------------------- solver

def _longitud_lineas(mask):
    """Longitud (px) de las lineas de la mascara = suma de 1/anchura-local sobre sus
    pixeles (anchura = 2 x distancia al fondo): una linea de 2 px de ancho y L de
    largo da ~L, y una mancha llena aporta ~su diametro, no su area. Se probo el
    esqueleto de 1 px y NO sirve: una mancha con huecos (jugador entre ruido) da una
    maraña de bucles y la longitud sale 10x mayor; el contar cresta de la transformada
    subestimaba ~35%. Esta no depende de la topologia."""
    din = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return float(max((1.0 / np.maximum(2.0 * din[mask > 0], 1.0)).sum(), 1.0))


class Campo:
    """Resuelve H para UNA mascara de lineas (uint8 0/1) de tamaño (h, w)."""

    def __init__(self, mask, tau=3.0):
        self.h, self.w = mask.shape
        self.mask = (mask > 0).astype(np.uint8)
        # distancia de cada pixel a la mascara (0 sobre ella)
        self.dt = cv2.distanceTransform((self.mask == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        self.mlen = _longitud_lineas(self.mask)
        self.tau = tau
        self.pen = 0.5
        self.pts_g, self.cont_g = muestrea(1.0)
        self.pts_f, self.cont_f = muestrea(0.25)

    # -- puntuacion ---------------------------------------------------------
    def puntua(self, Hs, fino=False, tau=None, chunk=1500):
        pts, cont = (self.pts_f, self.cont_f) if fino else (self.pts_g, self.cont_g)
        tau = self.tau if tau is None else tau
        Hs = np.asarray(Hs, float).reshape(-1, 3, 3)
        out = np.empty(len(Hs))
        for i in range(0, len(Hs), chunk):
            xy, den = proyecta(Hs[i:i + chunk], pts)
            ok = (den > 1e-6) & (xy[..., 0] >= 0) & (xy[..., 0] <= self.w - 1) \
                & (xy[..., 1] >= 0) & (xy[..., 1] <= self.h - 1)
            seg = ok[:, 1:] & ok[:, :-1] & cont[1:]
            xy = np.where(ok[..., None], xy, 0.0)
            L = np.hypot(xy[:, 1:, 0] - xy[:, :-1, 0], xy[:, 1:, 1] - xy[:, :-1, 1]) * seg
            mid = 0.5 * (xy[:, 1:] + xy[:, :-1])
            f = 1.0 - np.minimum(_bilineal(self.dt, mid) / tau, 1.0)
            M, V = (L * f).sum(1), L.sum(1)
            out[i:i + chunk] = 2.0 * M / (V + self.mlen)
        return out

    # -- via 1: chamfer sobre una rejilla de poses --------------------------
    def poses_rejilla(self):
        pos = [(cx, cy, cz) for cx in (-20, 0, 20) for cy in (-50, -70) for cz in (15, 25)]
        pan = np.radians(np.arange(-50, 51, 4))
        tilt = np.radians([8, 14, 20, 26, 32, 38])
        f = self.w * np.geomspace(0.6, 4.5, 8)
        g = np.array([(*p, a, t, ff) for p in pos for a in pan for t in tilt for ff in f])
        return g

    def busca_pose(self, top=12):
        Hs = pose_a_H(self.poses_rejilla(), self.w, self.h)
        # tau ancho: la rejilla es gruesa (la celda mas cercana queda a ~100 px), con tau
        # de unos pocos px la puntuacion es 0 entre celdas y no hay nada que comparar
        return self._mejores(Hs, self.puntua(Hs, tau=0.04 * self.w), top)

    # -- via 2: cuadruples de rectas -> 4 puntos -> H -------------------------
    def busca_lineas(self, top=8, n_max=7):
        lineas = detecta_lineas(self.mask, n_max)
        if len(lineas) < 4:
            return []
        Hs = hipotesis_lineas(lineas, self.w, self.h)
        if len(Hs) == 0:
            return []
        return self._mejores(Hs, self.puntua(Hs, tau=2.0 * self.tau), top)

    # -- comun: elegir distintas, refinar ------------------------------------
    def _mejores(self, Hs, sc, top, dist_min=0.03):
        """Top-`top` hipotesis DISTINTAS (sus 4 anclas en imagen a >dist_min*w) y refinadas."""
        ancla = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]) * [self.w, self.h]
        orden = np.argsort(-sc)[:400]
        elegidas, firmas = [], []
        for i in orden:
            Hi = np.linalg.inv(Hs[i])
            xy = (Hi @ np.c_[ancla, np.ones(4)].T).T
            with np.errstate(divide="ignore", invalid="ignore"):
                firma = (xy[:, :2] / xy[:, 2:]).ravel()      # donde caen las anclas de la imagen en el MUNDO
            if not np.isfinite(firma).all():
                continue
            if all(np.abs(firma - s).max() > 2.0 for s in firmas):     # >2 m de diferencia
                firmas.append(firma)
                elegidas.append(i)
            if len(elegidas) >= top:
                break
        res = []
        for i in elegidas:
            Hr, s = self.refina(Hs[i])
            res.append({"H": Hr, "score": s, "score0": float(sc[i]), "H0": Hs[i]})
        return sorted(res, key=lambda r: -r["score"])

    def refina(self, H0):
        """Perturba 4 puntos fijos de la imagen (8 params, en px) y maximiza la
        puntuacion fina; dos pasadas, primero con tau ancho para agrandar la cuenca."""
        src = (np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]) * [self.w, self.h]).astype(np.float32)

        def H_de(d):
            P = cv2.getPerspectiveTransform(src, src + d.reshape(4, 2).astype(np.float32))
            return P @ H0

        best = H0
        # cascada de tau: con 0.04*w la puntuacion es suave y recupera 100+ px de error;
        # las ultimas pasadas afinan. Solo la ultima con muestreo fino (mas caro).
        for tau in (0.04 * self.w, 0.015 * self.w, 0.006 * self.w, self.tau):
            fino = tau == self.tau
            # penalizacion pinhole: una H libre de 8 parametros puede "ganar" a la verdad
            # deformandose a algo que ninguna camara produce (medido: puntuacion igual a la de
            # la verdad con 1500-7800 px de error)
            r = minimize(lambda d: -(self.puntua(H_de(d)[None], fino, tau)[0]
                                     - self.pen * residuo_pinhole(H_de(d)[None], self.w, self.h)[1][0]),
                         np.zeros(8),
                         method="Powell", options={"xtol": 0.05, "ftol": 1e-4, "maxfev": 400})
            best = H_de(r.x)
            H0 = best
        return best, float(self.puntua(best[None], True)[0])


# ---------------------------------------------------------------- rectas de la mascara

def detecta_lineas(mask, n_max=7):
    """Rectas dominantes de la mascara: Hough probabilistico -> segmentos agrupados
    por angulo y posicion -> una recta (a,b,c), a*x+b*y+c=0 con a^2+b^2=1, por grupo,
    ordenadas por longitud total. (n,3)."""
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, int(0.03 * w),
                           minLineLength=int(0.06 * w), maxLineGap=int(0.02 * w))
    if segs is None:
        return np.zeros((0, 3))
    S = segs.reshape(-1, 4).astype(float)
    ln = np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1])
    ang = np.arctan2(S[:, 3] - S[:, 1], S[:, 2] - S[:, 0]) % np.pi
    libre = np.ones(len(S), bool)
    grupos = []
    for i in np.argsort(-ln):
        if not libre[i]:
            continue
        n = np.array([-np.sin(ang[i]), np.cos(ang[i])])
        c = -(n @ S[i, :2])
        d1 = np.abs(S[:, :2] @ n + c)
        d2 = np.abs(S[:, 2:] @ n + c)
        da = np.abs((ang - ang[i] + np.pi / 2) % np.pi - np.pi / 2)
        m = libre & (d1 < 0.012 * w) & (d2 < 0.012 * w) & (da < np.radians(4))
        libre &= ~m
        pts = np.concatenate([S[m][:, :2], S[m][:, 2:]]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        nn = np.array([-vy, vx])
        grupos.append((ln[m].sum(), np.array([nn[0], nn[1], -(nn @ [x0, y0])])))
    grupos.sort(key=lambda g: -g[0])
    return np.array([g[1] for g in grupos[:n_max]]) if grupos else np.zeros((0, 3))


def hipotesis_lineas(lineas, w, h, xs=X_LINEAS, ys=Y_LINEAS):
    """Todas las H que salen de asignar 2 rectas de la imagen a 2 rectas X=cte del
    campo y otras 2 a 2 rectas Y=cte (con las dos ordenaciones posibles de cada par).
    Descarta las que reflejan el campo, ponen algun punto detras de la camara o son
    degeneradas. (n,3,3)."""
    N = len(lineas)
    mundoV = [(i, j) for i in range(len(xs)) for j in range(len(xs)) if i != j]
    mundoH = [(k, l) for k in range(len(ys)) for l in range(len(ys)) if k != l]
    wv = np.array([(a, b) for a in mundoV for b in mundoH])                 # (1260, 2, 2) indices
    X = np.array(xs)[wv[:, 0]]      # (1260,2): x de las 2 rectas V
    Y = np.array(ys)[wv[:, 1]]      # (1260,2): y de las 2 rectas H
    # 4 puntos del mundo por hipotesis: (xi,yk), (xi,yl), (xj,yk), (xj,yl)
    Wp = np.stack([np.stack([X[:, 0], Y[:, 0]], 1), np.stack([X[:, 0], Y[:, 1]], 1),
                   np.stack([X[:, 1], Y[:, 0]], 1), np.stack([X[:, 1], Y[:, 1]], 1)], 1)   # (1260,4,2)
    out = []
    for A in itertools.combinations(range(N), 2):
        for B in itertools.combinations(range(N), 2):
            if set(A) & set(B):
                continue
            pi = []
            for a in A:
                for b in B:
                    q = np.cross(lineas[a], lineas[b])
                    pi.append(q)
            pi = np.array(pi)                                # (4,3): a1b1, a1b2, a2b1, a2b2
            if np.abs(pi[:, 2]).min() < 1e-9:
                continue
            uv = pi[:, :2] / pi[:, 2:]
            if not np.isfinite(uv).all() or np.abs(uv).max() > 20 * max(w, h):
                continue
            out.append(_dlt4(Wp, uv))
    if not out:
        return np.zeros((0, 3, 3))
    Hs = np.concatenate(out)
    return _filtra(Hs, w, h)


def _dlt4(Wp, uv):
    """Wp (n,4,2) mundo, uv (4,2) imagen (los mismos 4 puntos para las n) -> H (n,3,3)."""
    n = len(Wp)
    A = np.zeros((n, 8, 8))
    b = np.zeros((n, 8))
    for k in range(4):
        X, Y = Wp[:, k, 0], Wp[:, k, 1]
        u, v = uv[k]
        A[:, 2 * k, 0], A[:, 2 * k, 1], A[:, 2 * k, 2] = X, Y, 1
        A[:, 2 * k, 6], A[:, 2 * k, 7] = -u * X, -u * Y
        A[:, 2 * k + 1, 3], A[:, 2 * k + 1, 4], A[:, 2 * k + 1, 5] = X, Y, 1
        A[:, 2 * k + 1, 6], A[:, 2 * k + 1, 7] = -v * X, -v * Y
        b[:, 2 * k], b[:, 2 * k + 1] = u, v
    det = np.linalg.det(A)
    mal = ~(np.abs(det) > 1e-6 * np.abs(det).max() + 1e-300)
    A[mal] = np.eye(8)
    h8 = np.linalg.solve(A, b[..., None])[..., 0]
    H = np.concatenate([h8, np.ones((n, 1))], 1).reshape(n, 3, 3)
    H[mal] = np.nan
    return H


def residuo_pinhole(Hs, w, h):
    """(f2, mism) por H: f2 = focal^2 que sale de la ortogonalidad de las columnas 1 y 2
    (H = K [r1 r2 t], K con pixeles cuadrados y punto principal en el centro): f^2 =
    -a1.a2 / (z1 z2), con a_i la parte x,y de h_i respecto al centro y z_i su tercera
    componente. mism en [0,1]: cuanto difieren las normas de K^-1 h1 y K^-1 h2 (deben
    ser iguales); 1 si f2 no es valida."""
    a1 = Hs[:, :2, 0] - np.array([w / 2.0, h / 2.0]) * Hs[:, 2:3, 0]
    a2 = Hs[:, :2, 1] - np.array([w / 2.0, h / 2.0]) * Hs[:, 2:3, 1]
    z1, z2 = Hs[:, 2, 0], Hs[:, 2, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = -(a1 * a2).sum(1) / (z1 * z2)
        n1 = (a1 ** 2).sum(1) / f2 + z1 ** 2
        n2 = (a2 ** 2).sum(1) / f2 + z2 ** 2
        mism = np.abs(n1 - n2) / (n1 + n2)
    valido = np.isfinite(f2) & np.isfinite(mism) & (f2 > (0.4 * w) ** 2) & (f2 < (8 * w) ** 2)
    return f2, np.where(valido, np.clip(mism, 0, 1), 1.0)


def consistente_pinhole(Hs, w, h, tol=0.4):
    """H (n,3,3) -> bool (n,): pueden venir de una camara pinhole (residuo_pinhole)
    con la igualdad de normas cumplida a `tol` y f entre 0.4 y 8 anchos."""
    return residuo_pinhole(Hs, w, h)[1] < tol


def _filtra(Hs, w, h):
    """Quita NaN, campos reflejados, H con el centro del campo detras de la camara y
    las que no pueden ser una camara pinhole razonable (consistente_pinhole)."""
    Hs = Hs[np.isfinite(Hs).all((1, 2))]
    if len(Hs) == 0:
        return Hs
    # signo: el punto del mundo (0,0) delante de la camara si den>0; probamos con 2 puntos
    c = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    xy, den = proyecta(Hs, c)
    s = np.sign(den[:, 0])
    s[s == 0] = 1
    Hs = Hs * s[:, None, None]
    xy, den = proyecta(Hs, c)
    # orientacion: +X hacia la derecha de la imagen y +Y hacia ARRIBA (v decrece): cruce < 0
    dx, dy = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    cruce = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    ok = np.isfinite(cruce) & (cruce < 0) & (den > 0).all(1)
    Hs = Hs[ok]
    return Hs[consistente_pinhole(Hs, w, h)]


# ---------------------------------------------------------------- evaluacion

def error_reproy(H_est, H_gt, w, h, paso=1.0):
    """Mediana (px) de la distancia entre donde H_est y H_gt proyectan los puntos de
    la plantilla que H_gt deja DENTRO de la imagen. 1e4 si no hay ninguno."""
    pts, _ = muestrea(paso)
    xy_g, den_g = proyecta(H_gt[None], pts)
    xy_e, den_e = proyecta(H_est[None], pts)
    ok = (den_g[0] > 0) & (xy_g[0, :, 0] >= 0) & (xy_g[0, :, 0] < w) & (xy_g[0, :, 1] >= 0) & (xy_g[0, :, 1] < h)
    if ok.sum() < 5:
        return 1e4
    e = np.hypot(*(xy_e[0, ok] - xy_g[0, ok]).T)
    e = np.where((den_e[0, ok] > 0) & np.isfinite(e), e, 1e4)
    return float(np.median(e))
