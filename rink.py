"""Template de la pista para los 56 keypoints del modelo HockeyRink.

El modelo (SimulaMet-HOST/HockeyRink) no documenta que representa cada indice,
asi que la semantica se dedujo de las 661 anotaciones del dataset: alineando
todas las vistas (todo keypoint esta en el plano del hielo, luego cada vista es
una homografia del mismo layout) y contrastando el resultado con la geometria
real de una pista. Comprobaciones que confirmaron la lectura:

    postes 50/51            -> +-0.915 m (portria real 1.83 m)
    apice area de arbitro   -> 26.98 m frente a 26.95 esperado
    hash marks 36-39/42-45  -> separacion 1.70 m = 5 ft 7 in exactos
    trapecio 49/52 y 54/55  -> +-3.46 y +-4.48 m (reglamento +-3.35 y +-4.27)

OJO con las dimensiones: el dataset es de la liga sueca (IIHF, 60x30 m) pero un
partido NHL usa 60.96x25.91 m, con la linea de gol a 3.35 m del fondo en vez de
4 m y los puntos de faceoff a 13.41 m entre si en vez de 14 m. Usar el template
equivocado descuadra la homografia, por eso aqui van los dos.

Bloques de indices:
    0-19  zona A (un fondo)      36-55 zona B (el fondo opuesto)
    20-35 zona neutral
La zona A es el espejo de la B en x (no una rotacion de 180 grados), con la
correspondencia   A[i] = B[i+46] para i en 0..9   y   A[i] = B[i+26] para 10..19.
"""

import numpy as np

FT = 0.3048  # pie -> metro

# Espejo izquierda<->derecha de los 56 keypoints: MIRROR[i] es el gemelo de i en
# la otra mitad de la pista. Es el mismo array que flip_idx en hockeyrink_pose.yaml
# y POSE_FLIP_IDX en training/prepare_data.py, y debe mantenerse identico a ambos.
#
# Comprobado contra las 955 anotaciones ajustando una homografia por imagen y
# midiendo el residuo de reproyeccion sobre las imagenes que ven la zona A:
#
#     mapeo               inliers a 5px    error mediano
#     i+46 / i+26            36%              111 px     <- lo que habia
#     MIRROR (este)          50%              5.6 px
#
# (en datasets/hockeyrink_nhl, 54% -> 91% de inliers). Las formulas i+46 / i+26
# solo aciertan en 2..7 y 14,15; el resto de la zona A caia a 3-6 m de su sitio,
# que es lo que descuadraba findHomography aguas abajo.
MIRROR = [54, 55, 48, 49, 50, 51, 52, 53, 46, 47, 42, 43, 44, 45, 40, 41, 36, 37, 38, 39,
          34, 35, 32, 33, 24, 25, 26, 27, 31, 29, 30, 28, 22, 23, 20, 21,
          16, 17, 18, 19, 14, 15, 10, 11, 12, 13, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1]

RINK_NHL = dict(
    length=200 * FT, width=85 * FT, corner_r=28 * FT,
    goal_line_from_end=11 * FT, blue_from_end=75 * FT,
    circle_r=15 * FT, hash_half=(5 + 7 / 12) * FT / 2,
    dot_from_goal_line=20 * FT, dot_from_axis=22 * FT,
    neutral_dot_from_blue=5 * FT,
    post_half=3 * FT, ref_crease_r=10 * FT,
    trapezoid_at_goal=11 * FT, trapezoid_at_end=14 * FT,
    crease_half=4 * FT, crease_depth=4.5 * FT,
)

RINK_IIHF = dict(
    length=60.0, width=30.0, corner_r=8.5,
    goal_line_from_end=4.0, blue_from_end=22.86,
    circle_r=4.5, hash_half=0.85,
    dot_from_goal_line=6.0, dot_from_axis=7.0,
    neutral_dot_from_blue=1.5,
    post_half=0.915, ref_crease_r=3.05,
    trapezoid_at_goal=3.35, trapezoid_at_end=4.27,
    crease_half=1.22, crease_depth=1.57,
)

# Ajustado sobre 84 frames de camara principal de clip.mp4.webm (Bruins-Panthers).
# Parte de las medidas NHL -- el ancho de 25.91 m se confirmo barriendolo: 30 m
# empeora el ajuste, o sea que la pista del video es NHL -- pero dos parametros
# se desvian del reglamento porque el modelo esta entrenado en pistas IIHF y
# arrastra ese sesgo al colocar los keypoints. Error mediano de reproyeccion:
#     NHL puro 16.0 px | IIHF puro 10.9 px | este 6.3 px
RINK_NHL_FITTED = dict(
    RINK_NHL,
    goal_line_from_end=4.0,      # NHL: 3.35
    dot_from_goal_line=5.8,      # NHL: 6.10
    dot_from_axis=7.0,           # NHL: 6.71
    hash_half=0.75,              # NHL: 0.85
)


def build_template(p=RINK_NHL):
    """Devuelve un array (56,2) con la posicion en metros de cada keypoint.

    Origen en la esquina inferior izquierda; x a lo largo de la pista.
    """
    L, W = p["length"], p["width"]
    cy = W / 2
    T = np.full((56, 2), np.nan)

    goal_x = L - p["goal_line_from_end"]           # linea de gol de la zona B
    dot_x = goal_x - p["dot_from_goal_line"]
    dy = p["dot_from_axis"]
    dot_lo, dot_hi = cy - dy, cy + dy
    # las hash marks tocan el circulo, separadas 5'7" entre si
    hw = p["hash_half"]
    ha = np.sqrt(p["circle_r"] ** 2 - hw ** 2)

    # valla curvada a la altura de la linea de gol
    r = p["corner_r"]
    yb = r - np.sqrt(max(r ** 2 - (r - p["goal_line_from_end"]) ** 2, 0.0))

    B = {
        46: (goal_x - p["crease_depth"], cy - p["crease_half"]),   # esquinas del area de portera
        47: (goal_x - p["crease_depth"], cy + p["crease_half"]),
        48: (goal_x, yb),                                          # linea de gol contra la valla
        49: (goal_x, cy - p["trapezoid_at_goal"]),                 # esquinas del trapecio
        50: (goal_x, cy - p["post_half"]),                         # postes
        51: (goal_x, cy + p["post_half"]),
        52: (goal_x, cy + p["trapezoid_at_goal"]),
        53: (goal_x, W - yb),
        54: (L, cy - p["trapezoid_at_end"]),                       # trapecio contra el fondo
        55: (L, cy + p["trapezoid_at_end"]),
        36: (dot_x - hw, dot_lo - ha), 37: (dot_x - hw, dot_lo + ha),
        38: (dot_x - hw, dot_hi - ha), 39: (dot_x - hw, dot_hi + ha),
        42: (dot_x + hw, dot_lo - ha), 43: (dot_x + hw, dot_lo + ha),
        44: (dot_x + hw, dot_hi - ha), 45: (dot_x + hw, dot_hi + ha),
        40: (dot_x, dot_lo), 41: (dot_x, dot_hi),                  # puntos de faceoff de zona
    }
    for k, v in B.items():
        T[k] = v

    # zona A = espejo en x de la zona B, con la correspondencia de MIRROR (no las
    # formulas i+46 / i+26, que estaban mal en 0,1,8,9 y en 10-13/16-19).
    for a, b in enumerate(MIRROR[:20]):
        T[a] = (L - T[b, 0], T[b, 1])

    # zona neutral
    blue_lo, blue_hi = p["blue_from_end"], L - p["blue_from_end"]
    ndot = p["neutral_dot_from_blue"]
    cx = L / 2
    rc = p["ref_crease_r"]
    N = {
        20: (blue_lo, 0.0), 21: (blue_lo, W),                      # lineas azules contra la valla
        34: (blue_hi, 0.0), 35: (blue_hi, W),
        # los faceoff dots neutrales van DENTRO de la zona neutral, no fuera:
        # con el signo invertido caian a 3.03 m de su sitio real, asi a 0.03 m.
        22: (blue_lo + ndot, dot_lo), 23: (blue_lo + ndot, dot_hi), # puntos de faceoff neutrales
        32: (blue_hi - ndot, dot_lo), 33: (blue_hi - ndot, dot_hi),
        24: (cx, 0.0), 30: (cx, W),                                # linea central contra la valla
        25: (cx, cy - p["circle_r"]), 27: (cx, cy + p["circle_r"]),  # circulo central
        26: (cx, cy),                                              # punto central
        28: (cx - rc, W), 31: (cx + rc, W),                        # area del arbitro
        29: (cx, W - rc),
    }
    for k, v in N.items():
        T[k] = v
    return T


KEYPOINT_NAMES = {
    **{k: "esquina area portera" for k in (0, 1, 46, 47)},
    **{k: "linea de gol / valla" for k in (2, 7, 48, 53)},
    **{k: "trapecio en linea de gol" for k in (3, 6, 49, 52)},
    **{k: "poste" for k in (4, 5, 50, 51)},
    **{k: "trapecio en el fondo" for k in (8, 9, 54, 55)},
    **{k: "hash mark" for k in (10, 11, 12, 13, 16, 17, 18, 19, 36, 37, 38, 39, 42, 43, 44, 45)},
    **{k: "punto de faceoff de zona" for k in (14, 15, 40, 41)},
    **{k: "linea azul / valla" for k in (20, 21, 34, 35)},
    **{k: "punto de faceoff neutral" for k in (22, 23, 32, 33)},
    **{k: "linea central / valla" for k in (24, 30)},
    **{k: "circulo central" for k in (25, 27)},
    26: "punto central",
    **{k: "area del arbitro" for k in (28, 29, 31)},
}


def minimap_base(p=RINK_NHL_FITTED, scale=9, margin=12, bg=(245, 245, 245)):
    """Crea el lienzo cenital de la pista y la funcion metros -> pixel del minimapa."""
    import cv2  # noqa: F401  (draw_rink lo necesita)

    w = int(round(p["length"] * scale)) + 2 * margin
    h = int(round(p["width"] * scale)) + 2 * margin
    img = np.full((h, w, 3), bg, np.uint8)

    def to_img(pts):
        q = np.asarray(pts, dtype=float).reshape(-1, 2)
        return np.stack([q[:, 0] * scale + margin, q[:, 1] * scale + margin], 1)

    draw_rink(img, to_img, p, (150, 150, 150), 1)
    return img, to_img


def draw_rink(img, to_img, p=RINK_NHL, color=(70, 70, 70), thickness=2):
    """Dibuja las lineas de la pista sobre `img`, transformadas por `to_img`.

    `to_img` mapea metros -> pixeles: o una homografia 3x3 (vista de camara) o
    un callable para una vista cenital.
    """
    import cv2

    L, W = p["length"], p["width"]
    r, cy = p["corner_r"], W / 2
    gl = [p["goal_line_from_end"], L - p["goal_line_from_end"]]
    bl = [p["blue_from_end"], L - p["blue_from_end"]]

    def seg(a, b, n=60):
        t = np.linspace(0, 1, n)[:, None]
        return np.array(a) * (1 - t) + np.array(b) * t

    def arc(cx_, cy_, rad, a0, a1, n=40):
        t = np.radians(np.linspace(a0, a1, n))
        return np.stack([cx_ + rad * np.cos(t), cy_ + rad * np.sin(t)], 1)

    polys = [
        seg((r, 0), (L - r, 0)), seg((r, W), (L - r, W)),
        seg((0, r), (0, W - r)), seg((L, r), (L, W - r)),
        arc(r, r, r, 180, 270), arc(L - r, r, r, 270, 360),
        arc(L - r, W - r, r, 0, 90), arc(r, W - r, r, 90, 180),
        seg((L / 2, 0), (L / 2, W)),
        arc(L / 2, cy, p["circle_r"], 0, 360),
    ]
    for x in gl + bl:
        polys.append(seg((x, 0), (x, W)))
    for x in [gl[0] + p["dot_from_goal_line"], gl[1] - p["dot_from_goal_line"]]:
        for y in [cy - p["dot_from_axis"], cy + p["dot_from_axis"]]:
            polys.append(arc(x, y, p["circle_r"], 0, 360))

    for poly in polys:
        pts = poly.astype(np.float32).reshape(-1, 1, 2)
        if callable(to_img):
            out = to_img(poly).reshape(-1, 2)
        else:
            out = cv2.perspectiveTransform(pts, to_img).reshape(-1, 2)
        ok = np.isfinite(out).all(1)
        out = out[ok]
        if len(out) > 1:
            cv2.polylines(img, [out.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return img
