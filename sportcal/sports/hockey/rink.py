"""Hockey rink template: the 56 keypoints of the HockeyRink model and the line classes.

The public model (SimulaMet-HOST/HockeyRink) does not document what each keypoint
index means; the semantics were deduced from its 661 annotations by aligning all
views (every keypoint lies on the ice plane, so each view is a homography of the
same layout) and checking against real rink geometry (docs/experiments/hockey.md section 1).

DIMENSIONS - two standards, on purpose: the dataset is Swedish (IIHF, 60x30 m) but an
NHL game uses 60.96x25.91 m, the goal line 3.35 m from the end wall instead of 4 m,
and faceoff dots 13.41 m apart instead of 14 m. Using the wrong template skews the
homography, so both are kept (RINK_NHL, RINK_IIHF).

Keypoint blocks:
    0-19  zone A (one end)       36-55  zone B (the opposite end)
    20-35 neutral zone
Zone A is the x-mirror of zone B (NOT a 180-degree rotation), through MIRROR.

World frame: origin at the bottom-left corner, x along the length, metres.
"""
import numpy as np

FT = 0.3048  # feet -> metres

# Left<->right mirror of the 56 keypoints: MIRROR[i] is the twin of i in the other
# half of the rink. It must stay identical to `flip_idx` in configs/hockey/*.yaml.
# (Verified against 955 annotations: 5.6 px median reprojection error, versus 111 px
# for the i+46 / i+26 formulas it replaced.)
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

# Line classes of the segmentation task. Zones A/B are the x-low / x-high halves, as in
# build_template. Left and right are kept apart on purpose: a blue line without its
# side is useless as a homography correspondence. Do NOT confuse these indices with the
# 0-55 keypoint indices.
CLASSES = [
    "background",     # 0
    "boards",         # 1  whole perimeter (straight sides + corner arcs)
    "goal_line_A",    # 2
    "goal_line_B",    # 3
    "blue_line_A",    # 4
    "blue_line_B",    # 5
    "center_line",    # 6
    "center_circle",  # 7
    "faceoff_A_lo",   # 8
    "faceoff_A_hi",   # 9
    "faceoff_B_lo",   # 10
    "faceoff_B_hi",   # 11
]


def build_template(p=RINK_NHL):
    """(56, 2) array with the world position in metres of every keypoint."""
    L, W = p["length"], p["width"]
    cy = W / 2
    T = np.full((56, 2), np.nan)

    goal_x = L - p["goal_line_from_end"]  # zone B goal line
    dot_x = goal_x - p["dot_from_goal_line"]
    dy = p["dot_from_axis"]
    dot_lo, dot_hi = cy - dy, cy + dy
    # hash marks touch the circle and sit 5'7" apart
    hw = p["hash_half"]
    ha = np.sqrt(p["circle_r"] ** 2 - hw ** 2)

    # curved board at the height of the goal line
    r = p["corner_r"]
    yb = r - np.sqrt(max(r ** 2 - (r - p["goal_line_from_end"]) ** 2, 0.0))

    B = {
        46: (goal_x - p["crease_depth"], cy - p["crease_half"]),  # crease corners
        47: (goal_x - p["crease_depth"], cy + p["crease_half"]),
        48: (goal_x, yb),                                         # goal line meets the boards
        49: (goal_x, cy - p["trapezoid_at_goal"]),                # trapezoid corners
        50: (goal_x, cy - p["post_half"]),                        # posts
        51: (goal_x, cy + p["post_half"]),
        52: (goal_x, cy + p["trapezoid_at_goal"]),
        53: (goal_x, W - yb),
        54: (L, cy - p["trapezoid_at_end"]),                      # trapezoid on the end wall
        55: (L, cy + p["trapezoid_at_end"]),
        36: (dot_x - hw, dot_lo - ha), 37: (dot_x - hw, dot_lo + ha),
        38: (dot_x - hw, dot_hi - ha), 39: (dot_x - hw, dot_hi + ha),
        42: (dot_x + hw, dot_lo - ha), 43: (dot_x + hw, dot_lo + ha),
        44: (dot_x + hw, dot_hi - ha), 45: (dot_x + hw, dot_hi + ha),
        40: (dot_x, dot_lo), 41: (dot_x, dot_hi),                 # zone faceoff dots
    }
    for k, v in B.items():
        T[k] = v

    # zone A = x-mirror of zone B through MIRROR
    for a, b in enumerate(MIRROR[:20]):
        T[a] = (L - T[b, 0], T[b, 1])

    blue_lo, blue_hi = p["blue_from_end"], L - p["blue_from_end"]
    ndot = p["neutral_dot_from_blue"]
    cx = L / 2
    rc = p["ref_crease_r"]
    N = {
        20: (blue_lo, 0.0), 21: (blue_lo, W),                     # blue lines meet the boards
        34: (blue_hi, 0.0), 35: (blue_hi, W),
        # neutral faceoff dots sit INSIDE the neutral zone
        22: (blue_lo + ndot, dot_lo), 23: (blue_lo + ndot, dot_hi),
        32: (blue_hi - ndot, dot_lo), 33: (blue_hi - ndot, dot_hi),
        24: (cx, 0.0), 30: (cx, W),                               # centre line meets the boards
        25: (cx, cy - p["circle_r"]), 27: (cx, cy + p["circle_r"]),  # centre circle
        26: (cx, cy),                                             # centre dot
        28: (cx - rc, W), 31: (cx + rc, W),                       # referee crease
        29: (cx, W - rc),
    }
    for k, v in N.items():
        T[k] = v
    return T


KEYPOINT_NAMES = {
    **{k: "crease corner" for k in (0, 1, 46, 47)},
    **{k: "goal line / boards" for k in (2, 7, 48, 53)},
    **{k: "trapezoid at goal line" for k in (3, 6, 49, 52)},
    **{k: "post" for k in (4, 5, 50, 51)},
    **{k: "trapezoid at end wall" for k in (8, 9, 54, 55)},
    **{k: "hash mark" for k in (10, 11, 12, 13, 16, 17, 18, 19, 36, 37, 38, 39, 42, 43, 44, 45)},
    **{k: "zone faceoff dot" for k in (14, 15, 40, 41)},
    **{k: "blue line / boards" for k in (20, 21, 34, 35)},
    **{k: "neutral faceoff dot" for k in (22, 23, 32, 33)},
    **{k: "center line / boards" for k in (24, 30)},
    **{k: "center circle" for k in (25, 27)},
    26: "center dot",
    **{k: "referee crease" for k in (28, 29, 31)},
}


# --------------------------------------------------------------- line geometry

def seg(a, b, n=80):
    """Straight polyline from a to b with n points."""
    t = np.linspace(0, 1, n)[:, None]
    return np.array(a, float) * (1 - t) + np.array(b, float) * t


def arc(cx, cy, rad, a0, a1, n=80):
    """Circular arc from angle a0 to a1 (degrees) with n points."""
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rad * np.cos(t), cy + rad * np.sin(t)], 1)


def rink_polylines(p):
    """[(class index, world polyline in metres), ...] for every painted line."""
    L, W = p["length"], p["width"]
    r, cy = p["corner_r"], W / 2
    gl = [p["goal_line_from_end"], L - p["goal_line_from_end"]]
    bl = [p["blue_from_end"], L - p["blue_from_end"]]
    dot, dy = p["dot_from_goal_line"], p["dot_from_axis"]
    return [
        (1, seg((r, 0), (L - r, 0))), (1, seg((r, W), (L - r, W))),
        (1, seg((0, r), (0, W - r))), (1, seg((L, r), (L, W - r))),
        (1, arc(r, r, r, 180, 270)), (1, arc(L - r, r, r, 270, 360)),
        (1, arc(L - r, W - r, r, 0, 90)), (1, arc(r, W - r, r, 90, 180)),
        (2, seg((gl[0], 0), (gl[0], W))), (3, seg((gl[1], 0), (gl[1], W))),
        (4, seg((bl[0], 0), (bl[0], W))), (5, seg((bl[1], 0), (bl[1], W))),
        (6, seg((L / 2, 0), (L / 2, W))),
        (7, arc(L / 2, cy, p["circle_r"], 0, 360)),
        (8, arc(gl[0] + dot, cy - dy, p["circle_r"], 0, 360)),
        (9, arc(gl[0] + dot, cy + dy, p["circle_r"], 0, 360)),
        (10, arc(gl[1] - dot, cy - dy, p["circle_r"], 0, 360)),
        (11, arc(gl[1] - dot, cy + dy, p["circle_r"], 0, 360)),
    ]


# --------------------------------------------------------------- drawing

def draw_rink(img, to_img, p=RINK_NHL, color=(70, 70, 70), thickness=2):
    """Draw the rink lines on `img`, transformed by `to_img`.

    `to_img` maps metres -> pixels: either a 3x3 homography (camera view) or a callable
    for a top-down view.
    """
    import cv2

    for _, poly in rink_polylines(p):
        if callable(to_img):
            out = to_img(poly).reshape(-1, 2)
        else:
            out = cv2.perspectiveTransform(
                poly.astype(np.float32).reshape(-1, 1, 2), to_img).reshape(-1, 2)
        out = out[np.isfinite(out).all(1)]
        if len(out) > 1:
            cv2.polylines(img, [out.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return img


def minimap_base(p=RINK_NHL, scale=9, margin=12, bg=(245, 245, 245)):
    """Blank top-down rink canvas and its metres -> minimap-pixel function."""
    w = int(round(p["length"] * scale)) + 2 * margin
    h = int(round(p["width"] * scale)) + 2 * margin
    img = np.full((h, w, 3), bg, np.uint8)

    def to_img(pts):
        q = np.asarray(pts, dtype=float).reshape(-1, 2)
        return np.stack([q[:, 0] * scale + margin, q[:, 1] * scale + margin], 1)

    draw_rink(img, to_img, p, (150, 150, 150), 1)
    return img, to_img
