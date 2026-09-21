"""Playing-surface segmentation and line-response maps. Sport-agnostic: the sport only
chooses which colour model defines "the surface" (`surface="ice"` or `"grass"`).

Two separate things are estimated, and mixing them up is the classic mistake:
  * the SURFACE (colour): pixels that look like ice / grass. Painted lines are more
    saturated than ice, so the very threshold that isolates the ice expels the lines.
  * the PLAY REGION (area): the connected area the pitch occupies, lines and players
    included (`play_region`). Its outline is the boards / touchlines.

Measured comparisons between the methods below live in docs/experiments/hockey.md
section 3; nothing here should quote numbers.
"""
import cv2
import numpy as np

SURFACES = ("ice", "grass")

# Fixed HSV presets (OpenCV ranges: H 0-179, S/V 0-255).
ICE_S_MAX = 60      # ice is pale and neutral: low saturation...
ICE_V_MIN = 180     # ...and bright
GRASS_HUE = (35, 85)  # pure green is H=60
GRASS_S_MIN = 40    # excludes greys/whites
GRASS_V_MIN = 30    # excludes black shadows

# name -> (OpenCV conversion, channel letters, lightness channel index, chroma channel indices).
# Letters are single distinct characters per space so channels can be requested as
# "ab" or "l". In hsv the hue is circular and a Gaussian cannot model that (red sits at
# ~0 and ~179), so request channels="sv" there.
COLOR_SPACES = {
    "lab": (cv2.COLOR_BGR2LAB, "lab", 0, (1, 2)),
    "luv": (cv2.COLOR_BGR2LUV, "luv", 0, (1, 2)),
    "yuv": (cv2.COLOR_BGR2YUV, "yuv", 0, (1, 2)),
    "hsv": (cv2.COLOR_BGR2HSV, "hsv", 2, (1,)),
}


def _check_surface(surface):
    if surface not in SURFACES:
        raise ValueError("unsupported surface {!r}, use one of {}".format(surface, SURFACES))


# --------------------------------------------------------------- surface (colour)

def ice_hsv(img, s_max=ICE_S_MAX, v_min=ICE_V_MIN):
    """Fixed HSV threshold for ice. Calibrated for ONE arena; the baseline to beat."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] < s_max) & (hsv[:, :, 2] > v_min)).astype(np.uint8)


def grass_hsv(img, h_min=GRASS_HUE[0], h_max=GRASS_HUE[1], s_min=GRASS_S_MIN, v_min=GRASS_V_MIN):
    """Fixed HSV threshold on the green hue. Lines, players and ads fall outside."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 0] >= h_min) & (hsv[:, :, 0] <= h_max)
            & (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min)).astype(np.uint8)


def hsv_threshold(img, surface="ice"):
    """The sport's default fixed threshold (also the fallback when a model cannot fit)."""
    _check_surface(surface)
    return grass_hsv(img) if surface == "grass" else ice_hsv(img)


def gmm_surface(img, k=5, sample=30000, seed=0, surface="ice"):
    """Surface mask from a Lab Gaussian mixture, with no absolute thresholds.

    The GMM is fitted on a quarter-resolution copy (enough to find the colour modes).
    ALL matching components are kept, not just the largest: ice often splits into two
    or three modes (uneven lighting, scuffed areas) and striped grass into several
    lightness modes. Ice = bright, neutral components; grass = components whose centre
    hue falls in GRASS_HUE. Returns (mask, info dict or None if the fit failed).
    """
    _check_surface(surface)
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
        return hsv_threshold(img, surface), None

    means = em.getMeans()
    weights = em.getWeights().ravel()
    L = means[:, 0]
    chroma = np.hypot(means[:, 1] - 128.0, means[:, 2] - 128.0)

    if surface == "grass":
        centres = np.clip(means, 0, 255).astype(np.uint8).reshape(1, -1, 3)
        hsv_c = cv2.cvtColor(cv2.cvtColor(centres, cv2.COLOR_LAB2BGR),
                             cv2.COLOR_BGR2HSV)[0].astype(np.float64)
        green = (hsv_c[:, 0] >= GRASS_HUE[0]) & (hsv_c[:, 0] <= GRASS_HUE[1]) \
            & (hsv_c[:, 1] >= GRASS_S_MIN)
        if green.any():
            sel = np.where(green)[0]
        else:  # none is green: take the component whose hue is closest
            sel = np.array([int(np.argmin(np.abs(hsv_c[:, 0] - np.mean(GRASS_HUE))))])
    else:
        # the most "ice-like" component (bright, neutral, heavy), plus every component
        # close to it in lightness that is still neutral: recovers shaded ice without
        # fixing an absolute threshold
        score = weights * (L / 255.0) ** 2 * np.exp(-chroma / 8.0)
        anchor = int(np.argmax(score))
        sel = np.where((L > L[anchor] - 30) & (chroma < max(12.0, chroma[anchor] + 4)))[0]
        if anchor not in sel:
            sel = np.append(sel, anchor)

    # per-pixel posterior over the whole image, at half resolution
    half = cv2.resize(lab, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    Y = half.reshape(-1, 3).astype(np.float64)
    _, labels = em.predict(Y)
    lab_idx = labels.ravel().astype(np.int32) if labels.ndim > 1 else labels.astype(np.int32)
    if lab_idx.size != len(Y):  # predict returned log-likelihoods instead of labels
        lab_idx = np.argmax(labels, axis=1).astype(np.int32)
    mask_half = np.isin(lab_idx, sel).reshape(half.shape[:2]).astype(np.uint8)
    mask = cv2.resize(mask_half, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    var = np.asarray(em.getCovs())
    var = var.diagonal(axis1=1, axis2=2) if var.ndim == 3 else var
    info = {"L": L.round(0), "chroma": chroma.round(1), "w": weights.round(3), "sel": sel,
            "means": means, "var": var}
    return mask, info


def robust_surface(img, chi2=9.0, iters=4, space="lab", channels=None, info=False, surface="ice"):
    """Surface mask from ONE robust Gaussian fitted in `space`, re-seeded by percentiles.

    Equivalent to one-component EM with hard assignment. The seed uses percentiles of
    the image itself (bright + neutral relative to THIS image), which is what adapts it
    across arenas without recalibration. For grass the seed is the green HSV mask.

    space:    key of COLOR_SPACES.
    channels: which channels enter the Gaussian, as letters ("ab", "l") or a list; None
              = all. The seed always uses the space's lightness and chroma regardless.
    chi2:     threshold on the Mahalanobis distance, whose degrees of freedom are the
              number of channels (9.0 is ~97% with 3 channels, ~99% with 2, ~99.7% with 1).
    info:     also return the fitted model (dict: space, idx, mu, cov, chi2) or None if it
              fell back to the HSV threshold.
    """
    _check_surface(surface)
    if space not in COLOR_SPACES:
        raise ValueError("unsupported space {!r}, use one of {}".format(space, sorted(COLOR_SPACES)))
    code, letters, i_light, i_chroma = COLOR_SPACES[space]
    if channels is None:
        idx = list(range(3))
    else:
        asked = [c.lower() for c in channels]
        bad = [c for c in asked if c not in letters]
        if bad or not asked or len(set(asked)) != len(asked):
            raise ValueError("invalid channels {!r} for {!r} (letters: {})".format(
                channels, space, letters))
        idx = sorted(letters.index(c) for c in asked)

    conv = cv2.cvtColor(img, code).astype(np.float32)
    X = conv.reshape(-1, 3)
    # "neutral" = mid grey converted to this space, not a fixed constant
    grey = cv2.cvtColor(np.full((1, 1, 3), 128, np.uint8), code)[0, 0].astype(np.float32)
    L = X[:, i_light]
    chroma = np.sqrt(sum((X[:, i] - grey[i]) ** 2 for i in i_chroma))
    if surface == "grass":
        m = grass_hsv(img).ravel().astype(bool)
    else:
        m = (L > np.percentile(L, 55)) & (chroma < np.percentile(chroma, 45))
    if m.sum() < 1000:
        base = hsv_threshold(img, surface)
        return (base, None) if info else base
    Xs = X[:, idx]
    k = len(idx)
    model = None
    for _ in range(iters):
        mu = Xs[m].mean(0)
        cov = np.atleast_2d(np.cov(Xs[m].T)) + np.eye(k) * 2.0  # regularised: ice is nearly degenerate
        d = Xs - mu
        maha = np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
        new = maha < chi2
        if model is None or new.sum() >= 0.02 * len(X):
            model = {"space": space, "idx": idx, "mu": mu, "cov": cov, "chi2": chi2}
        if new.sum() < 0.02 * len(X):
            break
        m = new
    mask = m.reshape(conv.shape[:2]).astype(np.uint8)
    return (mask, model) if info else mask


# --------------------------------------------------------------- play region (area)

def play_region(surface_mask, k_frac=0.012, frac_keep=0.20):
    """From "surface pixels" to "play region": close, group and fill.

    Lines and players are holes inside the surface, not background. Filling them turns
    a COLOUR mask into an AREA mask, whose outline is the boards. Every component with
    at least `frac_keep` of the largest one's area is kept (a big centre logo can split
    the pitch in two; keeping only the largest loses half of it).
    """
    h, w = surface_mask.shape
    k = max(3, int(round(w * k_frac)) | 1)
    m = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(surface_mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.where(areas >= frac_keep * areas.max())[0]
    out = np.isin(lbl, keep).astype(np.uint8)
    flooded = out.copy()  # fill holes: flood from outside, keep what was not reached
    cv2.floodFill(flooded, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (out | (1 - flooded)).astype(np.uint8)


def solidity(mask):
    """Area / convex-hull area. A pitch is convex, so a low value means a broken region."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    hull_area = cv2.contourArea(cv2.convexHull(np.vstack(contours)))
    return float(np.count_nonzero(mask) / hull_area) if hull_area > 0 else 0.0


def best_region(img, surface="ice"):
    """Play region from the fixed threshold AND the GMM, keeping the more solid one.

    Neither wins alone (GMM: better median, worse tail; threshold: the opposite). The
    typical failure - keeping half the pitch because a logo split it - yields a
    non-convex region, which solidity detects without any ground truth.
    """
    region_hsv = play_region(hsv_threshold(img, surface))
    region_gmm = play_region(gmm_surface(img, surface=surface)[0])
    return region_gmm if solidity(region_gmm) >= solidity(region_hsv) else region_hsv


# --------------------------------------------------------------- line response

def local_chroma(img, win=61):
    """Colour deviation from the LOCAL surface: (da, db) in Lab.

    A large-window median estimates the surface colour because the line is a minority
    inside the window. Subtracting it cancels white balance, grading and lighting,
    which is what makes absolute hue useless.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    win = max(11, int(win) | 1)
    a_loc = cv2.medianBlur(lab[:, :, 1], win).astype(np.float32)
    b_loc = cv2.medianBlur(lab[:, :, 2], win).astype(np.float32)
    return lab[:, :, 1].astype(np.float32) - a_loc, lab[:, :, 2].astype(np.float32) - b_loc


def local_l(img, win=61):
    """Same as local_chroma but on lightness. The signal for WHITE lines on grass, which
    are near-neutral in colour (painted hockey lines are the opposite: colour on white)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    win = max(11, int(win) | 1)
    l_loc = cv2.medianBlur(lab[:, :, 0], win).astype(np.float32)
    return lab[:, :, 0].astype(np.float32) - l_loc


def ridge(resp, scales=(1.5, 3.0, 6.0, 10.0)):
    """Multi-scale ridge filter (normalised Laplacian). Visualisation only: it did not
    beat the raw response on the localisation metric (docs/experiments/hockey.md)."""
    out = np.zeros_like(resp)
    for s in scales:
        g = cv2.GaussianBlur(resp, (0, 0), s)
        out = np.maximum(out, -cv2.Laplacian(g, cv2.CV_32F, ksize=3) * (s ** 2))
    return out


def normals(pts):
    """Unit normal at every point of a polyline."""
    t = np.gradient(pts, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length[length < 1e-6] = 1.0
    return n / length


def integrate(R, pts, w, h, min_pts=50):
    """Mean of response map R along a curve, or None if too little of it is in the frame."""
    inb = (pts[:, 0] > 1) & (pts[:, 0] < w - 2) & (pts[:, 1] > 1) & (pts[:, 1] < h - 2)
    if inb.sum() < min_pts:
        return None
    q = pts[inb]
    return float(R[q[:, 1].astype(int), q[:, 0].astype(int)].mean())


def iou(a, b):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0
