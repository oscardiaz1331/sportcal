"""A circle on the field plane seen as an ellipse: what its outline says about the homography.

The image of a world circle is a conic (5 numbers), and a homography has 8, so an ellipse fixes 5 of them and leaves a
3-parameter family: the homographies that map the world circle onto that same conic. Those differ by an element of the
conic's stabiliser (rotation about the circle centre plus two "boosts"). A couple of point clicks (the circle centre and one
point of its outline, say) choose the member of the family, and any further point clicks over-determine the fit.

    conic_from_points     algebraic conic through >= 5 image points (the user's ellipse outline)
    sampson               first-order distance (px) from points to a conic
    circle_family         the H that map the world circle onto a given image conic, parametrised by 3 numbers
    fit_points_ellipse    least-squares H from point correspondences AND the ellipse outline

World coordinates are metres; the circle is centred at (cx, cy) with radius `radius`. Image coordinates are pixels.
"""
import numpy as np
import cv2
from scipy.linalg import expm
from scipy.optimize import least_squares

# generators of so(2, 1): rotation about the circle centre and two boosts (they preserve diag(1, 1, -1))
_JZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float)
_BX = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], float)
_BY = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], float)
_MAX_BOOST = 3.0
REAL_VIEW = -1.0   # orientation() of a genuine camera view: X to the right, Y away (up in the image)


def circle_conic(cx, cy, radius):
    """3x3 symmetric matrix of the world circle: [X Y 1] C [X Y 1]^T = 0 on it."""
    return np.array([[1, 0, -cx], [0, 1, -cy], [-cx, -cy, cx * cx + cy * cy - radius ** 2]], float)


def conic_from_points(pts):
    """Symmetric 3x3 conic through (n >= 5, 2) points: the algebraic least-squares fit (SVD null vector after Hartley
    normalisation). Returns None if the points do not determine a conic."""
    P = np.asarray(pts, float).reshape(-1, 2)
    if len(P) < 5:
        return None
    m, s = P.mean(0), max(float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())), 1e-9) / np.sqrt(2.0)
    x, y = ((P - m) / s).T
    D = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], 1)
    a, b, c, d, e, f = np.linalg.svd(D, full_matrices=False)[2][-1]
    Cn = np.array([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]])
    T = np.array([[1 / s, 0, -m[0] / s], [0, 1 / s, -m[1] / s], [0, 0, 1.0]])
    C = T.T @ Cn @ T
    return C / np.linalg.norm(C)


def sampson(C, pts):
    """Signed first-order distance (pixels) from (n, 2) points to conic C."""
    X = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(pts))]
    CX = X @ C
    g = np.maximum(2.0 * np.hypot(CX[:, 0], CX[:, 1]), 1e-12)
    return np.einsum("ni,ni->n", CX, X) / g


def _normalise(C):
    """(Cn, S) with C = S^-T Cn S^-1, S mapping a unit-scale frame centred on the ellipse to pixels, or None if C is not a
    real ellipse. Working in that frame keeps the eigen-decomposition well conditioned (pixel coordinates are ~1e3)."""
    C = np.asarray(C, float)
    A = C[:2, :2]
    if np.linalg.det(A) <= 0:
        return None
    if np.trace(A) < 0:
        C, A = -C, -A
    b = C[:2, 2]
    m = -np.linalg.solve(A, b)
    k = float(m @ A @ m - C[2, 2])          # A (x - m)^2 = k on the ellipse
    if k <= 0:
        return None
    s = np.sqrt(k / np.sqrt(np.linalg.det(A)))
    S = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1.0]])
    Cn = S.T @ C @ S
    return Cn / np.linalg.norm(Cn), S


def is_ellipse(C):
    """True if the conic is a real, non-degenerate ellipse."""
    return _normalise(C) is not None


def circle_family(C, radius=1.0, cx=0.0, cy=0.0):
    """Function g(params) -> (3, 3) H for the homographies mapping the world circle onto image conic C, or None.

    `params` = (angle, boost_x, boost_y): the 3 free numbers of the family (identity component; the mirror-image
    members are not returned). The circle centre is (cx, cy) with the given radius."""
    nrm = None if C is None else _normalise(C)
    if nrm is None:
        return None
    Cn, S = nrm
    lam, V = np.linalg.eigh(Cn)
    if not (np.sum(lam > 0) == 2 and np.sum(lam < 0) == 1):
        return None
    order = [i for i in np.argsort(-lam) if lam[i] > 0] + [i for i in range(3) if lam[i] < 0]
    lam, V = lam[order], V[:, order]
    W = np.diag(1.0 / np.sqrt(np.abs(lam)))       # Cn = V |L|^1/2 J |L|^1/2 V^T with J = diag(1, 1, -1)
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, radius]], float)
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])   # circle centred at origin -> at (cx, cy)
    base, Tinv = S @ V @ W, np.linalg.inv(T)

    def build(params, mirror):
        a, bx, by = params
        G = expm(a * _JZ + bx * _BX + by * _BY) @ mirror
        H = base @ G @ P @ Tinv
        return H / H[2, 2] if abs(H[2, 2]) > 1e-9 else H / np.linalg.norm(H)

    # the eigenvector handedness is arbitrary: pick the family (with or without a mirror of the world circle, which
    # leaves it unchanged) whose members are genuine, non-mirrored views
    mirror = np.eye(3) if orientation(build((0.0, 0.0, 0.0), np.eye(3)), (cx, cy)) == REAL_VIEW else np.diag([1.0, -1.0, 1.0])
    return lambda params: build(params, mirror)


def orientation(H, at=(0.0, 0.0), eps=1e-3):
    """Sign of the local Jacobian determinant of the world -> image map at a world point (-1 for a real camera above the
    plane with +Y going up in the image, +1 for a mirror image)."""
    def proj(p):
        q = H @ np.array([p[0], p[1], 1.0])
        return q[:2] / q[2]
    x0, y0 = at
    J = np.stack([(proj((x0 + eps, y0)) - proj((x0 - eps, y0))) / (2 * eps), (proj((x0, y0 + eps)) - proj((x0, y0 - eps))) / (2 * eps)], 1)
    return float(np.sign(np.linalg.det(J)))


def _project(H, world):
    q = np.c_[np.asarray(world, float).reshape(-1, 2), np.ones(len(world))] @ H.T
    return q[:, :2] / q[:, 2:3], q[:, 2]


def fit_points_ellipse(world, img, ellipse_pts, radius, center=(0.0, 0.0), weight_ellipse=1.0, n_angles=12):
    """H (world -> image, 3x3 with H[2,2] = 1) from point correspondences plus the outline of one world circle.

    world (n, 2) metres and img (n, 2) pixels are the clicked correspondences (n >= 2 when the ellipse is given;
    with n >= 4 the ellipse only refines the fit). ellipse_pts (m >= 5, 2): clicked points on the circle's outline.
    Returns {"H", "rms_points", "rms_ellipse", "n_solutions"} or None. `n_solutions` counts how many clearly different
    homographies fit the clicks about as well as the best (1 = well determined).

    The mirror family is excluded: a real camera never sees the mirror image, and it is the orientation that
    tells a genuine view from its reflection when all the clicked points lie on the halfway line."""
    world, img = np.asarray(world, float).reshape(-1, 2), np.asarray(img, float).reshape(-1, 2)
    E = np.asarray(ellipse_pts, float).reshape(-1, 2)
    C = conic_from_points(E)
    fam = circle_family(C, radius, *center)
    if fam is None or len(world) < 2:
        return None
    Cw = circle_conic(center[0], center[1], radius)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    circ = np.c_[center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)]
    ref = REAL_VIEW

    def resid_family(p):
        H = fam(p)
        xy, den = _project(H, np.vstack([world, circ]))
        if not np.all(np.isfinite(xy)) or (np.sign(den) != np.sign(den[0])).any() or orientation(H, center) != ref:
            return np.full(2 * len(world), 1e3)
        return (xy[:len(world)] - img).ravel()

    starts = [(a, 0.0, 0.0) for a in np.linspace(0, 2 * np.pi, n_angles, endpoint=False)]
    sols = []
    for p0 in starts:
        try:
            r = least_squares(resid_family, np.array(p0), bounds=([-4 * np.pi, -_MAX_BOOST, -_MAX_BOOST], [4 * np.pi, _MAX_BOOST, _MAX_BOOST]),
                              x_scale=[1.0, 0.5, 0.5], max_nfev=80)
        except Exception:
            continue
        sols.append((float(np.sqrt(np.mean(r.fun ** 2))), r.x))
    sols = [s for s in sols if s[0] < 1e2]
    if len(world) >= 4:            # enough points for a plain DLT start as well
        H0, _ = cv2.findHomography(world.astype(np.float32), img.astype(np.float32), 0)
        if H0 is not None and np.all(np.isfinite(H0)):
            sols.append((None, H0 / H0[2, 2]))
    if not sols:
        return None
    # polish every candidate over the full 8-parameter H with points and ellipse together
    Hc = []
    for c, x in sols:
        Hc.append(x if c is None else fam(x))
    results = []
    for H0 in Hc:
        H0 = H0 / H0[2, 2] if abs(H0[2, 2]) > 1e-12 else H0 / np.linalg.norm(H0)

        def resid(d, H0=H0):
            H = H0 @ (np.eye(3) + np.append(d, 0.0).reshape(3, 3))
            xy, den = _project(H, world)
            with np.errstate(all="ignore"):
                Ci = np.linalg.inv(H).T @ Cw @ np.linalg.inv(H)
            r_e = weight_ellipse * sampson(Ci, E)
            r = np.r_[(xy - img).ravel(), r_e]
            return np.where(np.isfinite(r), r, 1e3)

        try:
            r = least_squares(resid, np.zeros(8), x_scale=1e-2, max_nfev=200)
        except Exception:
            continue
        H = H0 @ (np.eye(3) + np.append(r.x, 0.0).reshape(3, 3))
        if abs(H[2, 2]) < 1e-12 or orientation(H, center) != ref:
            continue
        H = H / H[2, 2]
        xy, den = _project(H, np.vstack([world, circ]))
        if (np.sign(den) != np.sign(den[0])).any():
            continue
        rp = float(np.sqrt(np.mean(np.sum((xy[:len(world)] - img) ** 2, 1))))
        Ci = np.linalg.inv(H).T @ Cw @ np.linalg.inv(H)
        re = float(np.sqrt(np.mean(sampson(Ci, E) ** 2)))
        results.append((rp ** 2 + weight_ellipse ** 2 * re ** 2, H, rp, re))
    if not results:
        return None
    results.sort(key=lambda t: t[0])
    best = results[0]
    distinct = 1
    Pt = np.array([[x, y] for x in np.linspace(-30, 30, 5) for y in np.linspace(-20, 20, 3)], float)
    ref_xy = _project(best[1], Pt)[0]
    for cost, H, _, _ in results[1:]:
        if cost < max(2.0 * best[0], 4.0) and np.median(np.linalg.norm(_project(H, Pt)[0] - ref_xy, axis=1)) > 5.0:
            distinct += 1
    return {"H": best[1], "rms_points": best[2], "rms_ellipse": best[3], "n_solutions": distinct}
