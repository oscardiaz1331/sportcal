"""Pinhole-camera model for a planar field, batched over hypotheses.

A pose is (cx, cy, cz, pan, tilt, f): camera position in metres (Z up), pan/tilt in radians and
focal length in pixels; no roll, principal point at the image centre. The camera looks towards +Y
from the Y<0 side, so "up" in the image is +Y. H maps world (X, Y, 1) to image pixels.

These functions work on arrays of hypotheses so a search can score thousands of poses at once.
"""
import numpy as np


def pose_to_H(p, w, h):
    """Poses (n, 6) -> homographies (n, 3, 3), with positive homogeneous depth in front of the camera."""
    p = np.atleast_2d(p).astype(float)
    cx, cy, cz, pan, tilt, f = p.T
    sp, cp, st, ct = np.sin(pan), np.cos(pan), np.sin(tilt), np.cos(tilt)
    z = np.zeros_like(pan)
    right = np.stack([cp, -sp, z], 1)
    forward = np.stack([sp * ct, cp * ct, -st], 1)
    down = np.stack([-st * sp, -st * cp, -ct], 1)
    R = np.stack([right, down, forward], 1)  # (n, 3, 3): rows right, down, forward
    C = np.stack([cx, cy, cz], 1)
    t = -np.einsum("nij,nj->ni", R, C)
    M = np.stack([R[:, :, 0], R[:, :, 1], t], 2)  # plane Z=0: drop the third column of R
    K = np.zeros((len(p), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2.0, h / 2.0, 1.0
    return K @ M


def project(H, pts):
    """H (n, 3, 3), world points (m, 2) -> (xy (n, m, 2), depth (n, m))."""
    P = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    q = np.einsum("nij,mj->nmi", H, P)
    depth = q[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[..., :2] / depth[..., None]
    return xy, depth


def bilinear(img, xy):
    """Bilinear sample of a 2-D map at float pixel positions (clamped to the image)."""
    x = np.clip(xy[..., 0], 0, img.shape[1] - 1.001)
    y = np.clip(xy[..., 1], 0, img.shape[0] - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - x0, y - y0
    a, b, c, d = img[y0, x0], img[y0, x0 + 1], img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def pinhole_residual(Hs, w, h):
    """(f2, mismatch) per homography: how far it is from ANY square-pixel pinhole camera.

    For H = K [r1 r2 t] with K having the principal point at the centre, the columns' orthogonality
    gives f^2 = -a1.a2 / (z1 z2) (a_i: the x,y part of column i relative to the centre, z_i its third
    component). `mismatch` in [0, 1] measures how much the norms of K^-1 h1 and K^-1 h2 differ (they
    must be equal); it is 1 when f2 is not a valid focal length (outside 0.4-8 image widths).
    A free 8-parameter H can score as well as the truth while deforming into something no camera
    produces, so searches penalise this mismatch.
    """
    a1 = Hs[:, :2, 0] - np.array([w / 2.0, h / 2.0]) * Hs[:, 2:3, 0]
    a2 = Hs[:, :2, 1] - np.array([w / 2.0, h / 2.0]) * Hs[:, 2:3, 1]
    z1, z2 = Hs[:, 2, 0], Hs[:, 2, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = -(a1 * a2).sum(1) / (z1 * z2)
        n1 = (a1 ** 2).sum(1) / f2 + z1 ** 2
        n2 = (a2 ** 2).sum(1) / f2 + z2 ** 2
        mismatch = np.abs(n1 - n2) / (n1 + n2)
    valid = np.isfinite(f2) & np.isfinite(mismatch) & (f2 > (0.4 * w) ** 2) & (f2 < (8 * w) ** 2)
    return f2, np.where(valid, np.clip(mismatch, 0, 1), 1.0)


def is_pinhole_consistent(Hs, w, h, tol=0.4):
    """Bool (n,): the homography can come from a reasonable pinhole camera."""
    return pinhole_residual(Hs, w, h)[1] < tol
