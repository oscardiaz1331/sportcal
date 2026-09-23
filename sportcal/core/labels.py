"""YOLO-pose label files for a planar template (one object, N keypoints).

Line format: `cls cx cy w h  x1 y1 v1  x2 y2 v2 ...`, everything normalised to [0, 1].
Generated labels use v=1; the sport's template defines what each keypoint index means.
"""
import numpy as np

from sportcal.core.geometry import project_points

NKPT = 56  # hockey template size; every current label file uses it


def read_label(path):
    """(cls, box_xywh, kpts (NKPT, 3)) normalised, or None if the file has no object."""
    txt = path.read_text().split()
    if len(txt) < 5 + NKPT * 3:
        return None
    return (int(float(txt[0])), np.asarray(txt[1:5], float),
            np.asarray(txt[5:5 + NKPT * 3], float).reshape(NKPT, 3))


def write_label(path, cls, box, kpts):
    parts = [str(cls)] + ["{:.6f}".format(v) for v in box]
    for x, y, v in kpts:
        parts += ["{:.6f}".format(x), "{:.6f}".format(y), "{:g}".format(v)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(parts) + "\n")


def label_from_H(H, template, w, h, margin=0.0):
    """Reproject the template keypoints through H. Returns (kpts, box) or None when
    fewer than 6 keypoints are visible."""
    xy, ok = project_points(H, template, w, h)
    norm = xy / np.array([w, h])
    keep = (ok & (norm[:, 0] >= -margin) & (norm[:, 0] <= 1 + margin)
            & (norm[:, 1] >= -margin) & (norm[:, 1] <= 1 + margin))
    if keep.sum() < 6:
        return None
    kpts = np.zeros((NKPT, 3))
    kpts[keep, :2] = np.clip(norm[keep], 0.0, 1.0)
    kpts[keep, 2] = 1
    k = kpts[keep, :2]
    x0, y0 = k[:, 0].min(), k[:, 1].min()
    x1, y1 = k[:, 0].max(), k[:, 1].max()
    return kpts, np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])


# --------------------------------------------------------------- heatmap targets (keypoint + line models)

def visible_ends(H, a, b, w, h, n=1000):
    """((x, y), (x, y)): the ends, in image px, of the part of world segment a -> b that H puts in front of the camera
    and inside the image, ordered from a to b; None when less than that is visible. An end is the painted end when it is
    in view, the image border otherwise. Sampled, not solved, so a segment that runs behind the camera is still right.
    ponytail: n samples fix an end to within |ab| / n (6 cm on a 61 m line); solve the border crossing if that matters."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    xy, ok = project_points(H, np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t, w, h)
    ok &= (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
    idx = np.flatnonzero(ok)
    return None if len(idx) < 2 else (xy[idx[0]], xy[idx[-1]])


def render_heatmaps(H, points, segments, w, h, sigma=2.0):
    """(len(points) + 2 * len(segments), h, w) float32 targets: a Gaussian centred exactly (sub-pixel) where H puts each
    world point, then one channel per visible end of each world segment (`visible_ends`); a channel stays empty when its
    point is not in view. The off-centre peak keeps the sub-pixel position in the target, `heatmap_peaks` reads it back."""
    xy, ok = project_points(H, np.asarray(points, float).reshape(-1, 2), w, h)
    centres = [p if o else None for p, o in zip(xy, ok)]
    for a, b in segments:
        ends = visible_ends(H, a, b, w, h)
        centres += list(ends) if ends is not None else [None, None]
    out = np.zeros((len(centres), h, w), np.float32)
    r = int(3 * sigma) + 1
    for c, p in enumerate(centres):
        if p is None:
            continue
        cx, cy = int(round(p[0])), int(round(p[1]))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        y0, y1, x0, x1 = max(0, cy - r), min(h, cy + r + 1), max(0, cx - r), min(w, cx + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        out[c, y0:y1, x0:x1] = np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2.0 * sigma ** 2))
    return out


def heatmap_peaks(heat, thr=0.3):
    """[(x, y) or None] per channel of (C, h, w) heatmaps: the maximum when it reaches `thr`, refined to sub-pixel by
    the weighted centroid of its 3x3 neighbourhood."""
    C, h, w = heat.shape
    out = []
    for c, i in enumerate(heat.reshape(C, -1).argmax(1)):
        y, x = divmod(int(i), w)
        if heat[c, y, x] < thr:
            out.append(None)
            continue
        y0, y1, x0, x1 = max(0, y - 1), min(h, y + 2), max(0, x - 1), min(w, x + 2)
        patch = heat[c, y0:y1, x0:x1].astype(np.float64)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        out.append((float((xx * patch).sum() / patch.sum()), float((yy * patch).sum() / patch.sum())))
    return out
