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
