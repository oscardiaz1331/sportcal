import numpy as np

from sportcal.core import labels as L
from sportcal.sports import get

H = np.array([[30.0, 0, 200], [0, 30.0, 150], [0, 0, 1.0]])  # simple affine "camera", 1920x1080


def test_label_from_H_marks_visible_keypoints_and_boxes_them():
    template = get("hockey-nhl").keypoints()
    kpts, box = L.label_from_H(H, template, 1920, 1080)
    visible = kpts[:, 2] == 1
    assert visible.sum() >= 6
    assert (kpts[visible, :2] >= 0).all() and (kpts[visible, :2] <= 1).all()
    assert (kpts[~visible] == 0).all()
    assert 0 < box[2] <= 1 and 0 < box[3] <= 1


def test_label_from_H_returns_none_when_nothing_is_in_frame():
    far = H.copy()
    far[0, 2] = 1e6
    assert L.label_from_H(far, get("hockey-nhl").keypoints(), 1920, 1080) is None


def test_label_roundtrip(tmp_path):
    kpts, box = L.label_from_H(H, get("hockey-nhl").keypoints(), 1920, 1080)
    f = tmp_path / "sub" / "a.txt"
    L.write_label(f, 0, box, kpts)
    cls, box2, kpts2 = L.read_label(f)
    assert cls == 0 and np.allclose(box, box2, atol=1e-6) and np.allclose(kpts, kpts2, atol=1e-6)


def test_read_label_of_an_empty_file_is_none(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    assert L.read_label(f) is None
