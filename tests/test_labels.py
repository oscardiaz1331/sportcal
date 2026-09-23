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


def test_heatmap_targets_decode_back_to_their_homography():
    """render_heatmaps -> heatmap_peaks -> solve_points_lines gives back H: the chain every prediction of the keypoint +
    line model goes through, fed here with perfect heatmaps. The rink overflows the frame, so line ends get cut by the
    image border and the clipping path is exercised too."""
    import cv2
    from sportcal.core.geometry import geom_error, line_through, solve_points_lines
    from sportcal.core.labels import heatmap_peaks, render_heatmaps
    from sportcal.sports.hockey import rink
    p = rink.RINK_NHL
    L, W = p["length"], p["width"]
    tpl, segs = rink.build_template(p), [(a, b) for _, a, b in rink.straight_lines(p)]
    w, h = 960, 544
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [L, 0], [L, W], [0, W]]),
                                    np.float32([[-300, 80], [1250, 80], [1500, 620], [-500, 620]])).astype(float)
    peaks = heatmap_peaks(render_heatmaps(H, tpl, segs, w, h))
    K = len(tpl)
    wp = [tpl[k] for k in range(K) if peaks[k] is not None]
    ip = [peaks[k] for k in range(K) if peaks[k] is not None]
    ends = [(peaks[K + 2 * j], peaks[K + 2 * j + 1]) for j in range(len(segs))]
    on_border = [e for pair in ends for e in pair if e is not None and min(e[0], w - 1 - e[0], e[1], h - 1 - e[1]) <= 2]
    assert len(wp) >= 10 and len(on_border) >= 4
    wl = [line_through(a, b) for (a, b), (e0, e1) in zip(segs, ends) if e0 is not None and e1 is not None]
    il = [line_through(e0, e1) for e0, e1 in ends if e0 is not None and e1 is not None]
    Hh = solve_points_lines(wp, ip, wl, il, L, w, 8 * w / 1920)
    grid = np.stack(np.meshgrid(np.arange(0, L, 1.0), np.arange(0, W, 1.0)), -1).reshape(-1, 2)
    assert geom_error(Hh, H, grid, w, h) < 2.0
    # two points cannot fix H: the lines have to carry it
    assert geom_error(solve_points_lines(wp[:2], ip[:2], wl, il, L, w, 8 * w / 1920), H, grid, w, h) < 3.0
