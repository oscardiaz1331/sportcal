"""The keypoint + line model ("model A"), sport-agnostic: one heatmap per template keypoint and one per visible end of
every straight painted line; the template and a points + lines DLT turn them into H. The sport (`sports.Sport`) supplies
the keypoints, the straight lines and the extent; training is `lab/common/train_kpline.py`."""
import cv2
import numpy as np
import torch

from sportcal.core.geometry import line_through, solve_points_lines
from sportcal.core.labels import heatmap_peaks
from sportcal.models.unet import MEAN, STD, HalfResUNet

SIZE = (960, 544)                            # network input: a 16:9 frame at 960 px, padded at the bottom to /32
OUT_SIZE = (SIZE[0] // 2, SIZE[1] // 2)      # heatmaps at half the input (HalfResUNet)
KEYPOINT_SETS = ("base", "derived")


def n_channels(sport, kp):
    """Output channels for keypoints `kp` of `sport`: one per point, two per straight line."""
    return len(kp) + 2 * len(sport.straight_lines())


def keypoint_set_of(sport, channels):
    """The name of the keypoint set a network with `channels` outputs was trained on."""
    for name in KEYPOINT_SETS:
        try:
            if n_channels(sport, sport.keypoint_set(name)) == channels:
                return name
        except KeyError:
            continue
    raise ValueError("{} channels match no keypoint set of {}".format(channels, sport.name))


def load(weights, sport, device="cpu"):
    """(eval-mode network, its keypoints) from a checkpoint; the keypoint set is read from the output channels."""
    state = torch.load(weights, map_location=device)
    kp = sport.keypoint_set(keypoint_set_of(sport, state["final.1.weight"].shape[0]))
    model = HalfResUNet(ncls=n_channels(sport, kp), pretrained=False).to(device).eval()
    model.load_state_dict(state)
    return model, kp


def to_input(img):
    """(3 x SIZE tensor, scale): the BGR frame resized to SIZE's width, padded at the bottom, normalised."""
    s = SIZE[0] / img.shape[1]
    small = cv2.resize(img, (SIZE[0], int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    canvas[:small.shape[0]] = small[:SIZE[1]]
    x = (canvas[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), s


def estimate_H(heat, sport, w0, kp, thr=0.3):
    """H in original-frame px from one (channels, h, w) sigmoid stack, or None."""
    s = OUT_SIZE[0] / w0
    peaks = heatmap_peaks(heat, thr)
    nkp = len(kp)
    wp = [kp[k] for k in range(nkp) if peaks[k] is not None]
    ip = [np.asarray(peaks[k]) / s for k in range(nkp) if peaks[k] is not None]
    wl, il = [], []
    for j, (a, b) in enumerate(sport.straight_lines()):
        e0, e1 = peaks[nkp + 2 * j], peaks[nkp + 2 * j + 1]
        if e0 is not None and e1 is not None and np.hypot(e0[0] - e1[0], e0[1] - e1[1]) > 5:
            wl.append(line_through(a, b))
            il.append(line_through(np.asarray(e0) / s, np.asarray(e1) / s))
    x0, x1 = sport.box[:2]
    return solve_points_lines(wp, ip, wl, il, x1 - x0, w0, 8.0 * w0 / 1920)
