"""Team of a player from the colour of their jersey: an HSV histogram of the torso, two clusters per clip.

Sport-agnostic: it assumes only that the two teams wear shirts of different colours. Referees in black and white and
bare ice have too little saturation to give a histogram, so they get no team rather than a wrong one.
"""
import cv2
import numpy as np

SAT_MIN = 40                  # below: ice, white boards, reflections, skates - not jersey colour
MIN_COLOUR_FRACTION = 0.15    # a torso band with less colour than this fell on the background, not on the shirt


def jersey_histogram(frame, bbox):
    """(H, S) histogram of the torso band (25%-65% of the box height, below the helmet and above the skates) with the
    low-saturation pixels masked out, or None when too little of the band has colour: a crouched or occluded player
    puts the band on the ice or the boards, and a histogram of those is not a jersey."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    h = y2 - y1
    crop = frame[y1 + int(h * 0.25):y1 + int(h * 0.65), x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > SAT_MIN).astype(np.uint8)
    if mask.mean() < MIN_COLOUR_FRACTION:
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def fit_teams(hists):
    """(2, 128) centroids: 2-means over one clip's jersey histograms. Which team is 0 is arbitrary."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, _, centres = cv2.kmeans(np.asarray(hists, np.float32), 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centres


def team_of(hists, centres):
    """Nearest centroid of each histogram (n, 128): (teams, ratio), ratio = its distance over the other's - near 0 a
    clear call, near 1 an ambiguous one."""
    d = np.linalg.norm(np.asarray(hists, np.float32)[:, None, :] - centres[None], axis=2)
    return d.argmin(1), d.min(1) / np.maximum(d.max(1), 1e-9)
