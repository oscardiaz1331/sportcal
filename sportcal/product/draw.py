"""Top-down minimap and camera-view field lines for any registered sport, drawn from its template (`Sport.polylines`)."""
import cv2
import numpy as np

from sportcal.core.geometry import project_points

# (background, line colour) of the minimap per `Sport.surface`, BGR; any other surface gets COURT
SURFACES = {"ice": ((245, 245, 245), (150, 150, 150)), "grass": ((70, 130, 70), (255, 255, 255))}
COURT = ((150, 110, 60), (255, 255, 255))


def minimap(sport, max_w, max_h=None, apron=0.0, upright=False):
    """(image, to_px): a blank top-down field that fits in `max_w` x `max_h` pixels (as wide as `max_w` when `max_h` is
    None), and the map from world metres (N, 2) to its pixels (N, 2). The minimap looks like the main camera:

    * lying down (the default), for a camera at the side: +x to the right, +y down the image when `sport.y_down` and up
      otherwise;
    * `upright`, for a camera that looks along the field (tennis): the far end, -x, at the top and +y on the left - the
      naming `core.camera.canonical_mirror` gives an end view.

    `apron`: metres drawn around the field, for a sport played off it (tennis, behind the baselines)."""
    x0, x1, y0, y1 = sport.box
    x0, x1, y0, y1 = x0 - apron, x1 + apron, y0 - apron, y1 + apron
    across, down = (y1 - y0, x1 - x0) if upright else (x1 - x0, y1 - y0)  # metres along the image's width and height
    m = max(4, round(0.04 * max_w))
    s = (max_w - 2 * m) / across
    if max_h is not None:
        s = min(s, (max_h - 2 * m) / down)
    width, height = round(across * s) + 2 * m, round(down * s) + 2 * m

    def to_px(world):
        q = np.asarray(world, float).reshape(-1, 2)
        if upright:
            return np.c_[m + (y1 - q[:, 1]) * s, m + (q[:, 0] - x0) * s]
        y = q[:, 1] - y0 if sport.y_down else y1 - q[:, 1]
        return np.c_[m + (q[:, 0] - x0) * s, m + y * s]

    bg, ink = SURFACES.get(sport.surface, COURT)
    img = np.full((height, width, 3), bg, np.uint8)
    for _, pl in sport.polylines():
        cv2.polylines(img, [np.round(to_px(pl)).astype(np.int32)], False, ink, 1, cv2.LINE_AA)
    return img, to_px


def field_lines(H, sport, w, h, step=0.5):
    """The template lines as seen through H (world -> image): pixel runs (M, 2), one per stretch of a line that lies in
    front of the camera. Each line is resampled every `step` metres first, so a line crossing the horizon is cut where
    it leaves the view instead of joining its two ends with a bogus straight segment."""
    lines = []
    for _, pl in sport.polylines():
        pl = np.asarray(pl, float)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
        t = np.r_[np.arange(0, d[-1], step), d[-1]]
        lines.append(np.c_[np.interp(t, d, pl[:, 0]), np.interp(t, d, pl[:, 1])])
    # one call for the whole field: project_points tells the side in front of the camera from the median of the points
    # it gets, and a line lying wholly behind the camera would look in front of itself if projected alone
    xy, ok = project_points(np.asarray(H, float), np.vstack(lines), w, h)
    cut = np.cumsum([len(pts) for pts in lines])[:-1]
    runs = []
    for line_xy, line_ok in zip(np.split(xy, cut), np.split(ok, cut)):
        for run in np.split(np.arange(len(line_ok)), np.flatnonzero(~line_ok)):
            run = run[line_ok[run]]
            if len(run) > 1:
                runs.append(line_xy[run])
    return runs


def draw_field(img, H, sport, colour=(0, 255, 255), thickness=1):
    """`field_lines` drawn on `img` (BGR, in place)."""
    h, w = img.shape[:2]
    for run in field_lines(H, sport, w, h):
        cv2.polylines(img, [np.round(run).astype(np.int32)], False, colour, thickness, cv2.LINE_AA)
    return img


def ball_trail(img, trail, here, colour=(255, 255, 255)):
    """The ball's recent path on the minimap `img` (in place): `trail` is [((x, y) pixels, BGR colour)], oldest first,
    drawn fading into the past. With `here` (the ball was seen in this frame) the ball itself is drawn at the last
    point, with a dark ring: a white puck does not show on white ice without one."""
    for k in range(1, len(trail)):
        (p1, _), (p2, c) = trail[k - 1], trail[k]
        cv2.line(img, p1, p2, tuple(int(v * k / len(trail)) for v in c), 2)
    if here:
        cv2.circle(img, trail[-1][0], 4, colour, -1)
        cv2.circle(img, trail[-1][0], 5, (30, 30, 30), 1)
    return img
