"""Top-down minimap and camera-view field lines for any registered sport, drawn from its template (`Sport.polylines`)."""
import cv2
import numpy as np

from sportcal.core.geometry import project_points

# (background, line colour) of the minimap per `Sport.surface`, BGR; any other surface gets COURT
SURFACES = {"ice": ((245, 245, 245), (150, 150, 150)), "grass": ((70, 130, 70), (255, 255, 255))}
COURT = ((150, 110, 60), (255, 255, 255))


def minimap(sport, width_px):
    """(image, to_px): a blank top-down field `width_px` wide, and the map from world metres (N, 2) to its pixels (N, 2).
    +y points down the image when `sport.y_down` and up otherwise, so that the minimap looks like the main camera."""
    x0, x1, y0, y1 = sport.box
    m = max(4, round(0.04 * width_px))
    s = (width_px - 2 * m) / (x1 - x0)
    height = round((y1 - y0) * s) + 2 * m

    def to_px(world):
        q = np.asarray(world, float).reshape(-1, 2)
        y = q[:, 1] - y0 if sport.y_down else y1 - q[:, 1]
        return np.c_[m + (q[:, 0] - x0) * s, m + y * s]

    bg, ink = SURFACES.get(sport.surface, COURT)
    img = np.full((height, width_px, 3), bg, np.uint8)
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
