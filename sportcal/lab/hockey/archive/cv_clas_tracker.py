#!/usr/bin/env python3
"""
Hockey rink geometry / homography tracker without deep learning.

Main idea:
  1) Segment the ice using the user's robust Y > 185 rule in YUV.
  2) Clean/fill the ice mask so shadows/players create holes rather than
     breaking the rink into pieces.
  3) Detect long rink markings only INSIDE the ice mask.
  4) Keep the previous geometry when a frame is blurred/occluded.
  5) Track a set of rink keypoints with Lucas-Kanade optical flow.
  6) Re-estimate the image->rink homography from tracked points.

A practical bootstrap is included: press 'm' and click four points on a
known rink rectangle. These are then tracked automatically. This avoids
requiring a training dataset and is much more robust than asking Hough
circles to fire on every blurred frame.

The script also draws detected red/blue candidates and the projected rink
axes. World coordinates are arbitrary by default; change RINK_WIDTH and
RINK_LENGTH to real dimensions if desired.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

Y_THRESHOLD = 185
RINK_WIDTH = 60.0     # x coordinate range in the output top view
RINK_LENGTH = 200.0   # y coordinate range in the output top view

# OpenCV LK settings
LK_WIN = (31, 31)
LK_LEVELS = 4
LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01,
)


@dataclass
class LineCandidate:
    p1: np.ndarray
    p2: np.ndarray
    length: float
    angle_deg: float
    kind: str  # "red", "blue", "unknown"

    @property
    def midpoint(self) -> np.ndarray:
        return (self.p1 + self.p2) * 0.5


@dataclass
class TrackerState:
    H_img_to_world: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    track_points_img: Optional[np.ndarray] = None
    last_good_frame: int = -1
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Ice segmentation
# ---------------------------------------------------------------------------

def ice_mask_yuv(frame_bgr: np.ndarray, threshold: int = Y_THRESHOLD) -> np.ndarray:
    """Return a robust binary mask of the bright ice in YUV space."""
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]

    # User's empirically good rule.
    mask = (y >= threshold).astype(np.uint8) * 255

    h, w = mask.shape
    # Large closing fills shadows and gaps caused by players.
    k_close = max(9, int(round(min(h, w) * 0.035)) | 1)
    k_open = max(3, int(round(min(h, w) * 0.007)) | 1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (k_close, k_close))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (k_open, k_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    return largest_centered_component(mask)


def largest_centered_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest plausible ice component; prefer one containing image center."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask

    h, w = mask.shape
    cx, cy = w * 0.5, h * 0.55
    chosen = -1
    best_score = -1e18

    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 0.03 * w * h:
            continue
        x, y, ww, hh = stats[i, :4]
        contains_center = x <= cx <= x + ww and y <= cy <= y + hh
        center_dist = np.hypot(centroids[i, 0] - cx, centroids[i, 1] - cy)
        score = area * (2.5 if contains_center else 1.0) - center_dist * 100.0
        if score > best_score:
            best_score = score
            chosen = i

    if chosen < 0:
        return mask
    return (labels == chosen).astype(np.uint8) * 255


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed dark holes (players/shadows) inside the ice region."""
    flood = mask.copy()
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return mask | flood_inv


# ---------------------------------------------------------------------------
# Marking detection
# ---------------------------------------------------------------------------

def color_masks_yuv(frame_bgr: np.ndarray, ice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Loose red/blue masks. They are used as evidence, not as the rink detector."""
    # HSV is more convenient for hue, while the ice mask remains YUV-based.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    valid = ice > 0
    sat = s >= 45
    val = v >= 55

    red = (((h <= 12) | (h >= 168)) & sat & val & valid)
    blue = ((h >= 88) & (h <= 135) & sat & val & valid)

    red_m = red.astype(np.uint8) * 255
    blue_m = blue.astype(np.uint8) * 255

    # Join gaps from compression / motion blur.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_m = cv2.morphologyEx(red_m, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_m = cv2.morphologyEx(blue_m, cv2.MORPH_CLOSE, kernel, iterations=2)
    return red_m, blue_m


def hough_candidates(mask: np.ndarray, min_length: float) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    edges = cv2.Canny(mask, 60, 140)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=max(25, int(min_length * 0.12)),
        minLineLength=int(min_length),
        maxLineGap=max(12, int(min_length * 0.18)),
    )
    out = []
    if lines is None:
        return out

    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (float(v) for v in line)
        p1 = np.array([x1, y1], np.float32)
        p2 = np.array([x2, y2], np.float32)
        length = float(np.linalg.norm(p2 - p1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
        out.append((p1, p2, length, angle))
    return out


def detect_marking_lines(frame_bgr: np.ndarray, ice: np.ndarray) -> list[LineCandidate]:
    red, blue = color_masks_yuv(frame_bgr, ice)
    h, w = ice.shape
    min_len = max(80.0, 0.16 * w)

    candidates: list[LineCandidate] = []
    for kind, mask in (("red", red), ("blue", blue)):
        for p1, p2, length, angle in hough_candidates(mask, min_len):
            candidates.append(LineCandidate(p1, p2, length, angle, kind))
    return candidates


def draw_lines(frame: np.ndarray, lines: list[LineCandidate]) -> None:
    for line in lines:
        color = (0, 0, 255) if line.kind == "red" else (255, 80, 0)
        p1 = tuple(np.round(line.p1).astype(int))
        p2 = tuple(np.round(line.p2).astype(int))
        cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Manual bootstrap + optical-flow tracking
# ---------------------------------------------------------------------------

def mouse_select_four(frame: np.ndarray) -> np.ndarray:
    """Select four image points. They become the corners of a world rectangle."""
    clone = frame.copy()
    points: list[tuple[int, int]] = []
    win = "Select 4 rink points: TL, TR, BR, BL"

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    while True:
        view = clone.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(view, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(view, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(points) >= 2:
            cv2.polylines(view, [np.array(points, np.int32)], False,
                          (0, 255, 0), 2)

        cv2.putText(view, "Click TL TR BR BL | ENTER=accept | ESC=cancel",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            raise RuntimeError("Manual calibration cancelled")
        if key in (13, 10) and len(points) == 4:
            break

    cv2.destroyWindow(win)
    return np.array(points, dtype=np.float32)


def world_corners() -> np.ndarray:
    return np.array([
        [0.0, 0.0],
        [RINK_WIDTH, 0.0],
        [RINK_WIDTH, RINK_LENGTH],
        [0.0, RINK_LENGTH],
    ], dtype=np.float32)


def initialize_tracking(frame: np.ndarray, state: TrackerState) -> None:
    points = mouse_select_four(frame)
    H, _ = cv2.findHomography(points, world_corners(), 0)
    if H is None:
        raise RuntimeError("Could not calculate initial homography")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    state.H_img_to_world = H
    state.prev_gray = gray
    state.track_points_img = points.reshape(-1, 1, 2)
    state.confidence = 1.0


def track_keypoints(frame: np.ndarray, state: TrackerState) -> bool:
    """Track the four calibrated rink points.

    We deliberately require all four correspondences: arbitrary image
    features do not have known world coordinates and must not be fed to
    findHomography as if they did. When blur/occlusion breaks the quartet,
    the previous H is kept unchanged.
    """
    if state.prev_gray is None or state.track_points_img is None:
        return False

    points = state.track_points_img.reshape(-1, 1, 2).astype(np.float32)
    if len(points) != 4:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        state.prev_gray,
        gray,
        points,
        None,
        winSize=LK_WIN,
        maxLevel=LK_LEVELS,
        criteria=LK_CRITERIA,
    )

    ok_all = (
        next_pts is not None
        and status is not None
        and err is not None
        and int(status.reshape(-1).sum()) == 4
        and float(np.max(err)) < 45.0
    )

    if ok_all:
        old = points.reshape(-1, 2)
        new = next_pts.reshape(-1, 2)
        displacement = np.linalg.norm(new - old, axis=1)
        ok_all = bool(np.max(displacement) < 120.0)

    if ok_all:
        H_new, _ = cv2.findHomography(new, world_corners(), 0)
        if H_new is not None:
            state.H_img_to_world = H_new
            state.track_points_img = next_pts.astype(np.float32)
            mean_err = float(np.mean(err))
            state.confidence = float(np.exp(-mean_err / 45.0))
            state.last_good_frame += 1
            state.prev_gray = gray
            return True

    # Important: keep the old H. Do not let a blurred/occluded frame destroy
    # the geometry. Rebase LK on the current image using the last known points.
    state.confidence *= 0.97
    state.prev_gray = gray
    return False


def refresh_tracking_points(frame: np.ndarray, ice: np.ndarray, state: TrackerState) -> None:
    """Reserved hook for future automatic point reacquisition.

    Do not add arbitrary goodFeaturesToTrack points here: their world
    coordinates are unknown, so using them would corrupt the homography.
    """
    return


def project_player_foot(H: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    pt = np.array([[[0.5 * (x1 + x2), y2]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(world[0]), float(world[1])


# ---------------------------------------------------------------------------
# Automatic line-based recovery hint
# ---------------------------------------------------------------------------

def draw_rink_axes(frame: np.ndarray, H: np.ndarray) -> None:
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return

    # Rectangle + center line in world coordinates.
    world = np.array([
        [0, 0], [RINK_WIDTH, 0], [RINK_WIDTH, RINK_LENGTH],
        [0, RINK_LENGTH], [0, 0],
        [RINK_WIDTH * 0.5, 0], [RINK_WIDTH * 0.5, RINK_LENGTH]
    ], dtype=np.float32).reshape(-1, 1, 2)
    img = cv2.perspectiveTransform(world, H_inv).reshape(-1, 2)

    rect = img[:5].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [rect], False, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame,
             tuple(img[5].astype(int)), tuple(img[6].astype(int)),
             (0, 255, 0), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="clip2.mp4", help="Video path or camera index")
    p.add_argument("--output", default="hockey_geometry.mp4")
    p.add_argument("--display-mask", action="store_true")
    p.add_argument("--start-calibration", action="store_true")
    p.add_argument("--threshold", type=int, default=Y_THRESHOLD)
    return p.parse_args()


def open_capture(src: str) -> cv2.VideoCapture:
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {src}")
    return cap


def main() -> None:
    args = parse_args()
    cap = open_capture(args.input)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    state = TrackerState()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        vis = frame.copy()

        ice = ice_mask_yuv(frame, args.threshold)
        ice_filled = fill_holes(ice)
        lines = detect_marking_lines(frame, ice_filled)
        draw_lines(vis, lines)

        # Manual bootstrap at startup or by pressing 'm'.
        if state.H_img_to_world is None and args.start_calibration:
            initialize_tracking(frame, state)
        elif state.H_img_to_world is not None:
            tracked = track_keypoints(frame, state)
            if frame_idx % 10 == 0:
                refresh_tracking_points(frame, ice_filled, state)

            if tracked:
                state.last_good_frame = frame_idx

            draw_rink_axes(vis, state.H_img_to_world)

            # Draw active optical-flow points.
            if state.track_points_img is not None:
                for pt in state.track_points_img.reshape(-1, 2):
                    cv2.circle(vis, tuple(pt.astype(int)), 2, (255, 255, 0), -1)

        # Diagnostics.
        cv2.putText(vis, f"ice={np.mean(ice_filled > 0):.2f}  lines={len(lines)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, f"H={'YES' if state.H_img_to_world is not None else 'NO'}  conf={state.confidence:.2f}",
                    (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, "m=manual calibration | r=reset | q=quit",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        writer.write(vis)
        cv2.imshow("Hockey rink geometry", vis)
        if args.display_mask:
            cv2.imshow("Ice mask", ice_filled)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('m'):
            initialize_tracking(frame, state)
        elif key == ord('r'):
            state = TrackerState()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
