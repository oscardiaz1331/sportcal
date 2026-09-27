"""One video in, the product out (ADR 0006): the annotated video with the field lines and a minimap, the players'
positions in metres, and a status file the web app polls. Sport-agnostic: what differs per sport is a row of SPORTS.

The worker is its own process (`python -m sportcal.product.video`), started by the web app or by hand. Torch and
ultralytics load inside `process_video`, so importing this module (the server does, for SPORTS) stays light.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import psutil
import supervision as sv
from scenedetect.common import FrameTimecode
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import compute_downscale_factor
from trackers import ByteTrackTracker

from sportcal import sports
from sportcal.core.teams import fit_teams, jersey_histogram, team_of
from sportcal.paths import ROOT, RUNS
from sportcal.product import draw
from sportcal.product.kpline import KplineEstimator
from sportcal.product.pipeline import HomographyPipeline, project_to_field

JOBS_DIR = RUNS / "product"
ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[A-Za-z0-9_-]{1,40}$")

# What differs per sport (ADR 0006, "Scope of v1"). roles: detector class name -> role in the product.
SPORTS = {
    "hockey-nhl": dict(
        label="Hockey (NHL)",
        kpline=RUNS / "kpline" / "finetune" / "best_h.pt",
        detector=RUNS / "hockeyai" / "yolo26s" / "weights" / "best.pt",
        detector_hub=("SimulaMet-HOST/HockeyAI", "HockeyAI_model_weight.pt"),
        roles={"player": "player", "goalie": "goalie", "referee": "referee", "puck": "ball"},
        imgsz=800, hold_frames=15, smooth=0.1),
    "soccer-fifa": dict(
        label="Fútbol",
        kpline=RUNS / "kpline-soccer" / "pretrain-derived" / "best_h.pt",
        detector=ROOT / "yolo26m.pt",  # COCO; ultralytics downloads it by name when missing
        detector_hub=None,
        roles={"person": "player", "sports ball": "ball"},
        imgsz=1280, hold_frames=15, smooth=0.1),
}


class JobError(Exception):
    """A failure the owner can act on: its message (Spanish UI text) is shown as is."""


def _now():
    return datetime.now().isoformat(timespec="seconds")


def new_run_folder(jobs_dir, name):
    """A new `<YYYYMMDD-HHMMSS>-<label>` folder under `jobs_dir`, `label` being the file name `name` reduced to
    [A-Za-z0-9_-] - a client's file name is never used as a path. Never an existing folder: a clash gets a suffix."""
    label = re.sub(r"[^A-Za-z0-9_-]", "_", Path(name).stem)[:36] or "video"
    base = f"{datetime.now():%Y%m%d-%H%M%S}-{label}"
    Path(jobs_dir).mkdir(parents=True, exist_ok=True)
    for i in range(1, 100):
        folder = Path(jobs_dir) / (base if i == 1 else f"{base}-{i}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise FileExistsError(base)


def frame_range(n_frames, fps, start_s=0.0, end_s=None):
    """[first, last) frame indices of the stretch [start_s, end_s) seconds. With `n_frames` None (a container that
    does not say how long it is) and no `end_s`, `last` is None: the caller reads until the video ends."""
    first = int(round(start_s * fps))
    last = None if end_s is None else int(round(end_s * fps))
    if n_frames is not None:
        if first >= n_frames:
            raise JobError("el tramo empieza después del final del vídeo")
        last = n_frames if last is None else min(last, n_frames)
    if last is not None and last <= first:
        raise JobError("el tramo está vacío")
    return first, last


class Status:
    """The job's `status.json`, owned by the worker. Always written whole (temp file + `os.replace`), so a reader never
    sees half of it; at most once a second unless forced (start and end). Windows refuses to replace a file while
    another handle holds it open - the server polling it - so a replace is retried."""

    def __init__(self, out_dir, **fields):
        self.path, self.fields, self._last = Path(out_dir) / "status.json", dict(fields), 0.0

    def update(self, force=False, **fields):
        self.fields.update(fields)
        if not force and time.monotonic() - self._last < 1.0:
            return
        tmp = self.path.with_name("status.tmp")
        tmp.write_text(json.dumps(self.fields), encoding="utf-8")
        for _ in range(20 if force else 3):
            try:
                os.replace(tmp, self.path)
                break
            except PermissionError:
                time.sleep(0.05)
        else:
            print("status.json busy: update skipped", flush=True)
        self._last = time.monotonic()


class H264Writer:
    """BGR frames (all the same size) -> an H.264 mp4 that browsers play (`cv2.VideoWriter` writes mp4v, which they do
    not). yuv420p needs even sides: an odd width or height gets one replicated column or row. No file exists until
    the first frame is written."""

    def __init__(self, path, fps, width, height):
        self.pad = (width % 2, height % 2)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001),
                                                options={"crf": "23", "preset": "veryfast"})
        self.stream.width, self.stream.height = width + self.pad[0], height + self.pad[1]
        self.stream.pix_fmt = "yuv420p"

    def write(self, bgr):
        if any(self.pad):
            bgr = cv2.copyMakeBorder(bgr, 0, self.pad[1], 0, self.pad[0], cv2.BORDER_REPLICATE)
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(bgr, format="bgr24")):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


ROLE_COLOURS = {"goalie": (180, 0, 180), "referee": (0, 215, 255), "ball": (255, 255, 255)}  # BGR
TEAM_COLOURS = [(0, 0, 255), (255, 0, 0)]  # team 0 red, team 1 blue
NO_TEAM = (0, 200, 0)                       # a player with no team yet, or too little jersey colour to tell
TEAM_SAMPLES = 300                          # jerseys collected before the two teams are fitted
TRAIL, POSSESSION_M = 25, 3.0               # ball positions kept; the nearest player within this many metres owns it
GPU_FULL = "sin memoria en la GPU: usa CPU o espera"


def feet(xyxy):
    """Bottom centre of each box: the only point of a player on the field plane, which is what H maps."""
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    return np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]]


def track_rows(frame, fps, shot_id, ids, roles, teams, world, inside):
    """tracks.csv rows of one frame: the detections inside the field. A track ByteTrack has not confirmed yet (id -1)
    gets no id rather than a -1 shared by every such detection; a player with no team gets an empty team."""
    return [[frame, round(frame / fps, 3), shot_id, "" if ids[i] < 0 else int(ids[i]), roles[i],
             "" if teams[i] is None else teams[i], round(float(world[i, 0]), 3), round(float(world[i, 1]), 3)]
            for i in np.flatnonzero(inside)]


def detector_weights(cfg):
    """The detector's local weights, else its copy on the Hugging Face Hub, else the bare file name, which ultralytics
    downloads for its stock models."""
    path = Path(cfg["detector"])
    if path.exists():
        return path
    if cfg["detector_hub"]:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(*cfg["detector_hub"]))
    return Path(path.name)


def process_video(video, sport, out_dir, start_s=0.0, end_s=None, device="cuda:0", video_name=None):
    """Writes `out.mp4`, `tracks.csv` and `status.json` for `video` into `out_dir` (ADR 0006) and returns the final
    state: done, failed or cancelled. Refuses a folder that already holds a run: run folders are never reused."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists():
        raise FileExistsError(f"{out} already holds a run: a run folder is never reused")
    status = Status(out, state="running", pid=os.getpid(), pid_started=psutil.Process().create_time(), sport=sport,
                    video_name=video_name or Path(video).name, start_s=start_s, end_s=end_s, device=device,
                    frame=0, total=None, speed=0.0, error=None, created=_now(), finished=None)
    status.update(force=True)
    state, cap, writer, csv_file = "failed", None, None, None
    try:
        cfg = SPORTS[sport]
        if not Path(cfg["kpline"]).exists():
            raise JobError(f"faltan los pesos: {cfg['kpline']}")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise JobError(f"no se pudo abrir el vídeo {status.fields['video_name']}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first, last = frame_range(n if n > 0 else None, fps, start_s, end_s)
        status.update(force=True, total=None if last is None else last - first)

        from ultralytics import YOLO  # torch: only the worker loads models

        detector = YOLO(str(detector_weights(cfg)))
        roles = {cid: cfg["roles"][name] for cid, name in detector.names.items() if name in cfg["roles"]}
        if not roles:
            raise JobError(f"el detector no tiene ninguna de las clases {sorted(cfg['roles'])}")
        field = sports.get(sport)
        homography = HomographyPipeline([KplineEstimator(cfg["kpline"], sport=sport, device=device)],
                                        hold_frames=cfg["hold_frames"], smooth=cfg["smooth"])
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x0, x1, y0, y1 = field.box
        # 30% of the frame width, but never taller than 60% of its height (portrait or very wide videos)
        base, to_px = draw.minimap(field, int(min(0.3 * w, 0.6 * h * (x1 - x0) / (y1 - y0))))
        mh, mw = base.shape[:2]
        down = compute_downscale_factor(max(w, h))
        small = (round(w / down), round(h / down))  # the shot-cut detector only needs a ~256 px frame
        if first:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        writer = H264Writer(out / "out.mp4", fps, w, h)
        csv_file = open(out / "tracks.csv", "w", newline="", encoding="utf-8")
        rows = csv.writer(csv_file)
        rows.writerow(["frame", "time_s", "shot_id", "track_id", "role", "team", "x_m", "y_m"])
        cuts, tracker, shot_id = ContentDetector(), ByteTrackTracker(), 0
        jerseys, centres, trail = [], None, deque(maxlen=TRAIL)
        idx, t0, cancelled = first, time.monotonic(), False
        while last is None or idx < last:
            if (out / "cancel").exists():
                cancelled = True
                break
            ok, frame = cap.read()
            if not ok:
                break
            if cuts.process_frame(FrameTimecode(idx, fps=fps), cv2.resize(frame, small)):
                tracker, shot_id = ByteTrackTracker(), shot_id + 1  # ids and the carried H belong to their shot
                homography.reset()
                trail.clear()
            est = homography(frame)
            found = detector.predict(source=frame, conf=0.25, imgsz=cfg["imgsz"], classes=list(roles),
                                     verbose=False, device=device)[0]
            tracked = tracker.update(detections=sv.Detections.from_ultralytics(found))
            ids = tracked.tracker_id
            role = [roles[int(c)] for c in tracked.class_id]
            team = [None] * len(tracked)
            for i in (i for i, r in enumerate(role) if r == "player"):
                hist = jersey_histogram(frame, tracked.xyxy[i])
                if hist is None:
                    continue
                if centres is not None:
                    team[i] = int(team_of([hist], centres)[0][0])
                    continue
                jerseys.append(hist)  # teams are global to the video: fitted once, never reset at cuts
                if len(jerseys) >= TEAM_SAMPLES:
                    centres = fit_teams(jerseys)
            colour = [TEAM_COLOURS[t] if t is not None else ROLE_COLOURS.get(r, NO_TEAM) for r, t in zip(role, team)]
            vis, mini, ball = frame.copy(), base.copy(), False
            for (bx1, by1, bx2, by2), tid, c in zip(tracked.xyxy.astype(int), ids, colour):
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), c, 2)
                if tid >= 0:
                    cv2.putText(vis, f"#{tid}", (bx1, max(15, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            if est is not None:
                draw.draw_field(vis, est.H, field)
                world, inside = project_to_field(est, feet(tracked.xyxy), field)
                rows.writerows(track_rows(idx, fps, shot_id, ids, role, team, world, inside))
                owners = [(team[i], world[i]) for i in np.flatnonzero(inside)
                          if role[i] == "player" and team[i] is not None]
                for i in np.flatnonzero(inside):
                    px = tuple(int(v) for v in np.round(to_px(world[i])[0]))
                    if role[i] == "ball":
                        near = min(owners, key=lambda o: np.hypot(*(o[1] - world[i])), default=None)
                        own = near is not None and np.hypot(*(near[1] - world[i])) <= POSSESSION_M
                        trail.append((px, TEAM_COLOURS[near[0]] if own else NO_TEAM))
                        ball = True
                        continue
                    cv2.circle(mini, px, 5, colour[i], -1)
                    cv2.circle(mini, px, 6, (30, 30, 30), 1)
                    if ids[i] >= 0:
                        cv2.putText(mini, str(ids[i]), (px[0] + 7, px[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (30, 30, 30), 1)
            for k in range(1, len(trail)):  # the ball's recent path, fading into the past
                (p1, _), (p2, c) = trail[k - 1], trail[k]
                cv2.line(mini, p1, p2, tuple(int(v * k / len(trail)) for v in c), 2)
            if ball:
                cv2.circle(mini, trail[-1][0], 4, ROLE_COLOURS["ball"], -1)
            cv2.putText(vis, "H: sin calibrar" if est is None else f"H: {est.method}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            vis[h - mh - 10:h - 10, 10:10 + mw] = mini
            writer.write(vis)
            idx += 1
            done = idx - first
            status.update(frame=done, speed=round(done / (time.monotonic() - t0), 2))
            if done % 100 == 0:
                print(f"{done} frames, {status.fields['speed']} frames/s", flush=True)
        state = "cancelled" if cancelled else "done"
    except JobError as e:
        status.fields["error"] = str(e)
    except Exception as e:  # noqa: BLE001  any other failure ends the job as failed, traceback in worker.log
        traceback.print_exc()
        status.fields["error"] = GPU_FULL if "out of memory" in str(e).lower() else f"{type(e).__name__}: {e}"
        state = "failed"
    finally:
        if cap is not None:
            cap.release()
        if csv_file is not None:
            csv_file.close()
        try:
            if writer is not None:
                writer.close()  # what was processed stays playable, even after a failure or a cancel
        finally:
            status.update(force=True, state=state, finished=_now())
    return state


def main():
    ap = argparse.ArgumentParser(description="Annotated video, minimap and tracks.csv for one video (ADR 0006).")
    ap.add_argument("video")
    ap.add_argument("--sport", required=True, choices=sorted(SPORTS))
    ap.add_argument("--out", help="run folder (default: a new one under runs/product, listed by the web app)")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, help="seconds (default: the end of the video)")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--name", help="name shown for the video (default: its file name); pass it as --name=VALUE")
    a = ap.parse_args()
    out = Path(a.out) if a.out else new_run_folder(JOBS_DIR, Path(a.video).name)
    state = process_video(a.video, a.sport, out, a.start, a.end, a.device, a.name)
    print(f"{state}: {out}", flush=True)
    sys.exit(0 if state == "done" else 1)


if __name__ == "__main__":
    main()
