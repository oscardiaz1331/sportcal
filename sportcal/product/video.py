"""One video in, the product out (ADR 0006): the annotated video with the field lines and a minimap, the players'
positions in metres, and a status file the web app polls. Sport-agnostic: what differs per sport is a row of SPORTS.

The worker is its own process (`python -m sportcal.product.video`), started by the web app or by hand. Torch and
ultralytics load inside `process_video`, so importing this module (the server does, for SPORTS) stays light.
"""
import json
import os
import re
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import cv2

from sportcal.paths import ROOT, RUNS

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
