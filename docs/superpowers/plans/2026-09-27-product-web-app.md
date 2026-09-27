# Product Web App (ADR 0006) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app where the owner uploads a video, picks its sport, and gets back the video with the field
lines, the players and a minimap, plus their positions in metres, for hockey (NHL) and soccer.

**Architecture:** A Starlette server (`product/server.py`, 127.0.0.1 only) saves the upload into a new run folder and
starts a detached worker process (`python -m sportcal.product.video`) that writes `out.mp4`, `tracks.csv` and
`status.json` there; the page (`product/web/index.html`) polls the server, which derives everything from the run
folders. The worker is sport-agnostic: per-sport weights and classes live in one table (`video.SPORTS`); the minimap and
field lines are drawn from each sport's template (`product/draw.py`).

**Tech Stack:** Python 3.12 (venv), Starlette 1.6 + uvicorn 0.53, PyAV 18 (libx264), OpenCV 5 (contrib), ultralytics
YOLO26, `trackers` ByteTrack, PySceneDetect 0.7, psutil 7, plain HTML/JS. All installed already.

**Spec:** `docs/decisions/0006-product-web-app.md` (ADR 0006). Read it first; this plan implements it.

## Global Constraints

- Run every command from the repo root `REPO_ROOT` with `venv/Scripts/python.exe` (Git Bash
  paths below). Never `cd` elsewhere for a command that writes data.
- **A training is running on the GPU.** Do not edit anything the trainer imports: `sportcal/core/`, `sportcal/sports/`,
  `sportcal/models/`, `sportcal/lab/common/`, `sportcal/paths.py`. This plan touches only `sportcal/product/`, `tests/`,
  docs and `pyproject.toml`. Every test and manual check uses `device="cpu"` / `--device cpu`; nothing may run on the GPU.
- The server binds to `127.0.0.1` only. One job at a time. The server process never imports torch.
- A run folder is never reused: `mkdir(exist_ok=False)` for new folders; the worker refuses a folder that already has
  a `status.json`.
- v1 sports: `hockey-nhl` and `soccer-fifa` only.
- Code, comments and docstrings in English; UI text (page, error messages shown to the owner) in Spanish. Docstrings
  never quote experiment numbers.
- A deliberate simplification with a known ceiling gets a `# ponytail:` comment naming the ceiling and the upgrade path.
- No new dependencies to install. Fast tests: no GPU, no weights, no datasets. Anything needing weights or taking more
  than ~5 s gets `@pytest.mark.slow`.
- Fast suite: `venv/Scripts/python.exe -m pytest` (must stay green after every task). Everything:
  `venv/Scripts/python.exe -m pytest -m ""`.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Windows facts this plan relies on (all verified 2026-09-27):
  - `venv/Scripts/python.exe` is a launcher: the pid `Popen` returns is NOT the worker's pid. Liveness uses the pid the
    worker writes into `status.json` itself.
  - `os.kill(pid, 0)` on Windows kills the process. Use `psutil` to probe.
  - `os.replace` onto a file another process holds open raises `PermissionError`: retry.
  - PyAV writes no file at all when no frame was encoded.
  - argparse reads `--name -x.mp4` as an option: always pass `--name=VALUE`.

## Review Focus

1. **An upload whose name starts with `-`** (`-final.mp4`): argparse would take `--name -final.mp4` as an option, so
   the worker would never start. Expected: the job starts and lists with that name. Pinned in Task 5
   (`test_the_client_file_name_is_never_a_path`).
2. **A frame with no detections** (close-up, replay graphics): `project_to_field` with zero points crashed
   (`cv2.perspectiveTransform` returns None). Expected: empty arrays, the job goes on. Pinned in Task 1.
3. **Detections ByteTrack has not confirmed yet** (`tracker_id == -1`: the first frame of every track, most balls).
   Expected: drawn and logged with no id, never `#-1` or a shared `-1` in `tracks.csv`. Pinned in Task 4
   (`test_track_rows_keep_what_is_inside_and_leave_unconfirmed_ids_empty`).
4. **A container that does not report its frame count** (some `.mkv`/`.webm`: `CAP_PROP_FRAME_COUNT` is 0).
   Expected: processed until the video ends, total unknown; not "el tramo empieza después del final". Pinned in Task 3
   (`test_frame_range_clamps_to_the_video_and_refuses_what_is_not_in_it`).
5. **The server reading `status.json` at the instant the worker replaces it** (Windows refuses the replace). Expected:
   the worker retries; the job neither dies nor loses its final state. Pinned in Task 3
   (`test_status_waits_for_a_reader_that_holds_the_file`).

## File Map

| File | Task | Responsibility |
|---|---|---|
| `sportcal/product/pipeline.py` (modify) | 1 | `project_to_field`: use `sport.box`; accept zero points |
| `sportcal/product/draw.py` (create) | 2 | minimap and projected field lines for any `Sport` |
| `sportcal/product/video.py` (create) | 3, 4 | sport table, run folders, status file, H.264 writer, `process_video`, CLI |
| `sportcal/product/server.py` (create) | 5 | Starlette app: jobs API, static run files, page |
| `sportcal/product/web/index.html` (create) | 6 | the page |
| `sportcal/product/hockey_demo.py` (delete) | 4 | replaced by `video.py` |
| `tests/test_product.py` (modify) | 1 | `project_to_field` tests |
| `tests/test_product_draw.py` (create) | 2 | minimap orientation, lines behind the camera |
| `tests/test_product_video.py` (create) | 3, 4 | worker building blocks, failure paths, slow real clip |
| `tests/test_product_server.py` (create) | 5, 6 | jobs API with a stand-in worker, no torch in the server, page served |
| `pyproject.toml` (modify) | 3 | `product` extra |
| `CLAUDE.md`, `README.md`, `docs/architecture.md`, `sportcal/lab/hockey/README.md`, `docs/porting-status.md`, `docs/decisions/0006-product-web-app.md` (modify) | 4, 6 | commands and references |

## Before you start

- [ ] Check the baseline is green and nothing else is broken:

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: all pass (a few `slow` deselected).

---

### Task 1: `project_to_field` works for centred fields and for no points

`project_to_field` tests `0 <= x <= length`, which assumes the world origin at a corner (hockey). Soccer's origin is
the centre spot (`Sport.centred`), so every player in the left half (x < 0) is flagged outside. And with no points,
`core.geometry.image_to_world` crashes (`cv2.perspectiveTransform` returns None); that guard belongs in `core`, which the
running training imports, so it goes here with a `ponytail:` note.

**Files:**
- Modify: `sportcal/product/pipeline.py` (function `project_to_field`, end of file)
- Test: `tests/test_product.py` (append after `test_project_to_field_flags_points_outside_the_rink`)

**Interfaces:**
- Consumes: `Sport.box -> (x0, x1, y0, y1)` (existing, `sportcal/sports/base.py`).
- Produces: `project_to_field(estimate, img_xy, sport, margin=1.0) -> (world (N, 2) float, inside (N,) bool)`, now
  correct for any sport and for N = 0. Used by Task 4.

- [ ] **Step 1: Write the failing tests** - append to `tests/test_product.py` (it already defines `H`, an affine
  world -> image map, and `E(method)`, an `Estimate` with that `H`):

```python
def test_project_to_field_uses_the_box_of_a_centred_field():
    """Soccer's origin is the centre spot: a player in the left half has x < 0 and is still on the pitch."""
    sport = get("soccer-fifa")
    world_in, world_out = np.array([[-40.0, 10.0]]), np.array([[-60.0, 10.0]])
    img = np.vstack([np.c_[world_in, [1]] @ H.T, np.c_[world_out, [1]] @ H.T])[:, :2]
    world, inside = project_to_field(E("a"), img, sport)
    assert np.allclose(world, np.vstack([world_in, world_out]), atol=1e-6)
    assert inside.tolist() == [True, False]


def test_project_to_field_with_no_points_returns_empty_arrays():
    """A close-up with no player in it: nothing to project, and no crash."""
    world, inside = project_to_field(E("a"), np.zeros((0, 2)), get("hockey-nhl"))
    assert world.shape == (0, 2) and inside.shape == (0,)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_product.py -k project_to_field -v`
Expected: `test_project_to_field_uses_the_box_of_a_centred_field` FAILS (`[False, False] != [True, False]`);
`test_project_to_field_with_no_points_returns_empty_arrays` FAILS (`AttributeError: 'NoneType' object has no attribute
'reshape'`); the existing hockey test passes.

- [ ] **Step 3: Fix `project_to_field`** - replace the whole function at the end of `sportcal/product/pipeline.py`
  with:

```python
def project_to_field(estimate, img_xy, sport, margin=1.0):
    """Image pixels (e.g. players' feet) -> field metres, plus a mask of points that land
    inside the field (`sport.box`, with `margin` metres of slack). Only feet/ground contact points are
    meaningful: H is a plane-to-plane map."""
    img_xy = np.asarray(img_xy, float).reshape(-1, 2)
    # ponytail: cv2.perspectiveTransform returns None for no points; this guard belongs in core.geometry.image_to_world,
    # which a running training imports (CLAUDE.md) - move it there when core can be edited
    if not len(img_xy):
        return np.zeros((0, 2)), np.zeros(0, bool)
    world = image_to_world(estimate.H, img_xy)
    x0, x1, y0, y1 = sport.box
    inside = ((world[:, 0] >= x0 - margin) & (world[:, 0] <= x1 + margin)
              & (world[:, 1] >= y0 - margin) & (world[:, 1] <= y1 + margin))
    return world, inside
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_product.py -v`
Expected: all PASS (the slow kpline test is deselected).

- [ ] **Step 5: Commit**

```bash
git add sportcal/product/pipeline.py tests/test_product.py
git commit -m "product: project_to_field uses the sport's field box (soccer is centred) and accepts no points

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `product/draw.py` - minimap and field lines for any sport

The old demo drew hockey only (`sports.hockey.rink.minimap_base`, `rink.draw_rink`); soccer's drawing lives in the lab,
which the product may not import. Both can be drawn from the template every sport already has (`Sport.polylines()`,
`box`, `y_down`). The minimap must look like the main camera: +y down when `sport.y_down` (hockey), up otherwise
(soccer: the far touchline, +y, at the top). Field lines seen through H must not come back mirrored from behind the
camera: `core.geometry.project_points` already drops those points; lines are resampled so a line crossing the horizon
is cut instead of joined by a bogus segment.

**Files:**
- Create: `sportcal/product/draw.py`
- Test: `tests/test_product_draw.py`

**Interfaces:**
- Consumes: `Sport.polylines() -> [(class, (N, 2) world polyline)]`, `Sport.box`, `Sport.y_down`, `Sport.surface`;
  `core.geometry.project_points(H, world, w, h) -> (xy (N, 2), ok (N,) bool)`.
- Produces (used by Task 4):
  - `minimap(sport, width_px) -> (img (height, width_px, 3) uint8, to_px)`, `to_px(world (N, 2) or (2,)) -> (N, 2)
    float` minimap pixels.
  - `field_lines(H, sport, w, h, step=0.5) -> list[(M, 2) float]` pixel runs.
  - `draw_field(img, H, sport, colour=(0, 255, 255), thickness=1) -> img` (draws in place).

- [ ] **Step 1: Write the failing tests** - create `tests/test_product_draw.py`:

```python
"""The product's drawing for any sport (ADR 0006): minimap orientation, field lines behind the camera."""
import numpy as np

from sportcal.product.draw import field_lines, minimap
from sportcal.sports import get


def test_minimap_puts_the_field_corners_on_its_corners_the_way_the_main_camera_sees_them():
    hockey, soccer = get("hockey-nhl"), get("soccer-fifa")
    x0, x1, y0, y1 = hockey.box                      # corner origin; +y points down in the main camera
    cases = [(hockey, (x0, y0), (x1, y1))]
    x0, x1, y0, y1 = soccer.box                      # centred; +y (the far touchline) points up
    cases.append((soccer, (x0, y1), (x1, y0)))
    for sport, top_left, bottom_right in cases:
        img, to_px = minimap(sport, 400)
        h, w = img.shape[:2]
        assert w == 400, sport.name
        (xa, ya), (xb, yb) = to_px([top_left, bottom_right])
        assert xa == ya and 0 < xa < 40, sport.name            # the same margin on every side
        assert np.allclose((xb, yb), (w - xa, h - xa), atol=1), sport.name


def test_field_lines_keep_only_what_is_in_front_of_the_camera():
    """A camera that sees the rink up to x = 50 m: the last 11 m lie behind it and would come back mirrored, on the
    wrong side of the image (negative x), if they were drawn."""
    sport = get("hockey-nhl")
    H = np.array([[10.0, 0, 0], [0, 10.0, 0], [-1 / 50, 0, 1]])     # w = 1 - x/50: behind the camera past x = 50
    runs = field_lines(H, sport, 1920, 1080)
    pts = np.vstack(runs)
    assert len(runs) > 0 and np.isfinite(pts).all()
    assert (pts[:, 0] >= 0).all()                                     # nothing mirrored from behind the camera
    front = field_lines(np.diag([10.0, 10.0, 1.0]), sport, 1920, 1080)  # the whole rink in front
    assert sum(map(len, runs)) < sum(map(len, front))                  # the part behind was dropped, not kept
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_draw.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sportcal.product.draw'`.

- [ ] **Step 3: Write `sportcal/product/draw.py`**

```python
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
    H = np.asarray(H, float)
    runs = []
    for _, pl in sport.polylines():
        pl = np.asarray(pl, float)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
        t = np.r_[np.arange(0, d[-1], step), d[-1]]
        pts = np.c_[np.interp(t, d, pl[:, 0]), np.interp(t, d, pl[:, 1])]
        xy, ok = project_points(H, pts, w, h)
        for run in np.split(np.arange(len(pts)), np.flatnonzero(~ok)):
            run = run[ok[run]]
            if len(run) > 1:
                runs.append(xy[run])
    return runs


def draw_field(img, H, sport, colour=(0, 255, 255), thickness=1):
    """`field_lines` drawn on `img` (BGR, in place)."""
    h, w = img.shape[:2]
    for run in field_lines(H, sport, w, h):
        cv2.polylines(img, [np.round(run).astype(np.int32)], False, colour, thickness, cv2.LINE_AA)
    return img
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_draw.py tests/test_layering.py -v`
Expected: all PASS (the layering test confirms `product` imports only `core`).

- [ ] **Step 5: Mutation check** - in `minimap`, change `y = q[:, 1] - y0 if sport.y_down else y1 - q[:, 1]` to
  `y = q[:, 1] - y0`. Run `venv/Scripts/python.exe -m pytest tests/test_product_draw.py -v`: the minimap test must FAIL
  on `soccer-fifa`. Restore the line and re-run: PASS.

- [ ] **Step 6: Commit**

```bash
git add sportcal/product/draw.py tests/test_product_draw.py
git commit -m "product.draw: minimap and field lines for any sport, from its template

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `product/video.py` building blocks - sport table, run folders, status file, H.264 writer

The pieces of the worker that need no model, each with a fast test: the per-sport table of ADR 0006, run-folder
naming, the frame range of a stretch in seconds, the atomic `status.json`, and an H.264 writer (browsers do not play
the `mp4v` that `cv2.VideoWriter` writes).

**Files:**
- Create: `sportcal/product/video.py`
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)
- Test: `tests/test_product_video.py`

**Interfaces:**
- Consumes: `sportcal.paths.ROOT`, `sportcal.paths.RUNS`.
- Produces (used by Tasks 4 and 5):
  - `JOBS_DIR: Path` = `RUNS / "product"`; `ID_PATTERN: re.Pattern` = `^\d{8}-\d{6}-[A-Za-z0-9_-]{1,40}$`.
  - `SPORTS: dict[str, dict]`, keys `hockey-nhl`, `soccer-fifa`; each row has `label: str`, `kpline: Path`,
    `detector: Path`, `detector_hub: tuple[str, str] | None`, `roles: dict[str, str]` (detector class name -> one of
    `player`, `goalie`, `referee`, `ball`), `imgsz: int`, `hold_frames: int`, `smooth: float`.
  - `new_run_folder(jobs_dir, name) -> Path` (created, never an existing folder).
  - `class JobError(Exception)`: a failure whose message (Spanish) is shown to the owner as is.
  - `frame_range(n_frames: int | None, fps: float, start_s=0.0, end_s=None) -> (first: int, last: int | None)`.
  - `class Status(out_dir, **fields)` with `.fields: dict`, `.path: Path`, `.update(force=False, **fields)`.
  - `class H264Writer(path, fps, width, height)` with `.write(bgr)`, `.close()`.

- [ ] **Step 1: Write the failing tests** - create `tests/test_product_video.py`:

```python
"""The worker of the product (ADR 0006): building blocks and failure paths without GPU or weights; one slow test on a
real clip."""
import json
import sys
import threading

import av
import numpy as np
import pytest

from sportcal.product import video as V


def test_frame_range_clamps_to_the_video_and_refuses_what_is_not_in_it():
    assert V.frame_range(100, 25.0) == (0, 100)
    assert V.frame_range(100, 25.0, 1.0, 2.0) == (25, 50)
    assert V.frame_range(100, 25.0, 1.0, 60.0) == (25, 100)      # an end past the video: up to its end
    assert V.frame_range(None, 25.0, 1.0) == (25, None)           # the container does not say: read until the end
    for args in ((5.0,), (1.0, 1.0)):                              # starts after the end; an empty stretch
        with pytest.raises(V.JobError):
            V.frame_range(100, 25.0, *args)


def test_run_folders_are_new_and_named_from_a_clean_label(tmp_path):
    a = V.new_run_folder(tmp_path, "../../Mi vídeo final.mp4")
    b = V.new_run_folder(tmp_path, "../../Mi vídeo final.mp4")
    assert a.parent == tmp_path == b.parent and a != b             # same second, same name: still two folders
    assert a.name.endswith("-Mi_v_deo_final")
    long = V.new_run_folder(tmp_path, "x" * 100 + ".mp4")
    assert all(V.ID_PATTERN.match(f.name) for f in (a, b, long))


def test_h264_writer_pads_odd_sizes_and_writes_every_frame(tmp_path):
    w = V.H264Writer(tmp_path / "o.mp4", 29.97, 101, 51)
    for i in range(7):
        w.write(np.full((51, 101, 3), 30 * i, np.uint8))
    w.close()
    with av.open(str(tmp_path / "o.mp4")) as c:
        s = c.streams.video[0]
        assert (s.codec_context.name, s.width, s.height) == ("h264", 102, 52)
        assert sum(1 for _ in c.decode(s)) == 7


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to replace a file another handle holds open")
def test_status_waits_for_a_reader_that_holds_the_file(tmp_path):
    s = V.Status(tmp_path, state="running")
    s.update(force=True)
    reader = open(tmp_path / "status.json")                        # the server, mid-read
    threading.Timer(0.2, reader.close).start()
    s.update(force=True, state="done")                             # retried until the reader lets go
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "done"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_video.py -v`
Expected: FAIL with `ImportError: cannot import name 'video' from 'sportcal.product'`.

- [ ] **Step 3: Write `sportcal/product/video.py`**

```python
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
```

- [ ] **Step 4: Add the `product` extra** - in `pyproject.toml`, under `[project.optional-dependencies]`, after the
  `dev = ["pytest"]` line, add:

```toml
# the product: local web app + video worker (ADR 0006); the models come from `train`
product = ["starlette", "uvicorn", "python-multipart", "av", "trackers", "scenedetect", "psutil", "huggingface_hub"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_video.py tests/test_layering.py -v`
Expected: all PASS.

- [ ] **Step 6: Mutation checks** - (a) in `Status.update`, replace the retry loop by a bare `os.replace(tmp,
  self.path)`: `test_status_waits_for_a_reader_that_holds_the_file` must FAIL with `PermissionError`. (b) In
  `H264Writer.__init__`, set `self.pad = (0, 0)`: the writer test must FAIL. (c) In `frame_range`, delete the line
  `last = n_frames if last is None else min(last, n_frames)`: the frame-range test must FAIL. Restore each and re-run:
  PASS.

- [ ] **Step 7: Commit**

```bash
git add sportcal/product/video.py tests/test_product_video.py pyproject.toml
git commit -m "product.video: sport table, run folders, atomic status file, H.264 writer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `process_video` and its CLI; the old demo goes

The per-frame loop ported from `hockey_demo.py`, made sport-agnostic, plus the CLI the server starts. Order of the
early checks matters and is tested: a reused folder is refused before anything is written; missing weights fail before
the video is opened; a stretch outside the video fails with a message before any model loads.

**Files:**
- Modify: `sportcal/product/video.py` (imports, then append the loop, helpers and CLI)
- Delete: `sportcal/product/hockey_demo.py`
- Modify: `CLAUDE.md` (Commands block), `README.md` (line 39-40), `sportcal/lab/hockey/README.md` (line 24),
  `docs/porting-status.md` (item 3 and the `test.py` row)
- Test: `tests/test_product_video.py` (append)

**Interfaces:**
- Consumes: Task 1 `project_to_field`; Task 2 `draw.minimap`, `draw.draw_field`; Task 3 everything; existing
  `product.pipeline.HomographyPipeline(estimators, hold_frames, smooth)` (call it with a frame -> `Estimate(H,
  confidence, method)` or None; `.reset()` at shot cuts), `product.kpline.KplineEstimator(weights, sport, device)`,
  `core.teams.jersey_histogram(frame, bbox) -> hist | None`, `fit_teams(hists) -> centres`, `team_of(hists, centres) ->
  (teams, ratio)`.
- Produces (used by Task 5):
  - `process_video(video, sport, out_dir, start_s=0.0, end_s=None, device="cuda:0", video_name=None) -> "done" |
    "failed" | "cancelled"`; raises `FileExistsError` for a folder that already has `status.json`.
  - `status.json` fields: `state, pid, pid_started, sport, video_name, start_s, end_s, device, frame, total, speed,
    error, created, finished`.
  - CLI: `python -m sportcal.product.video VIDEO --sport S [--out DIR] [--start S] [--end S] [--device D]
    [--name=NAME]`; exit code 0 only for `done`.
  - `feet(xyxy) -> (N, 2)`, `track_rows(frame, fps, shot_id, ids, roles, teams, world, inside) -> list[list]`.

- [ ] **Step 1: Write the failing tests** - in `tests/test_product_video.py`, make the import block at the top read:

```python
import json
import sys
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from sportcal.paths import ROOT
from sportcal.product import video as V
```

then append to the end of the file:

```python
CLIP = ROOT / "nhl14.mp4"
START = 69.753  # frame 4181 of nhl14 (59.94 fps): a `fresh` frame the A2 model answers (tests/test_product.py)


def tiny_video(path, n=10, fps=10.0):
    """A synthetic clip OpenCV reads back."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), 20 * i, np.uint8))
    w.release()
    return path


def status_of(folder):
    return json.loads((folder / "status.json").read_text(encoding="utf-8"))


def test_a_run_folder_is_never_reused(tmp_path):
    (tmp_path / "status.json").write_text('{"state": "done"}')
    with pytest.raises(FileExistsError):
        V.process_video(tiny_video(tmp_path / "v.mp4"), "hockey-nhl", tmp_path)
    assert status_of(tmp_path) == {"state": "done"}                 # the earlier run is untouched


def test_missing_weights_fail_the_job_before_the_video_is_read(tmp_path, monkeypatch):
    monkeypatch.setitem(V.SPORTS, "hockey-nhl", {**V.SPORTS["hockey-nhl"], "kpline": tmp_path / "missing.pt"})
    assert V.process_video(tmp_path / "does-not-exist.mp4", "hockey-nhl", tmp_path / "run") == "failed"
    st = status_of(tmp_path / "run")
    assert st["state"] == "failed" and "missing.pt" in st["error"] and st["finished"]


def test_a_stretch_past_the_end_fails_with_a_message(tmp_path, monkeypatch):
    monkeypatch.setitem(V.SPORTS, "hockey-nhl", {**V.SPORTS["hockey-nhl"], "kpline": Path(__file__)})  # exists
    clip = tiny_video(tmp_path / "v.mp4")                           # 10 frames at 10 fps: 1 s
    assert V.process_video(clip, "hockey-nhl", tmp_path / "run", start_s=5.0) == "failed"
    assert status_of(tmp_path / "run")["error"] == "el tramo empieza después del final del vídeo"


def test_track_rows_keep_what_is_inside_and_leave_unconfirmed_ids_empty():
    world = np.array([[10.0, 5.0], [80.0, 5.0], [20.0, 3.0]])
    rows = V.track_rows(30, 30.0, 2, np.array([7, 8, -1]), ["player", "player", "ball"], [1, 0, None],
                        world, np.array([True, False, True]))
    assert rows == [[30, 1.0, 2, 7, "player", 1, 10.0, 5.0], [30, 1.0, 2, "", "ball", "", 20.0, 3.0]]


def needs_clip():
    if not (CLIP.exists() and V.SPORTS["hockey-nhl"]["kpline"].exists()):
        pytest.skip("needs nhl14.mp4 and the A2 weights")


@pytest.mark.slow
def test_a_real_clip_gives_a_playable_video_and_positions(tmp_path):
    needs_clip()
    run = tmp_path / "run"
    assert V.process_video(CLIP, "hockey-nhl", run, START, START + 0.5, device="cpu") == "done"
    st = status_of(run)
    assert st["frame"] == st["total"] == 30
    with av.open(str(run / "out.mp4")) as c:
        assert sum(1 for _ in c.decode(video=0)) == 30
    rows = (run / "tracks.csv").read_text().splitlines()
    assert rows[0] == "frame,time_s,shot_id,track_id,role,team,x_m,y_m" and len(rows) > 1


@pytest.mark.slow
def test_the_cancel_flag_stops_the_worker_and_leaves_a_playable_video(tmp_path):
    needs_clip()
    run = tmp_path / "run"

    def cancel_after_5_frames():
        while True:
            try:
                if status_of(run)["frame"] >= 5:
                    break
            except (FileNotFoundError, PermissionError, ValueError):
                pass
            time.sleep(0.05)
        (run / "cancel").touch()

    threading.Thread(target=cancel_after_5_frames, daemon=True).start()
    assert V.process_video(CLIP, "hockey-nhl", run, START, START + 10, device="cpu") == "cancelled"
    st = status_of(run)
    assert 5 <= st["frame"] < st["total"]
    with av.open(str(run / "out.mp4")) as c:
        assert sum(1 for _ in c.decode(video=0)) == st["frame"]
```

- [ ] **Step 2: Run the fast ones to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_video.py -v`
Expected: the four new fast tests FAIL with `AttributeError: module 'sportcal.product.video' has no attribute
'process_video'` (resp. `'track_rows'`); the two slow ones are deselected; Task 3's tests still PASS.

- [ ] **Step 3: Extend the imports of `sportcal/product/video.py`** - replace its import block (from `import json` to
  `from sportcal.paths import ROOT, RUNS`) with:

```python
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
```

- [ ] **Step 4: Append the loop, its helpers and the CLI** to the end of `sportcal/product/video.py`:

```python
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
```

- [ ] **Step 5: Run the fast tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_video.py tests/test_layering.py -v`
Expected: all fast tests PASS, the two slow ones deselected.

- [ ] **Step 6: Run the slow tests (CPU, real weights, ~2-3 min)**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_video.py -m slow -v`
Expected: both PASS. If `test_a_real_clip_gives_a_playable_video_and_positions` fails on `len(rows) > 1`, open
`out.mp4` from the test's tmp folder (path in the failure) and report what the frames show before changing anything.

- [ ] **Step 7: Mutation check** - in `track_rows`, replace `"" if ids[i] < 0 else int(ids[i])` by `int(ids[i])`:
  `test_track_rows_keep_what_is_inside_and_leave_unconfirmed_ids_empty` must FAIL. Restore; PASS.

- [ ] **Step 8: Delete the old demo and update what points to it**

```bash
git rm sportcal/product/hockey_demo.py
```

In `CLAUDE.md`, Commands block, replace the line

```
python -m sportcal.product.hockey_demo nhl11.mp4 [--max-frames 600] [--device cpu]  # minimap video, product pipeline
```

with

```
python -m sportcal.product.video nhl11.mp4 --sport hockey-nhl --end 20 [--device cpu]  # one video -> runs/product/<run>/
```

In `README.md`, replace the two lines

```
Whole video with detection, tracking, the rink lines drawn from the estimated H and a minimap (writes `tracked_out.mp4`,
`tracking_log.csv`): `python -m sportcal.product.hockey_demo nhl11.mp4 [--max-frames 600] [--device cpu]`.
```

with

```
Whole video with detection, tracking, the field lines drawn from the estimated H and a minimap, for hockey (NHL) or
soccer (writes `out.mp4`, `tracks.csv` and `status.json` into a new `runs/product/<run>/`, ADR 0006):
`python -m sportcal.product.video nhl11.mp4 --sport hockey-nhl [--start S] [--end S] [--device cpu]`.
```

In `sportcal/lab/hockey/README.md`, replace

```
`product/hockey_demo.py` uses the local weights when present and falls back to the HuggingFace ones.
```

with

```
`product/video.py` uses the local weights when present and falls back to the HuggingFace ones.
```

In `docs/porting-status.md`, replace item 3

```
3. **Split `product/hockey_demo.py`** (440 lines of module-level script: team classification, puck trail, minimap video)
   into a `main()` built on `HomographyPipeline`. A soccer estimator (`FieldSolver` behind an `Estimator`) is the natural
   second product method once soccer has a validation set.
```

with

```
3. ~~Split `product/hockey_demo.py`~~ **done 2026-09-27** (ADR 0006): replaced by the sport-agnostic worker
   `product/video.py` and the local web app `product/server.py`. Still open: a soccer estimator (`FieldSolver` behind an
   `Estimator`) as a second product method once soccer has a validation set.
```

and the table row

```
| `test.py` | `sportcal/product/hockey_demo.py` (script; refuses import) |
```

with

```
| `test.py` | `sportcal/product/hockey_demo.py`, replaced 2026-09-27 by `sportcal/product/video.py` (ADR 0006) |
```

- [ ] **Step 9: Check nothing else imports or names the demo**

Run: `git grep -n "hockey_demo" -- "*.py" "*.toml" CLAUDE.md README.md sportcal`
Expected: no output (the ADRs, the experiment write-ups and old plans under `docs/` keep their history and are not
searched).

- [ ] **Step 10: Run the fast suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: all PASS.

- [ ] **Step 11: Commit**

```bash
git add sportcal/product/video.py tests/test_product_video.py CLAUDE.md README.md sportcal/lab/hockey/README.md docs/porting-status.md
git commit -m "product.video: the sport-agnostic video worker and its CLI; hockey_demo.py removed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `product/server.py` - the jobs API

The Starlette app of ADR 0006 without the page (Task 6). All state is on disk; the only thing in memory is the lock
that makes "check nothing runs, create the folder, start the worker, wait for its first status" one step. Tests use a
stand-in worker: a short script with the real CLI's options that writes a `running` status like the real worker and
ends as `cancelled` when it sees the cancel flag.

**Files:**
- Create: `sportcal/product/server.py`
- Test: `tests/test_product_server.py`

**Interfaces:**
- Consumes: Task 3/4 `video.ID_PATTERN`, `video.JOBS_DIR`, `video.SPORTS`, `video.new_run_folder`, the CLI options and
  the `status.json` fields.
- Produces (used by Task 6):
  - `make_app(jobs=JOBS_DIR, worker=WORKER) -> Starlette`; `python -m sportcal.product.server` serves it on
    `127.0.0.1:8000`.
  - `GET /api/sports -> [{"id", "label"}]`; `GET /api/gpu -> {"used_mib": int | null}`;
    `GET /api/jobs -> [status fields + "id", "files": [names present among out.mp4, tracks.csv, worker.log],
    "log_tail": [last 20 lines] for failed/interrupted]`, newest first, `state` one of `running, done, failed,
    cancelled, interrupted`;
    `POST /api/jobs` (multipart `video`, `sport`, `start_s`, `end_s`, `device`) -> `201 {"id"}` | `400/409/500/507
    {"error"}`; `POST /api/jobs/{id}/cancel` -> `200 {"id"}` | `409`; `DELETE /api/jobs/{id}` -> `200 {"id"}` | `409`;
    bad id `400`, unknown id `404`; files under `/runs/{id}/<file>`.

- [ ] **Step 1: Write the failing tests** - create `tests/test_product_server.py`:

```python
"""The web app's job handling (ADR 0006) with a stand-in worker: no GPU, no weights."""
import json
import subprocess
import sys
import time

import psutil
import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

from sportcal.product import server  # noqa: E402

# Stands in for `python -m sportcal.product.video`: the same options, a `running` status written like the real worker
# (its own pid and start time, atomic replace retried), then it waits for the cancel flag (or 30 s) and ends cancelled.
FAKE_WORKER = r'''
import argparse, json, os, time, psutil
ap = argparse.ArgumentParser()
for opt in ("--sport", "--out", "--device", "--name"):
    ap.add_argument(opt)
ap.add_argument("video")
ap.add_argument("--start", type=float)
ap.add_argument("--end", type=float)
a = ap.parse_args()
def write(state):
    s = dict(state=state, pid=os.getpid(), pid_started=psutil.Process().create_time(), sport=a.sport,
             video_name=a.name, start_s=a.start, end_s=a.end, device=a.device, frame=0, total=10)
    tmp = os.path.join(a.out, "status.tmp")
    with open(tmp, "w") as f:
        json.dump(s, f)
    for _ in range(40):
        try:
            os.replace(tmp, os.path.join(a.out, "status.json"))
            break
        except PermissionError:
            time.sleep(0.05)
write("running")
t0 = time.time()
while not os.path.exists(os.path.join(a.out, "cancel")) and time.time() - t0 < 30:
    time.sleep(0.05)
write("cancelled")
'''


@pytest.fixture
def client(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    jobs = tmp_path / "jobs"
    with TestClient(server.make_app(jobs, [sys.executable, str(worker)])) as c:
        c.jobs = jobs
        yield c
        for f in jobs.iterdir():  # release any stand-in still waiting
            if f.is_dir():
                (f / "cancel").touch()


def post(client, name="clip.mp4", **form):
    data = {"sport": "hockey-nhl", "start_s": "0", "end_s": "", "device": "cpu", **form}
    return client.post("/api/jobs", files={"video": (name, b"not really a video", "video/mp4")}, data=data)


def wait_state(client, jid, state, timeout=20):
    t0 = time.time()
    while True:
        job = {j["id"]: j for j in client.get("/api/jobs").json()}[jid]
        if job["state"] == state:
            return job
        assert time.time() - t0 < timeout, f"{jid} never reached {state}: {job}"
        time.sleep(0.1)


def test_one_job_at_a_time_and_cancel_stops_it(client):
    r = post(client)
    assert r.status_code == 201, r.text
    jid = r.json()["id"]
    assert (client.jobs / jid / "input.mp4").read_bytes() == b"not really a video"
    assert wait_state(client, jid, "running")["sport"] == "hockey-nhl"
    assert post(client).status_code == 409                          # the GPU is not shared between jobs
    assert client.delete(f"/api/jobs/{jid}").status_code == 409      # nor a folder deleted under its worker
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200
    wait_state(client, jid, "cancelled")
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 409  # nothing left to cancel
    assert post(client).status_code == 201                          # the next job may start


@pytest.mark.parametrize("form", [dict(sport="curling"), dict(device="cuda:7"), dict(start_s="-1"),
                                  dict(start_s="nan"), dict(start_s="abc"), dict(start_s="10", end_s="5")])
def test_bad_requests_are_refused_and_leave_no_folder(client, form):
    assert post(client, **form).status_code == 400
    assert list(client.jobs.iterdir()) == []


def test_other_file_types_are_refused(client):
    assert post(client, name="notes.txt").status_code == 400
    assert list(client.jobs.iterdir()) == []


def test_the_client_file_name_is_never_a_path(client):
    for name in ("../../evil.mp4", "-starts-with-a-dash.mp4"):       # a traversal; an argparse option look-alike
        r = post(client, name=name)
        assert r.status_code == 201, r.text
        folder = client.jobs / r.json()["id"]
        assert folder.parent == client.jobs and (folder / "input.mp4").exists()
        assert wait_state(client, folder.name, "running")["video_name"] == name
        client.post(f"/api/jobs/{folder.name}/cancel")
        wait_state(client, folder.name, "cancelled")
    assert not (client.jobs.parent / "evil.mp4").exists()


def test_ids_that_are_not_a_run_folder_touch_nothing(client):
    victim = client.jobs.parent / "victim"
    victim.mkdir()
    for jid in ("%2e%2e", "..%2Fvictim", "20260927-120000-nope"):   # %2e%2e reaches the endpoint as ".."
        assert client.delete(f"/api/jobs/{jid}").status_code in (400, 404)
        assert client.post(f"/api/jobs/{jid}/cancel").status_code in (400, 404)
    assert victim.exists()


def test_a_finished_job_is_deleted_whole(client):
    folder = client.jobs / "20260927-120000-done"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"state": "done"}))
    (folder / "out.mp4").write_bytes(b"x")
    assert client.delete(f"/api/jobs/{folder.name}").status_code == 200
    assert list(client.jobs.iterdir()) == []


def test_a_running_status_without_its_worker_lists_as_interrupted(client):
    """The pid is alive (this test's own) but started at another time: a reused pid, not the worker."""
    folder = client.jobs / "20260927-120000-orphan"
    folder.mkdir()
    me = psutil.Process()
    (folder / "status.json").write_text(json.dumps({"state": "running", "pid": me.pid,
                                                    "pid_started": me.create_time() - 100}))
    (folder / "worker.log").write_text("Traceback: boom\n")
    job = client.get("/api/jobs").json()[0]
    assert job["state"] == "interrupted" and job["log_tail"] == ["Traceback: boom"]
    assert post(client).status_code == 201                          # a dead job does not hold the lock


def test_the_server_does_not_load_torch():
    """Models load in the worker only: the server stays light and never holds GPU memory."""
    code = "import sys, sportcal.product.server; assert 'torch' not in sys.modules, 'the server imported torch'"
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_server.py -v`
Expected: FAIL with `ImportError: cannot import name 'server' from 'sportcal.product'`.

- [ ] **Step 3: Write `sportcal/product/server.py`**

```python
"""The product's local web app (ADR 0006): upload a video, choose its sport, follow the job, watch the result.

Local only (127.0.0.1), one user. Each job runs in its own worker process (`python -m sportcal.product.video`), so a job
survives the browser tab and the server, and the GPU is free between jobs. All job state lives in the run folders
(`runs/product/<id>/status.json`, written by the worker); the server keeps nothing in memory but a lock.
"""
import asyncio
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from sportcal.paths import ROOT
from sportcal.product.video import ID_PATTERN, JOBS_DIR, SPORTS, new_run_folder

WORKER = [sys.executable, "-m", "sportcal.product.video"]
PAGE = Path(__file__).parent / "web" / "index.html"
VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DEVICES = {"cuda:0", "cpu"}
START_TIMEOUT_S = 60
# Windows: no console and its own process group, so closing the server's console or Ctrl+C there does not stop a job
DETACHED = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0


def error(message, code=400):
    return JSONResponse({"error": message}, status_code=code)


def read_status(folder):
    """The job's status.json, or None when it has none. A read can meet the worker's replace (Windows): retried."""
    for _ in range(3):
        try:
            return json.loads((folder / "status.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, ValueError):
            time.sleep(0.05)
    return None


def alive(status):
    """True while the worker that wrote `status` runs: its pid AND its start time, since a pid is reused after an exit.
    Never os.kill(pid, 0): on Windows that kills the process."""
    try:
        return abs(psutil.Process(status["pid"]).create_time() - status["pid_started"]) < 1.0
    except (psutil.Error, KeyError, TypeError):
        return False


def is_running(status):
    return bool(status) and status.get("state") == "running" and alive(status)


def folders(jobs):
    """The run folders, newest first (their names start with the creation time)."""
    return sorted((f for f in jobs.iterdir() if f.is_dir() and ID_PATTERN.match(f.name)), reverse=True)


def running(jobs):
    """The id of the job that is running, or None."""
    return next((f.name for f in folders(jobs) if is_running(read_status(f))), None)


def job(folder):
    """What the page shows of a run folder. `interrupted` is derived here, never written: a `running` status whose
    worker is gone, or no status at all."""
    st = read_status(folder) or {"state": "interrupted"}
    if st.get("state") == "running" and not alive(st):
        st["state"] = "interrupted"
    st["id"] = folder.name
    st["files"] = [f for f in ("out.mp4", "tracks.csv", "worker.log") if (folder / f).exists()]
    if st["state"] in ("failed", "interrupted") and "worker.log" in st["files"]:
        st["log_tail"] = (folder / "worker.log").read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    return st


def job_folder(request):
    """(folder, None) for the job named in the path, else (None, the error response). The id must look like a run
    folder name - so no `..` and no separator - and exist directly under the jobs folder."""
    jid, jobs = request.path_params["id"], request.app.state.jobs
    if not ID_PATTERN.match(jid):
        return None, error("identificador de trabajo no válido")
    folder = jobs / jid
    if not folder.is_dir():
        return None, error("no existe ese trabajo", 404)
    return folder, None


def seconds(text, default):
    """A form field in seconds; `default` when empty. ValueError unless a finite number >= 0 ("nan" parses as float)."""
    if text is None or not str(text).strip():
        return default
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise ValueError(text)
    return value


async def page(request):
    return FileResponse(PAGE)


def list_sports(request):
    return JSONResponse([{"id": k, "label": v["label"]} for k, v in SPORTS.items()])


def gpu(request):
    """MiB of GPU memory in use, from nvidia-smi (the server never loads torch), or null without an NVIDIA GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
        return JSONResponse({"used_mib": int(out.split()[0])})
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return JSONResponse({"used_mib": None})


def list_jobs(request):
    return JSONResponse([job(f) for f in folders(request.app.state.jobs)])


async def create_job(request):
    state = request.app.state
    async with request.form() as form:
        upload, sport, device = form.get("video"), form.get("sport"), form.get("device") or "cuda:0"
        try:
            start_s, end_s = seconds(form.get("start_s"), 0.0), seconds(form.get("end_s"), None)
        except ValueError:
            return error("inicio y fin: segundos, un número >= 0")
        if sport not in SPORTS:
            return error(f"deporte desconocido: {sport}")
        if device not in DEVICES:
            return error(f"dispositivo no válido: {device}")
        if end_s is not None and end_s <= start_s:
            return error("el fin tiene que ser posterior al inicio")
        if not isinstance(upload, UploadFile) or not upload.filename:
            return error("falta el vídeo")
        ext = Path(upload.filename).suffix.lower()
        if ext not in VIDEO_TYPES:
            return error(f"formato no soportado ({ext or 'sin extensión'}): usa {', '.join(sorted(VIDEO_TYPES))}")
        # ponytail: one uvicorn process and one global lock - check, create, start and first status are one step, so
        # two quick submissions cannot both start; a queue replaces the 409 when several videos must wait in line
        async with state.lock:
            if busy := running(state.jobs):
                return error(f"ya hay un trabajo en marcha: {busy}", 409)
            folder = new_run_folder(state.jobs, upload.filename)
            video = folder / f"input{ext}"
            try:
                with open(video, "wb") as fh:
                    while chunk := await upload.read(1 << 20):
                        fh.write(chunk)
            except OSError as e:  # disk full...: no half-made job left behind
                shutil.rmtree(folder, ignore_errors=True)
                return error(f"no se pudo guardar el vídeo: {e}", 507)
            # "--name=": a file name that starts with "-" would otherwise be read as an option
            cmd = [*state.worker, str(video), "--sport", sport, "--out", str(folder), "--start", str(start_s),
                   "--device", device, f"--name={upload.filename}"]
            if end_s is not None:
                cmd.append(f"--end={end_s}")
            with open(folder / "worker.log", "w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, creationflags=DETACHED)
            deadline = time.monotonic() + START_TIMEOUT_S
            while not (folder / "status.json").exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    tail = (folder / "worker.log").read_text(encoding="utf-8", errors="replace")[-2000:]
                    return error(f"el proceso no arrancó:\n{tail}", 500)
                await asyncio.sleep(0.1)
    return JSONResponse({"id": folder.name}, status_code=201)


def cancel_job(request):
    folder, err = job_folder(request)
    if err:
        return err
    if not is_running(read_status(folder)):
        return error("el trabajo no está en marcha", 409)
    (folder / "cancel").touch()  # the worker checks it every frame and stops cleanly
    return JSONResponse({"id": folder.name})


def delete_job(request):
    folder, err = job_folder(request)
    if err:
        return err
    if is_running(read_status(folder)):
        return error("el trabajo está en marcha: cancélalo antes", 409)
    trash = folder.with_name(f".deleting-{folder.name}")
    shutil.rmtree(trash, ignore_errors=True)  # left over by an earlier delete that failed half-way
    try:
        folder.rename(trash)  # all or nothing: Windows refuses while any file inside is open
    except OSError:
        return error("archivo en uso: cierra lo que tengas abierto de este trabajo y vuelve a intentarlo", 409)
    shutil.rmtree(trash, ignore_errors=True)
    return JSONResponse({"id": folder.name})


def make_app(jobs=JOBS_DIR, worker=WORKER):
    """The app over the run folders in `jobs`, starting jobs with the command `worker` (tests pass a stand-in)."""
    jobs = Path(jobs)
    jobs.mkdir(parents=True, exist_ok=True)
    app = Starlette(routes=[
        Route("/", page),
        Route("/api/sports", list_sports),
        Route("/api/gpu", gpu),
        Route("/api/jobs", list_jobs, methods=["GET"]),
        Route("/api/jobs", create_job, methods=["POST"]),
        Route("/api/jobs/{id}/cancel", cancel_job, methods=["POST"]),
        Route("/api/jobs/{id}", delete_job, methods=["DELETE"]),
        Mount("/runs", StaticFiles(directory=jobs)),  # out.mp4 with range requests: the player can seek
    ])
    app.state.jobs, app.state.worker, app.state.lock = jobs, list(worker), asyncio.Lock()
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(make_app(), host="127.0.0.1", port=8000)
```

- [ ] **Step 4: Run the tests to verify they pass** (the stand-in workers make this file take ~15 s)

Run: `venv/Scripts/python.exe -m pytest tests/test_product_server.py tests/test_layering.py -v`
Expected: all PASS (a `StarletteDeprecationWarning` about httpx in `starlette.testclient` is expected and harmless).

- [ ] **Step 5: Mutation checks** - (a) in `alive`, return `psutil.pid_exists(status["pid"])` instead of comparing
  start times: `test_a_running_status_without_its_worker_lists_as_interrupted` must FAIL. (b) In `create_job`, pass
  `"--name", upload.filename` instead of `f"--name={upload.filename}"`: `test_the_client_file_name_is_never_a_path` must
  FAIL (the stand-in never starts: 500). (c) In `job_folder`, drop the `ID_PATTERN` check:
  `test_ids_that_are_not_a_run_folder_touch_nothing` must FAIL or the victim folder must be gone. Restore each; PASS.

- [ ] **Step 6: Run the fast suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add sportcal/product/server.py tests/test_product_server.py
git commit -m "product.server: local jobs API over the run folders, one detached worker at a time

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: the page, the docs, and the end-to-end check

**Files:**
- Create: `sportcal/product/web/index.html`
- Test: `tests/test_product_server.py` (append)
- Modify: `CLAUDE.md` (Where things are, Commands), `README.md`, `docs/architecture.md` (layer list, Running),
  `docs/decisions/0006-product-web-app.md` (status, speed)

**Interfaces:**
- Consumes: Task 5's routes and JSON exactly as listed there.
- Produces: the page at `GET /`.

- [ ] **Step 1: Write the failing test** - append to `tests/test_product_server.py`:

```python
def test_the_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and 'id="form"' in r.text and "/api/jobs" in r.text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_server.py::test_the_page_is_served -v`
Expected: FAIL (`RuntimeError: File at path ...index.html does not exist.`).

- [ ] **Step 3: Write `sportcal/product/web/index.html`**

```html
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sportcal - minimapa</title>
<style>
  body { font: 15px/1.4 system-ui, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; color: #222; background: #fff; }
  form, .job { border: 1px solid #ccc; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }
  label { display: inline-block; margin: 4px 16px 4px 0; }
  input.time { width: 6em; }
  progress { width: 100%; }
  .muted { color: #666; font-size: 13px; }
  .warn { color: #b00; font-size: 13px; }
  .job h3 { margin: 0 0 4px; font-size: 16px; overflow-wrap: anywhere; }
  .job video { width: 100%; margin-top: 8px; background: #000; }
  .job pre { background: #f6f6f6; padding: 8px; overflow-x: auto; font-size: 12px; }
  .job button, .job a { margin: 6px 12px 0 0; }
</style>
</head>
<body>
<h1>Minimapa desde vídeo</h1>
<form id="form">
  <label>Vídeo <input type="file" name="video" accept="video/*" required></label><br>
  <label>Deporte <select name="sport" id="sport"></select></label>
  <label>Inicio <input class="time" id="start" placeholder="0:00"></label>
  <label>Fin <input class="time" id="end" placeholder="final"></label>
  <label>Dispositivo <select name="device"><option value="cuda:0">GPU</option><option value="cpu">CPU</option></select></label>
  <span id="gpu" class="muted"></span><br>
  <button type="submit">Procesar</button> <span id="msg"></span>
  <progress id="upload" hidden></progress>
</form>
<div id="jobs"></div>
<script>
const STATES = {running: "en marcha", done: "terminado", failed: "falló", cancelled: "cancelado", interrupted: "interrumpido"};
const $ = sel => document.querySelector(sel);
const cards = new Map();  // job id -> its card, updated in place: a <video> that is playing is never recreated

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text) e.textContent = text;  // never innerHTML: names come from the uploaded file
  if (cls) e.className = cls;
  return e;
}

function button(text, onclick) {
  const b = el("button", text);
  b.type = "button";
  b.onclick = onclick;
  return b;
}

function seconds(text) {  // "[[hh:]mm:]ss" -> seconds; "" -> null; NaN when malformed
  text = text.trim();
  if (!text) return null;
  if (!/^\d+(:\d{1,2}){0,2}(\.\d+)?$/.test(text)) return NaN;
  return text.split(":").reduce((total, part) => total * 60 + Number(part), 0);
}

function duration(s) {
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return h ? `${h} h ${m} min` : m ? `${m} min` : `${s} s`;
}

async function call(method, url) {
  const r = await fetch(url, {method});
  if (!r.ok) alert((await r.json().catch(() => ({}))).error || `error ${r.status}`);
  refresh();
}

function render(job) {
  let c = cards.get(job.id);
  if (!c) {
    c = {root: el("div", "", "job"), title: el("h3"), info: el("div", "", "muted"), bar: el("progress"),
         actions: el("div"), extra: el("div"), key: null};
    c.root.append(c.title, c.info, c.bar, c.actions, c.extra);
    cards.set(job.id, c);
  }
  const running = job.state === "running";
  c.title.textContent = job.video_name || job.id;
  const parts = [job.sport || "", STATES[job.state] || job.state];
  if (job.frame) parts.push(job.total ? `${job.frame} / ${job.total} frames` : `${job.frame} frames`);
  if (running && job.speed) parts.push(`${job.speed} frames/s`);
  if (running && job.speed && job.total) parts.push(`quedan ${duration((job.total - job.frame) / job.speed)}`);
  if (job.error) parts.push(job.error);
  c.info.textContent = parts.filter(Boolean).join(" · ");
  c.bar.hidden = !(running && job.total);
  if (job.total) { c.bar.max = job.total; c.bar.value = job.frame; }
  const key = `${job.state}|${job.files.join()}`;  // buttons, links and player change only with these
  if (c.key === key) return;
  c.key = key;
  c.actions.replaceChildren();
  c.extra.replaceChildren();
  c.actions.append(running
    ? button("Cancelar", () => call("POST", `/api/jobs/${job.id}/cancel`))
    : button("Borrar", () => confirm(`¿Borrar «${job.video_name || job.id}» y todos sus archivos?`)
                             && call("DELETE", `/api/jobs/${job.id}`)));
  for (const f of job.files) {
    const a = el("a", f);
    a.href = `/runs/${job.id}/${f}`;
    a.download = "";
    c.actions.append(a);
  }
  if (!running && job.files.includes("out.mp4")) {
    const v = el("video");
    v.controls = true;
    v.preload = "metadata";
    v.src = `/runs/${job.id}/out.mp4`;
    c.extra.append(v);
  }
  if (job.log_tail) c.extra.append(el("pre", job.log_tail.join("\n")));
}

async function refresh() {
  const jobs = await (await fetch("/api/jobs")).json();
  const ids = new Set(jobs.map(j => j.id));
  for (const [id, c] of cards) if (!ids.has(id)) { c.root.remove(); cards.delete(id); }
  for (const job of [...jobs].reverse()) {  // oldest first, each new card on top: existing cards never move
    const isNew = !cards.has(job.id);
    render(job);
    if (isNew) $("#jobs").prepend(cards.get(job.id).root);
  }
}

async function gpu() {
  const {used_mib} = await (await fetch("/api/gpu")).json();
  const busy = used_mib > 1024;
  $("#gpu").textContent = used_mib == null ? "" : `GPU: ${used_mib} MiB en uso${busy ? " — ¿hay un entrenamiento?" : ""}`;
  $("#gpu").className = busy ? "warn" : "muted";
}

$("#form").addEventListener("submit", ev => {
  ev.preventDefault();
  const start = seconds($("#start").value), end = seconds($("#end").value);
  if (Number.isNaN(start) || Number.isNaN(end)) { $("#msg").textContent = "Inicio y fin: [[hh:]mm:]ss"; return; }
  const data = new FormData(ev.target), bar = $("#upload"), xhr = new XMLHttpRequest();
  data.set("start_s", start ?? "");
  data.set("end_s", end ?? "");
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = e => { bar.hidden = false; bar.max = e.total; bar.value = e.loaded; };
  xhr.upload.onload = () => { $("#msg").textContent = "arrancando el proceso…"; };
  xhr.onload = () => {
    bar.hidden = true;
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (e) { /* not JSON: show the status code */ }
    $("#msg").textContent = xhr.status === 201 ? "" : (body.error || `error ${xhr.status}`);
    if (xhr.status === 201) { ev.target.reset(); refresh(); }
  };
  xhr.onerror = () => { bar.hidden = true; $("#msg").textContent = "no se pudo subir el vídeo"; };
  $("#msg").textContent = "subiendo…";
  xhr.send(data);
});

fetch("/api/sports").then(r => r.json()).then(list => {
  for (const s of list) { const o = el("option", s.label); o.value = s.id; $("#sport").append(o); }
});
refresh(); setInterval(refresh, 2000);
gpu(); setInterval(gpu, 5000);
</script>
</body>
</html>
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_product_server.py -v`
Expected: all PASS.

- [ ] **Step 5: Update the docs** -

In `CLAUDE.md`, "Where things are" table, replace `0005 template-conditioned model)` with
`0005 template-conditioned model, 0006 product web app)`. In the Commands block, add above the `product.video` line:

```
python -m sportcal.product.server                                                  # web app: http://127.0.0.1:8000 (ADR 0006)
```

In `README.md`, after the `product.video` paragraph added in Task 4, add:

```
The same from the browser - upload a video, pick the sport, follow the job, watch and download the result:
`python -m sportcal.product.server`, then open http://127.0.0.1:8000 (local only; ADR 0006).
```

In `docs/architecture.md`, layer list, replace

```
  product/   the deployed pipeline: Estimator chain, the sport-agnostic estimators, per-sport compositions
```

with

```
  product/   the deployed pipeline: Estimator chain, the sport-agnostic estimators, per-sport compositions,
             the video worker and the local web app (ADR 0006)
```

and in "Running", after the `pytest -m ""` line, add:

```
venv/Scripts/python.exe -m sportcal.product.server                   # the product: http://127.0.0.1:8000
```

- [ ] **Step 6: End-to-end check by hand (CPU; the GPU belongs to the training)** - start the server in the
  background:

Run: `venv/Scripts/python.exe -m sportcal.product.server`
Expected: uvicorn prints `Uvicorn running on http://127.0.0.1:8000`.

Then, in the browser at http://127.0.0.1:8000, check each point and note what you see:
1. The GPU line reads "GPU: N MiB en uso — ¿hay un entrenamiento?" in red while the training runs.
2. Upload `nhl11.mp4`, Hockey (NHL), Inicio `1:00`, Fin `1:10`, CPU, Procesar: the upload bar moves, then a card
   "en marcha" with frames / total, frames/s and "quedan ...".
3. Close the tab, open the page again: the job is still running.
4. When it ends ("terminado"): the video plays in the card and seeking works; boxes with ids, yellow field lines on the
   ice, the minimap bottom-left with coloured dots; `tracks.csv` downloads and has rows.
5. Upload `soccer.mp4`, Fútbol, `0:30` to `0:40`, CPU: a green minimap with dots on both halves of the pitch (the left
   half is the Task 1 fix).
6. Start a third job and press Cancelar after a few frames: "cancelado", and the partial video plays.
7. Borrar a finished job, confirm: the card disappears and its folder is gone from `runs/product/`.
8. Stop the server (Ctrl+C in its terminal) while no job runs.

Note the `speed` of job 2 (frames/s, from its card or `runs/product/<id>/status.json`) for Step 7. If any point fails,
stop and report it with what you saw instead of patching around it.

- [ ] **Step 7: Record the outcome in ADR 0006** - in `docs/decisions/0006-product-web-app.md`, replace the status line

```
Status: proposed, 2026-09-27. Deciders: the owner. Design agreed in conversation the same day; the implementation plan
goes to `docs/superpowers/plans/2026-09-27-product-web-app.md`.
```

with

```
Status: accepted 2026-09-27, implemented. Deciders: the owner. Implementation plan:
`docs/superpowers/plans/2026-09-27-product-web-app.md`.
```

and replace the "Speed" bullet of "Honest status" with the measured number from Step 6 (write the frames/s you noted
where the angle brackets are; this records a measurement):

```
* Speed, measured end to end (detector + homography model + drawing + encoding), CPU, `nhl11.mp4` 1:00-1:10:
  <frames/s from Step 6> frames/s. GPU not timed yet (a training held it). A 2-hour game is 216,000 frames.
```

- [ ] **Step 8: Run everything**

Run: `venv/Scripts/python.exe -m pytest -m "" -q`
Expected: all PASS (slow tests included; they use the CPU).

- [ ] **Step 9: Commit**

```bash
git add sportcal/product/web/index.html tests/test_product_server.py CLAUDE.md README.md docs/architecture.md docs/decisions/0006-product-web-app.md
git commit -m "product: the web page (upload, sport, progress, player, cancel, delete); ADR 0006 accepted

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
