# ADR 0006 - The product as a local web app

Status: accepted 2026-09-27, implemented and verified end to end 2026-10-02. Deciders: the owner. Implementation plan:
`docs/superpowers/plans/2026-09-27-product-web-app.md`.

## Context

The product is `product/hockey_demo.py`: a module-level script, hockey only, that cannot be imported or tested (its own
`ponytail:` comment says so). The calibration underneath is already sport-agnostic: `HomographyPipeline` and
`KplineEstimator` take any registered sport; only `product.hockey.build_pipeline` and the demo are hockey-specific.

The owner wants to upload a video, choose its sport (detected automatically later), get the homography of every frame
for that sport's minimap, detect the players with YOLO26 and see them on the minimap. Constraints:

* One user (the owner), on this PC. Nobody else connects.
* Videos range from clips of a few minutes to whole games: hours of processing, longer than a browser tab stays open.
* The 8 GB GPU is shared with training runs, which must have it alone (CLAUDE.md).
* Browsers do not play the `mp4v` that `cv2.VideoWriter` writes; PyAV (installed, with libx264) writes H.264.

## Decision

A Starlette app, bound to `127.0.0.1`, that runs each job in a **separate worker process**
(`python -m sportcal.product.video`), **one job at a time**, with all job state on disk in the job's run folder.

Rejected:

* **Streamlit** (already in the stack): it holds uploads in RAM, which is several GB for a whole game, and a job would
  live inside a page run unless paired with a worker anyway. The owner chose a plain server with its own page.
* **A worker thread inside the server**: the models would stay on the GPU between jobs, in the way of training runs,
  and a CUDA failure would take the server down with it.
* **A job queue, authentication, deployment**: one local user. See "What would change this".

## Scope of v1

| Sport | Homography model | Player detector | Roles | Ball | `imgsz` | Smoothing |
|---|---|---|---|---|---|---|
| `hockey-nhl` | A2, `runs/kpline/finetune/best_h.pt` | `runs/hockeyai/yolo26s/weights/best.pt`, fallback `SimulaMet-HOST/HockeyAI` on the Hub | player; goalie and referee in fixed colours | puck | 800 (measured in the old demo) | `hold_frames=15`, `smooth=0.1` (ADR 0003, hockey.md 14i) |
| `soccer-fifa` | E1, `runs/kpline-soccer/pretrain-derived/best_h.pt` | `yolo26m.pt` (COCO, repo root; ultralytics downloads it by name if missing) | person = player | sports ball | 1280 (not measured) | the same, **not measured on soccer** |

* Tennis joins when its model has an evaluation write-up (`tennis.md`): one row of this table.
* IIHF stays out: A2 predicts the NHL template.
* Sport auto-detection comes later, as one more value of the `sport` parameter (`auto`).
* No frame stride (process 1 frame in k), no audio in the output.

## Design

### Files

| File | What it holds |
|---|---|
| `product/video.py` | the sport table above, `process_video(video, sport, out_dir, start_s, end_s, device)`, and its CLI `python -m sportcal.product.video VIDEO --sport S [--out DIR] [--start S] [--end S] [--device D]` (without `--out`, a new run folder). Replaces `hockey_demo.py`, which is deleted. `product/hockey.py` stays: ADR 0003's full chain, of which the product uses the first stage, as the demo did |
| `product/draw.py` | the minimap and the projected field lines, for any `Sport`: from `polylines()`, `box` and `y_down` (the minimap puts +y down when `y_down`, up otherwise, so it matches the main camera). Projected lines go through `core.geometry.project_points`, which drops points behind the camera |
| `product/server.py` | the Starlette app; `python -m sportcal.product.server` serves on `127.0.0.1:8000` with uvicorn |
| `product/web/index.html` | the one page: plain HTML and JS, no framework, no build step |
| `product/pipeline.py` | fix: `project_to_field` tests `0 <= x <= length`, which assumes a corner origin and flags the whole left half of a centred field (soccer) as outside. It must use `sport.box` |
| `pyproject.toml` | extra `product = [starlette, uvicorn, python-multipart, av, trackers, scenedetect, psutil, huggingface_hub]`, all already installed and pinned in `requirements.txt`; install with `.[train,product]` |
| `CLAUDE.md`, `docs/architecture.md` | the server and CLI commands replace the `hockey_demo` one |

Nothing in `core/`, `sports/` or `models/` changes, and `train_kpline` imports nothing from `product/`: this can be
built while a training runs. The server never imports torch; the models load only in the worker.

### Per frame (worker), within `[start_s, end_s)`

1. **Shot cut** (PySceneDetect `ContentDetector` on a ~256 px frame, as in the old demo): reset the tracker, the
   homography pipeline (`reset()`) and the ball trail; new `shot_id`.
2. **Homography**: `HomographyPipeline([KplineEstimator])` with the sport's smoothing. No YOLO-keypoint fallback: a
   refusal covered by the hold is better than an answer ~100 px off (ADR 0003).
3. **Detection**: the sport's YOLO26 at `conf=0.25` and its `imgsz`, then ByteTrack (`trackers`) for the ids.
4. **Team**: `core.teams`; two teams fitted on the first 300 jerseys, kept for the whole video (not reset at cuts).
   Players without a team yet, or with too little jersey colour, are drawn as "no team".
5. **Position**: feet (bottom centre of the box) to metres with `project_to_field`; points more than 1 m outside the
   field are dropped.
6. **Ball**: a trail of the last 25 positions, each segment coloured by the team of the nearest player within 3 m.
7. **Output frame**: boxes and ids in team colours, the projected field lines, a label with `Estimate.method` (e.g.
   `kpline+klt`, `kpline+klt+hold`) or `sin calibrar`, and the minimap in the bottom-left corner, scaled to 30% of the
   frame width whatever the field size (at the old demo's 9 px/m a soccer minimap would be 945 px wide). H.264 through PyAV
   (`yuv420p`, odd frame sizes padded by 1 px).

Dropped from the old demo: the HSV "main camera" label and its `hsv_scan.csv`, the writer thread.

### Run folder

`runs/product/<YYYYMMDD-HHMMSS>-<label>/`, created with `exist_ok=False`; a folder that already has a `status.json` is
never reused (a reused run folder once overwrote a best checkpoint). `<label>` is the uploaded file name reduced to
`[A-Za-z0-9_-]`, at most 40 characters.

| File | Written by | Content |
|---|---|---|
| `input.<ext>` | server | the upload (CLI runs read their video where it is) |
| `status.json` | worker only | `state` (`running`, `done`, `failed`, `cancelled`), `pid`, `pid_started`, `sport`, `video_name`, `start_s`, `end_s`, `device`, `frame` (frames processed), `total` (frames in the range), `speed` (frames processed per second), `error`, `created` and `finished` (ISO 8601). Written first thing at start-up, then about once a second, atomically (temp file + `os.replace`) |
| `out.mp4` | worker | the annotated video, closed in a `finally` so that a failed or cancelled job keeps what it processed |
| `tracks.csv` | worker | `frame, time_s, shot_id, track_id, role, team, x_m, y_m`; `role` in `player, goalie, referee, ball`; rows only with a homography and inside the field |
| `worker.log` | worker (stdout, stderr) | progress and tracebacks |
| `cancel` | server | flag file; the worker checks it every frame and stops cleanly |

Two states are never written; the server derives them. `interrupted`: `status.json` says `running` but no process with
that `pid` and start time exists (killed, crashed, PC restarted), or the folder has no readable `status.json` (a power
cut once left one as NUL bytes). `starting`: the folder the server has just created, for the seconds until its worker
writes its first status; the server remembers its name in memory while `POST /api/jobs` waits, the page shows it as
"arrancando" with no buttons, and it cannot be deleted. Liveness uses `psutil`: on Windows `os.kill(pid, 0)` kills the
process instead of probing it.

### Server routes

| Route | Does |
|---|---|
| `GET /` | the page |
| `GET /api/sports` | the sports of the table in `video.py` (the single source) |
| `GET /api/gpu` | GPU memory in use, from `nvidia-smi` (no torch in the server); `null` without it |
| `POST /api/jobs` | multipart `video`, `sport`, `start_s`, `end_s`, `device` -> validate, create the folder, save the upload, launch the worker, wait until its first `status.json` exists, `201 {id}`. `409` while another job runs. If the worker exits, or writes no status within 60 s: `500` with the tail of `worker.log`, and the folder stays for inspection (listed as `interrupted`) |
| `GET /api/jobs` | every run folder, newest first, from its `status.json` (derived `starting` and `interrupted` included) |
| `POST /api/jobs/{id}/cancel` | create the `cancel` flag; `409` if the job is not running |
| `DELETE /api/jobs/{id}` | `409` if running; otherwise rename the folder, then delete it. On Windows the rename fails while any file inside is open (the video in the player), so the job is either deleted whole or not at all ("archivo en uso") |
| `/runs/...` | `StaticFiles` over `runs/product/`: `out.mp4` with range requests (seeking), `tracks.csv`, `worker.log` |

* **One job at a time**: an `asyncio.Lock` covers "check that nothing runs, create the folder, launch, wait for the
  first status", so two quick submissions cannot both start. `ponytail:` single uvicorn process and a global lock;
  upgrade to a queue when several videos need to wait in line.
* The worker is launched detached (Windows: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`), stdout and stderr to
  `worker.log`: closing the tab, stopping the server or closing its console does not stop a job. A restarted server
  finds the job through its `status.json`.
* The upload is spooled by Starlette to the system temp folder and then copied into the run folder: a whole game needs
  twice its size free on disk for a moment.

### Validation (a trust boundary, even locally)

* `sport` must be in the table; `device` is `cuda:0` or `cpu`; `0 <= start_s < end_s`, an empty `end_s` meaning the
  end of the video.
* Extension in `.mp4 .mov .mkv .avi .webm .m4v`, else `400`. The client file name is never used as a path, only reduced
  to the folder label.
* `{id}` must match `YYYYMMDD-HHMMSS-<label>` and name a direct child of `runs/product/`; anything else (`..`, a
  separator) is `400` and touches nothing.
* A video the worker cannot open ends as `failed` with the message; a broken or aborted upload removes the half-made
  folder.

### Page

* **Form**: file (upload progress through `XMLHttpRequest`, since `fetch` reports none), sport, start and end as
  `[[hh:]mm:]ss` (empty = whole video), GPU or CPU. Next to the device: "GPU: N MiB en uso" and, above ~1 GB, "¿hay un
  entrenamiento?".
* **Jobs**, polled every 2 s: name, sport, state, frames done / total, speed and time left, Cancel. When done: a
  `<video>` player and links to `out.mp4` and `tracks.csv`; when failed: the last lines of `worker.log`. Delete (with a
  confirmation) on any job that is not running. Polling must not recreate an existing `<video>` element, or playback
  restarts every 2 s.
* UI text in Spanish.

### Errors

| What happens | What the owner sees |
|---|---|
| Exception in the worker | `failed` with the message; traceback in `worker.log`; the partial `out.mp4` plays |
| CUDA out of memory (a training holds the GPU) | `failed`: "sin memoria en la GPU: usa CPU o espera" |
| Missing weights | `failed` before the video is opened, naming the expected path |
| Range outside the video (`start_s` past its end) | `failed`: "el tramo empieza después del final del vídeo" |
| Worker killed or crashed | `interrupted` |
| Server restarted | nothing lost: state is in the run folders |
| Upload broken, disk full | error message; the half-made folder is removed |

### Tests

Each check is broken once on purpose to see it go red.

1. `tests/test_product_server.py` (fast, no GPU): Starlette `TestClient`, `runs/product` in a temp folder, a fake
   worker (a few lines of Python that write `status.json`) instead of the real one. Covers: a valid job is `201` and a
   second one `409`; bad sport, range or device is `400` with no folder left; a file named `../../x.mp4` lands inside
   `runs/product`; delete is `409` while running, removes a finished job, and an id with `..` removes nothing; a
   `running` status with a dead pid lists as `interrupted`.
2. `tests/test_product_draw.py` (fast): the field corners land on the minimap corners with the right orientation for a
   corner-origin, `y_down` sport (hockey) and a centred, y-up one (soccer); projected lines drop the points behind the
   camera.
3. `tests/test_product.py`: `project_to_field` keeps a point in the left half of a centred field (fails before the fix).
4. A subprocess imports `sportcal.product.server` and checks that `torch` is not in `sys.modules`.
5. Slow (`@pytest.mark.slow`, real weights, CPU): 30 frames of a clip give an `out.mp4` that PyAV decodes to 30 frames,
   a `tracks.csv` with rows and state `done`; with the `cancel` flag the worker stops early, as `cancelled`, and its
   `out.mp4` still decodes.
6. By hand before calling it done: start the server, upload `nhl11.mp4` with a 30 s range, watch it in the browser.

## Honest status

* Soccer: the smoothing, `imgsz` 1280 and team colours are unmeasured there; COCO `person` does not tell referees or
  goalkeepers apart; E1 misses centre-circle views (soccer.md 20-22), which become refusals and holds.
* A wrong homography that a real camera could produce still passes the gate (ADR 0003).
* Speed, end to end (detector + homography model + drawing + encoding), 1080p: 5.7 frames/s for hockey and 6.5 for
  soccer on the GPU (one 10 s clip each, 2026-10-02); about 0.9 frames/s on CPU (2026-09-27, with a training running).
  A 2-hour game at 60 fps is 432,000 frames: about 21 hours on the GPU. A frame stride is the first thing to add for
  whole games.

## Verified end to end (2026-10-02)

Through the page in a browser, GPU: `nhl11.mp4` 1:00-1:10 (600 frames) and `soccer.mp4` 0:30-0:40 (250 frames) both
end as `done`; the video plays and seeks (range requests) and is not recreated by the polling; closing and reopening
the tab keeps the job; cancel stops a job at its 37th frame and the partial video plays; delete removes the folder.
The projected lines sit on the painted ones in both sports.

Seen on those two clips, not measured:

* Soccer: every player of the clip is in the left half (x from -48 to -8 m), the half `project_to_field` dropped before
  its fix. Team colours mix the two teams, and the ball trail jumps between false `sports ball` detections.
* Hockey: the detector calls some skaters referees; the browser's video controls cover part of the minimap while the
  video is paused.
* The first run of this check (2026-09-27, CPU job next to a GPU training) ended with the PC going down hard; the
  training died with it. Cause unknown (soccer.md 24 suspects the hardware): do not run a job next to a training.

## What would change this

* Someone else using it: authentication, a queue, perhaps a deployment. The worker and its CLI stay as they are.
* Many long videos: a queue and a frame stride.
* Sport auto-detection: a classifier behind `sport=auto`.
* The refined E1 run (`runs/kpline-soccer-fifa/pretrain-derived-refine`) beating E1 on `fresh`: one path in the table.
