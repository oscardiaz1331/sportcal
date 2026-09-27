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
