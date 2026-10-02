"""The web app's job handling (ADR 0006) with a stand-in worker: no GPU, no weights."""
import contextlib
import json
import subprocess
import sys
import threading
import time

import psutil
import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

from sportcal.product import server  # noqa: E402

# Stands in for `python -m sportcal.product.video`: the same options, a `running` status written like the real worker
# (its own pid and start time, atomic replace retried), then it waits for the cancel flag (or 30 s) and ends cancelled.
FAKE_WORKER = r'''
import argparse, ctypes, json, os, time, psutil
def console_window():  # True when this process sits in a console window the owner can see (and close)
    hwnd = ctypes.windll.kernel32.GetConsoleWindow() if os.name == "nt" else 0
    return bool(hwnd and ctypes.windll.user32.IsWindowVisible(hwnd))
ap = argparse.ArgumentParser()
for opt in ("--sport", "--out", "--device", "--name"):
    ap.add_argument(opt)
ap.add_argument("video")
ap.add_argument("--start", type=float)
ap.add_argument("--end", type=float)
a = ap.parse_args()
def write(state):
    s = dict(state=state, pid=os.getpid(), pid_started=psutil.Process().create_time(), sport=a.sport,
             video_name=a.name, start_s=a.start, end_s=a.end, device=a.device, frame=0, total=10,
             window=console_window())
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


@contextlib.contextmanager
def serving(tmp_path, worker_source):
    """A client of the app over an empty jobs folder, with `worker_source` (Python) as its worker."""
    worker = tmp_path / "fake_worker.py"
    worker.write_text(worker_source, encoding="utf-8")
    jobs = tmp_path / "jobs"
    app = server.make_app(jobs, [sys.executable, str(worker)])
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:  # the only host the app answers to
        c.jobs = jobs
        yield c
        for f in jobs.iterdir():  # release any stand-in still waiting
            if f.is_dir():
                (f / "cancel").touch()


@pytest.fixture
def client(tmp_path):
    with serving(tmp_path, FAKE_WORKER) as c:
        yield c


def post(client, name="clip.mp4", headers=None, **form):
    data = {"sport": "hockey-nhl", "start_s": "0", "end_s": "", "device": "cpu", **form}
    return client.post("/api/jobs", files={"video": (name, b"not really a video", "video/mp4")}, data=data,
                       headers=headers)


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
    job = wait_state(client, jid, "running")
    assert job["sport"] == "hockey-nhl"
    assert job["window"] is False                                   # no console window whose closing kills the job
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


def test_requests_from_another_site_are_refused(client):
    """A page of another site, open in the owner's browser, must not start, cancel or delete jobs here; nor may a
    name that resolves to 127.0.0.1 (DNS rebinding) read them."""
    folder = client.jobs / "20260927-120000-done"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"state": "done"}))
    evil = {"Origin": "http://evil.example"}
    assert post(client, headers=evil).status_code == 403
    assert client.delete(f"/api/jobs/{folder.name}", headers=evil).status_code == 403
    assert client.post(f"/api/jobs/{folder.name}/cancel", headers=evil).status_code == 403
    assert [f.name for f in client.jobs.iterdir()] == [folder.name]    # nothing started, nothing deleted
    assert client.get("/api/jobs", headers={"Host": "evil.example"}).status_code == 400
    assert client.delete(f"/api/jobs/{folder.name}", headers={"Origin": "http://127.0.0.1:8000"}).status_code == 200


def test_a_status_file_of_nul_bytes_lists_as_interrupted(client):
    """What a power cut left behind once: the file is there, 277 NUL bytes long."""
    folder = client.jobs / "20260927-191854-nhl11"
    folder.mkdir()
    (folder / "status.json").write_bytes(b"\0" * 277)
    assert client.get("/api/jobs").json()[0]["state"] == "interrupted"
    assert client.delete(f"/api/jobs/{folder.name}").status_code == 200


def test_a_job_whose_worker_is_starting_lists_as_starting_and_cannot_be_deleted(client):
    """Between the folder's creation and the worker's first status the job is starting, not interrupted."""
    folder = client.jobs / "20260927-120000-starting"
    folder.mkdir()
    client.app.state.starting = folder.name
    assert client.get("/api/jobs").json()[0]["state"] == "starting"
    assert client.delete(f"/api/jobs/{folder.name}").status_code == 409
    client.app.state.starting = None
    assert client.get("/api/jobs").json()[0]["state"] == "interrupted"


def test_a_job_lists_as_starting_while_its_worker_starts(tmp_path):
    """The real thing, not the state set by hand: a worker slow to start, the list read meanwhile."""
    with serving(tmp_path, "import time\ntime.sleep(1.5)\n" + FAKE_WORKER) as client:
        seen = []

        def poll():
            for _ in range(50):
                seen.extend(j["state"] for j in client.get("/api/jobs").json())
                time.sleep(0.1)

        reader = threading.Thread(target=poll)
        reader.start()
        assert post(client).status_code == 201
        reader.join()
        assert "starting" in seen and "running" in seen and "interrupted" not in seen


def test_a_worker_that_dies_at_once_is_an_error_with_its_last_words(tmp_path):
    with serving(tmp_path, 'print("faltan los pesos: señal")\nraise SystemExit(3)\n') as client:
        r = post(client)
        assert r.status_code == 500 and "señal" in r.json()["error"]        # its log, accents intact
        assert client.get("/api/jobs").json()[0]["state"] == "interrupted"
        assert post(client).status_code == 500                              # the lock was released: it tries again


def test_a_worker_that_never_reports_is_stopped_not_left_running(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "START_TIMEOUT_S", 2)
    with serving(tmp_path, "import time\ntime.sleep(60)\n") as client:
        assert post(client).status_code == 500
        time.sleep(0.5)
        script = str(tmp_path / "fake_worker.py")
        left = [p.pid for p in psutil.process_iter(["cmdline"]) if script in (p.info["cmdline"] or [])]
        assert left == []                                                   # launcher and interpreter both gone


def test_leftovers_of_a_delete_are_swept_at_start(tmp_path):
    jobs = tmp_path / "jobs"
    (jobs / ".deleting-20260927-120000-old").mkdir(parents=True)
    (jobs / ".deleting-20260927-120000-old" / "input.mp4").write_bytes(b"x")
    (jobs / "20260927-130000-kept").mkdir()
    server.make_app(jobs)
    assert [f.name for f in jobs.iterdir()] == ["20260927-130000-kept"]


def test_the_server_does_not_load_torch():
    """Models load in the worker only: the server stays light and never holds GPU memory."""
    code = "import sys, sportcal.product.server; assert 'torch' not in sys.modules, 'the server imported torch'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and 'id="form"' in r.text and "/api/jobs" in r.text
