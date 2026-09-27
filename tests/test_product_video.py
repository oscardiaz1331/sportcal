"""The worker of the product (ADR 0006): building blocks and failure paths without GPU or weights; one slow test on a
real clip."""
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
