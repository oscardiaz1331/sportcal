"""The worker of the product (ADR 0006): building blocks and failure paths without GPU or weights; one slow test on a
real clip."""
import csv
import json
import sys
import threading
import time
import types
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
import supervision as sv

from sportcal.paths import ROOT
from sportcal.product import video as V
from sportcal.product.pipeline import Estimate


def test_frame_range_clamps_to_the_video_and_refuses_what_is_not_in_it():
    assert V.frame_range(100, 25.0) == (0, 100)
    assert V.frame_range(100, 25.0, 1.0, 2.0) == (25, 50)
    assert V.frame_range(100, 25.0, 1.0, 60.0) == (25, 100)      # an end past the video: up to its end
    assert V.frame_range(None, 25.0, 1.0) == (25, None)           # the container does not say: read until the end
    for args in ((5.0,), (1.0, 1.0), (-2.0, 1.0)):                 # starts after the end; empty; starts before 0
        with pytest.raises(V.JobError):
            V.frame_range(100, 25.0, *args)


def test_run_folders_are_new_and_named_from_a_clean_label(tmp_path):
    a = V.new_run_folder(tmp_path, "../../Mi vídeo final.mp4")
    b = V.new_run_folder(tmp_path, "../../Mi vídeo final.mp4")
    assert a.parent == tmp_path == b.parent and a != b             # same second, same name: still two folders
    assert a.name.endswith("-Mi_v_deo_final")
    long = V.new_run_folder(tmp_path, "x" * 100 + ".mp4")
    assert all(V.ID_PATTERN.match(f.name) for f in (a, b, long))
    assert not V.ID_PATTERN.match(a.name + "\n")                   # an id is the whole string, nothing after it


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


def test_only_the_most_confident_ball_of_a_frame_is_kept():
    roles = ["player", "ball", "ball", "player", "ball"]
    assert V.one_ball(roles, np.array([0.9, 0.3, 0.8, 0.5, 0.4])).tolist() == [True, False, True, True, False]
    assert V.one_ball(["player", "player"], np.array([0.9, 0.5])).tolist() == [True, True]
    assert V.one_ball(["ball", "ball"], None).tolist() == [True, False]    # no confidences: the first one


# ---- the frame loop with stand-in models: no torch, no weights, a second or two

H_TINY = np.array([[5.0, 0, 8], [0, 5.0, 20], [0, 0, 1.0]])  # rink metres -> pixels of a 320x180 frame
H_COURT = np.array([[5.0, 0, 160], [0, 5.0, 90], [0, 0, 1.0]])  # the same for a field with its origin at the centre


def big_video(path, n=12, fps=30.0):
    """A synthetic 320x180 clip: room for the minimap inset, no shot cut."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    for i in range(n):
        w.write(np.full((180, 320, 3), 100 + 2 * i, np.uint8))
    w.release()
    return path


def box(x_m, y_m, h=30, H=H_TINY):
    """A box whose feet stand on the field point (x_m, y_m) seen through H (one of the two above)."""
    x, y = H[0, 0] * x_m + H[0, 2], H[1, 1] * y_m + H[1, 2]
    return [x - 6, y - h, x + 6, y]


class NoFrameCount:
    """cv2.VideoCapture for a container that does not say how many frames it holds (some .mkv / .webm)."""
    real = cv2.VideoCapture

    def __init__(self, *args):
        self._cap = NoFrameCount.real(*args)

    def get(self, prop):
        return 0.0 if prop == cv2.CAP_PROP_FRAME_COUNT else self._cap.get(prop)

    def __getattr__(self, name):
        return getattr(self._cap, name)


@pytest.fixture
def stand_ins(monkeypatch):
    """`process_video` with a scripted detector and a fixed homography in place of the models, for every product
    sport. The test sets `frames` - the detections of each frame, [(box, confidence, class name)] - and `H` when the
    field is centred, and reads `trackers`, the keyword arguments every ByteTrack was built with."""
    script = types.SimpleNamespace(frames=[], trackers=[], H=H_TINY)
    names = {0: "player", 1: "puck", 2: "person", 3: "sports ball"}

    class Detector:
        def __init__(self, weights):
            self.names, self.seen = names, 0

        def predict(self, source, **kwargs):
            dets = script.frames[self.seen] if self.seen < len(script.frames) else []
            self.seen += 1
            if not dets:
                return [sv.Detections.empty()]
            ids = {v: k for k, v in names.items()}
            return [sv.Detections(xyxy=np.array([d[0] for d in dets], float), confidence=np.array([d[1] for d in dets]),
                                  class_id=np.array([ids[d[2]] for d in dets]))]

    class Homography:
        name = "kpline"

        def __init__(self, weights, sport, device):
            pass

        def estimate(self, frame):
            return Estimate(script.H, 1.0, self.name)

    real_tracker = V.ByteTrackTracker
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Detector))
    monkeypatch.setattr(V.sv.Detections, "from_ultralytics", staticmethod(lambda result: result))
    monkeypatch.setattr(V, "KplineEstimator", Homography)
    monkeypatch.setattr(V, "ByteTrackTracker", lambda **kw: script.trackers.append(kw) or real_tracker(**kw))
    for sport, row in list(V.SPORTS.items()):
        monkeypatch.setitem(V.SPORTS, sport, {**row, "kpline": Path(__file__), "detector": Path(__file__)})
    return script


def test_the_loop_writes_every_frame_and_where_everyone_stands(tmp_path, stand_ins):
    two = [(box(20, 10), 0.9, "player"), (box(40, 15), 0.9, "player")]
    pucks = [(box(30, 12, 6), 0.8, "puck"), (box(10, 5, 6), 0.4, "puck")]
    stand_ins.frames = [two, [], two, two, two, two + pucks] + [two] * 6   # frame 1 is empty, frame 5 has two "pucks"
    run = tmp_path / "run"
    assert V.process_video(big_video(tmp_path / "v.mp4", 12), "hockey-nhl", run, device="cpu") == "done"
    st = status_of(run)
    assert st["frame"] == st["total"] == 12 and st["error"] is None
    with av.open(str(run / "out.mp4")) as c:
        assert sum(1 for _ in c.decode(video=0)) == 12
    rows = list(csv.DictReader(open(run / "tracks.csv", encoding="utf-8")))
    assert {r["frame"] for r in rows} == {str(i) for i in range(12)} - {"1"}   # a frame with nobody in it: no rows
    assert (float(rows[0]["x_m"]), float(rows[0]["y_m"])) == (20.0, 10.0)      # the feet, in rink metres
    assert rows[0]["track_id"] == "" and rows[-1]["track_id"] != ""            # not confirmed yet; confirmed later
    balls = [r for r in rows if r["role"] == "ball"]
    assert [(r["frame"], float(r["x_m"])) for r in balls] == [("5", 30.0)]     # the confident one of the two
    assert stand_ins.trackers and all(kw.get("frame_rate") == 30.0 for kw in stand_ins.trackers)


def test_a_video_that_does_not_say_its_length_is_read_to_its_end(tmp_path, stand_ins, monkeypatch):
    clip = big_video(tmp_path / "v.mp4", 12)
    monkeypatch.setattr(V.cv2, "VideoCapture", NoFrameCount)
    assert V.process_video(clip, "hockey-nhl", tmp_path / "run", device="cpu") == "done"
    st = status_of(tmp_path / "run")
    assert st["total"] is None and st["frame"] == 12
    # and a stretch past its end is a failure with a message, not a "done" with no video
    assert V.process_video(clip, "hockey-nhl", tmp_path / "late", start_s=5.0, device="cpu") == "failed"
    assert "tramo" in status_of(tmp_path / "late")["error"] and not (tmp_path / "late" / "out.mp4").exists()


def test_every_product_sport_is_a_registered_sport_with_a_full_row():
    from sportcal import sports

    assert set(V.SPORTS) == {"hockey-nhl", "soccer-fifa", "tennis-itf"}
    keys = set(V.SPORTS["hockey-nhl"])
    for name, row in V.SPORTS.items():
        assert sports.get(name).name == name and set(row) == keys, name


def test_tennis_keeps_a_player_standing_behind_the_baseline(tmp_path, stand_ins):
    """Tennis is played from behind the baselines: 3 m off the court is a player, not someone to drop (in hockey or
    soccer, 3 m outside the field is the bench or the crowd)."""
    stand_ins.H = H_COURT
    stand_ins.frames = [[(box(15.0, 0.0, H=H_COURT), 0.9, "person")]] * 6    # the near baseline is at x = 11.885
    clip = big_video(tmp_path / "v.mp4", 6)
    assert V.process_video(clip, "tennis-itf", tmp_path / "tennis", device="cpu") == "done"
    rows = list(csv.DictReader(open(tmp_path / "tennis" / "tracks.csv", encoding="utf-8")))
    assert len(rows) == 6 and {(r["role"], float(r["x_m"]), float(r["y_m"])) for r in rows} == {("player", 15.0, 0.0)}
    stand_ins.frames = [[(box(56.0, 0.0, H=H_COURT), 0.9, "person")]] * 6    # soccer: 3.5 m behind the goal line
    assert V.process_video(clip, "soccer-fifa", tmp_path / "soccer", device="cpu") == "done"
    assert (tmp_path / "soccer" / "tracks.csv").read_text().splitlines()[1:] == []


def test_a_cancel_flag_stops_the_loop(tmp_path, stand_ins):
    run = tmp_path / "run"
    run.mkdir()
    (run / "cancel").touch()
    assert V.process_video(big_video(tmp_path / "v.mp4"), "hockey-nhl", run, device="cpu") == "cancelled"
    assert status_of(run)["frame"] == 0 and not (run / "out.mp4").exists()


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
