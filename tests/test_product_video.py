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
