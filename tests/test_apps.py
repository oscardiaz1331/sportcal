"""Render the Streamlit apps headlessly (`AppTest`, no server) and drive their sport / method switches.

Unit tests cannot see a renamed key or function that only a UI branch calls (a renamed model-dict key once
broke the "robust Gaussian" method while every other test stayed green). Needs streamlit and at least one
video/image source on disk, otherwise the classical lab has nothing to draw and the test skips.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from sportcal.paths import ROOT  # noqa: E402

COMMON = Path(__file__).resolve().parent.parent / "sportcal" / "lab" / "common"
has_source = any(ROOT.glob("*.mp4")) or any((COMMON / "samples").glob("*.*"))


def _radio(at, label):
    return next(r for r in at.radio if r.label == label)


@pytest.mark.slow
@pytest.mark.skipif(not has_source, reason="no video/image source on disk for the lab to load")
def test_classical_lab_every_surface_and_method_renders_without_exceptions():
    at = AppTest.from_file(str(COMMON / "classical_cv_lab.py"), default_timeout=180)
    at.run()
    assert not at.exception
    surfaces, methods = _radio(at, "Superficie").options, _radio(at, "Metodo").options
    for s in surfaces:
        for m in methods:
            _radio(at, "Superficie").set_value(s)
            _radio(at, "Metodo").set_value(m)
            at.run()
            assert not at.exception, (s, m, [str(e.value)[:200] for e in at.exception])


@pytest.mark.slow
def test_annotate_app_renders_for_every_sport():
    at = AppTest.from_file(str(COMMON / "annotate_val_app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    for sport in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(sport).run()
        assert not at.exception, (sport, [str(e.value)[:200] for e in at.exception])
