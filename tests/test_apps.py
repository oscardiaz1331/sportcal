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


@pytest.mark.slow
@pytest.mark.skipif(not any(ROOT.glob("soccer*.mp4")), reason="no soccer clip on disk")
def test_annotate_app_soccer_motion_zone_and_every_line_method_render():
    at = AppTest.from_file(str(COMMON / "annotate_val_app.py"), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("futbol").run()
    assert not at.exception
    next(c for c in at.checkbox if c.key == "ver_klt").check()
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    for model in next(r for r in at.radio if r.key == "k_model").options:
        next(r for r in at.radio if r.key == "k_model").set_value(model)
        at.run()
        assert not at.exception, (model, [str(e.value)[:200] for e in at.exception])
    next(c for c in at.checkbox if c.key == "ver_grad").check()
    at.run()
    for method in next(r for r in at.radio if r.key == "g_metodo").options:
        next(r for r in at.radio if r.key == "g_metodo").set_value(method)
        at.run()
        assert not at.exception, (method, [str(e.value)[:200] for e in at.exception])


@pytest.mark.slow
@pytest.mark.skipif(not any(ROOT.glob("soccer*.mp4")), reason="no soccer clip on disk")
def test_annotate_app_soccer_ellipse_tool_fixed_two_clicks_and_klt_player():
    import numpy as np
    from sportcal.core import circle as CI
    from sportcal.core.camera import pose_to_H
    at = AppTest.from_file(str(COMMON / "annotate_val_app.py"), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("futbol").run()
    assert not at.exception
    # ellipse tool on, with an outline and two point clicks made from a synthetic pose (1920x1080 frames)
    H = pose_to_H(np.array([[-3.0, -68.0, 13.5, np.radians(-10), np.radians(12), 1.9 * 1920]]), 1920, 1080)[0]
    H = H / H[2, 2]
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    at.session_state["ell"] = [tuple(p) for p in CI._project(H, np.c_[9.15 * np.cos(ang), 9.15 * np.sin(ang)])[0]]
    at.session_state["pts"] = {6: tuple(CI._project(H, np.array([[0.0, 0.0]]))[0][0]), 7: tuple(CI._project(H, np.array([[0.0, -9.15]]))[0][0])}
    next(r for r in at.radio if r.key == "herr").set_value("Contorno del círculo central (elipse)")
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    assert any("puntos" in str(s.value) and "residuo" in str(s.value) for s in list(at.success) + list(at.warning)), "the two clicks + ellipse should give a fit"
    # the KLT zone: every region and the step-by-step player without a template
    next(c for c in at.checkbox if c.key == "ver_klt").check()
    at.run()
    for region in next(r for r in at.radio if r.key == "k_region").options:
        next(r for r in at.radio if r.key == "k_region").set_value(region)
        at.run()
        assert not at.exception, (region, [str(e.value)[:200] for e in at.exception])
    next(r for r in at.radio if r.key == "k_region").set_value(next(r for r in at.radio if r.key == "k_region").options[0])
    next(s for s in at.slider if s.key == "kv_n").set_value(5)
    next(c for c in at.checkbox if c.key == "kv_ref").uncheck()
    next(b for b in at.button if b.key == "kv_go").click()
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    assert next(s for s in at.slider if s.key == "kv_pos").max == 5
