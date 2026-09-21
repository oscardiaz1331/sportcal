# Archive: frozen experiments

Unmaintained. Kept in the tree so nothing needs a git checkout to be read, **not** because it should be used. The
tag `pre-restructure` holds the exact previous layout. Imports were mechanically updated to the package layout, but
these files are not tested and some reference names that no longer exist (e.g. `rink.RINK_NHL_FITTED`).

Results and the reasons each one was abandoned: [docs/experiments/hockey.md](../../../../docs/experiments/hockey.md).

| File | What it was | Why frozen | Write-up |
|---|---|---|---|
| `klt_propagate.py` | KLT tracking of H between frames | p50 158 px; drifts onto ice reflections | section 3 |
| `homography_head.py`, `cache_seg_probs.py` | Learned camera-pose head over the segmentation map (+ its cache) | ~340 px, under-fits; nearest-neighbour and refinement variants also negative | section 8 |
| `auto_label.py` | Classical auto-labelling gate anchored on YOLO | The four signals did not discriminate | section 4 |
| `video_line_explorer.py` | Interactive video viewer of the classical pipeline | Superseded by `lab/hockey/line_explorer.py` | section 3 |
| `cv_clas_tracker.py` | Early no-deep-learning homography tracker | Superseded | section 3 |
| `test_colors.py`, `test_lineas.py`, `test_rick.py`, `field_*.py`, `hsv_scan.py` | Early colour-space / HSV zone viewers | Colour-only region finding was replaced by `core/surface.py` | section 3 |
| `rink_annotate.py`, `rink_debug.py`, `rink_edges_debug.py` | Early keypoint annotation / debug viewers | Replaced by `click_labeler` and the metrics in `rink_metric` | sections 2, 10 |

If you revive one, port it: move reusable logic to `core/`, add `main()` behind a `__main__` guard, write it up.
