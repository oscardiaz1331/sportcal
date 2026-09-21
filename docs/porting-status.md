# Porting status

Snapshot of the restructure ([ADR 0001](decisions/0001-layered-architecture.md)). Update it when you finish a
follow-up pass. The previous layout is tagged `pre-restructure` (`git checkout pre-restructure`).

## Done

| Area | State |
|---|---|
| `sportcal/core/` (`geometry`, `camera`, `fitting`, `surface`, `labels`) | ported, English, covered by tests |
| `sportcal/sports/` (registry, `Sport`, hockey NHL + IIHF, soccer FIFA) | ported, English, covered by tests |
| `sportcal/lab/soccer/field_solver.py` | ported and translated; verified **bit-identical** to the original (`search_lines`, `search_pose`) and identical evaluation tables for the same seed |
| `sportcal/product/` (`pipeline`, `hockey` estimators) | new; pipeline logic tested with fakes; both model-backed estimators and the chain **verified on real weights** (ADR 0003) |
| `tests/` | 72 fast tests + 3 opt-in slow ones (`pytest -m ""`): layering, golden segmentation->H solver (hockey) and field solver (soccer), headless drives of both Streamlit apps; mutation-checked |
| `docs/` | architecture, ADR 0001-0003, English experiment records for hockey **and soccer**, Spanish hockey original archived verbatim |
| `lab/hockey/` live scripts | **moved and import-safe, comments still Spanish** (see below) |
| `lab/soccer/{evaluation,gradient,labeler}.py`, `lab/common/*` | **moved, imports and renamed APIs updated, comments/UI still Spanish** |
| `lab/hockey/archive/` | moved, frozen, untested |
| compat shims (`rink.py`, `training/{ice_lines_probe,robust_fit,click_labeler}.py`) | **deleted** - nothing needs them any more |

## Not done yet (follow-up passes, in suggested order)

1. **Translate the moved lab files.** Hockey live lab ~4,470 lines / 20 files (largest: `train_lines_seg` 419,
   `seg_to_homography` 382, `gen_synthetic_lines` 368, `ice_lines_probe` 344, `fit_homography_lines` 314); soccer
   `gradient` (~720), `labeler` (~430), `evaluation` (~240); `lab/common` apps (~1,300, their **UI labels are Spanish on
   purpose** - translate comments, decide separately about the UI). Rule: docstrings say what and why, results go to
   `docs/experiments/`, keep identifiers stable unless every caller is updated in the same change.
2. **Promote model + solver out of the lab** so `product/` stops importing `lab/` (ADR 0001, "Known debt"):
   `UNetResNet34` and `load_model/predict_probs` -> `sportcal/models/`, `solve_from_probs` and its helpers -> `core/`.
3. **Split `product/hockey_demo.py`** (440 lines of module-level script: team classification, puck trail, minimap video)
   into a `main()` built on `HomographyPipeline`. A soccer estimator (`FieldSolver` behind an `Estimator`) is the natural
   second product method once soccer has a validation set.
4. ~~Run the model-backed estimators against real weights~~ **done 2026-09-21** (ADR 0003, "Verified on real weights"): both estimators
   and the chain work; the YOLO stage is numerically noisy (RANSAC). Still open: a `@pytest.mark.slow` test that needs the weights on disk,
   and trying `cv2.USAC_MAGSAC` as a more stable fit.
5. Delete the temporary copy `training/hockeyrink_pose_rp.yaml` once you decide not to `resume` run `yolo26m-18`
   (its `args.yaml` points there).
6. Line endings: add a `.gitattributes` (`* text=auto eol=lf`) to silence the LF/CRLF warnings on Windows.

## Old path -> new path

| Old | New |
|---|---|
| `rink.py` | `sportcal/sports/hockey/rink.py` (`RINK_NHL_FITTED` removed; `rink_polylines`, `CLASSES`, `seg`, `arc` moved here from `make_line_masks`) |
| `training/robust_fit.py` | `sportcal/core/fitting.py` (`BORDE_MARGEN_PX` -> `BORDER_MARGIN_PX`; gains `detect_lines`) |
| DLT / refinement in `training/seg_to_homography.py` | `sportcal/core/geometry.py` |
| `make_line_masks.project_points`, `relabel_reproject.in_front`, `fit_homography_lines.geom_error`, `rink_metric.fit_homography` | `sportcal/core/geometry.py` (`project_points`, `in_front`, `geom_error`, `fit_homography_ransac`); the old modules re-export them |
| `relabel_reproject.read_label/write_label/NKPT`, `auto_label.label_from_H` | `sportcal/core/labels.py` |
| Surface functions of `training/ice_lines_probe.py` | `sportcal/core/surface.py`: `ice_gmm` -> `gmm_surface`, `ice_robust` -> `robust_surface`, `rink_region` -> `play_region`, `cesped_hsv` -> `grass_hsv`, `superficie_hsv` -> `hsv_threshold`, `ROBUST_SPACES` -> `COLOR_SPACES`, `CESPED_*` -> `GRASS_*`, `HSV_*_MAX/MIN` -> `ICE_*`; `deporte/espacio/canales` -> `surface/space/channels`; the fitted-model dict key `espacio` -> `space` |
| Hockey-specific rest of `ice_lines_probe.py` (true region, localisation z) | `sportcal/lab/hockey/ice_lines_probe.py` |
| `testing/soccer_field.py` template part (`polilineas`, `muestrea`, `ESQUINAS_MUNDO`, `X_LINEAS`..., `canonicaliza_H`, `HX/HY`) | `sportcal/sports/soccer/field.py` (`polylines`, `sample_template`, `CORNERS`, `X_LINES`..., `canonicalize_H`, `HALF_LENGTH/HALF_WIDTH`; Spanish polyline names became English) |
| `testing/soccer_labeler.py` `KEYPOINTS` (31 named points) | `sportcal/sports/soccer/field.py` (`KEYPOINTS`, `KEYPOINT_NAMES`, `KEYPOINT_COORDS`) |
| `testing/soccer_field.py` camera math (`pose_a_H`, `proyecta`, `_bilineal`, `residuo_pinhole`, `consistente_pinhole`) | `sportcal/core/camera.py` (`pose_to_H`, `project`, `bilinear`, `pinhole_residual`, `is_pinhole_consistent`) |
| `testing/soccer_field.py` `detecta_lineas` | `sportcal/core/fitting.py::detect_lines` |
| `testing/soccer_field.py` solver (`Campo`, `hipotesis_lineas`, `error_reproy`) | `sportcal/lab/soccer/field_solver.py` (`FieldSolver`, `line_hypotheses`, `reprojection_error`; methods `puntua/busca_pose/busca_lineas/busca_elipse/pon_evidencia/snap_esquinas/refina` -> `score/search_pose/search_lines/search_ellipse/set_evidence/snap_corners/refine`) |
| `testing/soccer_eval.py`, `soccer_grad.py`, `soccer_labeler.py` | `sportcal/lab/soccer/{evaluation,gradient,labeler}.py` |
| `testing/classical_cv_lab.py`, `color_spaces.py`, `samples/` | `sportcal/lab/common/` |
| `testing/annotate_val_app.py` | `sportcal/lab/common/annotate_val_app.py` |
| `training/*.py` (live) | `sportcal/lab/hockey/*.py` (same file names) |
| `fetch_clips.py`, `split_val.py`, `rink_reference.py` | `sportcal/lab/hockey/` (`fetch_clips`, `split_val` now have `main()`) |
| `training/{klt_propagate,homography_head,cache_seg_probs,auto_label,video_line_explorer}.py` | `sportcal/lab/hockey/archive/` |
| `training/test-cv-clas.py` | `sportcal/lab/hockey/archive/cv_clas_tracker.py` |
| `test_colors.py`, `test_lineas.py`, `test_rick.py`, `rink_annotate.py`, `rink_debug.py`, `rink_edges_debug.py`, `field_*.py`, `hsv_scan.py`, `hsv_scan.csv` | `sportcal/lab/hockey/archive/` |
| `test.py` | `sportcal/product/hockey_demo.py` (script; refuses import) |
| `training/*.yaml` | `configs/hockey/*.yaml` (`hockeyrink_pose.yaml.bak` deleted) |
| `training/README.md` | `sportcal/lab/hockey/README.md` (translated) |
| `rink_keypoints_map.png` | `docs/assets/rink_keypoints_map.png` |
| `CLAUDE.md` (1088-line Spanish log) | `docs/experiments/hockey.es.md` (verbatim) + `docs/experiments/hockey.md` (English, condensed); `CLAUDE.md` is now a short rules file |
| `python training/<x>.py`, `python testing/soccer_eval.py` | `python -m sportcal.lab.hockey.<x>`, `python -m sportcal.lab.soccer.evaluation` |
| `streamlit run testing/<app>.py` | `streamlit run sportcal/lab/common/<app>.py` |
