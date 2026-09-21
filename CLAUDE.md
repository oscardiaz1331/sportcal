# sportcal - project rules for Claude

Camera calibration (planar homography) from broadcast sports video, to project player tracking
onto a minimap. Per-sport **labs** evaluate approaches; one **product** pipeline composes the
winners. Start with [docs/architecture.md](docs/architecture.md).

## Where things are

| Need | Look at |
|---|---|
| Layers, dependency rules, how to add a sport / experiment / product method | `docs/architecture.md` |
| Why things are the way they are | `docs/decisions/` (ADRs) |
| What was measured, with numbers, and what was discarded | `docs/experiments/hockey.md` (Spanish long-form: `hockey.es.md`), `docs/experiments/soccer.md` |
| Old path -> new path, what is still untranslated / unported | `docs/porting-status.md` |
| Code | `sportcal/{core,sports,lab,product}`; tests in `tests/` |

## Rules

* **Check the number before recommending.** Most of what matters was measured, and several
  reasonable ideas measured badly (all listed in `docs/experiments/hockey.md`, section 3 and 8).
  Do not re-propose a documented dead end without new evidence.
* Results, decisions and reasoning go in markdown (`docs/`), code holds mechanisms only. Code, comments
  and docstrings in English; docstrings never quote experiment numbers.
* Respect the import layers (`tests/test_layering.py` enforces them). New sport-agnostic code goes in
  `core/`, never in a lab.
* A lab module must be import-safe: `main()` behind `if __name__ == "__main__"`. Never "smoke-import"
  lab scripts without checking for that guard first (an unguarded `split_val` once re-split a dataset).
* Run `venv/Scripts/python.exe -m pytest` after touching `core/`, `sports/` or `product/`
  (72 fast tests, ~8 s, no GPU). Before committing run everything: `pytest -m ""` (75, ~45 s: golden solver runs and headless
  drives of the Streamlit apps). When adding a test for a numeric property, break the code once to see it fail.
* Data (`datasets/`, `runs/`, videos) is gitignored and lives beside the repo; locations come from `sportcal/paths.py`.
* GPU is 8 GB: never load a second model (`rink_metric`, auto-labellers, the product) while a training runs -
  check `nvidia-smi`.
* UI code (the two Streamlit apps) is only checked by `tests/test_apps.py`: after renaming anything in `core/` that an app
  reads, run that test. A renamed model-dict key once broke an app method while every unit test stayed green.
* opencv: keep **`opencv-contrib-python`** only. Installing plain `opencv-python` next to it overwrites `cv2.pyd` and
  removes `cv2.ximgproc` (docs/experiments/soccer.md section 7).

## Facts that are easy to get wrong

* Best YOLO weights: `runs/hockeyrink/yolo26m-17/weights/best_homography.pt` - never `last.pt`/`best.pt`.
  `rink_metric.sweep()` writes it. Pose mAP is not the metric; reprojection error is.
* Reference validation set: `hockeyrink_nhl_valh` (76 hand-labelled frames, independent of the DLT). Do not compare
  models across different val sets.
* Two hockey templates: `hockey-nhl` (60.96x25.91 m) and `hockey-iihf` (60x30 m). The keypoint index lands on different
  world coordinates; `RINK_NHL_FITTED` no longer exists.
* Line-class indices (0-11) and keypoint indices (0-55) are unrelated numbering schemes.
* The cost gate of `solve_from_probs` does not detect wrong-geometry videos (ADR 0002): sample visually per source
  video before merging auto-labelled data.
* YOLO homography numbers are noisy: the same weights and frames gave 10.5% or 2.6% under 25 px depending on whether images were
  predicted in batches or one by one (RANSAC on ~20 keypoints; ADR 0003). Do not read a few-frame difference between checkpoints as signal.
* Current standing (target is ~8 px): YOLO 102 px median at 88% coverage on `valh`; segmentation-DLT 15-56 px but
  ~0-4% coverage on random frames (ADR 0003).

## Commands

```bash
venv/Scripts/python.exe -m pytest                                        # fast suite
venv/Scripts/python.exe -m sportcal.lab.hockey.seg_to_homography --selftest
venv/Scripts/python.exe -m sportcal.lab.hockey.rink_metric <weights.pt>  # YOLO reprojection metric (needs GPU)
venv/Scripts/python.exe -m streamlit run sportcal/lab/common/annotate_val_app.py   # hand-label hockey / soccer frames
venv/Scripts/python.exe -m streamlit run sportcal/lab/common/classical_cv_lab.py   # classical-CV lab, any sport
venv/Scripts/python.exe -m sportcal.lab.soccer.evaluation --sintetico --n 10       # soccer solver on synthetic poses
```
