# sportcal

Camera calibration (planar homography) for sports broadcast video: find where the field is in each frame, so
player tracking can be projected onto a top-down minimap.

The repository is organised as **per-sport labs** (experiments that evaluate approaches: classical CV, semantic
segmentation, keypoint models) and **one product pipeline** that composes the approaches that won.

```
sportcal/
  core/      sport-agnostic geometry, pinhole camera, robust fitting, surface segmentation, label I/O
  sports/    template geometry per sport (hockey NHL/IIHF, soccer FIFA)
  lab/       experiments per sport (`python -m sportcal.lab.<sport>.<name>`), lab/common for shared apps, archive/ for dead ends
  product/   deployable pipeline: chain of estimators with fallback
tests/       fast, deterministic tests (no GPU / data)
docs/        architecture, decisions (ADRs), experiment results
configs/     training configs
```

## Quick start

```bash
uv pip install --python venv/Scripts/python.exe -e ".[dev]"      # add ,train / ,lab for models / streamlit apps
venv/Scripts/python.exe -m pytest                                # 72 fast tests in ~8 s (pytest -m "" for all 75)
```

```python
from sportcal import sports
from sportcal.product.hockey import build_pipeline
from sportcal.product.pipeline import project_to_field

pipe = build_pipeline("runs/hockeyrink/yolo26m-17/weights/best_homography.pt",
                      seg_weights=None, hold_frames=15)          # seg_weights enables the DLT stage
est = pipe(frame)                                                # Estimate(H, confidence, method) or None
if est:
    xy_m, inside = project_to_field(est, feet_px, sports.get("hockey-nhl"))
```

## Where to read next

* [docs/architecture.md](docs/architecture.md) - layers, dependency rules, how to add a sport / experiment / product method
* [docs/experiments/hockey.md](docs/experiments/hockey.md), [docs/experiments/soccer.md](docs/experiments/soccer.md) - what was measured, what worked, what was discarded (with numbers)
* [docs/decisions/](docs/decisions/) - ADRs, including the current (provisional) product composition
* [docs/porting-status.md](docs/porting-status.md) - restructure ledger and old -> new path map

## Status

Nothing reaches the ~8 px accuracy a metric minimap needs. Best full-coverage model: YOLO keypoints, 102 px median at 88%
coverage on the hand-labelled validation set. The classical segmentation + DLT path is precise (15-56 px) but answers on
only a few percent of frames. Details and the open questions: `docs/experiments/hockey.md`, sections 10 and 13.
