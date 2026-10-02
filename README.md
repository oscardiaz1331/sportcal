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
venv/Scripts/python.exe -m pytest                                # fast tests, ~10 s (pytest -m "" for all)
```

```python
from sportcal import sports
from sportcal.product.hockey import build_pipeline
from sportcal.product.pipeline import project_to_field

pipe = build_pipeline("runs/hockeyrink/yolo26m-17/weights/best_homography.pt",   # None: refuse rather than a coarse answer
                      kpline_weights="runs/kpline/finetune/best_h.pt", hold_frames=15)
est = pipe(frame)                                                # Estimate(H, confidence, method) or None
if est:
    xy_m, inside = project_to_field(est, feet_px, sports.get("hockey-nhl"))
```

Whole video with detection, tracking, the field lines drawn from the estimated H and a minimap, for hockey (NHL) or
soccer (writes `out.mp4`, `tracks.csv` and `status.json` into a new `runs/product/<run>/`, ADR 0006):
`python -m sportcal.product.video nhl11.mp4 --sport hockey-nhl [--start S] [--end S] [--device cpu]`.

The same from the browser - upload a video, pick the sport, follow the job, watch and download the result:
`python -m sportcal.product.server`, then open http://127.0.0.1:8000 (local only; ADR 0006).

## Where to read next

* [docs/architecture.md](docs/architecture.md) - layers, dependency rules, how to add a sport / experiment / product method
* [docs/experiments/hockey.md](docs/experiments/hockey.md), [docs/experiments/soccer.md](docs/experiments/soccer.md) - what was measured, what worked, what was discarded (with numbers)
* [docs/decisions/](docs/decisions/) - ADRs, including the current (provisional) product composition
* [docs/porting-status.md](docs/porting-status.md) - restructure ledger and old -> new path map

## Status

Best model: keypoint + line heatmaps (model A2, `lab/common/train_kpline.py`) with a camera plausibility gate, 7.2 px
median at 97% coverage on 4 NHL arenas nothing was tuned on - the first to reach the ~8 px a metric minimap needs. YOLO
keypoints (102 px median) remain as an optional coverage fallback. Open: outdoor games and wrong homographies a real
camera could produce. Details: `docs/decisions/0003-product-composition.md`, `docs/experiments/hockey.md` sections 14g-14h.
