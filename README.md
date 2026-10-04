# sportcal

Camera calibration (planar homography) for sports broadcast video: find where the field is in each frame, so
player tracking can be projected onto a top-down minimap.

The repository is organised as **per-sport labs** (experiments that evaluate approaches: classical CV, semantic
segmentation, keypoint models) and **one product pipeline** that composes the approaches that won.

```
sportcal/
  core/      sport-agnostic geometry, pinhole camera, robust fitting, surface segmentation, label I/O
  sports/    template geometry per sport (hockey NHL/IIHF, soccer FIFA, tennis ITF, basketball FIBA)
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

Whole video with detection, tracking, the field lines drawn from the estimated H and a minimap, for hockey (NHL),
soccer or tennis (writes `out.mp4`, `tracks.csv` and `status.json` into a new `runs/product/<run>/`, ADR 0006):
`python -m sportcal.product.video nhl11.mp4 --sport hockey-nhl [--start S] [--end S] [--device cpu]`.

The same from the browser - upload a video, pick the sport, follow the job, watch and download the result:
`python -m sportcal.product.server`, then open http://127.0.0.1:8000 (local only; ADR 0006).

## Where to read next

* [docs/architecture.md](docs/architecture.md) - layers, dependency rules, how to add a sport / experiment / product method
* [docs/experiments/](docs/experiments/): [hockey.md](docs/experiments/hockey.md), [soccer.md](docs/experiments/soccer.md), [tennis.md](docs/experiments/tennis.md), [basketball.md](docs/experiments/basketball.md) - what was measured, what worked, what was discarded (with numbers)
* [docs/decisions/](docs/decisions/) - ADRs, including the current (provisional) product composition
* [docs/porting-status.md](docs/porting-status.md) - restructure ledger and old -> new path map

## Status

The same model - keypoint + line heatmaps (`lab/common/train_kpline.py`) with a camera plausibility gate and one shared
solver - runs every sport (ADR 0004); each sport has its own weights. On frames nothing was tuned on (target ~8 px):

| Sport | Result | Details |
|---|---|---|
| Hockey (NHL) | 7.2 px median at 97% coverage on 4 arenas; YOLO keypoints (102 px) stay as an optional fallback | ADR 0003, hockey.md 14g-14h |
| Soccer | 7.7 px at 100% coverage on our hand labels, centre-circle views included (PnLCalib 10.3 px); SoccerNet test 6.5 px | soccer.md 19-25 |
| Tennis | 1.7 px on the test split (labels from a court detector); ten labelled frames are enough | tennis.md 2-3 |
| Basketball (DeepSportRadar) | 4.6 px, 90% under 10 px on three held-out arenas with 550 labels; about 200 labels for full coverage | basketball.md 3 |

Open: outdoor games and wrong homographies a real camera could produce; the template-conditioned model for a sport with
no labels (ADR 0005, not built); the PC resets under combined load (soccer.md 24).
