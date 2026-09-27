# Hockey lab: training the YOLO26 models

Two independent models (not one multi-head model):

| Model | Base | HF dataset | Task |
|---|---|---|---|
| Player / puck / rink-mark detection | `yolo26s` | `SimulaMet-HOST/HockeyAI` (2101 img, 7 classes) | detect |
| Rink keypoints | `yolo26m-pose` | `SimulaMet-HOST/HockeyRink` (661 img, 56 kpts) | pose |

Results and what was learned from all of this: [docs/experiments/hockey.md](../../../docs/experiments/hockey.md).
Training configs live in `configs/hockey/`.

## Flow

```bash
# 1. Download and prepare the datasets (YOLO train/val format, hardlinks)
python -m sportcal.lab.hockey.prepare_data

# 2. Train (RTX 3060 Ti 8 GB; FIXED batch, not autobatch)
python -m sportcal.lab.hockey.train_detect     # -> runs/hockeyai/yolo26s/weights/best.pt
python -m sportcal.lab.hockey.train_pose       # -> runs/hockeyrink/<run>/weights/best_homography.pt
```

`product/video.py` uses the local weights when present and falls back to the HuggingFace ones.

## Decisions

* **Detection classes** keep the order/names of the deployed model (`centriod, faceoff, goal, goalie,
  player, puck, referee`) so name lookups in the demo do not break.
* **`mosaic=0.0` in pose**: each image has a single rink filling most of the frame; mosaic adds nothing.
* **`fliplr`**: the yaml defines `flip_idx` (left/right symmetry map, identical to `rink.MIRROR`), so flipping is
  valid and effectively doubles the dataset. If `flip_idx` is removed from the yaml, set `fliplr=0.0` again.
* **Train/val split**: by MD5 hash of the file name, stable across runs. (`split_val` re-derives a block-based split
  for `hockeyrink_nhl` and moves files - read its docstring before running it.)
* **`imgsz`**: the puck is tiny at 1080p; lowering the resolution loses it.

## VRAM (8 GB): why the batch is fixed

AutoBatch (`batch=0.85`) probes batch 1, 2, 4... and fits a line. At 1024-1280 px each image costs ~3-4 GB, so only
batch 1-2 fit; the batch-4 probe OOMs (`backward = nan`), AutoBatch extrapolates from 2 points and over-picks, then OOMs
in epoch 1. The scripts therefore fix:

| | imgsz | batch |
|---|---|---|
| detection (yolo26s) | 1024 | 8 |
| pose (yolo26m-pose) | 1024 | 4 |

Also `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` against fragmentation. On OOM: halve the batch. Spare VRAM:
`imgsz=1280`, batch 4/2.

## Inference

The demo runs the rink model every `RINK_EVERY` frames and reuses the last homography in between (the camera is nearly
static within a shot), forcing a recompute after every shot cut. The reusable version of this is
`sportcal.product.pipeline.HomographyPipeline(hold_frames=...)`.
