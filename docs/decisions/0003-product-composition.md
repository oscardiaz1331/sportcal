# ADR 0003 - Product composition (provisional)

Status: provisional - no method meets the accuracy target yet

## Decision

The hockey product is `product.hockey.build_pipeline`: an ordered chain, first answer wins.

| Order | Estimator | Why here | Measured |
|---|---|---|---|
| 1 (optional, needs seg weights) | `SegDltEstimator` | Precise when it answers, refuses otherwise | 15.5 px (IIHF) / 56 px (NHL) median on curated val, but coverage only 5-14%; ~0% on random frames |
| 2 | `YoloKeypointEstimator` | Only method with near-full coverage | 88% coverage, 102 px median, 10.5% of all frames < 25 px on the hand-labelled val (`valh`) |

Optional `hold_frames` reuses the last H when every stage refuses (camera nearly static within a
shot); call `pipeline.reset()` at shot cuts.

## Honest status

The target for a reliable minimap is ~8 px. **Nothing reaches it.** The composition exists so that
an improvement in any stage lands in the product without touching the others, not because the
current numbers are good enough. Use the product for coarse positioning only; do not rely on it for
metric distances.

## What would change this

* A segmentation model whose DLT coverage on random NHL frames is well above 4% (see the open
  questions in `experiments/hockey.md`, section 13).
* A verified way to reject wrong-geometry frames (ADR 0002).
* A learned solver that beats YOLO's 102 px at full coverage (the direct pose head reached ~340 px).

## Verified on real weights (2026-09-21)

Run on the 76 hand-labelled `valh` frames with `yolo26m-17/best_homography.pt` and `runs/lineas_seg/best.pt`:

| Check | Result |
|---|---|
| Lab metric (`rink_metric.evaluate`, batches of file paths) | 88.2% coverage, p50 102.2, p90 3195, 10.5% < 25 px - **exactly the documented numbers** |
| Product `YoloKeypointEstimator` (one ndarray per frame) | 85.5% coverage, p50 107.5, p90 3210, **2.6% < 25 px** |
| Full chain `build_pipeline(yolo, seg)`, first 40 frames | 0.34 s/frame with inference; YOLO answered 31, seg-DLT 0, none 9; YOLO p50 90.9 px, max 25,625 px |

**The two YOLO rows disagree, and the wrapper is not the cause.** Raw keypoints from batch-of-paths and single-array inference
differ by at most 0.1 px (GPU batch numerics). Pushed through the *same* RANSAC fit, that changes the homography of 15 of 65
frames by more than 5 px, of 7 by more than 50 px (worst 2,319 px), and flips the coverage of 2 frames. The homography from ~20
keypoints with an 8 px RANSAC threshold is numerically chaotic on a fifth of the frames.

Consequences:

* The product's YOLO stage is only reproducible to within this noise; the "10.5% < 25 px" of section 2 of the hockey write-up
  is one draw from a wide distribution (2.6% - 10.5% observed for the SAME model and frames).
* Differences between YOLO checkpoints that are a handful of frames wide (e.g. yolo26m-17 vs -18 on < 25 px) cannot be
  trusted without repeating the evaluation under both input paths.
* Candidate fix, **not tried**: a more stable estimator than plain `cv2.RANSAC` (`cv2.USAC_MAGSAC`), or a refinement over the
  inliers. It would change every documented YOLO number, so it needs its own experiment.
* The seg-DLT stage answered on 0 of 40 frames, consistent with the ~0-4% coverage measured on random frames; in this
  composition it costs inference time and adds nothing yet.
