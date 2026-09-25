# ADR 0003 - Product composition

Status: accepted 2026-09-25. Replaces the provisional composition of 2026-09-21 (segmentation-DLT, then YOLO), under which
no method reached the accuracy target.

## Decision

The hockey product is `product.hockey.build_pipeline`: an ordered chain, first answer wins. A stage whose weights are
`None` is left out.

| Order | Estimator (argument) | Why here | Measured |
|---|---|---|---|
| 1 | `KplineEstimator` (`kpline_weights`: model A2, `runs/kpline/finetune/best_h.pt`) | Precise at near-full coverage; refuses the homographies no camera gives (`core.camera.is_plausible_view`) | 4 arenas nothing was tuned on (`fresh`, 38 frames): 97% coverage, p50 7.2 px, p90 10.9, 97% < 25 px, no answer > 50 px. `test` (72 frames): 76% coverage, p50 7.2, p90 65 |
| 2 (optional) | `SegDltEstimator` (`seg_weights`) | Kept from the previous composition; it adds little behind stage 1 | 15.5 px (IIHF) / 56 px (NHL) median on curated val, answers on ~0-4% of random frames |
| 3 (optional) | `YoloKeypointEstimator` (`yolo_weights`) | Coverage for the frames stage 1 refuses | 88% coverage, 102 px median on `valh`; not measured on `fresh` |

Sources: `experiments/hockey.md` sections 14g (A2, `test`) and 14h (`fresh`, the gates); ADR 0004 for why the network
outputs heatmaps and one solver turns them into H.

Optional `hold_frames` reuses the last H when every stage refuses (camera nearly static within a shot); call
`pipeline.reset()` at shot cuts.

**On video, pass `smooth=0.1`.** Each answer is then blended into the previous H carried along the camera motion (KLT),
and a hold follows the camera instead of freezing: frame-to-frame jitter falls from 29.6 to 3.1 px (p50), with the same
or slightly better accuracy than per-frame answers (`experiments/hockey.md` section 14i). Leave it `None` for single
images.

**Which stages to use.** For metric work (distances, speeds) pass `yolo_weights=None`: a refusal, covered by
`hold_frames`, is better than a YOLO answer ~100 px off. Keep YOLO when an answer on every frame matters more than its
accuracy (coarse positioning).

## Honest status

On games nothing was tuned on, stage 1 reaches the ~8 px a metric minimap needs. Limits:

* 38 `fresh` frames with a single gross answer: the plausibility gate is confirmed on one case.
* A wrong homography that a real camera could produce still goes through. Outdoor `nhl9` in `test` is the example: with
  the gate, 30% of its frames are answered, at p50 248 px. Outdoor games are out of reach.
* End views are named by the view (`core.camera.canonical_mirror`), not by the physical end of the arena; which end a
  frame shows needs context (the tracker, or asymmetric markings).
* All numbers are NHL broadcasts. IIHF rinks (`sport="hockey-iihf"`) are untested in the product.

## What would change this

* A check that catches camera-like wrong homographies (open question 4 of `experiments/hockey.md` section 13), e.g.
  agreement with the H carried from neighbouring frames of the shot (`core.motion`).
* The template-conditioned model (ADR 0004) replacing the per-sport one, if it matches it on `fresh`.
* A measurement of YOLO on the frames stage 1 refuses: if it does no better than refusing, drop stage 3.

## Verified on real weights (2026-09-25): the kpline stage

`KplineEstimator` with the A2 weights, one ndarray per frame, CPU (4 threads), on the 38 `fresh` frames: 97% coverage,
p50 7.2, p90 10.9, max 14.0 px, 82% < 10 px; answered 37, refused 1 (nhl13 frame 10575, the frame the lab gate
refuses). The same numbers as the lab measure of section 14h. 0.85 s/frame on CPU; GPU speed not timed. The slow test
`tests/test_product.py::test_kpline_estimator_answers_a_new_arena_and_refuses_an_impossible_camera` repeats it on two of
those frames.

## Verified on real weights (2026-09-21): the YOLO and seg-DLT stages

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
