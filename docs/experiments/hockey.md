# Hockey experiments - what was measured and what was decided

Condensed, maintained English record of the hockey lab. The original long-form Spanish
research log (with the full narrative of each dead end) is preserved verbatim in
[hockey.es.md](hockey.es.md); its file paths predate the restructure - use the old→new map in
[../porting-status.md](../porting-status.md). **Before recommending anything "because it
seems reasonable", check the number here: several reasonable ideas measured badly, on purpose
documented.**

Conventions: pixel errors are normalised to a 1920 px frame width. "p50 / p90" = median / 90th
percentile over frames. Commands are `python -m sportcal.lab.hockey.<module>`.

## 0. Datasets and validation sets

| Name | What | Notes |
|---|---|---|
| `hockeyrink` | SimulaMet HockeyRink, Swedish league, IIHF 60x30 m | homogeneous source; 574 train / 61 val (lines) |
| `hockeyrink_nhl` | own NHL clips (nhl3...nhl10) | 688 train / 73 val frames; ~80% of labels are geometric reprojections, not hand-clicked - self-consistent, **not independent** |
| `hockeyrink_nhl_valx` | val 73 + 17 DLT-accepted frames = 90 | **biased**: only "easy" frames the DLT solves (circular for measuring DLT coverage); 10/17 new frames are consecutive nhl4; 6 of nhl9 were solved by the UDA model that saw unlabelled nhl9 |
| `hockeyrink_nhl_valh` | 76 frames **hand-labelled** with 4-8 clicks (`click_labeler`) | independent of the DLT, includes hard frames, label error p50 2.0 px. **The reference val from now on** |

Never compare models on different val sets, and never trust a difference smaller than the
sampling noise: with 10-25 curves per class one miss = ±8 pp.

## 0b. Per-frame H index, per-video split, one label convention - `lab/hockey/build_h_index.py` (2026-09-23)

**Question:** one label source for the keypoint + line model (every target is rendered from one H per frame plus the
template), and a test set that no training or selection frame can leak into.

**Method:** fit H to every YOLO label of `hockeyrink_nhl` (NHL template) and `hockeyrink` (SHL, IIHF template) with
the relabel gate lowered from 8 to 6 visible points (the 6-7 point frames are the sparse views; +61 NHL / +51 IIHF
frames), take the `valh` H from its `clicks.jsonl`, mirror every H into one convention (zone A on the image left,
y = 0 boards above the y = W ones, so y = W is the camera side), and split by video with **nhl4, nhl10 and nhl9 held
out**. Output: `datasets/hockey_h.jsonl`, one frame per line (`build_h_index` docstring has the fields).

**Result:**

| split | frames | videos |
|---|---|---|
| train | NHL 585 (282 auto_seg + 303 older labels) + IIHF 574 | clip, clip2, nhl3, nhl5, nhl7, nhl8 + SHL |
| dev (checkpoint selection) | NHL 62 + IIHF 61 | same videos |
| test (report here) | 53 hand-labelled | nhl4 15, nhl10 18, nhl9 20 |
| test_leaky | 23 hand-labelled | nhl3, nhl5, nhl7, nhl8 |
| excluded | 96 | the labelled frames of nhl4 and nhl10 |

Left out: 18 NHL and 26 IIHF frames with fewer than 6 visible keypoints. Fit residual p50 0.64 px, p90 2.78 px (the
labels are mostly reprojections, section 0).

**Half of the auto-labelled frames are mirrored in y.** In the index convention, 163 of the 329 segmentation-DLT
frames need a y mirror, spread inside every video (nhl5 39/76, nhl7 47/93, nhl8 55/113, nhl10 22/47), against 3 of
414 older NHL labels, 2 of 635 SHL labels and 2 of 76 hand labels (both nhl4). The camera does not change side inside
a video, so the DLT picks the y sign at random (the lo/hi faceoff classes are symmetric): in half of the frames merged
for `yolo26m-18` (section 2b) every lo/hi keypoint had the other side's name and the referee crease was on the wrong
side. The index mirrors them back; the YOLO label files are untouched. Checked by eye on nhl7 f345 (mirrored: the
template's referee crease lands on the painted one), nhl7 f5 and nhl3 f2610.

**Decision:** the index is the label source of the per-sport NHL model; train / dev / test as above. SHL frames have
no video id, so the IIHF part keeps the dataset's own split.

**Audit of the stored labels** (`audit_labels`, 2026-09-23): 40 frames (30 auto_seg, 10 older labels) sampled evenly in
time per video by `audit_labels --sample` and hand-clicked by the owner in `datasets/hockeyrink_nhl_audit`, then
compared with the H stored in the index. Two measures, px at 1920: *grid* = median over a 1 m rink grid inside the frame,
*clicks* = median distance between the clicks and where the stored H puts the clicked points (direct, no
extrapolation). The hand labels are self-consistent: median click residual 2.5 px (9 of 40 used only 4 clicks, an
exact fit with no check).

| stored labels | n | grid p50 / p90 | clicks p50 / p90 | < 10 px at the clicks | > 50 px at the clicks |
|---|---|---|---|---|---|
| auto_seg (segmentation + DLT) | 30 | 17.2 / 24.5 | 21.6 / 51.6 | 0% | 4 |
| older NHL labels (`nhl_prior`) | 10 | 15.6 / 50.8 | 10.8 / 33.0 | 40% | 1 |

By eye (nhl5 f2540, nhl7 f1630, nhl8 f2270): the hand H sits on the painted lines; the auto H drifts in the half of
the frame nearest the camera (the centre circle comes out too big towards the bottom), and nhl8 f2270 is simply wrong.

**Consequence:** no NHL training label source is within the ~8 px target. The auto_seg frames are ~15-25 px off
(besides the y mirror above) and the older labels ~11-16 px with occasional gross errors; the only NHL labels at
~2-3 px are the hand ones (76 valh + 40 audit). The auto error is per frame and structured (the whole H is off, most in
the near field), so it is not expected to average out the way independent click noise would - an expectation, not
measured.

**Decision (owner, 2026-09-23):** the 40 audit frames enter the index with their hand H (`source` hand, the label
they replace kept as `H_label` / `label_source`): 25 train, 5 dev, and 10 from nhl10 / nhl4 that join the clean test
set (now 63 frames). NHL train = 260 auto_seg + 300 older + 25 hand. Model A is trained in two phases: every train
label for pretraining, then only the hand labels of training videos (train + test_leaky, 48 frames) for fine-tuning
(section 14).

    python -m sportcal.lab.hockey.audit_labels

## 1. Template geometry (`sports/hockey/rink.py`)

Two real bugs, both fixed and covered by `tests/test_sports.py`:

* **Zone-A mirror mapping** wrong (`i+46`/`i+26` only right for a subset). Correct mapping is
  `MIRROR` (= `flip_idx` in the yaml). Model selection on real labels: 111 px reprojection error
  with the old formula, 5.6 px with `MIRROR`.
* **Neutral faceoff dots** had inverted sign (1.5 m outside the neutral zone). Fixed, matches to 3 cm.

With both fixed the whole pipeline ceiling is **1.0 px** reprojection error using ground-truth
keypoints: geometry and template are correct; every remaining failure is the model or the method.
NHL (60.96x25.91 m) and IIHF (60x30 m) are separate templates (`RINK_NHL`, `RINK_IIHF`); the old
`RINK_NHL_FITTED` was fitted on the broken mapping and was removed.

## 2. YOLO26 keypoints (56 points) - `lab/hockey/train_pose.py`

Nine runs stalled because of a chain of training bugs (all fixed in `train_pose.py` /
`configs/hockey/hockeyrink_pose.yaml`):

| Bug | Fix |
|---|---|
| `optimizer="auto"` ignores `lr0` (AdamW 0.002 fixed) | set the optimizer explicitly |
| `kpt_oks_sigmas=0.10` saturates the OKS loss (89 px error -> loss 0.038, no gradient) | 0.02 (same error -> 0.62). **Training from COCO must start at 0.10 and decrease**, else it saturates the other way |
| Box head starved once pose loss grew ~5x | `box=15.0` (val frames with no box: 13 -> 48 of 138 before) |
| Ultralytics `fitness` (box mAP) picks the wrong checkpoint (yolo26m-13: best.pt 85.2 px, last.pt 63.4; yolo26m-17: last 82.1, epoch80 61.4) | `save_period=10` + `rink_metric.sweep()` writes the winner to `best_homography.pt`. **Always use that file, never `last.pt`** |
| Early stop on box mAP (peaks at epoch 1) killed runs while pose improved | `patience=250` |

**Pose mAP is not the metric.** A model with mAP 0.50 and 0% usable homographies existed.
`lab/hockey/rink_metric.py` fits H by RANSAC on predicted keypoints and measures reprojection
error against ground truth. Take the best box **without** confidence filtering (median box conf
0.59, p25 0.03; filtering loses ~42% of frames).

Progress (median keypoint error): 88.6 → 76.1 → 63.4 → **61.4 px** (`yolo26m-17`, `epoch80`).
On the hand-labelled val: **88.2% coverage, p50 102 px, p90 3195 px, 10.5% of all frames < 25 px**.
Best full-coverage model measured; target is ~8 px. Decision: YOLO is the product's fallback
(ADR 0003), no longer the research priority.

**Reproducibility caveat (2026-09-21):** these figures are one draw of a noisy process. The same weights on the same frames give
10.5% < 25 px when predicted in batches of file paths (what `rink_metric` does) and 2.6% when predicted one image at a time,
because 0.1 px differences in the raw keypoints change the RANSAC homography of ~20% of frames by tens of px (ADR 0003).
Treat differences of a few frames between checkpoints as noise.

Relabelling by reprojection (`relabel_reproject.py`): gain is **outlier cleaning**, not filling
missing points (only ~14 of 56 keypoints are physically in frame); 2789 hand-placed points
(~27% of IIHF annotations) were RANSAC outliers and got corrected. Modest impact.

## 2b. YOLO26 retrain with the auto-labelled NHL frames - `lab/hockey/train_pose.py` (run `yolo26m-18`, interrupted)

**Question:** does fine-tuning with the 329 auto-labelled NHL frames (`hockeyrink_nhl_rp`, 688 train images) improve
homography quality over `yolo26m-17`?
**Method:** `train_pose.py` unchanged (fine-tune from `yolo26m-17/best_homography.pt`, 250 epochs planned, cosine LR).
Stopped by hand at epoch 56 (~168 min, ~180 s/epoch) to free the GPU, so the automatic end-of-run sweep never ran. Every
saved checkpoint (epoch0/10/20/30/40/50, last) evaluated with `rink_metric.evaluate` on `valh` (76 frames, same set as
the reference). The standard sweep on the old val (138 frames) was then run by hand and picked `epoch50.pt`, which is
now `runs/hockeyrink/yolo26m-18/weights/best_homography.pt` (selection independent of `valh`).
**Result:**

| model | val | coverage % | p50 px | p90 px | <25 px, % of all frames | <10 px |
|---|---|---|---|---|---|---|
| yolo26m-17 | valh (76) | 88.2 | 102.2 | 3195 | 10.5 | 0% |
| yolo26m-18 epoch50 | valh (76) | 96.1 | 65.6 | 3308 | 17.1 | 0% |
| yolo26m-18 last (ep56) | valh (76) | 92.1 | 70.0 | 2518 | 15.8 | - |
| yolo26m-17 | old val (138) | 68.1 | 61.4 | 12440 | 18.1 | 2.2% |
| yolo26m-18 epoch50 | old val (138) | 94.2 | 63.5 | 4445 | 18.8 | 1.4% |

Per-checkpoint `valh` p50 (px): ep0 82.5, ep10 125.8, ep20 87.3, ep30 98.9, ep40 91.3, ep50 65.6, ep56 70.0; coverage
90.8-97.4%; <25 px 5.3-18.4%.
**Decision:** modest gain, not a step change. Coverage rises a lot (old val 68 → 94%, valh 88 → 96%) and the tail
shrinks (old val p90 12440 → 4445 px); the median improves on `valh` (102 → ~66 px) but is unchanged on the old val
(61.4 → 63.5 px), and accuracy at the good end does not move (<10 px stays 0-2%; target is ~8 px). Adopting
`yolo26m-18` as the reference weights is left to the owner (`CLAUDE.md` still points at `yolo26m-17`).
**Caveats:** the run stopped at epoch 56/250 with the LR still high (no annealing phase); consecutive checkpoints swing
p50 by 30-60 px (ep0 82.5, ep10 125.8), so single p50 differences under ~30 px are noise on 76 frames. Part of the gain
is already there at epoch 0 (one epoch on the new data: 82.5 px, 90.8% coverage). A full run to the end of the cosine
schedule is the untested variant.

## 3. Classical CV (colour + geometry)

Explorable with `python -m sportcal.lab.hockey.line_explorer`. Generic parts now in
`core/surface.py` and `core/fitting.py`.

**Works**

| Piece | Result |
|---|---|
| Play-region segmentation: fixed HSV + Lab GMM, **chosen by solidity** (`core.surface.best_region`) | IoU 0.91-0.93, worst case 0.34-0.71. Neither method wins alone; solidity (pitch is convex) roughly halves each one's failure rate |
| Local chroma (`local_chroma`: deviation from LOCAL ice, not absolute colour) | Base of everything else; absolute hue of a painted line under ice is noise, its deviation from the neighbouring ice is stable |
| **Yellow kickplate** as a homography term (`fit_homography_lines.kickplate_score`) | z = 79.9 (IIHF) / 48.5 (NHL): 8-16x stronger than any painted line, covers the whole perimeter. Peak is ~8 px **outside** the outline (vertical surface): use a FIXED 8 px offset; searching offsets biases -4 px and drops z from 20 to 4 |
| Geometric corridor (restrict Hough/blobs to a window from an approximate H) | Hough precision 8.8% -> 22.2% of real segments, ~11x fewer candidates. **Needs an approximate H to start** |

**Discarded (do not retry)**

| Idea | Measured |
|---|---|
| Convex hull of the ice mask | worse than filling holes in every combination |
| Robust Gaussian instead of the GMM (`robust_surface`) | worse than GMM and than fixed threshold on both datasets |
| Multi-scale ridge filter | no gain over raw response (z 3.1 vs 3.3) |
| Kickplate as whole-region segmenter | catastrophic (p10 IoU 0.007): only 17-38% of high-Δb pixels are near the true outline; the rest is stands/ads. The same signal is z=80 *along a known curve* and useless without the shape constraint |
| Blob shape classification (line vs logo) | distributions overlap (aspect ratio 2.7 real vs 2.0 false); 220 of 1735 components (12.7%) were real lines; lines fragment under occlusion |
| Hough (lines and circles) on the whole frame | real candidate drowned: 1 of 122 segments, 1 of 22 circles |
| "Growing" circle from a seed (ellipse fit, twice) | signal limit: at the exact true edge the colour response is at noise level (median 1.0 vs σ=4.06). Everything that works integrates over hundreds of pixels |
| KLT tracking between frames (`archive/klt_propagate.py`) | p50 158 px, 20% < 25 px, 24% catastrophic (>500 px). LK never reports loss (status=1) but drifts onto specular ice reflections (not fixed to the plane); ~90% of seeded points lost in 1 s. Tracking inside the chroma field: worse (87.8 vs 45.9 px) |
| Joint homography fit over all curves (`fit_homography_lines.fit`) | does not converge even from the exact truth (σ=0 gives 6.8-11 px); spurious maxima, ~3 curves per frame = no RANSAC redundancy. **Worsens** a real YOLO anchor (13 of 22 frames worse, 33→283 px). Do not use as blind refinement |
| Bootstrap H from the region outline, no anchor (`minAreaRect` and a real quadrilateral, all 8 rotation/mirror combos) | detected "corners" are mostly where the ice region is **clipped by the frame border** (3 of 4 at x=0 or y=h-1); broadcast frames rarely contain the whole rink. The input data does not exist |

## 4. Classical auto-labelling v1 (`archive/auto_label.py`) - superseded

Gate: YOLO anchor → 4 signals (contour IoU, kickplate z, curves ok, YOLO inliers) → reproject 56
keypoints. Calibrated on ground truth: correlation with real error `inliers_yolo` r=-0.47,
`iou_contorno` -0.16, `zocalo_z` -0.08, `curvas_ok` +0.12; best single-signal precision 38%
(kickplate z, recall 86%). The gate did **not discriminate** because the classical signals are
*refiners* (need H within 10-20 px), and YOLO was at 60-100 px. Superseded by segmentation
auto-labelling (§8). `label_from_H` / `NKPT` moved to `core/labels.py`.

## 5. Semantic line segmentation - `lab/hockey/train_lines_seg.py`

12 classes (0 background, 1 boards, 2-3 goal lines, 4-5 blue lines, 6 centre line, 7 centre
circle, 8-11 faceoff circles; see `sports/hockey/rink.py::CLASSES` - **not** the 0-55 keypoint
indices). Masks are free: `make_line_masks.py` reprojects the template through each label's
fitted H. U-Net with ResNet34 encoder (not DeepLabV3: without a "+" decoder it outputs 1/8
resolution and blurs 3-5 px lines); class-weighted CE (inverse *square root* of frequency: imbalance
is 1:2173 and pure inverse gives ~2000x weights) + Dice, `ignore_index`, flip with class remap.

**Val IoU alone is misleading** (train loss keeps falling, line IoU stalls at 0.25-0.26 from epoch
~43). The test that matters is `diagnose_lines_seg.py`: slide each template curve along its normal
on the network's probability map and z-score the peak, same criterion as the classical numbers.

| Val | Result |
|---|---|
| IIHF (48 frames) | Beats classical by orders of magnitude: goal lines / centre / faceoff / circle z in the hundreds-thousands (kickplate: 79.9), displacement 0.5-2.5 px; 83% of frames have ALL curves right. `blue_line_A` weak: 56%, d=5 px, z=11.5 |
| NHL (57 frames) | Much weaker: `blue_line_B` and `center_circle` fail (d=35-40 px, at the search limit); only faceoff and `goal_line_A` hold (z 12-200); 46% of frames all-right |

Interpretation: hypothesis validated where data is homogeneous (IIHF); NHL lacks data (344 images
over many broadcasts/rinks). **Synthetic domain randomisation** (`gen_synthetic_lines.py`, 3000
images, reusing ~955 real H as camera geometry; val is always 100% real; 50/50 real/synthetic
sampler) gave mixed results: `center_circle` d 35-40 → 16 px, `blue_line_B` 35-40 → 26.5 px, but
IoU flat and "all curves right" 46% → 39%.

## 6. Segmentation → homography (`core/geometry.py`, `lab/hockey/seg_to_homography.py`)

Goal/blue/centre lines are world lines X=const → **line** correspondences (dual DLT
`l_w ~ H^T l_i`), never sampled into fake points. Circles give point correspondences (centre ↔ centre;
known bias: an ellipse centre is not the projection of the circle centre). Boards give one Y=const line.

Result requiring dof ≥ 10 (5+ correspondences):

| | coverage | p50 | p90 | < 25 px |
|---|---|---|---|---|
| IIHF val | 12% | 16.4 px | 20.7 px | 100% |
| NHL val | 7% | 37.8 px | 156 px | 25% (75% < 60 px) |

dof ≥ 8 (the mathematical minimum) raises coverage to 47%/31% but p50 explodes to 300-500 px:
**redundancy, not a laxer threshold, buys reliability.** Equal or better than YOLO in the frames
it covers, using no keypoint. Six real bugs found on the way (all fixed, several covered by
`tests/test_geometry.py`):

1. **DLT without Hartley normalisation** (rows of magnitude ~1000 vs ~10-60): numerically useless SVD (p50 1858 px).
2. **Circle arc clipped by the image border**: ellipse fit of a partial arc is ill-conditioned (476 px centre error). Filter: mask touches border or arc covers < 360°-150°.
3. **Parallel lines never fix the transverse direction**: all goal/blue/centre lines are X=const; one circle carried it (each correspondence < 16 px, result 2189 px); the SVD condition number looked clean. Fixed by adding the boards (Y=const).
4. **Board-sign choice by algebraic residual: 0/27 correct** (anti-correlated). Fixed by solving the full DLT+refinement for each sign and comparing refinement cost.
5. **`soft_l1` with a mis-calibrated `f_scale` kills the gradient** (residual ~570 px at start). Fixed with two passes: linear loss, then robust loss calibrated near the optimum.
6. **Numerically degenerate seed (~1e14)**: 6 of 8 Jacobian columns exactly zero. Guard `max|Hn| < 50`.

Class confusion (wrong circle/line class) was investigated and **ruled out** (47/47 circles, 93/94 lines correct under the true H).
The visual debug overlay (`solve_from_probs(debug=True)` + `--figuras N`; green = true H, magenta = estimated)
found bugs 3 and 6, which aggregate numbers could not.

Cost gate: see [ADR 0002](../decisions/0002-seg-cost-gate.md).

## 7. Auto-labelling through segmentation (`lab/hockey/auto_label_seg.py`)

`--calibrar` before running on raw video. Run on nhl5-10 (`--cada 5`): 400 accepted frames (2-4% per
video). **Visual QA per video (4 random frames each) is mandatory**:

| Video | Good | Cause of failure |
|---|---|---|
| nhl5 | 3/4 | noise |
| nhl6 | **0/4** | not an NHL rink (Spanish amateur arena): wrong template |
| nhl7 | 4/4 | |
| nhl8 | 4/4 | |
| nhl9 | **0/4** | NHL Stadium Series (outdoor): boards appearance confuses segmentation |
| nhl10 | 4/4 | |

Merged only nhl5/7/8/10 (329 frames) into `hockeyrink_nhl`; nhl6/nhl9 stay quarantined in
`datasets/hockeyrink_auto_seg`. Lesson: sample **per video/source** before merging; a failure can
be systematic for a whole source and the cost filter will not see it.

## 8. Learned camera-pose head - `archive/homography_head.py` (negative)

Motivation: replace per-frame DLT by a network mapping the 12-channel probability map to camera
pose (6D rotation, camera centre, focal), trained with reprojection loss over the 56 template points
(always supervised, in or out of frame). Four real design bugs fixed (broadcasting `(B,1)*(B,)`;
global average pooling collapsed spatial layout → CoordConv + 4x6 pooling; `H/H[2,2].clamp(min=1e-6)`
destroys the sign → separate sign and magnitude; supervising far-out-of-frame points in pixels is ill-conditioned →
clamp prediction and truth to a frame-plus-margin box).

Result after all fixes: **does not work**. Val p50 ≈ 340 px, p90 ≈ 1150 px, 1% < 25 px; train loss flat
(under-fitting, not over-fitting). Diagnostics:

* Parametrisation ceiling (fit R, C, f directly to each true H, no network): p50 4.65 px, p90 40 px,
  85% < 25 px → the pinhole model represents the H fine; the failure is learning the regression.
* Nearest neighbour by cosine similarity of the probability map (train → val): p50 121 px, 11% < 25 px
  (possibly optimistic: near-duplicate frames across splits). Dictionary oracle (best neighbour in H): p50 50 px.
* Nearest neighbour + differentiable refinement against the segmentation (16 val frames): refined top-1
  **worse** than the raw neighbour (IIHF 53 → 234 px). In 8/12 frames the refined pose scores *higher* than
  the ground-truth pose → the objective (probability mass on template polylines) has spurious maxima; root
  cause is likely segmentation quality (IoU ≈ 0.25), not the optimiser.

Decision: no learned solver beats the classical DLT (IIHF 15.5 / NHL 56 px, coverage 14% / 5%) or YOLO
(~61-102 px, full coverage). Common bottleneck = segmentation quality.

## 8b. Literature coverage (`docs/hockey_research.md`)

| Stage | Done | Outcome |
|---|---|---|
| 0 NHL template | yes | see §1 |
| 1 Line segmentation | yes (U-Net, not DeepLabV3+) | §5 |
| 2 Synthetic pre-training + domain randomisation | yes | mixed, §5 |
| 3 Classical pre-annotator (Hough/ellipse/chamfer) | partial | Hough 1/122, ellipse seed fails, contour quad fails |
| 4 DLT + refinement + temporal smoothing | DLT yes; TVCalib-style refinement naive/negative; smoothing negative (KLT) | §6, §3 |
| 5 Domain adaptation (MIC, unlabelled video) | yes | §9, negative |
| Not tried | | NBJW/PnLCalib weights, real TVCalib, keypoints from line×board intersections and circle tangents (Sportlight, 57 pts), Jiang learned error, Shi self-supervised, synthetic pan/tilt/zoom dictionary + chamfer, random-H perturbation augmentation for the head |

Unproven hypothesis about low DLT coverage: each line/circle counts as ONE correspondence; PnLCalib/Sportlight gain
redundancy from many derived points (line×board intersections, circle tangents).

## 9. Appearance augmentation + UDA (MIC) - negative

`lab/hockey/appearance_aug.py` (copy-paste of REAL players from HockeyAI crops, logo augmentation on background
pixels only), `extract_uda_frames.py` (983 unlabelled frames, excludes nhl6 and ±300 frames around val), and
`train_lines_seg.py --uda` (MIC: EMA teacher α=0.999 pseudo-labels the unmasked target frame, student trained on
64 px patches masked at 50%; λ=0.5 ramp 3 epochs). Run 40 epochs from `runs/lineas_seg/best.pt` into
`runs/lineas_seg_uda/`.

Final (NHL val, reference → UDA): `azul_A` 23→54%, `gol_B` 42→58%, `faceoff_B_lo` 75→90%; but `circulo_central`
30→10%, `vallas` 49→37%, `linea_central` 50→42%, "all curves right" 39→32%; DLT p50 56→338 px. On the hand-labelled
val the reference wins or ties almost everywhere (gol_A +20, centre +17, circle +27 pp). **Decision: do not adopt
UDA + augmentation.** Effects of augmentation and UDA cannot be separated (launched together). The weakest classes in
BOTH models are the neutral-zone ones: blue lines, centre line, centre circle.

Also: an earlier experiment retraining with the 329 auto-labelled NHL frames made NHL worse in all three metrics
(coverage 7→5%, p50 37.8→56.0 px, p90 156→174 px), so "more data for the classical DLT" is not the path.

## 10. Reference numbers on the hand-labelled val (`valh`, 71 frames with curves)

Curve hit rate (reference net `best.pt` / UDA net): faceoff_A_hi 69/88, faceoff_A_lo 90/90, faceoff_B_hi 95/89,
faceoff_B_lo 83/78, goal_line_A 80/60, goal_line_B 76/81, boards 70/71 (n=128), blue_line_A 0/20 (n=10, d=37 px),
blue_line_B 60/53, center_line 50/33, center_circle 36/9 (n=11), all-curves-right 46/42.
DLT on random frames: coverage 0% (reference) / 4% with garbage H (UDA). YOLO: see §2 and §2b.

## 11. Tools

| Old command | New command |
|---|---|
| `python training/rink_metric.py <weights> [--sweep]` | `python -m sportcal.lab.hockey.rink_metric <weights> [--sweep]` |
| `python training/diagnose_lines_seg.py [--dataset hockeyrink_nhl] [--n N]` | `python -m sportcal.lab.hockey.diagnose_lines_seg ...` (`LINEAS_PESOS` env var selects a checkpoint) |
| `python training/seg_to_homography.py [--selftest] [--figuras N]` | `python -m sportcal.lab.hockey.seg_to_homography ...` |
| `python training/auto_label_seg.py --calibrar --n 80` | `python -m sportcal.lab.hockey.auto_label_seg --calibrar --n 80` |
| `python training/line_explorer.py` | `python -m sportcal.lab.hockey.line_explorer` |
| `python training/diagnose_lines_fit.py` / `ice_lines_probe.py` / `fit_homography_lines.py` | same names under `sportcal.lab.hockey` |
| `python training/make_line_masks.py [--overlay N]`, `gen_synthetic_lines.py` | same names under `sportcal.lab.hockey` |
| `python training/build_val_nhl.py`, click labeler app | `sportcal.lab.hockey.build_val_nhl`; `streamlit run sportcal/lab/common/annotate_val_app.py` |

Click labeller warnings (`click_labeler.ajusta`; the soccer labeller uses the same measure): the fit is flagged
unstable when `core.geometry.click_sensitivity` (median px at 1920 that the drawn rink moves when every click is ~2 px
off, over the image pixels that fall on the rink +-2 m) exceeds 30. Calibrated against known truth on 352
configurations of 4 random points over 19 val frames: real error ~0.42 x sensitivity (median; p90 ~1.2x); the threshold
warns on 56% of those configurations, and of the ones it does not warn on only 6% are > 15 px off. A first version
averaged over the whole image and warned on 91-100% of fits even at 7 px real error: stands and horizon blow up with
any fit. The "IMPOSIBLE" warning is the plausibility gate of section 14d.

GPU note: an 8 GB card (RTX 3060 Ti) cannot run a pose training (~7.8 GB at imgsz 1024) and any other model
at once; check `nvidia-smi` before running `rink_metric` or the auto-labellers.

## 12. Facts worth not re-deriving

* Own videos `nhl3.mp4`...`nhl10.mp4` are indexed: sequential `cv2.VideoCapture` reads give exactly the frame named
  by the numeric suffix in `datasets/hockeyrink_nhl/` (mean diff 1.4/255, JPEG). `nhl3.mp4` has no shot cuts in
  frames 0-9523 (159 s at 60 fps).
* ~80% of `hockeyrink_nhl` labels are reprojections (0.0 px error against themselves): bootstrap-style validation on
  them is artificially confident.
* `split_val` re-derives the whole partition and moves files; running it after new frames were merged silently
  changes the validation set (it happened once by accident and was restored from `hockeyrink_nhl_rp`).

## 13. Open questions (highest value first)

1. Why is DLT coverage on random NHL frames ~0-4%? Derived correspondences (line×board intersections, circle
   tangents) are the untested hypothesis.
2. Neutral-zone classes (blue lines, centre line/circle) fail in NHL in every model: more/better data for those,
   not more epochs.
3. A larger independent val: keep annotating `valh` (`sportcal/lab/common/annotate_val_app.py`) and consider using the
   hand-labelled frames (2 px, best labels that exist) as **train** for hard cases with a per-video split.
4. Reject wrong-geometry frames (nhl6, nhl9 class of failure) automatically: the plausibility gate (section 14d) refuses
   the homographies no camera gives; a wrong but camera-like H still goes through.

## 14. Keypoint + line model for NHL ("model A") - `lab/hockey/train_kpline.py`

**Question:** does a PnLCalib-style model - heatmaps for named keypoints and for the ends of the straight lines, then one
DLT over points and lines - get closer to the ~8 px target than YOLO (section 2) and the segmentation DLT (section 6)?

**Setup:**

* Targets rendered on the fly from the index H (section 0b), never stored: the 56 template keypoints plus both visible
  ends of 9 straight lines (4 board runs, 2 goal lines, 2 blue lines, the centre line: `rink.straight_lines`) = 74
  channels, sub-pixel Gaussians of sigma 1.5 px at 480 (6 px at 1920).
* U-Net / ResNet34 of section 5 (ImageNet encoder), 960x544 input, heatmaps at half resolution: full resolution ran out
  of memory (6 GB at batch 2 on the CPU). CenterNet focal loss, final bias at p = 0.01.
* Augmentation: a turn / zoom of the camera about its centre (`core.camera.ptz_warp`, exact: the label becomes G H), a
  mirror (the index convention renames the points), brightness / contrast.
* Decoding: the peak of each channel (>= 0.3), RANSAC over the points, then one DLT over the inlier points plus the lines
  (`core.geometry.solve_points_lines`).
* Two phases: pretrain on every train label (NHL + IIHF, 1159 frames), fine-tune on the 48 hand labels of training
  videos. Checkpoints are chosen by the median H error on dev (stored labels, ~11-16 px off, so a few px between
  checkpoints is noise) and the result is reported on the 63 clean test frames.
* Floors measured without a network: reading perfect half-resolution heatmaps back costs ~1 px at 1920; perfect
  targets through the whole decode + solver chain come back within 0.4-0.5 px (`tests/test_labels.py`, at 960).

**Commands:**

    python -m sportcal.lab.hockey.train_kpline --phase pretrain
    python -m sportcal.lab.hockey.train_kpline --phase finetune --init runs/kpline/pretrain/best_h.pt
    python -m sportcal.lab.hockey.train_kpline --eval runs/kpline/finetune/best_h.pt --split test

**Result** (2026-09-23, batch 4 / workers 4 throughout, default CUDA path, no OOM at any point):

| phase | epochs | wall time | batch |
|---|---|---|---|
| pretrain | 60 | ~82 min (~80 s/epoch) | 4 |
| finetune | 30 | ~2 min (~4 s/epoch after the first, which pays the DataLoader worker start-up) | 4 |

Checkpoint within each phase is the one with the lowest median H error on **dev** (stored labels, ~11-16 px off per
the section 0b audit - a few px between checkpoints is noise, not signal). Reported below on the two held frame sets
of section 0b:

| model | split | n | coverage | p50 | p90 | < 10 px | < 25 px |
|---|---|---|---|---|---|---|---|
| pretrain `best_h.pt` | test | 63 | 62% | 9.7 px | 129.7 px | 33% | 49% |
| pretrain `best_h.pt` | test_leaky | 23 | 100% | 18.3 px | 59.6 px | 26% | 61% |
| finetune `best_h.pt` | test | 63 | 78% | 9.2 px | 639.2 px | 44% | 59% |
| finetune `best_h.pt` | test_leaky | 23 | 100% | 6.4 px | 27.3 px | 65% | 87% |

**Decision:** promising but not conclusive - needs more data before adoption. Fine-tuning helps on `test` (p50 9.7 →
9.2 px, < 25 px 49% → 59%, coverage 62% → 78%), but coverage on `test` stays well short of full and the p90 explosion
(639 px) shows some solves fail outright rather than land imprecisely. Not yet compared with YOLO (section 2) or the
segmentation DLT (section 6) on this same 63-frame test set - both were measured on different val sets, so no
same-set verdict against them exists yet. Next step before further investment: run YOLO on this `test` split and
decide whether the coverage gap is a decoding threshold (the 0.3 peak cut in Setup) or a genuine model failure.

**Caveats:**

* `test_leaky` is not a held-out set: those 23 frames are the training-video hand labels also used for fine-tuning
  (section 0b, "train + test_leaky, 48 frames" both enter the fine-tune data). Its near-full coverage and low p50
  reflect near-train performance, not generalisation - only `test` (63 frames, nhl4/nhl9/nhl10, never trained on) is
  the real number.
* Checkpoint selection ran on noisy dev labels (~11-16 px off): several dev-epoch checkpoints in the 89-90% coverage
  range are indistinguishable at that noise floor, so "the best epoch" within a phase is a soft pick, not a precise
  one.
* p90 on `test` swings hard between phases (129.7 px pretrain vs 639.2 px finetune) on only 63 frames - a couple of
  outlier solves dominate it; treat single p90 figures as high-variance, unlike p50 and coverage.

**Same-set comparison with YOLO and failure analysis** (2026-09-23): both methods on the same 63 `test` frames, same
measure (`geom_error`, 1 m grid, points the label puts in frame), one frame at a time
(`train_kpline --eval <weights> [--yolo]`). YOLO's H is canonicalized first (its labels mixed mirror conventions,
section 0b). The YOLO figures differ from section 2 because the set, the measure and the inference path all differ
(section 2 is valh-76, keypoint reprojection error, batched inference - ADR 0003).

| method | test (63): coverage / p50 / < 10 px / < 25 px | nhl10 (26) | nhl4 (17) | nhl9 (20, outdoor) |
|---|---|---|---|---|
| model A, finetune | 78% / 9.2 / 44% / 59% | 100% / **7.2** / 73% / 100% | 59% / 7.0 / 41% / 47% | 65% / 512 / 10% / 15% |
| model A, pretrain | 62% / 9.9 / 32% / 49% | 100% / 10.2 / 50% / 85% | 47% / 8.3 / 41% / 47% | 25% / 154 / 0% / 5% |
| YOLO `yolo26m-18` | 100% / 84.5 / 0% / 2% | 100% / 41.7 / 0% / 4% | 100% / 851 / 0% / 0% | 100% / 243 / 0% / 0% |
| YOLO `yolo26m-17` | 90% / 204 / 0% / 0% | 96% / 135 / 0% / 0% | 76% / 1336 / 0% / 0% | 95% / 212 / 0% / 0% |

On the two indoor held-out videos together (nhl4 + nhl10, 43 frames) the fine-tuned model answers 84% with p50 7.1 px,
p90 17.2 px, 60% < 10 px, 79% < 25 px of all frames; `yolo26m-18` answers all with p50 50.6 px and 2% < 25 px.

* **nhl9 is where it breaks:** the outdoor Stadium Series game, quarantined from training (section 7), with nothing
  like it in the train set. 10 of the 12 answers over 50 px are nhl9 frames.
* **Refusals are sparse views:** the 14 refused frames (7 nhl4, 7 nhl9) had 1-3 keypoints and 0-3 line ends above the
  0.3 peak threshold. Lowering it buys coverage with accuracy: 0.3 -> 78% / 9.2 px, 0.2 -> 90% / 11.6 px, 0.1 ->
  100% / 18.9 px (test, p50 of the answered frames). Keep 0.3.
* **Candidate confidence gate (found on test, not validated):** every answer built from >= 7 keypoints above threshold
  was within 18 px (34 frames, p50 6.9 px), and all 12 answers over 50 px came from <= 6 keypoints. Picked on the test
  set itself, so it needs confirming on frames the choice never saw before it is used anywhere.

**Decision (revised, proposed - owner's call):** model A replaces the "promising, not conclusive" reading above. It is
the best NHL method measured: ~10x lower median than YOLO on the same frames, and on standard indoor broadcasts of
videos it never saw it reaches the ~8 px target when it answers. Proposed for ADR 0003 as the first product stage with
YOLO as fallback, once the keypoint-count gate is confirmed on independent frames. Open: outdoor / unusual rinks (need
training frames of that kind) and sparse views (refused, not wrong).

## 14b. Why model A refuses or fails: missing views, not missing markings (2026-09-24)

**Question:** the refused test frames looked "sparse". Is the fix more keypoints per frame (derived points, circles), or
something else?
**Method:** for the 26 `test` frames model A (fine-tune) refuses or misses by > 50 px, count what the hand label puts in
frame: template keypoints, straight lines, circles with > 25% of their arc in view; compare with the keypoints the model
finds above 0.3. `core.camera.view_angle` (angle of the rink's long axis in the image) separates the side camera from a
camera behind the goal.
**Result:**

| frames | template keypoints in frame | straight lines | circles | keypoints the model finds |
|---|---|---|---|---|
| nhl4, refused / > 50 px (9) | 12-50 | 2-8 | 2-5 | 1-5 |
| nhl9, refused / > 50 px (17) | 11-20 | 2-5 | 1-3 | 2-6 |
| test + test_leaky frames within 25 px (57) | median 17 (min 7) | median 4 | >= 1 in all | - |

The markings are there; the network does not find them. Six of the 17 nhl4 test frames come from an end-zone camera
(view angle > 45 degrees), a view with 4 frames among the 585 NHL train frames: model A answers 1 of the 6, wrongly.
On the side-camera test frames (57, nhl9 included) it answers 84% with p50 8.6 px. nhl9 is the outdoor game.
**Decision:** more keypoints per frame would not rescue these frames - the network would have to learn them in the views
it has not seen. The fix is training data of those views: `mine_views` finds end-zone shots in unlabelled video by
thumbnail nearest-neighbour against the labelled end / side frames. In the train videos it found the only two such
shots (clip 0-5 s, nhl7 ~161-165 s) and one side-view false positive (nhl3); 39 frames are queued in
`datasets/hockeyrink_nhl_endview` for hand labels, which `build_h_index` puts in train. Two shots from two arenas are
little diversity: new NHL games (`fetch_clips`) are the real source.

    python -m sportcal.lab.hockey.mine_views          # -> datasets/hockeyrink_nhl_endview/queue.json

## 14c. Keypoints derived from the circles ("model B1") - `train_kpline --keypoints derived`

**Question:** do PnLCalib-style derived keypoints make model A more precise or more robust where it already answers?
Section 14b says they are not the fix for the refused views; this measures what they add on the rest.
**Setup:** `rink.derived_keypoints` adds 34 points to the 56: on each of the 5 painted circles the two points where the
tangent runs across the rink and the four at 45 degrees (the hash marks and centre-line crossings already sit near the
other two extremes), and the middle of each rounded board corner - 108 output channels instead of 74. Everything else as
section 14. The set is closed under the rink's mirrors (`tests/test_sports.py`), so mirrored frames rename them. Runs go
to `runs/kpline/<phase>-derived`; `--eval` reads the set from the weights. Trained on the same index as model A so that
the comparison isolates the keypoints; the end-view labels of 14b go into a later run.
**Caveat:** the keypoint-count gate of section 14 (>= 7) was measured on 56 points; with 90 it has to be measured again.

    python -m sportcal.lab.hockey.train_kpline --phase pretrain --keypoints derived
    python -m sportcal.lab.hockey.train_kpline --phase finetune --keypoints derived --init runs/kpline/pretrain-derived/best_h.pt
    python -m sportcal.lab.hockey.train_kpline --eval runs/kpline/finetune-derived/best_h.pt --split test

**Result** (2026-09-24, same index as model A, batch 4 / workers 4 throughout, no OOM):

| phase | epochs | wall time | batch |
|---|---|---|---|
| pretrain-derived | 60 | ~113 min (~110 s/epoch) | 4 |
| finetune-derived | 30 | ~5 min (~5 s/epoch after the first) | 4 |

~40% slower per epoch than model A's pretrain (110 s vs 80 s, section 14): the extra 34 derived points add 34 of the
108 output channels. Checkpoint chosen the same way as model A (lowest median H error on dev). Compared with model A
fine-tune on the same 63-frame `test` split, total and per video:

| model | split | n | coverage | p50 | p90 | < 10 px | < 25 px |
|---|---|---|---|---|---|---|---|
| A, finetune | test (total) | 63 | 78% | 9.2 | 639.2 | 44% | 59% |
| B1, finetune-derived | test (total) | 63 | 81% | 10.3 | 249.2 | 38% | 57% |
| A, finetune | nhl10 | 26 | 100% | 7.2 | 15.8 | 73% | 100% |
| B1, finetune-derived | nhl10 | 26 | 100% | 7.7 | 13.9 | 65% | 92% |
| A, finetune | nhl4 | 17 | 59% | 7.0 | 326.0 | 41% | 47% |
| B1, finetune-derived | nhl4 | 17 | 53% | 7.9 | 148.0 | 41% | 47% |
| A, finetune | nhl9 (outdoor) | 20 | 65% | 512.0 | 1902.3 | 10% | 15% |
| B1, finetune-derived | nhl9 (outdoor) | 20 | 80% | 196.9 | 1279.6 | 0% | 20% |

**Decision:** derived keypoints do not make model A more precise where it already answers - p50 on the two clean indoor
videos (nhl10, nhl4) is the same or slightly worse than the 56-point model - but they buy some robustness: total-`test`
p90 falls from 639 to 249 px, and on nhl9 (the outdoor game, out of distribution) coverage rises 65% -> 80% and p50
falls 512 -> 197 px, still far from usable. Consistent with the section 14b finding: derived points are not the fix for
refusals on views the network has not seen, they mainly shrink how badly the frames that do fail miss. Not a
replacement for model A as proposed in section 14; the keypoint-count confidence gate (>= 7, measured on 56 points in
section 14) has not been re-measured for the 90-point set.

**Caveats:**

* Only `test` was evaluated here (the brief for this run), not `test_leaky`; the `test_leaky` leakage caveat of
  section 14 still applies if that split is measured later.
* Same dev-noise caveat as section 14: checkpoints were chosen on stored dev labels ~11-16 px off, so small per-epoch
  differences within a phase are not reliable signal.
* One run each (pretrain-derived + finetune-derived), no repeats: per ADR 0003, treat the p50/p90 gaps against model A
  as indicative, not as a settled effect size.
* A later evaluation of the fine-tuned B1 reported by the owner (same command, 2026-09-24 ~20:00) gave test 84% /
  p50 10.5 / p90 218 / < 25 px 57%; nhl10 100% / 7.9, nhl4 53% / 7.2, nhl9 90% / 172 (coverage / p50 px). With the
  plausibility gate (section 14d, `--gate`): 79% / 9.5 / p90 194; nhl4 47% / 7.0, nhl9 80% / 168. The gate removes far
  less from B1 than from model A (A: p90 639 -> 17, 11 of 12 gross answers refused): B1's wrong nhl9 answers look like
  real cameras, so they are harder to catch. Decision unchanged: keep model A.

## 14d. Homographies no camera can produce: a plausibility gate - `core.camera.is_plausible_view` (2026-09-24)

**Question:** an overlay showed a rink line crossing a faceoff circle: impossible for a real camera, easy for a free
8-parameter H. Is "physically impossible" a usable sign of a wrong H, for model answers and for hand labels?
**Signs:** (a) part of the rink behind the camera - the homogeneous depth (third row of H) changes sign over a 25 x 11
grid of the rink; drawn, that part comes back mirrored across the horizon, which is the line through the circle;
(b) no valid focal length - `pinhole_residual` finds none between 0.4 and 8 image widths (mismatch 1).
**Result:** model A fine-tune, its 72 answers on `test` + `test_leaky` (12 of them > 50 px), and the hand labels of the
same 86 frames:

| sign | answers > 50 px | answers <= 50 px | hand labels |
|---|---|---|---|
| part of the rink behind | 12/12 | 15/60 | 31/86 |
| no valid focal | 11/12 | 7/60 | 18/86 |
| **both** (the gate) | **11/12** | **0/60** | **0/86** |
| focal < 0.8 widths (rejected) | 1/12 | 0/60 | 6/86 |

Neither sign alone works because real views show each: a side camera at pan ~0 has the rink's long axis parallel to the
image plane and no measurable focal (the orthogonality equation divides by zero), and a camera panned along the rink has
part of it behind. Together they are the fold. A minimum focal would catch the one gross answer the gate misses (nhl9
frame 8160, f = 0.44 widths, part of the rink behind) but refuses 6 correct nhl4 hand labels (0.34-0.66 widths): the
focal of a real view is too poorly conditioned to bound.

With the gate (`train_kpline --eval <weights> --gate`), model A fine-tune:

| set | coverage | p50 | p90 | > 50 px | < 25 px (of all frames) |
|---|---|---|---|---|---|
| test (63), no gate | 78% | 9.2 | 639 | 12 | 59% |
| test (63), gate | 60% | 6.9 | 16.8 | 1 | 59% |
| test + test_leaky (86), no gate | 84% | 8.3 | 510 | 12 | 66% |
| test + test_leaky (86), gate | 71% | 7.2 | 18.8 | 1 | 66% |

It refuses no answer under 25 px. Per video on `test`: nhl10 stays at 100%; nhl4 59% -> 47% (its 2 gross answers);
nhl9 65% -> 20% (9 of its 10 gross answers). Over the whole index it refuses 0 of the 116 hand labels and 11 of the 1338
stored ones: 4 `auto_seg` (nhl7 frames 9725, 9895, 10770; nhl8 2130), 5 `nhl_prior` (clip 810, 1020, 1110; clip2 180 in
dev; nhl4 1800, excluded) and 2 IIHF - 9 train and 1 dev label that no camera gives. None of the 18 soccer hand labels
trips it (`datasets/soccer_labels`, field box +-52.5 x +-34 m).

**Decision:** the gate is an evaluation option (`--gate`; checkpoints are still chosen ungated) and a warning in both
click labellers ("IMPOSIBLE", not a block). It has no tuned threshold, only a physical rule, but it was checked on these
test frames only: confirm it on new frames before a product stage relies on it. At the next index rebuild (after B1),
drop the 10 train/dev labels it refuses.

## 14e. Fewer clicks for the end-view queue: carry a hand label along the shot; Claude as the clicker (2026-09-24)

**Question:** the end-view queue (section 14b) is bursts of frames 0.5 s apart inside one shot. Can one hand label per
few frames be carried to the rest by the camera motion (`core.motion`), and can Claude click the points instead?
**Propagation - method:** the 24 labelled queue frames (3 shots: clip 0-4 s, nhl3 166-170 s, nhl7 160-164 s). Step
motions every 5 frames from `core.motion.track_pair` (ice surface), chained from each hand label to every other label of
its shot (backwards with the inverse), scored at the target's clicks (median px at 1920, min over the rink's mirrors:
these labels were named without one convention). The error includes both labels' own noise (each label fits its own
clicks within 1-6 px).

| where the features come from | 0.5 s: p50 / max | 1.0 s | 1.5 s |
|---|---|---|---|
| whole frame (`region_kind="all"`) | **5.4 / 13.8** | **7.5 / 19.5** | **9.5 / 21.5** |
| background only (stands, boards) | 5.0 / 78 | 6.9 / 91 | 9.2 / 84 |
| ice only | 18 / 216 | 28 / 314 | 43 / 375 |
| no motion (the start label as is; rink-grid measure) | ~60 | ~110 | ~150 |

Background-only is best on nhl7 (2-3 px on the grid) but misses some clip pairs by 78-91 px; the whole frame never
fails badly. Labelling one frame in three and carrying it to the other two gives ~5-8 px, worst ~20: better than the stored
train labels (median 11-22 px, section 0b), not good enough for test. Three shots of one camera type.
**In the labeller** (`click_labeler.propaga`, 2026-09-24): opening a hockey frame proposes the saved label of the same
video nearest before and after it (<= 2 s), carried with the whole-frame motion every 5 frames; "Aceptar la propuesta"
fixes it with 6 spread points (drags still move the fit). Leave-one-out on the 28 end-view labels then saved: the nearest
proposal lands p50 5.8 px from the frame's own clicks, max 34.9 (clip frame 270 from 300); ~2 s per proposal. A saved
proposal is recorded (`"origen"` in clicks.jsonl) and `build_h_index` gives it source `propagated`: train, or excluded
for a held-out video - never test, and not in the hand-only fine-tune.
**Status 2026-09-24 evening:** 36 end-view labels (clip 0-330 and 930-1110, nhl3 9990-10170, nhl7 9630-9900); 7 of
them are accepted proposals, and clip 960-1110 is a chain (each carried from the previous accepted one, so its error can
add up along the chain). Some of the day's later labels were clicked with the browser zoomed in, without the whole frame
in view: they are being checked against their neighbours before the next index rebuild.
Checked 2026-09-25: carried from both neighbours, 22 of the last 28 agree within 1-9 px; the 6 that disagree more
(clip 240/270/300/1110, nhl7 9870/9900) fit the paint better than the carried proposal on inspection (fast, blurred pans
break the tracking). None changed.

**Index rebuild 2026-09-25** (`build_h_index`, previous index kept as `scratch_frames/hockey_h_before_endview.jsonl`):
1495 frames. New: the 36 end-view labels (29 `hand` + 7 `propagated`, all train) and 16 new `valh` hand labels (9 `test`:
nhl10 x6, nhl9 x3; 7 `test_leaky`). Gone: the 11 stored labels the plausibility gate refuses (section 14d; `build_h_index`
now drops them). `test` is now 72 frames (nhl10 32, nhl4 17, nhl9 23), so section 14's numbers on 63 frames are not
directly comparable: re-evaluate model A on the new `test` next to the retrain.

**Claude clicking - method:** one labelled frame (nhl7 frame 9750, 7 hand clicks) clicked by Claude from zoomed crops
with a pixel grid before seeing the hand label; one unlabelled frame (clip frame 990, a glass-level centre-ice view)
judged by eye by the owner. **Result:** 13 points at first; the least-squares residuals (up to 46 px) exposed 2 wrong
identities: both posts (seen through the net mesh by the owner) were placed where a crease line meets the goal line and
where the goal line disappears behind the net frame. Without them (11 points, self-fit median 5.9 px) Claude's H is
5.8 px (median) from the hand clicks and 5.7 px from the hand H over the rink grid; with them, 9.3 and 16.7. The hand H misses Claude's clicks on the right circle by 33-44 px, where the hand
H extrapolates (no hand click there); neither H matches that circle's outer edge (lens distortion is the likely reason).
**Decision:** n = 1, not a measure of Claude as a labeller. The failure mode that matters is identity, not pixels: a
wrong name costs tens of px and only a residual check or a human catches it. Claude's clicks stay candidates for a
human to accept, never labels on their own.

## 14f. Model A retrained with the end-view labels ("A1", 2026-09-25)

**Setup:** section 14 unchanged (pretrain 60 epochs + fine-tune 30), on the index rebuilt in section 14e (36 end-view
labels in train, the 11 impossible stored labels out). Compared with the previous model ("A0", `runs/kpline/*-A0`) on
the new 72-frame `test`; errors as in section 14 (names as labelled), plus "any mirror": the error with the label taken
in whichever of the rink's 4 mirrors fits best - the end-view labels were named without one convention (section 14b),
so a geometrically right end-view answer can carry the other mirror's names.

| model | coverage | p50 | < 25 px | > 50 px | nhl10 cov / p50 | nhl4 cov / p50 | nhl4 < 25 px, any mirror | nhl9 cov / p50 |
|---|---|---|---|---|---|---|---|---|
| A0 | 79% | 9.2 | 60% | 13 | 100% / 7.2 | 59% / 7.0 | 47% | 65% / 512 |
| A1 | 89% | 11.2 | 61% | 18 | 100% / 6.6 | 100% / 85 | **76%** | 65% / 321 |
| A0, `--gate` | 62% | 7.0 | 60% | 1 | 100% / 7.2 | 47% / 6.5 | - | 22% / 24 |
| A1, `--gate` | 74% | 8.8 | 61% | 7 | 100% / 6.6 | 82% / 11.0 | - | 30% / 15 |

(Evaluated on CPU from the same weights; the owner's GPU run of A1 gave the same figures within a frame or two.)

The 6 end-view `test` frames (nhl4): A0 answered 1 of 6 (wrong); A1 answers all 6, and with any mirror 5 of them are
1.9, 8.0, 13.3, 18.1 and 31.8 px off, one is 416 px off. As named they are 260-415 px off: **the network learned the end
view; what it cannot learn is which end is which, because the training labels do not say it consistently.** That is
also why the plausibility gate does not help there (the answers are real cameras, just renamed). nhl10 is unchanged
(p50 7.2 -> 6.6).
**Decision:** keep A1 as the working model. The end-view symmetry has to be settled before these answers are usable:
either a naming rule for end views (e.g. the zone the camera sits behind is always zone B), applied to the labels and to
`canonicalize`, or the planned temporal resolution from the tracker. The gate's advantage on the side views remains.

## 14g. One naming rule for side and end views - `core.camera.canonical_mirror` (2026-09-25)

**Why:** section 14f - the model learned the end view but named it at random, because `canonicalize` ("zone A on the
image left") is undefined when the rink's long axis runs up the image, and the end-view labels came in all 4 mirrors.
**Rule** (sport-agnostic, for any field symmetric in x and y; hockey's `build_h_index.canonicalize` now calls it):
measured at the field centre - side view (long axis within 45 degrees of the image horizontal): +x right, +y down, the
old hockey convention unchanged; end view: +x down (the end the camera sits behind is zone B) and +y left, the side rule
turned a quarter. It names the view, not the arena: which physical end a frame shows still needs context (the tracker,
or asymmetric markings such as the referee crease).
**Effect on the index** (rebuilt; previous one in `scratch_frames/hockey_h_before_endrule.jsonl`): 33 labels renamed,
28 of them end views (20 end-view train labels, 6 dev, 2 test). The 24 pairs of end-view labels 0.5 s apart in one shot
now all carry the same names (the best-matching mirror is the identity for every pair). Side views: 5 renamed (long
axis close to 45 degrees at the field centre). Next: retrain model A on it and score the end views as named.
**Result - model "A2"** (section 14 recipe retrained on this index; owner's GPU run, 2026-09-25), 72-frame `test`:

| model | coverage | p50 | p90 | < 10 px | < 25 px | nhl10 cov / p50 / < 25 | nhl4 cov / p50 / < 25 | nhl9 cov / p50 / < 25 |
|---|---|---|---|---|---|---|---|---|
| A0 (section 14f) | 79% | 9.2 | - | 44% | 60% | 100% / 7.2 / 97% | 59% / 7.0 / 47% | 65% / 512 / 17% |
| A1 (section 14f) | 89% | 11.2 | - | 39% | 61% | 100% / 6.6 / 94% | 100% / 85 / 47% | 65% / 321 / 26% |
| **A2** | 83% | **8.2** | 269 | **54%** | **67%** | 100% / 7.0 / 97% | 94% / **6.3** / **88%** | 52% / 292 / 9% |
| **A2, `--gate`** | 76% | **7.2** | **65** | 54% | 67% | 100% / 7.0 / 97% | 94% / 6.3 / 88% | 30% / 248 / 9% |

(A0 and A1 were scored before the rule renamed 2 end-view `test` labels; with the rule they only get the "any mirror"
credit of section 14f, 76% < 25 px for A1 on nhl4.) The rule is what made the end-view data usable: nhl4, the video
with the 6 end-view `test` frames, goes from 47% to 88% of frames within 25 px, named as labelled, while nhl10 holds.
nhl9 (outdoor) is still wrong and the gate still removes most of its answers.
**Decision:** A2 is the working model (`runs/kpline/finetune`; A0 and A1 kept as `*-A0`, `*-A1`). Open: nhl9, and the
keypoint-count / plausibility gates, still to be confirmed on frames they were not chosen on.

## 14h. A "fresh" set from 4 new games, to confirm the gates (2026-09-25)

**Why:** the keypoint-count gate (section 14) and the plausibility gate (section 14d) were both read off `test`; they
need frames nothing was chosen on. **Videos:** nhl11-nhl14, 4 minutes each (`fetch_clips --first-index 11`), new
arenas. **Queue** (`queue_fresh`): shots cut from a grey thumbnail, kept where the HockeyRink detector sees the rink,
the middle of each shot plus one frame every 10 s inside long shots, 10 per game spread over the clip - 40 frames, none
picked by our own models. A look at the sheet: side and end views of every game, one close-up that slipped through
(nhl11 frame 8052, to skip). Labels go to `hockeyrink_nhl_fresh`, split `fresh` (never train or test, and a saved
proposal is excluded), clicked by hand with no model proposal, so the set stays independent of what it measures.

**Result** (2026-09-25): 38 frames labelled (2 skipped: the nhl11 close-up and nhl14 frame 5796), split `fresh` of the
rebuilt index (1533 frames). Model A2 (`runs/kpline/finetune`), scored on CPU with the section 14 measure:

| | coverage | p50 | p90 | < 10 px | < 25 px | > 50 px |
|---|---|---|---|---|---|---|
| no gate | 100% | 7.2 | 11.0 | 82% | 97% | 1 |
| plausibility gate (14d) | 97% | 7.2 | 10.9 | 82% | 97% | 0 |
| >= 7 keypoints (14) | 89% | 7.1 | 10.6 | 76% | 89% | 0 |

Per game p50: nhl11 7.4, nhl12 6.5, nhl13 6.4, nhl14 8.1 px; every game answered on every frame. The only gross answer
(nhl13 frame 10575, a close view of a goal crease, 4 keypoints, 549 px) is the one the plausibility gate refuses; it
refuses nothing else. The keypoint-count gate also drops it but costs 3 good answers (6.2, 8.7 and 14 px, 6 keypoints
each). The 4 end views of the set (nhl12 x2, nhl14 x2) are 3.2-4.5 px off, named as labelled: the naming rule of section
14g holds in new arenas.
**Decision:** on games nothing was tuned on, A2 reaches the ~8 px target the product asked for (ADR 0003) at full
coverage, and the plausibility gate is confirmed as the gate (the count gate is not needed). Adopted in ADR 0003
(2026-09-25): A2 + plausibility gate as the first product stage (`product.hockey.KplineEstimator`), YOLO as optional
fallback. Caveats: 38 frames, 1 gross answer - the gate is
confirmed on one case; outdoor games (nhl9) remain out of reach.

## 14i. Temporal smoothing along the camera motion - `product.pipeline.HomographyPipeline(smooth=...)` (2026-09-25)

**Why:** on video (the product demo on nhl11) the minimap shakes. Each frame's H is estimated on its own, and a few px
of error per frame move the whole projected rink from one frame to the next.
**Method:** a complementary filter. The camera motion between consecutive frames (`core.motion.estimate_motion`, KLT
over the whole frame, one homography by RANSAC at 1 working px) carries the last output to the new frame, and the new
answer only pulls it a fraction `smooth` of the way (`core.geometry.blend_homographies`, mixed on the image of the
frame's inner quad). An answer more than 150 px (at 1920, largest distance on that quad) away from the carried H counts
as a refusal unless it persists for more than 5 frames (a missed cut), which resets the filter. A hold follows the
camera instead of freezing.
**Measure:** the 38 `fresh` labels of section 14h, each with the 30 frames before it (960 px JPG, matched to the label
image by pixels), model A2 answers cached per frame. Error at the labelled frame against the hand label (section 14
measure), and jitter: for each pair of consecutive frames, the largest distance on the frame's inner quad between the
output H and the previous output carried by the camera motion, i.e. the frame-to-frame change the camera does not
explain (px at 1920). Holds and resets are counted over the 1178 frames.

| smooth | answered | p50 | p90 | < 10 px | max | jitter p50 | jitter p90 | holds | resets |
|---|---|---|---|---|---|---|---|---|---|
| none (per frame) | 37/38 | 5.9 | 10.7 | 31 | 17 | 29.6 | 104.3 | - | - |
| 0.05 | 37/38 | 6.7 | 10.9 | 32 | 13 | 1.5 | 4.0 | 52 | 4 |
| **0.1** | 37/38 | 6.3 | **9.6** | **35** | **11** | 2.8 | 7.7 | 41 | 3 |
| 0.2 | 37/38 | 6.6 | 9.6 | 33 | 12 | 5.1 | 15.2 | 36 | 3 |
| 0.3 | 37/38 | 6.5 | 10.7 | 30 | 21 | 7.6 | 22.1 | 40 | 3 |

**Two settings that looked harmless were not:**

* The RANSAC threshold of the motion. With `core.motion`'s default of 3 working px (at `smooth` 0.1): p50 7.8, p90 16.4,
  max 25 px. Static overlays (score bug, channel logo) sit within 3 px of a slow pan, pull the fitted motion towards
  zero, and the carried H lags by a bias the filter only corrects at rate `smooth`. At 1 px (0.5 gives the same) they
  drop out. Taking the motion from the stands only (region "background") or from the ice ("field") was worse (p50 6.7 /
  10.7, max 45 / 47 px at the 3 px threshold).
* The jump limit. At 40 px, normal answer noise on the far corners of the quad crossed it: 40% of the frames became
  holds and the filter reset 35 times (a visible jump each time) - its low jitter came from ignoring the model, not
  from filtering it. At 150 px: 3.5% holds, 3 resets. 100 and 300 px give similar numbers.

**Decision:** `smooth=0.1` in the demo and for video (ADR 0003): jitter p50 from 29.6 to 2.8 px, accuracy the same or
slightly better than per frame (4 more frames under 10 px, max 17 -> 11 px). Single images keep `smooth=None` (the
filter needs consecutive frames).
**Caveats:** 30-frame sequences, so drift over a long shot is corrected continuously but not measured beyond that. The
jitter reference is the same KLT motion the filter uses, so a wrong motion would not show in it; the error at the
labelled frame is the check for that. The detection boxes jitter too (the feet point is the bottom of the box); that is
not addressed here.
