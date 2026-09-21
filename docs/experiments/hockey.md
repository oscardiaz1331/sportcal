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
4. Reject wrong-geometry frames (nhl6, nhl9 class of failure) automatically: nothing measured does it yet.
