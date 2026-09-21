# Soccer experiments - what was measured and what was decided

Soccer is the second sport, built to test whether the hockey ideas (colour → play region → local chroma → line
geometry) transfer, and to exercise the sport-agnostic `core/`. It is **much smaller than hockey**: one match video,
2-5 hand-picked frames per claim, no validation set, no trained model. Treat every number below as "an observation on a
handful of frames", not a benchmark. Sources: the lab code docstrings and the session notes of 2026-09-20/21, condensed
here when the lab was ported into `sportcal`.

Sport: `sports/soccer/field.py` (`soccer-fifa`, 105x68 m, origin at the centre, +Y = far touchline = image up).
Commands (`python -m sportcal.lab.soccer.<module>`, apps via `streamlit run <path>`):

| Tool | Command |
|---|---|
| Synthetic + real evaluation of the solver | `python -m sportcal.lab.soccer.evaluation --sintetico --n 40` / `--video soccer.mp4 --frames 900,3000,5100 --salida montaje.jpg` |
| Interactive classical-CV lab (any sport) | `streamlit run sportcal/lab/common/classical_cv_lab.py` |
| Click labelling (hockey and soccer) + solver suggestions | `streamlit run sportcal/lab/common/annotate_val_app.py` |

Data: `soccer.mp4` = minutes 27:00-31:00 of one YouTube match (`4YGmrxdAq_c`), plus `soccer2.mp4`. Frames used:
900, 1500, 2400, 3000, 5100. Hand labels go to `datasets/soccer_labels/` (or `$SOCCER_LABELS_DIR`); as of the port the
directory holds no labels yet (only `skipped.json`).

## 1. The 180-degree twin (a real solver bug) - `sports/soccer/field.py::canonicalize_H`

The field is identical under the world turn (X, Y) → (-X, -Y): `H` and `H @ Rot180` draw the same lines with the same
score. The line-intersection solver's orientation filter only removed *reflections*, not this rotation, so every hypothesis
appeared twice and sometimes the twin won, with every template point on its antipode (per-point error **2377 px with
identical drawn lines**). Fix: `_filter` requires +Y up, and every result is passed through `canonicalize_H` (near touchline
at the bottom). Effect on synthetic frames (12 per level, top-1 with < 10 px error at 1920 px):

| Dirt level | before | after |
|---|---|---|
| clean | 25% (p50 2371 px) | **83%** (p50 1.3 px) |
| light (player holes, small residue, spurious lines, 10% dropout) | 25% | 58% |
| heavy | 8% | 33% |
| unremoved player blobs | 25% | 50% |

The chamfer generator was unaffected (its poses are already canonical), and the "best of 8 candidates" oracle no longer beats
top-1: the earlier ties "0.857 vs 0.857", blamed on sliding along a line, were twins. **Any synthetic figure or stored H
computed before this fix may be the twin: canonicalise before comparing per point** (line metrics are unaffected).

## 2. Line mask baseline - `lab/soccer/evaluation.py::etapas_mascara`

Settings fixed by the owner on 2026-09-20 as the reference for other videos (everything else default):

* Play region: **robust Gaussian** (`core.surface.robust_surface`, `surface="grass"`), chi2 = 10.17.
* Lines: local Lab responses (window 61 px at 1920), thresholds **a+ ≥ 5, b- ≥ 5, L+ ≥ 8.5**, combined with **OR**,
  clipped to the region. Lines come out clean and straight, but **players also pass the filters**.
* Player removal (ad hoc): morphological opening (~9 px at 1920); compact, taller-than-wide components that survive are
  players and get subtracted (dilated). Near-camera lines are thicker than a distant player, so thickness alone is not
  perspective-invariant.
* **Elongation rule** (`elong_max = 8`): "opening + component taller than wide = player" deleted the whole (thick,
  vertical) halfway line in frames 1500 and 2400. Elongation (sqrt of the covariance eigenvalue ratio) is 32 and 22 on
  the line versus median 2.2-2.6 and max 4.4 on players; 8 sits in that gap. **Threshold from 3 frames of one video,
  unvalidated elsewhere.**
* Fixing the mask did **not** fix the solver in frame 1500 (chamfer 0.217→0.242, still ~25 px off; intersections got worse
  0.344→0.277). Vertical/horizontal gradient alone does not separate players from lines (a player's sides light the
  horizontal gradient); it is useful to group lines into families.
* A close-up (frame 5100: goalkeeper + stands) breaks the premise: mask 26% and full of crowd. A per-frame "is this a
  wide field shot?" filter is missing.

## 3. Field solver - `lab/soccer/field_solver.py`

Three hypothesis generators sharing one length-weighted Dice score and one refinement:

| Generator | Idea | Observations |
|---|---|---|
| `search_pose` (chamfer) | grid of pinhole poses scored against the mask | ~8 s/frame; clean-mask median error 37 px (n=2) |
| `search_lines` | detected lines → quadruples → 4 point correspondences | ~5 s/frame; clean-mask 1-2 px, 100% < 10 px (n=2); best generator on clean masks |
| `search_ellipse` | centre-circle ellipse as a conic → pose | see §5 |

Design decisions that were measured (details in the module docstrings):

* **Length of mask lines** = sum of 1 / local width. A 1 px skeleton gave a tangle of loops on noisy masks (length ~10x
  too big) and counting ridge maxima underestimated ~35%.
* **Pinhole penalty**: a free 8-parameter H can score as well as the truth while deforming into something no camera
  produces (equal score to the truth at 1500-7800 px error). `core.camera.pinhole_residual` penalises that.
* **tau cascade** (0.04w → 0.015w → 0.006w → tau) in refinement: a wide tau smooths the score and recovers 100+ px of error;
  only the last pass uses fine sampling.
* **Point evidence** (`set_evidence`): without a term from the detected ellipse/corners, refinement on a dirty mask (low
  Dice) dragged away hypotheses the ellipse had already got right. With ideal evidence in synthetic tests it beats chamfer
  (clean: median 27 → 5 px). In real frames it fixes f2400 and does not regress f900/f3000; f1500 has neither ellipse nor
  corners and still fails.
* Refining **from the truth** (the ceiling of the refinement) reaches 0.5-0.75 px.

Earlier synthetic numbers (12 frames per level, top-1 error < 10 px at 1920; chamfer was unaffected by the twin fix):
chamfer clean 42% / light 0% / heavy 0%. Failure modes measured on the real clip (`soccer.mp4`, checked by eye): line
intersections get the area + arc shots right (f900, f3000); chamfer gets the centre circle + halfway line shot right (f1500,
~15-25 px off); close-ups are garbage with scores 0.09-0.24. Cost is ~13-18 s per 960 px frame. Known weaknesses:

1. The Dice penalises template lines the mask lacks (the halfway line was missing from the f1500 mask) and rewards
   hypotheses that draw less.
2. Chamfer is blind to sliding along a line (wrong hypotheses scoring like the truth, "0.857 vs 0.857" - later found to be
   partly the twin bug of section 1).
3. Refining from the truth stays at 1-3 px on the older setup (0.5-0.75 px measured after the port).
4. The first synthetic generator punched holes in player blobs and sank the truth's score from 0.9 to 0.1: a generator
   artefact, not a solver property.

Equivalence check made during the port: the original `soccer_eval.py --sintetico --n 2 --seed 0` and the new
`evaluation` module print **identical tables** for all four dirt levels, and the old `Campo` / new `FieldSolver` return
bit-identical H for `search_lines` and `search_pose`.

## 3b. The broadcast camera looks like a fixed-centre PTZ (key finding, 2026-09-21)

Decomposing each solved H as K [r1 r2 t] (focal from column orthogonality, principal point centred) gives nearly the same
camera centre in two frames 84 s apart with different pans: **(-3.2, -68.0, 13.3) m in f900 and (-3.4, -67.7, 13.3) m in f3000**
(0.3 m apart). In f2400 (only the circle, the halfway line and little else) the decomposition is absurd (30, -34, 5.6; pan -85°):
ill-conditioned with few primitives.

Test: fix C = (-3.3, -67.9, 13.3) and search only pan / tilt / focal (3-D grid, 73k poses) with ellipse evidence
(EdgeDrawing / findEllipses). That **fixes f1500 and f2400 by eye** (circle on the circle, vertical halfway line in place, touchline
within ~20 px in f1500), the two cases no 8-DoF solver had solved. Without the ellipse evidence the fixed-C halfway line comes out
diagonal (wrong). C was calibrated on f900 / f3000, which are different from the test frames. **Limits:** one camera, one clip,
2 test frames, judged by eye; the refinement is still a free 8-DoF fit and drifts away from C; C is not calibrated automatically
yet. This is the most promising soccer lead: if the camera centre really is fixed per broadcast, the solver's search space drops
from 8 DoF to 3.

## 4. Line extraction by gradient on the full image - `lab/soccer/gradient.py`

Continuous response R = max(L+/8.5, a+/5, b-/5); Canny edges; individual RANSAC lines (an edge only supports a line if its
gradient is perpendicular to it) or Hough; paired opposite-gradient edges give the painted line's centre. Measured
against the solver's template in f900 and f3000 (reference biased towards the mask; 2 frames; RANSAC varies ~15 pp across
seeds; thresholds tuned on the same 34 lines):

| Variant | recall (f900 / f3000) | precision (f900 / f3000) |
|---|---|---|
| mask + Hough (reference) | 78% / 86% | 100% / 86% |
| gradient, no region, all lines | 67% / 43% | 38% / 17% (stands, boards) |
| gradient with grass region, all lines | 74% / 62% | 87% / 57% |
| + "grass on both sides ≥ 0.3" | 63% / 57% | 100% / 80% |

"Grass on both sides" is what separates (median 0.63 on correct lines vs 0.05). The union with the mask **does not add
recall**. What the gradient route does add over the mask: the centre-circle ellipse (right in f1500, f2400, f3000) and
person boxes (YOLO on CPU, yolo26s at 1280 = 0.5 s).

Other line detectors on R with the mask applied first (recall / precision per frame, f900 and f3000, reference biased towards the
mask): mask + Hough 78/100 and 86/86; field gradient + RANSAC 74/100 and 48/66; `FastLineDetector` (length 20, merged) 67/100 and
71/62 at 8-35 ms (RANSAC takes 2-3 s); `EdgeDrawing.detectLines` similar but a little worse. All raise raw recall (100 / 86%) at
the cost of precision (64 / 43%) unless the "grass on both sides" filter is applied. Corners are almost never detected (0 in 3
of 4 frames), so corner snapping has effectively never fired.

**Grass region before the response** (owner's rule, applied 2026-09-21): the region is applied BEFORE computing R and its
gradient (outside is filled with the median grass first) rather than clipping edges afterwards. Result is mixed: f900 improves
(recall 63→74% at 100% precision), f3000 gets slightly worse (57→48%); seeds vary ±15 pp.

## 5. Ellipse detection for the centre circle

The owner rejected robust `cv2.fitEllipse` ("a fit to already selected points is not a detector") in favour of real
detectors from `cv2.ximgproc` (needs `opencv-contrib-python`, see below). Observations on soccer.mp4:

| | f1500 (full circle) | f2400 | f3000 (clipped by frame) | f900 (no centre circle) |
|---|---|---|---|---|
| `EdgeDrawing.detectEllipses` | exact (centre 510,235; semi-axes 197x41) | misses | misses; spurious r=10 circle | none (correct) |
| `findEllipses` (score ≥ 0.3) | right | right but over-sizes (249 vs ~190) | none | none |
| robust `fitEllipse` (old) | fails | right | right | false positive |

They are complementary; clipped arcs still need handling. **Open problem**: as solver evidence neither fixes f1500 / f2400:
in f1500 the solver assigns the ellipse to a *penalty arc* instead of the centre circle (all three circles measure 9.15 m);
unimplemented hint: the ellipse centre lies on the long vertical detected line (the halfway line) → it is the centre circle.
In f2400 the "ellipse" suggestion wins on score but rotates the rest wrongly while "intersections" is right: **scores with
evidence are not comparable across candidates**.

## 6. Labelling tool - `lab/soccer/labeler.py`

31 named keypoints (`sports/soccer/field.py::KEYPOINTS`) → H by least squares, canonicalised for the half-turn like hockey.
The instability warning threshold (sensitivity > 30 px) is **inherited from hockey and uncalibrated for soccer**. Output is
YOLO-pose with 31 keypoints, provisional (no soccer training pipeline exists).

## 7. Environment trap

`cv2.ximgproc` (findEllipses, EdgeDrawing) exists only in `opencv-contrib-python`. Having `opencv-python` installed next to it
overwrites `cv2.pyd` with a build without ximgproc (happened on 2026-09-20 20:51 via an editable install of `sportcal`).
`pyproject.toml` and `requirements.txt` now both pin contrib. If it recurs: stop servers/kernels, uninstall both, reinstall only
`opencv-contrib-python==5.0.0.93` (the good `cv2.pyd` is 112,898,048 bytes; 86,293,504 is the bad one).

## 8. Open questions

1. No validation set: the first hand-labelled soccer frames (`annotate_val_app`) are the prerequisite for any number that is
   more than an anecdote.
2. A per-frame "wide field shot?" filter (close-ups poison the mask).
3. Comparable scores across generators (evidence terms make them incomparable).
4. Which circle is the centre circle (ellipse centre on the halfway line).
5. Whether the hockey segmentation route (train a line-segmentation U-Net on reprojected masks, then DLT) is worth repeating
   for soccer once labels exist; the hockey results (`hockey.md` §5-6) say it needs homogeneous data.
