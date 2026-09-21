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

### 3c. Automatic centre calibration and the fixed-centre solver, measured (2026-09-21)

`lab/soccer/camera_center.py` solves 60 frames spread over the clip with `search_lines`, keeps those with score >= 0.6, pinhole mismatch
<= 0.05 and a plausible position, and takes the median of the decomposed centres. 13 of 60 frames were accepted (the rest are
close-ups or have too few lines). The individual centres are scattered (y from -57.6 to -69.7 m, x from -7.2 to 1.3 m: frames with two
or three primitives are ill-conditioned) but their median, **(-2.75, -68.2, 13.6) m from 10 votes**, lands within 0.6 m of the manual
value of 3b. Lesson: the first run (16 frames) had only 3 accepted frames, and a 2 m outlier gate around the median of 3 discarded
the good ones and returned a centre 3.6 m off; the gate is now 4 m and the tool now warns when fewer than 8 frames voted.

`search_fixed_center` with that centre against the pseudo-GT homographies of f900 / f3000 (grid of pitch points, median pixel distance
in 960-px working coordinates): **7.4 px and 5.2 px**, against 3.1 px and 19.9 px for the 8-DoF line solver on the same frames. On f1500 and
f2400 (no reference) the fixed-centre search scores 0.45 / 0.45 where the 8-DoF line solver scores 0.28 / 0.33. Refine at least the top 4 coarse
winners: refining only the best one failed on f900 / f3000 (score 0.35 / 0.22) because a 2-degree pan step is ~60 px at f/w 1.9, wider than the coarse
tolerance, so the true pose is not always the best node. Widening the coarse tolerance made it worse (a wide tau flattens the scores),
and a finer grid (0.7 deg) gave the same result at 10x the time. A synthetic test (`tests/test_soccer_solver.py`) shows a 25 m error in the
centre costs >= 5x in accuracy; sensitivity on real frames: 0.5 m ~ 4-5 px, 5 m ~ 24-61 px.

**Caveats:** one clip, one camera, two frames with a reference, and those frames are near (not identical to) calibration frames 944 / 3047.
The centre needs a wide shot with lines to calibrate; a broadcast that changes camera position would break the assumption.

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

**Line methods integrated in `gradient.procesa(metodo=...)`** (2026-09-21, grass region first, field lines only = "grass on both
sides" filter, single seed; f900 / f3000, reference biased towards the mask):

| `metodo` | recall | precision | time per frame |
|---|---|---|---|
| `ransac` | 78% / 57% | 100% / 67% | 0.8-1.0 s |
| `hough` (Hough on the gradient edges) | 78% / 71% | 100% / 83% | 0.1 s |
| `fld` (`FastLineDetector`, length 20, merged) | 67% / 57% | 100% / 57% | 0.1 s |

With the region applied first, Hough on the gradient edges went from unusable (recall 22% / 14% in the first unmasked test) to the
best gradient-side method, at a tenth of the RANSAC time; it is now comparable to mask + Hough. Two frames and a reference that
favours the mask: read it as "worth keeping", not as a ranking.

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
6. Long-horizon propagation: how often to re-anchor the chained homography on a solved frame, and how to detect a shot cut
   (the mean absolute difference between consecutive frames jumps from ~7 to ~48 at the cut found in §9) so the chain resets.
7. Reuse the previous frame's background mask (warped by the estimated motion) instead of re-running the grass segmentation at every
   step: the segmentation is most of the ~0.2 s per pair.

## 9. Camera motion from the stands - `core/motion.py`, `lab/soccer/camera_motion.py`

**Question:** can the image-to-image motion of the broadcast camera be recovered from the NON-grass region (stands, boards) alone, so a
field homography solved on a rich frame can be carried to frames where the field has almost no lines? For a camera that only pans,
tilts and zooms about a fixed centre (§3b) any two frames are related by a homography whatever the depth of the scene, so a distant
textured background is a valid source. (A KLT tracker on the ice plane was a dead end for hockey - `hockey.md`, specular reflections
on the ice; this one never touches the playing surface.)

**Method:** Shi-Tomasi corners inside the eroded non-grass region -> pyramidal Lucas-Kanade forward and backward -> keep tracks whose
round trip lands within 1 px -> RANSAC homography (3 px) -> refuse the fit below 30 inliers. Players, the score bug and animated
boards move differently and fall out as outliers. Independent check: warp the field-line mask of frame 0 by the estimated motion and
measure how many line pixels land within 3 px of a line pixel of frame `t+gap` (the field lines were never seen by the tracker).
soccer.mp4 from frame 900 (wide shot with the left penalty area), 800 features, 960 px working width.

    python -m sportcal.lab.soccer.camera_motion --video soccer --frame 900 --gaps 1,3,5,10,25,50
    python -m sportcal.lab.soccer.camera_motion --video soccer --frame 900 --chain-step 5 --chain-n 60 --check-every 5

**Result 1 - background motion carries the unseen field lines** (fraction of warped field-line pixels within 3 px, with / without motion):

| gap (frames) | 1 | 3 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| inliers / 800 | 800 | 736 | 721 | 718 | 713 | 512 |
| median inlier residual (px) | 0.11 | 0.15 | 0.07 | 0.07 | 0.12 | 0.15 |
| lines within 3 px, KLT | 97% | 93% | 94% | 93% | 91% | 90% |
| lines within 3 px, no motion | 98% | 96% | 89% | 72% | 59% | 33% |

The camera barely moves in this stretch (2-5 px per 1-10 frames), so the benefit shows from a few frames onwards. The median
line-to-line distance saturates at 0 (the painted lines are several pixels thick), hence the 3 px fraction.

**Result 2 - chaining the field homography** (solved at f900 with the intersection solver, then only multiplied by the estimated
motions; compared at checkpoints with the solver's own solution for that frame: score on that frame's line mask and median distance
between the two projected templates, px at 1920):

| step | checkpoints | distance between templates | outcome |
|---|---|---|---|
| 10 frames x 8 (3.2 s) | every step | 1.7, 2.0, 6.6, 1.9, 4.9, 6.2, 6.8, 2.9 | propagated score 0.66-0.76 vs solved 0.71-0.76 |
| 25 frames x 8 | every step | 2.8, 4.9, then 15.8, 184, 4381 | lost from f975 (inliers 85, 60, 35) |
| 5 frames x 60 (up to 4 s) | every 5 steps | 2.9, 5.2, 4.8, 5.6 | lost at f1016-1018 (below) |

The 25-frame failure is not the pyramid: 5-6 levels and 31-41 px windows give the same counts (inliers at 25 frames: 85 default, 95,
82, 77, 131). Tracks vanish because the camera pans at ~2.3 px/frame with motion blur, so most points of the start frame are gone or
unrecognisable 25 frames later. Short steps (1-5 frames) keep ~670-800 inliers.

The loss at f1016-1018 is a **real shot cut**: the mean absolute difference between consecutive frames jumps from ~7 to 47.6 and the
inliers drop to 0 for ~40 frames (a close-up, no non-grass features to speak of), then the wide shot returns at f1056 with 538 inliers
and the same motion. The estimator failing there is correct.

**Bug found on the way:** at f1050-1054 a fit on 7 and 9 inliers was returned with zoom x16.8 and dx -3416 px. `estimate_motion` now
refuses fits below `min_inliers` (30); the unit test constructs a case where the fit exists but is weakly supported and fails when the
guard is removed.

**Decision:** adopt for propagation with **short steps (<= 5 frames)** and a reset on a shot cut; it is not in the product yet and needs a
cut detector plus a re-anchoring policy (open questions 6-7). Drift stayed at ~5 px (1920) over 4 s against the solver's own solution.
The forward-backward check and RANSAC are both load-bearing: removing either turns a synthetic unit test red
(`tests/test_motion.py`, mutation-checked).

**Caveats:** one clip, one camera, one starting frame. The reference for chaining is the solver's own solution, whose noise is a few
pixels, so distances of 2-7 px are at the noise floor. The line-alignment fraction cannot exceed ~0.94 because the line masks contain
player residue and occlusions. Field-line alignment was not run for the fast-pan stretch beyond the chained checkpoints.

## 10. Are the grass stripes normative? (2026-09-21)

**No rule fixes them.** IFAB Law 1 only asks for a rectangular pitch with continuous lines and, for artificial turf, a green colour; nothing
about mowing. The stripes are not paint: a reel mower bends the blades toward or away from the camera, so the two shades are the same grass
seen with different reflection, and **they swap when seen from the opposite side of the pitch**. Direction, width and pattern (straight, diagonal,
checkerboard, circles) are the groundsman's choice for looks, turf wear and TV, and change from week to week. Secondary sources say some leagues
regulate it (Serie A: cutting height and rectangular parallel bands, for television yield) but I found no primary text, so treat that as unverified.
A rotary mower or a single-direction cut gives no stripes at all.

Measured on our two clips (L channel of the grass region, perspective trend removed, Otsu split into two classes; an upper bound because
players and line residue add to it): `soccer` shows two shades ~11-16 L apart, `soccer2` 9-25 L apart, and the gap changes with the view
angle inside the same clip (soccer2: 24 L at f400, 9 L at f4500). Both stadiums are striped, but with different width and contrast.

**Consequences:** the grass model must accept two shades of green (the robust Gaussian with chi2 10.17 does), the mask must never rely on stripes,
and stripe edges must not be read as field lines (the "grass on both sides" filter of §4 is what protects against that). Stripes are at best a soft
cue (their edges are usually parallel to the pitch axes) that cannot be assumed; not used.

Sources: theifab.com/laws/latest/the-field-of-play, killingley.co.uk (mowing patterns), brightview.com (striping), archysport.com (Serie A).

**soccer2 (Real Madrid - Barcelona broadcast, 8200 frames):** the automatic centre calibration accepted only 2 of 60 frames (centres 6 m apart, spread 6.4 m,
z ~9-10 m against 13.6 m in `soccer`), so no centre was saved for it. Solver scores on sampled frames were 0.32-0.61: this broadcast is closer and
frames rarely show the two whole area boxes. Whether the fixed-centre idea holds here is not established.

## 11. KLT on the painted lines, line refinement per step, and labelling with an ellipse (2026-09-21)

**KLT on the field lines: no.** `camera_motion --region field` takes features from the play surface with its lines closed in. On `soccer` f900
it finds ~100 corners (against 800 in the stands; grass has almost no texture and a painted line only fixes the position across it, the
aperture problem), 43 survive RANSAC at gap 5 and the fit already agrees with "the camera did not move" (line alignment 0.89 = identity,
dx -2.4 px where the stands say +4.8), and at gaps 10 and 25 fewer than the 30 inliers needed remain, so there is no motion at all. `--region all` is
indistinguishable from the stands alone (the corners are stand corners). The stands and boards stay the motion source.

**What does help: pulling the chained H back onto the lines at every step** (`FieldSolver.refine(H, taus=(0.015, 0.006))`, ~0.6 s a step). Chain
from f900, 5-frame steps, 20 steps, distance to an independent re-solve of that frame / line score: unrefined 5.6 px and 0.61, refined
3.5 px and 0.70 (the re-solve scores 0.70). The score gain is partly circular (refinement maximises that score), the distance to the re-solve
is the independent part. It is on by default in the viewer. One clip, one start frame.

**Viewer:** the KLT zone plays the tracking over the clip (`track_video`, `render_step`): from the current frame and template, step by step,
green = background points supporting the step, yellow = carried template; stops on its own when the background is lost.

**Labelling with the centre-circle ellipse** (`core/circle.py`, `lab/soccer/labeler.py::ajusta_elipse`). The outline of the circle is a conic:
5 of the 8 numbers of H. The remaining 3 are a rotation about the circle centre and two "boosts" (the stabiliser of the conic), so **the ellipse
plus 2 point clicks** (field centre and a circle x halfway-line crossing) fix the field; more clicks over-determine it and the ellipse then refines. The
mirror-image family is excluded by orientation (a real camera never sees the mirror), which leaves only the 180-degree twin the labeller already
canonicalises. If all the clicked points lie on the halfway line the fit can be ambiguous; the app says so and asks for a point off that line.
Checked only on synthetic scenes (`tests/test_circle.py`: exact with 2 clicks and a clean outline, within a few px with 1 px of click noise, and
better than a 4-point DLT when the four clicks are poorly spread); **not measured yet on hand-labelled real frames**.

