# ADR 0004 - What the networks output, and one geometric solver for every sport

Status: the contract (section "Decision") is accepted, 2026-09-25; the representation inside it is open and decided by
the experiments below, which are proposed.

## Context

Every calibration network in the project has to answer the same question: what should come out of it? The candidates:

1. **H directly** (8 numbers, or the 4-corner parametrisation).
2. **The camera, K R t** (focal, rotation, position).
3. **Keypoints** of the field template, as regressed coordinates or as heatmaps, turned into H by a solver.
4. **Keypoints plus other elements** (line ends, lines, circles/ellipses) as in PnLCalib, that fix H or yield more
   points.

The choice affects the per-sport models (hockey model A, the soccer model), the product, and the later multi-sport
model that receives the field template as input and should reach a sport it was not trained on.

## What is already measured

| Output | Result | Where |
|---|---|---|
| K R t regressed (6D rotation, centre, focal) from segmentation | Does not learn: val p50 ~340 px, 1% < 25 px, flat train loss. The pinhole parametrisation itself is not the limit (fitted to the true H: p50 4.65 px) | hockey.md 8 |
| Keypoint coordinates regressed (YOLO pose, 56 points) | p50 84.5 px, 2% < 25 px on the 63-frame `test` | hockey.md 14 |
| Keypoint + line-end heatmaps, H by points + lines DLT (model A) | p50 9.2 px on the same 63 frames; A2: p50 8.2 px, 67% < 25 px on the 72-frame `test`, nhl4 end views 88% < 25 px | hockey.md 14, 14g |
| Plus keypoints derived from the circles (B1) | No gain in precision | hockey.md 14c |
| Line segmentation -> DLT (no keypoints) | Precise when it answers, coverage 7-12% | hockey.md 6 |
| Keypoint + line heatmaps with PnL refinement (PnLCalib, soccer) | 3-12 px on our hand labels, including centre-circle-only views where our classical solvers failed | soccer.md 18 |

Two further reasons that do not depend on these numbers:

* A regressed H or pose carries no per-point evidence: no residual, no plausibility check (`is_plausible_view`), no
  keypoint-count gate, no way to know when it is wrong.
* An H is expressed in one template's coordinates. It means nothing for a new template; an image point does.

## Decision

**The networks output image-plane evidence with a confidence; the geometry is solved outside them, by one solver in
`core/` shared by every sport.**

* Network: named keypoints and line ends as heatmaps today; any future primitive (ellipses, class-agnostic segments,
  template-conditioned points) keeps the same shape - image coordinates plus a confidence.
* Solver (`core`): evidence + the sport's template -> H (points + lines DLT, RANSAC) -> K R t where a camera is needed
  (refinement, fixed-centre PTZ, tracking) -> gates (`is_plausible_view`, evidence count) -> naming convention
  (`canonical_mirror`).
* K R t is the solver's parametrisation, not a network output.

This holds whichever representation wins below, so experiments stay comparable: same solver, same gates, same
metrics (coverage, p50, p90, < 25 px, answers > 50 px, with and without the gate), same splits (`test` and `fresh` per
sport).

## Experiments that decide the representation

| # | Experiment | Decides | Cost |
|---|---|---|---|
| E0 | Keypoint + line-end heatmaps per sport (hockey A2, soccer from SoccerNet) | the baseline | done / running |
| E1 | E0 plus circles and arcs as their own outputs (ellipse points or parameters; an ellipse fixes 5 of the 8 dof) | whether PnLCalib's extra elements pay per sport: ~22% of SoccerNet frames have < 4 straight markings; faceoff circles are the most visible hockey marking in many views | medium |
| E2 | Template-conditioned keypoints: each template point enters as a query (normalised world position + type) and gets one heatmap. Train on NHL, test on IIHF with the IIHF template; then hockey + soccer. Designed in ADR 0005 | whether conditioning on the template generalises across field dimensions | medium |
| E3 | Class-agnostic primitives (segments, arcs, intersections) + a solver that matches them to any template | whether the zero-data path works, and at what precision | high |
| E4 | E2 and E3 on a sport kept out of training (e.g. basketball, which has public calibration data) | the end goal: a new sport with zero or few labels | high |
| - | H or K R t regressed directly | already answered (hockey.md 8); only as a row of the portfolio comparison table, on soccer's larger set | low, optional |

Order: E0 -> E1 and E2 (cheap, answer "per sport" and "conditioned") -> E3 / E4 once E2 says whether conditioning is
enough. A mixture-of-experts style shared backbone with per-sport heads is only considered if a shared model loses
accuracy against the per-sport ones; it cannot reach a new sport by itself (no data, no head).

## Consequences

* The per-sport models keep the model-A recipe; the soccer model is its first reuse (`train_kpline --sport soccer-fifa`).
* New geometry lands in `core` (the solver, gates, conventions), never inside a network.
* Every experiment reports on the same splits with the same solver; a result that changes the representation is
  written up in the sport's experiment file and summarised here.
