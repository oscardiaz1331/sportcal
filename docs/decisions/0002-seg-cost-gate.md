# ADR 0002 - The refinement-cost gate is not a sufficient safety net

Status: accepted, evidence in [experiments/hockey.md](../experiments/hockey.md) (sections 6-7)

## Context

`solve_from_probs` (segmentation -> DLT -> non-linear refinement) refuses unless >= 5
correspondences are in view (dof >= 10). A second signal, the median refinement residual
("cost"), was proposed as an accept/reject gate for auto-labelling and for the product.

## Evidence

| Measurement | Result |
|---|---|
| Cost vs real error, 9 frames (IIHF) | r = 0.97 |
| Same, large sample (~955 reference frames) | r = 0.35 (IIHF), r = 0.31 (NHL) |
| NHL, keep the 50% lowest-cost frames | p90 error still ~487 px |
| Visual QA of 24 auto-labelled frames, 6 videos | nhl5/7/8/10: 15/16 good; nhl6 (amateur rink, wrong template) 0/4; nhl9 (outdoor game) 0/4 - cost did not flag them |
| Random frames of the hand-labelled val (`valh`) | DLT coverage 0% (reference net) / 4% with garbage H (UDA net) |

## Decision

* The cost gate is a *pre-filter*, never the only line of defence. Anything auto-accepted through
  it must be sampled visually **per source video** before being merged into training data.
* In the product, `SegDltEstimator` may only sit in front of a fallback (ADR 0003), never alone.
* `--max-costo` in `lab/hockey/auto_label_seg.py` stays at 0.001, but treat its output as
  "candidates", not "labels".

## Why it fails (hypothesis, not proven)

A homography that is internally consistent but built on the wrong geometry (wrong template) or
mis-segmented boards still yields a small residual in normalised space, exactly as the algebraic
residual failed to pick the board sign (0/27) earlier.
