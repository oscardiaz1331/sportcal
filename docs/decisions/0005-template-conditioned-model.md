# ADR 0005 - E2: one keypoint + line model that takes the field template as input

Status: proposed, 2026-09-27. Deciders: the owner. Refines experiments E2 and E4 of ADR 0004 (tennis replaces
basketball as the held-out sport, `tennis.md`).

## Context

Model A (`models/kpline.py`) has one output channel per named template element: 56 points + 2 x 9 line ends for
hockey, 39 + 2 x 17 for soccer (E1), 14 + 2 x 9 for tennis. A new sport needs a new head and labels. The end goal of
ADR 0004 is a sport with zero or few labels, which needs a network that is told *which* elements to find instead of
having them baked into its last layer.

What the code and data already give:

* The contract of ADR 0004 holds unchanged: the network outputs one heatmap per requested element, the shared solver
  (`estimate_H` -> `solve_points_lines`, gates, `canonical_mirror`) turns them into H. Only the head changes.
* `train_kpline` is already per-row template aware: targets are rendered from each row's H and its own template
  (`Frames`, `render_heatmaps`), `evaluate` solves each row with its own template.
* Train frames per template: NHL 614, IIHF 572 (one league, SHL), soccer 8831, tennis 6612.
* **NHL and IIHF share their 56 point names and 9 lines**; only the coordinates differ. A named-channel model trained
  on NHL alone can already be scored on IIHF by handing the IIHF template to the solver
  (`--sport hockey-iihf --eval ...`, the channel counts match). So NHL -> IIHF tests "the same markings at other
  distances", which model A may pass without any conditioning; it does not test reading a template.
* Four real templates in total. A conditioned model trained on two or three of them can memorise them as lookup
  tables; only a held-out sport measures whether it reads the template.

## Decision

1. **Controls before code.** (a) A2 on IIHF `dev` with the IIHF template: the in-distribution bound, no code.
   (b) Model A pretrained on NHL rows only, scored on all 633 IIHF frames: needs one training filter
   (`--templates hockey-nhl`). If (b) is within ~2x of (a), NHL -> IIHF is answered by named channels and E2 goes
   straight to the held-out sport; if not, the conditioned model is also run NHL -> IIHF.
   **Result (2026-10-02, hockey.md section 16):** (a) 100% coverage, p50 7.5 px; (b) 25-43% coverage, p50 21-37 px
   gated - no transfer. But the points (b) finds on IIHF are where the markings are (7.9 px, as at home); it finds
   19% of them, and only 35% on NHL itself. The failure is recognition (614 rows, another broadcast), not rink
   geometry, so conditioning on the template would not repair it: NHL -> IIHF is dropped as a test of conditioning
   and E2 goes to the held-out sport, where the training sports have 11k (soccer) and 1.2k (hockey) frames.
2. **Conditioning: a template raster and a dynamic head** (option C below). The template's painted lines are drawn
   top-down into a fixed canvas (field box fitted, aspect kept, canonical orientation); a small CNN encodes it. A point
   query is the raster feature bilinearly sampled at the point, plus its normalised position; a line query is the
   mean of the features sampled along a -> b plus both ends' positions. An MLP turns each query into a 1x1 kernel
   over the U-Net's per-pixel features (two kernels for a line: the a-side and b-side visible ends, the order
   `visible_ends` already uses). Queries do not see each other, so the output does not depend on their order or number.
3. **Data mixing.** Equal probability per sport (`WeightedRandomSampler`, weight 1 / frames of the row's sport), an
   epoch is a fixed number of steps. Batches mix sports, queries padded to the largest template and masked in the loss
   (single-sport batches would skew the BatchNorm statistics towards one sport per step). Same focal loss, same
   augmentation.
4. **Checkpoint selection** by the mean of the per-sport dev p50 over the *training* sports only. The held-out sport's
   dev is never read during training or threshold choice.
5. **The experiment**, all on tennis `test` (1014 frames, ADR 0004 metrics, with and without the gate):

   | Row | Model | Tennis labels |
   |---|---|---|
   | upper bound | model A, tennis only | all 6612 |
   | cost of sharing | conditioned, hockey + soccer + tennis | all |
   | zero-shot | conditioned, hockey + soccer | none |
   | few-shot | conditioned (zero-shot weights) fine-tuned | 10 / 50 / 200 |
   | few-shot control | model A backbone from hockey + soccer, new named tennis head | 10 / 50 / 200 |

   The few-shot control is the lazy alternative: if a new named head on a shared backbone matches the conditioned
   model at 50 labels, the conditioning is not worth its code for adding sports, and only the zero-shot row argues for it.

## Options considered

| | Option | Zero-shot | Cost | Verdict |
|---|---|---|---|---|
| A | Shared backbone, one named head per sport | no (no head without labels) | lowest, exists | the few-shot control |
| B | Hand-made query descriptor (normalised position + type from a fixed vocabulary: dot, T, L, cross, line end) -> MLP -> dynamic kernel | yes | low | the fallback if C does not train |
| C | Template raster encoded by a CNN, sampled at each element -> dynamic kernel | yes | low-medium | **chosen** |
| D | Transformer decoder, queries cross-attend image features (DETR-like) | yes | high: memory on 8 GB, slow to converge on ~17k frames | only if C and B fail with a sign that queries need image context |

B's weakness is the vocabulary: normalised position means different things on a 105 m pitch and a 24 m court, and
every new marking kind needs a new type. C learns the local shape from the drawing (a service-line T and a penalty-box
T look alike in the raster) and keeps the global position that separates symmetric elements. C falls back to B by
replacing the raster CNN with a lookup, so B needs no separate design.

## Consequences

* Code: the conditioned head and the template raster are sport-agnostic, so they live in `models/` (the raster from
  `Sport.polylines`, next to `kpline.py`); `estimate_H` and the solver are reused as they are. The trainer gains
  `--templates` (control b), the sampler and the padded queries; it is edited in a worktree, never while a training
  runs (Windows DataLoader workers re-import it).
* A result that the conditioned model memorises (zero-shot tennis fails while "cost of sharing" is fine) is an
  answer, not a bug: with three training templates that is the expected risk, and the few-shot rows then decide
  whether it still pays. More templates (IIHF already counts; padel, volleyball later) are the upgrade.
* Tennis views are homogeneous (`tennis.md` section 1): the zero-shot row measures reading a new geometry in a
  familiar kind of view, not new camera placements.
* Numbers go to `docs/experiments/tennis.md` (held-out rows) and `hockey.md` (NHL -> IIHF); this ADR gets the verdict.

## Action items

1. [x] Control (a): `train_kpline --sport hockey-iihf --eval runs/kpline/finetune/best_h.pt --split dev --gate`.
2. [x] Tennis per-sport model (the upper bound): test p50 1.7 px, 100% coverage (`tennis.md` section 2).
3. [x] `--templates` filter, control (b) on NHL only, scored on IIHF train + dev (hockey.md section 16).
4. [ ] Template raster + dynamic head in `models/`, one test: the raster sample of an element depends only on the
       template, and permuting queries permutes the outputs.
5. [ ] Multi-sport trainer (sampler, padding, selection on training sports), then the five rows of the table.
