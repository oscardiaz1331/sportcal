# Basketball - the candidate "hard" new sport

Why a fourth sport: tennis needs ten labels (tennis.md section 3), but it is one view of one court. The question left
is what a sport with many kinds of view costs with the model we have, before building the template-conditioned model
(ADR 0005). Status: DeepSportRadar indexed and checked by eye (section 2); the conditioned model is built and its
zero-shot row is 0% coverage, fine-tuned on 50 labels it does what the per-sport model needs 200 for (section 4); the
control with a named head on the same backbone is at 18.5 px with 50 labels, so the gain is the conditioning; the Roboflow
NBA set downloaded, not used.

## 1. Which sport has the best public data (survey, 2026-10-02)

Wanted: real footage, labels that fix a per-frame H (ground-plane keypoints or a calibration), enough frames to hold
out whole games or venues for the test, varied views, a template the rules give.

| sport | dataset | frames | footage | labels | access | verdict |
|---|---|---|---|---|---|---|
| basketball | DeepSportRadar Basketball Instants (MMSports 2022) | 728 instants, 15 arenas, 37 LNB Pro A games; split 480 / 164 / 84, 3 arenas held out | fixed 2-5 Mpx cameras of an automated production system, each on a part of the court; not a TV broadcast | exact K, R, T per image (cm, origin at a court corner) | Kaggle (account), CC BY-NC-ND 4.0 or NC-SA (sources differ) | **first choice**: exact labels, venue-held-out test, published metric (MSE in cm) and results (KaliCalib) to compare with |
| basketball | Roboflow `basketball-court-detection-2` | ~1.2k (the trained version; source count not verified) | NBA TV broadcast | 33 hand-placed court keypoints | Roboflow Universe (account) | **second**: the product's domain, but size, games and label quality could not be checked (the site refuses automated reads) |
| basketball | 3DMPB (Pose2UV, TIP 2022) | 10k-13.6k | not confirmed | camera parameters for body meshes; a court frame is not confirmed | GitHub / Google Drive, MIT | unverified: check before counting on it |
| basketball | College Basketball (Sha et al., CVPR 2020) | 640 | broadcast | homographies | not public | out |
| American football | SportsFields (Nie et al., WACV 2021) | 2967 over 5 sports | broadcast | homographies | not public | out |
| American football | Roboflow hobby sets ("Football Field Key Points") | ~430 | not verified | yard-line keypoints, 2 classes | Roboflow (account), CC BY 4.0 | out: small, and yard lines repeat - a point's identity needs the painted numbers, which a named-keypoint model does not read |
| volleyball | Roboflow / Kaggle court keypoints; Sha et al. (470) | ~500 | one view from behind the court | 4-8 keypoints | account | out: one view, the tennis case again |
| rugby | HF `abinazmi/rugby-pitch-keypoints-detection` | ~445 | not verified | YOLO keypoints (a Roboflow export) | open | out: small, unverified |
| athletics | Baumgartner & Klatt (CVPRW 2023) | 10k | synthetic (Unreal Engine) | pinhole | GitHub | out: not real footage |

American football has no public academic set; the one paper that covers it (Nie et al.) reports it as the hardest of
its five sports because the field looks the same everywhere.
**Decision (proposed):** basketball. DeepSportRadar first for the label-count curve (10 / 50 / 200 / all, test on the
held-out arenas, our px error next to the published cm metric); the NBA broadcast set second, as the check in the
product's domain, once its content is looked at.
**Caveats:** neither basketball set is large (hundreds to ~1.2k frames), so the curve stops early; DeepSportRadar's
cameras are fixed, so "many kinds of view" there means many arenas and camera placements, not pans and zooms; both
downloads need the owner's account; the non-commercial licence of DeepSportRadar rules it out of a product model.
Sources: arXiv 2404.09807 (table of calibration datasets), github.com/DeepSportradar/camera-calibration-challenge,
arXiv 2209.07795 (KaliCalib), github.com/roboflow/sports, github.com/boycehbz/3DMPB-dataset.

## 2. DeepSportRadar: template and per-frame H index - `lab/basketball/deepsport_h.py`, `sports/basketball/court.py` (2026-10-04)

**Data on disk** (owner's download, `datasets/basket`): 728 frames = 364 instants x 2 fixed cameras, 15 arenas, 1624 x
1234 px in most (two arenas have larger frames: 3115 x 1752 and 3652 x 2054 after padding). Each frame has a pinhole
calibration (K, R by rows, T, centimetres, origin at a court corner, z down); `-R^T T` reproduces the stored camera
position to 0.01 cm, which fixes the reading of R. All 728 are marked valid.
**Template:** FIBA 28 x 15 m centred on the court: 26 keypoints (court and lane corners, the three-point stretches and
the top of the arc, the centre circle on the halfway line), 15 straight lines and the arcs for drawing, 56 channels in the
model. `basketball-fiba` in the registry; world frame in metres like tennis.
**Index:** H is computed from the calibration (`calib_to_H`: pixel ~ K (R X + T), X in cm from the corner), not fitted.
The model assumes 16:9 and would crop the bottom quarter of a 4:3 frame, so the frames are copied to
`datasets/basket_169` with black bars at the sides and H shifted (`pad_to_16_9`; `models/` untouched). Splits are by
arena: the challenge's three test arenas (Caen, Limoges, Roanne; 84 frames, as the challenge publishes) and the two
arenas with the lowest name hash of the rest as `dev` (Blois, Strasbourg; 94 frames); train 550. No frame was dropped
by the plausibility gate. Pixel errors are reported at 1920 px of the padded frame, so they are 0.88x what the original
width would give.
**Check by eye:** the template drawn on 24 frames (one or two per arena, all 15 arenas) and on two frames at full size
(Le Mans, train; Gravelines): sidelines, baselines, lane, free-throw line and circle, the three-point arc and the centre
circle sit on the painted lines in every arena; each camera sees about half the court, so the rest projects outside the
image. The frames are undistorted, as the dataset says: no residual bow was visible. The first drawing showed the
three-point arc and the free-throw circle mirrored (a sign in `court.py`, not in the labels); fixed, and
`tests/test_basketball.py` checks the arc's distance to the basket and its side (mutation-checked).
**Roboflow NBA set** (`datasets/basketball-court-detection-2.v15i.coco`, CC BY 4.0), looked at while at it: 1460 images
but only 850 distinct frames (610 in train, each twice with a brightness change; 124 valid; 116 test), from 23 playoff
games, 33 keypoints with 11-14 labelled per frame, stretched to 1280 x 1280. Its own valid and test splits share 15 of
their 17-18 games with train, so it would have to be re-split by game, and the stretch undone, before any
generalization number means something. The template's 33 point positions are not published with it.

    python -m sportcal.lab.basketball.deepsport_h        # -> datasets/basketball_h.jsonl (728 rows), datasets/basket_169/

## 3. How many labels does basketball need? - `train_kpline --limit`, DeepSportRadar (2026-10-04)

**Question:** tennis needed ten labels (tennis.md section 3) because it is one view of one court. A sport whose cameras
differ from arena to arena: what does the same per-sport model A cost?
**Method:** as tennis: `--limit N` (first N train rows of one shuffle, repeated), the ImageNet encoder, ~3300 steps each
(24 epochs of 550 rows; 33 epochs of 400 rows for N = 200), base keypoints (56 channels), usual augmentation. Train is 550
frames from 10 arenas; scored on the 84 test frames of the 3 held-out arenas (Caen 3 games, Limoges, Roanne), px at 1920
of the padded frame, no gate. The owner ran it from his console (`runs/basketball_curve.cmd`).

| labels | arenas seen | checkpoint | coverage | p50 | p90 | < 10 px | < 25 px |
|---|---|---|---|---|---|---|---|
| 10 | ~7 | last | 7% | 4.3 | 6.0 | 7% | 7% |
| 50 | ~9 | last | 75% | 12.6 | 606 | 31% | 44% |
| 200 | 10 | last | 100% | 5.2 | 11.7 | 83% | 95% |
| 550 (all) | 10 | last | 100% | 4.6 | 9.1 | 90% | 98% |
| 550 (all) | 10 | best on dev | 100% | 4.2 | 8.3 | 93% | 99% |

The best-on-dev checkpoint of the smaller runs is worse than the last (N = 50: 70% coverage, p50 20.5; N = 200: p50 7.5,
< 25 px 77%; N = 10: 7%): dev is two other arenas, and an early epoch can win it by luck. A new sport has no labelled dev
anyway, so the last-checkpoint rows are the honest ones. Dev during training: N = 10 loses coverage as it trains
(61% -> 47%); N = 200 is at p50 2.8-3.8 on dev from epoch 7 on.
**Per arena** (last checkpoint, 200 -> 550 labels, p50, < 10 px): Caen 4.8-6.8 -> 3.8-4.9 px (75-88% -> 94-100%), Limoges 5.9 -> 6.1
px (94% -> 62%), Roanne 4.0 -> 4.3 px (100% both; 6 frames). The worst frames of the 550 run are Limoges (33.6, 17.6, 15.6 px)
and one Caen frame (25.3).
**Reading:** against tennis' ten, basketball needs on the order of two hundred labelled frames for coverage to
reach 100% (50 gives three quarters of the frames, and a wild p90), and the whole 550 only improves precision (< 10 px 83%
-> 90%). Once the model answers, its error is 4-5 px, the same as the sports with thousands of frames: the cost is
recognition across arenas (floor colour, lighting, camera height), not geometry. Ten labels do not work.
**For ADR 0005:** this is the regime the held-out-sport test needs and tennis could not give: 50 to 200 labels is where a
template-conditioned model that starts from hockey + soccer + tennis would have to beat the per-sport row above. Fixed cameras
and half-court views remain; the licence is non-commercial.
**Caveats:** 84 test frames, 3 arenas (Roanne has 6 frames, Limoges 16); one run per N and one subset per N (the 50 run's
p90 of 606 px is one or two frames); the 10 labels come from ~7 arenas and the 50 from ~9, a different mix from
tennis' one-view-per-video; px are 0.88x what the original frame width would give; not comparable to the challenge's cm MSE.
**A corrupted batch again:** the N = 200 run skipped one batch at epoch 24 (`train_kpline.corrupted`; rows from Gravelines,
Nancy, Nantes, ...). The guard worked and the run is unaffected, but it is the first since 2026-09-27 (soccer.md section 24)
and it happened on the owner's machine under a lone job: the cause is still open.

    runs\basketball_curve.cmd        # 10 / 50 / 200 / 550 labels, then test, last and best; ~1 h

## 4. Zero-shot from hockey + soccer + tennis: the template-conditioned model - `lab/common/train_conditioned.py` (2026-10-04)

**Question:** ADR 0005. Does one network that takes the field template as input (`models/conditioned.py`), trained on
hockey (NHL + IIHF), soccer and tennis, find a basketball court it has no labels of? And what does sharing one network
cost the training sports?
**Method:** frame part from A2 (`--backbone runs/kpline/finetune/best_h.pt`), template CNN and query MLP from scratch; 40
epochs of 500 steps at batch 4, a third of the samples per sport (NHL and IIHF pooled as hockey, each row with its own
template), base keypoints for every sport, lr 3e-4 cosine, the usual augmentation; 149 s per epoch, ~1 h 50 min. Checkpoint:
epoch 29, by the mean over hockey, soccer and tennis of the dev p50 (60 dev frames per template; mean 7.7 px); epochs 34
and 39 did not beat it. Basketball was not read during training. Px at 1920, percentages of all frames.

**Zero-shot, basketball `test`** (84 frames, 3 held-out arenas): **coverage 0%**, with and without the gate. It is not the
solver refusing: no point reaches the 0.3 peak threshold. The peak value of the elements in view (p50 0.029, p90 0.057,
p99 0.21) is that of the elements out of view (0.026, 0.052, 0.25): the network does not respond to the court at all.

**Cost of sharing** (gated; coverage / p50 / p90 / < 10 px / < 25 px):

| sport, split | conditioned, shared | per-sport model A |
|---|---|---|
| hockey-nhl `test` (72) | 65% / 9.7 / 20.5 / 35% / 62% | NHL + IIHF pretrain: 75% / 10.3 / - / - / 61% (hockey.md 16) |
| hockey-iihf `dev` (61) | 98% / 7.9 / 15.5 / 72% / 93% | pretrain: 100% / 8.1 / 17.8 / 59% / 97%; A2: 100% / 7.5 / 17.1 / 67% / 97% |
| soccer-fifa `test` (2135) | 77% / 7.1 / 19.8 / 53% / 72% | E1, derived keypoints: 89% / 6.5 / 17.0 / 64% / 84% (soccer.md 25) |
| tennis-itf `test` (1014) | 100% / 1.8 / 3.5 / 98% / 100% | 100% / 1.7 (tennis.md 2) |

NHL `test` per video: nhl10 100% / 10.7, nhl4 82% / 8.5, nhl9 (outdoor) 4%; without the gate 69% / 9.9.

**Per element** (peaks >= 0.3 against the rendered targets, as hockey.md 16):

| weights, frames | points: recall | image error p50 | > 25 px | false / frame | line ends: recall |
|---|---|---|---|---|---|
| conditioned, NHL `test` | 47% | 7.6 px | 13% | 0.03 | 46% |
| model A NHL + IIHF pretrain, NHL `test` | 55% | 7.7 px | 15% | 0.1 | 49% |
| conditioned, IIHF `dev` | 92% | 4.5 px | 4% | 0.02 | 80% |
| model A NHL + IIHF pretrain, IIHF `dev` | 95% | 4.6 px | 4% | 0.0 | 85% |
| conditioned, basketball `test` | 0% | - | - | 0.00 | 0% |

**Does it mix up point names?** Hardly: of the 85 NHL `test` points more than 25 px off, 14 sit within 10 px of another
template point (2% of the ~650 points found); IIHF `dev` 0 of 28. NHL `dev` reads p50 24.1 px during training because its
labels are off, not the model: 56 of its 61 frames are `nhl_prior` fits (11-16 px off, hockey.md 0b), model A's own
pretrain scored 23.7 px on it and A2 16.0, and the same points are 12.2 px from the `dev` labels but 7.6 px from the hand
labels of `test`.
**Reading:**
* Zero-shot fails outright. With three sports (four templates) the network learned its training templates, not how to
  read one: the risk ADR 0005 named. It cannot be told apart here from not recognising the arenas (the per-sport model
  also fails with 10 labels, section 3): the frame features and the queries are both out of distribution.
* Sharing is free for tennis and IIHF. NHL loses 10 points of coverage against model A's pretrain at the same precision:
  it finds fewer points (47% against 55%), having seen each NHL row ~17 times at the chosen checkpoint against 60.
  Soccer loses 12 points of coverage against E1, not separable from the base keypoints (the circle views need the derived
  ones, soccer.md 20) and from ~11x fewer soccer samples (20k against 225k).
* What model A had for NHL and this run does not: the fine-tune on the hand labels (hockey.md 14).

**Few-shot** (`runs\cond_fewshot.cmd`): the zero-shot weights fine-tuned on the same N basketball labels as section 3
(`--limit N`, seed 0), same lr, 24 epochs of 137 steps (~3300 steps, ~15 min each), scored on `test`, last checkpoint, no
gate (coverage / p50 / p90 / < 10 px / < 25 px):

| labels | conditioned, from the zero-shot weights | control: named model-A head on the same frame part | per-sport model A from ImageNet (section 3) |
|---|---|---|---|
| 10 | 96% / 66.5 / 835.7 / 18% / 30% | 39% / 333.7 / 861.5 / 7% / 12% | 7% / 4.3 / 6.0 / 7% / 7% |
| 50 | **100% / 5.3 / 10.9 / 89% / 94%** | 92% / 18.5 / 105.9 / 33% / 54% | 75% / 12.6 / 606 / 31% / 44% |
| 200 | 100% / 5.3 / 11.6 / 79% / 99% | 100% / 7.4 / 19.7 / 67% / 94% | 100% / 5.2 / 11.7 / 83% / 95% |
| 550 | - | - | 100% / 4.6 / 9.1 / 90% / 98% |

The control (`runs\cond_control.cmd`) is model A trained with `train_kpline --init` from the conditioned weights minus the
template CNN and the query MLP, plus an untrained 56-channel head (`runs\kpline-cond\shared\backbone_as_model_a_basketball.pt`):
the same frame part, labels, steps and lr, with named channels instead of template queries.

With the gate: N = 10 69% / 39.8 / 112.6 / 18% / 30%, N = 50 96% / 5.1 / 9.4 / 89% / 94%, N = 200 unchanged. Per arena at
N = 10: Roanne 100% / 4.7, Limoges 100% / 41.2, the three Caen games 92-100% / 107-109; at N = 50 every arena answers all
its frames at p50 3.8-6.8 (one Caen game: p90 86, 65% < 10 px). Dev during training: N = 10 67-77% coverage at p50 ~4;
N = 50 and 200 100% at p50 4.3-6.1. No corrupted batch in the three runs.
* **50 labels are enough.** The conditioned model at 50 labels is where the per-sport model is with 200 to 550 (100%
  coverage, 89% < 10 px, against 75% and 31% at the same 50): about four times fewer labels for a sport with many views.
* **200 labels: no difference.** Both are at 100% coverage and p50 5.2-5.3.
* **10 labels still do not work, and fail worse:** the per-sport model refuses (7% coverage), this one answers 96% of
  the frames and is right on 30%; the gate removes a quarter of the answers, not the error. Only the 6 Roanne frames
  come out right (4.7 px); the Caen games (~108 px) and Limoges (41 px) do not.
  (Correction of the first write-up of this section: it said the model "learned the arena of its labels (Roanne)";
  Roanne is a held-out arena, none of the 10 labels comes from it.)

**The control separates conditioning from backbone** (gated, N = 10 / 50 / 200: 19% / 20.6, 81% / 12.9, 99% / 7.3 coverage / p50):
* **50 labels: the gain is the conditioning, not the shared backbone.** The same frame part with a named head reaches
  92% coverage but only 18.5 px and 33% < 10 px, the numbers of the per-sport model from ImageNet (31% < 10 px); the
  conditioned head reaches 5.3 px and 89% < 10 px. What the pretrained backbone buys is answering (75% -> 92% coverage
  at 50 labels); where the points land comes from the template-conditioned head.
* **200 labels:** the control gets close (7.4 px, 67% < 10 px against 5.3 px and 79%) and the per-sport model from
  ImageNet is as good (5.2 px, 83%): the advantage is small and gone against the plain per-sport model.
* **10 labels:** nothing works; the control is worse than the conditioned model (39% coverage, 334 px).
* The best-on-dev checkpoint was not scored; dev during the control training: N = 10 49-57% coverage, N = 50 64-93%,
  N = 200 99-100% at p50 5-6 px.

**Decision (ADR 0005, section 5 rows):** zero-shot: no (0% coverage). Few-shot: yes, at 50 labels - about four times fewer
than the per-sport model needs, and not explained by the backbone; at 200 labels it is not worth more than the per-sport
model. So a new sport with many camera views can start from this network with ~50 labels instead of ~200, a saving of
~150 labels per sport and nothing at all for a one-view sport (tennis, tennis.md 3). Not enough for the product to
switch: the per-sport route already works with the labels the product sports have, and the held-out sport is
non-commercial licence. Open: more training sports (padel, volleyball) could make zero-shot work; 10 labels is still the
wall.
**Caveats:** one run, one seed and one label subset per N (the 50-label gap, 5.3 against 18.5 px, is large; the 200-label
gap, 5.3 against 7.4 px, is not separable from seed noise); 84 test frames from 3 arenas (Roanne 6, Limoges 16); template
reading and arena appearance are confounded in the zero-shot row; soccer is not like for like (base against derived
keypoints, fewer samples); the per-element numbers come from a one-off script that is not in the repo; IIHF `dev` shares
its videos with IIHF train; every few-shot and control run is scored on its last checkpoint, with dev on 60 (conditioned)
or all 94 (control) of the dev frames.

    python -m sportcal.lab.common.train_conditioned --backbone runs/kpline/finetune/best_h.pt
    python -m sportcal.lab.common.train_conditioned --eval runs/kpline-cond/shared/best_h.pt --sport basketball-fiba --split test [--gate]
    python -m sportcal.lab.common.train_conditioned --eval runs/kpline-cond/shared/best_h.pt --sport hockey-nhl --split test --gate
    runs\cond_fewshot.cmd            # 10 / 50 / 200 basketball labels from the zero-shot weights, then test; ~45 min
    runs\cond_control.cmd            # the control: a named model-A head on the same frame part, same labels
