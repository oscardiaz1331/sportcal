# Tennis - the "new sport" of the multi-sport plan

Why tennis: the final experiment of the plan (ADR 0004) needs a sport the template-conditioned model has never been
trained on, with labels good enough to score it. Candidates were surveyed on 2026-09-25 (basketball DeepSportRadar and
Roboflow, volleyball, padel, badminton, American football...); tennis has the largest real-broadcast set whose points all
lie on the ground plane, and the ITF rules give the template.

## 1. Data and per-frame H index - `lab/tennis/tennis_h.py`, `sports/tennis/court.py` (2026-09-26)

**Source:** TennisCourtDetector (github.com/yastrebksv/TennisCourtDetector; mirror
`hf download Gholamreza/tennis_court_keypoints_dataset --repo-type dataset --local-dir datasets/tennis_court`, 7.3 GB zip,
no licence in the original repo, MIT on the mirror). 8841 frames at 1280x720 from 498 YouTube highlight videos (hard,
clay and grass courts), one frame every 50, 14 court keypoints each. The labels come from a classical court detector with
bad results removed by hand, not from clicks.
**Template:** ITF doubles court 23.77 x 10.97 m, origin at the centre of the net, X along the length; 9 straight lines
(baselines, doubles and singles sidelines, service lines, centre service line), the net and the 10 cm centre marks left
out. The 14 points are the dataset's, in its order.
**Index:** one H per frame by least squares on the 14 points, renamed with `core.camera.canonical_mirror` (end views: the
camera sits behind the +X baseline). The published train/val split shares 434 of 498 videos, so frames are re-split by a
stable hash of the video id (80 / 10 / 10 train / dev / test).
**What the fit says about the labels:** every frame fits one homography to within 1.6 px (max over the 14 points, at
1920; median 0.55). That is how a detector that fits a court model would label, so the residual cannot reveal a wrong
frame - only an overlay check can. 1 of 8841 frames gives an H no camera can give.
**Overlay check** (2026-09-27): the court drawn from the index H on one random frame of each of 47 random videos: every
one sits on the painted lines at contact-sheet resolution (480 px wide), on clay, grass and hard courts of every colour.
The views are homogeneous: nearly all are the main camera behind a baseline with the whole court in frame (one zoomed
serve view in 47), so the line ends mostly coincide with the keypoints and tennis is an easy target for a per-sport
model; for the generalization test that is the point - the court geometry is new, the kind of view is not.
**Index built:** 8840 frames (train 6612, dev 1214, test 1014), all images present and complete.
**Recipe check:** targets rendered from 300 labels decode back through `train_kpline.estimate_H` (14 points + 18 line
ends = 32 channels) at p50 0.5 px, max 1.2 px.

    python -m sportcal.lab.tennis.tennis_h                                 # -> datasets/tennis_h.jsonl
    python -m sportcal.lab.common.train_kpline --sport tennis-itf --phase pretrain --epochs 20
    python -m sportcal.lab.common.train_kpline --sport tennis-itf --eval runs/kpline-tennis-itf/pretrain/best_h.pt --split test [--gate]

The per-sport tennis model is the upper bound the template-conditioned model is compared against when tennis is left out
of its training.

## 2. Per-sport tennis model - `runs/kpline-tennis-itf/pretrain` (scored 2026-10-02)

**What:** the weights in `runs/kpline-tennis-itf/pretrain` (`best_h.pt`, 32 channels, files dated 2026-09-26; the run's
log and its number of epochs were not kept), scored on the current index:

| split | frames | coverage | p50 | p90 | < 10 px | < 25 px |
|---|---|---|---|---|---|---|
| test | 1014 | 100% | 1.7 | 3.3 | 98% | 100% |
| dev | 1214 | 100% | 1.7 | 3.1 | 98% | 100% |

The plausibility gate changes nothing on `test`. A 340-step smoke run already reached p50 2.0 px on `dev` (2026-09-27):
as section 1 expected, homogeneous views make tennis easy for a per-sport model.
**Decision:** this is the upper bound for the held-out-tennis rows of ADR 0005; no further per-sport training needed.
**Caveats:** the labels come from a court detector (section 1), so 1.7 px is agreement with that detector, not with hand
clicks; there are no `fresh` tennis labels. The run predates the 2026-09-27 rebuild of the index (same videos, split by a
stable hash of the video id).

    python -m sportcal.lab.common.train_kpline --sport tennis-itf --eval runs/kpline-tennis-itf/pretrain/best_h.pt --split test --gate

## 3. How many labels does tennis need? - `train_kpline --limit` (2026-10-02)

**Question:** before building the template-conditioned model (ADR 0005), what does a new sport cost with the model we
have - a per-sport model A trained on a few labels?
**Method:** `--limit N` keeps the first N train rows of one shuffle (seed 0; the 10 are among the 50, the 50 among the
200) and repeats them, so every run makes the same ~3300 steps (two epochs of the full set's length) from the ImageNet
encoder, with the usual augmentation. None of the N frames shares a video with `dev` or `test`. Scored on `test` (1014
frames, 58 videos), px at 1920:

| labels | videos | checkpoint | coverage | p50 | p90 | < 10 px | < 25 px |
|---|---|---|---|---|---|---|---|
| 10 | 10 | last | 100% | 2.0 | 3.8 | 98% | 100% |
| 50 | 47 | last | 100% | 1.7 | 3.3 | 98% | 100% |
| 200 | 138 | last | 100% | 1.9 | 3.6 | 98% | 100% |
| 6612 (section 2) | 380 | best on dev | 100% | 1.7 | 3.3 | 98% | 100% |

The last checkpoint is the honest row (a new sport has no labelled `dev` to choose with); the best-on-dev checkpoints
score the same to 0.1 px. After the first epoch (~1650 steps) the 10-label run was already at p50 2.4 px on `dev`.
**Reading:** ten labelled frames give what 6612 give. Tennis broadcasts are one view of one court, and the exact camera
augmentation (`ptz_warp`) covers the pans and zooms around it. The differences between 10, 50 and 200 (2.0 / 1.7 / 1.9)
are one run each and within noise.
**Decision:** for a sport like tennis a new sport costs a template and about ten labels, so the template-conditioned
model has no practical case here; and tennis cannot separate few-shot methods (ADR 0005's few-shot rows would all read
~2 px). What is left for the held-out-tennis test is the zero-shot row only.
**Caveats:** the labels and the test "truth" come from the same court detector (section 1), so this is agreement with
it; one seed, one subset per N; the 10 frames come from 10 different videos, the favourable way to spend ten labels.
This says nothing about sports with many kinds of view: hockey with 614 NHL rows still finds a third of its points
(hockey.md section 16), and soccer was trained on thousands of frames.

    python -m sportcal.lab.common.train_kpline --sport tennis-itf --phase pretrain --limit 10 --epochs 2 --eval-every 1 --tag=-n10
    python -m sportcal.lab.common.train_kpline --sport tennis-itf --eval runs/kpline-tennis-itf/pretrain-n10/last.pt --split test
