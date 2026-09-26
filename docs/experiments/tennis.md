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
frame - only an overlay check can (to do once the images are extracted). 1 of 8841 frames gives an H no camera can give.
**Recipe check:** targets rendered from 300 labels decode back through `train_kpline.estimate_H` (14 points + 18 line
ends = 32 channels) at p50 0.5 px, max 1.2 px.

    python -m sportcal.lab.tennis.tennis_h                                 # -> datasets/tennis_h.jsonl
    python -m sportcal.lab.hockey.train_kpline --sport tennis --phase pretrain --epochs 20
    python -m sportcal.lab.hockey.train_kpline --sport tennis --eval runs/kpline-tennis/pretrain/best_h.pt --split test [--gate]

The per-sport tennis model is the upper bound the template-conditioned model is compared against when tennis is left out
of its training.
