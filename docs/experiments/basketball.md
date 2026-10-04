# Basketball - the candidate "hard" new sport

Why a fourth sport: tennis needs ten labels (tennis.md section 3), but it is one view of one court. The question left
is what a sport with many kinds of view costs with the model we have, before building the template-conditioned model
(ADR 0005). Status: DeepSportRadar indexed and checked by eye (section 2); the Roboflow NBA set downloaded, not used yet.

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
