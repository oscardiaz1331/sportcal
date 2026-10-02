# Basketball - the candidate "hard" new sport

Why a fourth sport: tennis needs ten labels (tennis.md section 3), but it is one view of one court. The question left
is what a sport with many kinds of view costs with the model we have, before building the template-conditioned model
(ADR 0005). Status: survey only, no data downloaded yet.

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
