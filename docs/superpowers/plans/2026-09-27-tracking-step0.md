# Tracking step 0 (do the stock trackers swap players?) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how often ByteTrack and BoT-SORT swap two hockey players' identities within a shot, using indirect
signals in rink metres, and decide by a pre-registered rule whether a metric-space tracker is worth building.

**Architecture:** One GPU pass per video caches everything (cuts, the product's H, COCO person boxes, jersey
histograms, the two stock trackers' IDs) in `runs/tracking/<video>.npz`. Everything else runs offline on the cache:
feet projected to the rink, four signals per track (jump, impossible turn, team change, fragmentation), H jumps
separated from swaps, contact sheets for labelling by eye, precision and real swaps per minute from the hand labels.
The jersey-colour team code moves from the demo script to `core/teams.py` so lab and demo share it.

**Tech Stack:** Python 3.12, numpy, scipy (`ndimage.median_filter`), opencv-contrib, ultralytics (`yolo26m.pt` COCO),
roboflow `trackers` (`ByteTrackTracker`, `BoTSORTTracker`), supervision, scenedetect, the product pipeline
(`product.hockey.build_pipeline`).

**Spec:** `docs/experiments/hockey.md` section 15 ("Player tracking in metres, step 0"). Read it before any task.

## Global Constraints

- Code, comments, docstrings in English; docstrings never quote experiment numbers (link hockey.md section 15).
- Layers: `core` imports nothing above it; `lab` may import `product` only lazily inside a function, with a comment
  (precedent: `lab/hockey/eval_yolo.py`). `tests/test_layering.py` must stay green.
- A lab module is import-safe: `main()` only behind `if __name__ == "__main__"`; heavy imports (ultralytics, trackers,
  supervision, scenedetect, product) inside the function that needs them, so `pytest` stays fast.
- Paths from `sportcal/paths.py` (`RUNS`, `ROOT`); commands run from the repo root.
- Stage explicit paths only when committing: the working tree holds the owner's unrelated uncommitted changes
  (CLAUDE.md, ADR 0004/0005, soccer files, `hsv_scan.csv`). Never `git add -A` / `git add .`.
- GPU is 8 GB: check `nvidia-smi` shows no training before a cache run.
- Deliberate simplifications with a known ceiling get a `# ponytail:` comment naming the ceiling and the upgrade path.
- Exact values from the spec: `V_MAX = 12.0` m/s; `A_MAX = 15.0` m/s^2; team confident when distance ratio `< 0.8`;
  COCO `yolo26m.pt`, class 0, `conf >= 0.1`; H from `build_pipeline(None, kpline_weights=runs/kpline/finetune/best_h.pt,
  hold_frames=15, smooth=0.1)`; turn windows 0.5 s each needing >= 80% of their frames observed; team majority over
  1 s, a change needs >= 1 s of each team; flags on one track within 1 s merge into one event; H jump = more than half of
  the tracks seen on a frame, and >= 3, jump on it; swap flags (jump, turn) within 0.5 s of an H-jump frame are dropped;
  `margin` = 2 x the 99th percentile of the distance between a track's position and its median over 0.1 s; 20 random
  events per sample.
- Videos: development `nhl4 nhl7 nhl10` (margins may be adjusted here); final `nhl11 nhl12 nhl13 nhl14` (`fresh`:
  nothing adjusted, `--margin` required); `nhl9` excluded.
- Fast suite: `venv/Scripts/python.exe -m pytest` (~10 s) after touching `core/` or `product/`.

## Review Focus

- Tracker IDs restart at every cut (trackers are recreated): the same ID in two shots must be two tracks -> pinned in
  Task 3 (`test_observations_keep_measurable_tracked_feet_only_and_split_ids_by_shot`).
- Frames without H (refusals), feet off the rink, and untracked boxes (ID -1) must be dropped, never projected with a
  stale or NaN H -> pinned in Task 3 (same test).
- nhl4 is 30 fps, the others 60: every threshold is in seconds, a normal skater must fire nothing at either rate ->
  pinned in Task 2 (`test_a_skater_at_record_speed_or_on_a_tight_curve_fires_nothing`, parametrized on fps).
- Running `signals` on the `fresh` games without the margin frozen on development would tune on `fresh` -> refused,
  pinned in Task 5 (`test_fresh_games_need_the_margin_frozen_on_development`).
- Re-running `signals` must never wipe the hand labels in `runs/tracking/labels.csv` -> pinned in Task 5
  (`test_labels_file_is_created_once_and_never_overwritten`).

---

### Task 1: Team from jersey colour in `core/teams.py`, shared by the demo

**Files:**
- Create: `sportcal/core/teams.py`
- Create: `tests/test_teams.py`
- Modify: `sportcal/product/hockey_demo.py` (imports near line 29; delete lines 81-123; lines 291, 299-310, 312)

**Interfaces:**
- Consumes: nothing.
- Produces: `jersey_histogram(frame: np.ndarray, bbox) -> np.ndarray (128,) float32 | None`;
  `fit_teams(hists) -> np.ndarray (2, 128) float32`;
  `team_of(hists (n, 128), centres (2, 128)) -> (teams: int array (n,), ratio: float array (n,))`.

- [ ] **Step 1: Write the failing test** - `tests/test_teams.py`:

```python
"""Team from jersey colour (sportcal/core/teams.py): synthetic crops of two shirt colours and one of bare ice."""
import numpy as np

from sportcal.core.teams import fit_teams, jersey_histogram, team_of


def _crop(bgr, rng):
    """Histogram of a 40 x 100 box: white ice around, a noisy shirt colour on the torso band (rows 25-65)."""
    frame = np.full((100, 40, 3), 235, np.uint8)
    frame[25:65, 5:35] = np.clip(np.array(bgr) + rng.normal(0, 12, (40, 30, 3)), 0, 255).astype(np.uint8)
    return jersey_histogram(frame, (0, 0, 40, 100))


def test_two_shirt_colours_split_into_two_confident_teams_and_bare_ice_has_no_team():
    rng = np.random.default_rng(0)
    green = [_crop((40, 180, 40), rng) for _ in range(15)]
    blue = [_crop((200, 60, 30), rng) for _ in range(15)]
    teams, ratio = team_of(green + blue, fit_teams(green + blue))
    assert len(set(teams[:15])) == 1 and len(set(teams[15:])) == 1 and teams[0] != teams[15]
    assert ratio.max() < 0.8
    assert jersey_histogram(np.full((100, 40, 3), 235, np.uint8), (0, 0, 40, 100)) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_teams.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sportcal.core.teams'`

- [ ] **Step 3: Write `sportcal/core/teams.py`**

```python
"""Team of a player from the colour of their jersey: an HSV histogram of the torso, two clusters per clip.

Sport-agnostic: it assumes only that the two teams wear shirts of different colours. Referees in black and white and
bare ice have too little saturation to give a histogram, so they get no team rather than a wrong one.
"""
import cv2
import numpy as np

SAT_MIN = 40                  # below: ice, white boards, reflections, skates - not jersey colour
MIN_COLOUR_FRACTION = 0.15    # a torso band with less colour than this fell on the background, not on the shirt


def jersey_histogram(frame, bbox):
    """(H, S) histogram of the torso band (25%-65% of the box height, below the helmet and above the skates) with the
    low-saturation pixels masked out, or None when too little of the band has colour: a crouched or occluded player
    puts the band on the ice or the boards, and a histogram of those is not a jersey."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    h = y2 - y1
    crop = frame[y1 + int(h * 0.25):y1 + int(h * 0.65), x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > SAT_MIN).astype(np.uint8)
    if mask.mean() < MIN_COLOUR_FRACTION:
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def fit_teams(hists):
    """(2, 128) centroids: 2-means over one clip's jersey histograms. Which team is 0 is arbitrary."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, _, centres = cv2.kmeans(np.asarray(hists, np.float32), 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centres


def team_of(hists, centres):
    """Nearest centroid of each histogram (n, 128): (teams, ratio), ratio = its distance over the other's - near 0 a
    clear call, near 1 an ambiguous one."""
    d = np.linalg.norm(np.asarray(hists, np.float32)[:, None, :] - centres[None], axis=2)
    return d.argmin(1), d.min(1) / np.maximum(d.max(1), 1e-9)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_teams.py -v`
Expected: PASS

- [ ] **Step 5: Mutation check** - in `jersey_histogram` change `if mask.mean() < MIN_COLOUR_FRACTION:` to
  `if False:`, run the test, expect FAIL on the bare-ice assert; restore the line and re-run: PASS.

- [ ] **Step 6: Make the demo use it** - in `sportcal/product/hockey_demo.py`:

  After `from sportcal.paths import RUNS` add:

```python
from sportcal.core.teams import fit_teams, jersey_histogram, team_of
```

  Delete the block from the comment `# por debajo de esto un pixel se considera "sin color"` through the end of
  `def classify_team(...)` (the constants `JERSEY_SAT_THRESHOLD`, `JERSEY_MIN_COLOR_FRACTION` and the functions
  `player_jersey_histogram`, `classify_team`).

  Replace `hist = player_jersey_histogram(frame, bbox)` with `hist = jersey_histogram(frame, bbox)`.

  Replace

```python
                        if len(team_calib_histograms) >= TEAM_CALIB_SAMPLES:
                            calib_data = np.array(team_calib_histograms, dtype=np.float32)
                            best_labels = np.zeros((len(calib_data), 1), dtype=np.int32)
                            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
                            _, _, centers = cv2.kmeans(
                                calib_data, 2, best_labels, criteria, 10, cv2.KMEANS_PP_CENTERS
                            )
                            team_centroids = centers
```

  with

```python
                        if len(team_calib_histograms) >= TEAM_CALIB_SAMPLES:
                            team_centroids = fit_teams(team_calib_histograms)
```

  Replace `team = classify_team(hist, team_centroids)` with `team = int(team_of([hist], team_centroids)[0][0])`.

- [ ] **Step 7: Smoke-run the demo from a temporary directory** (it writes `tracked_out.mp4`, `tracking_log.csv` and
  `hsv_scan.csv` to the working directory; the repo root's `hsv_scan.csv` is the owner's untracked data and must not be
  overwritten):

```bash
D=$(mktemp -d) && cd "$D" && REPO_ROOT/venv/Scripts/python.exe -m sportcal.product.hockey_demo REPO_ROOT/nhl10.mp4 --max-frames 300; cd REPO_ROOT
```

Expected: runs to the end and prints `[equipos] calibrados con 300 muestras en el frame N`.

- [ ] **Step 8: Fast suite**

Run: `venv/Scripts/python.exe -m pytest`
Expected: all pass (including `test_layering.py`).

- [ ] **Step 9: Commit**

```bash
git add sportcal/core/teams.py tests/test_teams.py sportcal/product/hockey_demo.py
git commit -m "core.teams: team from jersey colour, moved out of the demo script so the lab can use it"
```

---

### Task 2: The swap signals (pure functions) in `lab/hockey/track_swaps.py`

**Files:**
- Create: `sportcal/lab/hockey/track_swaps.py`
- Create: `tests/test_track_swaps.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (all arrays numpy; `t` seconds sorted ascending, `xy` metres (n, 2), one track):
  `jumps(t, xy, v_max, margin) -> int array of indices`;
  `turns(t, xy, a_max, fps, window=0.5, min_seen=0.8) -> int array`;
  `team_changes(t, team, hold=1.0) -> int array` (`team` int: 0/1, -1 unsure);
  `h_jump_frames(jump_frames, obs_frames, min_tracks=3) -> int array of frame numbers`;
  `merge(key, t, idx, gap=1.0) -> int array` (`key`/`t` over all rows sorted by (track, time));
  `noise(xy, k) -> float array`; `_tracks(key) -> list[(start, end)]`;
  constants `V_MAX`, `A_MAX`, `TEAM_RATIO`, `TRACKERS`, `SIGNALS`, `FRESH_VIDEOS`, `KPLINE`, `OUT`.

- [ ] **Step 1: Write the failing tests** - `tests/test_track_swaps.py`:

```python
"""Swap signals of tracking step 0 (sportcal/lab/hockey/track_swaps.py, hockey.md section 15), on synthetic tracks."""
import numpy as np
import pytest

from sportcal.lab.hockey.track_swaps import A_MAX, V_MAX, h_jump_frames, jumps, merge, team_changes, turns

MARGIN = 0.3   # m of feet-point noise the jump test tolerates in these tests


def _track(v=(6.0, 0.0), start=(10.0, 5.0), fps=60.0, seconds=4.0, noise=0.03, seed=0):
    """(t, xy) of a skater at constant velocity whose feet point jitters."""
    t = np.arange(int(seconds * fps)) / fps
    rng = np.random.default_rng(seed)
    return t, np.array(start) + t[:, None] * np.array(v) + rng.normal(0, noise, (len(t), 2))


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_a_skater_at_record_speed_or_on_a_tight_curve_fires_nothing(fps):
    t, xy = _track(v=(11.0, 0.0), fps=fps)
    assert len(jumps(t, xy, V_MAX, MARGIN)) == 0 and len(turns(t, xy, A_MAX, fps)) == 0
    theta = 1.5 * t     # 6 m/s on a 4 m radius: 9 m/s^2 of turning, hard but real
    curve = np.c_[20 + 4 * np.cos(theta), 10 + 4 * np.sin(theta)] + np.random.default_rng(1).normal(0, 0.03, (len(t), 2))
    assert len(jumps(t, curve, V_MAX, MARGIN)) == 0 and len(turns(t, curve, A_MAX, fps)) == 0


def test_an_id_moving_to_a_player_one_metre_away_is_one_jump_at_the_swap():
    t, a = _track()
    _, b = _track(start=(10.0, 6.0), seed=1)
    xy = np.r_[a[:120], b[120:]]
    assert list(jumps(t, xy, V_MAX, MARGIN)) == [120]


def test_an_id_moving_between_two_crossing_players_is_an_impossible_turn_not_a_jump():
    t, a = _track(v=(6.0, 0.0), start=(10.0, 5.0))
    _, b = _track(v=(0.0, 6.0), start=(22.0, -7.0), seed=1)     # both pass (22, 5) at t = 2 s, frame 120
    xy = np.r_[a[:120], b[120:]]
    assert len(jumps(t, xy, V_MAX, MARGIN)) == 0
    flags = turns(t, xy, A_MAX, 60.0)
    assert len(flags) > 0 and np.abs(flags - 120).max() <= 8
    assert len(merge(np.zeros(len(t), int), t, flags)) == 1


def test_a_team_change_needs_a_second_of_each_team():
    t = np.arange(240) / 60.0
    team = np.r_[np.zeros(120, int), np.ones(120, int)]
    changes = team_changes(t, team)
    assert len(changes) == 1 and abs(changes[0] - 120) <= 1
    flicker = np.zeros(240, int)
    flicker[100:130] = 1                       # half a second of misread shirts
    assert len(team_changes(t, flicker)) == 0
    unsure = team.copy()
    unsure[::2] = -1                           # unconfident votes do not count
    assert len(team_changes(t, unsure)) == 1


def test_everyone_jumping_on_one_frame_is_the_homography():
    seen = np.repeat(np.arange(100), 4)        # 4 tracks on each of 100 frames
    assert list(h_jump_frames(np.array([50, 50, 50, 70]), seen)) == [50]


def test_flags_within_a_second_on_one_track_are_one_event():
    t = np.arange(240) / 60.0
    assert list(merge(np.zeros(240, int), t, np.array([40, 10, 12, 200]))) == [10, 200]
    two = np.r_[np.zeros(100, int), np.ones(140, int)]
    assert list(merge(two, t, np.array([95, 100]))) == [95, 100]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_track_swaps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sportcal.lab.hockey.track_swaps'`

- [ ] **Step 3: Write `sportcal/lab/hockey/track_swaps.py`** (the module grows in Tasks 3-5):

```python
"""Tracking step 0 (hockey.md section 15): do the stock trackers swap players' identities within a shot?

No hockey tracks with IDs exist here, so swaps are counted through indirect signals in rink metres - a jump no skater
makes, a change of course no skater makes, a change of jersey colour - and those signals are checked by eye on
contact sheets.

    python -m sportcal.lab.hockey.track_swaps cache nhl4.mp4 nhl7.mp4 nhl10.mp4      # GPU -> runs/tracking/<video>.npz
    python -m sportcal.lab.hockey.track_swaps signals nhl4 nhl7 nhl10                 # measures the noise margin
    python -m sportcal.lab.hockey.track_swaps signals nhl11 nhl12 nhl13 nhl14 --margin <frozen margin>

Hand labels go to runs/tracking/labels.csv, one `event_id,label` line per event read off the sheets in
runs/tracking/sheets/ (label: swap / no / unclear).
"""
import numpy as np
from scipy.ndimage import median_filter

from sportcal.paths import RUNS

V_MAX = 12.0        # m/s, fastest skater on record rounded up (source: hockey.md section 15)
A_MAX = 15.0        # m/s^2, above the peak push-off acceleration of elite skaters (source: hockey.md section 15)
TEAM_RATIO = 0.8    # a box's team counts when its distance ratio to the two jersey centroids is below this
TRACKERS = ("bytetrack", "botsort")
SIGNALS = ("jump", "turn", "team", "any")
FRESH_VIDEOS = {"nhl11", "nhl12", "nhl13", "nhl14"}
KPLINE = RUNS / "kpline" / "finetune" / "best_h.pt"
OUT = RUNS / "tracking"


def jumps(t, xy, v_max, margin):
    """Indices i where the step from observation i-1 to i is longer than a skater covers in that time plus the
    feet-point noise `margin`: the ID moved to someone else, or the homography moved."""
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.flatnonzero(step > v_max * np.diff(t) + margin) + 1


def turns(t, xy, a_max, fps, window=0.5, min_seen=0.8):
    """Indices where the mean velocity over the `window` s before and over the `window` s after differ by more than
    a_max * window (the two window centres are `window` s apart): a change of course no skater makes. Each window
    needs `min_seen` of its frames observed, so gaps and the ends of a track do not fire it."""
    idx = np.arange(len(t))
    lo = np.searchsorted(t, t - window)
    hi = np.searchsorted(t, t + window, "right") - 1
    need = min_seen * window * fps
    ok = (idx - lo + 1 >= need) & (hi - idx + 1 >= need) & (t - t[lo] > 0) & (t[hi] - t > 0)
    v_before = (xy - xy[lo]) / np.maximum(t - t[lo], 1e-9)[:, None]
    v_after = (xy[hi] - xy) / np.maximum(t[hi] - t, 1e-9)[:, None]
    return np.flatnonzero(ok & (np.linalg.norm(v_after - v_before, axis=1) > a_max * window))


def team_changes(t, team, hold=1.0):
    """Indices where a track's confident team votes (team >= 0) switch: the majority over the `hold` s around each vote
    holds one team for >= `hold` s, then the other for >= `hold` s. Shorter flickers are misread shirts, not swaps."""
    v = np.flatnonzero(team >= 0)
    if len(v) < 2:
        return np.array([], int)
    tv = t[v]
    lo, hi = np.searchsorted(tv, tv - hold / 2), np.searchsorted(tv, tv + hold / 2, "right")
    ones = np.r_[0, np.cumsum(team[v])]
    major = ((ones[hi] - ones[lo]) / (hi - lo) > 0.5).astype(int)
    b = np.r_[0, np.flatnonzero(np.diff(major)) + 1, len(v)]
    dur = tv[b[1:] - 1] - tv[b[:-1]]
    return np.array([v[b[k]] for k in range(1, len(b) - 1) if dur[k - 1] >= hold and dur[k] >= hold], int)


def h_jump_frames(jump_frames, obs_frames, min_tracks=3):
    """Frames on which more than half of the tracks seen, and at least `min_tracks`, jump at once: the homography
    moved, not the players."""
    seen = np.bincount(obs_frames)
    jumped = np.bincount(jump_frames, minlength=len(seen))
    return np.flatnonzero((jumped >= min_tracks) & (jumped > seen / 2))


def merge(key, t, idx, gap=1.0):
    """The first of each run of flags on one track that follow each other within `gap` s: one event per run. `key` and
    `t` are sorted by (track, time), as `observations` returns them."""
    idx = np.unique(idx)
    if len(idx) == 0:
        return idx
    return idx[np.r_[True, (key[idx][1:] != key[idx][:-1]) | (np.diff(t[idx]) > gap)]]


def noise(xy, k):
    """Distance from each position to the median of the k consecutive positions around it (k odd): the jitter of the
    feet point, which the jump test must tolerate."""
    # ponytail: consecutive rows, ignores time gaps inside a track (a gap inflates the estimate slightly); use a
    # time-windowed median if the margin ever looks too loose
    return np.linalg.norm(xy - median_filter(xy, size=(k, 1), mode="nearest"), axis=1)


def _tracks(key):
    """(start, end) of each track's rows in arrays sorted by track."""
    if len(key) == 0:
        return []
    s = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return list(zip(s, np.r_[s[1:], len(key)]))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_track_swaps.py -v`
Expected: 7 passed (the parametrized test counts twice).

- [ ] **Step 5: Mutation check** - in `jumps` delete `+ margin`, run the tests, expect FAIL in
  `test_a_skater_at_record_speed_or_on_a_tight_curve_fires_nothing` (the jitter alone now reads as jumps); restore,
  re-run: PASS. Then in `team_changes` change `dur[k - 1] >= hold and dur[k] >= hold` to `True`, expect FAIL on the
  flicker assert; restore, re-run: PASS.

- [ ] **Step 6: Commit**

```bash
git add sportcal/lab/hockey/track_swaps.py tests/test_track_swaps.py
git commit -m "track_swaps: swap signals in rink metres (hockey.md 15)"
```

---

### Task 3: From a cache to measurable observations (`observations`)

**Files:**
- Modify: `sportcal/lab/hockey/track_swaps.py` (imports; add `observations` after `_tracks`)
- Modify: `tests/test_track_swaps.py` (append one test)

**Interfaces:**
- Consumes: the cache layout written by Task 4 (a dict-like of arrays): `fps` (scalar), `shot` (n_frames,) int,
  `H` (n_frames, 3, 3) world->image with NaN where the product refused, `det_frame` (n_det,) int,
  `xyxy` (n_det, 4) float32, `hist` (n_det, 128) float16 with NaN rows for no jersey colour,
  `id_bytetrack` / `id_botsort` (n_det,) int with -1 for untracked. `core.teams.fit_teams`, `team_of`.
- Produces: `observations(c, tracker: str, sport) -> dict` of arrays sorted by (track, frame): `key` int64
  (`shot * 10**6 + tracker id`), `frame` int, `t` float s, `xy` (n, 2) m, `team` int (0/1, -1 unsure),
  `xyxy` (n, 4).

- [ ] **Step 1: Append the failing test** to `tests/test_track_swaps.py`:

```python
def test_observations_keep_measurable_tracked_feet_only_and_split_ids_by_shot():
    from sportcal import sports
    from sportcal.lab.hockey.track_swaps import observations

    H = np.array([[20.0, 0, 100], [0, 20.0, 50], [0, 0, 1]])      # world metres -> image px

    def box(x, y):                                                   # a box whose feet are at world (x, y)
        u, v = 100 + 20 * x, 50 + 20 * y
        return [u - 10, v - 60, u + 10, v]

    c = {"fps": np.float64(60), "shot": np.array([0, 0, 1]), "H": np.stack([H, np.full((3, 3), np.nan), H]),
         "det_frame": np.array([0, 0, 0, 1, 2]),
         "xyxy": np.array([box(10, 5), box(-5, 5), box(20, 8), box(10, 5), box(10, 5)], np.float32),
         "hist": np.full((5, 128), np.nan, np.float16),
         "id_bytetrack": np.array([3, 4, -1, 3, 3]), "id_botsort": np.zeros(5, int)}
    o = observations(c, "bytetrack", sports.get("hockey-nhl"))
    # kept: box 0 (frame 0) and box 4 (frame 2, next shot); dropped: off the rink (1), untracked (2), frame without H (3)
    assert list(o["frame"]) == [0, 2]
    np.testing.assert_allclose(o["xy"], [[10, 5], [10, 5]], atol=1e-4)
    assert o["key"][0] != o["key"][1]          # the same tracker ID in two shots is two tracks
    assert list(o["team"]) == [-1, -1]         # no jersey colour, no team
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_track_swaps.py -k observations -v`
Expected: FAIL with `ImportError: cannot import name 'observations'`

- [ ] **Step 3: Implement** - add to the imports of `track_swaps.py`:

```python
import cv2

from sportcal.core.teams import fit_teams, team_of
```

  and after `_tracks`:

```python
def observations(c, tracker, sport):
    """The boxes of one cached video that `tracker` tracked and that can be measured - on a frame with H, feet inside
    the field. Dict of arrays sorted by (track, frame): key (shot * 10**6 + tracker id, as IDs restart at every cut),
    frame, t (s), xy (m), team (0/1, -1 when unsure or without jersey colour), xyxy."""
    f = c["det_frame"]
    H = c["H"][f]
    has_h = ~np.isnan(H[:, 0, 0])
    feet = np.c_[(c["xyxy"][:, 0] + c["xyxy"][:, 2]) / 2, c["xyxy"][:, 3], np.ones(len(f))]
    xy = np.full((len(f), 2), np.nan)
    if has_h.any():
        w = np.einsum("nij,nj->ni", np.linalg.inv(H[has_h]), feet[has_h])
        xy[has_h] = w[:, :2] / w[:, 2:]
    x0, x1, y0, y1 = sport.box
    inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    team = np.full(len(f), -1)
    hist = c["hist"].astype(np.float32)
    coloured = inside & ~np.isnan(hist[:, 0])
    if coloured.sum() >= 2:
        cv2.setRNGSeed(0)
        teams, ratio = team_of(hist[coloured], fit_teams(hist[coloured]))
        team[coloured] = np.where(ratio < TEAM_RATIO, teams, -1)
    ids = c[f"id_{tracker}"]
    key = c["shot"][f].astype(np.int64) * 10**6 + ids
    sel = np.flatnonzero(inside & (ids >= 0))
    sel = sel[np.lexsort((f[sel], key[sel]))]
    return {"key": key[sel], "frame": f[sel], "t": f[sel] / float(c["fps"]), "xy": xy[sel], "team": team[sel],
            "xyxy": c["xyxy"][sel]}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_track_swaps.py -v`
Expected: 8 passed

- [ ] **Step 5: Mutation check** - replace `np.linalg.inv(H[has_h])` with `H[has_h]` (projecting the wrong way),
  expect FAIL on the `xy` assert; restore, re-run: PASS.

- [ ] **Step 6: Commit**

```bash
git add sportcal/lab/hockey/track_swaps.py tests/test_track_swaps.py
git commit -m "track_swaps: observations - tracked feet on the rink, keyed per shot"
```

---

### Task 4: The GPU pass (`cache`) and the command line

**Files:**
- Modify: `sportcal/lab/hockey/track_swaps.py` (imports; add `_hist_or_nan`, `cache`, `main`)

**Interfaces:**
- Consumes: `core.teams.jersey_histogram`; `product.hockey.build_pipeline` (lazy import); `KPLINE`, `OUT`.
- Produces: `cache(video, device="cuda:0", imgsz=1280, max_frames=0) -> None`, writing
  `runs/tracking/<stem>.npz` (or `<stem>-<max_frames>f.npz` when `max_frames` is set, so a smoke run never replaces a
  full cache) with the keys listed in Task 3 plus `video` (absolute path str), `method` (n_frames,) str,
  `conf` (n_det,) float32. `main()` with subcommand `cache`.

- [ ] **Step 1: Implement** - extend the imports of `track_swaps.py` so they read:

```python
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

from sportcal.core.teams import fit_teams, jersey_histogram, team_of
from sportcal.paths import ROOT, RUNS
```

  then append:

```python
def _hist_or_nan(frame, box):
    h = jersey_histogram(frame, box)
    return np.full(128, np.nan) if h is None else h


def cache(video, device="cuda:0", imgsz=1280, max_frames=0):
    """One GPU pass over `video`: shot cuts, the product's H, COCO person boxes with their jersey histogram, and the IDs
    the two stock trackers give them (both recreated at every cut). Saved to runs/tracking/<stem>[-<max_frames>f].npz."""
    import supervision as sv
    from scenedetect.common import FrameTimecode
    from scenedetect.detectors import ContentDetector
    from scenedetect.scene_manager import compute_downscale_factor
    from tqdm import tqdm
    from trackers import BoTSORTTracker, ByteTrackTracker
    from ultralytics import YOLO

    from sportcal.product.hockey import build_pipeline     # lab -> product: only here, to measure what the product sees

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = compute_downscale_factor(max(w, h))
    small = (round(w / k), round(h / k))
    yolo = YOLO(str(ROOT / "yolo26m.pt"))
    pipe = build_pipeline(None, kpline_weights=KPLINE, hold_frames=15, device=device, smooth=0.1)
    cuts = ContentDetector()

    def new_trackers():
        return {"bytetrack": ByteTrackTracker(frame_rate=fps), "botsort": BoTSORTTracker(frame_rate=fps)}

    trk, shot_no = new_trackers(), 0
    shot, Hs, method, det = [], [], [], defaultdict(list)
    for i in tqdm(range(max_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), unit="frame"):
        ok, frame = cap.read()
        if not ok:
            break
        if cuts.process_frame(FrameTimecode(i, fps=fps), cv2.resize(frame, small)):
            trk, shot_no = new_trackers(), shot_no + 1
            pipe.reset()
        shot.append(shot_no)
        est = pipe(frame)
        Hs.append(np.full((3, 3), np.nan) if est is None else est.H)
        method.append("none" if est is None else est.method)
        d = sv.Detections.from_ultralytics(
            yolo.predict(frame, classes=[0], conf=0.1, imgsz=imgsz, device=device, verbose=False)[0])
        d.data["i"] = np.arange(len(d))       # trackers reorder and drop rows; this maps them back
        det["det_frame"].append(np.full(len(d), i, np.int32))
        det["xyxy"].append(d.xyxy.astype(np.float32))
        det["conf"].append(d.confidence.astype(np.float32))
        det["hist"].append(np.array([_hist_or_nan(frame, b) for b in d.xyxy], np.float16).reshape(-1, 128))
        for name, tr in trk.items():
            out = tr.update(d, frame=frame) if name == "botsort" else tr.update(d)   # only BoT-SORT uses the frame
            ids = np.full(len(d), -1, np.int32)
            if len(out):
                ids[out.data["i"]] = out.tracker_id
            det[f"id_{name}"].append(ids)
    path = OUT / (Path(video).stem + (f"-{max_frames}f" if max_frames else "") + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    Hs = np.array(Hs)
    np.savez_compressed(path, video=str(Path(video).resolve()), fps=fps, shot=np.array(shot, np.int32), H=Hs,
                        method=np.array(method), **{k: np.concatenate(v) for k, v in det.items()})
    print(f"{path}: {len(shot)} frames, {np.isfinite(Hs[:, 0, 0]).mean():.0%} with H, "
          f"{sum(map(len, det['xyxy']))} boxes, {shot_no + 1} shots")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cache", help="one GPU pass per video -> runs/tracking/<video>.npz")
    a.add_argument("videos", nargs="+")
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--imgsz", type=int, default=1280)
    a.add_argument("--max-frames", type=int, default=0, help="smoke runs: stop here, cache named <video>-<n>f")
    args = ap.parse_args()
    if args.cmd == "cache":
        for v in args.videos:
            cache(v, args.device, args.imgsz, args.max_frames)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check the GPU is free**

Run: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
Expected: no `python.exe` training process. If one is running, stop here and ask the owner.

- [ ] **Step 3: Smoke run on 600 frames of nhl10, timed**

Run: `time venv/Scripts/python.exe -m sportcal.lab.hockey.track_swaps cache nhl10.mp4 --max-frames 600`
Expected: ends with `...runs\tracking\nhl10-600f.npz: 600 frames, N% with H, M boxes, K shots`. Note the wall time: a
full 4-minute video is ~14400 frames (nhl4: ~7200), so the full runs of Task 6 take (time / 600) x that.

- [ ] **Step 4: Inspect the cache**

```bash
venv/Scripts/python.exe -c "
import numpy as np; c = np.load('runs/tracking/nhl10-600f.npz')
print({k: c[k].shape for k in c.files})
n = len(c['shot']); print('boxes/frame', len(c['det_frame']) / n)
for t in ('bytetrack', 'botsort'): print(t, 'tracked', (c['id_' + t] >= 0).mean().round(2), 'ids', len(set(c['id_' + t]) - {-1}))
print('methods', {m: int((c['method'] == m).sum()) for m in set(c['method'])})
print('hist rows with colour', (~np.isnan(c['hist'][:, 0].astype(float))).mean().round(2))"
```

Expected: consistent lengths (`shot`, `H`, `method` = 600; all `det_*`/`id_*`/`xyxy`/`conf`/`hist` the same length);
tens of boxes per frame; most boxes tracked by both trackers; methods mostly `kpline...` variants. Anything else
(nothing tracked, all `none`) means a bug: stop and debug before Task 5.

- [ ] **Step 5: Fast suite and import safety**

Run: `venv/Scripts/python.exe -m pytest`
Expected: all pass (the module imports without running anything or loading ultralytics).

- [ ] **Step 6: Commit**

```bash
git add sportcal/lab/hockey/track_swaps.py
git commit -m "track_swaps: cache - one GPU pass with the product's H, COCO people, jersey colour and stock tracker IDs"
```

---

### Task 5: Events, rates, contact sheets and precision (`signals`)

**Files:**
- Modify: `sportcal/lab/hockey/track_swaps.py` (imports; add `find_events`, `read_labels`, `_around`, `_crop`,
  `_sheets`, `signals`; extend `main`)
- Modify: `tests/test_track_swaps.py` (append two tests)

**Interfaces:**
- Consumes: `observations`, `jumps`, `turns`, `team_changes`, `h_jump_frames`, `merge`, `noise`, `_tracks`,
  constants; caches from Task 4.
- Produces: `find_events(o, fps, margin) -> ({signal: int array of observation indices}, h_jump_frames array)`;
  `read_labels(path: Path) -> dict[str, str]`; `signals(names, margin=None, n_sample=20, seed=0) -> None`;
  subcommand `signals`. Event id format: `<video>:<tracker>:<key>:<frame>`. Sheets in
  `runs/tracking/sheets/<names joined by ->/<tracker>_<signal>_<k>.png`.

- [ ] **Step 1: Append the failing tests** to `tests/test_track_swaps.py`:

```python
def test_labels_file_is_created_once_and_never_overwritten(tmp_path):
    from sportcal.lab.hockey.track_swaps import read_labels

    p = tmp_path / "labels.csv"
    assert read_labels(p) == {} and p.read_text() == "event_id,label\n"
    p.write_text("event_id,label\nnhl10:bytetrack:3:120,swap\n")
    assert read_labels(p) == {"nhl10:bytetrack:3:120": "swap"}
    assert "swap" in p.read_text()


def test_fresh_games_need_the_margin_frozen_on_development():
    from sportcal.lab.hockey.track_swaps import signals

    with pytest.raises(SystemExit, match="margin"):
        signals(["nhl10", "nhl12"])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_track_swaps.py -v`
Expected: the two new tests FAIL with `ImportError: cannot import name 'read_labels'` / `'signals'`.

- [ ] **Step 3: Implement** - add `import csv` next to `import argparse` and `from sportcal import sports` next to the
  other `sportcal` imports; then add, before `def main():`:

```python
def find_events(o, fps, margin):
    """Where events start, as observation indices per signal ('any' = the union of jump, turn and team), flags merged
    per track within 1 s. Jump and turn flags within 0.5 s of an H-jump frame are dropped: the homography moved, not a
    player. Also returns those H-jump frames."""
    raw = defaultdict(list)
    for a, b in _tracks(o["key"]):
        t, xy = o["t"][a:b], o["xy"][a:b]
        raw["jump"].append(a + jumps(t, xy, V_MAX, margin))
        raw["turn"].append(a + turns(t, xy, A_MAX, fps))
        raw["team"].append(a + team_changes(t, o["team"][a:b]))
    flags = {s: np.concatenate(raw[s] or [np.array([], int)]).astype(int) for s in ("jump", "turn", "team")}
    hj = h_jump_frames(o["frame"][flags["jump"]], o["frame"])
    if len(hj):
        for s in ("jump", "turn"):
            f = o["frame"][flags[s]]
            flags[s] = flags[s][np.abs(f[:, None] - hj[None]).min(1) > 0.5 * fps]
    flags["any"] = np.concatenate([flags["jump"], flags["turn"], flags["team"]])
    return {s: merge(o["key"], o["t"], v) for s, v in flags.items()}, hj


def read_labels(path):
    """{event_id: label} typed by hand into `path` (swap / no / unclear). Created with its header when missing and never
    rewritten: it holds the only copy of the hand labels."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            f.write("event_id,label\n")
    with open(path, newline="") as f:
        return {r["event_id"]: r["label"].strip() for r in csv.DictReader(f)}


def _around(o, i, dt=0.5):
    """(frame, box) of the track of observation i at dt s before, at i, and dt s after (clipped to the track)."""
    a = np.searchsorted(o["key"], o["key"][i])
    b = np.searchsorted(o["key"], o["key"][i], "right")
    rows = [min(a + np.searchsorted(o["t"][a:b], o["t"][i] + d), b - 1) for d in (-dt, 0.0, dt)]
    return [(int(o["frame"][r]), o["xyxy"][r]) for r in rows]


def _crop(img, box, size=(300, 200)):
    """The region around a box, 3 box heights tall (at least 200 px), with the box drawn: enough to see who is in it."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    hh = max(1.5 * (y2 - y1), 100.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r0, r1 = int(max(0, cy - hh)), int(min(img.shape[0], cy + hh))
    c0, c1 = int(max(0, cx - 1.5 * hh)), int(min(img.shape[1], cx + 1.5 * hh))
    return cv2.resize(img[r0:r1, c0:c1], size)


def _sheets(folder, stem, items, per_png=10):
    """PNG contact sheets: per event its id, then three crops (0.5 s before, at, 0.5 s after) with the track's box."""
    folder.mkdir(parents=True, exist_ok=True)
    caps = {}
    for p in range(0, len(items), per_png):
        rows = []
        for eid, video, shots in items[p:p + per_png]:
            if video not in caps:
                caps[video] = cv2.VideoCapture(video)
            tiles = []
            for f, box in shots:
                caps[video].set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, img = caps[video].read()
                tiles.append(_crop(img, box) if ok else np.zeros((200, 300, 3), np.uint8))
            head = np.zeros((22, 900, 3), np.uint8)
            cv2.putText(head, eid, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            rows += [head, np.hstack(tiles)]
        cv2.imwrite(str(folder / f"{stem}_{p // per_png}.png"), np.vstack(rows))


def signals(names, margin=None, n_sample=20, seed=0):
    """Swap signals of the cached videos `names`: rates per video and tracker, contact sheets of `n_sample` random
    events per tracker and signal, and, for the events labelled in runs/tracking/labels.csv, each signal's precision
    and the real swaps per minute of play with H."""
    fresh = sorted(set(names) & FRESH_VIDEOS)
    if fresh and margin is None:
        raise SystemExit(f"{fresh} are `fresh` games: pass --margin, the value frozen on the development videos "
                         "(hockey.md section 15)")
    sport = sports.get("hockey-nhl")
    caches = {n: dict(np.load(OUT / f"{n}.npz")) for n in names}
    obs = {(n, tr): observations(caches[n], tr, sport) for n in names for tr in TRACKERS}
    if margin is None:
        margin = 2 * np.percentile(np.concatenate(
            [noise(o["xy"][a:b], max(3, round(0.1 * float(caches[n]["fps"])) | 1))
             for (n, _), o in obs.items() for a, b in _tracks(o["key"])]), 99)
        print(f"margin measured on these videos: {margin:.2f} m - freeze it in hockey.md 15, pass --margin on `fresh`")
    pool, minutes = defaultdict(list), defaultdict(float)
    print(f"{'video':8} {'tracker':9} {'min H':>6} {'IDs/player':>10} {'len s':>6} "
          + " ".join(f"{s + '/min':>9}" for s in SIGNALS) + f" {'Hjump/min':>9}")
    for (n, tr), o in obs.items():
        c = caches[n]
        fps = float(c["fps"])
        ev, hj = find_events(o, fps, margin)
        m = max(np.isfinite(c["H"][:, 0, 0]).sum() / fps / 60, 1e-9)
        minutes[tr] += m
        for s, idx in ev.items():
            pool[tr, s] += [(n, int(i)) for i in idx]
        shots = c["shot"][o["frame"]]
        ids = sum(len(np.unique(o["key"][shots == s])) for s in np.unique(shots))
        players = sum(np.median(np.unique(o["frame"][shots == s], return_counts=True)[1]) for s in np.unique(shots))
        spans = _tracks(o["key"])
        length = np.median([o["t"][b - 1] - o["t"][a] for a, b in spans]) if spans else 0.0
        print(f"{n:8} {tr:9} {m:6.1f} {ids / max(players, 1):10.2f} {length:6.1f} "
              + " ".join(f"{len(ev[s]) / m:9.2f}" for s in SIGNALS) + f" {len(hj) / m:9.2f}")
    labels = read_labels(OUT / "labels.csv")
    rng = np.random.default_rng(seed)
    folder = OUT / "sheets" / "-".join(names)
    for tr in TRACKERS:
        for s in SIGNALS:
            events = pool[tr, s]
            pick = [events[j] for j in rng.choice(len(events), min(n_sample, len(events)), replace=False)]
            eids = [f"{n}:{tr}:{obs[n, tr]['key'][i]}:{obs[n, tr]['frame'][i]}" for n, i in pick]
            _sheets(folder, f"{tr}_{s}",
                    [(e, str(caches[n]["video"]), _around(obs[n, tr], i)) for e, (n, i) in zip(eids, pick)])
            got = [labels[e] for e in eids if labels.get(e) in ("swap", "no")]
            line = f"{tr:9} {s:5} events {len(events):5}  labelled {len(got):2}/{len(eids)}"
            if got:
                p = got.count("swap") / len(got)
                line += f"  precision {p:.2f}"
                if s == "any":
                    line += f"  real swaps/min {len(events) / minutes[tr] * p:.2f}"
            print(line)
    print(f"sheets: {folder}")
```

  and extend `main()` - after the `cache` subparser add:

```python
    s = sub.add_parser("signals", help="swap signals, rates, contact sheets and, with labels.csv, precision")
    s.add_argument("names", nargs="+", help="cache names under runs/tracking, e.g. nhl4 nhl7 nhl10")
    s.add_argument("--margin", type=float, help="feet-point noise margin (m): measured when omitted, required on fresh")
```

  and replace the dispatch with:

```python
    if args.cmd == "cache":
        for v in args.videos:
            cache(v, args.device, args.imgsz, args.max_frames)
    else:
        signals(args.names, args.margin)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_track_swaps.py -v`
Expected: 10 passed

- [ ] **Step 5: Mutation check** - in `read_labels` change `if not path.exists():` to `if True:`, expect FAIL
  (the hand label is wiped); restore, re-run: PASS.

- [ ] **Step 6: Smoke run on the 600-frame cache of Task 4**

Run: `venv/Scripts/python.exe -m sportcal.lab.hockey.track_swaps signals nhl10-600f`
Expected: a margin line, a table with two rows (bytetrack, botsort) of finite numbers, eight `events ... labelled 0/N`
lines, and PNGs in `runs/tracking/sheets/nhl10-600f/`. Open one PNG: each event shows its id and three crops with a
yellow box on a person. A box on no one, or crops from the wrong moment, is a bug (frame seek or `_around`): fix before
Task 6. `runs/tracking/labels.csv` now exists with only its header.

- [ ] **Step 7: Fast suite**

Run: `venv/Scripts/python.exe -m pytest`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add sportcal/lab/hockey/track_swaps.py tests/test_track_swaps.py
git commit -m "track_swaps: signals - events, rates, contact sheets, precision from hand labels"
```

---

### Task 6: Run the experiment and write it up

**Files:**
- Modify: `docs/experiments/hockey.md` section 15 (fill **Result**, **Decision**; freeze the margin in **Method**)
- Data (gitignored): `runs/tracking/*.npz`, `runs/tracking/labels.csv`, `runs/tracking/sheets/`

**Interfaces:**
- Consumes: the `cache` and `signals` subcommands.
- Produces: the numbers and the decision in hockey.md section 15.

- [ ] **Step 1: GPU free** - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`, no training.

- [ ] **Step 2: Development caches** (long: time from Task 4 Step 3; run in the background)

Run: `venv/Scripts/python.exe -m sportcal.lab.hockey.track_swaps cache nhl4.mp4 nhl7.mp4 nhl10.mp4`
Expected: three `runs/tracking/<video>.npz` summary lines.

- [ ] **Step 3: Development signals**

Run: `venv/Scripts/python.exe -m sportcal.lab.hockey.track_swaps signals nhl4 nhl7 nhl10`
Expected: the measured margin, the rates table, sheets in `runs/tracking/sheets/nhl4-nhl7-nhl10/`.

- [ ] **Step 4: Label the development sheets** - for every event on the `*_jump_*`, `*_turn_*` and `*_team_*` sheets
  of both trackers, append a line `<event id>,<swap|no|unclear>` to `runs/tracking/labels.csv`: `swap` when the box is
  on a different player after the event than before, `no` when it stays on the same player (an H jump or noise),
  `unclear` when the crops do not tell. The owner, or Claude reading the PNGs (as in hockey.md section 14e).

- [ ] **Step 5: Precision on development** - re-run Step 3's command (same margin, same seed, same samples).
  Expected: `labelled N/N  precision p` per tracker and signal. If a signal fires mostly on noise (e.g. jumps from
  feet jitter), the margin may be raised here and only here - write the reason down. Then write the frozen margin into
  the **Method** of hockey.md section 15 (`margin = <value> m, measured on nhl4, nhl7, nhl10`).

- [ ] **Step 6: Final caches** (long, background)

Run: `venv/Scripts/python.exe -m sportcal.lab.hockey.track_swaps cache nhl11.mp4 nhl12.mp4 nhl13.mp4 nhl14.mp4`

- [ ] **Step 7: Final signals, label, report**

Run: `venv/Scripts/python.exe -m sportcal.lab.hockey.track_swaps signals nhl11 nhl12 nhl13 nhl14 --margin <frozen margin>`
Label every event on the `*_any_*` sheets of both trackers (Step 4 rules), then re-run the same command.
Expected: `real swaps/min` for each tracker. Change nothing in the code or the margin between the two runs.

- [ ] **Step 8: Write up** - in hockey.md section 15 replace `**Result:** not run yet.` with: the rates table of the
  final run (per video and tracker: minutes with H, IDs per player, median track length, events/min per signal,
  H jumps/min), the development precision per signal (n labelled), the final precision of `any` and real swaps per
  minute per tracker (n labelled), and the **Decision** the pre-registered rule gives (< 1 real swap/min for the
  better tracker: keep it, next is level 2; otherwise build the metric-space tracker). Add any new caveat seen on the
  sheets. Numbers only here, never in code.

- [ ] **Step 9: Commit**

```bash
git add docs/experiments/hockey.md
git commit -m "hockey.md 15: tracking step 0 result - stock tracker swaps per minute and the decision"
```

- [ ] **Step 10: Next** - if the decision is to build the metric-space tracker, that is a new design (brainstorming
  -> plan), judged by the same signals plus fragmentation. If it is to keep, the next design is level 2 (identity
  across cuts). Either way, stop here and report to the owner.
