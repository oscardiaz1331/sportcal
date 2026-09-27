# sportcal - project rules for Claude

Camera calibration (planar homography) from broadcast sports video, to project player tracking onto a minimap.
Per-sport **labs** evaluate approaches; one **product** pipeline composes the winners. Sports: hockey (NHL/IIHF),
soccer, tennis (the "new sport" of the multi-sport plan, ADR 0004). Start with `docs/architecture.md`.

## How to work here: the ponytail way (lazy senior developer)

Use the `ponytail` skill whenever it is available; if not, these are its rules. Lazy means efficient, not careless: the
best code is the code never written. Before writing code, stop at the first rung that holds:
1. Does it need building at all? 2. Does the stdlib do it? 3. A platform feature? 4. An installed dependency
(numpy, cv2, scipy, torch - check `core/` first, most geometry already exists)? 5. Can it be one line? 6. Only then
the minimum code that works.

* No abstractions, boilerplate or new dependencies nobody asked for. Deletion over addition, boring over clever,
  fewest files. Edit in place, surgically; never rewrite a file wholesale.
* Question complex requests: "do you need X, or does Y already cover it?"
* When two stdlib options are the same size, take the edge-case-correct one.
* **`ponytail:` comments.** Every deliberate simplification with a known ceiling (a global lock, an O(n^2) scan, a
  naive heuristic, a lazy import of lab code) gets a `# ponytail:` comment naming the ceiling and the upgrade path.
  `grep -rn "ponytail:" sportcal` is the project's debt list; read the ones near code you touch.
* Not lazy about: input validation at trust boundaries, error handling that prevents data loss (two index files and a
  best checkpoint were once overwritten by a wrong path or a reused run folder), security, anything explicitly asked.
* Non-trivial logic leaves ONE runnable check (one small test, no frameworks or fixtures). Break the code once to see
  it fail (mutation check) before trusting it. Trivial one-liners need no test.

## Skills, MCPs and agents: use them before hand-rolling

* **Library/API questions** (torch, ultralytics, opencv, scipy, streamlit...): the Context7 MCP (`resolve-library-id`,
  then `query-docs`) before answering from memory - versions here are recent (torch 2.14, opencv 5).
* **Broad searches** across many files: an `Explore` subagent, so only the conclusion enters the context. For a
  single known file or symbol, `Grep`/`Read` directly.
* **Before claiming done**: the verification skill if installed (`superpowers:verification-before-completion`) -
  run the tests / the eval and read the output first.
* **Bugs and surprising numbers**: `superpowers:systematic-debugging` / `engineering:debug` - reproduce, isolate,
  then fix. **New features**: `superpowers:brainstorming` for the design, `superpowers:test-driven-development` for
  the one check. **Decisions**: `engineering:architecture` writes the ADR in `docs/decisions/`.
* **Literature / datasets**: `anthropic-skills:deep-research` or parallel subagents; record findings in `docs/`.
* **Charts** of results: the `dataviz` skill. **Independent tasks**: `superpowers:dispatching-parallel-agents`.
* Isolated refactors while a training runs: `superpowers:using-git-worktrees` (see the GPU rule below).

## Context and memory

* This file is loaded every session: keep it short and current. Details live in `docs/`; link, do not copy.
* Results, decisions and reasoning go to `docs/experiments/<sport>.md` (template: `docs/experiments/README.md`) and
  ADRs - never only in chat. An experiment without a write-up did not happen.
* Read the relevant section of an experiment doc, not the whole file (hockey.md and soccer.md are long: grep the
  section headings first).
* Big outputs (per-frame tables, logs) go to a file and are summarised; do not paste thousands of lines into the chat.
* When a fact in this file goes stale (a new best model, a renamed command), update it in the same commit.

## Where things are

| Need | Look at |
|---|---|
| Layers, dependency rules, adding a sport / experiment / product method | `docs/architecture.md` |
| Why things are the way they are | `docs/decisions/` (ADRs; 0003 product composition, 0004 network output + shared solver) |
| What was measured, and the dead ends | `docs/experiments/hockey.md` (Spanish long-form `hockey.es.md`), `soccer.md`, `tennis.md` |
| Code | `sportcal/{core,sports,models,lab,product}`; tests in `tests/` |
| Porting ledger (old -> new paths) | `docs/porting-status.md` |

## Rules

* **Check the number before recommending.** Most things were measured, and several reasonable ideas measured badly
  (hockey.md sections 3, 8; soccer.md). Do not re-propose a documented dead end without new evidence.
* Code, comments, docstrings in English; UI text (Streamlit apps, printed guidance for the owner) may be Spanish.
  Docstrings never quote experiment numbers.
* Import layers are enforced by `tests/test_layering.py`: `core` <- `sports` <- `models` <- `lab`/`product`.
  Sport-agnostic code goes to `core/` or `models/`, what every sport's lab uses to `lab/common/`, never a sport's lab.
* A lab module must be import-safe (`main()` behind `if __name__ == "__main__"`).
* Run `venv/Scripts/python.exe -m pytest` after touching `core/`, `sports/`, `models/` or `product/` (~10 s, no GPU);
  before committing `pytest -m ""` (slow tests need weights). UI code is only covered by `tests/test_apps.py`.
* Data (`datasets/`, `runs/`, videos) is gitignored; paths come from `sportcal/paths.py`. Commands run from the repo
  root (a relative `--root` from elsewhere once wrote an empty index).
* **GPU is 8 GB**: nothing else on it while a training runs (`nvidia-smi`). On Windows the DataLoader workers
  re-import the training module at every dev evaluation: do not move or edit `lab/common/train_kpline.py` or what it
  imports while a training runs - use a worktree.
* A continued training goes to a new folder (`--tag -cont`): the first dev evaluation always writes `best_h.pt`.
* opencv: keep **`opencv-contrib-python`** only (plain `opencv-python` removes `cv2.ximgproc`).

## Facts that are easy to get wrong

* Current standing (target ~8 px on frames nothing was tuned on):
  * Hockey: model A2 (`runs/kpline/finetune/best_h.pt`) + plausibility gate, 97% coverage, p50 7.2 px on the `fresh`
    games; on video with `smooth=0.1` (KLT) the jitter falls ~10x (ADR 0003, hockey.md 14g-14i). YOLO: ~100 px.
  * Soccer: model E1 (`runs/kpline-soccer/pretrain-derived`), SoccerNet test p50 6.5 px; on our `fresh` labels it
    misses centre-circle views for lack of such frames in the index (soccer.md 20-22; `fit_circle` recovers them).
  * Tennis: template + index ready (`tennis.md`), no model yet.
* Keypoint + line model commands: `python -m sportcal.lab.common.train_kpline --sport <hockey-nhl|soccer-fifa|tennis-itf>`;
  new runs go to `runs/kpline-<sport>/`; the index is `datasets/<family>_h.jsonl`.
* Splits: `fresh` = hand labels of games nothing was tuned on - never train on it or pick thresholds with it.
* The plausibility gate (`core.camera.is_plausible_view`) refuses impossible H; a wrong but camera-like H still passes.
* End views are named by the view rule (`core.camera.canonical_mirror`), not by the physical end of the arena.
* Two hockey templates: `hockey-nhl` (60.96x25.91 m) and `hockey-iihf` (60x30 m).
* YOLO homography numbers are noisy (batch vs single-frame inference changes them, ADR 0003): few-frame differences
  are not signal. Best YOLO weights: `runs/hockeyrink/yolo26m-17/weights/best_homography.pt`.

## Commands

```bash
venv/Scripts/python.exe -m pytest                                                  # fast suite
python -m sportcal.lab.common.train_kpline --sport hockey-nhl --eval runs/kpline/finetune/best_h.pt --split fresh --gate
python -m sportcal.product.hockey_demo nhl11.mp4 [--max-frames 600] [--device cpu]  # minimap video, product pipeline
python -m sportcal.lab.soccer.soccernet_h --root datasets/calibration-2023          # soccer H index
python -m sportcal.lab.tennis.tennis_h                                             # tennis H index
python -m streamlit run sportcal/lab/common/annotate_val_app.py                    # hand-label hockey / soccer frames
```
