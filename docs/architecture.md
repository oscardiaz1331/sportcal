# Architecture

`sportcal` estimates the homography between a sport field and broadcast video, so player
tracking can be projected onto a minimap. The repository has two jobs that must not be
mixed up:

* **Labs** - per-sport experiments that evaluate approaches (classical CV, segmentation,
  keypoint models...) and record what worked, with numbers.
* **Product** - one deployable pipeline that *composes* the approaches that won.

Code holds the mechanisms. **Results, decisions and reasoning live in markdown** (`docs/`),
never in code comments or docstrings.

## Layers

```
sportcal/
  core/      sport-agnostic building blocks: geometry (DLT, refinement), pinhole camera,
             robust fitting, surface segmentation, frame-to-frame background motion, label I/O
  sports/    one subpackage per sport: template geometry + line classes (pure data)
  lab/       per-sport experiments, runnable with `python -m sportcal.lab.<sport>.<name>`
    <sport>/archive/   frozen dead ends and superseded tools (unmaintained)
    common/            sport-agnostic labs and the Streamlit apps (`streamlit run sportcal/lab/common/<app>.py`)
  product/   the deployed pipeline: Estimator chain + per-sport estimators
tests/       fast, deterministic, no GPU / weights / datasets
docs/        architecture, decisions (ADRs), experiment write-ups, porting status
configs/     training configs (yaml), per sport
```

Data is **not** in the package. `datasets/`, `runs/` (weights), videos and scratch output are
large, gitignored and shared between sports; `sportcal/paths.py` is the single place that
knows where they are (override the root with `SPORTCAL_ROOT`).

## Dependency rules

Arrows read "may import".

```
product  ->  core, sports          (lab only via lazy import inside a constructor: temporary)
lab      ->  core, sports
sports   ->  core                  (in practice only numpy + sports.base: they are pure data)
core     ->  (nothing above it)    never imports sports, lab or product
```

`tests/test_layering.py` enforces these by reading the imports (no code is executed), so a
violation fails CI instead of surfacing as a tangled import a year later. One rule is
deliberately loose today: `product/hockey.py` lazily imports two lab modules
(see [ADR 0001](decisions/0001-layered-architecture.md), "Known debt").

## Adding things

**A sport** (e.g. soccer): create `sportcal/sports/<name>/`, build `Sport` instances
(`sports/base.py`: size, surface, line classes, polylines, optional keypoints), `register()`
them, import the subpackage from `sports/__init__.py`. Nothing in `core/` changes; if you
find yourself editing `core/` for a sport, the abstraction is wrong - write it down in an ADR.

**An experiment**: add `sportcal/lab/<sport>/<name>.py` with a `main()` guarded by
`if __name__ == "__main__"` (importing a lab module must never run it - an unguarded script
once re-split a dataset on import). Reusable pieces belong in `core/`. Then write it up in
`docs/experiments/<sport>.md` using the template in
[experiments/README.md](experiments/README.md): hypothesis, method, numbers, decision,
reproduce command. An experiment without a write-up did not happen.

**A product method**: implement `Estimator` (`product/pipeline.py`): `estimate(frame)` returns
an `Estimate` or `None` (refuse). Insert it in the chain where its trust belongs and record the
reason in [ADR 0003](decisions/0003-product-composition.md). A method only enters the product
after a lab write-up shows its numbers on the reference validation set.

**A test**: `tests/test_<module>.py`, pure logic on synthetic data. Anything needing weights or
data, or that takes more than ~5 s, gets `@pytest.mark.slow`: the default `pytest` run deselects those, `pytest -m ""` runs everything.
A test that cannot fail is noise: when you add one for a numerical property, break the
code on purpose once and check it goes red (the Hartley-normalisation test was added this way).

## Running

```bash
uv pip install --python venv/Scripts/python.exe -e ".[dev]"     # once (add [train] / [lab] as needed)
venv/Scripts/python.exe -m pytest                              # 72 fast tests, ~8 s
venv/Scripts/python.exe -m pytest -m ""                        # everything (75), ~45 s
venv/Scripts/python.exe -m sportcal.lab.hockey.seg_to_homography --selftest
```

## Conventions

* Code, comments and docstrings are English; docstrings say what a function does and why it is
  shaped that way, never quote experiment numbers (link the write-up instead).
* Pixel errors are always reported normalised to a 1920 px frame width (`core.geometry.REF_WIDTH`).
* World frame: metres, X along the length, Y across the width, origin at a corner (hockey) or the
  centre (soccer). H maps world -> image.
* A deliberate simplification with a known ceiling gets a `ponytail:` comment naming the ceiling
  and the upgrade path.
