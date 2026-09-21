# ADR 0001 - Layered architecture: core / sports / lab / product

Status: accepted (2026-09-20)

## Context

The repository grew as a research log: ~14k lines in a flat namespace (`rink.py`, `test*.py`,
`field_*.py` at the root; `training/`, `testing/`), ~35 files doing
`sys.path.insert(0, ROOT/"training")` to import each other by bare name, a copy-pasted Windows
PATH workaround in ~10 scripts, and soccer code importing hockey's `ice_lines_probe` for generic
surface segmentation. Adding a sport meant copying hockey code; nothing said which results were
"the product" and which were dead ends.

## Decision

1. One installable package, `sportcal`, in four layers (`core`, `sports`, `lab`, `product`)
   with import rules enforced by `tests/test_layering.py` (see `docs/architecture.md`).
2. Labs are per sport and keep experiments runnable as `python -m sportcal.lab.<sport>.<name>`;
   the *knowledge* they produce lives in `docs/experiments/`, not in code comments.
3. The product is a composition (`product/pipeline.py`) of methods that won in a lab, chosen by
   documented numbers (ADR 0003).
4. Sport-agnostic code moved to `core/` (DLT + refinement, robust fitting, surface segmentation,
   label I/O). `surface` is a parameter (`"ice"` | `"grass"`), replacing the Spanish `deporte=` strings.
5. Dead ends are **archived, not deleted** (`lab/<sport>/archive/`, unmaintained), and the
   pre-restructure state is tagged `pre-restructure`.
6. Data stays where it is on disk; `sportcal/paths.py` is the only module that knows locations.

## Consequences

+ No `sys.path` hacks; tests import the same modules users run.
+ A new sport touches `sports/<name>/` and its lab, never `core/`.
+ The Windows CUDNN PATH workaround lives once, in `sportcal/__init__.py`.
- Moved-but-not-yet-translated lab files still carry Spanish comments (tracked in
  `docs/porting-status.md`).

## Known debt

* `product/hockey.py::SegDltEstimator` lazily imports `lab/hockey/diagnose_lines_seg.py` and
  `seg_to_homography.py`, because the segmentation network and `solve_from_probs` still live in the
  lab. Upgrade path: promote them to `sportcal/models/` and `core/` when the lab port lands.
* (Resolved 2026-09-21) `testing/` was ported: soccer template -> `sports/soccer`, camera math -> `core/camera`, solver ->
  `lab/soccer`, apps -> `lab/common`, and the four temporary compat shims were deleted. Lab code of different sports must not
  import each other (enforced by `tests/test_layering.py`); what both need goes to `core/` or `lab/common/`.
