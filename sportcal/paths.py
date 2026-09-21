"""Single source of truth for on-disk locations.

Data is deliberately NOT inside the package: datasets, weights and videos are large,
gitignored and shared between sports. Set SPORTCAL_ROOT to relocate everything
(e.g. to a bigger drive); by default it is the repository root.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SPORTCAL_ROOT", Path(__file__).resolve().parent.parent))
DATASETS = ROOT / "datasets"
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"
SCRATCH = ROOT / "scratch_frames"
