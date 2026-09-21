"""sportcal: sport-field camera calibration (homography from broadcast video).

Layout:
    core/      sport-agnostic building blocks (geometry, robust fitting, surface segmentation)
    sports/    one subpackage per sport: template geometry + line classes (see sports/base.py)
    lab/       per-sport experiments; every result is written up in docs/experiments/
    product/   the deployed pipeline, composed from the methods that won in the lab

See docs/architecture.md for the dependency rules between these layers.
"""
import os

# Windows workaround: a system-wide NVIDIA CUDNN directory on PATH shadows the DLLs
# bundled with torch and crashes the import. Scrubbing it here is enough because
# importing any `sportcal.*` module runs this file first, i.e. before torch is loaded.
_CUDNN_DIR = os.sep.join(("NVIDIA", "CUDNN"))
os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if _CUDNN_DIR not in p
)
