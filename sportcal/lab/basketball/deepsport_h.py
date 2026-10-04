"""Per-frame H index for basketball from the DeepSportRadar "basketball instants" dataset (MMSports 2022 calibration
challenge: 728 frames from 364 instants, 2 fixed cameras each, 15 French league arenas), the basketball counterpart of
`lab/tennis/tennis_h`. Results: docs/experiments/basketball.md.

Each frame comes with an exact pinhole calibration (K, R, T, in cm, origin at a court corner, x along the length, z
pointing down, images undistorted), so H is computed, not fitted: no label noise to measure, only a convention to check
(`overlay` draws the template). The challenge holds three arenas out for its test set (`TEST_ARENAS`); two more arenas
(chosen by a stable hash of their name) form `dev`, so no arena is in two splits.

The keypoint model (`models/kpline.py`) assumes 16:9 frames and would crop the bottom quarter of these 1624 x 1234 ones;
the frames are copied with black bars at the sides to 16:9 instead (`pad_to_16_9`).
ponytail: padding keeps `models/` untouched but wastes a quarter of the input width; letterbox inside `to_input` and
`estimate_H` if a non-16:9 sport ever matters for the product.

    kaggle datasets download deepsportradar/basketball-instants-dataset   (unzip into datasets/basket)
    python -m sportcal.lab.basketball.deepsport_h          # -> datasets/basketball_h.jsonl, datasets/basket_169/
"""
import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

from sportcal.core.camera import canonical_mirror, is_plausible_view
from sportcal.paths import DATASETS
from sportcal.sports.basketball import court as C

ROOT = DATASETS / "basket"
PADDED = DATASETS / "basket_169"
OUT = DATASETS / "basketball_h.jsonl"
BOX = (-C.HALF_LENGTH, C.HALF_LENGTH, -C.HALF_WIDTH, C.HALF_WIDTH)
TEST_ARENAS = ("KS-FR-CAEN", "KS-FR-LIMOGES", "KS-FR-ROANNE")     # the challenge's held-out arenas (84 frames)
N_DEV_ARENAS = 2


def calib_to_H(calib):
    """World -> pixel homography (3, 3) for the plane z = 0, from a DeepSportRadar calibration dict, world in metres
    centred on the court: pixel ~ K (R X + T) with X in cm from the corner, so X_cm = 100 (X_m + (14, 7.5))."""
    K = np.array(calib["KK"], float).reshape(3, 3)
    R = np.array(calib["R"], float).reshape(3, 3)
    T = np.array(calib["T"], float)
    corner = 100.0 * np.array([C.HALF_LENGTH, C.HALF_WIDTH])
    H = K @ np.c_[100.0 * R[:, 0], 100.0 * R[:, 1], R[:, :2] @ corner + T]
    return H / H[2, 2]


def pad_to_16_9(img, H):
    """(image, H) with black bars at the left and right until the frame is 16:9; H follows the shift."""
    h, w = img.shape[:2]
    pad = max(0, int(round(h * 16 / 9)) - w)
    left = pad // 2
    out = cv2.copyMakeBorder(img, 0, 0, left, pad - left, cv2.BORDER_CONSTANT, value=0)
    return out, np.array([[1.0, 0, left], [0, 1.0, 0], [0, 0, 1.0]]) @ H


def splits(arenas):
    """{arena: split}: the challenge's test arenas, then the `N_DEV_ARENAS` lowest hashes of the rest as dev."""
    rest = sorted((a for a in arenas if a not in TEST_ARENAS), key=lambda a: zlib.crc32(a.encode()))
    return {a: "test" if a in TEST_ARENAS else "dev" if a in rest[:N_DEV_ARENAS] else "train" for a in arenas}


def overlay(img, H, colour=(0, 255, 0)):
    """The template's polylines drawn on a copy of the image through H."""
    img = img.copy()
    for pl in C.polylines().values():
        q = np.c_[pl, np.ones(len(pl))] @ H.T
        if (q[:, 2] > 0).all():
            cv2.polylines(img, [np.round(q[:, :2] / q[:, 2:]).astype(np.int32)], False, colour, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT), help="the unzipped dataset (arena folders and basketball-instants-dataset.json)")
    args = ap.parse_args()
    root = Path(args.root)
    if not (root / "basketball-instants-dataset.json").exists():
        raise SystemExit("{} has no basketball-instants-dataset.json: nothing written".format(root.resolve()))
    files = sorted(root.glob("*/*/camcourt*[0-9].json"))
    split = splits({f.parts[-3] for f in files})
    rows, dropped = [], {"no image": 0, "invalid": 0, "implausible": 0}
    for f in files:
        meta = json.load(open(f, encoding="utf-8"))
        img = cv2.imread(str(f.with_name(f.stem + "_0.png")))
        if img is None:
            dropped["no image"] += 1
            continue
        if not meta.get("valid", 1):
            dropped["invalid"] += 1
            continue
        arena, game = f.parts[-3], f.parts[-2]
        H = calib_to_H(meta["calibration"])
        h0, w0 = img.shape[:2]
        if not is_plausible_view(H, w0, h0, BOX):
            dropped["implausible"] += 1
            continue
        img, H = pad_to_16_9(img, H)
        H, flips = canonical_mirror(H, (0.0, 0.0))
        dest = PADDED / arena / game / (f.stem + ".jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"id": "basketball/{}/{}/{}".format(arena, game, f.stem), "image": "datasets/basket_169/{}/{}/{}.jpg".format(
                     arena, game, f.stem), "dataset": "deepsport", "video": "{}/{}".format(arena, game), "arena": arena,
                     "template": "basketball-fiba", "source": "deepsport", "w": img.shape[1], "h": img.shape[0],
                     "H": H.tolist(), "flips": flips, "fit": {"method": "calibration"}, "split": split[arena]})
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
    print("{} frames -> {}  {}  dropped {}".format(len(rows), OUT, counts, dropped))
    print("dev arenas:", sorted(a for a, s in split.items() if s == "dev"))


if __name__ == "__main__":
    main()
