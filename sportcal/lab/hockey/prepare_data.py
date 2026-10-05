"""Descarga los datasets de HuggingFace y los deja en formato YOLO (train/val).

    datasets/hockeyai/    -> deteccion, 7 clases   (SimulaMet-HOST/HockeyAI)
    datasets/hockeyrink/  -> pose, 56 keypoints    (SimulaMet-HOST/HockeyRink)

Usa hardlinks (mismo volumen NTFS) para no duplicar ~1 GB de imagenes; si el
hardlink falla copia el fichero. Idempotente: si la carpeta destino ya existe
no rehace nada salvo que pases --force.

    python -m sportcal.lab.hockey.prepare_data            # ambos
    python -m sportcal.lab.hockey.prepare_data --only pose --force
"""
import argparse
import hashlib
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from sportcal.paths import CONFIGS, ROOT
DATASETS = ROOT / "datasets"
TRAINING = CONFIGS / "hockey"

# el orden/nombres se toman del modelo desplegado (SimulaMet-HOST/HockeyAI) para
# no romper test.py, que busca las clases por nombre ("goalie", "player", ...)
DET_NAMES = ["centriod", "faceoff", "goal", "goalie", "player", "puck", "referee"]

# Espejo izquierda<->derecha de los 56 keypoints del rink: al voltear la imagen la zona A
# pasa a ser la B (0,1<->54,55  2..7<->48..53  8,9<->46,47  10..13<->42..45  14,15<->40,41
# 16..19<->36..39, y en la zona neutral 20<->34  21<->35  22<->32  23<->33  28<->31).
# Con esto train_pose.py puede usar fliplr, que dobla de hecho el dataset.
POSE_FLIP_IDX = [54, 55, 48, 49, 50, 51, 52, 53, 46, 47, 42, 43, 44, 45, 40, 41, 36, 37, 38, 39,
                 34, 35, 32, 33, 24, 25, 26, 27, 31, 29, 30, 28, 22, 23, 20, 21,
                 16, 17, 18, 19, 14, 15, 10, 11, 12, 13, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1]

# kpt_oks_sigmas fija la tolerancia de la OKS loss: e = d^2 / ((2*sigma)^2 * area * 2). Con la
# caja del rink (68% del frame, ~718k px^2 a 1024) la distancia donde e=1 sale 449 px@1920 con
# sigma 0.10 y 90 px con 0.02. Con 0.10 el modelo convergia con 89 px de error mediano por
# keypoint porque para la loss eso ya era perfecto (loss 0.038): pose mAP alto y homografias
# inservibles. 0.02 deja ese error justo en e=1, donde el gradiente es maximo.
#
# OJO: 0.02 es para AFINAR desde un checkpoint que ya ronda los 90 px. Entrenando desde cero
# (COCO) la loss se satura a 1.0 y el gradiente se anula, que era el motivo original de 0.10.
# Ver docs/experiments/hockey.md (kpt_oks_sigmas) y sportcal/lab/hockey/rink_metric.py.
POSE_OKS_SIGMA = 0.02

# frames propios etiquetados sobre pistas NHL (relativo a `path`); se entrena y se valida con
# los dos datasets juntos, asi las metricas (y el best.pt que elige Ultralytics) tienen en
# cuenta las pistas NHL. El reparto train/val de hockeyrink_nhl lo hace split_val.py.
POSE_EXTRA_TRAIN = ["../hockeyrink_nhl/images/train"]
POSE_EXTRA_VAL = ["../hockeyrink_nhl/images/val"]


def _split(stem: str, val_fraction: float) -> str:
    """Reparte train/val por hash del nombre: estable entre corridas, sin fugas."""
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
    return "val" if (h % 10_000) / 10_000 < val_fraction else "train"


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    # snapshot_download deja los ficheros como symlinks al blob de la cache de HF;
    # hay que resolverlos al fichero real antes de hacer el hardlink
    real = Path(os.path.realpath(src))
    try:
        os.link(real, dst)
    except OSError:
        shutil.copy2(real, dst)


def _materialize(pairs, out: Path, val_fraction: float) -> dict:
    n = {"train": 0, "val": 0}
    for img, lbl in pairs:
        sp = _split(lbl.stem, val_fraction)
        _link(img, out / "images" / sp / img.name)
        _link(lbl, out / "labels" / sp / lbl.name)
        n[sp] += 1
    return n


def prepare_detection(force: bool) -> None:
    out = DATASETS / "hockeyai"
    if out.exists() and not force:
        print(f"[det] {out} ya existe, salto (--force para rehacer)")
        return
    if force and out.exists():
        shutil.rmtree(out)

    zip_path = hf_hub_download(
        "SimulaMet-HOST/HockeyAI", "HockeyAI_Dataset.zip", repo_type="dataset"
    )
    raw = DATASETS / "_hockeyai_raw"
    if not (raw / "SHL").exists():
        print(f"[det] extrayendo {zip_path} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw)

    frames, labels = raw / "SHL" / "frames", raw / "SHL" / "annotations"
    pairs = [
        (frames / f"{lbl.stem}.jpg", lbl)
        for lbl in sorted(labels.glob("*.txt"))
        if (frames / f"{lbl.stem}.jpg").exists()
    ]
    n = _materialize(pairs, out, val_fraction=0.10)
    print(f"[det] train={n['train']} val={n['val']}")

    (TRAINING / "hockeyai_det.yaml").write_text(
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(DET_NAMES)),
        encoding="utf-8",
    )
    print("[det] -> configs/hockey/hockeyai_det.yaml")


def prepare_pose(force: bool) -> None:
    out = DATASETS / "hockeyrink"
    if out.exists() and not force:
        print(f"[pose] {out} ya existe, salto (--force para rehacer)")
        return
    if force and out.exists():
        shutil.rmtree(out)

    raw = Path(
        snapshot_download(
            "SimulaMet-HOST/HockeyRink",
            repo_type="dataset",
            allow_patterns=["frames/*", "annotations/*"],
        )
    )
    frames, labels = raw / "frames", raw / "annotations"
    pairs = [
        (frames / f"{lbl.stem}.jpg", lbl)
        for lbl in sorted(labels.glob("*.txt"))
        if (frames / f"{lbl.stem}.jpg").exists()
    ]
    n = _materialize(pairs, out, val_fraction=0.12)
    print(f"[pose] train={n['train']} val={n['val']}")

    train_dirs = ["images/train"] + POSE_EXTRA_TRAIN
    val_dirs = ["images/val"] + POSE_EXTRA_VAL
    (TRAINING / "hockeyrink_pose.yaml").write_text(
        f"path: {out.as_posix()}\n"
        f"train: [{', '.join(train_dirs)}]\n"
        f"val: [{', '.join(val_dirs)}]\n"
        "kpt_shape: [56, 3]\n"
        f"flip_idx: {POSE_FLIP_IDX}\n"
        f"kpt_oks_sigmas: {[POSE_OKS_SIGMA] * 56}\n"
        "names:\n  0: rink\n",
        encoding="utf-8",
    )
    print("[pose] -> configs/hockey/hockeyrink_pose.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rehace aunque ya exista")
    ap.add_argument("--only", choices=["det", "pose"], help="solo uno de los dos")
    args = ap.parse_args()

    TRAINING.mkdir(parents=True, exist_ok=True)  # configs/hockey/ is not tracked: a fresh clone has no such folder
    if args.only != "pose":
        prepare_detection(args.force)
    if args.only != "det":
        prepare_pose(args.force)
