"""Reentrena la deteccion de los 56 keypoints del rink sobre YOLO26m-pose.

    python -m sportcal.lab.hockey.train_pose

Salida: runs/hockeyrink/yolo26m-N/weights/ (Ultralytics numera solo)
        best_homography.pt  <- el que hay que usar, ver mas abajo

METRICA: pose mAP no dice si esto sirve. Lo que importa es la homografia que sale
de los keypoints, y se mide con sportcal/lab/hockey/rink_metric.py. Progreso medido sobre los
138 frames de val (error mediano de keypoint contra ground truth, px a 1920):

    techo (keypoints ground truth)          1.0 px   homografia usable en el 100 %
    yolo26m-8    sigma 0.10, pose 20       88.6 px
    yolo26m-12   sigma 0.02, pose 20       76.1 px
    yolo26m-13   sigma 0.02, pose  8       63.4 px   <- de aqui parte este run

NO USAR best.pt. Ultralytics elige el checkpoint por `fitness`, que pondera el mAP
de CAJA, y en este problema la caja se degrada mientras los keypoints mejoran: en
yolo26m-13, best.pt da 85.2 px y last.pt 63.4 px. Por eso este script guarda
checkpoints cada SAVE_PERIOD epochs y al terminar los barre con rink_metric.sweep(),
que deja el ganador en weights/best_homography.pt.

Por lo mismo patience va MUY alta: en yolo26m-13 el early stop salto en el epoch 42
porque el mAP de caja hizo su maximo en el epoch 1, justo cuando el de keypoints
marcaba su mejor valor en el 39. Cortaba por una metrica que no nos importa.

box=15.0 (default 7.5): con sigma 0.02 la pose loss vale ~3-4 y la de RLE ~6, y
entre las dos ahogan a la cabeza de deteccion. Se nota: los frames de val sin
NINGUNA caja pasaron de 13 a 48 de 138, y eso limita la cobertura de homografia al
65 % por mucho que mejoren los keypoints. Subir box compensa el reparto.

OJO con la cabeza de deteccion aguas abajo: el rink es un unico objeto que ocupa
casi todo el frame y el modelo es inseguro con la caja. Hay que coger siempre la
mejor caja, no filtrar por confianza.

fliplr=0.5: hockeyrink_pose.yaml ya define flip_idx (el mapeo de simetria
izquierda<->derecha de los 56 puntos), asi que el flip horizontal reetiqueta bien
y duplica de hecho el dataset. Si quitas flip_idx del yaml, vuelve a poner 0.0.

mosaic=0.0 a proposito: cada imagen tiene un unico objeto "rink" cuya caja es
casi el frame entero; el mosaico lo encogeria a un cuadrante y no aporta.

IMPORTANTE: el dataset yaml DEBE llevar kpt_oks_sigmas. Con sigma 0.10 la OKS loss
daba 0.038 para un error de 89 px -- o sea, ya convergida -- y el modelo no tenia
ningun gradiente que lo empujase a ser mas preciso. Ver el comentario largo de
hockeyrink_pose.yaml.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

from ultralytics import YOLO

from sportcal.paths import CONFIGS, ROOT

from sportcal.lab.hockey import rink_metric

DATA = CONFIGS / "hockey" / "hockeyrink_pose_rp.yaml"
PESOS = ROOT / "runs" / "hockeyrink" / "yolo26m-17" / "weights" / "best_homography.pt"
IMGSZ = 1024
DEVICE = 0
SAVE_PERIOD = 10   # epochs entre checkpoints; alimenta el barrido final


def main() -> None:
    if not DATA.exists():
        raise SystemExit("falta el dataset: corre antes  python -m sportcal.lab.hockey.prepare_data")
    if not PESOS.exists():
        raise SystemExit("faltan los pesos de partida: {}".format(PESOS))

    model = YOLO(str(PESOS))
    model.train(
        data=str(DATA),
        epochs=250,          # solo 1093 imagenes: necesita muchas vueltas
        # yolo26m-pose es mas pesado que yolo26s; en 8 GB: 1024 / batch 4.
        imgsz=IMGSZ,
        batch=4,             # fijo, NO autobatch
        cache=False,
        device=DEVICE,
        project=str(ROOT / "runs" / "hockeyrink"),
        name="yolo26m",
        patience=250,        # <- ver docstring: el early stop cortaba por el mAP de caja
        save_period=SAVE_PERIOD,
        cos_lr=True,
        fliplr=0.5,          # <- ver docstring (necesita flip_idx en el yaml)
        flipud=0.0,
        degrees=0.0,
        translate=0.04,
        scale=0.25,
        mosaic=0.0,          # <- ver docstring
        mixup=0.0,
        pose=8.0,            # bajado de 20: con sigma 0.02 la loss cruda ya es 5x mayor
        box=15.0,            # <- ver docstring: la caja se estaba quedando sin gradiente
        optimizer="AdamW",
        lr0=3e-4,
        lrf=0.05,
        warmup_bias_lr=0.0,
    )

    # El checkpoint bueno no es el que elige Ultralytics: se barren todos y se
    # escoge por error de reproyeccion de la homografia. Aqui la GPU ya esta libre.
    save_dir = getattr(model.trainer, "save_dir", None) if model.trainer else None
    pesos_dir = Path(save_dir) / "weights" if save_dir else None
    if pesos_dir and pesos_dir.exists():
        print("\n" + "=" * 70)
        print("  barrido de checkpoints por homografia")
        print("=" * 70)
        rink_metric.sweep(pesos_dir, imgsz=IMGSZ, device=DEVICE)


if __name__ == "__main__":
    main()
