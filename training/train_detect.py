"""Reentrena la deteccion de HockeyAI (7 clases) sobre YOLO26s.

    python training/train_detect.py

Salida: runs/hockeyai/yolo26s/weights/best.pt
test.py lo coge automaticamente si existe (si no, cae al modelo de HuggingFace).
"""
import os

# mismo apaño que test.py: evita mezclar el cuDNN del sistema con el del wheel de torch
os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if "NVIDIA\\CUDNN" not in p
)
# reduce la fragmentacion del allocator de CUDA (clave con 8 GB justos)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "hockeyai_det.yaml"


def main() -> None:
    if not DATA.exists():
        raise SystemExit("falta el dataset: corre antes  python training/prepare_data.py")

    model = YOLO("yolo26s.pt")
    model.train(
        data=str(DATA),
        epochs=250,
        # 1280 seria ideal (puck diminuto en 1080p) pero no cabe en 8 GB ni a
        # batch 2; 1024 + batch 8 es el equilibrio. Si tienes mas VRAM: 1280 / batch 4.
        imgsz=1024,
        batch=8,             # fijo, NO autobatch: a esta resolucion AutoBatch elige de mas y peta
        cache=False,         # 'ram' cachearia ~1.5 GB de imagenes y come VRAM efectiva via SO
        device=0,
        project=str(ROOT / "runs" / "hockeyai"),
        name="yolo26s",
        patience=25,
        cos_lr=True,
        close_mosaic=15,
        # video de broadcast: flip horizontal si (no hay keypoints), nada vertical
        # ni rotaciones; el puck se pierde con escalados agresivos
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.3,
        mosaic=1.0,
        mixup=0.0,
    )
    model.val(data=str(DATA), split="val")


if __name__ == "__main__":
    main()
