# Reentrenamiento de los modelos (YOLO26)

Dos modelos independientes (no un modelo multi-head):

| Modelo | Base | Dataset HF | Task |
|---|---|---|---|
| Detección jugadores/puck/rink-marks | `yolo26s` | `SimulaMet-HOST/HockeyAI` (2101 img, 7 clases) | detect |
| Keypoints del rink | `yolo26m-pose` | `SimulaMet-HOST/HockeyRink` (661 img, 56 kpts) | pose |

## Flujo

```bash
# 1. Descargar y preparar los datasets (formato YOLO train/val, con hardlinks)
python training/prepare_data.py

# 2. Entrenar (RTX 3060 Ti 8 GB; batch FIJO, no autobatch)
python training/train_detect.py     # -> runs/hockeyai/yolo26s/weights/best.pt
python training/train_pose.py       # -> runs/hockeyrink/yolo26m/weights/best.pt
```

`test.py` usa automáticamente los `best.pt` locales si existen; si no, cae a los
pesos de HuggingFace.

## Decisiones

- **Clases de detección**: se mantiene el orden/nombres del modelo desplegado
  (`centriod, faceoff, goal, goalie, player, puck, referee`) para no romper la
  búsqueda por nombre de `test.py`.
- **`fliplr=0.0` en pose**: los 56 puntos del rink no tienen `flip_idx` (mapeo de
  simetría izquierda↔derecha), así que un flip horizontal mezclaría etiquetas.
- **`mosaic=0.0` en pose**: cada imagen tiene un único "rink" que ocupa casi todo
  el frame; el mosaico no aporta.
- **Split train/val**: por hash MD5 del nombre de fichero → estable entre corridas.
- **`imgsz=1280`**: el puck es diminuto en 1080p; bajar resolución lo pierde.

## VRAM (8 GB) — por qué batch fijo

AutoBatch (`batch=0.85`) mide el coste a batch 1,2,4… y ajusta una recta. A
1280² con yolo26s cada imagen pide ~3-4 GB, así que solo caben batch 1-2; el
probe de batch 4 hace OOM (`backward = nan`), AutoBatch extrapola con 2 puntos y
elige de más → OOM en la epoch 1. Por eso los scripts fijan:

| | imgsz | batch |
|---|---|---|
| detección (yolo26s) | 1024 | 8 |
| pose (yolo26m-pose) | 1024 | 4 |

Además `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` para la fragmentación.
Si aún hay OOM: baja el batch a la mitad. Si te sobra VRAM: `imgsz=1280`, batch 4/2.

## Inferencia

`test.py` ahora corre el modelo de rink 1 de cada `RINK_EVERY` frames (=15) y
reutiliza la última homografía entre medias (la cámara es casi estática dentro de
un plano). Se fuerza recálculo tras cada corte de plano.
