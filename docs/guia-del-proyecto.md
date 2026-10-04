# Guía del proyecto - lo que hay que saber de `sportcal` (y cómo contarlo como trabajo de visión por computador y deep learning)

Complementa a [lecciones-aprendidas.md](lecciones-aprendidas.md) (los hallazgos) y a [architecture.md](architecture.md)
(el diseño del código). Aquí: qué problema es, qué se construyó, qué conceptos de CV/DL hay detrás, qué hay que saber contar
con soltura y cuáles son las limitaciones honestas.

## 1. El problema en tres frases

Una emisión de TV de hockey, fútbol o tenis muestra el campo en perspectiva. Para proyectar a los jugadores en un minimapa
(posición en metros, velocidades, distancias) hay que saber, **para cada frame, la homografía H** que lleva un punto del plano del
campo (metros) a un píxel. La cámara se mueve (pan, tilt, zoom), las líneas se ocultan, hay publicidad, hielo reflectante y cortes de
plano, y el objetivo de precisión es ~8 px a 1920 de ancho.

* H es una matriz 3x3 con 8 grados de libertad: cuatro correspondencias punto-punto la fijan (DLT).
* Con una cámara pinhole, H = K [r1 r2 t]: de ahí salen focal, rotación y posición; sirve para detectar H imposibles.
* La tarea es **calibración de cámara plana**, no detección de objetos: un buen mAP no implica buena H.

## 2. Qué hay construido (mapa mental)

```
vídeo -> corte de plano -> modelo keypoint+línea (U-Net, heatmaps) -> RANSAC + DLT puntos+rectas -> gates -> H por frame
                                                                                  |                      |
                                                           suavizado con movimiento KLT           minimapa + detección de jugadores (YOLO26)
                                                                                                         + seguimiento (ByteTrack) + equipo por color
```

| Pieza | Dónde | Qué hace |
|---|---|---|
| Plantillas de campo | `sportcal/sports/` | hockey NHL/IIHF, fútbol FIFA, tenis ITF, baloncesto FIBA: geometría pura (keypoints, rectas, polilíneas) |
| Núcleo geométrico | `sportcal/core/` | DLT con normalización de Hartley, refinamiento robusto, cámara pinhole, `canonical_mirror`, `is_plausible_view`, movimiento KLT, segmentación de superficie, mezcla de homografías |
| Modelo | `sportcal/models/` | U-Net con ResNet34 + decodificación heatmaps -> puntos -> H (`kpline`) |
| Laboratorio | `sportcal/lab/<deporte>/` | experimentos, índices de H por frame, etiquetadores (Streamlit) |
| Producto | `sportcal/product/` | cadena de estimadores, worker por vídeo, app web local (Starlette, ADR 0006) |
| Documentación | `docs/` | arquitectura, ADRs (decisiones), experimentos por deporte, plan de porting |

Regla de capas (comprobada por `tests/test_layering.py`): `core` <- `sports` <- `models` <- `lab`/`product`.

## 3. El modelo que ganó ("modelo A" / E1)

* **Entrada**: frame a 960x544. **Red**: U-Net, encoder ResNet34 (ImageNet), heatmaps a media resolución.
* **Salidas**: un canal por keypoint con nombre de la plantilla + dos canales por cada recta (sus dos extremos visibles) (hockey: 56 + 2x9 = 74;
  con derivados 108; fútbol 39 + 2x17 = 73; tenis 14 + 2x9 = 32). Gaussianas de sigma 1.5 px a 480.
* **Pérdida**: focal tipo CenterNet, bias final con p = 0.01.
* **Targets generados al vuelo** desde una H por frame + la plantilla: nunca se guardan máscaras; etiquetar = tener una H.
* **Aumento exacto de cámara** (`core.camera.ptz_warp`): giro/zoom alrededor del centro, la etiqueta pasa a G·H sin error; más espejo
  (renombra puntos) y brillo/contraste.
* **Decodificación**: pico por canal (>= 0.3) -> RANSAC sobre puntos -> un DLT con los inliers + las rectas -> gate de plausibilidad.
* **Entrenamiento en dos fases** (hockey): preentrenar con todas las etiquetas (60 épocas, ~82 min) y afinar con solo las manuales (30 épocas, ~2 min).
* **Selección de checkpoint** por la mediana del error de H en `dev`, nunca por la loss.

Por qué esta arquitectura y no otra (ADR 0004): la red devuelve *evidencia* con confianza; el solver geométrico, común a todos los deportes,
da H, y permite gates, rechazar y reutilizar. Regresar H o la pose directamente no aprende (p50 ~340 px).

## 4. Conceptos que hay que dominar para defender este trabajo

**Geometría**
* Homografía, DLT, por qué la **normalización de Hartley** es imprescindible (sin ella el SVD es inútil, p50 1858 px).
* Cámara pinhole: K, R, t; descomponer H; por qué la focal está mal condicionada con vistas laterales.
* Correspondencias de **recta** (dual DLT) frente a puntos; por qué las paralelas no fijan una dirección.
* RANSAC y su caos numérico (un umbral de 8 px sobre ~20 puntos hace la H inestable en ~20% de frames).
* Ambigüedades de simetría (180º, espejo) y la regla de nombres `canonical_mirror`.
* Elipses (círculos del campo vistos en perspectiva) como cónicas: 5 de los 8 números de H.
* PTZ de centro fijo: 3 grados de libertad (pan, tilt, focal) en vez de 8; distorsión radial k1.

**Aprendizaje profundo**
* Regresión de coordenadas frente a heatmaps; por qué los heatmaps conservan la localización.
* U-Net, CoordConv, pérdida focal, desbalance de clases extremo (pesos 1/sqrt(freq), Dice).
* Fine-tuning con pocas etiquetas; curva de etiquetas (10 etiquetas = 6612 en tenis).
* Aumentos que conservan la etiqueta exactamente (warp de cámara) frente a los que no.
* Dominio: UDA (MIC, profesor EMA, enmascarado) y datos sintéticos; por qué aquí no ayudaron.
* Evaluación: cobertura (responde o rechaza), p50, p90, % < 10 / 25 px, respuestas > 50 px, con y sin gate; split por vídeo; ruido de muestreo.

**Visión clásica**
* Segmentación de superficie por color local; KLT (Lucas-Kanade) con comprobación hacia atrás y RANSAC; filtro complementario temporal.
* Por qué el KLT sobre el hielo falla (reflejos) y sobre las gradas funciona.

**Ingeniería**
* Experimentos reproducibles: comando por resultado, índices de datos, `fresh` intocable, ADR para cada decisión.
* Pruebas de mutación: romper el código una vez para ver que el test falla.

## 5. Cómo se ve el resultado

| Deporte | Estado | Cifra principal |
|---|---|---|
| Hockey NHL | modelo A2 + gate | `fresh`: 97% cobertura, p50 7.2 px; con vídeo (suavizado 0.1) el jitter cae ~10x (29.6 -> 3.1 px). YOLO base: ~100 px |
| Fútbol FIFA | E1 sobre índice refinado + gate | SoccerNet `test` 89% / p50 6.5 px; `fresh` 100% / 7.7 px (PnLCalib 10.3) |
| Tenis ITF | modelo por deporte | `test` p50 1.7 px (contra etiquetas de un detector) |
| Baloncesto FIBA | modelo por deporte (DeepSportRadar, no comercial) | `test` en 3 pabellones no vistos: 550 etiquetas 100% / p50 4.6 px; 200 etiquetas 100% / 5.2 px; 10 etiquetas no funcionan |

Producto: app local (`python -m sportcal.product.server` -> http://127.0.0.1:8000); un worker por vídeo, un trabajo a la vez, estado en
disco (`status.json`), salida `out.mp4` + `tracks.csv` (posición en metros por jugador y frame) (ADR 0006).

## 6. La historia que puedes contar (de menos a más técnico)

1. **Punto de partida**: un modelo público de keypoints de hockey (661 frames de la liga sueca, plantilla IIHF) que no servía para
   la NHL: otro tamaño de pista y otro dominio.
2. **Primer error de planteamiento**: se midió con mAP; un modelo con mAP 0.50 daba 0% de homografías útiles. Se pasó a error de H.
3. **Probar lo clásico y lo aprendido de forma ordenada y documentarlo**: segmentación + DLT (precisa pero cubre 7-12%), regresión de
   pose (no aprende), UDA y sintéticos (no ayudan). Todo con su cifra.
4. **Cambio decisivo**: adoptar el planteamiento de PnLCalib (heatmaps de puntos y rectas + DLT) con targets generados desde una H por frame.
   Pasó de ~100 px a ~7-9 px.
5. **Auditar etiquetas y datos** (la parte que más enseñó): etiquetas automáticas a 15-25 px, espejos aleatorios, QA por vídeo, `fresh`.
6. **Gates**: la regla física de plausibilidad frente a umbrales de confianza que no discriminan.
7. **Reutilizar la receta en fútbol y tenis** sin tocar `core/`: solo plantilla + índice de H.
8. **Producto**: app web local con vídeo anotado y trayectorias en metros.
9. **Siguiente pregunta de investigación**: un deporte con muchas vistas cuesta ~200 etiquetas (baloncesto, medido); falta ver si una red condicionada por la plantilla lo baja (ADR 0005, sin construir)

Frases cortas que resumen el aprendizaje: "la métrica de entrenamiento no era la del problema", "la calidad de las etiquetas fue el techo",
"las mejoras vinieron de datos y de reglas físicas, no de arquitectura", "cada resultado negativo está documentado con su cifra".

## 7. Limitaciones honestas (decirlas antes de que las pregunten)

* Los números de hockey son de emisiones NHL; los estadios exteriores (nhl9) siguen sin funcionar; las pistas IIHF no se han probado en el
  producto.
* `fresh` de hockey son 38 frames (1 respuesta gruesa): la puerta de plausibilidad está confirmada en un caso. `fresh` de fútbol son 19.
* Una H "equivocada pero con aspecto de cámara real" pasa la gate.
* Las etiquetas del tenis vienen de un detector, no de clics humanos; baloncesto: 84 frames de test en 3 pabellones, un run por N.
* Las comparaciones son un solo run por configuración; el PC tuvo reinicios y lotes corruptos que se mitigaron, pero cuya causa no se encontró.
* El seguimiento de jugadores (identidad a lo largo del tiempo) está diseñado (h 15) pero no medido.
* Licencias: PnLCalib es GPL-2.0, DeepSportRadar es no comercial.

## 8. Cómo ejecutarlo (desde la raíz del repo)

```bash
venv/Scripts/python.exe -m pytest                                   # tests rápidos (~10 s)
python -m sportcal.lab.common.train_kpline --sport hockey-nhl --eval runs/kpline/finetune/best_h.pt --split fresh --gate
python -m sportcal.product.video nhl11.mp4 --sport hockey-nhl --end 20   # un vídeo -> runs/product/<run>/
python -m sportcal.product.server                                    # la app web
streamlit run sportcal/lab/common/annotate_val_app.py                # etiquetar frames a mano
```

Reglas prácticas: un trabajo pesado a la vez (GPU de 8 GB), no tocar `train_kpline.py` mientras entrena, continuar un entrenamiento en carpeta
nueva (`--tag=-cont`), `opencv-contrib-python` solo, comandos siempre desde la raíz.

## 9. Dónde mirar para cada pregunta

| Pregunta | Documento |
|---|---|
| ¿Por qué esta capa/estructura? | `architecture.md`, ADR 0001 |
| ¿Por qué heatmaps + solver y no regresión? | ADR 0004, hockey.md §8, §14 |
| ¿Cómo se compone el producto? | ADR 0003, ADR 0006 |
| ¿Qué cuesta un deporte nuevo? | tennis.md §3, basketball.md, ADR 0005 |
| ¿Qué se probó y falló? | `lecciones-aprendidas.md` §4, hockey.md §3-§9 |
| ¿Estado de cada deporte? | `CLAUDE.md` ("Current standing") |
| Literatura y estado del arte | `hockey_research.md` |
| Dónde estaba cada archivo antes de reorganizar | `porting-status.md` |
