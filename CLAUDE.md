# hockey — registro del proyecto

Homografía cámara↔pista de hockey a partir de vídeo de difusión, para proyectar
tracking de jugadores sobre un minimapa. Dos vías en paralelo: un modelo YOLO26
que regresiona 56 keypoints, y visión clásica (color + geometría) tanto para
refinar como para generar datos de entrenamiento en bulk.

Este documento es un resumen de una sesión larga de investigación. Antes de
recomendar nada "porque sí", comprueba el número aquí abajo — casi todo lo
importante viene de medir, no de intuición, y varias intuiciones razonables
salieron mal (están documentadas como tal, a propósito).

## Estado ahora mismo (2026-09-16)

- **Modelo YOLO**: `runs/hockeyrink/yolo26m-17/weights/best_homography.pt`.
  Error mediano de keypoint ~61 px (normalizado a 1920 px de ancho). Bajando de
  forma sostenida entre runs (88.6 → 76.1 → 63.4 → 61.4 px) pero lejos del
  objetivo (~8 px) para que la homografía sea fiable.
- **`training/train_pose.py`**: `PESOS` ya apunta a ese checkpoint. Listo para
  lanzar otra vez sin tocar nada.
- **Vía clásica (color+geometría)**: funciona muy bien como *refinador de
  precisión* (parte de un error de 10-20 px y lo baja a ~7). No sirve como
  *rescate desde lejos* — con el error actual de YOLO (60-100 px) no tiene
  ventana de búsqueda válida. Ver sección "Auto-etiquetado" más abajo:
  calibrado y en espera de que YOLO mejore.
- **PIVOTE VALIDADO: segmentación semántica de líneas** (sección 6) supera por
  mucho a la vía clásica de color en precisión de localización — pero solo
  donde hay datos homogéneos suficientes (IIHF). En NHL (pocos datos, muy
  variados) el mismo modelo falla en varias clases. Se generó un dataset
  sintético (`training/gen_synthetic_lines.py`) para tapar ese hueco.
- **`training/seg_to_homography.py`: resuelve H directamente de la
  segmentación, sin keypoints.** Con dof≥10 + filtro de coste de refinamiento
  (`--max-costo`, r=0.97 con el error real en IIHF): igual o mejor que YOLO
  (~61px) en los frames que cubre, en vídeo real. Ver sección 7.
- **AUTO-ETIQUETADO DESBLOQUEADO Y PROBADO SOBRE VÍDEO REAL**:
  `training/auto_label_seg.py` reemplaza a `auto_label.py` (sección 5). Corrido
  sobre `nhl5,6,7,8,9,10.mp4` → 400 frames aceptados. **QA visual manual (24
  muestras) descubrió que el filtro de coste NO detecta todos los fallos**:
  nhl6 y nhl9 salieron 0/4 buenos (nhl6 es una pista amateur española, no NHL;
  nhl9 es un Stadium Series al aire libre) mientras nhl5/7/8/10 salieron
  15/16 buenos. Fusionados solo los 4 vídeos limpios (329 frames) en
  `hockeyrink_nhl` — ver sección 7bis, es lectura obligatoria antes de fiarse
  de `--max-costo` a ciegas.
- **DECIDIDO: pivotar de YOLO a un solver aprendido sobre la segmentación.**
  El experimento condicionado (reentrenar con datos NHL nuevos y remedir
  cobertura dof≥10) dio negativo -- NHL empeoró en las tres métricas (7%→5%
  cobertura, 37.8→56.0px p50, 156→174px p90), confirmando que el DLT clásico
  es el techo, no la falta de datos de segmentación. Construyendo la cabeza
  aprendida ahora. Ver sección 7ter.

## 1. El template geométrico (`rink.py`) — dos bugs reales, corregidos

- **Mapeo de la zona A mal calculado.** Las fórmulas `i+46`/`i+26` para reflejar
  la zona A sobre la B solo acertaban en un subconjunto de índices. El mapeo
  correcto es `MIRROR` (idéntico a `flip_idx` del yaml). Verificado por
  selección de modelo: 111 px de error de reproyección con la fórmula vieja,
  5.6 px con `MIRROR`, sobre las etiquetas reales.
- **Signo invertido en los faceoff dots neutrales** (quedaban 1.5 m fuera de la
  zona neutral en vez de dentro). Corregido, cuadra a 3 cm.
- Con las dos correcciones, el **techo de toda la tubería es 1.0 px** de error
  de reproyección usando keypoints ground truth. O sea: la geometría y el
  template están bien: todo lo que falla es el modelo o la vía clásica, nunca
  el template.

## 2. Por qué el modelo YOLO se atascaba (ya corregido)

Nueve runs (yolo26m-1 a -9) no mejoraban por una cadena de bugs de
entrenamiento, todos en `training/train_pose.py` y `hockeyrink_pose.yaml`:

- **`optimizer="auto"` ignoraba `lr0`.** Ultralytics usa AdamW a 0.002 fijo si
  no fijas `optimizer` explícitamente. Confirmado en los CSV: `lr/pg0` nunca
  se movía del valor de la fórmula automática.
- **`kpt_oks_sigmas=0.10` saturaba la OKS loss.** Con la caja del rink al 68%
  del frame, un error de keypoint de 89 px ya daba loss=0.038 — el optimizador
  lo consideraba "perfecto" y no había gradiente para mejorar. Bajado a 0.02:
  el mismo error ahora da loss≈0.62, justo donde el gradiente es máximo. **Si
  entrenas desde cero (COCO) hay que arrancar en 0.10** e ir bajando, o la
  pose_loss se satura al revés y se queda plana desde el epoch 1 (ver el
  comentario largo en `hockeyrink_pose.yaml`).
- **La cabeza de caja se quedaba sin gradiente.** Al bajar σ la pose_loss+RLE
  pasó a valer ~5x más que antes y ahogaba a `box` en su default (7.5). Subido
  a `box=15.0`. Antes del fix, frames de val sin ninguna caja detectada: 13→48
  de 138 — limitaba la cobertura de homografía por mucho que mejorasen los
  keypoints.
- **`fitness` de Ultralytics elige el checkpoint equivocado en este problema.**
  Pondera mAP de caja, que se degrada mientras los keypoints mejoran (medido en
  yolo26m-13: `best.pt`=85.2 px, `last.pt`=63.4 px; en yolo26m-17: `last.pt`
  82.1 px, `epoch80.pt` 61.4 px — **ni el último es siempre el mejor**). Por
  eso `train_pose.py` guarda checkpoints con `save_period=10` y al terminar
  llama a `rink_metric.sweep()`, que evalúa todos y deja el ganador en
  `weights/best_homography.pt`. **Usa siempre ese fichero, nunca `last.pt` a
  ciegas.**
- **`patience` muy alta a propósito (250).** El early stop cortaba en yolo26m-13
  por el mAP de caja (que hizo su máximo en el epoch 1) justo cuando el mAP de
  pose iba mejorando.

### Métrica: pose mAP no sirve, usa `training/rink_metric.py`

Un modelo con mAP alto y error de reproyección enorme es perfectamente posible
(lo tuvimos: σ=0.10 daba mAP 0.50 con 0% de homografías usables). `rink_metric.py`
ajusta H con los keypoints predichos por RANSAC y mide error de reproyección
contra el ground truth:

```
python training/rink_metric.py <pesos.pt>              # evalua un checkpoint
python training/rink_metric.py <carpeta weights> --sweep # evalua todos, copia el mejor a best_homography.pt
```

Ojo: la cabeza de detección es insegura (confianza mediana 0.59, p25=0.03) —
por defecto se coge siempre la mejor caja sin filtrar por confianza, o se
pierde ~42% de los frames sin que los keypoints tengan la culpa.

## 3. Datos: `relabel_reproject.py` — funciona, impacto modesto

Reetiqueta reproyectando el template por la H ajustada con los keypoints
marcados a mano. Genera `datasets/hockeyrink_rp/` y `hockeyrink_nhl_rp/`
(usado por defecto en `train_pose.py`). **Corrección a una expectativa
temprana**: la ganancia NO viene de "rellenar puntos que faltaban" — de media
solo ~14 de los 56 keypoints caen dentro del frame por razones físicas de
encuadre, así que no hay mucho que rellenar. La ganancia real fue **limpieza
de outliers**: 2789 puntos marcados a mano que el RANSAC descarta y se
corrigen a su posición geométrica (~27% de las anotaciones del dataset IIHF).

## 4. Vía clásica — mapa completo de lo que funciona y lo que no

Todo esto vive en `training/`. Se puede explorar visualmente con
`python training/line_explorer.py` (peor-a-mejor, 11 etapas, teclas en el
docstring del fichero).

### Funciona, con números

| Pieza | Fichero | Resultado |
|---|---|---|
| Segmentación de región de pista (HSV fijo + GMM Lab, elegido por solidez) | `ice_lines_probe.best_region()` | IoU 0.91-0.93, peor caso 0.34-0.71. Ni el umbral fijo ni el GMM ganan solos; elegir por solidez (la pista es convexa, una región rota se detecta sola) casi reduce a la mitad la tasa de fallos de cada uno por separado. |
| Croma local (Δa/Δb respecto al hielo LOCAL, no color absoluto) | `ice_lines_probe.local_chroma()` | Es la base de todo lo demás. El tono absoluto de una línea pintada bajo el hielo es casi neutro (ruido puro); su desviación del hielo de al lado es estable. |
| **Zócalo amarillo de las vallas** como término de homografía | `fit_homography_lines.kickplate_score()` | **z=79.9 (IIHF) / 48.5 (NHL)** deslizando el contorno por su normal — 8-16x más fuerte que cualquier línea pintada, y cubre todo el perímetro. Hallazgo del usuario, confirmado y ahora integrado. **Detalle crítico**: el pico no cae en el contorno sino ~8 px hacia fuera (el zócalo es una superficie vertical). Hay que usar un offset FIJO (8 px), no buscar un máximo sobre un abanico de offsets — eso mete un sesgo de -4 px y hunde el z de 20 a 4. |
| Corredor geométrico (restringir Hough/blobs a una ventana predicha por una H aproximada) | prototipo en sesión, no integrado como módulo aparte | Multiplica por ~2.5 la precisión de Hough (8.8%→22.2% de segmentos reales) y reduce los candidatos en ~11x. Confirmado en 3 frames independientes. **Pero necesita una H aproximada de partida** — no resuelve el problema de arrancar desde cero. |

### Descartado, con la cifra exacta (para no repetirlo)

Todos documentados en los docstrings de los ficheros correspondientes:

- **Casco convexo** sobre la máscara de hielo (`ice_lines_probe.rink_region`,
  docstring) — peor que rellenar huecos en todas las combinaciones medidas.
- **Gaussiana robusta** como alternativa al GMM (`ice_lines_probe.ice_robust`)
  — peor que el GMM y que el umbral fijo en los dos datasets.
- **Filtro de cresta multiescala** — no mejora sobre la respuesta cruda
  (z=3.1 vs 3.3 de Δa sin filtrar). Se deja solo para visualizar estructura.
- **Zócalo como segmentador de región completa** (no solo como término de
  ajuste) — catastrófico, p10=0.007 de IoU. Solo el 17-38% de los píxeles con
  Δb alto en el frame están cerca del contorno real; el resto es grada,
  anuncios, cualquier cosa cálida de la imagen. La misma señal que da z=80
  sobre una curva conocida es inútil sin esa restricción de forma. Documentado
  en `ice_lines_probe.region_kickplate_global()`.
- **Clasificación de blobs por forma** (aspect ratio línea vs mancha) para
  separar líneas reales de publicidad pintada bajo el hielo / camisetas —
  las distribuciones casi se solapan (ratio mediano 2.7 real vs 2.0 falso), y
  de 1735 componentes solo 220 (12.7%) eran líneas reales. Las líneas se
  fragmentan por oclusión en trozos tan compactos como un logo.
- **Hough (líneas y círculos) sobre el frame ENTERO, sin restricción** —
  encuentra el candidato real, pero ahogado: 1 de 122 segmentos, 1 de 22
  círculos. Útil solo si se combina con el corredor geométrico (arriba).
- **Círculo "creciendo" desde semilla** (ajuste de elipse iterativo, tipo
  active-shape-model / starburst, sugerido por el usuario) — implementado
  DOS veces (elipse libre y luego con forma fija derivada de H, que es la
  forma correcta de hacerlo). Ambas fallan por un límite de señal, no de
  algoritmo: **en el punto EXACTO del borde real, la respuesta de color está
  al nivel del ruido** (mediana 1.0 vs sigma=4.06). Todo lo que sí funciona en
  esta sesión integra sobre docenas/cientos de píxeles a la vez (zócalo,
  curvas completas); decidir el borde píxel a píxel no tiene margen para
  promediar el ruido. Es un límite del dato, no una implementación mejorable.
- **Tracking KLT** (`training/klt_propagate.py`) para propagar la homografía
  entre frames de un mismo plano — falla rápido y mal. Con checkpoints
  independientes cada ~1s: error p50 158 px, solo 20% de los intentos <25 px,
  24% de fallos catastróficos (>500 px). Causa medida: LK nunca reporta
  pérdida de pista (status=1 todo el rato) pero deriva en silencio hacia
  reflejos especulares del hielo (que NO están fijos al plano físico, violan
  la asunción de homografía fija). En 1 segundo se pierde ~90% de los puntos
  sembrados, solo por RANSAC descartándolos tras derivar, nunca por LK
  reportando fallo. Probado también trackeando dentro del campo de croma
  filtrado (hipótesis del usuario) — **peor todavía** (87.8 px vs 45.9 px),
  porque ese campo es más ruidoso frame a frame que la imagen cruda.
- **Ajuste conjunto de homografía sobre TODAS las curvas a la vez**
  (`fit_homography_lines.fit()`) — no converge ni partiendo de la verdad
  exacta (σ=0 da 6.8-11 px "tras ajustar", debería dar 0). El objetivo tiene
  máximos espurios (una curva puede enganchar con otra o con publicidad) y con
  mediana de solo 3 curvas en cuadro por frame no hay redundancia para
  RANSAC. Grueso-a-fino (desenfoque decreciente) ayuda pero no arregla el
  fondo. **Confirmado además que EMPEORA partiendo de una ancla YOLO real**
  (no perturbación de verdad): en la calibración del auto-etiquetado, de 22
  frames el refinamiento mejoró en ~9 y empeoró en ~13, a veces
  dramáticamente (33→283 px). **No usar `F.fit()` como refinamiento ciego.**
- **Bootstrap de homografía desde el contorno de la región, sin ninguna
  ancla** — dos intentos, los dos descartados por motivos distintos:
  - `minAreaRect` (fuerza un rectángulo) — mal por diseño: bajo perspectiva la
    pista es un cuadrilátero general, no un rectángulo girado. IoU de región
    decente (0.6-0.8) pero error geométrico ~1100 px.
  - Cuadrilátero real vía `approxPolyDP` (con las 8 combinaciones de
    rotación+espejo probadas y confirmadas, no es bug de correspondencia) —
    **sigue fallando igual de mal**. Causa real, diagnosticada con certeza:
    las "esquinas" detectadas muchas veces no son esquinas de la pista, son
    **donde la región de hielo queda recortada por el borde del frame**
    (verificado: 3 de 4 esquinas detectadas caían literalmente en x=0 o
    y=h-1). En vídeo de difusión la pista casi nunca cabe entera en el
    encuadre. No es un bug arreglable con más candidatos de correspondencia,
    es que el dato de entrada (4 esquinas reales) no existe en la mayoría de
    los frames.

## 5. Auto-etiquetado masivo (`training/auto_label.py`) — construido, calibrado, EN ESPERA

Objetivo del usuario: no usar la vía clásica para producción en tiempo real,
sino para generar muchas más etiquetas de entrenamiento offline. Para eso el
listón es mucho más bajo — no hace falta acertar siempre, solo saber decir
"de este me fío" y descartar el resto sin piedad.

Pipeline: YOLO ancla (H0) → [refinamiento clásico — **actualmente
desactivado**, ver arriba] → puerta de aceptación con 4 señales → si se
acepta, reproyecta los 56 keypoints y escribe la etiqueta.

**Calibración hecha sobre datos con ground truth — resultado: la puerta NO
discrimina todavía.**

```
correlacion con el error real, señal medida sobre H0 (ancla YOLO cruda):
  inliers_yolo   r=-0.47   (la unica con algo de señal)
  iou_contorno   r=-0.16
  zocalo_z       r=-0.08
  curvas_ok      r=+0.12   (sin utilidad, incluso mal signo)

mejor precision alcanzable con una sola señal: 38% (zocalo_z, recall 86%)
```

**Por qué**: las tres señales clásicas (contorno, zócalo, curvas) necesitan
que la H candidata ya esté razonablemente cerca (10-20 px) para buscar en la
ventana correcta. Con el error actual de YOLO (mediana 60-100 px, a veces
catastrófico) esas ventanas de búsqueda muchas veces no contienen lo que
buscan. No es un problema de umbral, es una limitación estructural de estas
herramientas de verificación: son refinadores de precisión, no rescatadores
desde lejos.

**Qué hacer con esto**: no lanzar `auto_label.py --video ...` en serio hasta
que un run de `--calibrar` dé precisión razonable (>80-90%). Cada vez que
termine un entrenamiento nuevo, correr:

```
python training/auto_label.py --calibrar --pesos <ultimo best_homography.pt> --n 60
```

y mirar si `inliers_yolo` (la señal con más correlación, r=-0.47) empieza a
separar bien. Es plausible que esto se resuelva solo según el modelo YOLO siga
bajando de error — revisar cuando la mediana de keypoint baje de ~20-25 px.

## 6. Segmentación semántica de líneas — pivote validado, con reserva de generalización

Motivado por revisión de literatura (Rink-Agnostic/Waterloo, SimulaMet
HockeyRink, KpSFR/TVCalib/NBJW/PnLCalib): en vez de 56 keypoints dispersos,
segmentar las líneas como clases de píxel. `make_line_masks.py` genera las
máscaras GRATIS (reproyectando `rink.py` por la H ya ajustada de cada
etiqueta), 12 clases (0=fondo, 1=vallas, 2-3=goles, 4-5=azules, 6=central,
7=círculo central, 8-11=faceoff — ver sección 7, no confundir con índices de
keypoints).

`training/train_lines_seg.py`: U-Net con encoder ResNet34 preentrenado en
ImageNet (no DeepLabV3 — sin decoder tipo "+" sale a 1/8 de resolución y
difumina líneas de 3-5 px), CE ponderada por clase (inverso de la raíz de la
frecuencia, no inverso puro — el desbalance es 1:2173 en el peor caso y el
inverso puro da pesos ~2000x que desestabilizan) + Dice loss, `ignore_index`,
flip horizontal con remapeo de clase (`FLIP_CLS`, involución verificada).

**IoU de val por sí solo es engañoso.** Entrenado 60 epochs sobre
`hockeyrink_lines` (574/61) + `hockeyrink_nhl_lines` (344/70) juntos: la loss
de train sigue bajando sin parar mientras `IoU_lineas` de val se estanca en
0.25-0.26 desde el epoch ~43 — sobreajuste, más epochs sobre estos mismos
datos no ayuda. Pero un IoU bajo no dice si la máscara está bien centrada y
ancha (útil) o sistemáticamente desplazada (inútil) — dos casos con IoU
parecido.

**El test que sí importa**: `training/diagnose_lines_seg.py` reutiliza el
mismo criterio de `diagnose_lines_fit.py` (deslizar cada curva del template
por su normal y ver dónde pica la respuesta) pero usando el mapa de
probabilidad de la red como respuesta en vez del color clásico — comparación
directa, mismo z-score, contra los números clásicos ya documentados arriba
(zócalo z=79.9/48.5).

```
python training/diagnose_lines_seg.py [--dataset hockeyrink_nhl] [--n N]
```

Resultado sobre val (frames que la red NUNCA vio en entrenamiento):

```
IIHF (48 frames val):  goleada a la via clasica.
  linea_gol_A/B, linea_central, faceoff, circulo_central: z de cientos a miles
  (el zocalo clasico, la mejor senal previa, daba z=79.9), desplazamiento
  mediano 0.5-2.5 px. 100% de aciertos en casi todas las clases.
  linea_azul_A floja: 56% acierta, d=5px, z=11.5 (coincide con ser la clase
  de IoU mas bajo en el training).
  83% de los frames: TODAS sus curvas aciertan.

NHL (57 frames val):  mucho mas debil.
  linea_azul_B y circulo_central practicamente fallan (d=35-40px, al limite
  del rango de busqueda -- ruido, no localizacion). Solo faceoff y
  linea_gol_A se sostienen (z=12-200, d~2px).
  Solo 46% de los frames con todas las curvas correctas.
```

**Interpretación**: la hipótesis del pivote queda validada — donde hay datos
homogéneos suficientes (IIHF), la red localiza órdenes de magnitud mejor que
cualquier cosa clásica de color. La brecha en NHL no es un fallo del método,
es falta de datos: 344 imágenes repartidas entre muchas retransmisiones y
pistas distintas, poca redundancia de apariencia por escena (frente a IIHF,
fuente mucho más homogénea). Coincide con lo que predice la literatura
(Rink-Agnostic soluciona esto exactamente con preentrenamiento sintético +
domain randomization).

**Hecho**: generador de datos sintéticos por domain randomization
(`training/gen_synthetic_lines.py`, 3000 imágenes generadas) para ampliar la
diversidad de apariencia sin depender de más etiquetado manual. Reutiliza el
pool de ~955 homografías reales ya ajustadas como geometría de cámara (no
inventa vistas de camara desde cero) y randomiza fuerte la apariencia (hielo,
pintura, zócalo, publicidad, graderío, jugadores toscos como oclusión). `val`
se queda siempre 100% real (nunca sintético) para que la medición no se
autoengañe. `train_lines_seg.py --synth` lo incluye por defecto con muestreo
balanceado 50/50 real/sintético (`WeightedRandomSampler`) y carga el
checkpoint anterior si existe (antes cada run empezaba de cero, se perdía lo
ya aprendido).

## 7. `training/seg_to_homography.py` — homografía directa desde la segmentación, sin keypoints

El paso que le faltaba al pivote de segmentación: convertir las curvas
identificadas por clase en una H real, no solo medir que localizan bien.
Ventaja sobre la vía clásica de color: allí el cuello de botella era la
CORRESPONDENCIA (qué línea del mundo es este píxel rojo); aquí la red ya dice
"esto es linea_azul_A", la correspondencia viene gratis.

**Resultado, exigiendo redundancia suficiente (dof≥10, ver más abajo por qué)**:

```
            cobertura   p50      p90      <25px
IIHF val      12%       16.4px   20.7px   100%
NHL val        7%       37.8px   156px     25%  (75% <60px)
```

Igual o mejor que YOLO (~61px mediana) en los frames que cubre, con datos de
vídeo real (val), sin usar ni un keypoint. La cobertura baja es el precio de
la fiabilidad (ver más abajo); con dof≥8 a secas la cobertura sube a 47%/31%
pero la precisión se hunde (p50 300-500px) — la redundancia extra es lo que de
verdad compra la fiabilidad, no solo un umbral más generoso.

Geometría: goles/azules/central son rectas de mundo X=cte, usadas como
correspondencias de RECTA (DLT dual, `l_mundo ~ H^T l_imagen`) — sabemos que
cada píxel de esa clase está en algún punto de esa recta, pero no en cuál
exactamente, así que muestrear puntos arbitrarios a lo largo de ella
inventaría una correspondencia que no existe. Círculos (central y faceoff) dan
correspondencia de punto (centro ajustado ↔ centro del template). Vallas
aporta una recta Y=cte, la única familia NO paralela a las demás.

**Seis bugs reales encontrados y corregidos, en orden**:

1. **DLT sin normalizar Hartley** — mezclar filas con magnitud ~1000 (px de
   imagen) y ~10-60 (metros de mundo) da un SVD numéricamente inútil aunque
   las ecuaciones sean matemáticamente correctas (verificado con self-test a
   escala unidad: pasa perfecto; a escala real: p50 sube a 1858px). Arreglado
   con `scale_transform()` (Hartley 2004, 4.4.4) antes de cualquier SVD.
2. **Arco de círculo recortado por el borde de imagen** — si el centro real
   cae fuera del frame, ajustar una elipse al arco parcial visible es un
   problema mal condicionado clásico (visto: 476px de error en un centro).
   Filtro: descartar si la máscara toca el borde, o si el arco cubre menos de
   360°-150° alrededor del centro ajustado (`fit_circle_center`).
3. **Líneas paralelas nunca fijan la dirección transversal** — goles/azules/
   central son TODAS X=cte en el mundo; por muchas que haya, no hay forma de
   que fijen por sí solas la otra dirección de H. Con 1 solo círculo cargando
   toda esa responsabilidad, su propio ruido de ajuste (10-20px) se amplifica
   sin freno (caso real: cada correspondencia a <16px de su verdad, resultado
   final a 2189px). Ni el número de condición de la SVD lo detecta (salía
   "limpio"). Arreglado añadiendo vallas (recta Y=cte, dirección distinta).
4. **Signo de vallas por residuo algebraico: falla 100%, no al azar** — no se
   sabe si la recta ajustada de vallas es Y=0 o Y=W sin conocer H. Elegir por
   el residuo del DLT (última singular value) se probó contra la verdad en 27
   frames reales: **acertó 0/27**. No es ruido, está anticorrelacionado (el
   residuo algebraico ya sabíamos que no discrimina bien, bug 3). Arreglado
   resolviendo DLT+refinamiento COMPLETO para cada signo candidato y
   comparando por el coste del refinamiento (error de reproyección real), que
   sí mide lo que dice medir.
5. **Pérdida robusta con `f_scale` mal calibrado mata el gradiente** — al
   refinar con `soft_l1` y `f_scale=0.05` (pensado para el residuo ya cerca del
   óptimo) sobre un punto de partida con residuo real de ~570px, la pérdida
   satura desde el primer paso y el optimizador "converge" (xtol) sin moverse.
   Arreglado con dos pasadas: 1ª sin robustez (loss lineal, gradiente completo
   en todo el rango) y 2ª robusta con `f_scale` calibrado al residuo YA CERCA
   del óptimo de la 1ª pasada, no al inicial.
6. **Semilla numéricamente degenerada (~1e14) hace inútil el refinamiento** —
   una H del DLT casi singular, al normalizar, puede dar coeficientes de
   ~10^14; con parámetros a esa escala un paso de optimizador normal no
   cambia nada detectable en float64 (Jacobiano con 6 de 8 columnas EXACTAMENTE
   a cero, confirmado). Arreglado con una guarda de sanidad numérica antes de
   refinar (`max|Hn|<50`) que descarta la semilla en vez de fingir refinarla.

**Confusión de clase (círculo/línea equivocados) — investigada y descartada**
como causa: comprobado contra la verdad en el val completo de IIHF, 47/47
círculos y 93/94 líneas coinciden con su clase más cercana bajo la H real. No
es donde estaba el problema (el problema era 3-6, arriba).

**dof≥8 (el mínimo matemático) no es suficiente** — con exactamente 4
correspondencias (cero redundancia) no hay margen para que nada delate una
configuración mala, incluso con direcciones NO paralelas y cada correspondencia
por debajo de 25px de su verdad (caso real: 2 círculos + 2 líneas en
direcciones distintas, aun así colapsó — columna Y de H prácticamente a cero).
Exigir dof≥10 (5+ correspondencias) es lo que de verdad compra fiabilidad.

**Herramienta de depuración visual**: `solve_from_probs(..., debug=True)` +
`dibuja_debug()` pintan sobre la imagen real: las correspondencias extraídas
(círculos en amarillo, líneas ajustadas en cian, vallas en naranja) y la
plantilla proyectada por la H real (verde) vs la H estimada (magenta) — donde
coincidan, va bien; donde no, ahí está fallando. Con `--figuras N` en
`seg_to_homography.py` guarda las N peores y N mejores en
`scratch_frames/seg_to_h/`. Fue clave para encontrar los bugs 3 y 6 (visualmente
obvio que la H estimada no se parecía en nada a una pista, pese a
correspondencias de entrada razonables).

**dof≥10 tampoco es perfecto en el límite exacto** — sobre una muestra más
amplia (split train, 72+67 frames), justo los casos con dof=10 raspado
(ninguna correspondencia de sobra) seguían dando fallos catastróficos de vez
en cuando (2 de 9 en IIHF: 1048px y 5294px). Arreglado con una SEGUNDA señal,
gratis porque ya se calculaba: el coste del refinamiento no lineal (mediana
del residuo final, espacio normalizado) correlaciona con el error real
**r=0.97 (log) en IIHF** — filtrar por `costo<=0.0009` (p75 de los aceptados)
tira esos dos casos y baja el p90 de 1897px a 16.4px, perdiendo solo 2/9
frames. En NHL la muestra de aceptados (n=6) es aun insuficiente para fiarse
de esa correlación (salió r=-0.16, con tan pocos puntos es ruido) — recalibrar
cuando haya mas frames NHL de referencia.

**Pendiente, no explorado todavía**: por qué la cobertura es tan baja (12%/7%)
— casi siempre por no alcanzar dof≥10, no por correspondencias erróneas. Subir
la cobertura sin perder fiabilidad probablemente necesita más clases
simultáneas en cuadro (mejorar el propio modelo de segmentación, sección 6).

## 7bis. `training/auto_label_seg.py` — auto-etiquetado vía segmentación, DESBLOQUEADO

Sustituye a `auto_label.py` (sección 5), que se quedó esperando a que YOLO
bajara de 20-25px. Mismo patrón (`--calibrar` antes de soltarlo sobre vídeo
crudo, escribe en formato YOLO reutilizando `label_from_H`/`write_label` de
siempre), pero la puerta de aceptación es `seg_to_homography.solve_from_probs`
(dof≥10) + el filtro de coste de refinamiento (`--max-costo`, por defecto
0.001, calibrado en IIHF — ver arriba). No hay umbrales de color que calibrar.

```
python training/auto_label_seg.py --calibrar --n 80          # repetir tras cada cambio
python training/auto_label_seg.py --video nhl5.mp4 --cada 5 --salida hockeyrink_auto_seg
```

**Probado sobre 6 vídeos reales** (`nhl5,6,7,8,9,10.mp4`, `--cada 5`) → 400
frames aceptados en total (2-4% según vídeo), en `datasets/hockeyrink_auto_seg/`.

**La recalibración con muestra grande (`--n 600`, ~955 frames de referencia)
corrigió el número optimista de arriba**: la correlación coste↔error real que
parecía r=0.97 (medida con solo 9 frames) baja a **r=0.35 en IIHF y r=0.31 en
NHL** con muestra grande. En NHL además el filtro NO mejora el caso peor:
incluso quedándote solo con el 50% de menor coste, el p90 de error sigue en
~487px. Conclusión: el número `r=0.97` de la sección 7 estaba sobreajustado a
una muestra pequeña, no es mentira pero tampoco hay que fiarse a ciegas del
coste como único filtro.

**QA visual manual — el hallazgo importante**: en vez de fiarte del coste,
`scripts/review_auto_seg.py` (ad-hoc, en el scratchpad de la sesión, no
commiteado) reconstruye la H desde la propia etiqueta escrita
(`RP.read_label` + `MM.fit_from_label`, igual que hace `calibrar()`) y dibuja
`rink.draw_rink()` encima del frame real — el mismo truco visual de siempre,
aplicado a una muestra de 4 frames al azar POR VÍDEO (24 en total) en vez de
solo 4 del primer vídeo. Resultado, muy desigual por vídeo, no ruido
aleatorio:

```
nhl5:   3/4 bien   (1 fallo suelto, ruido normal)
nhl6:   0/4 bien   -- NO es un partido NHL: banners "Alé Majadahonda",
                      banderas de Madrid/España -- pista amateur/juvenil
                      española. Plantilla RINK_NHL incorrecta para ese vídeo
                      sea cual sea su tamaño real.
nhl7:   4/4 bien   (San Jose)
nhl8:   4/4 bien   (Minnesota Wild)
nhl9:   0/4 bien   -- NHL Stadium Series (partido al aire libre, pista
                      temporal) -- geometria SI es NHL estandar, pero el
                      aspecto de las vallas/dasher parece confundir la
                      segmentacion. Sin diagnosticar la causa exacta todavia.
nhl10:  4/4 bien   (Vegas, Stanley Cup Final)
```

El coste de refinamiento no detecta estos dos fallos sistemáticos — lo más
probable es que una H internamente consistente pero basada en geometría
equivocada (nhl6) o en vallas mal segmentadas (nhl9) siga dando un residuo
"limpio" en espacio normalizado, igual que el bug 4 de la sección 7 (el signo
de vallas por residuo algebraico fallaba 0/27 y tampoco lo detectaba la SVD).

**Hecho**: fusionados solo nhl5/7/8/10 (329 de los 400 frames) en
`datasets/hockeyrink_nhl` (train), y regenerados `hockeyrink_nhl_rp` (743
imgs densificadas, para `train_pose.py`) y `hockeyrink_nhl_lines` (743
máscaras, para `train_lines_seg.py`). nhl6 y nhl9 se quedan fuera, sin tocar,
en `datasets/hockeyrink_auto_seg`.

**Lección para la próxima vez que se auto-etiquete un vídeo nuevo**: no basta
con mirar 3-4 frames del vídeo entero como se hizo la primera vez con nhl5 —
hay que muestrear varios frames POR VÍDEO/POR FUENTE antes de fusionar,
porque el fallo puede ser sistemático de todo un vídeo (geometría o cámara
distinta) y el filtro de coste no lo detecta.

## 7ter. ¿Seguir con YOLO o pivotar el solver? DECIDIDO: pivotar (cabeza aprendida)

Pregunta del usuario tras revisar el estudio de la sección 6: si la
segmentación ya da 4x más precisión que YOLO donde puede resolver (16-38px
vs 61px), ¿tiene sentido seguir invirtiendo en la arquitectura de 56
keypoints dispersos, o pivotar a otra cosa?

**Diagnóstico**: el cuello de botella no es el detector (la segmentación ya
gana), es el SOLVER. `seg_to_homography.py` resuelve cada frame con DLT
clásico + SVD (ver sección 7bis del texto: "DLT sin normalizar Hartley" etc.)
de forma independiente, sin memoria entre frames ni prior aprendido sobre
qué poses de cámara son plausibles. Por eso exige dof≥10 (5+ correspondencias
no paralelas simultáneas) y se niega a responder si no las hay — de ahí la
cobertura del 7-12%. Los papers que motivaron el pivote (TVCalib, NBJW-calib,
PnLCalib) no tienen ese problema porque no usan DLT clásico: usan una cabeza
de regresión APRENDIDA entrenada end-to-end con pérdida geométrica, que
interpola con pocas líneas visibles porque generaliza sobre miles de vistas
parciales vistas en entrenamiento — el DLT no puede hacer eso, cada frame es
un problema de álgebra lineal aislado.

**Qué es el DLT clásico, en una frase**: dado un conjunto de correspondencias
punto/recta mundo↔imagen, se arma un sistema lineal `A h = 0` (una fila por
correspondencia, ver `dlt_point_rows`/`dlt_line_rows` en
`seg_to_homography.py`) y se resuelve por SVD tomando el vector singular más
pequeño de `A` — es exacto y óptimo en mínimos cuadrados algebraicos SI hay
suficientes correspondencias bien repartidas, pero no tiene ninguna noción de
"esto no puede ser una pista de hockey real": si las correspondencias son
pocas o casi degeneradas (p.ej. todas paralelas), da una solución
matemáticamente válida pero geométricamente absurda, y nada en el álgebra lo
avisa (de ahí que en la sección 7 el número de condición de la SVD saliera
"limpio" en casos que eran un desastre real).

**Qué sería la cabeza aprendida, en una frase**: en vez de SVD, una red
pequeña (CNN ligera o incluso una MLP) que toma como entrada el mapa de
probabilidad por clase de `train_lines_seg.py` (12 canales, la misma salida
que ya existe) y regresiona directamente los 8 parámetros de H (o una
parametrización mejor condicionada: p.ej. la pose de cámara — rotación +
traslación + focal — en vez de la matriz 3x3 cruda), entrenada con una
pérdida de error de reproyección de puntos del template (no con los
parámetros de H directamente, que no son físicamente interpretables ni están
bien condicionados para backprop). El dataset para entrenarla ya existe:
cada frame de `hockeyrink`/`hockeyrink_nhl` tiene una H de verdad ajustada
(`fit_from_label`), así que es supervisión directa, no hace falta anotar
nada nuevo. La ventaja sobre el DLT: puede aprender "con estas 3 líneas
paralelas y este trozo de círculo, la pose más plausible es X" en vez de
negarse a responder.

**El experimento — HECHO, resultado: se estanca (de hecho empeora en la cola).**
Reentrenado `train_lines_seg.py` desde el checkpoint anterior (epoch 29) con
los 329 frames NHL reales nuevos (sección 7bis) + sintéticos, 80 épocas
(mejor checkpoint en epoch 31, `runs/lineas_seg/best.pt`; el `IoU_lineas` de
val se queda plano en 0.24-0.26 desde la época 1 — mismo patrón de
sobreajuste ya documentado en la sección 6). Repetido el mismo comando de
`seg_to_homography.py` con el mismo split de val:

```
                cobertura dof≥10      p50       p90      <25px
NHL   antes          7 %            37.8px    156px      25%
NHL   despues        5 %            56.0px    174px       0%    <- peor, no mejor
IIHF  antes         12 %            16.4px     20.7px    100%
IIHF  despues       14 %            15.5px     89.3px     88%   <- mediana igual, cola peor
```

**NHL no solo no sube, empeora en las tres métricas a la vez** (cobertura,
mediana y cola). IIHF se mantiene en mediana pero también empeora de cola
(p90 20.7→89.3px). Con muestras tan pequeñas (58 y 57 frames con H) hay que
leer esto con cautela — la diferencia de cobertura son 2-3 frames de nada —
pero la dirección es clara: NO es la mejora de "7%→15-20%" que habría
validado "solo hacía falta más diversidad de datos". Dos lecturas posibles,
no excluyentes:

1. **Techo estructural del solver** (la hipótesis original): frames con
   contenido genuinamente insuficiente para dof≥10, que ninguna mejora de red
   puede arreglar. Consistente con que `diagnose_lines_seg.py` SÍ mejoró en
   las clases que antes fallaban del todo (`circulo_central` d=35-40px→16px,
   `linea_azul_B` d=35-40px→26.5px) sin que eso se tradujera en más H's
   resueltas — la red localiza mejor pero sigue sin tener suficientes
   clases simultáneas en cuadro.
2. **Los 329 frames auto-etiquetados meten ruido de etiqueta neto**: su H de
   origen es del propio `seg_to_homography.py` (con su error real de
   16-38px), así que las máscaras derivadas de ellos son más ruidosas que las
   de las etiquetas verificadas a mano -- podría estar empeorando la
   PRECISION de la red en vez de (o ademas de) no ayudar a la cobertura.

**Decisión, con el usuario ya de acuerdo independientemente de este resultado
("tiene sentido este tipo de cabeza, vamos a hacerla")**: construir la cabeza
aprendida (siguiente paso). Este experimento no distingue entre las dos
lecturas de arriba, pero tampoco hace falta que lo haga para justificar la
decisión -- en ambos casos "más datos con el DLT clásico" no es el camino
que sube la cobertura, así que el pivote al solver aprendido está motivado
de cualquier forma.

En cualquiera de los dos casos, YOLO como arquitectura de keypoints dispersos
pierde prioridad frente a la vía de segmentación — la pregunta que decide el
experimento es solo si además hace falta sustituir el DLT por un solver
aprendido, o si mejorar la segmentación ya es suficiente.

## 7quater. `training/homography_head.py` — cabeza aprendida de pose de cámara (en marcha)

Implementación del pivote decidido en 7ter. Dos ficheros nuevos:

- `training/cache_seg_probs.py`: precalcula y cachea el mapa de 12 canales de
  `train_lines_seg.py` reescalado pequeño (128x72 por defecto) + la H de
  verdad, un `.npz` por frame, en `datasets/<nombre>/seg_probs/<split>/`.
  Necesario porque repetir el forward de segmentación en cada época de la
  cabeza (a ~3-19 fps) costaría minutos por época solo en eso. Cacheados
  1151 train / 115 val entre `hockeyrink` + `hockeyrink_nhl`.
- `training/homography_head.py`: arquitectura + entrenamiento.

**Arquitectura, y por qué**:

- **Entrada**: el mapa de 12 canales ya existente (softmax de
  `train_lines_seg.py`), no la imagen cruda — la red de segmentación ya hizo
  el trabajo de decidir qué es cada píxel.
- **Salida: pose de cámara, NO los 8 coeficientes de H.** Regresionar H cruda
  es justo el problema que ya vimos en la sección 7 (bug 6: coeficientes
  normalizados a ~1e14, pésimamente condicionado para gradiente). En su
  lugar: rotación en **representación 6D** (Zhou et al. 2019 — Euler y
  cuaternión tienen discontinuidades que rompen el aprendizaje por
  gradiente cerca de los límites de rango; 6D + Gram-Schmidt es continua en
  todo el dominio), centro de cámara en metros de mundo (3 números, acotado
  y físicamente interpretable), y focal (1 número). De ahí, H se construye
  analíticamente vía el modelo pinhole estándar para un plano
  (H = K·[r1 r2 t]) — misma matemática que ya usa `seg_to_homography.py`,
  pero con parámetros de entrada bien condicionados.
- **Pérdida de reproyección** sobre los 56 puntos de `rink.py`, no MSE sobre
  los parámetros — mismo principio que `refine_nonlinear`. Huber (no L2
  puro), por la misma lección que calibrar `f_scale` en el refinamiento
  clásico (bug 5, sección 7): un punto muy fuera de cuadro puede dar miles
  de px de error y ahogaría el gradiente sin saturar. **Los 56 puntos se
  supervisan SIEMPRE, estén o no en cuadro** — la pieza central de todo esto:
  así la red aprende a extrapolar una pose plausible con pocas pistas
  visibles, la propiedad que al DLT por-frame le falta por diseño.
- **Cobertura 100% por construcción**: no hay puerta de aceptación ni dof
  mínimo — la red siempre da una H, con la precisión que pueda. Cambia el
  problema de "0% u homografía buena" a "homografía que degrada suave con
  menos pistas".
- Init de las capas de salida NO aleatorio: sesgado hacia una pose típica de
  retransmisión (cámara alta, lateral, mirando al centro) para converger más
  rápido que arrancando de una pose degenerada.

**Cuatro bugs/decisiones de diseño reales, en orden**:

1. **`f` (focal) con forma `(B,1)` multiplicada por `w` con forma `(B,)` sin
   squeeze** — el broadcasting de PyTorch lo convertía en `(B,B)` en vez de
   dar error claro, y `pose_to_H` fallaba con un stack de tamaños
   inconsistentes. Corregido devolviendo `f` ya como `(B,)`.
2. **`AdaptiveAvgPool2d(1)` al final del encoder colapsaba TODA la
   información espacial antes del MLP.** Primer entrenamiento completo (200
   épocas): p50 de val estancado en ~1000px, `<25px` en 0% todo el run —
   muy por debajo incluso del DLT clásico. Diagnosticado inspeccionando la
   pose predicha en 8 frames de val distintos: el centro de cámara predicho
   era casi idéntico entre arenas completamente distintas (variación <3% en
   focal) — la red había colapsado a ~2 poses en vez de diferenciar cada
   frame. Causa: un vector global-average-pooled no distingue "la línea azul
   está a la izquierda" de "está a la derecha" si la activación media por
   canal es parecida — exactamente la información que promediar el espacio
   entero destruye, para una tarea que es literalmente "dónde está cada
   cosa". Arreglado con dos cambios: (a) canales de coordenada explícitos
   (CoordConv, Liu et al. 2018) concatenados a la entrada, (b) pool final a
   una rejilla 4x6 en vez de a 1x1, conservando disposición espacial gruesa
   antes de aplanar. Verificado en un sanity-check de 5 épocas: los centros
   de cámara predichos pasaron de 2 clusters casi fijos a un rango continuo
   y creíble entre frames distintos.
3. **Normalizar H por `H[2,2].clamp(min=1e-6)` no preserva el signo.**
   Con el pooling ya arreglado, el segundo entrenamiento completo (200
   épocas) seguía estancado en p50≈890-910px de val, sin bajar apenas entre
   la época 100 y la 200 — ya no olía a "falta de entrenar", olía a un techo
   estructural. Diagnosticado construyendo una H a mano con una pose
   razonable (rotación identidad, C=(30,-25,18), f=2200) y comparando el
   resultado con el cálculo por álgebra: `H[2,2]` antes de normalizar daba
   -18 (negativo, normal según la orientación de cámara), pero
   `.clamp(min=1e-6)` lo subía a +0.000001 en vez de dejarlo en -18 — dividir
   toda la matriz por eso la disparaba x~10⁷ (confirmado exacto: factor
   18/1e-6). La red llevaba dos entrenamientos completos compensando a
   duras penas una normalización rota en vez de aprender la pose. Arreglado
   separando signo y magnitud: `H / (signo(H[2,2]) · |H[2,2]|.clamp(min=1e-6))`.
4. **Supervisar la posición en píxeles de puntos MUY fuera de cuadro es un
   objetivo mal condicionado — el bug de diseño real, no de implementación.**
   Con los tres bugs de arriba corregidos, el tercer entrenamiento completo
   seguía clavado en p50≈840-910px, casi sin mejorar de la época 100 a la
   200. Verificado que `project_batch` (torch) coincide EXACTO con
   `MM.project_points` (numpy, ya validada) sobre la misma H de verdad —
   no era un bug de proyección. La causa: de los 56 puntos del template,
   muchos caen fuera de cuadro (19/56 en un frame cualquiera), algunos a
   decenas de miles de px (visto: -66000px) — un error angular mínimo mueve
   esa posición miles de px, así que exigir precisión en píxeles sobre ella
   es imposible por diseño, y con 56 puntos así de inestables en cada frame
   el entrenamiento nunca afina. Arreglado recortando (`clamp`) tanto la
   predicción COMO la verdad a una caja ampliada (1 ancho de frame de margen
   a cada lado) antes de medir distancia: un punto cuya verdad ya cae fuera
   del margen deja de pedir precisión (basta con que la predicción también
   caiga fuera, en la misma zona) — degradación suave en vez de objetivo
   imposible. Confirmado en sanity-check de 5 épocas: p90 pasó de
   12000-14000px a 1200-3400px, y el p50 "en cuadro" (el comparable con
   `seg_to_homography.py`) bajó de 896 a 589px en solo 5 épocas.

**Limitación conocida, no arreglada todavía**: `--init` para continuar un run
solo restaura los pesos del modelo, no el estado del optimizador/scheduler —
el coseno de LR se reinicia desde el pico, dando un salto que empeora las
primeras épocas antes de recuperarse. No afecta a un run limpio (sin
`--init`), que es como se está entrenando ahora.

**Resultado del run completo (200 épocas, los 4 fixes aplicados): NO sirve
todavía.** Val en cuadro: p50≈340px, p90≈1150px, `<25px`=1%; la loss de train
se queda plana en ~25.9k, que con Huber δ=50 equivale a un error medio de
train de ~500px → **infraajuste, no sobreajuste** (la red ni siquiera ajusta
el train). Muy lejos del DLT clásico (NHL 56px, IIHF 15.5px) y de YOLO (~61px).
`best.pt` (epoch 199, 339px) es indistinguible de las épocas vecinas (342-350).

Dos diagnósticos sin entrenar (scripts ad-hoc, no commiteados) que acotan dónde
está el problema:

1. **Techo de la parametrización** (ajustar R, C, f directamente a la H de
   verdad de cada frame de val, sin red): **p50=4.65px, p90=40px, 85%<25px**
   en cuadro (init cerrada por rejilla de f + Adam). Focal necesaria: 850 a
   6800px (mediana 2970). Conclusión: el modelo pinhole con punto principal
   fijo REPRESENTA bien las H; el fallo es aprender la regresión, no la
   parametrización.
2. **Línea base sin entrenar: vecino más cercano por similitud coseno del
   mapa de probabilidad (train→val)**: p50=121px, p90=853px, 11%<25px,
   30%<60px. Ya ~3x mejor que la cabeza entrenada. Y un diccionario de las
   1151 H de train cubre val con p50=50px, p90=100px (oráculo: mejor vecino
   en H) — techo de la recuperación pura. Ojo: val puede tener frames casi
   duplicados de train (mismos clips), así que 121px puede ser optimista.

3. **Experimento: vecino más cercano (K=8) + refinado diferenciable de la pose
   contra el mapa de segmentación** (scratch `refine_from_nn.py`, no
   commiteado; muestreo bilinear de los mapas a 256x144 en las polilíneas del
   template, pirámide de blur σ=6→0.8, Adam sobre rot6D/C/logf, 550 pasos,
   ~40 s/frame). **Resultado NEGATIVO** sobre 16 frames de val (9 IIHF, 7 NHL),
   error en cuadro:

   ```
                                     TODOS p50   IIHF p50   NHL p50   <25px
   vecino top-1, sin refinar           142px       53px      543px     9%
   top-1 refinado                      324px      234px      374px     3%   <- EMPEORA
   mejor de K por score del mapa       209px      126px      293px    21%
   mejor de K por error real (oráculo) 154px       35px      169px    23%
   ```

   El refinado ingenuo aleja la pose de la verdad (IIHF 53→234px), y ni el
   oráculo sobre los K refinados baja de 154px. El vecino en NHL es muy malo
   (543px: pistas/cámaras mucho más variadas que IIHF). Comprobación del
   objetivo: puntuando la pose de VERDAD frente al vecino de inicio,
   score(verdad) > score(vecino) en 30/40, 31/40, 28/40 y 25/40 frames según
   σ=6/3/1.5/0.8 — el objetivo es informativo en promedio. **Comprobación
   decisiva (12 frames, score a σ=0.8, pose de verdad frente al mejor
   refinado de K): en 8/12 el refinado puntúa MÁS que la verdad** → el
   objetivo (masa de probabilidad de segmentación muestreada en las
   polilíneas) tiene máximos espurios más altos que la pose correcta, y en
   los otros 4/12 la verdad puntúa más (fallo de optimización). Dominante:
   objetivo mal planteado, no optimizador. Causa probable: la propia
   segmentación es poco fiable (IoU val ≈0.25, clases NHL débiles), así que
   cualquier objetivo que confíe en ella tiene máximos falsos.
   **Trampa del propio experimento, corregida a mitad**: la primera versión
   inicializaba con una descomposición H→(R,C,f) cerrada (rejilla de f) que
   por sí sola da ~57px de error (medido en el oráculo), lo que ensuciaba tanto
   los inicios como la "pose de verdad" del diagnóstico; con la
   descomposición refinada por Adam el score de la verdad sube (0.001-0.05 →
   0.02-0.16) pero la conclusión no cambia: top-1 refinado p50=558px frente a
   148px del vecino sin refinar (12 frames).

**Estado tras todo esto (honesto)**: ninguna vía aprendida por ahora supera al
DLT clásico (IIHF 15.5px / NHL 56px, cobertura 14% / 5%) — cabeza directa
~340px, vecino ~120px (optimista), vecino+refinado peor que el vecino. Y a
cobertura completa el mejor número medido sigue siendo YOLO (~61px). El cuello
de botella común parece la calidad de la segmentación (IoU ≈0.25), no el
solver. Línea razonable: DLT como puerta de alta confianza para auto-etiquetar
(ya funciona) y reentrenar YOLO con los 329 frames NHL ya fusionados
(`hockeyrink_nhl_rp` regenerado) en vez de seguir con la cabeza.

```
python training/cache_seg_probs.py                 # una vez
python training/homography_head.py --epochs 200
```

## 7quinquies. Cobertura del estudio de literatura (`docs/hockey_research.md`) — qué se probó y qué no

El estudio (Rink-Agnostic/Waterloo, HockeyRink/SimulaMet, TVCalib, NBJW/PnLCalib,
Sharma, Chen&Little, Jiang, Shi...) recomienda 5 etapas. Estado y resultado de cada una:

| Etapa / método | ¿Hecho? | Resultado medido |
|---|---|---|
| **0** Plantilla NHL 61×26 | Sí | `RINK_NHL`/`RINK_IIHF` separados; 2 bugs de plantilla corregidos; techo 1.0px con keypoints de verdad |
| **1** Segmentación de líneas (+vallas) | Sí (U-Net ResNet34, no DeepLabV3+) | IIHF excelente (0.5-2.5px, z cientos-miles); NHL flojo (azules y círculo central casi fallan), IoU val plano ~0.25 |
| **2** Preentreno sintético + domain randomization | Sí (2700 sint., 50/50) | Mixto: `circulo_central` 35-40→16px de desplazamiento, pero IoU plano y "todas las curvas aciertan" 46%→39% |
| **3** Preanotador clásico (Hough/elipse/chamfer) | Parcial, sin CVAT/Nuclio | Hough entero: 1/122 segmentos; elipse-semilla: falla (señal=ruido); cuadrilátero de contorno: falla; croma local: solo sirve con H a 10-20px |
| **4** DLT + refinado + suavizado temporal | DLT sí; refinado TVCalib y suavizado no/negativo | DLT IIHF 15.5px / NHL 56px con cobertura 14% / 5%; KLT p50 158px (deriva en reflejos) |
| **5** Adaptación de dominio (MIC, vídeo sin etiquetar) | **No** (en curso, ver 7sexies) | — |
| Keypoints/pose (HockeyRink) | Sí, 17 runs YOLO26 | 88.6→61.4px de mediana por keypoint; objetivo ~8px no alcanzado; 5 bugs de entrenamiento; mAP alto con H inservible (confirmado) |
| Regresión directa de pose (DeTone/Waterloo) | Sí (`homography_head.py`) | ~340px, infraajusta; techo de la parametrización 4.65px |
| Diccionario por recuperación (Sharma) | Parcial (mapas reales, no sintéticos/chamfer) | vecino p50 121px (optimista); oráculo del diccionario 50px |
| Refinado diferenciable (TVCalib-like) | Sí, versión ingenua | Negativo: 8/12 frames refinado puntúa > verdad (objetivo mal planteado) |
| **No probado**: pesos/código NBJW-PnLCalib, TVCalib real, keypoints por intersección/tangente (Sportlight, 57 pts), Jiang (error aprendido), Shi (autosupervisado), diccionario sintético pan/tilt/zoom+chamfer, perturbación aleatoria de H como aumentación de la cabeza, copy-paste de jugadores REALES, logo augmentation | | |

**Regla de decisión del propio estudio ("Benchmarks que cambian el plan")**: si la
segmentación falla en una arena nueva → invertir en UDA (etapa 5) y más aumentación
de apariencia. Es justo lo medido (NHL: IoU ~0.25, azules y círculo débiles), y
las sesiones de solver (7ter-7quater) fueron etapa 4 antes de arreglar la etapa 1-2.
**Hipótesis mía sin probar** sobre la baja cobertura del DLT: contamos cada
línea/círculo como UNA correspondencia; PnLCalib/Sportlight ganan redundancia con
muchos puntos derivados (intersección línea×valla, tangentes de círculo).

## 7sexies. Etapa 5 del estudio: aumentación de apariencia + adaptación de dominio (MIC) — EN CURSO

Siguiendo la regla de decisión de 7quinquies (la segmentación falla en NHL → UDA y
más aumentación), tres piezas nuevas, todas opt-in (el comportamiento por defecto de
`train_lines_seg.py` no cambia):

- `training/appearance_aug.py` (`--aug-apariencia`): **copy-paste de jugadores
  REALES** (recortes de las cajas goalie/player/referee de `datasets/hockeyai`,
  1500 recortes, borde difuminado para que el rectángulo no sea una pista trivial)
  y **logo augmentation** (rectángulos/elipses/texto de color aleatorio pintados
  SOLO sobre píxeles de fondo, nunca sobre línea). Las etiquetas de línea no se
  tocan: la máscara es geometría reproyectada, no visibilidad, así que la red
  aprende a inferir la línea bajo un jugador. Verificado a ojo sobre un frame
  (`scratch_frames/aug_check.jpg`). Los workers del DataLoader se re-siembran
  (copias con el mismo rng habrían pegado los mismos jugadores).
- `training/extract_uda_frames.py`: 983 frames SIN etiquetar (1 cada 1.5 s) de
  nhl3,4,5,7,8,9,10 en `datasets/uda_target/`, a 1024x576. Excluye nhl6 (pista
  amateur, otra geometría) y todo frame a <300 frames de un frame de VAL del mismo
  vídeo (para que la adaptación no vea vecinos casi idénticos de lo que se mide).
- `train_lines_seg.py --uda`: **MIC** (Hoyer et al. 2023) — profesor EMA (α=0.999)
  que da pseudo-etiquetas sobre el frame objetivo SIN tapar (umbral de confianza
  0.7 líneas / 0.95 fondo, ponderadas por la fracción fiable, estilo DAFormer) y
  alumno entrenado sobre el mismo frame con parches de 64px tapados al 50%;
  λ=0.5 con rampa de 3 épocas.
- `diagnose_lines_seg.py`: `PESOS` ahora se puede cambiar con la variable de
  entorno `LINEAS_PESOS` (también lo respetan `seg_to_homography.py`,
  `auto_label_seg.py`, `cache_seg_probs.py`, que importan de ahí) — para evaluar un
  checkpoint experimental sin pisar `runs/lineas_seg/best.pt`.

**Experimento lanzado**: 40 épocas desde `runs/lineas_seg/best.pt` (epoch 31), con
`--aug-apariencia --uda`, salida en `runs/lineas_seg_uda/` (NO pisa el modelo actual).

```
python training/extract_uda_frames.py            # una vez
python training/train_lines_seg.py --epochs 40 --aug-apariencia --uda --salida runs/lineas_seg_uda
# evaluar SIN pisar nada (mismo val, mismos N):
set LINEAS_PESOS=runs\lineas_seg_uda\best.pt
python training/diagnose_lines_seg.py --dataset hockeyrink_nhl --n 100
python training/seg_to_homography.py --dataset hockeyrink_nhl --n 100
```

**Resultado intermedio (snapshot época 24 de 40, evaluado en CPU para no competir
por la GPU): NO mejora.** IoU de val: el mejor checkpoint del run es la época 1
(0.230 vs 0.257 de la referencia, todas las clases por debajo). `diagnose_lines_seg`
NHL val (56 frames; 10-25 curvas por clase, un fallo = ±8pp, hay que leerlo con
cautela): mejoran `azul_B` 42→50% (d 26.5→11.5px), `azul_A` 23→31%, "todas
aciertan" 39→43%; empeoran `circulo_central` 30→10%, `linea_central` 50→42% (d
12→30px), `linea_gol_B` 42→33%, `linea_gol_A` z 78→25. `seg_to_homography`: cobertura
3% (≈2 frames), p50 255px (referencia 5% / 56px). Sin ganancia clara.

**Resultado FINAL (40 épocas, `runs/lineas_seg_uda/last.pt`): sin mejora neta, NO
adoptar.** IoU val: mínimo 0.194 (ep 13), recupera a ~0.229, mejor 0.231 (ref 0.257).
`diagnose_lines_seg` NHL val vs referencia: sube `azul_A` 23→54%, `linea_gol_B`
42→58%, `faceoff_B_lo` 75→90%; baja `circulo_central` 30→10%, `vallas` 49→37%,
`linea_central` 50→42%, `gol_A` 100→88%; "todas aciertan" 39→**32%**, mayoría
54→46%. `seg_to_homography`: NHL cobertura 5→7% (≈+1 frame) pero p50 56→**338px**
(p90 614); IIHF 12% / p50 13.7px / p90 114px / 86% <25px (≈ igual que antes).
No se puede separar el efecto de la aumentación del de UDA (se lanzaron juntas).
**Limitación de fondo de TODAS las decisiones recientes**: el val NHL son ~56
frames con 10-25 curvas por clase; un fallo = ±8pp, así que cambios de ±10-15pp por
clase (como los de arriba, en las dos direcciones) no son distinguibles del ruido.
Antes de seguir comparando modelos hace falta un val NHL mucho mayor y verificado.

**Cómo se decide** (referencia = modelo actual, NHL val): `diagnose_lines_seg`:
azul_A 23% / azul_B 42% / central 50% / círculo 30% de acierto, "todas aciertan" 39%;
`seg_to_homography` cobertura 5%, p50 56px. Mejora real = suben las clases débiles
Y la cobertura dof≥10; el IoU de val solo, ya sabemos que no basta (sección 6).
Ojo con el riesgo conocido de UDA con pseudo-etiquetas: si el profesor casi no ve
líneas en NHL, las pseudo-etiquetas son casi todo fondo y el alumno puede aprender
a no predecir líneas — comprobarlo mirando que las clases NO caen.

## 7septies. Val NHL ampliado y verificado a ojo (`hockeyrink_nhl_valx`) — 90 frames, con sesgos

`training/build_val_nhl.py` (`--generar` / revisión visual / `--finalizar`): muestrea
1 frame/s de nhl3,4,5,7,8,9,10 en zonas a ≥120 frames de cualquier frame de train,
corre los DOS modelos (referencia y UDA, para no favorecer a ninguno) con
`solve_from_probs` (dof≥10, sin filtro de coste), guarda candidatos + hojas de
contacto 2x2 (amarillo = A, cian = B) en `datasets/nhl_val_cand/` y, tras revisar
cada hoja a ojo, escribe `datasets/hockeyrink_nhl_valx/` = val original (73) +
aceptados. Se usa con `--dataset hockeyrink_nhl_valx` (el nombre contiene "nhl", así
que `diagnose_lines_seg`/`seg_to_homography` eligen la plantilla NHL solos).

**Rendimiento real**: ~960 frames muestreados → 35 candidatos (3.6%) → **17 aceptados
(49%)**: 10 nhl4, 1 nhl8, 6 nhl9. Rechazados 18: mallas de líneas caóticas, planos
de portería, círculos que no encajan. Total valx = 73 + 17 = **90** (72 con curvas
evaluables), no los 150-200 buscados: el cuello de botella es que el generador solo
propone frames donde el DLT resuelve.

**Sesgos conocidos (leer antes de fiarse de valx)**:
1. Solo frames "fáciles" (≥5 correspondencias en cuadro): sirve para medir
   localización con más muestra, NO la cobertura del DLT (circular).
2. 10/17 nuevos son nhl4 consecutivos (1 s entre ellos, mismo plano largo,
   centro de hielo): muestras muy correlacionadas, N efectivo bastante menor.
3. Los 6 de nhl9 los resolvió el modelo UDA (B) y el UDA vio frames SIN etiquetar de
   nhl9 cada 1.5 s durante el entrenamiento (val no está excluido de UDA, solo de
   train): ventaja transductiva para B en esos frames.
4. La verificación visual a 1000px de ancho (hoja 2x2) detecta fallos gruesos, no
   errores de 10-20px: un H "aceptado" puede tener ese error.

**Resultado (valx, 72 frames con curvas) referencia vs UDA final**: gol_A 100/88,
faceoff_A 92/92, faceoff_B_lo 75/90, `linea_central` 67/67, `azul_B` 67/62, `azul_A`
50/**73**, `vallas` 50/**38** (n=118), `circulo_central` 48/40, `gol_B` 42/58, "todas
aciertan" 42/**33**, mayoría 56/49. **Sin ganador claro (otra vez)**: UDA gana en azul_A,
faceoff_B_lo, gol_B; la referencia en vallas (la clase con más muestra, ≈1.8σ), gol_A y
"todas aciertan". Con más muestra `circulo_central` de la referencia sube de 30% a 48%
y `linea_central` de 50% a 67% (los frames nuevos son centro de hielo, más fáciles):
las cifras absolutas del val anterior estaban pesimistas por composición, no solo por ruido.
Veredicto sin cambios: no adoptar UDA+aumentación.

**Siguiente paso real para un val independiente y grande**: etiquetado humano de unos
pocos puntos por frame (4-6 clics sobre puntos reconocibles: intersecciones línea×valla,
centros de faceoff) + `fit_from_label`; así entran también los frames difíciles y se
rompe la circularidad con el DLT. `streamlit-image-coordinates` ya está instalado
(lo usa `testing/classical_cv_lab.py`).

### Etiquetador por clics (`testing/annotate_val_app.py`, lógica en `training/click_labeler.py`)

Val independiente del DLT: TÚ marcas 4-8 puntos reconocibles por frame y se ajusta la
H por mínimos cuadrados. Entran también frames DIFÍCILES (no se filtra por si el DLT
resuelve). Salida en `datasets/hockeyrink_nhl_valh/{images,labels}/val` (+
`clicks.jsonl` con los clics originales, `skipped.json`), misma estructura que valx →
`--dataset hockeyrink_nhl_valh`.

```
venv/Scripts/python.exe -m streamlit run testing/annotate_val_app.py
```

Flujo: clic en un punto del minimapa (se pone rojo, 56 puntos numerados de `rink.py`)
→ clic en ese punto del frame → repetir; con ≥4 puntos se dibuja la plantilla en
amarillo y hay tabla de residuos por punto (`Quitar` el que falle) → Guardar y siguiente
/ Saltar. Los frames salen de nhl3,4,5,7,8,9,10 (sin nhl6), 1 cada 2 s, a ≥120 frames de
cualquier frame de train, barajados con semilla fija (482 candidatos), sin repetir los
ya anotados/saltados (se puede cerrar y seguir otro día).

**Precisión medida con verdad conocida** (19 frames de val, 6 puntos con ruido de clic
gaussiano de 1.5px): error de las líneas **p50 2.0px / p90 2.4px**, 10x mejor que las
etiquetas automáticas (16-56px). **Convención de clases**: medido sobre las H de
train, +x→derecha en el 99% (zona A a la izquierda); la pista es simétrica, así que
`click_labeler.canonicalizar` aplica el espejo en x cuando la H sale al revés (dibuja
las mismas líneas, solo cambia qué clase es A/B) — no hay que pensar en A/B al clicar.
El eje y no se fuerza (en NHL los datos tienen ambas orientaciones, 76/24; faceoff
lo/hi es intrínsecamente ambiguo). Probado con AppTest (render, ajuste, guardado)
inyectando puntos por estado de sesión; los clics reales sobre el componente de imagen
no se pueden simular, así que esa parte la valida el primer uso.

**Marcadores arrastrables y aviso de estabilidad (etiquetador)**: con ≥4 puntos, el
resto de puntos de la plantilla que la H sitúa en cuadro salen como marcadores naranja
(proyectados con la H SIN espejar: con la canonicalizada saldría el punto simétrico) y
se arrastran a su sitio (`streamlit-image-coordinates` 0.4.1 `click_and_drag`); simulado
con verdad conocida, 4 puntos + ~1.5 arrastres dan las líneas a 1.2-1.7px (p50, p90 3px).
**Error a no repetir**: una primera guarda rechazaba 4 puntos si su casco cubría <6% de
la pista o <2% del frame — bloqueaba justo los planos cerrados, donde solo se ve un
trozo pequeño y no se pueden pedir esquinas opuestas. Sustituida por
`click_labeler.sensibilidad`: cuánto se mueve la plantilla SOBRE LA PISTA (no la grada:
promediar toda la imagen avisaba en el 91-100% de los casos) si cada clic se desvía 2px;
error real ≈ 0.42×esa medida (calibrado en 352 configuraciones), umbral 30 (avisa al 56%
de las configuraciones de 4 puntos aleatorios; de las NO avisadas solo el 6% supera
15px de error real). Es un aviso, NO bloquea; solo bloquean colinealidad numérica y
"casi toda la plantilla cae fuera del frame". El orden de candidatos es estable
(se baraja la lista completa y se filtran los hechos después).

### Primeras medidas sobre `hockeyrink_nhl_valh` (76 frames anotados A MANO, 71 con curvas)

Val independiente del DLT (frames al azar, incluye difíciles: nhl9 ×20, nhl10 ×18,
nhl4 ×15, nhl8 ×11, nhl3 ×7, nhl5 ×3, nhl7 ×2) y con H de ~2px de error (medido en
simulación). Sesgo residual que queda: el UDA vio frames SIN etiquetar de estos vídeos
(ventaja para B); aun así:

```
                        referencia (best.pt)   UDA final
faceoff_A_hi                69%                   88%
faceoff_A_lo                90%                   90%
faceoff_B_hi                95%                   89%
faceoff_B_lo                83%                   78%
linea_gol_A                 80%                   60%
linea_gol_B                 76%                   81%
vallas (n=128)              70%                   71%
linea_azul_A (n=10)          0% (d=37px)          20%
linea_azul_B                60%                   53%
linea_central               50%                   33%
circulo_central (n=11)      36%                    9%
todas aciertan              46%                   42%
```

Con n=10-30 curvas por clase ninguna diferencia suelta es significativa, pero la
dirección es consistente: **la referencia gana o empata en casi todo** (gol_A +20,
central +17, círculo +27) y UDA solo gana en faceoff_A_hi y azul_A. Refuerza "no adoptar
UDA+aumentación". Las clases que fallan en AMBOS modelos son las de la zona neutral:
**azul_A/azul_B, línea central y círculo central** — ahí está el problema real de la
segmentación en NHL.

**DLT sobre frames al azar (sin filtrar por que resuelva): cobertura 0% (referencia) /
4% (UDA, con H basura: p50=inf).** Coherente con el 2-4% de aceptación medido en vídeo
crudo: sobre imágenes NO seleccionadas el DLT prácticamente nunca dispara. Los 5-14% de
"cobertura" de secciones anteriores eran sobre val curado por humanos para tener
keypoints visibles.

**YOLO (yolo26m-17, `best_homography.pt`) sobre valh**: cobertura 88.2% (9/76 sin H),
error p50 **102px** (no los ~61px medidos sobre el val antiguo), p90 3195px, 0% <10px,
10.5% de TODOS los frames <25px, inliers RANSAC 54%. Es el número a batir; a cobertura
casi completa sigue siendo el mejor modelo entero, pero está lejos de ser fiable.
(`rink_metric.template_for` ahora acepta nombres con sufijo, p.ej. `hockeyrink_nhl_valh`.)

**Uso recomendado de aquí en adelante**: valh es el val de referencia (crece según se
anota; conviene fijar un subconjunto para comparar entre modelos, o comparar siempre
sobre el mismo N). Los frames anotados a mano son además la etiqueta más precisa que
existe (2px frente a 16-56px de las automáticas): valorar usarlos también como TRAIN
para casos difíciles, con un split por vídeo/zona para no contaminar la evaluación.

## 8. Herramientas de diagnóstico, para cuando algo no cuadre

- `python training/line_explorer.py [--dataset hockeyrink_nhl] [--video nhl4] [--n N] [--solo-con-circulo]`
  — matplotlib interactivo, 11 etapas del procesado clásico sobre frames reales
  (ordenados de peor a mejor). Flechas = frame, arriba/abajo o 1-9 = etapa,
  `c` = siguiente curva (perfil de respuesta al deslizar), `r` = recalcular.
- `python training/diagnose_lines_fit.py [--dataset ...] [--n N]` — por cada
  curva del template, dónde cree ELLA que está (desliza por su normal). Tabla
  de qué clases son de fiar y qué frames son recuperables. Genera figuras
  peores/mejores con la curva real (continua) vs donde pica (discontinua).
- `python training/ice_lines_probe.py [--dataset ...] [--n N]` — banco de
  comparación de métodos de segmentación de región + medida de localización
  de curva. Genera figuras de 6 paneles.
- `python training/fit_homography_lines.py [--n N] [--sigmas 0,10,20,40]` —
  mide radio de captura del ajuste conjunto (ya sabemos que es malo; útil solo
  si se retoma esa vía).
- `python training/make_line_masks.py [--overlay N]` — genera las 1049
  máscaras de línea (gratis, sin anotar) para entrenamiento de segmentación.
  Datasets: `hockeyrink_lines/`, `hockeyrink_nhl_lines/`.
- `python training/diagnose_lines_seg.py [--dataset hockeyrink_nhl] [--n N]`
  — igual que `diagnose_lines_fit.py` pero con el mapa de probabilidad de la
  red de segmentación (`runs/lineas_seg/best.pt`) en vez de color clásico.
  Es el test real para saber si la red localiza bien, no el IoU de val (ver
  sección 6).
- `python training/gen_synthetic_lines.py --n N [--overlay]` — genera dataset
  sintético por domain randomization (ver sección 6). `--overlay` genera unas
  pocas con máscara superpuesta para revisar antes de comprometerse.
- `python training/seg_to_homography.py [--dataset ...] [--n N] [--selftest] [--figuras N]`
  — resuelve H desde la segmentación (ver sección 7). `--selftest` verifica el
  DLT con datos sintéticos antes de tocar nada real (correrlo siempre después
  de tocar `dlt_line_rows`/`dlt_point_rows`/normalización). `--figuras N`
  guarda las N peores y N mejores en `scratch_frames/seg_to_h/`: verde = H
  real, magenta = H estimada, amarillo = centros de círculo extraídos,
  cian/naranja = líneas/vallas extraídas — la herramienta que de verdad
  encontró los bugs de esa sección, no solo los números agregados.
- `python training/auto_label_seg.py --calibrar [--n N]` — puerta de
  aceptación de `auto_label_seg.py` (ver sección 7bis) contra datos con
  verdad conocida; imprime la correlación coste↔error real para recalibrar
  `--max-costo` cuando cambie el modelo de segmentación o crezca el dataset
  NHL de referencia.

## 9. Cosas que dan por sentadas y no hay que re-derivar

- `rink.py` tiene `MIRROR`, `RINK_IIHF`, `RINK_NHL` (¡y `RINK_NHL_FITTED` que
  ya NO se debería usar — sus parámetros se ajustaron sobre el mapeo de zona A
  roto, quedó obsoleto; `test.py:39` ya se cambió a `RINK_NHL` a secas — si
  aparece `RINK_NHL_FITTED` en código nuevo, es una señal de que se copió de
  una versión vieja).
- `make_line_masks.CLASSES` (12 clases de líneas, 0=fondo, 1=vallas, 2-3=goles,
  4-5=azules, 6=central, 7=círculo central, 8-11=faceoff) es el esquema de
  índices de LÍNEAS, **completamente distinto** del índice 0-55 de keypoints
  de `rink.py`. No reutilizar números entre los dos esquemas (bug real que
  costó tiempo en esta sesión: confundir `tpl[8]` (keypoint 56) con clase 8
  de `rink_polylines` (faceoff_A_lo) dio centros de círculo completamente
  falsos).
- El dataset `hockeyrink_nhl` tiene ~80% de sus etiquetas generadas por
  reproyección geométrica (reproyectan a 0.0 px de error consigo mismas), no
  clicadas a mano. Son autoconsistentes pero no independientes — cualquier
  validación tipo bootstrap sobre esas etiquetas sale artificialmente segura.
- Los vídeos propios (`nhl3.mp4` … `nhl7.mp4`, `clip.mp4.webm`, `clip2.mp4`)
  están indexados: leer secuencialmente con `cv2.VideoCapture` da exactamente
  el mismo frame que el sufijo numérico del nombre de archivo en
  `datasets/hockeyrink_nhl/` (verificado en `nhl3.mp4`, diff medio 1.4/255,
  jpg compression). `nhl3.mp4` no tiene cortes de plano entre los frames 0 y
  9523 (159 s a 60 fps) — es un plano largo y estable, útil si se retoma la
  vía temporal más adelante con una técnica más robusta que KLT plano.
- GPU de 8 GB (RTX 3060 Ti): un training de pose a imgsz 1024 ocupa ~7.8 GB.
  No lanzar nada que cargue un segundo modelo (`rink_metric.py`,
  `auto_label.py`) mientras haya un training corriendo — revienta o compite
  por VRAM. Comprobar con `nvidia-smi` antes.
