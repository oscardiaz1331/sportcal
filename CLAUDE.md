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
- **DECISIÓN PENDIENTE, condicionada a un experimento**: ¿seguir invirtiendo en
  YOLO (keypoints dispersos) o pivotar a un solver aprendido sobre la
  segmentación? Ver sección 7ter.

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

## 7ter. Decisión pendiente: ¿seguir con YOLO o pivotar el solver? (condicionada a un experimento)

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

**El experimento que decide esto** (no lanzado todavía): reentrenar
`train_lines_seg.py --synth` con los datos NHL nuevos (329 frames reales
diversos recién fusionados, sección 7bis) y volver a correr
`seg_to_homography.py --dataset hockeyrink_nhl --n N` (N grande, todo el val)
para medir la cobertura dof≥10 ANTES vs DESPUÉS del reentreno, con el mismo
split de val y mismo N ambas veces para que sea comparable:

```
# ANTES de reentrenar (número de referencia, val NHL): cobertura 7%, p50=37.8px
python training/seg_to_homography.py --dataset hockeyrink_nhl --n <todo val>

# reentrenar, luego repetir exactamente el mismo comando
```

- **Si la cobertura sube apreciablemente** (p.ej. 7%→15-20%+) manteniendo o
  mejorando la precisión: el problema era calidad/diversidad de datos de
  segmentación, no el solver. Seguir por ese camino (más vídeo auto-etiquetado
  → mejor segmentación → más cobertura, en bucle) es la apuesta correcta, y
  NO hace falta la cabeza aprendida todavía.
- **Si se estanca** (cobertura se mueve poco, digamos <10 puntos porcentuales,
  a pesar de +329 frames reales diversos): es señal de que el techo es
  estructural del solver (frames con contenido genuinamente insuficiente para
  dof≥10, sin importar cuánto mejore la red) — ahí es cuando construir la
  cabeza aprendida está justificado, y sería el momento de parar de invertir
  en YOLO (56 keypoints) en favor de esto.

En cualquiera de los dos casos, YOLO como arquitectura de keypoints dispersos
pierde prioridad frente a la vía de segmentación — la pregunta que decide el
experimento es solo si además hace falta sustituir el DLT por un solver
aprendido, o si mejorar la segmentación ya es suficiente.

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
