# Lecciones aprendidas - lo más llamativo de la investigación, para futuros proyectos

Extracto de `docs/experiments/{hockey,soccer,tennis,basketball}.md`, los ADR 0001-0006 y `hockey_research.md`. Cada punto
lleva el número que lo respalda y la sección donde está el detalle (`h` = hockey.md, `s` = soccer.md, `t` = tennis.md,
`b` = basketball.md, `ADR n`). Los errores en px están normalizados a un frame de 1920 px de ancho, p50 = mediana.
La sección 1 es la más reutilizable en cualquier proyecto de visión, no solo de deportes.

## 1. Las diez lecciones que más pesan (si solo lees una sección, esta)

1. **La métrica de entrenamiento no es la métrica del problema.** Existió un modelo de pose con mAP 0.50 y 0% de
   homografías utilizables (h 2). Ultralytics elige el checkpoint por mAP de caja y eligió mal: `best.pt` 85.2 px de error
   frente a 63.4 de `last.pt`. Solución: medir el error de reproyección de la H final (`rink_metric`), no el mAP.
2. **La calidad de las etiquetas es el techo del modelo.** Auditoría con 40 frames clicados a mano: las etiquetas
   automáticas (segmentación + DLT) están a 15-25 px de la verdad (0% por debajo de 10 px) y las antiguas a 11-16 px;
   las manuales a 2-3 px (h 0b). Con ese ruido en `dev`, elegir checkpoint "por unos px" es elegir ruido.
3. **Salida de la red = evidencia en la imagen (heatmaps); la geometría se resuelve fuera.** Regresar la H o la pose
   directamente: p50 ~340 px y 1% < 25 px, con la loss plana; regresar 56 keypoints (YOLO pose): 84.5 px; heatmaps +
   solver DLT: 9.2 px, mismos frames (h 8, h 14, ADR 0004). Diagnóstico decisivo: la pinhole sí representa bien la H
   (ajustada a la H real: p50 4.65 px), así que el fallo era aprender la regresión, no el modelo de cámara.
4. **Una puerta de plausibilidad física vale más que un umbral de confianza.** `is_plausible_view` (parte del campo
   "detrás" de la cámara + ninguna focal válida) rechaza 11 de 12 respuestas > 50 px, 0 de 60 respuestas buenas y 0 de 86
   etiquetas manuales; no tiene umbral ajustado, es una regla física (h 14d). En cambio: el coste de refinamiento
   correlaciona r = 0.97 en 9 frames y r = 0.35 en 955 (ADR 0002); el residuo interno de PnLCalib no ordena el error
   (Spearman 0.20); un gate por nº de keypoints (>= 7) tiraba 3 respuestas buenas (h 14h).
5. **Casi todas las mejoras vinieron de datos, no de arquitectura.** Cuántos datos hacen falta depende de la variedad de vistas: tenis (una vista)
   necesita ~10 etiquetas, baloncesto (15 pabellones) ~200, y una vez que responde el error es el mismo, 4-5 px (b 3): el coste es reconocer, no la geometría.
   * Vistas "de fondo de pista" (4 de 585 frames de entrenamiento): la red no las respondía. 36 etiquetas nuevas + una
     regla de nombres consistente: 47% -> 88% de frames < 25 px en nhl4 (h 14b, 14g).
   * Fútbol, círculo central: 11% -> 64% de cobertura al añadir al índice los frames descartados por tener pocas rectas
     (+27% de frames, s 21-25). Los mismos keypoints, otros datos.
   * Keypoints derivados de círculos (34 más en hockey): no mejoran la precisión donde la red ya responde (h 14c).
6. **Los números de pocos frames son ruido.** El mismo modelo y los mismos frames dan 10.5% o 2.6% de frames < 25 px
   según se infiera en lote o de uno en uno: diferencias de 0.1 px en los keypoints cambian la H de ~20% de los frames en
   decenas de px (RANSAC caótico, ADR 0003). Con 10-25 curvas por clase un fallo = ±8 pp. Entre checkpoints consecutivos
   el p50 oscila 30-60 px (h 2b).
7. **Valida con vídeos/partidos que nada ha visto y guarda un conjunto que nunca toques (`fresh`).** Las gates se
   eligieron sobre `test` y se confirmaron en 38 frames de 4 partidos nuevos: p50 7.2 px, 97% de cobertura, 1 respuesta
   gruesa (rechazada por la gate) (h 14h). `test_leaky` (mismos vídeos que el fine-tune) da p50 6.4 pero no es
   generalización.
8. **Cuidado con la validación circular.** `valx` solo contenía frames fáciles que el DLT sabía resolver; ~80% de las
   etiquetas de `hockeyrink_nhl` eran reproyecciones con error 0.0 contra sí mismas (h 0, h 12). `skipped.json` (frames
   "saltados") parecía un set de negativos gratis: 7 de 8 eran tomas anchas normales (s 17). Comprueba qué significa
   realmente un fichero antes de usarlo como verdad.
9. **El error que importa a veces no es de píxeles sino de identidad.** Claude clicando keypoints: 5.8 px de mediana si se
   quitan 2 puntos con identidad mal puesta; con ellos 9.3 y 16.7 px (h 14e). Y la red aprendió la vista de fondo pero no
   *qué* fondo era cuál, porque las etiquetas no lo decían de forma consistente (260-415 px "mal nombrada") hasta que se fijó una
   regla de nombres (h 14f-g).
10. **Un experimento sin write-up no ocurrió.** La disciplina de este repo (resultado + decisión + comando de reproducción
    + callejones sin salida con su cifra) es lo que ha evitado repetir ideas que ya midieron mal.

## 2. Evaluación y datos: trampas que costaron tiempo

| Trampa | Qué pasó | Ref. |
|---|---|---|
| Etiquetas reproyectadas = autoconsistentes | validar contra ellas da confianza artificial; no son independientes | h 0, 12 |
| Mitad de los frames auto-etiquetados en espejo | 163 de 329: el DLT elige el signo de y al azar (las clases lo/hi son simétricas); el índice los reespeja | h 0b |
| Fuente entera mala que el filtro no ve | nhl6 (pista amateur, plantilla equivocada) y nhl9 (estadio exterior): 0/4 buenos y el filtro de coste no los marcó -> QA visual por vídeo antes de fusionar | h 7, ADR 0002 |
| `split_val` recalcula la partición y mueve ficheros | cambió el set de validación sin avisar | h 12 |
| Script sin `if __name__ == "__main__"` | volvió a repartir un dataset al importarlo | arq. |
| Carpeta "basketball" con datos de fútbol | era SoccerNet; el dataset de baloncesto estaba en otra carpeta | sesión 2026-10-04 |
| Roboflow baloncesto: 1460 imágenes, solo 850 distintas | cada una duplicada con cambio de brillo; sus splits valid/test comparten 15 de 17-18 partidos con train: fuga | b 2 |
| Set de tenis etiquetado por un detector clásico | 1.7 px es acuerdo con ese detector, no con clics humanos; el residuo (< 1.6 px) no detecta frames malos | t 1-2 |
| Reutilizar una carpeta de run | el primer evaluado sobrescribió `best_h.pt` de E0; ahora `--init` dentro de la carpeta de salida se rechaza | s 19 |
| `--tag -cont` | argparse lo lee como opción; hay que usar `--tag=-cont` | s 24 |
| Splits por vídeo/arena | el único modo honesto: frames consecutivos son casi duplicados | h 0b, b 2 |

Medida de los clics: `click_sensitivity` (px que se movería el dibujo si cada clic se mueve ~2 px) calibrada con 352
configuraciones: error real ~0.42 x sensibilidad. Una primera versión promediaba sobre toda la imagen (gradas, horizonte) y avisaba
en 91-100% de los ajustes (h 11).

## 3. Geometría y solver: errores reales encontrados

* **DLT sin normalización de Hartley**: filas de magnitud ~1000 frente a ~10-60 -> SVD inútil, p50 1858 px (h 6).
* **Mapa de espejos mal hecho**: 111 px de error frente a 5.6 con el correcto; plantilla con las dots neutras invertidas
  1.5 m. Con ambos arreglados el techo de la tubería con keypoints perfectos es **1.0 px**: todo lo que falla después es
  modelo o método (h 1). Hacer siempre esta prueba "con verdad terreno, ¿cuál es el techo?".
* **El gemelo de 180 grados** (fútbol): H y H·Rot180 dibujan las mismas líneas con la misma puntuación; con el gemelo ganador
  los puntos caían en su antípoda (2377 px con rectas idénticas). Arreglo: canonicalizar -> 25% a 83% de aciertos (s 1).
  Regla general para campos simétricos: `core.camera.canonical_mirror` nombra la *vista*, no el extremo físico.
* **Rectas paralelas no fijan la dirección transversal**: todas las líneas de gol/azules/central son X = const; se arregló
  añadiendo las vallas (Y = const). El número de condición del SVD "se veía limpio" (h 6).
* **Elección del signo de las vallas por residuo algebraico**: 0/27 correctas (anticorrelado). Se resolvió probando los
  dos signos y comparando el coste tras refinar.
* **`soft_l1` con `f_scale` mal calibrado mata el gradiente**: dos pasadas (lineal y luego robusta).
* **Una H libre de 8 parámetros puede puntuar igual que la verdad deformándose en algo imposible** (error 1500-7800 px con la misma
  puntuación): penalización pinhole (`pinhole_residual`) (s 3).
* **Redundancia, no umbral laxo, compra fiabilidad**: exigir dof >= 8 (mínimo matemático) sube cobertura de 12% a 47%, pero el p50 se
  va a 300-500 px (h 6).
* **Rectas como correspondencias de recta** (DLT dual `l_w ~ H^T l_i`), nunca como puntos muestreados falsos.
* **Cámara de TV = PTZ de centro fijo**: la posición de la cámara sale casi igual en dos frames separados 84 s (0.3 m); fijándola,
  la búsqueda baja de 8 a 3 grados de libertad (pan, tilt, focal) (s 3b). Pero el suelo de 15-34 px en vistas "solo círculo" no
  lo explicaba ni el ruido de clic (1-4 px a centro correcto), ni el detector, ni la distorsión.
* **Distorsión de lente (k1) real y medible, pero no era la causa**: k1 = 0.171 con 10 frames filtrados por cámara; en los frames
  con solo círculo, este queda al 19-26% de la distancia de la esquina, donde la distorsión radial (~r²) apenas pesa (s 15-16).
  Un buen ejemplo de descartar una hipótesis con una prueba directa.
* **Quién ancla a quién**: una elipse da 5 de los 8 números de H; 2 clics más fijan el resto (s 11).

## 4. Qué funcionó y qué no, en un vistazo

| Funcionó | Cifra |
|---|---|
| Heatmaps de keypoints + extremos de recta + DLT puntos+rectas (modelo A) | hockey `fresh`: p50 7.2 px, 97% cobertura (h 14h) |
| Aumento por cámara exacta (`ptz_warp`: giro/zoom -> la etiqueta pasa a `G H`) + espejo | base de todos los modelos |
| Misma receta en fútbol (E1 sobre índice refinado) | `test` 89% cobertura / p50 6.5; `fresh` 100% / 7.7 (PnLCalib 10.3) |
| Tenis: ~10 etiquetas | p50 2.0 px, igual que con 6612 (t 3) |
| Baloncesto: ~200 etiquetas | 100% de cobertura, p50 5.2 px; 550 solo mejora la precisión (< 10 px 83% -> 90%) (b 3) |
| Suavizado temporal (filtro complementario con movimiento KLT) | jitter p50 29.6 -> 3.1 px, misma precisión (h 14i) |
| Movimiento de cámara desde las gradas, pasos <= 5 frames | ~5 px de deriva en 4 s (s 9) |
| Refit de la H con los puntos (en vez de DLT de intersecciones) | +27% de frames de entrenamiento (s 23) |

| No funcionó (no reintentar sin evidencia nueva) | Cifra |
|---|---|
| Cabeza de pose de cámara aprendida desde segmentación | p50 ~340 px, 1% < 25 px, train loss plana (h 8) |
| Refinar la H contra la segmentación | peor que el vecino sin refinar (53 -> 234 px); objetivo con máximos espurios (h 8) |
| UDA (MIC) + aumentos de apariencia | DLT p50 56 -> 338 px; no adoptar (h 9) |
| Más datos auto-etiquetados para el DLT clásico | cobertura 7% -> 5%, p50 37.8 -> 56 px (h 9) |
| Síntesis (domain randomization) para segmentación | mixto: círculo central 35-40 -> 16 px pero "todo bien" 46% -> 39% (h 5) |
| KLT sobre el hielo | 24% catastrófico: se pega a los reflejos especulares; ~90% de puntos perdidos en 1 s (h 3) |
| Kickplate como segmentador de región | catastrófico (p10 IoU 0.007): z = 80 *a lo largo de una curva conocida*, inútil sin la forma (h 3) |
| Ajuste conjunto de homografía con curvas | no converge ni desde la verdad; empeora un ancla de YOLO (33 -> 283 px) (h 3) |
| Cuadrilátero del contorno como H inicial | los "vértices" detectados son recortes por el borde del frame; el dato no existe (h 3) |
| NHL -> IIHF con canales con nombre | 25-43% de cobertura: falla el reconocimiento (614 filas, otra emisión), no la geometría (h 16) |
| Refinar la H de la cadena KLT línea por línea | 5 de 10 pares peor (hasta 31 px) frente a las etiquetas manuales; mejoraba solo contra el propio solver (s 11-12) |
| Distorsión k1 por frame | varía 0.00-0.48 sin relación con el zoom; k1 y pose se compensan (s 15) |
| Gate por línea en SoccerNet | no separa buenos de malos (la peor línea del DLT de confianza: p50 6.6 / p99 677 px) (s 23) |

Detalles que se aprenden una vez:
* Máscara de líneas por color local: usa la **desviación respecto al hielo/césped vecino**, no el color absoluto (h 3).
* Segmentación de la región de juego: HSV fijo y GMM Lab, **elegido por solidez** (convexidad), reduce a la mitad los fallos de cada uno.
* El césped está segado en bandas que **se invierten** al mirar desde el otro lado; la máscara nunca debe depender de ellas (s 10).
* Umbral RANSAC del movimiento: con 3 px los logos estáticos y el marcador arrastran el movimiento hacia cero; con 1 px caen
  fuera (p50 7.8 -> 6.6). El límite de salto: 40 px provocaba 40% de retenciones y 35 reinicios; 300 px es lo razonable (h 14i).
* Los cortes de plano son reales y hay que reiniciar: la diferencia media entre frames sube de ~7 a ~48 (s 9).

## 4b. Cifras de referencia (estado a 2026-10-04)

| Deporte | Modelo | Conjunto | Cobertura | p50 | Nota |
|---|---|---|---|---|---|
| Hockey NHL | A2 + gate | `fresh` (38 frames, 4 partidos nuevos) | 97% | 7.2 px | YOLO ~100 px; exteriores (nhl9) fuera de alcance |
| Fútbol FIFA | E1 refinado + gate | SoccerNet `test` (2135) / `fresh` (19) | 89% / 100% | 6.5 / 7.7 px | las vistas de círculo siguen siendo lo más débil |
| Tenis ITF | modelo por deporte | `test` (1014) | 100% | 1.7 px | etiquetas de un detector, sin `fresh` |
| Baloncesto FIBA | modelo por deporte, 550 etiquetas | DeepSportRadar `test` (84 frames, 3 pabellones no vistos) | 100% | 4.6 px (90% < 10 px) | con 200 etiquetas ya 100% / 5.2 px; con 50, 75% de cobertura; con 10 no funciona (b 3) |

## 5. Ingeniería de entrenamiento y entorno (Windows, GPU de 8 GB)

* **`kpt_oks_sigmas`**: 0.10 satura la loss OKS (89 px de error -> loss 0.038, sin gradiente); 0.02 sí da gradiente. Entrenando
  desde COCO hay que empezar en 0.10 y bajar. `optimizer="auto"` ignora `lr0`; `box=15` para que la cabeza de caja no se
  muera; `patience=250` (la mAP de caja alcanza su pico en la época 1). Nueve runs estancados por esta cadena (h 2).
* **Pérdida focal (CenterNet)**, bias final a p = 0.01, gaussianas de sigma 1.5 px a 480 (6 px a 1920); heatmaps a media
  resolución (a resolución completa se agotó la memoria). Perfectos, los targets decodifican a 0.4-0.5 px; la prueba "targets
  perfectos -> decodificar -> H" es la mejor red de seguridad (h 14).
* **U-Net con encoder ResNet34** en vez de DeepLabV3 sin "+": sin decoder da 1/8 de resolución y emborrona líneas de 3-5 px.
  Pérdida CE con pesos = inverso de la **raíz** de la frecuencia (desbalance 1:2173; el inverso puro da pesos de ~2000x) + Dice.
* **El IoU de validación engaña**: la loss de train sigue bajando y el IoU se queda en 0.25. La prueba que importa es deslizar cada
  curva de la plantilla sobre el mapa de probabilidad y medir el z-score del pico (h 5).
* **8 GB de GPU**: un entrenamiento de pose (~7.8 GB a imgsz 1024) no admite nada más. En Windows los workers del DataLoader reimportan
  el módulo de entrenamiento en cada evaluación -> no editar `train_kpline.py` ni sus imports con un entrenamiento en marcha (usar
  worktree).
* **El PC se reinició bajo carga CPU + GPU** (Kernel-Power 41 sin bugcheck, dos veces el 2026-09-27) y alimentó **lotes corruptos**
  (entradas de hasta 2.85e21, targets de -1.3e17) que dejaron **todas** las estadísticas de BatchNorm en NaN, con los pesos finitos.
  Defensa barata y útil: `train_kpline.corrupted` rechaza todo lote fuera de rangos físicos antes del forward. Regla: un trabajo pesado
  a la vez; un run que muere continúa desde `last.pt` en otra carpeta (s 24).
* **opencv**: `opencv-python` junto a `opencv-contrib-python` sobrescribe `cv2.pyd` y se pierde `cv2.ximgproc` (EdgeDrawing, findEllipses).
  Comprobar la API real de la versión instalada antes de apoyarse en ella (s 7).
* **Procesos largos**: `DETACHED_PROCESS` en Windows deja una consola visible propia cuyo cierre mata el trabajo; `os.kill(pid, 0)` mata el
  proceso en vez de sondearlo (usar `psutil`) (ADR 0006).

## 6. Datos públicos que conviene recordar

| Deporte | Fuente | Para qué sirvió / aviso |
|---|---|---|
| Hockey | SimulaMet HockeyRink (SHL, IIHF 60x30, 661 frames), HockeyAI | modelo inicial; la plantilla IIHF no vale en NHL (61x26): sesgo sistemático |
| Fútbol | SoccerNet calibration-2023 | ~15.5k frames indexados; PnLCalib es GPL-2.0 (no usar en producto) |
| Tenis | TennisCourtDetector (8841 frames, 498 vídeos) | vistas homogéneas -> con ~10 etiquetas basta |
| Baloncesto | DeepSportRadar (728 frames, K R T exactos, CC BY-NC-ND) / Roboflow NBA (33 keypoints) | el primero es mejor para evaluar; licencia no comercial |
| Descartados | rugby (HF, 445 frames sin verificar), fútbol americano (las líneas se repiten, hace falta leer los números), atletismo (sintético) | ver b 1 |

Referencias de la literatura que guiaron el diseño (`hockey_research.md`): PnLCalib / No Bells Just Whistles (heatmaps de
puntos y extremos + DLT), TVCalib (calibración diferenciable), Rink-Agnostic de Waterloo (segmentación + H sintética + UDA), Sportlight
(57 keypoints derivados de rectas y elipses), Sharma 2018 (diccionario sintético), Shi 2022 (autosupervisado).
Resultado práctico: la receta de PnLCalib (keypoints + rectas + DLT) fue la que funcionó; segmentación + DLT clásico y regresión de
pose, no.

## 7. Ideas todavía abiertas

1. Por qué el DLT de segmentación cubre ~0-4% de frames NHL aleatorios (hipótesis sin probar: puntos derivados de intersecciones y tangentes).
2. Una comprobación que atrape homografías "de cámara real pero equivocada" (acuerdo con la H arrastrada por el movimiento de los vecinos).
3. Modelo condicionado por la plantilla (ADR 0005): construido y entrenado con hockey + fútbol + tenis. Zero-shot en baloncesto: 0% de
   cobertura, no enciende ningún punto. Con 50 etiquetas da 100% de cobertura y 5.3 px, lo que el modelo por deporte da con 200 (con 50 se
   queda en 75%); con 10 sigue fallando. Falta el control (cabeza de canales fijos sobre el mismo backbone) que diga si la ganancia es la
   plantilla o el preentrenamiento con otros deportes (b 4).
4. Seguimiento de jugadores en metros (h 15): ¿los trackers de stock cambian la identidad? Diseño y reglas de decisión escritos, **no ejecutado**.
5. Fútbol: re-calibrar el centro fijo por zoom; `soccer2` (otra emisión) no lo sostuvo (2 de 60 frames válidos).
6. Frames con 4+ rectas que el índice sigue descartando (3393 con línea > 3 px).
