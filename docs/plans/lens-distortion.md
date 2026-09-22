# Plan: modelar la distorsión de lente en la cámara de fútbol

> **Estado (2026-09-22):** pasos 1 y 2 hechos, confiables, con test. Paso 3 (ajuste conjunto pose+k1) implementado
> y con test, pero el k1 que devuelve por frame NO es fiable como calibración — ver la sección "Lo que se encontró
> en los pasos 3-4" al final, con la recomendación de cómo seguir. Pasos 5 y 6 no hechos todavía.

Guía de implementación para ti (el modelo grande te ayuda función a función cuando se complique). No es un ADR ni
un resultado medido todavía — es un plan de trabajo. Cuando algo se mida de verdad, eso va a
`docs/experiments/soccer.md`, no aquí.

**Por qué:** en `docs/experiments/soccer.md` sección 14 medimos que, con el centro de cámara correcto, el ruido de
clic solo produce 1-4 px de error de homografía en cualquier zoom. Pero en los frames reales de solo círculo el
error medido es 15-34 px. Esa diferencia es una señal real, no ruido, y correlaciona con el zoom (f/w) y no con el
tiempo — la firma típica de distorsión de lente sin modelar, no de una cámara que se mueve físicamente.

**Alcance:** un modelo de distorsión radial simple (2 coeficientes), calibrado una vez por zoom a partir de frames
anchos con líneas, aplicado como corrección antes de las funciones que ya existen (`core/circle.py`,
`field_solver.py`). No se toca el DLT de hockey ni nada fuera de `sportcal/lab/soccer` y, como mucho, dos funciones
nuevas en `core/camera.py`.

---

## 0. Qué tienes que entender antes de escribir nada

### 0.1 Repaso del modelo pinhole que ya existe
Lee `sportcal/core/camera.py` entero (es corto). Fíjate en:
- `pose_to_H(p, w, h)`: de una pose `(cx, cy, cz, pan, tilt, f)` construye `H = K [r1 r2 t]`, la homografía que
  manda un punto del plano del mundo (Z=0) a un píxel. Esto es un pinhole **sin distorsión**: cualquier recta del
  mundo sale recta en la imagen.
- `project(H, pts)`: aplica esa H a un lote de puntos.
- `pinhole_residual` / `decompose_H`: el camino inverso, de una H cualquiera a una pose pinhole (si se puede).

Repasa también `refine_ptz` en `sportcal/lab/soccer/field_solver.py`: es un ejemplo de cómo el proyecto ya hace
ajuste no lineal de una pose (aquí con `scipy.optimize.minimize`, método Powell, sobre una cascada de tau
decreciente). Tu función de distorsión seguirá el mismo patrón: una función de residuo + un optimizador.

### 0.2 Qué es la distorsión radial (lo mínimo para poder escribir la fórmula)
Una lente real no es un pinhole perfecto: dobla la luz de forma distinta cerca del eje óptico que en los bordes.
El modelo estándar (Brown-Conrady, el que usa OpenCV) trabaja en coordenadas normalizadas de cámara, **antes** de
multiplicar por la focal:

```
(x, y) = coordenadas normalizadas sin distorsión (las que ya calculas al dividir por la profundidad)
r2 = x*x + y*y
x_d = x * (1 + k1*r2 + k2*r2^2)
y_d = y * (1 + k1*r2 + k2*r2^2)
pixel = K @ [x_d, y_d, 1]
```

Con `k1, k2` los dos coeficientes radiales. Hay también distorsión tangencial (`p1, p2`, por desalineación de
lente/sensor) pero para una lente decente es casi siempre despreciable frente a la radial — empieza sin ella.

**Por qué esto rompe la H:** una H es exactamente el mapa de un pinhole sin distorsión. En cuanto metes `k1, k2`
el mapa mundo→píxel deja de ser lineal/proyectivo: ya no hay una matriz 3x3 que lo represente. Por eso las
funciones nuevas de este plan NO devuelven una H, devuelven píxeles directamente.

### 0.3 Apóyate en OpenCV, no reinventes la proyección
El proyecto ya depende de `opencv-contrib-python` en todos lados. Para esto usa:
- `cv2.Rodrigues` para pasar de tu matriz de rotación `R` a un `rvec` (si haces la proyección con `cv2.projectPoints`
  en vez de a mano, como hicimos aquí).
- `cv2.undistortPoints(pixeles, K, distCoeffs)` → "limpia" puntos clicados/detectados antes de metértelos a las
  funciones que ya existen (`core/circle.py`, que asume pinhole puro). Este es el que de verdad se usó en los
  diagnósticos de abajo.

No necesitas `cv2.fisheye.*` — eso es para lentes de gran angular extremo, no para una lente de zoom de
retransmisión.

---

## 1. `core/camera.py`: proyección con distorsión — HECHO

`project_distorted(pose, k1, k2, world_pts, w, h)` en `sportcal/core/camera.py`. Acepta k1/k2 escalares o un valor
por pose (como `pose_to_H` acepta una pose o un lote). Devuelve `(xy, depth)`, igual que `project()`.

Bug real que salió al escribirla la primera vez, por si te vuelve a pasar: se le añadió una columna de unos a los
puntos del mundo (para homogeneizar, como hace `project`) pero luego solo se multiplicaba por 2 columnas de `R`
(r1, r2) sumando `t` aparte — las formas no casaban. La homogeneización solo tiene sentido si multiplicas por las
3 columnas de `R` juntas (`[r1, r2, t]`, como hace `pose_to_H`); si sumas `t` aparte, el punto debe ser solo
`[X, Y]`, sin el 1.

Tests en `tests/test_camera.py`: k1=k2=0 reproduce exactamente `project(pose_to_H(...))`; un k1 real curva puntos
del mundo que estaban alineados, creciendo con `|k1|`; acepta escalar o un valor por pose. Mutación comprobada.

## 2. Diagnóstico: ¿de verdad hay curvatura medible? — HECHO, resultado positivo

En vez de asumir que la distorsión es real solo porque correlaciona con el zoom (sección 14 de
`docs/experiments/soccer.md`), se midió directamente: sobre la máscara de líneas de un frame ancho, se detecta la
línea más larga (`core/fitting.py::detect_lines` + inliers), se toman sus puntos binados a lo largo del eje largo
(igual que `skeleton_points`, para no dejar que el ancho de la pintura cerca de la cámara domine), y se mide el
**"bow"**: la desviación perpendicular máxima respecto a la cuerda que une sus dos extremos. Una recta del mundo
proyectada sin distorsión da bow≈0 salvo ruido de detección; con distorsión real, esa desviación tiene forma de
arco y cambia con `k1` al "desdistorsionar" los puntos con `cv2.undistortPoints`.

**Resultado**, en píxeles a la resolución nativa del vídeo: f2200 (línea de 924 px, f/w≈1,44) tiene un mínimo nítido
y acotado por los dos lados en k1≈0,25 (bow 5,25→1,24 px). f2250 (906 px, f/w≈1,43) en k1≈0,175 (3,73→0,92 px).
f1000 (645 px, f/w≈1,84) en k1≈0,175 (2,03→1,38 px). f900 y f950 no muestran ningún mínimo claro en el mismo rango
de k1 — su línea más larga disponible es más corta (578 y 499 px), probablemente más cerca del centro de la
imagen, con menos palanca para delatar distorsión (coherente: el efecto radial crece con el cuadrado de la
distancia al centro óptico, así que una línea corta y central apenas se mueve por mucho k1 que haya).

**Conclusión: hay distorsión real y medible, del orden de k1≈0,17-0,25, pero solo la delatan las líneas que
llegan lejos del centro de la imagen — no cualquier línea del frame.** Esto importa mucho para el paso 3.

## 3. Ajuste conjunto de pose + distorsión en un frame — HECHO, pero ver la sección de abajo antes de fiarte del k1

`FieldSolver.score_distorted(pose, k1, k2, fine, tau, radius_weight)` y `FieldSolver.refine_distorted(pose0, k1_0,
k2_0, fit_k2)` en `sportcal/lab/soccer/field_solver.py`, junto a `refine_ptz`. Por defecto `fit_k2=False`: solo se
ajusta k1 (7 parámetros libres, no 8) porque k1 y k2 son fuertemente colineales sobre el rango de radio que cubre
un solo plano — el término de cuarto grado casi no se separa del de segundo grado salvo muy cerca del borde de la
imagen, y ajustar los dos a la vez desde un solo frame converge en un par que se compensa mutuamente en vez del k1
real (medido en escenas sintéticas antes de poner esta guarda).

Tests en `tests/test_soccer_solver.py` (síntéticos, con verdad conocida): el ajuste mejora sustancialmente el
encaje en espacio de píxeles frente al punto de partida sin distorsión, y no inventa distorsión en una máscara
limpia sin ella. **Ojo:** el primer test comprobaba que `k1` se recuperase con tolerancia ±0,03 del valor real y
fallaba de forma intermitente según la semilla — se cambió el criterio a error de reproyección (más robusto y más
honesto sobre lo que de verdad se puede pedir de un solo frame) tras confirmar en varias semillas que el score sí
mejora consistentemente pero el k1 exacto recuperado varía. Esto ya anticipaba el problema de la sección siguiente.

## 4. ¿Es una distorsión fija, o cambia con el zoom? — hecho parcialmente, con un hallazgo importante

Repetir el ajuste conjunto del paso 3 en varios frames reales dio k1 muy inconsistentes entre sí (de 0,00 a 0,48,
sin relación limpia con el zoom), contradiciendo el paso 2. La causa, encontrada al comparar ambos métodos sobre
el mismo frame (f2200): `score_distorted` pesa cada segmento del contorno de la plantilla por su LONGITUD en
píxeles, igual que el resto de puntuaciones del proyecto — y las líneas cortas cerca del centro de la imagen (con
poca palanca para delatar distorsión, según el paso 2) son mucho más numerosas que las pocas líneas largas
informativas, así que dominan la suma y diluyen la señal. Se añadió `radius_weight=True` a `score_distorted`
(multiplica cada segmento por el cuadrado de su distancia al centro de la imagen, normalizado), usado solo dentro
de `refine_distorted`, sin tocar la puntuación de nadie más. Mejoró parcialmente pero NO resolvió el problema de
fondo — ver la sección siguiente.

---

## Lo que se encontró en los pasos 3-4, y cómo seguir

Incluso con la puntuación ponderada por radio, el ajuste conjunto de pose+k1 sobre f2200 converge a k1≈0,00 (barrido
con la pose fija, puntuación ponderada), **contradiciendo el mínimo nítido en k1≈0,25 que el mismo frame dio en el
paso 2.** La diferencia de fondo entre los dos métodos:

- El **paso 2** (bow de una línea detectada) no asume nada sobre la pose de la cámara ni sobre a qué línea del
  mundo corresponde cada punto: solo pregunta "¿estos píxeles, que ya sé que pertenecen a una misma línea recta,
  se enderezan al quitarles k1?". Es una medida de curvatura pura.
- El **ajuste conjunto** (pasos 3-4) proyecta la PLANTILLA DEL MUNDO completa con una pose candidata y compara
  contra la máscara observada. Si la pose de partida (`search_lines`, sin distorsión) tiene aunque sea un pequeño
  error de posición u orientación — y la sección 14 ya midió que eso pasa —, el optimizador puede "explicar" ese
  error moviendo k1 en la dirección que sea, o directamente no moverlo si mover la pose ya basta para tapar el
  hueco. La distorsión y el error de pose quedan mezclados y el ajuste conjunto no los separa de forma fiable
  frame a frame.

**Recomendación para seguir (no implementada todavía):** no fiarse de un k1 por frame salido del ajuste conjunto.
En vez de eso, una calibración tipo "plumb-line" de verdad:
1. Sobre VARIOS frames anchos a la vez, extraer (como en el paso 2) las líneas MÁS LARGAS detectadas — sin pose,
   sin plantilla — y medir su "bow" en función de k1, igual que el paso 2 ya hace por frame.
2. Ajustar un k1 COMPARTIDO por todos esos frames (o por bucket de zoom) minimizando la suma de "bow" al cuadrado
   de todas las líneas largas a la vez, con las k1 buscándose de forma independiente de cualquier pose — esto es
   exactamente lo que ya funcionó en el paso 2, solo que combinando varios frames en vez de mirar cada uno suelto.
3. Solo DESPUÉS de tener ese k1 (por zoom o único), aplicarlo como corrección fija (des-distorsionar los puntos
   clicados/detectados con `cv2.undistortPoints` antes de pasarlos a `core/circle.py`) y recién ahí SÍ merece la
   pena re-ajustar la pose con k1 ya fijo, no libre — eso evita la mezcla que rompió los pasos 3-4.

Esto es más trabajo que lo que quedaba de plan original (pasos 5-6 asumían que el k1 por frame del paso 3 ya era
fiable). Antes de construirlo, vale la pena parar y decidir si merece la pena frente a la vía barata de calibración
por rango de zoom que se descartó al principio de esta sesión — con esta nueva evidencia (k1 real, ~0,17-0,25, pero
solo detectable con líneas largas y lejos del centro) puede que ninguna de las dos por sí sola baste, y haga falta
alguna combinación de las dos.

---

## Recordatorios del estilo del repo mientras hagas esto

- Código y comentarios en inglés; los docstrings nunca citan números de experimento (los números van en
  `docs/experiments/soccer.md`, no en el código).
- `sportcal/core/camera.py` es código agnóstico de deporte: no metas nada específico de fútbol ahí.
- Cualquier script de calibración por frames va en `sportcal/lab/soccer/`, con `main()` detrás de
  `if __name__ == "__main__":`.
- Corre `venv/Scripts/python.exe -m pytest` después de tocar `core/`. La suite completa (`pytest -m ""`) antes de
  dar por cerrado el trabajo.
