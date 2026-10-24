# PROMPT — Edición automática de video vertical con Python (sin IA en el render)

Copia **todo** el bloque de abajo en Claude Code (o en el agente que uses), adjunta tu
video y ejecútalo. Está escrito para que funcione con **cualquier** video de persona
hablando a cámara: el agente analiza el tuyo y adapta cortes, textos y gráficos.

---

## ▶️ EL PROMPT (copiar desde aquí)

```text
Quiero que edites el video que te adjunto y me devuelvas un MP4 vertical listo para
publicar. Hazlo TODO con código Python (numpy + Pillow + ffmpeg). No uses herramientas
de IA generativa para la imagen ni para la edición: cortes, transiciones, efectos y
motion graphics se programan. La única excepción permitida es transcribir el audio con
un modelo local (faster-whisper) para saber qué se dice y en qué segundo; si te doy un
.srt, úsalo y no transcribas nada.

ENTREGABLE
- MP4 1080x1920, 30 fps, H.264 yuv420p, +faststart, audio AAC 192k normalizado a -14 LUFS.
- Un script reutilizable con un archivo de configuración (JSON) separado del código,
  para que yo pueda cambiar colores, textos, palabras clave y duraciones sin tocar
  el código, y para que cualquier persona pueda correrlo con SU propio video.
- Un comando de una línea: python -m autoedit -i mi_video.mp4 -o final.mp4

PASO 1 — ANÁLISIS (antes de editar)
1. ffprobe: resolución, duración, fps, audio.
2. Transcribe con tiempos por palabra (faster-whisper local, idioma es) o lee mi .srt.
3. Envolvente RMS del audio en ventanas de 10 ms para detectar silencios. El umbral
   debe ser RELATIVO al nivel de voz del propio video (percentil 85 menos ~16 dB), no
   un valor fijo, para que funcione igual con móvil, micro de solapa o Zoom.
4. Detecta la ZONA DEL SUJETO sin modelos de IA: muestrea ~48 frames reducidos, calcula
   la varianza temporal por píxel y la diferencia entre frames consecutivos; la región
   con más variación es la persona (cara, boca, manos). Saca su bounding box.

PASO 2 — MONTAJE (cuidado con los cortes)
- Elimina solo silencios de más de ~0.5 s, dejando ~0.14 s de aire a cada lado.
- NUNCA cortes dentro de una palabra: protege el rango de cada palabra de la
  transcripción (±0.08 s) y recorta únicamente la parte del silencio que queda fuera.
- Une los tramos con un crossfade de audio de ~18 ms para que no se oiga el corte.
- Si el discurso es continuo y no hay silencios que quitar, NO fuerces cortes de tiempo:
  divide en planos por frase (mínimo ~2.4 s por plano) manteniendo el audio contiguo, y
  usa esos límites para cambiar encuadre y meter transición. Eso da ritmo sin romper el
  mensaje.
- Alinea todo a la rejilla de frames para que audio y video no se desincronicen.

PASO 3 — LA PERSONA APARECE DE LA NADA (apertura)
- Arranca con la escena VACÍA, sin el sujeto: construye un "plato" de fondo por código,
  tejiendo franjas limpias de los bordes del encuadre (espejadas y mosaicadas) mezcladas
  con una versión muy desenfocada del cuadro, y oscurécelo un poco.
- Durante ~1.5 s el sujeto se materializa: máscara de ruido suave con gradiente vertical
  que avanza, el sujeto entra desenfocado y se va enfocando, con separación RGB que se
  cierra y un destello final de luz acompañado de un golpe grave.
- Cierra el video con la operación inversa: el sujeto se desvanece y queda la escena
  vacía con la tarjeta de cierre.

PASO 4 — DINAMISMO (todo programado, con moderación)
- Cámara virtual por plano: push-in, pull-out, drift lateral y flotación tipo cámara en
  mano (ruido de valor suave, nunca aleatorio puro). Zoom base ~2 % para tener margen.
- Transiciones rotando entre: flash blanco, whip-pan con motion blur horizontal, glitch
  por bandas con separación RGB, y zoom punch. Siempre cortas (≤ 0.34 s) y con micro
  shake que decae.
- Look: curva en S, split-tone cálido en altas y frío en sombras, saturación +8 %,
  bloom suave en luces, viñeta y grano fino. Sutil: tiene que verse cine, no filtro.
- SFX sintetizados con numpy (nada de packs ni IA): whoosh con barrido de filtro en cada
  transición, riser + impacto en la aparición, pop corto en cada entrada de texto.
  Mézclalos por debajo de la voz y pasa el máster por highpass + compresor + loudnorm.

PASO 5 — MOTION GRAPHICS (obligatorios, sincronizados con lo que dice)
- Subtítulos karaoke palabra por palabra: bloques de máximo 3 palabras / 20 caracteres,
  la palabra que se está diciendo en ese instante va resaltada en una píldora de color
  de marca con el resto en blanco con contorno oscuro. Entrada con rebote (spring) y
  salida suave. Dos bloques NUNCA pueden verse a la vez: recorta el final de cada bloque
  al inicio del siguiente.
- Chips con icono vectorial (dibujados con primitivas, no imágenes) que entran con
  rebote cuando se pronuncia una palabra clave de mi lista (ej.: "minutos" → rayo,
  "horas" → reloj, "carpetas" → carpeta, "profesional" → check). Alterna lado
  izquierdo/derecho y limita a uno cada ~4 s para que no se sature.
- Al menos una tarjeta de datos animada que refuerce el argumento cuando se dice la
  palabra que yo indique: título + filas con barras que crecen con easing y cifras que
  aparecen al terminar la barra. Mientras la tarjeta está en pantalla, oculta subtítulos
  y chips para no amontonar.
- Barra de progreso fina abajo (retención) y tarjeta de cierre con CTA.
- Corrige en los subtítulos las palabras que el transcriptor escribe mal (mapa
  configurable, ej. "Cloud" → "Claude").

PASO 6 — REGLA DE ORO DE LA TIPOGRAFÍA (no negociable)
- NINGÚN texto ni gráfico puede quedar sobre la cara de la persona.
- NINGÚN texto en la franja superior del encuadre (ahí no va nada: ni marca, ni hashtag,
  ni título).
- Todo el texto vive en una BANDA SEGURA calculada a partir de la caja del sujeto del
  Paso 1: el borde superior de la banda es el borde inferior del sujeto + ~5 % de alto,
  acotado entre el 60 % y el 76 % de la altura, y el borde inferior deja ~8.5 % de aire
  para la barra de progreso. Dentro de esa banda: la tarjeta/chips arriba y los
  subtítulos abajo, centrados.
- Corrige la banda por el zoom máximo de la cámara virtual (el encuadre se agranda, la
  cara baja): proyecta la caja con z_max antes de decidir las alturas.
- Equilibrio horizontal: ningún bloque de texto puede superar el 86 % del ancho; si se
  pasa, redúcelo proporcionalmente en vez de partirlo mal. Márgenes laterales iguales.
- Si el plano es tan cerrado que no queda banda libre (< 17 % de alto), baja la banda al
  mínimo y añade una cinta de contraste en degradado SOLO en esa franja inferior (cubre
  pecho y fondo, nunca la cara) para que el texto se lea.
- Jerarquía: subtítulos ~84 px, chips ~34 px, tarjeta ~44 px sobre 1080 de ancho, con
  escalado proporcional si la salida cambia de tamaño.

PASO 7 — ROBUSTEZ PARA QUE OTRA PERSONA PUEDA USARLO CON SU VIDEO
- Nada hardcodeado del video de ejemplo: todo (textos de intro/cierre, colores, palabras
  clave, tarjetas, duraciones, umbrales) vive en el JSON de configuración.
- Si no hay red para bajar tipografías (Google Fonts vía API CSS), usa una del sistema.
- Si no se puede transcribir, edita igual: sin subtítulos pero con cortes, cámara,
  transiciones, look y gráficos genéricos.
- Si el video es horizontal, reencuádralo a 9:16 centrando el sujeto detectado.
- Opción --preview N para renderizar solo N frames y revisar rápido antes del render final.
- Imprime un log por etapa: cortes detectados, segundos eliminados, planos, banda segura
  calculada y total de efectos.

CRITERIO DE ACEPTACIÓN (revísalo tú mismo antes de entregarme el archivo)
1. Extrae fotogramas del resultado y comprueba de verdad que ningún texto toca la cara
   ni aparece en la franja superior, y que nada se sale de los márgenes.
2. Ninguna palabra queda cortada por un corte de montaje.
3. Nunca hay dos bloques de subtítulo simultáneos ni gráficos superpuestos.
4. El audio no tiene clics en las uniones y el nivel es constante.
5. Dime con números qué hizo el script: duración original vs final, cuántos cortes,
   cuántos gráficos y dónde quedó la banda de texto.
```

## ▶️ FIN DEL PROMPT

---

## Cómo lo usa otra persona con su propio video

```bash
pip install -r requirements.txt            # numpy, Pillow (+ faster-whisper opcional)
python -m autoedit -i mi_video.mp4 -o final.mp4
python -m autoedit -i mi_video.mp4 -o final.mp4 --srt mis_subtitulos.srt   # sin transcribir
python -m autoedit -i mi_video.mp4 -o prueba.mp4 --preview 150             # prueba rápida
python -m autoedit -i mi_video.mp4 -o final.mp4 -c mi_config.json          # mi marca
```

Lo mínimo que conviene tocar en el JSON antes de publicar:

| Clave | Para qué |
|---|---|
| `brand.accent` / `accent2` | Colores de marca de las píldoras, chips y barras |
| `intro.title` / `outro.title` | Gancho de apertura y llamada a la acción |
| `keywords` | Palabra hablada → chip con icono (`spark`, `bolt`, `clock`, `folder`, `check`) |
| `cards` | Tarjeta de datos y la palabra que la dispara |
| `captions.fix_words` | Arreglar nombres que el transcriptor escribe mal |
| `layout` | Márgenes de la banda segura si tu plano es más abierto o más cerrado |
| `cut.min_silence` | Cuán agresivo es el recorte de silencios |
