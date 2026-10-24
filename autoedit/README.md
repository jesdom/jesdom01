# autoedit

Edición automática de video vertical (9:16) **con código**: corte de silencios, cámara
virtual, transiciones, color, SFX y motion graphics sincronizados con lo que se dice.

En el render no interviene ninguna IA: todo es `numpy` + `Pillow` + `ffmpeg`. El único
paso opcional con modelo es la transcripción local (`faster-whisper`), que solo sirve
para saber **qué** se dice y **en qué segundo**; puedes sustituirla por un `.srt`.

El prompt reutilizable para pedirle esto a un agente está en [`PROMPT.md`](PROMPT.md).

## Uso

```bash
pip install -r requirements.txt
python -m autoedit -i mi_video.mp4 -o final.mp4
```

| Flag | Qué hace |
|---|---|
| `-i, --input` | Video en bruto (vertical u horizontal) |
| `-o, --out` | MP4 de salida (1080×1920, H.264, AAC −14 LUFS) |
| `-c, --config` | JSON de marca y comportamiento (por defecto `config.example.json`) |
| `--srt` | Usa subtítulos existentes en vez de transcribir |
| `--preview N` | Renderiza solo N frames para revisar rápido |
| `--no-captions` | Edita sin subtítulos |
| `--model`, `--lang` | Modelo y idioma de la transcripción local |

## Qué hace, paso a paso

1. **Análisis** — `ffprobe`, transcripción con tiempos por palabra y envolvente RMS del
   audio cada 10 ms. El umbral de silencio es relativo al nivel de voz del propio video.
2. **Zona segura** (`layout.py`) — la región de mayor varianza temporal es la persona.
   De ahí sale la banda inferior donde se puede escribir **sin tapar la cara**; nada se
   dibuja en la franja superior.
3. **Montaje** (`timeline.py`) — quita silencios > 0.5 s sin partir ninguna palabra
   (los rangos de cada palabra están protegidos) y divide en planos por frase cuando el
   discurso es continuo, para cambiar encuadre sin tocar el audio.
4. **Apertura** (`effects.py`) — la escena empieza vacía: el fondo se teje con franjas
   limpias del encuadre y el sujeto se materializa con una disolución de ruido,
   desenfoque que se cierra, separación RGB y un destello. El cierre es la inversa.
5. **Dinamismo** (`render.py`) — push-in / pull-out / drift / flotación handheld,
   transiciones (flash, whip-pan con motion blur, glitch por bandas, zoom punch),
   curva en S con split-tone, bloom, viñeta y grano.
6. **Motion graphics** (`graphics.py`) — subtítulos karaoke palabra por palabra, chips
   con iconos vectoriales disparados por palabra clave, tarjeta de datos con barras
   animadas y barra de progreso.
7. **Audio** (`audio.py`) — uniones con crossfade de 18 ms, SFX sintetizados con numpy
   (whoosh, riser, impacto, pop) y máster con highpass + compresor + loudnorm.

## Configuración

Todo lo editable vive en el JSON: `brand`, `fonts`, `intro`, `outro`, `cut`, `camera`,
`captions` (incluido `fix_words` para corregir nombres mal transcritos), `grade`,
`layout`, `keywords`, `cards`, `ui` y `export`. El código no tiene nada específico del
video de ejemplo.

## Requisitos

`ffmpeg` y `ffprobe` en el PATH, Python 3.9+, `numpy`, `Pillow`. `faster-whisper` solo
si quieres subtítulos automáticos. Las tipografías se descargan de Google Fonts y, si no
hay red, se usa una del sistema.
