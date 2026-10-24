"""Descarga de tipografias (Google Fonts) y sintesis de efectos de sonido.

No se usa ningun servicio de IA: las fuentes se bajan de fonts.googleapis.com
(licencia OFL) y los SFX se sintetizan con numpy, asi que el pipeline funciona
sin librerias de audio externas ni packs de sonido de pago.
"""
import os
import re
import urllib.request

import numpy as np

UA = {"User-Agent": "Mozilla/5.0 (autoedit)"}
CSS = "https://fonts.googleapis.com/css2?family={fam}:wght@{w}&display=swap"
SYSTEM_FALLBACK = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_font(family, weight, out_dir):
    """Devuelve la ruta local de una Google Font; cae a fuente del sistema."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{family.replace(' ', '')}-{weight}.ttf")
    if os.path.exists(path) and os.path.getsize(path) > 20000:
        return path
    try:
        css = _get(CSS.format(fam=family.replace(" ", "+"), w=weight)).decode("utf8")
        urls = re.findall(r"url\((https://fonts\.gstatic\.com[^)]+)\)", css)
        if urls:
            data = _get(urls[-1])
            if len(data) > 20000:
                with open(path, "wb") as f:
                    f.write(data)
                return path
    except Exception as exc:  # sin red -> fuente del sistema
        print(f"  [assets] no se pudo bajar {family} ({exc}); uso fuente del sistema")
    for cand in SYSTEM_FALLBACK:
        if os.path.exists(cand):
            return cand
    raise RuntimeError("No hay ninguna tipografia disponible")


# ------------------------------------------------------------------ SFX ----
def _env(n, attack, release, sr):
    e = np.ones(n, dtype=np.float32)
    a = max(1, int(attack * sr))
    r = max(1, int(release * sr))
    e[:a] = np.linspace(0, 1, a)
    e[-r:] *= np.linspace(1, 0, r)
    return e


def whoosh(sr=48000, dur=0.42, seed=0, direction=1):
    """Ruido filtrado con barrido de frecuencia: transicion / whip pan."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    noise = rng.standard_normal(n).astype(np.float32)
    # barrido de paso-bajo simulado con media movil de ventana variable
    out = np.zeros(n, dtype=np.float32)
    acc = 0.0
    sweep = np.linspace(0.02, 0.45, n) if direction > 0 else np.linspace(0.45, 0.02, n)
    for i in range(n):  # filtro de un polo con cutoff animado
        acc += sweep[i] * (noise[i] - acc)
        out[i] = acc
    out *= _env(n, 0.05 * dur, 0.6 * dur, sr)
    out /= (np.abs(out).max() or 1)
    return out * 0.5


def riser(sr=48000, dur=1.1, seed=1):
    """Riser tonal + ruido para la aparicion del sujeto."""
    n = int(dur * sr)
    t = np.linspace(0, dur, n, dtype=np.float32)
    f = 120 * (2 ** (3.2 * t / dur))
    tone = np.sin(2 * np.pi * np.cumsum(f) / sr).astype(np.float32)
    rng = np.random.default_rng(seed)
    air = np.cumsum(rng.standard_normal(n).astype(np.float32)) * 0.002
    sig = 0.55 * tone + 0.45 * air
    sig *= np.linspace(0.05, 1.0, n) ** 2 * _env(n, 0.02, 0.08, sr)
    sig /= (np.abs(sig).max() or 1)
    return sig * 0.38


def impact(sr=48000, dur=0.5):
    """Golpe grave para el flash / aparicion."""
    n = int(dur * sr)
    t = np.linspace(0, dur, n, dtype=np.float32)
    f = 140 * np.exp(-7 * t) + 42
    sig = np.sin(2 * np.pi * np.cumsum(f) / sr).astype(np.float32)
    sig *= np.exp(-5.5 * t)
    sig /= (np.abs(sig).max() or 1)
    return sig * 0.55


def pop(sr=48000, dur=0.09, freq=950):
    """Click corto para la entrada de un texto/chip."""
    n = int(dur * sr)
    t = np.linspace(0, dur, n, dtype=np.float32)
    sig = np.sin(2 * np.pi * freq * t) * np.exp(-28 * t)
    sig += 0.3 * np.sin(2 * np.pi * freq * 2.5 * t) * np.exp(-40 * t)
    sig /= (np.abs(sig).max() or 1)
    return sig.astype(np.float32) * 0.22
