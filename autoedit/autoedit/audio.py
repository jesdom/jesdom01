"""Analisis y construccion de la pista de audio (numpy + ffmpeg)."""
import subprocess

import numpy as np

from . import assets

SR = 48000


def load_audio(path, sr=SR):
    """Audio mono float32 del archivo de entrada."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def envelope(a, sr=SR, hop=0.01):
    """RMS en dB por ventana de 'hop' segundos."""
    h = int(sr * hop)
    n = len(a) // h
    frames = a[:n * h].reshape(n, h)
    rms = np.sqrt((frames.astype(np.float32) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12), hop


def detect_silences(db, hop, floor_offset=-16.0, min_sil=0.45, abs_floor=-45.0):
    """Tramos de silencio como lista de (inicio, fin) en segundos.

    El umbral es relativo al nivel de voz del propio video (percentil 85),
    por eso funciona igual con un movil, un microfono de solapa o un Zoom.
    """
    speech_level = np.percentile(db, 85)
    thr = max(speech_level + floor_offset, abs_floor)
    quiet = db < thr
    sil, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * hop >= min_sil:
                sil.append((start * hop, i * hop))
            start = None
    if start is not None and (len(quiet) - start) * hop >= min_sil:
        sil.append((start * hop, len(quiet) * hop))
    return sil, float(thr)


def emphasis_peaks(db, hop, min_gap=0.8):
    """Instantes con subida fuerte de energia (palabras acentuadas)."""
    d = np.diff(db, prepend=db[0])
    thr = np.percentile(d, 97)
    peaks, last = [], -99
    for i, v in enumerate(d):
        t = i * hop
        if v >= thr and t - last >= min_gap and db[i] > np.percentile(db, 60):
            peaks.append(round(t, 3))
            last = t
    return peaks


# ------------------------------------------------------- pista de salida ----
def _xfade(a, b, n):
    """Union equipotencia entre dos bloques (evita clicks en los cortes)."""
    n = min(n, len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])
    w = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    tail = a[-n:] * np.cos(w) + b[:n] * np.sin(w)
    return np.concatenate([a[:-n], tail, b[n:]])


def build_track(src, ranges, intro_s, outro_s, sfx_plan, sr=SR, xfade_ms=18):
    """Concatena los tramos conservados + silencio de intro/outro + SFX."""
    out = np.zeros(int(intro_s * sr), dtype=np.float32)
    body = None
    for (s, e) in ranges:
        chunk = src[int(s * sr):int(e * sr)].copy()
        body = chunk if body is None else _xfade(body, chunk, int(sr * xfade_ms / 1000))
    out = np.concatenate([out, body if body is not None else np.zeros(1, np.float32),
                          np.zeros(int(outro_s * sr), dtype=np.float32)])

    for item in sfx_plan:
        t, kind = item["t"], item["kind"]
        gain = item.get("gain", 1.0)
        if kind == "whoosh":
            s = assets.whoosh(sr, seed=item.get("seed", 0), direction=item.get("dir", 1))
        elif kind == "riser":
            s = assets.riser(sr, dur=item.get("dur", 1.1))
        elif kind == "impact":
            s = assets.impact(sr)
        elif kind == "pop":
            s = assets.pop(sr, freq=item.get("freq", 950))
        else:
            continue
        i = int(t * sr)
        if i < 0:
            s, i = s[-i:], 0
        j = min(len(out), i + len(s))
        if j > i:
            out[i:j] += s[:j - i] * gain
    peak = np.abs(out).max() or 1.0
    if peak > 0.99:
        out = out / peak * 0.99
    return out


def write_wav(path, a, sr=SR):
    import wave
    pcm = np.clip(a, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
