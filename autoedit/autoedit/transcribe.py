"""Transcripcion opcional con faster-whisper (local, offline tras la 1a vez).

Es el UNICO paso que usa un modelo; es opcional y previo al render:
  - si ya tienes un .srt o un transcript.json, el render no necesita nada de esto.
  - el montaje, los efectos y los motion graphics son 100% codigo.
"""
import json
import os
import re
import subprocess

import numpy as np


def _audio16k(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", "16000", "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def transcribe(video, out_json, model="small", lang="es"):
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(_audio16k(video), language=lang,
                           word_timestamps=True, vad_filter=True)
    data = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
             "words": [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
                       for w in (s.words or [])]} for s in segs]
    with open(out_json, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data


def _ts(x):
    h, rest = divmod(float(x), 3600)
    m, s = divmod(rest, 60)
    return int(h), int(m), s


def from_srt(path):
    """Lee un .srt (sin tiempos por palabra: se reparten proporcionalmente)."""
    txt = open(path, encoding="utf8").read()
    pat = re.compile(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)\s*\n(.+?)(?=\n\s*\n|\Z)", re.S)
    out = []
    for a, b, body in pat.findall(txt):
        def sec(x):
            x = x.replace(",", ".")
            hh, mm, ss = x.split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
        s, e = sec(a), sec(b)
        text = " ".join(l.strip() for l in body.strip().splitlines())
        ws = text.split()
        n = max(1, len(ws))
        words = [{"w": w, "s": round(s + (e - s) * i / n, 2),
                  "e": round(s + (e - s) * (i + 1) / n, 2)} for i, w in enumerate(ws)]
        out.append({"start": s, "end": e, "text": text, "words": words})
    return out


def load_or_make(video, workdir, srt=None, model="small", lang="es"):
    if srt and os.path.exists(srt):
        print(f"  [transcribe] uso subtitulos: {srt}")
        return from_srt(srt)
    cache = os.path.join(workdir, "transcript.json")
    if os.path.exists(cache):
        print(f"  [transcribe] uso cache: {cache}")
        return json.load(open(cache, encoding="utf8"))
    try:
        print("  [transcribe] generando transcripcion local (faster-whisper)...")
        return transcribe(video, cache, model=model, lang=lang)
    except Exception as exc:
        print(f"  [transcribe] sin transcripcion ({exc}); se edita sin subtitulos")
        return []
