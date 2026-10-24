"""CLI: python -m autoedit --input video.mp4 --out final.mp4"""
import argparse
import json
import os
import shutil
import subprocess
import sys

from . import audio, render, timeline, transcribe

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, check=True).stdout
    d = json.loads(out)
    st = d["streams"][0]
    dur = float(st.get("duration") or d["format"]["duration"])
    return st["width"], st["height"], dur


def deep_update(base, extra):
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description="Edicion automatica de video vertical")
    ap.add_argument("--input", "-i", required=True, help="video en bruto")
    ap.add_argument("--out", "-o", default="salida_editada.mp4")
    ap.add_argument("--config", "-c", default=os.path.join(HERE, "config.example.json"))
    ap.add_argument("--srt", help="subtitulos ya existentes (opcional)")
    ap.add_argument("--work", default=os.path.join(HERE, ".work"))
    ap.add_argument("--model", default="small", help="modelo de transcripcion local")
    ap.add_argument("--lang", default="es")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--preview", type=int, help="renderiza solo N frames (prueba rapida)")
    a = ap.parse_args(argv)

    if not shutil.which("ffmpeg"):
        sys.exit("Necesitas ffmpeg en el PATH")
    os.makedirs(a.work, exist_ok=True)
    cfg = json.load(open(a.config, encoding="utf8"))
    cfg["_font_dir"] = os.path.join(HERE, ".cache", "fonts")
    if a.no_captions:
        cfg["captions"]["enabled"] = False

    W, H, dur = probe(a.input)
    fps = cfg["export"]["fps"]
    size = (cfg["export"]["width"], cfg["export"]["height"])
    print(f"[1/5] entrada {W}x{H} {dur:.1f}s -> salida {size[0]}x{size[1]} @{fps}fps")

    tr = transcribe.load_or_make(a.input, a.work, srt=a.srt, model=a.model, lang=a.lang)
    print(f"[2/5] transcripcion: {len(tr)} frases")

    pcm = audio.load_audio(a.input)
    db, hop = audio.envelope(pcm)
    sil, thr = audio.detect_silences(db, hop, min_sil=cfg["cut"]["min_silence"])
    ranges = timeline.keep_ranges(dur, sil, timeline.protected_spans(tr), fps=fps,
                                  pad=cfg["cut"]["pad"], min_silence=cfg["cut"]["min_silence"])
    if cfg["cut"]["split_on_sentences"]:
        ranges = timeline.split_on_sentences(ranges, tr, cfg["cut"]["min_shot"], fps)
    kept = sum(e - s for s, e in ranges)
    print(f"[3/5] cortes: umbral {thr:.1f} dB | {len(sil)} silencios | "
          f"{len(ranges)} planos | {dur - kept:.2f}s eliminados")

    beats = audio.emphasis_peaks(db, hop) if cfg["camera"].get("beat_punch", True) else []
    plan = timeline.build(tr, ranges, cfg, fps=fps, seed=cfg.get("seed", 7), beats=beats)
    wav = os.path.join(a.work, "mix.wav")
    audio.write_wav(wav, audio.build_track(pcm, ranges, plan["intro"], plan["outro"], plan["sfx"]))
    print(f"[4/5] audio montado: {plan['total']:.2f}s, {len(plan['sfx'])} efectos, "
          f"{len(plan['broll'])} escenas B-roll, {len(plan['punches'])} acentos")

    out = render.render(a.input, plan, cfg, a.out, wav, a.work, size=size, fps=fps,
                        preview_frames=a.preview)
    print(f"[5/5] listo -> {out}")
    return out


if __name__ == "__main__":
    main()
