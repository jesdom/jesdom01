"""Construccion del plan de edicion: cortes, camara, transiciones y graficos."""
import random
import unicodedata

from . import util


def _norm(w):
    w = unicodedata.normalize("NFD", w.lower())
    return "".join(c for c in w if unicodedata.category(c) != "Mn" and c.isalnum())


def protected_spans(transcript, pad=0.08):
    """Rangos donde NO se puede cortar: cada palabra detectada (+ margen)."""
    spans = []
    for seg in transcript:
        for w in seg.get("words", []):
            spans.append((max(0.0, w["s"] - pad), w["e"] + pad))
    return spans


def keep_ranges(duration, silences, protected, fps=30,
                pad=0.14, min_silence=0.5, min_keep=0.45, head_pad=0.10):
    """Tramos a conservar: quita silencios largos sin tocar ninguna palabra."""
    cuts = []
    for (s, e) in silences:
        seg = [(s + pad, e - pad)]            # deja aire a ambos lados del corte
        for (ps, pe) in protected:            # nunca recorta dentro de una palabra
            nxt = []
            for (a, b) in seg:
                if pe <= a or ps >= b:
                    nxt.append((a, b))
                    continue
                if ps > a:
                    nxt.append((a, min(b, ps)))
                if pe < b:
                    nxt.append((max(a, pe), b))
            seg = [(a, b) for (a, b) in nxt if b > a]
        for (a, b) in seg:
            if b - a >= min_silence:
                cuts.append((a, b))
    cuts.sort()

    ranges, t = [], 0.0
    for (s, e) in cuts:
        if s - t > min_keep:
            ranges.append((t, s))
        t = e
    if duration - t > 0.2:
        ranges.append((t, duration))
    if not ranges:
        ranges = [(0.0, duration)]
    # ajuste a rejilla de frames para que audio y video no se desincronicen
    snapped = []
    for (s, e) in ranges:
        s = round(s * fps) / fps
        e = round(e * fps) / fps
        if e - s >= 2 / fps:
            snapped.append((max(0.0, s - head_pad if s > head_pad else s), e))
    return snapped


CAMERA_MOVES = ["push_in", "pull_out", "drift_left", "drift_right", "static_float"]
TRANSITIONS = ["flash", "whip_left", "whip_right", "glitch", "zoom_punch"]


def build(transcript, ranges, cfg, fps=30, seed=7, beats=None):
    """Devuelve el plan completo del montaje."""
    rng = random.Random(seed)
    intro = cfg["intro"]["duration"]
    outro = cfg["outro"]["duration"]

    clips, out_t = [], intro
    for i, (s, e) in enumerate(ranges):
        n = max(1, int(round((e - s) * fps)))
        move = CAMERA_MOVES[i % len(CAMERA_MOVES)] if cfg["camera"]["enabled"] else "static_float"
        if i == 0:
            move = "push_in"
        clips.append({
            "i": i, "src_start": s, "src_end": e, "n": n,
            "out_start": out_t, "out_end": out_t + n / fps,
            "move": move,
            "zoom_base": 1.0 + cfg["camera"]["base_zoom"] + rng.uniform(0, 0.012),
            "transition": None if i == 0 else TRANSITIONS[(i - 1) % len(TRANSITIONS)],
            "seed": rng.randrange(10 ** 6),
        })
        out_t = clips[-1]["out_end"]
    total = out_t + outro

    # --- mapa tiempo fuente -> tiempo de salida (para subtitulos y graficos) --
    def to_out(t_src):
        for c in clips:
            if c["src_start"] <= t_src <= c["src_end"]:
                return c["out_start"] + (t_src - c["src_start"])
        prev = None
        for c in clips:
            if c["src_start"] > t_src:
                return c["out_start"]
            prev = c
        return prev["out_end"] if prev else intro

    # ------------------------------------------------- subtitulos karaoke ----
    fix = {_norm(k): v for k, v in cfg["captions"].get("fix_words", {}).items()}
    if fix:
        for seg in transcript:
            for w in seg.get("words", []):
                key = _norm(w["w"])
                if key in fix:
                    tail = "".join(c for c in w["w"] if not c.isalnum())
                    w["w"] = fix[key] + (tail if tail in ",.!?;:" else "")

    fix = {_norm(k): v for k, v in cfg["captions"].get("fix_words", {}).items()}
    for seg in transcript or []:
        for w in seg.get("words", []):
            if _norm(w["w"]) in fix:
                punct = w["w"][-1] if w["w"][-1:] in ",.!?;:" else ""
                w["w"] = fix[_norm(w["w"])] + punct

    chunks = []
    if cfg["captions"]["enabled"] and transcript:
        maxw = cfg["captions"]["max_words"]
        maxc = cfg["captions"]["max_chars"]
        for seg in transcript:
            cur = []
            for w in seg.get("words", []):
                cand = cur + [w]
                if len(cand) > maxw or sum(len(x["w"]) + 1 for x in cand) > maxc:
                    chunks.append(cur)
                    cur = [w]
                else:
                    cur = cand
            if cur:
                chunks.append(cur)
    caption_items = []
    for ch in chunks:
        if not ch:
            continue
        words = [{"w": w["w"], "s": to_out(w["s"]), "e": to_out(w["e"])} for w in ch]
        s = words[0]["s"]
        e = max(words[-1]["e"] + 0.12, s + 0.35)
        if e - s < 0.12:
            continue
        caption_items.append({"start": s, "end": e, "words": words})
    for a, b in zip(caption_items, caption_items[1:]):   # sin solapes
        a["end"] = min(a["end"], b["start"] - 0.02)
    caption_items = [c for c in caption_items if c["end"] - c["start"] > 0.12]

    # --------------------------------------------------- chips de palabras ---
    kw_cfg = {_norm(k): v for k, v in cfg["keywords"].items()}
    chips, used = [], {}
    if transcript:
        for seg in transcript:
            for w in seg.get("words", []):
                key = _norm(w["w"])
                if key in kw_cfg and used.get(key, -9) + 4.0 < w["s"]:
                    spec = kw_cfg[key]
                    used[key] = w["s"]
                    chips.append({
                        "label": spec.get("label", w["w"]).upper(),
                        "icon": spec.get("icon", "spark"),
                        "start": max(intro, to_out(w["s"]) - 0.08),
                        "dur": spec.get("dur", 2.0),
                        "slot": len(chips) % 2,
                    })

    # ------------------------------- escenas B-roll (el sujeto desaparece) ---
    def find_word(word):
        trig = _norm(word)
        for seg in transcript or []:
            for w in seg.get("words", []):
                if _norm(w["w"]) == trig:
                    return w["s"], seg
        return None, None

    broll = []
    for spec in cfg.get("broll", []):
        t_src, seg = find_word(spec.get("after_word", ""))
        if t_src is None:
            continue
        items = spec.get("items")
        if not items and seg:                 # tipografia cinetica: la frase dicha
            items = [w["w"] for w in seg.get("words", [])][:10]
        start = max(intro, to_out(t_src) + spec.get("delay", 0.15))
        broll.append({"type": spec.get("type", "bars"), "title": spec.get("title", ""),
                      "items": items or [], "start": start, "dur": spec.get("dur", 2.6),
                      "pip": spec.get("pip", False),
                      "captions": spec.get("captions", False),
                      "wipe_in": spec.get("wipe_in", 0.30),
                      "wipe_out": spec.get("wipe_out", 0.30)})
    broll.sort(key=lambda b: b["start"])
    for a, b in zip(broll, broll[1:]):        # sin solapes entre escenas
        a["dur"] = min(a["dur"], max(0.6, b["start"] - a["start"] - 0.4))

    # --------------------------------------- acentos ritmicos del audio -----
    punches = []
    for tb in (beats or []):
        ot = to_out(tb)
        if ot > intro + 0.2:
            punches.append({"t": ot, "amount": 1.0})

    # -------------------------------------- tarjetas de datos (refuerzo) ----
    cards = []
    for spec in cfg.get("cards", []):
        trig = _norm(spec.get("after_word", ""))
        hit = None
        for seg in transcript or []:
            for w in seg.get("words", []):
                if _norm(w["w"]) == trig:
                    hit = w["s"]
                    break
            if hit is not None:
                break
        if hit is None:
            continue
        cards.append({"title": spec.get("title", ""), "items": spec.get("items", []),
                      "start": max(intro, to_out(hit) + spec.get("delay", 0.25)),
                      "dur": spec.get("dur", 2.4)})

    # -------------------------------------- tarjetas de datos (refuerzo) ----
    cards = []
    for spec in cfg.get("cards", []):
        trig = _norm(spec.get("after_word", ""))
        hit = None
        for seg in transcript or []:
            for w in seg.get("words", []):
                if _norm(w["w"]) == trig:
                    hit = w["s"]
                    break
            if hit is not None:
                break
        if hit is None:
            continue
        cards.append({"title": spec.get("title", ""), "items": spec.get("items", []),
                      "start": max(intro, to_out(hit) + spec.get("delay", 0.25)),
                      "dur": spec.get("dur", 2.4)})

    # ------------------------------------------------------- SFX + shakes ---
    sfx = [{"t": 0.05, "kind": "riser", "dur": max(0.5, intro - 0.15), "gain": 0.9},
           {"t": intro - 0.22, "kind": "impact", "gain": 0.9}]
    shakes = [{"t": intro - 0.22, "amount": 1.0, "dur": 0.3}]
    for c in clips[1:]:
        kind = c["transition"]
        sfx.append({"t": c["out_start"] - 0.10, "kind": "whoosh", "gain": 0.55,
                    "seed": c["seed"] % 999, "dir": 1 if "left" in str(kind) else -1})
    for it in caption_items[:40]:
        sfx.append({"t": it["start"], "kind": "pop", "gain": 0.35,
                    "freq": 820 + (int(it["start"] * 7) % 5) * 90})
    for b in broll:
        sfx.append({"t": b["start"] - 0.12, "kind": "whoosh", "gain": 0.6, "seed": 5, "dir": 1})
        sfx.append({"t": b["start"] + b["dur"] - 0.12, "kind": "whoosh", "gain": 0.5,
                    "seed": 9, "dir": -1})
        chips[:] = [c for c in chips
                    if c["start"] + c["dur"] < b["start"] - 0.1
                    or c["start"] > b["start"] + b["dur"] + 0.3]
        cards[:] = [c for c in cards
                    if c["start"] + c["dur"] < b["start"] - 0.1
                    or c["start"] > b["start"] + b["dur"] + 0.3]
    for cd in cards:   # un chip y una tarjeta nunca compiten por la misma banda
        chips[:] = [c for c in chips
                    if c["start"] + c["dur"] < cd["start"] - 0.1
                    or c["start"] > cd["start"] + cd["dur"] + 0.3]
        sfx.append({"t": cd["start"], "kind": "whoosh", "gain": 0.4, "seed": 77, "dir": -1})
    for ch in chips:
        sfx.append({"t": ch["start"], "kind": "pop", "gain": 0.5, "freq": 1250})
    for cd in cards:
        sfx.append({"t": cd["start"], "kind": "whoosh", "gain": 0.4, "seed": 77, "dir": -1})
        chips[:] = [c for c in chips
                    if c["start"] + c["dur"] < cd["start"] - 0.1
                    or c["start"] > cd["start"] + cd["dur"] + 0.3]

    return {
        "fps": fps, "intro": intro, "outro": outro, "total": total,
        "clips": clips, "captions": caption_items, "chips": chips, "cards": cards,
        "broll": broll, "punches": punches, "cards": cards,
        "sfx": sfx, "shakes": shakes,
        "ranges": ranges,
    }


def camera_at(clip, t_local, cfg, noise):
    """Zoom/desplazamiento/rotacion de la camara virtual en un instante."""
    dur = max(0.001, clip["n"] / 30.0)
    p = util.clamp(t_local / dur)
    amp = cfg["camera"]["move_amount"]
    z, dx, dy = clip["zoom_base"], 0.0, 0.0
    m = clip["move"]
    if m == "push_in":
        z += amp * util.ease_in_out(p)
    elif m == "pull_out":
        z += amp * (1 - util.ease_in_out(p))
    elif m == "drift_left":
        dx = -amp * 420 * util.ease_in_out(p)
        z += amp * 0.6
    elif m == "drift_right":
        dx = amp * 420 * util.ease_in_out(p)
        z += amp * 0.6
    # flotacion tipo camara en mano, siempre activa (muy sutil)
    k = cfg["camera"]["handheld"]
    i = int(t_local * 30) % len(noise[0])
    dx += noise[0][i] * 9 * k
    dy += noise[1][i] * 9 * k
    rot = noise[2][i] * 0.35 * k
    return z, dx, dy, rot


def split_on_sentences(ranges, transcript, min_shot=2.4, fps=30):
    """Divide los planos largos en los limites de frase.

    No elimina audio (el corte es contiguo): sirve para cambiar el encuadre y
    meter una transicion, que es lo que hace un editor cuando el discurso es
    continuo y no hay silencios que recortar.
    """
    if not transcript:
        return ranges
    bounds = sorted({round(s["start"], 3) for s in transcript})
    word_starts = sorted({round(w["s"], 3) for seg in transcript
                          for w in seg.get("words", [])})
    out = []
    for (s, e) in ranges:
        pts = [s] + [b for b in bounds if s + min_shot <= b <= e - min_shot] + [e]
        merged = [pts[0]]
        for p in pts[1:]:
            if p - merged[-1] >= min_shot:
                merged.append(p)
        merged[-1] = e
        # si un plano sigue siendo muy largo, se parte en el inicio de palabra
        # mas cercano a su mitad: mantiene el audio intacto y da ritmo
        grown = True
        while grown:
            grown = False
            for i in range(len(merged) - 1):
                a, b = merged[i], merged[i + 1]
                if b - a >= 2 * min_shot:
                    mid = (a + b) / 2
                    cand = [x for x in word_starts if a + min_shot <= x <= b - min_shot]
                    if cand:
                        merged.insert(i + 1, min(cand, key=lambda x: abs(x - mid)))
                        grown = True
                        break
        for a, b in zip(merged, merged[1:]):
            a = round(a * fps) / fps
            b = round(b * fps) / fps
            if b - a >= 2 / fps:
                out.append((a, b))
    return out or ranges
