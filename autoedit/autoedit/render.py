"""Render frame a frame: camara virtual, transiciones, grade y graficos."""
import subprocess

import numpy as np
from PIL import Image

from . import effects as E
from . import layout
from .broll import Scenes, wipe
from . import util
from .graphics import Graphics
from .timeline import camera_at


class FrameSource:
    """Lectura secuencial de frames RGB desde ffmpeg (permite saltar hacia delante)."""

    def __init__(self, path, w, h, fps):
        self.w, self.h, self.fps = w, h, fps
        self.n = w * h * 3
        self.idx = -1
        self.cur = None
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h},fps={fps}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, bufsize=self.n * 4)

    def _next(self):
        buf = self.p.stdout.read(self.n)
        if len(buf) < self.n:
            return None
        self.idx += 1
        self.cur = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        return self.cur

    def at(self, i):
        """Frame i (solo hacia delante; si ya se paso, devuelve el ultimo)."""
        if self.cur is None:
            self._next()
        while self.idx < i:
            if self._next() is None:
                break
        return self.cur

    def close(self):
        try:
            self.p.stdout.close()
            self.p.wait(timeout=5)
        except Exception:
            pass


def transform(img, zoom, dx, dy, rot, W, H):
    """Zoom + desplazamiento + rotacion sin bordes negros."""
    if abs(rot) > 0.02:
        img = img.rotate(rot, resample=Image.BICUBIC, expand=False)
    z = max(1.0, zoom)
    nw, nh = int(W * z), int(H * z)
    img = img.resize((nw, nh), Image.BICUBIC)
    cx = (nw - W) / 2 + dx
    cy = (nh - H) / 2 + dy
    cx = min(max(cx, 0), nw - W)
    cy = min(max(cy, 0), nh - H)
    return img.crop((int(cx), int(cy), int(cx) + W, int(cy) + H))


def render(video_in, plan, cfg, out_path, audio_wav, workdir, size=(1080, 1920), fps=30,
           preview_frames=None):
    W, H = size
    zone = layout.safe_zone(video_in, size, cfg)
    zoom_max = 1.0 + cfg["camera"]["base_zoom"] + cfg["camera"]["move_amount"] + 0.02
    zone["text_top"] = min(0.78, (zone["text_top"] - 0.5) * zoom_max + 0.5)
    zone["caption_y"] = min(zone["floor"] - 0.055, (zone["caption_y"] - 0.5) * zoom_max + 0.5)
    zone["chip_y"] = min(zone["caption_y"] - 0.085, (zone["chip_y"] - 0.5) * zoom_max + 0.5)
    gfx = Graphics(cfg, size, cfg["_font_dir"], zone=zone)
    scenes = Scenes(gfx, cfg, size)
    lut = E.build_lut(**cfg["grade"]["lut"])
    vig = E.vignette(W, H, cfg["grade"]["vignette"])
    grains = E.grain_bank(W, H, amount=cfg["grade"]["grain"])
    noise = [util.value_noise_1d(900, seed=s) for s in (11, 22, 33)]
    rng = np.random.default_rng(cfg.get("seed", 7))

    intro, outro = plan["intro"], plan["outro"]
    total = plan["total"]
    clips = plan["clips"]
    src = FrameSource(video_in, W, H, fps)

    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-i", audio_wav,
         "-c:v", "libx264", "-preset", cfg["export"]["preset"], "-crf", str(cfg["export"]["crf"]),
         "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart",
         "-af", "highpass=f=85,acompressor=ratio=3:threshold=-20dB:makeup=2,loudnorm=I=-14:TP=-1.5:LRA=11",
         "-c:a", "aac", "-b:a", "192k", "-shortest", out_path],
        stdin=subprocess.PIPE)

    first = src.at(int(round(clips[0]["src_start"] * fps)))
    first_img = Image.fromarray(first)
    plate = E.background_plate(first_img, darken=cfg["intro"]["plate_darken"])
    n_intro = int(round(intro * fps))
    n_total = int(round(total * fps))
    written = 0

    def active_broll(t):
        for b in plan["broll"]:
            if b["start"] - b["wipe_in"] <= t <= b["start"] + b["dur"] + b["wipe_out"]:
                return b
        return None

    def apply_broll(img, b, t):
        """Sustituye la toma por la escena, con barrido de entrada y salida."""
        age = t - b["start"]
        p = util.clamp(age / max(0.3, b["dur"]))
        scene = scenes.render(b, p, t, img)
        if age < 0:
            return wipe(img, scene, 1 - (-age / b["wipe_in"]), 1, gfx.accent)
        out_age = age - b["dur"]
        if out_age > 0:
            return wipe(scene, img, out_age / b["wipe_out"], -1, gfx.accent)
        return scene

    def finish(img, t):
        """Grade + grano + vineta + graficos comunes."""
        arr = E.apply_lut(np.asarray(img), lut)
        img = Image.fromarray(arr.astype(np.uint8))
        if cfg["grade"]["saturation"] != 1.0:
            img = E.saturate(img, cfg["grade"]["saturation"])
        if cfg["grade"]["bloom"]:
            img = E.bloom(img, strength=cfg["grade"]["bloom"])
        a = np.asarray(img, dtype=np.float32) * vig
        if cfg["grade"]["grain"] > 0:
            a = E.add_grain(a, grains[int(t * fps) % len(grains)])
        img = Image.fromarray(np.uint8(np.clip(a, 0, 255)))
        if active_broll(t) is None:
            img = gfx.scrim(img)
        img = img.convert("RGBA")
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        b = active_broll(t)
        card_on = False
        for cd in plan["cards"]:
            card_on = gfx.card(layer, cd, t) or card_on
        show_text = not card_on and (b is None or b.get("captions"))
        if show_text:
            gfx.captions(layer, plan["captions"], t)
            gfx.chips(layer, plan["chips"], t)
        gfx.brand(layer, t, total)
        gfx.progress(layer, t, total, intro)
        if t < intro + 0.05:
            gfx.intro(layer, t, intro)
        if t >= total - outro * 0.72:
            gfx.outro(layer, t, total - outro * 0.72, outro * 0.72)
        img.alpha_composite(layer)
        return img.convert("RGB")

    def push(img, t):
        nonlocal written
        enc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        written += 1

    # ---------------------------------------------------------------- intro
    for k in range(n_intro):
        t = k / fps
        p = util.ease_out_cubic(util.clamp((t - 0.12) / (intro * 0.72)))
        base = transform(first_img, 1.10 - 0.08 * util.ease_in_out(t / max(0.1, intro)),
                         0, 0, 0, W, H)
        pl = transform(plate, 1.10 - 0.08 * util.ease_in_out(t / max(0.1, intro)), 0, 0, 0, W, H)
        img = E.materialize(base, pl, p, np.random.default_rng(1000 + k // 2))
        if p > 0.92:  # golpe de luz al completarse
            img = E.flash(img, (1 - (p - 0.92) / 0.08) * 0.22)
        push(finish(img, t), t)
        if preview_frames and written >= preview_frames:
            enc.stdin.close(); enc.wait(); src.close(); return out_path

    # ----------------------------------------------------------- cuerpo ----
    for c in clips:
        start_src = int(round(c["src_start"] * fps))
        for k in range(c["n"]):
            t = c["out_start"] + k / fps
            frame = src.at(start_src + k)
            if frame is None:
                break
            img = Image.fromarray(frame)
            z, dx, dy, rot = camera_at(c, k / fps, cfg, noise)
            tr, age = c["transition"], k / fps
            extra_blur = 0.0
            if tr and age < 0.34:
                q = 1 - util.ease_out_expo(age / 0.34)
                if tr == "zoom_punch":
                    z += 0.14 * q
                elif tr == "whip_left":
                    dx -= 260 * q; extra_blur = 42 * q
                elif tr == "whip_right":
                    dx += 260 * q; extra_blur = 42 * q
                # micro shake en todo corte
                dx += np.sin(age * 90) * 10 * q
                dy += np.cos(age * 77) * 8 * q
            for pu in plan["punches"]:          # acento: micro zoom en los picos
                d = t - pu["t"]
                if 0 <= d <= 0.26:
                    z += 0.030 * (1 - util.ease_out_expo(d / 0.26)) * pu["amount"]
            img = transform(img, z, dx, dy, rot, W, H)
            if extra_blur > 1:
                img = E.motion_blur_h(img, extra_blur)
            if tr and age < 0.34:
                q = 1 - util.ease_out_expo(age / 0.34)
                if tr == "flash":
                    img = E.flash(img, 0.55 * q)
                elif tr == "glitch":
                    img = E.slice_glitch(img, 0.55 * q, rng)
                    img = E.rgb_split(img, 10 * q)
            b = active_broll(t)
            if b is not None:
                img = apply_broll(img, b, t)
            # realce ritmico en las palabras acentuadas
            for sh in plan["shakes"]:
                d = t - sh["t"]
                if 0 <= d <= sh["dur"]:
                    q = (1 - d / sh["dur"]) * sh["amount"]
                    img = E.rgb_split(img, 5 * q)
            push(finish(img, t), t)
            if preview_frames and written >= preview_frames:
                enc.stdin.close(); enc.wait(); src.close(); return out_path

    # ---------------------------------------------------------------- outro
    last = Image.fromarray(src.cur)
    plate_out = E.background_plate(last, darken=cfg["intro"]["plate_darken"])
    n_out = max(0, n_total - written)
    for k in range(n_out):
        t = written / fps
        p = 1 - util.ease_in_out(util.clamp((k / fps - 0.35) / max(0.2, outro - 0.5)))
        z = 1.02 + 0.06 * util.ease_out_cubic(k / max(1, n_out))
        img = E.materialize(transform(last, z, 0, 0, 0, W, H),
                            transform(plate_out, z, 0, 0, 0, W, H),
                            p, np.random.default_rng(2000 + k // 2))
        push(finish(img, t), t)

    enc.stdin.close()
    enc.wait()
    src.close()
    return out_path
