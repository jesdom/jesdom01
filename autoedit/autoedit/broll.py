"""Escenas B-roll a pantalla completa, generadas por codigo.

Sustituyen la imagen del sujeto durante unos segundos para que el video no sea
un plano fijo de alguien hablando: barras comparativas, pasos de un flujo,
lista de verificacion y tipografia cinetica. Fondo animado (degradado + rejilla
en movimiento + blobs con parallax + grano) dibujado con Pillow/numpy.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import util

RGBA = "RGBA"


class Scenes:
    def __init__(self, gfx, cfg, size):
        self.g = gfx
        self.cfg = cfg
        self.W, self.H = size
        self.s = gfx.s
        self._bg_cache = None

    # ------------------------------------------------------------- fondo ---
    def backdrop(self, t):
        """Degradado de marca + rejilla que se desplaza + blobs con parallax."""
        W, H = self.W, self.H
        if self._bg_cache is None:
            ink = np.array(self.g.ink, dtype=np.float32)
            top = np.clip(ink * 1.9 + 8, 0, 255)
            grad = np.linspace(0, 1, H, dtype=np.float32)[:, None] ** 1.2
            col = top[None, :] * (1 - grad) + ink[None, :] * grad      # (H, 3)
            self._bg_cache = np.repeat(col[:, None, :], W, axis=1)     # (H, W, 3)
        a = self._bg_cache.copy()
        img = Image.fromarray(np.uint8(a)).convert(RGBA)

        # blobs de color con parallax
        blob = Image.new(RGBA, (W, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blob)
        for i, col in enumerate((self.g.accent, self.g.accent2, self.g.accent)):
            ph = t * (0.18 + 0.07 * i)
            cx = W * (0.25 + 0.5 * (0.5 + 0.5 * np.sin(ph * 2 * np.pi + i)))
            cy = H * (0.22 + 0.55 * (0.5 + 0.5 * np.cos(ph * 2 * np.pi * 0.7 + i)))
            r = W * (0.34 + 0.08 * np.sin(ph * 3 + i))
            bd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (42,))
        blob = blob.filter(ImageFilter.GaussianBlur(W * 0.09))
        img.alpha_composite(blob)

        # rejilla en movimiento
        grid = Image.new(RGBA, (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grid)
        step = int(96 * self.s)
        off = int((t * 34 * self.s) % step)
        for x in range(-step, W + step, step):
            gd.line([(x + off, 0), (x + off, H)], fill=(255, 255, 255, 16), width=1)
        for y in range(-step, H + step, step):
            gd.line([(0, y + off), (W, y + off)], fill=(255, 255, 255, 12), width=1)
        img.alpha_composite(grid)
        return img.convert("RGB")

    # --------------------------------------------------------------- PIP ---
    def pip(self, img, frame_img, p):
        """Sujeto en pequeno, con marco: sigue presente pero no ocupa la toma."""
        w = int(self.W * 0.30)
        h = int(w * 1.28)
        src = frame_img.resize((int(w * 1.5), int(w * 1.5 * self.H / self.W)), Image.BICUBIC)
        left = (src.width - w) // 2
        src = src.crop((left, int(src.height * 0.06), left + w, int(src.height * 0.06) + h))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=int(34 * self.s),
                                               fill=255)
        k = util.spring(util.clamp(p / 0.35), 2.0, 7.0)
        x = self.W - w - int(56 * self.s)
        y = int(self.H * 0.085) + int((1 - k) * 90 * self.s)
        shadow = Image.new(RGBA, (w + 40, h + 40), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle([20, 24, w + 20, h + 24],
                                                 radius=int(34 * self.s), fill=(0, 0, 0, 150))
        img.paste(Image.alpha_composite(img.convert(RGBA).crop(
            (x - 20, y - 20, x + w + 20, y + h + 20)), shadow).convert("RGB"), (x - 20, y - 20))
        img.paste(src, (x, y), mask)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([x, y, x + w, y + h], radius=int(34 * self.s),
                            outline=self.g.accent, width=int(4 * self.s))
        return img

    # ------------------------------------------------------------ escenas --
    def _title(self, d, text, y, p=1.0, font=None, left=False):
        """Titulo de escena con barra de acento que crece debajo."""
        font = font or self.g.f_card
        a = int(255 * util.clamp(p / 0.18))
        bb = font.getbbox(text)
        w = bb[2] - bb[0]
        x = int(self.W * 0.11) if left else (self.W - w) / 2
        off = (1 - util.ease_out_expo(util.clamp(p / 0.22))) * 34 * self.s
        d.text((x, y + off), text, font=font, fill=self.g.paper + (a,))
        grow = util.ease_out_expo(util.clamp((p - 0.06) / 0.3))
        bw = w * grow
        bx = x if left else (self.W - bw) / 2
        d.rounded_rectangle([bx, y + off + (bb[3] - bb[1]) + 22 * self.s,
                             bx + bw, y + off + (bb[3] - bb[1]) + 32 * self.s],
                            radius=int(5 * self.s), fill=self.g.accent + (a,))
        return bb[3] - bb[1]

    def bars(self, scene, p, t):
        img = self.backdrop(t).convert(RGBA)
        d = ImageDraw.Draw(img)
        self._title(d, scene["title"].upper(), self.H * 0.30, p, left=scene.get("pip", False))
        rows = scene["items"]
        x0 = int(self.W * 0.11)
        x1 = int(self.W * 0.89)
        for i, (label, value, frac) in enumerate(rows):
            y = int(self.H * (0.46 + 0.20 * i))
            grow = util.ease_out_expo(util.clamp((p - 0.12 - i * 0.14) / 0.55))
            col = self.g.accent if i == len(rows) - 1 else self.g.accent2
            d.text((x0, y - int(66 * self.s)), label, font=self.g.f_item,
                   fill=self.g.paper + (215,))
            h = int(64 * self.s)
            d.rounded_rectangle([x0, y, x1, y + h], radius=h // 2, fill=(255, 255, 255, 26))
            bw = max(int(10 * self.s), int((x1 - x0) * float(frac) * grow))
            d.rounded_rectangle([x0, y, x0 + bw, y + h], radius=h // 2, fill=col + (250,))
            vb = self.g.f_big.getbbox(str(value))
            d.text((x0, y + h + int(14 * self.s)), str(value), font=self.g.f_big,
                   fill=col + (int(255 * util.clamp(grow * 1.5)),))
        return img.convert("RGB")

    def steps(self, scene, p, t):
        img = self.backdrop(t).convert(RGBA)
        d = ImageDraw.Draw(img)
        self._title(d, scene["title"].upper(), self.H * 0.26, p, left=scene.get("pip", False))
        items = scene["items"]
        x0, x1 = int(self.W * 0.12), int(self.W * 0.88)
        top = self.H * 0.40
        gap = self.H * 0.155
        for i, label in enumerate(items):
            on = util.ease_out_back(util.clamp((p - 0.10 - i * 0.16) / 0.42))
            y = top + gap * i
            h = int(118 * self.s)
            alpha = int(255 * util.clamp(on))
            col = self.g.accent if on > 0.6 else self.g.accent2
            w = (x1 - x0) * (0.55 + 0.45 * util.clamp(on))
            d.rounded_rectangle([x0, y, x0 + w, y + h], radius=int(24 * self.s),
                                fill=self.g.ink + (int(220 * util.clamp(on)),),
                                outline=col + (alpha,), width=int(3 * self.s))
            r = int(20 * self.s)
            cx, cy = x0 + int(52 * self.s), y + h / 2
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (alpha,))
            d.text((cx + r + int(30 * self.s), cy - int(28 * self.s)), label,
                   font=self.g.f_item, fill=self.g.paper + (alpha,))
            if i < len(items) - 1:                       # conector que crece
                prog = util.clamp((p - 0.22 - i * 0.16) / 0.3)
                d.line([(cx, y + h), (cx, y + h + gap * prog - h * 0.0)],
                       fill=col + (alpha,), width=int(5 * self.s))
        return img.convert("RGB")

    def checks(self, scene, p, t):
        img = self.backdrop(t).convert(RGBA)
        d = ImageDraw.Draw(img)
        self._title(d, scene["title"].upper(), self.H * 0.28, p, left=scene.get("pip", False))
        items = scene["items"]
        x0 = int(self.W * 0.13)
        for i, label in enumerate(items):
            on = util.clamp((p - 0.14 - i * 0.18) / 0.3)
            y = self.H * (0.44 + 0.135 * i)
            b = int(78 * self.s)
            col = self.g.accent if on > 0.5 else (255, 255, 255)
            d.rounded_rectangle([x0, y, x0 + b, y + b], radius=int(16 * self.s),
                                outline=col + (235,), width=int(4 * self.s),
                                fill=col + (int(220 * util.clamp((on - 0.4) / 0.4)),))
            if on > 0.45:
                k = util.ease_out_expo((on - 0.45) / 0.55)
                pts = [(x0 + b * .22, y + b * .52), (x0 + b * .44, y + b * .74),
                       (x0 + b * .80, y + b * .28)]
                seg = pts[:2] if k < 0.5 else pts
                d.line(seg, fill=self.g.ink + (255,), width=int(7 * self.s), joint="curve")
            d.text((x0 + b + int(34 * self.s), y + int(14 * self.s)), label,
                   font=self.g.f_item, fill=self.g.paper + (int(235 * util.clamp(on * 2)),))
        return img.convert("RGB")

    def kinetic(self, scene, p, t):
        """Tipografia grande: las palabras de la frase entran escalonadas."""
        img = self.backdrop(t).convert(RGBA)
        d = ImageDraw.Draw(img)
        words = scene["items"]
        n = max(1, len(words))
        font = self.g.f_title
        lines, cur = [], []
        for w in words:
            cand = cur + [w]
            if len(" ".join(cand)) > 14 and cur:
                lines.append(cur)
                cur = [w]
            else:
                cur = cand
        if cur:
            lines.append(cur)
        total_h = len(lines) * 132 * self.s
        y = self.H * 0.42 - total_h / 2
        idx = 0
        for line in lines:
            txt = " ".join(line)
            bb = font.getbbox(txt)
            x = (self.W - (bb[2] - bb[0])) / 2
            for w in line:
                on = util.ease_out_back(util.clamp((p - 0.08 - idx * 0.09) / 0.32))
                wb = font.getbbox(w + " ")
                alpha = int(255 * util.clamp(on * 1.4))
                off = (1 - util.clamp(on)) * 40 * self.s
                hot = idx == min(n - 1, int(util.clamp(p * 1.1) * n))
                if hot and alpha > 200:
                    pad = int(12 * self.s)
                    d.rounded_rectangle([x - pad, y - pad * .3 + off,
                                         x + (wb[2] - wb[0]) - pad, y + 108 * self.s + off],
                                        radius=int(14 * self.s), fill=self.g.accent + (235,))
                    d.text((x, y + off - bb[1] + 8 * self.s), w, font=font,
                           fill=self.g.ink + (255,))
                else:
                    d.text((x, y + off - bb[1] + 8 * self.s), w, font=font,
                           fill=self.g.paper + (alpha,))
                x += wb[2] - wb[0]
                idx += 1
            y += 132 * self.s
        return img.convert("RGB")

    def _footer(self, img, scene, p):
        """Pie de escena: puntos de avance + filo de acento (equilibra el bajo)."""
        d = ImageDraw.Draw(img)
        n = max(1, len(scene.get("items", [])))
        r = int(9 * self.s)
        gap = int(34 * self.s)
        y = int(self.H * 0.86)
        x0 = (self.W - (n * gap)) / 2 + gap / 2
        for i in range(n):
            on = util.clamp((p - 0.1 - i * 0.16) / 0.25)
            col = self.g.accent if on > 0.5 else (255, 255, 255)
            a = int(90 + 165 * on)
            d.ellipse([x0 + i * gap - r, y - r, x0 + i * gap + r, y + r], fill=col + (a,))
        w = self.W * 0.42 * util.ease_out_expo(util.clamp(p / 0.35))
        d.rounded_rectangle([(self.W - w) / 2, y + int(34 * self.s),
                             (self.W + w) / 2, y + int(40 * self.s)],
                            radius=int(3 * self.s), fill=self.g.accent + (150,))

    # --------------------------------------------------------- despacho ----
    def render(self, scene, p, t, frame_img):
        kind = scene.get("type", "bars")
        fn = {"bars": self.bars, "steps": self.steps,
              "checks": self.checks, "kinetic": self.kinetic}.get(kind, self.bars)
        img = fn(scene, p, t).convert(RGBA)
        self._footer(img, scene, p)
        if scene.get("pip", False) and frame_img is not None:
            img = self.pip(img, frame_img, p)
        return img.convert("RGB")


def wipe(video_img, scene_img, p, direction=1, accent=(255, 255, 255)):
    """Barrido diagonal con filo de color entre video y escena."""
    W, H = video_img.size
    p = util.clamp(p)
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    diag = (xx * 0.55 + yy * 0.45) if direction > 0 else (1 - (xx * 0.55 + yy * 0.45))
    edge = 0.16
    m = np.clip((p * (1 + edge) - diag) / edge, 0, 1)
    a = np.asarray(video_img, dtype=np.float32)
    b = np.asarray(scene_img, dtype=np.float32)
    out = a * (1 - m[:, :, None]) + b * m[:, :, None]
    line = np.clip(1 - np.abs(p * (1 + edge) - diag) / (edge * 0.28), 0, 1)[:, :, None]
    out = out * (1 - line) + np.array(accent, dtype=np.float32)[None, None, :] * line
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)))
