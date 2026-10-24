"""Motion graphics dibujados con PIL: subtitulos karaoke, chips, barras, cards."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import assets, util

RGBA = "RGBA"


def rounded(draw, box, r, fill=None, outline=None, width=3):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


class Graphics:
    def __init__(self, cfg, size, font_dir, zone=None):
        self.cfg = cfg
        self.zone = zone or {"text_top": 0.66, "caption_y": 0.80, "chip_y": 0.70,
                             "floor": 0.915, "scrim": False, "band_h": 0.25}
        self.W, self.H = size
        s = self.W / 1080.0
        self.s = s
        head = assets.fetch_font(cfg["fonts"]["display"], cfg["fonts"]["display_weight"], font_dir)
        body = assets.fetch_font(cfg["fonts"]["body"], cfg["fonts"]["body_weight"], font_dir)
        self.f_cap = ImageFont.truetype(head, int(cfg["captions"]["size"] * s))
        self.f_chip = ImageFont.truetype(head, int(34 * s))
        self.f_title = ImageFont.truetype(head, int(104 * s))
        self.f_sub = ImageFont.truetype(body, int(38 * s))
        self.f_tag = ImageFont.truetype(body, int(28 * s))
        self.f_card = ImageFont.truetype(head, int(44 * s))
        self.f_cardv = ImageFont.truetype(head, int(52 * s))
        self.f_small = ImageFont.truetype(body, int(30 * s))
        self.accent = util.hex_rgb(cfg["brand"]["accent"])
        self.accent2 = util.hex_rgb(cfg["brand"]["accent2"])
        self.ink = util.hex_rgb(cfg["brand"]["ink"])
        self.paper = util.hex_rgb(cfg["brand"]["paper"])
        self._word_cache = {}

    # ------------------------------------------------------------ helpers --
    def _word_img(self, text, active):
        key = (text, active)
        if key in self._word_cache:
            return self._word_cache[key]
        f = self.f_cap
        pad = int(18 * self.s)
        bbox = f.getbbox(text, stroke_width=int(8 * self.s))
        w = bbox[2] - bbox[0] + pad * 2
        h = bbox[3] - bbox[1] + pad * 2
        im = Image.new(RGBA, (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        x, y = pad - bbox[0], pad - bbox[1]
        if active:
            box = (pad * 0.30, pad * 0.40, w - pad * 0.30, h - pad * 0.40)
            d.rounded_rectangle([box[0] + 5 * self.s, box[1] + 7 * self.s,
                                 box[2] + 5 * self.s, box[3] + 7 * self.s],
                                radius=int(16 * self.s), fill=(0, 0, 0, 110))
            rounded(d, box, int(16 * self.s), fill=self.accent + (255,))
            d.text((x, y), text, font=f, fill=self.ink + (255,))
        else:
            d.text((x, y + 5 * self.s), text, font=f, fill=(0, 0, 0, 130),
                   stroke_width=int(9 * self.s), stroke_fill=(0, 0, 0, 130))
            d.text((x, y), text, font=f, fill=self.paper + (255,),
                   stroke_width=int(9 * self.s), stroke_fill=(10, 10, 14, 235))
        self._word_cache[key] = im
        return im

    # ------------------------------------------------------- subtitulos ----
    def captions(self, layer, items, t):
        cfg = self.cfg["captions"]
        live = [it for it in items if it["start"] - 0.18 <= t <= it["end"] + 0.18]
        started = [it for it in live if it["start"] <= t]
        live = started[-1:] if started else live[:1]
        for it in live:
            age = t - it["start"]
            life = it["end"] - it["start"]
            inp = util.spring(util.clamp(age / 0.22), 2.4, 9.0)
            outp = 1.0 - util.ease_out_cubic(util.clamp((t - it["end"]) / 0.18))
            scale = (0.82 + 0.18 * inp) * (0.94 + 0.06 * outp)
            alpha = util.clamp(age / 0.10) * outp
            if alpha <= 0.02:
                continue
            words = it["words"]
            imgs = []
            for w in words:
                active = w["s"] - 0.02 <= t <= w["e"] + 0.06
                im = self._word_img(w["w"], active)
                if active:  # la palabra activa late un poco
                    k = 1.0 + 0.07 * (1 - util.clamp((t - w["s"]) / 0.18))
                    im = im.resize((int(im.width * k), int(im.height * k)), Image.BICUBIC)
                imgs.append(im)
            gap = int(6 * self.s)
            tw = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
            th = max(i.height for i in imgs)
            strip = Image.new(RGBA, (tw, th), (0, 0, 0, 0))
            x = 0
            for im in imgs:
                strip.alpha_composite(im, (x, (th - im.height) // 2))
                x += im.width + gap
            sw, sh = max(1, int(tw * scale)), max(1, int(th * scale))
            strip = strip.resize((sw, sh), Image.BICUBIC)
            if alpha < 0.999:
                a = strip.getchannel("A").point(lambda v: int(v * alpha))
                strip.putalpha(a)
            cx = self.W // 2
            cy = int(self.H * self.zone["caption_y"]) + int((1 - inp) * 26 * self.s)
            cy = min(cy, int(self.H * self.zone["floor"]) - sh // 2)
            layer.alpha_composite(strip, (cx - sw // 2, cy - sh // 2))

    # ------------------------------------------------------------- chips ---
    def _icon(self, d, x, y, r, kind, col):
        if kind == "bolt":
            d.polygon([(x, y - r), (x - r * .55, y + r * .12), (x - r * .08, y + r * .12),
                       (x - r * .25, y + r), (x + r * .6, y - r * .2),
                       (x + r * .05, y - r * .2)], fill=col)
        elif kind == "clock":
            d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=int(4 * self.s))
            d.line([x, y, x, y - r * .6], fill=col, width=int(4 * self.s))
            d.line([x, y, x + r * .45, y + r * .2], fill=col, width=int(4 * self.s))
        elif kind == "folder":
            d.rounded_rectangle([x - r, y - r * .6, x + r, y + r * .75], radius=int(5 * self.s), fill=col)
            d.rounded_rectangle([x - r, y - r * .95, x - r * .05, y - r * .45],
                                radius=int(4 * self.s), fill=col)
        elif kind == "check":
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
            d.line([(x - r * .45, y), (x - r * .1, y + r * .42), (x + r * .5, y - r * .45)],
                   fill=self.ink + (255,), width=int(6 * self.s), joint="curve")
        else:  # spark
            for ang, ln in ((0, 1), (90, 1), (45, .6), (135, .6)):
                a = np.deg2rad(ang)
                dx, dy = np.cos(a) * r * ln, np.sin(a) * r * ln
                d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=col, width=int(5 * self.s))

    def chips(self, layer, items, t):
        for ch in items:
            age = t - ch["start"]
            if not (-0.1 <= age <= ch["dur"] + 0.4):
                continue
            inp = util.spring(util.clamp(age / 0.30), 2.2, 8.0)
            outp = util.ease_out_cubic(util.clamp((age - ch["dur"]) / 0.35))
            if outp >= 1:
                continue
            txt = ch["label"]
            pad = int(26 * self.s)
            tb = self.f_chip.getbbox(txt)
            r = int(20 * self.s)
            w = tb[2] - tb[0] + pad * 3 + r * 2
            h = int(78 * self.s)
            im = Image.new(RGBA, (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            rounded(d, (0, 0, w - 1, h - 1), h // 2, fill=self.ink + (232,),
                    outline=self.accent + (255,), width=int(3 * self.s))
            self._icon(d, pad + r * .6, h / 2, r * .8, ch["icon"], self.accent + (255,))
            d.text((pad * 1.6 + r, h / 2 - (tb[3] - tb[1]) / 2 - tb[1]), txt,
                   font=self.f_chip, fill=self.paper + (255,))
            slide = int((1 - inp) * 160 * self.s + outp * 90 * self.s)
            alpha = util.clamp(age / 0.12) * (1 - outp)
            a = im.getchannel("A").point(lambda v: int(v * alpha))
            im.putalpha(a)
            y = int(self.H * self.zone["chip_y"]) - h // 2
            if ch["slot"] % 2 == 0:
                x = int(self.W * 0.06) - slide
            else:
                x = self.W - w - int(self.W * 0.06) + slide
            layer.alpha_composite(im, (x, y))

    # -------------------------------------------------------- barra/brand ---
    def progress(self, layer, t, total, intro):
        if not self.cfg["ui"]["progress_bar"]:
            return
        p = util.clamp((t - intro) / max(0.1, total - intro))
        h = int(9 * self.s)
        y = self.H - h - int(8 * self.s)
        d = ImageDraw.Draw(layer)
        d.rectangle([0, y, self.W, y + h], fill=(255, 255, 255, 46))
        w = int(self.W * p)
        if w > 2:
            d.rectangle([0, y, w, y + h], fill=self.accent + (235,))
            d.ellipse([w - h, y - h * .3, w + h * .4, y + h * 1.3], fill=self.paper + (240,))

    def brand(self, layer, t, total):
        tag = self.cfg["ui"].get("tag")
        if not tag:
            return
        a = util.clamp((t - 0.6) / 0.5) * (1 - util.clamp((t - (total - 1.2)) / 0.4))
        if a <= 0.02:
            return
        im = Image.new(RGBA, (int(520 * self.s), int(60 * self.s)), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([0, int(18 * self.s), int(6 * self.s), int(46 * self.s)],
                    fill=self.accent + (255,))
        d.text((int(18 * self.s), int(16 * self.s)), tag, font=self.f_tag,
               fill=self.paper + (int(220 * a),))
        im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
        layer.alpha_composite(im, (int(40 * self.s),
                                   int(self.H * self.zone["floor"]) + int(10 * self.s)))

    # ------------------------------------------------------ intro / outro ---
    def intro(self, layer, t, intro_dur):
        cfg = self.cfg["intro"]
        title, sub = cfg.get("title", ""), cfg.get("subtitle", "")
        if not title:
            return
        a_in = util.clamp((t - 0.12) / 0.35)
        a_out = 1 - util.clamp((t - intro_dur * 0.52) / (intro_dur * 0.30))
        a = a_in * a_out
        if a <= 0.02:
            return
        d = ImageDraw.Draw(layer)
        lines = title.split("\n")
        y = self.H * self.zone["text_top"] + 10 * self.s
        if cfg.get("center", True):
            y = self.H * 0.40 - len(lines) * 40 * self.s
        for i, line in enumerate(lines):
            bb = self.f_title.getbbox(line)
            w = bb[2] - bb[0]
            x = (self.W - w) / 2
            off = (1 - util.ease_out_expo(a_in)) * 40 * self.s
            yy = y + i * 132 * self.s + off
            d.text((x, yy + 6 * self.s), line, font=self.f_title, fill=(0, 0, 0, int(150 * a)))
            d.text((x, yy), line, font=self.f_title, fill=self.paper + (int(255 * a),))
            if i == len(lines) - 1:   # subrayado que crece
                ul = util.ease_out_expo(util.clamp((t - 0.25) / 0.5))
                d.rounded_rectangle([x, yy + 126 * self.s, x + w * ul, yy + 138 * self.s],
                                    radius=int(6 * self.s), fill=self.accent + (int(255 * a),))
        if sub:
            bb = self.f_sub.getbbox(sub)
            d.text(((self.W - (bb[2] - bb[0])) / 2, y + 160 * self.s), sub,
                   font=self.f_sub, fill=self.paper + (int(200 * a),))

    def outro(self, layer, t, start, dur):
        cfg = self.cfg["outro"]
        title, sub = cfg.get("title", ""), cfg.get("subtitle", "")
        age = t - start
        if age < 0 or not title:
            return
        a = util.clamp(age / 0.3)
        d = ImageDraw.Draw(layer)
        k = util.spring(util.clamp(age / 0.45), 2.0, 7.0)
        bb = self.f_title.getbbox(title)
        w, hgt = bb[2] - bb[0], bb[3] - bb[1]
        x, y = (self.W - w) / 2, self.H * 0.44
        pad = int(34 * self.s)
        rounded(d, (x - pad, y - pad * .7, x + w + pad, y + hgt + pad),
                int(28 * self.s), fill=self.ink + (int(226 * a),),
                outline=self.accent + (int(255 * a),), width=int(4 * self.s))
        d.text((x, y - bb[1] + (1 - k) * 18 * self.s), title, font=self.f_title,
               fill=self.paper + (int(255 * a),))
        if sub:
            sb = self.f_sub.getbbox(sub)
            d.text(((self.W - (sb[2] - sb[0])) / 2, y + hgt + pad * 1.9), sub,
                   font=self.f_sub, fill=self.accent + (int(240 * a),))

    # ------------------------------------------------- cinta de contraste ---
    def scrim(self, img):
        """Degradado oscuro SOLO en la banda inferior (pecho/fondo, nunca cara)."""
        if not self.cfg["layout"].get("scrim", True):
            return img
        top = int(self.H * self.zone["text_top"]) - int(60 * self.s)
        h = self.H - top
        g = np.linspace(0, 1, h, dtype=np.float32) ** 0.85
        strength = self.cfg["layout"].get("scrim_strength", 0.42)
        a = np.asarray(img, dtype=np.float32)
        a[top:] *= (1 - g * strength)[:, None, None]
        return Image.fromarray(np.uint8(np.clip(a, 0, 255)))

    # ---------------------------------------------- tarjeta de datos/barras --
    def card(self, layer, card, t):
        """Motion graphic de refuerzo: barras comparativas o dato grande."""
        age = t - card["start"]
        dur = card["dur"]
        if age < -0.1 or age > dur + 0.45:
            return False
        inp = util.spring(util.clamp(age / 0.38), 2.0, 7.5)
        outp = util.ease_out_cubic(util.clamp((age - dur) / 0.35))
        if outp >= 1:
            return False
        alpha = util.clamp(age / 0.12) * (1 - outp)
        pad = int(30 * self.s)
        w = int(self.W * 0.82)
        rows = card.get("items", [])
        h = int((96 + 86 * len(rows)) * self.s)
        im = Image.new(RGBA, (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        rounded(d, (0, 0, w - 1, h - 1), int(30 * self.s), fill=self.ink + (236,),
                outline=self.accent + (190,), width=int(3 * self.s))
        d.text((pad, int(26 * self.s)), card["title"].upper(), font=self.f_small,
               fill=self.paper + (210,))
        bar_x = int(w * 0.40)
        bar_w = w - bar_x - pad
        for i, (label, value, frac) in enumerate(rows):
            y = int((96 + 86 * i) * self.s)
            d.text((pad, y + int(10 * self.s)), label, font=self.f_small,
                   fill=self.paper + (190,))
            grow = util.ease_out_expo(util.clamp((age - 0.25 - i * 0.14) / 0.55))
            col = self.accent if i == len(rows) - 1 else self.accent2
            bw = max(int(6 * self.s), int(bar_w * float(frac) * grow))
            rounded(d, (bar_x, y + int(4 * self.s), bar_x + bar_w, y + int(46 * self.s)),
                    int(14 * self.s), fill=(255, 255, 255, 28), width=0)
            rounded(d, (bar_x, y + int(4 * self.s), bar_x + bw, y + int(46 * self.s)),
                    int(14 * self.s), fill=col + (245,), width=0)
            vb = self.f_card.getbbox(str(value))
            d.text((bar_x + bar_w - (vb[2] - vb[0]), y + int(52 * self.s)), str(value),
                   font=self.f_card, fill=col + (int(255 * util.clamp(grow * 1.4)),))
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
        y0 = int(self.H * self.zone["text_top"]) - int(40 * self.s)
        y0 = min(y0, int(self.H * self.zone["floor"]) - h - int(10 * self.s))
        off = int((1 - inp) * 120 * self.s + outp * 80 * self.s)
        layer.alpha_composite(im, ((self.W - w) // 2, y0 + off))
        return True
