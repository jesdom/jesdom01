"""Efectos de imagen: grade, grano, vineta, glitch, flash, aparicion."""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from . import util


# ------------------------------------------------------------- color -------
def build_lut(contrast=1.10, lift=0.01, gamma=0.98, warm=0.03):
    """LUT 256x3: curva en S + split-tone calido en altas, frio en sombras."""
    x = np.linspace(0, 1, 256, dtype=np.float32)
    y = np.clip((x - 0.5) * contrast + 0.5, 0, 1) ** gamma
    y = np.clip(y + lift * (1 - y), 0, 1)
    r = np.clip(y + warm * (y ** 2), 0, 1)
    g = np.clip(y + warm * 0.25 * (y ** 2), 0, 1)
    b = np.clip(y - warm * 0.55 * (y ** 2) + 0.02 * (1 - y), 0, 1)
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def apply_lut(arr, lut):
    return np.take_along_axis(
        lut[arr.reshape(-1, 3)], np.arange(3)[None, :], axis=1
    ).reshape(arr.shape) if False else np.stack(
        [lut[:, 0][arr[:, :, 0]], lut[:, 1][arr[:, :, 1]], lut[:, 2][arr[:, :, 2]]], axis=2)


def saturate(img, amount):
    return ImageEnhance.Color(img).enhance(amount) if amount != 1.0 else img


# ------------------------------------------------------- mascaras fijas ----
def vignette(w, h, strength=0.35, radius=0.95):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    d = np.sqrt(((xx - cx) / (w * radius / 2)) ** 2 + ((yy - cy) / (h * radius / 2)) ** 2)
    m = np.clip(1 - strength * np.clip(d - 0.55, 0, None) ** 1.6, 0, 1)
    return m.astype(np.float32)[:, :, None]


def grain_bank(w, h, n=6, amount=5.0, seed=3):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((h // 2, w // 2, 1)).astype(np.float32) * amount)
            for _ in range(n)]


def add_grain(arr, g):
    small = Image.fromarray(np.uint8(np.clip(g[:, :, 0] * 8 + 128, 0, 255)))
    full = np.asarray(small.resize((arr.shape[1], arr.shape[0]), Image.BILINEAR),
                      dtype=np.float32)[:, :, None]
    return np.clip(arr.astype(np.float32) + (full - 128) / 8.0, 0, 255)


# ------------------------------------------------------------ glitch -------
def rgb_split(img, px):
    if px < 0.5:
        return img
    a = np.asarray(img, dtype=np.uint8)
    p = int(round(px))
    out = a.copy()
    out[:, p:, 0] = a[:, :-p or None, 0]
    out[:, :-p or None, 2] = a[:, p:, 2]
    return Image.fromarray(out)


def slice_glitch(img, amount, rng):
    """Desplazamiento horizontal de bandas: corte 'digital'."""
    if amount <= 0.01:
        return img
    a = np.asarray(img).copy()
    h = a.shape[0]
    bands = rng.integers(6, 14)
    for _ in range(int(bands)):
        y0 = int(rng.integers(0, h - 30))
        hh = int(rng.integers(12, 90))
        sh = int(rng.normal(0, 70 * amount))
        if sh:
            a[y0:y0 + hh] = np.roll(a[y0:y0 + hh], sh, axis=1)
    return Image.fromarray(a)


def flash(img, amount, color=(255, 255, 255)):
    if amount <= 0.003:
        return img
    base = np.asarray(img, dtype=np.float32)
    col = np.array(color, dtype=np.float32)
    return Image.fromarray(np.uint8(np.clip(base + (col - base) * amount, 0, 255)))


def motion_blur_h(img, px):
    if px < 1:
        return img
    a = np.asarray(img, dtype=np.float32)
    acc = np.zeros_like(a)
    taps = 5
    for i in range(taps):
        sh = int(round(-px / 2 + px * i / (taps - 1)))
        acc += np.roll(a, sh, axis=1)
    return Image.fromarray(np.uint8(acc / taps))


def bloom(img, threshold=215, strength=0.28, radius=18):
    a = np.asarray(img, dtype=np.float32)
    lum = a.mean(2)
    mask = np.clip((lum - threshold) / max(1, 255 - threshold), 0, 1)[:, :, None]
    hi = Image.fromarray(np.uint8(a * mask)).filter(ImageFilter.GaussianBlur(radius))
    out = a + np.asarray(hi, dtype=np.float32) * strength
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)))


# ------------------------------------------- fondo vacio + materializacion --
def background_plate(frame_img, blur=3, darken=0.82):
    """Plato de fondo SIN sujeto: se teje con franjas limpias de los bordes.

    Funciona con cualquier video de busto parlante (el sujeto ocupa el centro),
    y se mezcla con una version desenfocada del cuadro para conservar la luz.
    """
    a = np.asarray(frame_img)
    h, w, _ = a.shape
    sw = max(24, w // 9)
    left, right = a[:int(h * 0.55), :sw], a[:int(h * 0.55), w - sw:]
    strip = left if left.std() <= right.std() else right
    tile = np.concatenate([strip, strip[:, ::-1]], axis=1)
    tile = np.concatenate([tile, tile[::-1]], axis=0)
    ry = int(np.ceil(h / tile.shape[0]))
    rx = int(np.ceil(w / tile.shape[1]))
    plate = np.tile(tile, (ry, rx, 1))[:h, :w]
    plate_img = Image.fromarray(plate).filter(ImageFilter.GaussianBlur(blur))
    soft = frame_img.resize((w // 8, h // 8)).filter(
        ImageFilter.GaussianBlur(10)).resize((w, h), Image.BILINEAR)
    mixed = Image.blend(plate_img, soft, 0.35)
    return ImageEnhance.Brightness(mixed).enhance(darken)


def materialize(frame_img, plate_img, p, rng, bands=46):
    """p=0 -> solo fondo; p=1 -> sujeto nitido. Disolucion por bandas + ruido."""
    p = util.clamp(p)
    if p >= 0.999:
        return frame_img
    w, h = frame_img.size
    # mascara: umbral de ruido suave que avanza de arriba a abajo
    small_h, small_w = bands, max(8, bands * w // h)
    noise = rng.random((small_h, small_w)).astype(np.float32)
    grad = np.linspace(0.0, 1.0, small_h, dtype=np.float32)[:, None]
    thr = (noise * 0.55 + grad * 0.45)
    m = np.clip((p * 1.6 - thr) * 3.2, 0, 1)
    mask = Image.fromarray(np.uint8(m * 255)).resize((w, h), Image.BILINEAR)
    mask = mask.filter(ImageFilter.GaussianBlur(14))
    # el sujeto llega desenfocado y se enfoca al materializarse
    blur_px = (1 - util.ease_out_cubic(p)) * 26
    sharp = frame_img.filter(ImageFilter.GaussianBlur(blur_px)) if blur_px > 0.6 else frame_img
    sharp = rgb_split(sharp, (1 - p) * 9)
    out = Image.composite(sharp, plate_img, mask)
    return flash(out, (1 - p) ** 2 * 0.35 * (p > 0.05))
