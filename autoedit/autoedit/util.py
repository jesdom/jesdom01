"""Utilidades: easing, color, cache de numpy."""
import math
import numpy as np


# ---------------------------------------------------------------- easing ----
def clamp(v, a=0.0, b=1.0):
    return a if v < a else (b if v > b else v)


def ease_out_cubic(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp(t)
    return 3 * t * t - 2 * t * t * t


def ease_out_back(t, s=1.70158):
    t = clamp(t)
    t -= 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_expo(t):
    t = clamp(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def spring(t, freq=3.0, damp=6.0):
    """0 -> 1 con rebote amortiguado."""
    t = clamp(t)
    return 1 - math.exp(-damp * t) * math.cos(freq * math.pi * t)


def smoothstep(edge0, edge1, x):
    if edge1 == edge0:
        return 0.0
    t = clamp((x - edge0) / (edge1 - edge0))
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------- color ----
def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


# ----------------------------------------------------------------- ruido ----
def value_noise_1d(n, seed=0, octaves=3):
    """Ruido suave 1D en [-1, 1] para movimiento tipo handheld."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)
    amp = 1.0
    for o in range(octaves):
        pts = max(2, int(n / (2 ** (5 - o))) + 2)
        base = rng.standard_normal(pts).astype(np.float32)
        x = np.linspace(0, pts - 1, n, dtype=np.float32)
        out += amp * np.interp(x, np.arange(pts), base).astype(np.float32)
        amp *= 0.5
    m = np.abs(out).max() or 1.0
    return out / m
