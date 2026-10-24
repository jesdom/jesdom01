"""SFX per reveal style, synthesized and aligned to the frame timings in styles.py (30 fps).
usage: python3 fx/sfx_styles.py <parpadeo|aura|glitch>  -> fx/sfx_<style>.wav"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

STYLE = sys.argv[1]
SR, DUR = 44100, 3.9
N = int(SR * DUR)
rng = np.random.default_rng(11)
L, R = np.zeros(N), np.zeros(N)
fr = lambda f: f / 30


def env(n, a, d):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / d)


def bp(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], 'bandpass', fs=SR, output='sos'), x)


def add(sig, t0, pan=0.0, gain=1.0):
    i = int(t0 * SR)
    sig = sig[: max(0, N - i)]
    L[i:i + len(sig)] += sig * np.sqrt((1 - pan) / 2) * gain
    R[i:i + len(sig)] += sig * np.sqrt((1 + pan) / 2) * gain


def riser(t_end, dur=0.3, gain=0.2):
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    x = np.concatenate([bp(noise[k:k + 1024], 800 + 6000 * k / n, 1600 + 9000 * k / n) for k in range(0, n, 1024)])[:n]
    add(x * np.linspace(0, 1, n) ** 2, t_end - dur, gain=gain)


def boom(t0, gain=0.6, f0=90, f1=36, dec=0.25):
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    f = (f0 - f1) * np.exp(-t * 3) + f1
    add(np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.005, dec), t0, gain=gain)


def tick(t0, gain=0.45, pitch=2600):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    click = bp(rng.standard_normal(n), 3000, 9000) * env(n, 0.001, 0.012)
    ping = np.sin(2 * np.pi * pitch * t) * env(n, 0.001, 0.04) + 0.5 * np.sin(2 * np.pi * pitch * 1.5 * t) * env(n, 0.001, 0.03)
    add(click * 0.6 + ping * 0.35, t0, gain=gain)


def whoosh_split(t0, dur=0.55, gain=0.32):
    n = int(dur * SR)
    for side in (-1, 1):
        noise = rng.standard_normal(n)
        w = np.concatenate([bp(noise[k:k + 512], (900 + 2600 * np.sin(np.pi * k / n)) * 0.6,
                                 (900 + 2600 * np.sin(np.pi * k / n)) * 1.4) for k in range(0, n, 512)])[:n]
        w *= np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
        pan = side * np.linspace(0, 0.9, n)
        i = int(t0 * SR); m = min(n, N - i)
        L[i:i + m] += w[:m] * np.sqrt((1 - pan[:m]) / 2) * gain
        R[i:i + m] += w[:m] * np.sqrt((1 + pan[:m]) / 2) * gain


def shimmer(t0, gain=0.035):
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    sh = sum(np.sin(2 * np.pi * f * t + ph) for f, ph in [(3520, 0), (4186, 1), (5274, 2), (6272, .5)])
    sh *= (0.6 + 0.4 * np.sin(2 * np.pi * 14 * t)) * env(n, 0.03, 0.25)
    add(sh, t0, -0.3, gain); add(sh, t0 + 0.03, 0.3, gain)


def blink_hit(t0, gain=0.5):
    """Dark 'thump-shutter' for the deep-blue blinks."""
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * 140 * t * (1 - 0.4 * t)) * env(n, 0.002, 0.05)
    air = bp(rng.standard_normal(n), 1500, 7000) * env(n, 0.001, 0.03)
    add(body * 0.8 + air * 0.5, t0, gain=gain)


def glitch_burst(t0, dur=0.12, gain=0.3, pan=0.0):
    """Bit-crushed, chopped digital stutter."""
    n = int(dur * SR)
    x = rng.standard_normal(n)
    x = np.round(x * 3) / 3                                   # bit crush
    hold = 12
    x = np.repeat(x[::hold], hold)[:n]                        # sample-rate reduction
    gate = (np.floor(np.arange(n) / (SR * 0.012)) % 2 == 0)   # 12 ms chop
    sq = np.sign(np.sin(2 * np.pi * rng.choice([440, 660, 880, 1320]) * np.arange(n) / SR)) * 0.4
    add((x * 0.6 + sq) * gate * env(n, 0.001, dur / 2), t0, pan, gain)


if STYLE == 'parpadeo':
    riser(fr(36), 0.3, 0.2)
    blink_hit(fr(36), 0.55)
    blink_hit(fr(40), 0.65)
    boom(fr(40), 0.7, 100, 34, 0.3)
    whoosh_split(fr(40), 0.4, 0.22)
    shimmer(fr(41), 0.04)
elif STYLE == 'aura':
    riser(fr(36), 0.28, 0.18)
    for f in (36, 52, 56):
        tick(fr(f), 0.45)
    boom(fr(37), 0.55)
    whoosh_split(fr(37), 0.55, 0.32)
    shimmer(fr(38), 0.035)
elif STYLE == 'glitch':
    riser(fr(35), 0.25, 0.15)
    glitch_burst(fr(35), 0.18, 0.35)
    boom(fr(36), 0.65, 110, 32, 0.28)
    for f, p in [(38, -0.5), (40, 0.5), (42, -0.3), (45, 0.3)]:
        glitch_burst(fr(f), 0.07, 0.25, p)
    whoosh_split(fr(37), 0.45, 0.2)
    tick(fr(49), 0.3, 3200)

out = np.stack([L, R], 1)
out /= max(1e-9, np.abs(out).max())
out *= 0.5
wavfile.write(f'fx/sfx_{STYLE}.wav', SR, (out * 32767).astype(np.int16))
print('sfx', STYLE)
