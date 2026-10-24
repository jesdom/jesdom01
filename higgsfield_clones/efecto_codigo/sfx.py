"""Clean clone-split SFX synthesized from scratch, stereo 44.1k, 3.9 s, aligned to the video effect."""
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

SR = 44100
DUR = 3.9
N = int(SR * DUR)
rng = np.random.default_rng(7)
L = np.zeros(N)
R = np.zeros(N)


def env(n, a, d):
    """Attack/decay envelope (seconds) of length n samples."""
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / d)
    return e


def add(sig, t0, pan=0.0, gain=1.0):
    i = int(t0 * SR)
    sig = sig[: max(0, N - i)]
    gl = np.sqrt((1 - pan) / 2) * gain
    gr = np.sqrt((1 + pan) / 2) * gain
    L[i:i + len(sig)] += sig * gl
    R[i:i + len(sig)] += sig * gr


def bp(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], 'bandpass', fs=SR, output='sos'), x)


T_FLASH1 = 36 / 30   # first blink frame
T_FLASH2 = 40 / 30   # second blink frame
T_SPLIT = 37 / 30    # clones start leaving the center

# 1) short airy riser into the effect
n = int(0.28 * SR)
noise = rng.standard_normal(n)
riser = np.concatenate([bp(noise[k:k + 1024], 800 + 6000 * k / n, 1600 + 9000 * k / n) for k in range(0, n, 1024)])[:n]
riser *= np.linspace(0, 1, n) ** 2
add(riser, T_FLASH1 - 0.28, gain=0.18)

# 2) two blink "ticks" (soft camera-shutter / glass ping)
for t0 in (T_FLASH1, T_FLASH2):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    click = bp(rng.standard_normal(n), 3000, 9000) * env(n, 0.001, 0.012)
    ping = np.sin(2 * np.pi * 2600 * t) * env(n, 0.001, 0.04) + 0.5 * np.sin(2 * np.pi * 3900 * t) * env(n, 0.001, 0.03)
    add(click * 0.6 + ping * 0.35, t0, gain=0.45)

# 3) soft low "bwoom" when the duplicates appear
n = int(0.7 * SR)
t = np.arange(n) / SR
f = 85 * np.exp(-t * 3) + 38
boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.006, 0.22)
add(boom, T_SPLIT, gain=0.55)

# 4) whoosh splitting center -> left and center -> right
n = int(0.55 * SR)
for side in (-1, 1):
    noise = rng.standard_normal(n)
    chunks = []
    for k in range(0, n, 512):
        p = k / n
        c = 900 + 2600 * np.sin(np.pi * p)
        chunks.append(bp(noise[k:k + 512], c * 0.6, c * 1.4))
    w = np.concatenate(chunks)[:n] * np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    pan_curve = side * np.linspace(0, 0.9, n)
    i = int(T_SPLIT * SR)
    m = min(n, N - i)
    L[i:i + m] += w[:m] * np.sqrt((1 - pan_curve[:m]) / 2) * 0.32
    R[i:i + m] += w[:m] * np.sqrt((1 + pan_curve[:m]) / 2) * 0.32

# 5) light shimmer tail
n = int(0.9 * SR)
t = np.arange(n) / SR
sh = sum(np.sin(2 * np.pi * fr * t + ph) for fr, ph in [(3520, 0), (4186, 1), (5274, 2), (6272, 0.5)])
sh *= (0.6 + 0.4 * np.sin(2 * np.pi * 14 * t)) * env(n, 0.03, 0.25)
add(sh, T_SPLIT + 0.05, pan=-0.3, gain=0.035)
add(sh[::-1][::-1] * 0.9, T_SPLIT + 0.08, pan=0.3, gain=0.035)

out = np.stack([L, R], 1)
out /= max(1e-9, np.abs(out).max())
out *= 0.5  # headroom; final level set in the mix
wavfile.write('fx/sfx.wav', SR, (out * 32767).astype(np.int16))
print('sfx ok')
