"""SFX for the screens block (30 fps timeline, 6.97 s)."""
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile
SR, DUR = 44100, 209 / 30
N = int(SR * DUR); rng = np.random.default_rng(5)
L, R = np.zeros(N), np.zeros(N)
fr = lambda f: f / 30
def env(n, a, d):
    t = np.arange(n) / SR; return np.minimum(1, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / d)
def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], 'bandpass', fs=SR, output='sos'), x)
def add(sig, t0, pan=0.0, gain=1.0):
    i = int(t0 * SR); sig = sig[:max(0, N - i)]
    L[i:i + len(sig)] += sig * np.sqrt((1 - pan) / 2) * gain; R[i:i + len(sig)] += sig * np.sqrt((1 + pan) / 2) * gain
def rise(t0, pan, gain=0.25, dur=0.22):
    n = int(dur * SR); x = rng.standard_normal(n)
    y = np.concatenate([bp(x[k:k + 512], 600 + 5000 * k / n, 1500 + 9000 * k / n) for k in range(0, n, 512)])[:n]
    add(y * np.linspace(0, 1, n) ** 1.5, t0 - dur, pan, gain)
def pop(t0, pan, gain=0.5, f=900):
    n = int(0.25 * SR); t = np.arange(n) / SR
    body = np.sin(2 * np.pi * (f * (1 + 1.2 * np.exp(-t * 40))) * t) * env(n, 0.002, 0.05)
    shim = sum(np.sin(2 * np.pi * h * t) for h in (2637, 3520, 4699)) * env(n, 0.005, 0.12) * 0.25
    add(body + shim, t0, pan, gain)
def tick(t0, pan, gain):
    n = int(0.03 * SR); t = np.arange(n) / SR
    add(np.sin(2 * np.pi * 2200 * t) * env(n, 0.001, 0.006), t0, pan, gain)
def swipe(t0, pan, gain=0.28):
    n = int(0.25 * SR); x = rng.standard_normal(n)
    y = np.concatenate([bp(x[k:k + 512], 2500 - 1800 * k / n, 6000 - 3500 * k / n) for k in range(0, n, 512)])[:n]
    add(y * np.sin(np.pi * np.linspace(0, 1, n)), t0, pan, gain)
# screens open: left (f30), right (f35), center (f40); pop when the panel snaps open (+5)
for f, pan in ((30, -0.6), (35, 0.6), (40, 0.0)):
    rise(fr(f + 5), pan); pop(fr(f + 6), pan, 0.45, 800 + 150 * (f - 30) / 5)
# follower counter ticking on the left screen, accelerating then settling
t = fr(44)
while t < fr(150):
    p = (t - fr(44)) / (fr(150) - fr(44)); tick(t, -0.6, 0.10 * (1 - 0.6 * p)); t += 0.045 + 0.12 * p ** 2
# reel swipe on the right screen
swipe(fr(112), 0.6)
# center screen collapses
n = int(0.25 * SR); tt = np.arange(n) / SR
add(np.sin(2 * np.pi * (1400 * np.exp(-tt * 9) + 200) * tt) * env(n, 0.003, 0.08), fr(128), 0.0, 0.35)
out = np.stack([L, R], 1); out /= max(1e-9, np.abs(out).max()); out *= 0.5
wavfile.write('b2/sfx2.wav', SR, (out * 32767).astype(np.int16)); print('sfx ok')
