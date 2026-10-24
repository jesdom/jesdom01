"""Build a .cube LUT that moves the clone video toward the next scene's look while keeping skin alive:
cooler/muted surroundings (hot oranges tamed, deep shadows teal, whites slightly cool), but skin keeps
its warmth and midtone contrast. usage: python3 fx/grade.py <out.cube> [strength]"""
import sys
import cv2
import numpy as np

OUT = sys.argv[1]
S = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
N = 33

g = np.linspace(0, 1, N, dtype=np.float32)
# .cube order: R fastest, then G, then B
b, gg, r = np.meshgrid(g, g, g, indexing='ij')
rgb = np.stack([r, gg, b], -1).reshape(-1, 1, 3)
lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)  # float: L 0..100, a/b signed
L, A, B = lab[..., 0] / 100, lab[..., 1], lab[..., 2]
C = np.sqrt(A ** 2 + B ** 2)
hue = np.degrees(np.arctan2(B, A))

# skin: warm hue, moderate chroma, mid luminance
skin = (np.exp(-((hue - 52) / 16) ** 2) * np.exp(-((C - 30) / 16) ** 2) * np.exp(-((L - 0.6) / 0.25) ** 2))

# tone: a bit darker overall, gentle S-curve so faces keep depth, highlights roll off, true blacks
Lt = L ** (1 + 0.28 * S)
Lt = Lt + 0.12 * S * np.sin(2 * np.pi * Lt) * (Lt * (1 - Lt)) * 2   # midtone contrast
Lt = Lt * (1 - 0.08 * S * Lt ** 4)                                  # soft whites
Lt = np.clip(Lt, 0, 1)

# chroma: surroundings muted (oranges much more), skin only lightly
orange = np.exp(-((hue - 60) / 28) ** 2) * np.clip((C - 38) / 20, 0, 1)
k_env = 1 - S * (0.22 + 0.33 * orange)
k_skin = 1 - S * 0.05
k = k_env * (1 - skin) + k_skin * skin
A, B = A * k, B * k
# a touch of rose in the skin so it doesn't go grey-orange
A = A + 1.5 * S * skin

# cool only the deep shadows and the brightest whites, never the skin
deep = np.clip(1 - Lt / 0.45, 0, 1) ** 2 * (1 - skin)
high = np.clip((Lt - 0.8) / 0.2, 0, 1) * (1 - skin)
B = B - 8 * S * deep - 4 * S * high
A = A - 1.0 * S * deep

out = cv2.cvtColor(np.concatenate([Lt[..., None] * 100, A[..., None], B[..., None]], -1).astype(np.float32), cv2.COLOR_LAB2RGB)
out = np.clip(out.reshape(-1, 3), 0, 1)
with open(OUT, 'w') as f:
    f.write('TITLE "clones_to_ref"\nLUT_3D_SIZE %d\n' % N)
    for v in out:
        f.write('%.6f %.6f %.6f\n' % tuple(v))
print('lut', OUT)
