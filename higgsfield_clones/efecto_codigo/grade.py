"""Build a .cube LUT that moves the clone video toward the reference grade:
lower exposure, soft highlight roll-off, ~25% less chroma (extra on hot oranges), blue/teal shadows.
usage: python3 fx/grade.py <out.cube> [strength]"""
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

# tone: pull mids down (skin 151 -> ~128 in 8-bit L), roll off highlights, keep a soft black floor
Lt = L ** (1 + 0.5 * S)
Lt = Lt * (1 - 0.10 * S * Lt ** 3)            # highlight roll-off: whites stop short of clipping
Lt = Lt + 0.025 * S * (1 - Lt) ** 4           # slight lift of the blacks (log-ish feel)

# chroma: global desaturation, extra on hot oranges/reds (floor, warm skin)
C = np.sqrt(A ** 2 + B ** 2)
hue = np.degrees(np.arctan2(B, A))
orange = np.exp(-((hue - 60) / 30) ** 2) * np.clip((C - 35) / 25, 0, 1)
k = 1 - S * (0.2 + 0.3 * orange)
A, B = A * k, B * k

# shadows to blue-teal, like the reference's walls and shirt
shadow = (1 - Lt) ** 2.2
B = B - 9 * S * shadow
A = A - 1.5 * S * shadow

out = cv2.cvtColor(np.concatenate([Lt[..., None] * 100, A[..., None], B[..., None]], -1).astype(np.float32), cv2.COLOR_LAB2RGB)
out = np.clip(out.reshape(-1, 3), 0, 1)
with open(OUT, 'w') as f:
    f.write('TITLE "clones_to_ref"\nLUT_3D_SIZE %d\n' % N)
    for v in out:
        f.write('%.6f %.6f %.6f\n' % tuple(v))
print('lut', OUT)
