"""Luminance-only LUT: lower exposure and soften whites, colors untouched."""
import sys, cv2, numpy as np
N = 33
g = np.linspace(0, 1, N, dtype=np.float32)
b, gg, r = np.meshgrid(g, g, g, indexing='ij')
lab = cv2.cvtColor(np.stack([r, gg, b], -1).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
L = lab[..., 0] / 100
Lt = L ** 1.3
Lt = Lt * (1 - 0.06 * Lt ** 4)
lab[..., 0] = Lt * 100
out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).reshape(-1, 3), 0, 1)
with open(sys.argv[1], 'w') as f:
    f.write('LUT_3D_SIZE %d\n' % N)
    for v in out:
        f.write('%.6f %.6f %.6f\n' % tuple(v))
