"""Run the watershed split once per frame and cache left/right clone masks."""
import sys, cv2, numpy as np, importlib.util
sys.argv = ['x', '999']
spec = importlib.util.spec_from_file_location('c', 'fx/comp.py'); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
for i in range(37, 118):
    o = cv2.imread(f'fx/orig/{i:03d}.png').astype(np.float32); a = cv2.imread(f'fx/ai/{i:03d}.png').astype(np.float32)
    ro, ra = o[:650].reshape(-1, 3), a[:650].reshape(-1, 3); a = (a - ra.mean(0)) / (ra.std(0) + 1e-6) * ro.std(0) + ro.mean(0)
    l, r, mo, ma = c.clone_masks(i, o, a)
    cv2.imwrite(f'fx/cm/l{i:03d}.png', l); cv2.imwrite(f'fx/cm/r{i:03d}.png', r); cv2.imwrite(f'fx/cm/ma{i:03d}.png', ma)
print('cached')
