"""Clone-split composite: the real center person stays untouched; the two AI clones slide out of him
right as he starts turning his head, with two quick blinks."""
import os
import sys
import cv2
import numpy as np

F_BLINK1, F_BLINK2 = 36, 40          # 1-based frame numbers (30 fps)
F_START, F_END = 37, 50              # clones travel from the center to their seat
LAST = 117
OUT = 'fx/comp'
os.makedirs(OUT, exist_ok=True)
only = int(sys.argv[1]) if len(sys.argv) > 1 else None


def ease(p):
    p = np.clip(p, 0, 1)
    return 1 - (1 - p) ** 3


def soft(mask, grow, blur):
    m = mask
    if grow > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    elif grow < 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-2 * grow + 1, -2 * grow + 1)))
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), blur)


def clone_masks(i, o, a):
    ma = cv2.imread(f'fx/ma/{i:03d}.png', 0)
    mo = cv2.imread(f'fx/mo/{i:03d}.png', 0)
    _, ma = cv2.threshold(ma, 110, 255, cv2.THRESH_BINARY)
    _, mo_b = cv2.threshold(mo, 60, 255, cv2.THRESH_BINARY)
    center = cv2.dilate(mo_b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71)))
    rest = cv2.bitwise_and(ma, cv2.bitwise_not(center))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(rest)
    left = np.zeros_like(rest)
    right = np.zeros_like(rest)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 15000:
            continue  # stray bits of the AI's own center person
        (left if cent[k][0] < 540 else right)[lab == k] = 255
    return left, right, mo


cache = {}
for i in range(1, LAST + 1):
    if only and i != only:
        continue
    o = cv2.imread(f'fx/orig/{i:03d}.png').astype(np.float32)
    frame = o.copy()
    if i >= F_START:
        a = cv2.imread(f'fx/ai/{i:03d}.png').astype(np.float32)
        # match the AI render's grade to the real footage using the static wall above the sofa
        ref_o, ref_a = o[:650].reshape(-1, 3), a[:650].reshape(-1, 3)
        a = (a - ref_a.mean(0)) / (ref_a.std(0) + 1e-6) * ref_o.std(0) + ref_o.mean(0)
        left, right, mo = clone_masks(i, o, a)
        p = (i - F_START + 1) / (F_END - F_START + 1)
        center_x = 540
        layers = []
        for m in (left, right):
            if m.max() == 0:
                continue
            xs = np.where(m.max(0) > 0)[0]
            cx = (xs.min() + xs.max()) / 2
            alpha = soft(m, 2, 2.5)
            layers.append((alpha, center_x - cx))
        # ghost trails first (behind), then the clone itself
        trails = [(0.30, 0.35), (0.16, 0.18)] if p < 1 else []
        for lag, strength in trails + [(0.0, 1.0)]:
            pp = ease(p - lag)
            if p - lag <= 0:
                continue
            vis = np.clip((p - lag) / 0.35, 0, 1) * strength
            for alpha, dx in layers:
                shift = (1 - pp) * dx
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                al = cv2.warpAffine(alpha, M, (1080, 1920)) * vis
                rgb = cv2.warpAffine(a, M, (1080, 1920))
                frame = frame * (1 - al[..., None]) + rgb * al[..., None]
        # real center person always on top
        ca = soft(cv2.threshold(mo, 60, 255, cv2.THRESH_BINARY)[1], 4, 3)
        frame = frame * (1 - ca[..., None]) + o * ca[..., None]
    # two quick clean blinks
    flash = {F_BLINK1: 0.55, F_BLINK1 + 1: 0.2, F_BLINK2: 0.45, F_BLINK2 + 1: 0.15}.get(i, 0)
    if flash:
        frame = frame * (1 - flash) + 255 * flash
    cv2.imwrite(f'{OUT}/{i:03d}.png', np.clip(frame, 0, 255).astype(np.uint8))
print('done')
