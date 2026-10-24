"""Clone-split composite: the real center person stays untouched; the two AI clones slide out of him
right as he starts turning his head, with two quick blinks."""
import os
import sys
import cv2
import numpy as np

F_BLINK1, F_BLINK2 = 36, 40          # 1-based frame numbers (30 fps)
BLUE = np.array([150, 52, 28], np.float32)  # BGR, deep 'blue rose' blue
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
    _, ma = cv2.threshold(ma, 100, 255, cv2.THRESH_BINARY)
    _, mo_b = cv2.threshold(mo, 60, 255, cv2.THRESH_BINARY)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    # pixels that differ from the real empty sofa catch limbs the segmenter missed
    diff = cv2.GaussianBlur(np.abs(a - o).max(2), (0, 0), 2)
    changed = ((diff > 30) * 255).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, k(3))
    fg = cv2.bitwise_or(ma, changed)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k(9))
    # watershed: split the three people along real image edges instead of cutting a fixed band
    markers = np.zeros(ma.shape, np.int32)
    markers[cv2.dilate(fg, k(18)) == 0] = 1                     # background
    markers[cv2.erode(mo_b, k(12)) > 0] = 2                     # center (real position)
    far = cv2.dilate(mo_b, k(70)) == 0
    seed = (cv2.erode(ma, k(14)) > 0) & far
    xs = np.arange(ma.shape[1])[None, :]
    markers[seed & (xs < 540)] = 3                              # left clone
    markers[seed & (xs >= 540)] = 4                             # right clone
    # dark jeans on the dark sofa fool both cues at the frame edges; the clones own those strips
    markers[950:1750, :70] = 3
    markers[950:1750, -70:] = 4
    lab = cv2.watershed(np.clip(a, 0, 255).astype(np.uint8), markers)
    left = ((lab == 3) * 255).astype(np.uint8)
    right = ((lab == 4) * 255).astype(np.uint8)
    out = []
    for m in (left, right):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k(2))
        n, cc, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
            m = ((cc == big) * 255).astype(np.uint8)
        out.append(m)
    return out[0], out[1], mo, ma


LIT = np.array([235, 72, 22], np.float32)  # BGR, luminous version of the blue-rose blue for glows

# clone glow schedule: (tint, glow). Strong while they travel, then they blink blue twice and settle.
CLONE_FX = {51: (.25, .45), 52: (.65, 1.0), 53: (.7, 1.0), 54: (.15, .3), 55: (.1, .2),
            56: (.6, .95), 57: (.6, .9), 58: (.3, .5), 59: (.15, .25), 60: (.06, .1)}
# the real person "charges" blue right before splitting
CENTER_GLOW = {34: .3, 35: .7, 36: 1.0, 37: .6, 38: .3}


def tint(rgb, t):
    """Recolor toward the blue-rose blue, keeping luminance so the person stays readable."""
    lum = rgb.mean(2, keepdims=True)
    blue = lum / 255 * LIT * 1.15 + BLUE * 0.2
    return rgb * (1 - t) + blue * t


def screen(frame, mask, strength):
    g = mask[..., None] * strength * LIT / 255
    return 255 - (255 - frame) * (1 - np.clip(g, 0, 1))


for i in range(1, LAST + 1):
    if only and i != only:
        continue
    o = cv2.imread(f'fx/orig/{i:03d}.png').astype(np.float32)
    frame = o.copy()
    clone_alpha = None
    if i >= F_START:
        a = cv2.imread(f'fx/ai/{i:03d}.png').astype(np.float32)
        # match the AI render's grade to the real footage using the static wall above the sofa
        ref_o, ref_a = o[:650].reshape(-1, 3), a[:650].reshape(-1, 3)
        a = (a - ref_a.mean(0)) / (ref_a.std(0) + 1e-6) * ref_o.std(0) + ref_o.mean(0)
        left, right, mo, ma = clone_masks(i, o, a)
        p = (i - F_START + 1) / (F_END - F_START + 1)
        t_amt, g_amt = (0.55 * (1 - 0.5 * p), 1.0) if p < 1 else CLONE_FX.get(i, (0, 0))
        center_x = 540
        # glow only where the AI frame really differs from the empty real sofa (i.e. a person)
        diff = cv2.GaussianBlur(np.abs(a - o).max(2), (0, 0), 3)
        person = cv2.morphologyEx(((diff > 30) * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        person = cv2.dilate(person, np.ones((11, 11), np.uint8))
        layers = []
        for m in (left, right):
            if p < 1:
                m = cv2.bitwise_and(m, ma)  # while moving, carry only the person, not sofa
            if m.max() == 0:
                continue
            xs = np.where(m.max(0) > 0)[0]
            cx = (xs.min() + xs.max()) / 2
            alpha = soft(m, 2, 2.5)
            body = soft(cv2.morphologyEx(cv2.bitwise_and(m, ma), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))), 2, 4)  # glow hugs the person, not the sofa
            layers.append((alpha, body, center_x - cx))
        clone_alpha = np.zeros((1920, 1080), np.float32)
        # ghost trails first (behind), then the clone itself
        trails = [(0.30, 0.4), (0.16, 0.25)] if p < 1 else []
        for lag, strength in trails + [(0.0, 1.0)]:
            pp = ease(p - lag)
            if p - lag <= 0:
                continue
            vis = np.clip((p - lag) / 0.35, 0, 1) * strength
            for alpha, body, dx in layers:
                shift = (1 - pp) * dx
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                al = cv2.warpAffine(alpha, M, (1080, 1920)) * vis
                rgb = cv2.warpAffine(a, M, (1080, 1920))
                rgb = tint(rgb, 0.9 if lag > 0 else t_amt)
                frame = frame * (1 - al[..., None]) + rgb * al[..., None]
                if lag == 0:
                    clone_alpha = np.maximum(clone_alpha, cv2.warpAffine(body, M, (1080, 1920)))
        # real center person always on top
        ca = soft(cv2.threshold(mo, 60, 255, cv2.THRESH_BINARY)[1], 4, 3)
        frame = frame * (1 - ca[..., None]) + o * ca[..., None]
        if g_amt > 0:
            # rim light: strongest on the silhouette edge, soft outer bloom, the body itself stays readable
            edge = np.clip(cv2.GaussianBlur(clone_alpha, (0, 0), 5) * 1.0 - cv2.erode(clone_alpha, np.ones((15, 15), np.uint8)) * 0.85, 0, 1)
            bloom = cv2.GaussianBlur(clone_alpha, (0, 0), 26) * 0.9
            halo = np.clip(edge * 1.3 + bloom * (1 - clone_alpha * 0.7), 0, 1)
            frame = screen(frame, halo * (1 - 0.85 * ca), g_amt)
    cg = CENTER_GLOW.get(i, 0)
    if cg:
        mo = cv2.imread(f'fx/mo/{i:03d}.png', 0)
        m = cv2.GaussianBlur(mo.astype(np.float32) / 255, (0, 0), 18) * 1.4
        rim = np.clip(m - 0.85 * mo.astype(np.float32) / 255, 0, 1)
        frame = screen(frame, rim, cg * 0.9)
        frame = screen(frame, np.full((1920, 1080), 0.07, np.float32), cg)  # faint room-wide blue pulse
    cv2.imwrite(f'{OUT}/{i:03d}.png', np.clip(frame, 0, 255).astype(np.uint8))
print('done')
