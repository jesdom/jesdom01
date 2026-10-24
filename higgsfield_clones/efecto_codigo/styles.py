"""Three blue-rose clone-reveal styles over the real take.

usage: python3 fx/styles.py <parpadeo|aura|glitch> [frame]
The real person is always the original footage. Clones come from the AI render (color-matched).
Once the reveal settles, the AI frame is the base and only the real person's silhouette is laid on top,
so the clones keep their full outline (no straight cuts between them and the center).
"""
import os
import sys
import cv2
import numpy as np

STYLE = sys.argv[1]
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT = f'fx/out_{STYLE}'
os.makedirs(OUT, exist_ok=True)
W, H = 1080, 1920
BLUE_DEEP = np.array([110, 38, 18], np.float32)   # BGR, dark blue-rose
BLUE_LIT = np.array([235, 72, 22], np.float32)    # BGR, luminous blue-rose for glows
rng = np.random.default_rng(3)
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def load(i):
    o = cv2.imread(f'fx/orig/{i:03d}.png').astype(np.float32)
    a = cv2.imread(f'fx/ai/{i:03d}.png').astype(np.float32)
    ro, ra = o[:650].reshape(-1, 3), a[:650].reshape(-1, 3)
    a = (a - ra.mean(0)) / (ra.std(0) + 1e-6) * ro.std(0) + ro.mean(0)
    mo = cv2.imread(f'fx/mo/{i:03d}.png', 0)
    return o, a, mo


def clone_mask(i, tight):
    """Union of both clones; tight=True drops sofa so a moving clone carries only the person."""
    l, r = cv2.imread(f'fx/cm/l{i:03d}.png', 0), cv2.imread(f'fx/cm/r{i:03d}.png', 0)
    if tight:
        ma = cv2.imread(f'fx/cm/ma{i:03d}.png', 0)
        l, r = cv2.bitwise_and(l, ma), cv2.bitwise_and(r, ma)
    return l, r


def soft(m, grow=0, blur=2.5):
    if grow:
        m = cv2.dilate(m, K(grow))
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), blur)


def person_alpha(mo):
    return soft(((mo > 60) * 255).astype(np.uint8), 6, 3)


def body_for_glow(m):
    return soft(cv2.morphologyEx(m, cv2.MORPH_OPEN, K(20)), 0, 4)


def over(dst, src, al):
    return dst * (1 - al[..., None]) + src * al[..., None]


def screen(frame, mask, strength, color=BLUE_LIT):
    g = np.clip(mask[..., None] * strength * color / 255, 0, 1)
    return 255 - (255 - frame) * (1 - g)


def tint(rgb, t):
    lum = rgb.mean(2, keepdims=True)
    blue = lum / 255 * BLUE_LIT * 1.15 + BLUE_DEEP * 0.3
    return rgb * (1 - t) + blue * t


def rim(alpha, strength):
    edge = np.clip(cv2.GaussianBlur(alpha, (0, 0), 5) - cv2.erode(alpha, np.ones((15, 15), np.uint8)) * 0.85, 0, 1)
    bloom = cv2.GaussianBlur(alpha, (0, 0), 26) * 0.9
    return np.clip(edge * 1.3 + bloom * (1 - alpha * 0.7), 0, 1) * strength


def shift(img, dx, dy=0):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_CONSTANT)


def punch(frame, s, dx=0, dy=0):
    """Zoom punch / shake around the frame center."""
    if s == 1 and dx == 0 and dy == 0:
        return frame
    M = cv2.getRotationMatrix2D((W / 2, H * 0.55), 0, s)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def settled(o, a, mo, i):
    """Final look: AI frame as base, real person on top."""
    return over(a, o, person_alpha(mo))


def ease(p):
    p = np.clip(p, 0, 1)
    return 1 - (1 - p) ** 3


# ---------------------------------------------------------------- styles
def parpadeo(i):
    """Two deep-blue blinks. Clones pop in on the second blink with a zoom punch, then a blue cool-down."""
    o, a, mo = load(i) if i >= 30 else (cv2.imread(f'fx/orig/{i:03d}.png').astype(np.float32), None, None)
    if i < 40:
        f = o
    else:
        f = settled(o, a, mo, i)
        l, r = clone_mask(i, True)
        cl = body_for_glow(cv2.bitwise_or(l, r))
        t = {40: 0.9, 41: 0.7, 42: 0.5, 43: 0.35, 44: 0.2, 45: 0.1}.get(i, 0)
        if t:
            f = over(f, tint(f, 1.0), cl * t)
            f = screen(f, rim(cl, 1.0), t * 1.2)
    # the blinks: room drops to deep blue with a lit core on the person
    blink = {36: 0.85, 37: 0.4, 40: 0.95, 41: 0.45}.get(i, 0)
    if blink:
        lum = f.mean(2, keepdims=True) / 255
        blue = BLUE_DEEP * 0.6 + lum * BLUE_LIT * 0.9
        f = f * (1 - blink) + blue * blink
    s = {40: 1.07, 41: 1.05, 42: 1.03, 43: 1.015}.get(i, 1)
    return punch(f, s)


def aura(i):
    """Clones slide out of him leaving blue afterimages, rim-lit, then blink blue twice."""
    F0, F1 = 37, 50
    if i < 34:
        return cv2.imread(f'fx/orig/{i:03d}.png').astype(np.float32)
    o, a, mo = load(i)
    pa = person_alpha(mo)
    if i < F0:
        f = o
        clone_al = None
    elif i <= F1:
        p = (i - F0 + 1) / (F1 - F0 + 1)
        l, r = clone_mask(i, True)
        f = o.copy()
        clone_al = np.zeros((H, W), np.float32)
        for m in (l, r):
            if m.max() == 0:
                continue
            xs = np.where(m.max(0) > 0)[0]
            dx = W / 2 - (xs.min() + xs.max()) / 2
            al0, body = soft(m), body_for_glow(m)
            for lag, strength in [(0.30, 0.4), (0.16, 0.25), (0.0, 1.0)]:
                if p - lag <= 0:
                    continue
                sh = (1 - ease(p - lag)) * dx
                vis = np.clip((p - lag) / 0.35, 0, 1) * strength
                rgb = tint(shift(a, sh), 0.9 if lag else 0.55 * (1 - 0.5 * p))
                f = over(f, rgb, shift(al0, sh) * vis)
                if not lag:
                    clone_al = np.maximum(clone_al, shift(body, sh))
        f = over(f, o, pa)
        if i == F1:
            f = settled(o, a, mo, i)
    else:
        f = settled(o, a, mo, i)
        l, r = clone_mask(i, True)
        clone_al = body_for_glow(cv2.bitwise_or(l, r))
    sched = {51: (.25, .45), 52: (.65, 1.0), 53: (.7, 1.0), 54: (.15, .3), 55: (.1, .2),
             56: (.6, .95), 57: (.6, .9), 58: (.3, .5), 59: (.15, .25), 60: (.06, .1)}
    t_amt, g_amt = ((0, 1.0) if F0 <= i <= F1 else sched.get(i, (0, 0)))
    if clone_al is not None and (t_amt or g_amt):
        if t_amt:
            f = over(f, tint(f, 1.0), clone_al * t_amt)
        f = screen(f, rim(clone_al, 1.0) * (1 - 0.85 * pa), g_amt)
    cg = {34: .3, 35: .7, 36: 1.0, 37: .6, 38: .3}.get(i, 0)
    if cg:
        m = mo.astype(np.float32) / 255
        f = screen(f, np.clip(cv2.GaussianBlur(m, (0, 0), 18) * 1.4 - 0.85 * m, 0, 1), cg * 0.9)
    s = {37: 1.04, 38: 1.03, 39: 1.02, 40: 1.01, 52: 1.025, 53: 1.015, 56: 1.025, 57: 1.015}.get(i, 1)
    return punch(f, s)


def glitch(i):
    """Hologram: clones flicker in as blue scan-lined holograms with slice displacement and a camera shake."""
    if i < 35:
        return cv2.imread(f'fx/orig/{i:03d}.png').astype(np.float32)
    o, a, mo = load(i)
    pa = person_alpha(mo)
    # on/off flicker pattern for the clones (1 = visible)
    vis = {37: .0, 38: .8, 39: .0, 40: 1, 41: .3, 42: 1, 43: 1, 44: .0, 45: 1, 46: 1, 47: .6, 48: 1}.get(i, 1 if i > 48 else 0)
    holo = np.clip((54 - i) / 12, 0, 1) if i >= 37 else 0      # hologram look fades out by frame 54
    f = settled(o, a, mo, i) if i >= 37 else o.copy()
    if i >= 37:
        l, r = clone_mask(i, True)
        cm = cv2.morphologyEx(cv2.bitwise_or(l, r), cv2.MORPH_OPEN, K(12))  # person only, no sofa blocks
        lf, rf = clone_mask(i, False)
        cm_full = cv2.bitwise_or(lf, rf)
        al = soft(cm) * vis
        clean = settled(o, a, mo, i)
        if holo > 0:
            h = tint(a, 0.75) * 1.1
            h[::3] *= 0.35                                                     # scan lines
            # horizontal slice displacement + chroma split, only on the clones
            for _ in range(int(10 * holo)):
                y0 = int(rng.integers(700, 1850)); hh = int(rng.integers(8, 60))
                h[y0:y0 + hh] = np.roll(h[y0:y0 + hh], int(rng.integers(-60, 60)), axis=1)
            h[..., 0] = np.roll(h[..., 0], int(14 * holo), axis=1)
            h[..., 2] = np.roll(h[..., 2], -int(14 * holo), axis=1)
            base = o.copy()
            f = over(base, h, al)
            f = over(f, clean, soft(cm_full) * (1 - holo) * vis)
            f = over(f, o, pa)
            f = screen(f, rim(body_for_glow(cm), 1.0) * vis * (1 - 0.85 * pa), 0.7 * holo + 0.1)
        else:
            f = clean
    # glitch hit on the whole frame at the start: blue flash + RGB split + shake
    hit = {35: .5, 36: 1.0, 37: .7, 38: .4, 40: .5, 42: .3}.get(i, 0)
    dx = dy = 0
    if hit:
        k = int(18 * hit)
        f[..., 0] = np.roll(f[..., 0], k, axis=1)
        f[..., 2] = np.roll(f[..., 2], -k, axis=1)
        f = screen(f, np.full((H, W), 0.35 * hit, np.float32), 1.0)
        dx, dy = rng.integers(-14, 15, 2) * hit
    s = {36: 1.06, 37: 1.04, 38: 1.025, 39: 1.01}.get(i, 1)
    return punch(f, s, dx, dy)


fn = {'parpadeo': parpadeo, 'aura': aura, 'glitch': glitch}[STYLE]
for i in range(1, 118):
    if ONLY and i != ONLY:
        continue
    cv2.imwrite(f'{OUT}/{i:03d}.png', np.clip(fn(i), 0, 255).astype(np.uint8))
print(STYLE, 'done')
