"""Clone composite of the three takes + floating holographic phone screens above each head.
Take A = src 1-200 (left), B = 201-400 (right), C = 401-609 (center, stands up and leaves)."""
import os
import sys
import cv2
import numpy as np
sys_path_added = __import__('sys').path.insert(0, 'b4')
from walkmask import walk_mask, PLATE

NF = 209
OUT = 'b4/out'
os.makedirs(OUT, exist_ok=True)
ONLY = int(sys.argv[1]) if len(sys.argv) > 1 else None
W, H = 1080, 1920
# A/B -> C alignment (ECC, ~1% focus breathing + small shift); used with WARP_INVERSE_MAP
WARP = np.float32([[0.9897, 0.0003, 5.30], [-0.00013, 0.9903, 10.3]])
BLUE = np.array([255, 110, 40], np.float32)     # BGR luminous blue-rose for glows
DEEP = np.array([140, 50, 22], np.float32)

SW, SH = 286, 572            # on-screen size of each phone screen
# screen centers x, bottom y (from head-top positions), open frame
SCREENS = {'L': dict(x=213, bottom=715, start=30), 'R': dict(x=828, bottom=707, start=35), 'C': dict(x=522, bottom=715, start=40)}
C_CLOSE = 128                # center screen collapses before he stands up


def ease(p):
    p = np.clip(p, 0, 1)
    return 1 - (1 - p) ** 3


def back(p):                 # ease-out-back (overshoot)
    p = np.clip(p, 0, 1); c = 1.7
    return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2


def src(i):
    return cv2.imread(f'b2/src/{i:03d}.jpg').astype(np.float32)


def mask(i, warp=False):
    m = cv2.imread(f'b2/m/{i:03d}.png', 0)
    if warp:
        m = cv2.warpAffine(m, WARP, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    m = cv2.dilate((m > 90).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.2)


def align(img):
    return cv2.warpAffine(img, WARP, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def over(dst, src_, a):
    return dst * (1 - a[..., None]) + src_ * a[..., None]


def screen_layer(frame, t, key, cfg):
    """Draw beam + holographic screen. Returns frame."""
    s0 = cfg['start']
    if t < s0:
        return frame
    if key == 'C' and t >= C_CLOSE + 10:
        return frame
    cx, bottom = cfg['x'], cfg['bottom']
    head_y = bottom + 45
    # open / close progress
    if key == 'C' and t >= C_CLOSE:
        k = 1 - (t - C_CLOSE) / 10
        beam_p, open_p = min(1, k * 2), max(0, k * 1.6 - 0.6)
    else:
        beam_p = ease((t - s0) / 6)
        open_p = back((t - s0 - 5) / 9) if t >= s0 + 5 else 0
    bob = 4 * np.sin((t - s0) / 11 + cx)
    top = bottom - SH + bob
    glow = np.zeros((H, W), np.float32)
    # beam: from a dot above the head up to the screen
    if False:
        ytop = head_y - (head_y - (bottom + bob)) * beam_p
        pts = np.array([[cx - 10, head_y], [cx + 10, head_y], [cx + 120 * beam_p, ytop], [cx - 120 * beam_p, ytop]], np.int32)
        cone = np.zeros((H, W), np.float32); cv2.fillPoly(cone, [pts], 1.0)
        grad = np.clip((head_y - np.arange(H)) / max(1, head_y - ytop), 0, 1)[:, None]
        cone *= (0.35 + 0.35 * grad) * (1 - min(1, open_p)); cone = cv2.GaussianBlur(cone, (0, 0), 6)
        frame = 255 - (255 - frame) * (1 - np.clip(cone[..., None] * 0.55 * BLUE / 255, 0, 1))
        cv2.circle(glow, (int(cx), int(head_y)), 7, 1.0 - min(1, open_p), -1)
    if open_p > 0.02:
        sh = max(4, int(SH * min(open_p, 1.08)))
        sw = SW if open_p > 0.15 else int(SW * (0.3 + 4.6 * open_p))
        content = cv2.imread(f'b4/scr/{key}/{min(t, NF - 1):03d}.png')
        content = cv2.resize(content, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        x0, y0 = int(cx - sw / 2), int(bottom + bob - sh)
        rr = np.zeros((sh, sw), np.uint8)
        cv2.rectangle(rr, (14, 0), (sw - 15, sh - 1), 255, -1); cv2.rectangle(rr, (0, 14), (sw - 1, sh - 15), 255, -1)
        for cxx, cyy in [(14, 14), (sw - 15, 14), (14, sh - 15), (sw - 15, sh - 15)]:
            cv2.circle(rr, (cxx, cyy), 14, 255, -1)
        a = cv2.GaussianBlur(rr.astype(np.float32) / 255, (0, 0), 0.8) * min(1, open_p * 2)
        if key == 'L':                                   # he walks right in front of it: hide it while he covers it
            a = a * float(np.interp(t, [189, 193, 198, 202], [1, 0, 0, 1]))
        ys, xs = slice(max(0, y0), min(H, y0 + sh)), slice(max(0, x0), min(W, x0 + sw))
        cy0, cx0 = ys.start - y0, xs.start - x0
        sub_a = a[cy0:cy0 + (ys.stop - ys.start), cx0:cx0 + (xs.stop - xs.start)]
        sub_c = content[cy0:cy0 + (ys.stop - ys.start), cx0:cx0 + (xs.stop - xs.start)]
        frame[ys, xs] = over(frame[ys, xs], sub_c, sub_a)
    g = cv2.GaussianBlur(glow, (0, 0), 2) * 1.2
    return 255 - (255 - frame) * (1 - np.clip(g[..., None] * BLUE / 255, 0, 1))


for t in range(NF):
    if ONLY is not None and t != ONLY:
        continue
    a = t if t < 200 else 199 - (t - 199)          # ping-pong the 9 extra frames of A/B
    iA, iB, iC = 1 + a, 201 + a, 401 + t
    C = src(iC)
    f = C.copy()
    f = over(f, align(src(iA)), mask(iA, True))
    f = over(f, align(src(iB)), mask(iB, True))
    mA, mB, mC = mask(iA, True), mask(iB, True), mask(iC)
    # where the center layer overlaps a side clone, keep it in front only on his actual body
    # (pixels that differ from the empty-room plate); its stray sofa must not erase the clone
    body = cv2.GaussianBlur(np.abs(C - PLATE).max(2), (0, 0), 2) > 22
    body = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    body = cv2.GaussianBlur(cv2.dilate(body, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.5)
    side = np.clip(np.maximum(mA, mB) * 3, 0, 1)
    if t >= 172:                                      # walking out in front of the left clone: he is fully in front
        side = side * 0
    mC = mC * (1 - side) + np.minimum(mC, body) * side
    f = over(f, C, mC)                                # center person in front
    near = lambda m: cv2.GaussianBlur((m > 0.5).astype(np.float32), (0, 0), 7)
    side = np.maximum(near(mA), near(mB))
    gap = np.clip(side * near(mC) * 4, 0, 1) * (1 - np.maximum(np.maximum(mA, mB), mC))
    f = f * (1 - 0.6 * gap[..., None])
    for key in ('L', 'R', 'C'):
        f = screen_layer(f, t, key, SCREENS[key])
    if t >= 150:                                      # walking out near the camera: he passes in front of the screens
        mw = walk_mask(C, cv2.imread(f'b2/m/{iC:03d}.png', 0))
        mw = cv2.dilate(mw, np.ones((7, 7), np.uint8))
        band = np.zeros((H, W), np.float32); band[:760] = 1; band = cv2.GaussianBlur(band, (0, 0), 15)
        f = over(f, C, np.maximum(mC, mw * band * (1 - side) + np.minimum(mw, body) * band * side))
    cv2.imwrite(f'{OUT}/{t:03d}.jpg', np.clip(f, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 96])
print('comp done')
