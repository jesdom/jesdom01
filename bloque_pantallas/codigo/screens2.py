"""Screen content v2, rendered at 540x960 (2x) per frame.
L: real TikTok profile header of @grannyspills (public page screenshot) + animated follower count & growth curve.
C: AI-influencer reel playing (moving video).
R: For-You feed that swipes several times and lands on a moving POV video."""
import math
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 540, 960
NF = 209
OUT = 'b3/scr'
for s in 'LCR':
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
F = '/usr/share/fonts/opentype/inter/'
font = lambda w, s: ImageFont.truetype(F + {'b': 'Inter-Bold.otf', 's': 'Inter-SemiBold.otf', 'r': 'Inter-Regular.otf', 'x': 'Inter-ExtraBold.otf'}[w], s)
IMG = 'b2/img/'


def ease(p):
    p = min(max(p, 0), 1)
    return 1 - (1 - p) ** 3


def fmt(n):
    if n >= 1e6:
        return f'{n / 1e6:.1f}M'
    if n >= 1000:
        return f'{n / 1000:.1f}K'
    return str(int(n))


def cover(im, w, h, zoom=1.0, cx=0.5, cy=0.5):
    s = max(w / im.width, h / im.height) * zoom
    im = im.resize((int(im.width * s + 1), int(im.height * s + 1)), Image.LANCZOS)
    x = int((im.width - w) * cx); y = int((im.height - h) * cy)
    return im.crop((x, y, x + w, y + h))


def video_frames(path, n):
    """Decode a clip to n RGB frames at 30 fps."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vf', 'fps=30,scale=540:960:force_original_aspect_ratio=increase,crop=540:960',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, 960, 540, 3)
    return [Image.fromarray(arr[min(k, len(arr) - 1)]) for k in range(n)]


def heart(d, size, x, y, fill):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 48):
        hx = 16 * math.sin(t) ** 3
        hy = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((x + hx * size / 34, y + hy * size / 34))
    d.polygon(pts, fill=fill)


def reel_ui(im, handle, caption, likes, comments, shares):
    """TikTok-style overlay."""
    g = Image.new('RGBA', (W, 300)); gd = ImageDraw.Draw(g)
    for y in range(300):
        gd.line((0, y, W, y), fill=(0, 0, 0, int(190 * (y / 300) ** 1.3)))
    im.alpha_composite(g, (0, H - 300))
    d = ImageDraw.Draw(im)
    d.text((W / 2 - 128, 26), 'Siguiendo', font=font('s', 22), fill=(255, 255, 255, 170))
    d.text((W / 2 + 16, 26), 'Para ti', font=font('b', 22), fill='white')
    d.line((W / 2 + 30, 58, W / 2 + 70, 58), fill='white', width=3)
    d.text((22, H - 128), '@' + handle, font=font('b', 24), fill='white')
    d.text((22, H - 92), caption, font=font('r', 21), fill='white')
    x = W - 52
    for k, val in enumerate([likes, comments, shares]):
        y = H - 430 + k * 100
        if k == 0:
            heart(d, 44, x, y + 18, (255, 255, 255))
        elif k == 1:
            d.ellipse((x - 22, y - 2, x + 22, y + 36), fill='white'); d.polygon([(x - 14, y + 30), (x - 22, y + 44), (x - 2, y + 34)], fill='white')
        else:
            d.polygon([(x - 22, y + 8), (x + 22, y + 18), (x - 22, y + 30), (x - 12, y + 18)], fill='white')
        tw = d.textlength(val, font=font('b', 19)); d.text((x - tw / 2, y + 50), val, font=font('b', 19), fill='white')


# ---------------- L: real profile + growth
head = Image.open('b3/tt_grannyspills.png').convert('RGB').crop((0, 0, 1170, 610))
head = head.resize((W, int(610 * W / 1170)), Image.LANCZOS)
HS = W / 1170


def screen_L(i):
    im = Image.new('RGBA', (W, H), (255, 255, 255, 255))
    im.paste(head, (0, 0))
    d = ImageDraw.Draw(im)
    p = ease((i - 40) / 125)
    n = 970700 * p ** 1.8
    # replace the static follower number with the live one
    d.rectangle((548 * HS, 300 * HS, 684 * HS, 372 * HS), fill='white')
    txt = fmt(n)
    tw = d.textlength(txt, font=font('x', 16))
    d.text((682 * HS - tw, 318 * HS), txt, font=font('x', 16), fill=(22, 24, 35))
    top = head.height + 26
    d.text((24, top), 'Seguidores', font=font('s', 22), fill=(110, 110, 120))
    big = fmt(n)
    d.text((24, top + 30), big, font=font('x', 64), fill=(22, 24, 35))
    if p > 0.25:
        a = min(1, (p - 0.25) * 6)
        d.rounded_rectangle((24, top + 112, 300, top + 152), 20, fill=(254, 44, 85, int(255 * a)))
        d.text((40, top + 119), 'en sus primeras semanas', font=font('b', 19), fill=(255, 255, 255, int(255 * a)))
    # growth curve drawing itself
    cx0, cy0, cw, ch = 30, top + 190, W - 60, 300
    for k in range(5):
        y = cy0 + ch * k / 4; d.line((cx0, y, cx0 + cw, y), fill=(232, 232, 236), width=2)
    pts = []
    steps = int(120 * p) + 1
    for k in range(steps):
        u = k / 120
        pts.append((cx0 + cw * u, cy0 + ch - ch * (u ** 2.6)))
    if len(pts) > 1:
        fillpoly = pts + [(pts[-1][0], cy0 + ch), (cx0, cy0 + ch)]
        layer = Image.new('RGBA', (W, H)); ImageDraw.Draw(layer).polygon(fillpoly, fill=(254, 44, 85, 45)); im.alpha_composite(layer)
        d = ImageDraw.Draw(im)
        d.line(pts, fill=(254, 44, 85), width=6, joint='curve')
        ex, ey = pts[-1]; d.ellipse((ex - 9, ey - 9, ex + 9, ey + 9), fill=(254, 44, 85))
    d.text((cx0, cy0 + ch + 14), 'semana 1', font=font('r', 18), fill=(140, 140, 150))
    tw = d.textlength('hoy', font=font('r', 18)); d.text((cx0 + cw - tw, cy0 + ch + 14), 'hoy', font=font('r', 18), fill=(140, 140, 150))
    d.text((24, H - 46), 'TikTok · @grannyspills · cuenta creada con IA', font=font('r', 17), fill=(150, 150, 160))
    return im


# ---------------- C: AI influencer reel, moving video
vidC = video_frames('b3/v_sofi.mp4', NF)


def screen_C(i):
    im = vidC[max(0, i - 40)].convert('RGBA')
    likes = fmt(182000 + 260000 * ease((i - 40) / 90))
    reel_ui(im, 'sofi.virtual', 'mi rutina de los domingos ☕'.replace(' ☕', ''), likes, '3.4K', '12K')
    return im


# ---------------- R: feed with several swipes, last one is a moving video
feed = [('rome.png', 'historia.pov', 'POV: vives en la antigua Roma', '612K'),
        ('p1_b.png', 'sofi.virtual', 'atardeceres así', '98K'),
        ('p1_d.png', 'sofi.virtual', 'noche en la ciudad', '143K'),
        ('p1_a.png', 'sofi.virtual', 'outfit del día', '76K')]
feed_imgs = []
for f, h, c, l in feed:
    im = cover(Image.open(IMG + f).convert('RGB'), W, H, 1.04).convert('RGBA')
    reel_ui(im, h, c, l, '2.1K', '5K')
    feed_imgs.append(im)
vidR = video_frames('b3/v_medieval.mp4', NF)
SWIPES = [58, 76, 94, 112]          # frame where each swipe starts (10 frames long)


def screen_R(i):
    k = sum(1 for s in SWIPES if i >= s + 10)            # fully passed swipes
    cur = None
    def page(idx):
        if idx < len(feed_imgs):
            return feed_imgs[idx]
        v = vidR[max(0, i - SWIPES[-1])].convert('RGBA')
        reel_ui(v, 'historia.pov', 'POV: vives en la Edad Media', fmt(95000 + 600000 * ease((i - 112) / 80)), '5.1K', '9K')
        return v
    im = Image.new('RGBA', (W, H), (0, 0, 0, 255))
    active = [s for s in SWIPES if s <= i < s + 10]
    if active:
        sp = ease((i - active[0]) / 10); y = int(-H * sp)
        im.paste(page(k), (0, y)); im.paste(page(k + 1), (0, H + y))
    else:
        im.paste(page(k), (0, 0))
    return im


for i in range(NF):
    for s, fn in (('L', screen_L), ('C', screen_C), ('R', screen_R)):
        fn(i).convert('RGB').save(f'{OUT}/{s}/{i:03d}.png')
print('screens2 ok')
