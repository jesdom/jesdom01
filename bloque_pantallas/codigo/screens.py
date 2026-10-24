"""Render the content of the three floating phone screens, one RGBA PNG per frame per screen.
Fictional accounts styled after real fast-growing AI-influencer formats."""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 270, 480          # screen content size (9:16)
NF = 209                 # frames in the composite (30 fps)
OUT = 'b2/scr'
for s in ('L', 'C', 'R'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)

F = '/usr/share/fonts/opentype/inter/'
font = lambda w, s: ImageFont.truetype(F + {'b': 'Inter-Bold.otf', 's': 'Inter-SemiBold.otf', 'r': 'Inter-Regular.otf', 'x': 'Inter-ExtraBold.otf'}[w], s)
IMG = 'b2/img/'


def cover(path, w, h, zoom=1.0, cx=0.5, cy=0.5):
    im = Image.open(path).convert('RGB')
    s = max(w / im.width, h / im.height) * zoom
    im = im.resize((int(im.width * s + 1), int(im.height * s + 1)), Image.LANCZOS)
    x = int((im.width - w) * cx); y = int((im.height - h) * cy)
    return im.crop((x, y, x + w, y + h))


def circle(im, d):
    im = im.resize((d, d), Image.LANCZOS)
    m = Image.new('L', (d, d), 0); ImageDraw.Draw(m).ellipse((0, 0, d - 1, d - 1), fill=255)
    out = Image.new('RGBA', (d, d)); out.paste(im, (0, 0), m)
    return out


def ease(p):
    p = min(max(p, 0), 1)
    return 1 - (1 - p) ** 3


def fmt(n):
    if n >= 1000:
        return f'{n / 1000:.1f}K'.replace('.0K', 'K')
    return str(int(n))


def heart(d, size, x, y, alpha):
    r = size / 4
    pts = []
    for t in np.linspace(0, 2 * math.pi, 40):
        hx = 16 * math.sin(t) ** 3
        hy = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((x + hx * size / 34, y + hy * size / 34))
    d.polygon(pts, fill=(255, 48, 92, int(255 * alpha)))


# ---------------- Left: profile with exploding follower count
grid = [IMG + f for f in ('p1_a.png', 'p1_b.png', 'p1_c.png', 'p1_d.png', 'p1_main.png', 'p1_b.png')]
grid_tiles = [cover(g, 89, 89) for g in grid]
avatar = circle(cover(IMG + 'p1_main.png', 200, 200, 1.6, 0.45, 0.2), 64)


def screen_L(i):
    im = Image.new('RGBA', (W, H), (12, 12, 14, 255)); d = ImageDraw.Draw(im)
    d.text((14, 12), 'sofi.virtual', font=font('b', 15), fill='white')
    d.ellipse((112, 17, 124, 29), fill=(56, 151, 240)); d.line((115, 23, 117.5, 26, 121.5, 20), fill='white', width=2)
    # avatar with story ring
    d.ellipse((12, 42, 84, 114), outline=(225, 48, 108), width=3)
    im.alpha_composite(avatar, (16, 46))
    p = ease((i - 40) / 120)
    followers = 127400 * p ** 1.6
    for k, (num, lab) in enumerate([('24', 'posts'), (fmt(followers), 'seguidores'), ('12', 'seguidos')]):
        x = 104 + k * 56
        f = font('x', 16 if k != 1 else 17)
        tw = d.textlength(num, font=f)
        col = 'white' if k != 1 else (255, 255, 255) if p < 1 else (120, 200, 255)
        d.text((x + 24 - tw / 2, 60), num, font=f, fill=col)
        lw = d.textlength(lab, font=font('r', 10))
        d.text((x + 24 - lw / 2, 82), lab, font=font('r', 10), fill=(200, 200, 205))
    d.text((14, 122), 'Sofía · AI', font=font('s', 12), fill='white')
    d.text((14, 139), 'Influencer virtual · creada con IA', font=font('r', 11), fill=(215, 215, 220))
    d.text((14, 155), 'Madrid · lifestyle · colabs', font=font('r', 11), fill=(215, 215, 220))
    # growth badge pops when the counter is flying
    if 60 < i:
        a = min(1, (i - 60) / 8)
        d.rounded_rectangle((14, 176, 162, 198), 11, fill=(28, 70, 190, int(230 * a)))
        d.text((24, 180), '+100K en 3 semanas', font=font('b', 11), fill=(255, 255, 255, int(255 * a)))
    d.rounded_rectangle((14, 206, 130, 232), 7, fill=(0, 149, 246)); d.text((52, 212), 'Seguir', font=font('s', 12), fill='white')
    d.rounded_rectangle((136, 206, 256, 232), 7, fill=(38, 38, 42)); d.text((166, 212), 'Mensaje', font=font('s', 12), fill='white')
    # grid scrolls up slowly
    off = int(18 * ease((i - 50) / 140))
    for k, t in enumerate(grid_tiles):
        x = 1 + (k % 3) * 90; y = 244 + (k // 3) * 90 - off
        im.paste(t, (x, y))
    for k, t in enumerate(grid_tiles[:3]):
        im.paste(t, (1 + k * 90, 244 + 180 - off))
    return im


# ---------------- Center: granny reel, likes climbing
hearts = [(np.random.default_rng(k).uniform(200, 248), np.random.default_rng(k + 50).uniform(0, 160)) for k in range(14)]


def reel_ui(im, i, handle, caption, likes, comments):
    d = ImageDraw.Draw(im)
    g = Image.new('RGBA', (W, 140), (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
    for y in range(140):
        gd.line((0, y, W, y), fill=(0, 0, 0, int(170 * y / 140)))
    im.alpha_composite(g, (0, H - 140))
    d.text((12, 12), 'Reels', font=font('x', 16), fill='white')
    d.ellipse((12, H - 78, 34, H - 56), fill=(90, 90, 100)); d.text((40, H - 76), handle, font=font('s', 12), fill='white')
    d.rounded_rectangle((40 + d.textlength(handle, font=font('s', 12)) + 8, H - 77, 40 + d.textlength(handle, font=font('s', 12)) + 54, H - 59), 5, outline='white')
    d.text((40 + d.textlength(handle, font=font('s', 12)) + 14, H - 76), 'Seguir', font=font('s', 10), fill='white')
    d.text((12, H - 48), caption, font=font('r', 11), fill='white')
    for k, (ic, val) in enumerate([('h', likes), ('c', comments), ('s', '')]):
        y = H - 200 + k * 46
        if ic == 'h':
            heart(d, 22, W - 24, y + 8, 1)
        elif ic == 'c':
            d.ellipse((W - 35, y - 2, W - 13, y + 18), outline='white', width=2)
        else:
            d.polygon([(W - 34, y), (W - 12, y + 9), (W - 34, y + 18), (W - 29, y + 9)], outline='white')
        if val:
            tw = d.textlength(val, font=font('s', 10)); d.text((W - 24 - tw / 2, y + 22), val, font=font('s', 10), fill='white')


def screen_C(i):
    z = 1.05 + 0.10 * ease(i / NF)
    im = cover(IMG + 'granny.png', W, H, z + 0.42, 0.3, 0.0).convert('RGBA')  # crop out the baked-in app UI
    likes = fmt(48000 + 900000 * ease((i - 40) / 110) ** 1.3)
    reel_ui(im, i, 'abuela.ia', 'Hoy les cuento cómo conocí a su abuelo', likes, fmt(3100 + 9000 * ease((i - 40) / 110)))
    d = ImageDraw.Draw(im)
    for k, (x0, ph) in enumerate(hearts):   # floating hearts
        t = (i * 2.2 + ph) % 160
        if i > 44:
            heart(d, 12 + k % 3 * 4, x0 + 10 * math.sin(t / 12 + k), H - 210 - t * 1.4, max(0, 1 - t / 160))
    return im


# ---------------- Right: history POV reels, swipe from Rome to Middle Ages
def screen_R(i):
    S = 112                          # swipe frame
    sp = ease((i - S) / 10)
    a = cover(IMG + 'rome.png', W, H, 1.06 + 0.06 * ease(i / S), 0.5, 0.45).convert('RGBA')
    b = cover(IMG + 'medieval.png', W, H, 1.06 + 0.06 * ease((i - S) / 90), 0.5, 0.45).convert('RGBA')
    reel_ui(a, i, 'historia.pov', 'POV: vives en la antigua Roma', fmt(210000 + 400000 * ease((i - 40) / 70)), '8.4K')
    reel_ui(b, i, 'historia.pov', 'POV: vives en la Edad Media', fmt(95000 + 600000 * ease((i - S) / 80)), '5.1K')
    y = int(-H * sp)
    im = Image.new('RGBA', (W, H), (0, 0, 0, 255))
    im.paste(a, (0, y)); im.paste(b, (0, H + y))
    return im


for i in range(NF):
    for s, fn in (('L', screen_L), ('C', screen_C), ('R', screen_R)):
        fn(i).save(f'{OUT}/{s}/{i:03d}.png')
print('screens ok')
