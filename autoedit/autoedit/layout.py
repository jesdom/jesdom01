"""Zona segura: donde SI se puede poner texto sin tapar la cara.

No usa modelos de IA. El sujeto de un video hablado es la region de la imagen
con mayor varianza temporal (boca, ojos, pelo, manos). Se muestrean frames,
se mide esa varianza y se obtiene la caja del sujeto; los graficos se anclan
SIEMPRE por debajo de esa caja, dentro del tercio inferior del encuadre.
"""
import subprocess

import numpy as np
from PIL import Image, ImageFilter


def _sample(path, n, w, h, fps):
    """n frames repartidos por todo el video, en pequeno."""
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)
    if len(a) > n:
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        a = a[idx]
    return a


def subject_box(path, out_w, out_h, sample_fps=2.0, n=48):
    """Caja (x0, y0, x1, y1) normalizada 0..1 del sujeto en movimiento."""
    sw, sh = 96, int(96 * out_h / out_w)
    frames = _sample(path, n, sw, sh, sample_fps)
    if len(frames) < 3:
        return (0.18, 0.05, 0.82, 0.62)
    var = frames.std(axis=0).mean(axis=2)
    # tambien pesa la diferencia entre frames consecutivos (movimiento real)
    mot = np.abs(np.diff(frames, axis=0)).mean(axis=(0, 3))
    score = var / (var.max() + 1e-6) + mot / (mot.max() + 1e-6)
    score = np.asarray(Image.fromarray(np.uint8(np.clip(score / score.max() * 255, 0, 255)))
                       .filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255.0
    thr = np.percentile(score, 88)
    mask = score >= thr
    rows = mask.sum(1).astype(np.float32)
    cols = mask.sum(0).astype(np.float32)
    rth, cth = rows.max() * 0.22, cols.max() * 0.22
    ys = np.where(rows >= rth)[0]
    xs = np.where(cols >= cth)[0]
    if len(ys) < 2 or len(xs) < 2:
        return (0.18, 0.05, 0.82, 0.62)
    return (xs[0] / sw, ys[0] / sh, (xs[-1] + 1) / sw, (ys[-1] + 1) / sh)


def safe_zone(path, size, cfg):
    """Devuelve el plan de colocacion de graficos en pixeles.

    - 'text_top': ninguna grafica de texto empieza por encima de esta linea.
    - 'caption_y': centro de los subtitulos.
    - 'chip_y': fila para los chips/motion graphics, SIEMPRE bajo la cara.
    - 'scrim': True si el sujeto llega tan abajo que hace falta una cinta de
      contraste (nunca sobre la cara: cubre pecho/fondo).
    """
    W, H = size
    g = cfg.get("layout", {})
    margin = g.get("face_margin", 0.045)
    try:
        box = subject_box(path, W, H)
    except Exception as exc:
        print(f"  [layout] deteccion no disponible ({exc}); uso zona segura por defecto")
        box = (0.18, 0.05, 0.82, 0.62)
    face_bottom = box[3]
    floor = 1.0 - g.get("bottom_margin", 0.085)        # barra de progreso + aire
    top = min(max(face_bottom + margin, g.get("min_top", 0.58)), g.get("max_top", 0.76))
    band_h = floor - top
    scrim = band_h < g.get("min_band", 0.16)
    if scrim:                                           # plano muy cerrado
        top = min(floor - g.get("min_band", 0.16), g.get("max_top", 0.78))
        band_h = floor - top
    caption_y = top + band_h * 0.62
    chip_y = top + band_h * 0.14
    info = {
        "box": box, "text_top": top, "caption_y": caption_y, "chip_y": chip_y,
        "floor": floor, "scrim": scrim, "band_h": band_h,
    }
    print(f"  [layout] sujeto y:{box[1]:.2f}-{box[3]:.2f} x:{box[0]:.2f}-{box[2]:.2f} | "
          f"banda de texto {top:.2f}-{floor:.2f}{' + cinta' if scrim else ''}")
    return info
