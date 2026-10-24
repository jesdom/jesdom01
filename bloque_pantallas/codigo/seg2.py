from rembg import new_session, remove
from PIL import Image
import numpy as np
hs = new_session('u2net_human_seg')
for i in range(1, 610):
    im = Image.open(f'b2/src/{i:03d}.jpg').convert('RGB')
    Image.fromarray(np.asarray(remove(im, session=hs, only_mask=True))).save(f'b2/m/{i:03d}.png')
    if i % 50 == 0:
        print(i, flush=True)
print('DONE', flush=True)
