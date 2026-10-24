import sys, glob, os
from rembg import new_session, remove
from PIL import Image, ImageChops
import numpy as np
start=int(sys.argv[1])
isn=new_session('isnet-general-use'); hs=new_session('u2net_human_seg')
for i in range(start,118):
    o=Image.open(f'fx/orig/{i:03d}.png').convert('RGB')
    m1=np.asarray(remove(o,session=isn,only_mask=True),np.uint8); m2=np.asarray(remove(o,session=hs,only_mask=True),np.uint8)
    Image.fromarray(np.maximum(m1,m2)).save(f'fx/mo/{i:03d}.png')
    a=Image.open(f'fx/ai/{i:03d}.png').convert('RGB')
    Image.fromarray(np.asarray(remove(a,session=isn,only_mask=True),np.uint8)).save(f'fx/ma/{i:03d}.png')
    print(i,flush=True)
