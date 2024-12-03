import numpy as np
import pandas as pd
import cv2
from PIL import Image
import math
from tqdm.auto import tqdm
import os, sys
from pathlib import Path

ds_dir = 'truby'
new_ds_dir = 'ds'
img_shape = (384, 384)

if not Path(new_ds_dir).exists():
    os.mkdir(new_ds_dir)

    fns = os.listdir(ds_dir)
    for i, fn in tqdm(enumerate(fns)):
        new_fn = os.path.join(new_ds_dir, f'{i}.jpg')
        img = Image.open(os.path.join(ds_dir, fn))
        img = img.resize(img_shape)
        img = np.array(img)
        cv2.imwrite(new_fn, img)