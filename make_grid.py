import sys
from pathlib import Path

import numpy as np
from skimage import io


for im_path in sorted(Path(sys.argv[1]).glob('*.jpg')):
    im = io.imread(im_path)
    h, w, _ = im.shape
    hn, wn = int(round(h / 2.4)), int(round(w / 2.4))
    hs, ws = int(round(hn / 10)), int(round(wn / 10))
    hl, wl = int(round(h / 10)), int(round(w / 10))
    mask_s = np.zeros((h, w), dtype=np.uint8)
    mask_l = np.ones((h, w), dtype=np.uint8)
    mask_s[:hn, :wn] = 1
    mask_l[:hn, :wn] = 0
    grid_s = np.zeros((h, w), dtype=np.uint8)
    grid_l = np.zeros((h, w), dtype=np.uint8)
    grid_s[::hs] = 1
    grid_s[:, ::ws] = 1
    grid_l[::hl] = 1
    grid_l[:, ::wl] = 1
    grid = grid_s * mask_s + grid_l * mask_l
    grid = grid.astype(bool)
    im[..., 0][grid] = 255
    im[..., 1][grid] = 0
    im[..., 2][grid] = 0
    io.imsave(im_path, im)
    break
