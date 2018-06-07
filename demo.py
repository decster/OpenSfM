import scipy.misc
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge

from opensfm import csfm

def draw_features(img, pts, patches):
    h, w = img.shape

    plt.figure()
    ax = plt.axes()
    ax.imshow(img, cmap='gray')
    weds = []
    for pt in pts:
        weds.append(Wedge((pt[0], pt[1]), pt[2]*7.5, pt[3]+1, pt[3]-1))
    col = PatchCollection(weds, alpha=0.4)
    ax.add_collection(col)

    plt.figure()
    SZ = patches[0].shape[0]
    N = len(patches)
    NW = 8
    NH = (N + NW - 1) // NW
    grid = np.zeros((NH*32, NW*32), dtype=np.float32)
    for i in range(N):
        gy = i // NW
        gx = i % NW
        grid[gy*32:gy*32+31, gx*32:gx*32+31] = patches[i]
    plt.imshow(grid, cmap='gray')

    plt.show()

img = scipy.misc.imread('data/desk/images/IMG_20180606_104124.jpg', mode='L')
ims = scipy.misc.imresize(img, (2048 // 4 * 3, 2048))
img = ims.astype(np.float32) / 255
pts, patches = csfm.hadetect(img, target_num_features=1000, sigma=0.1)

idx = np.logical_and(pts[:,2] >= 6, pts[:,2] <= 8)
draw_features(img, pts[idx][:2], patches[idx][:2])
