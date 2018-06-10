import sys

import numpy as np
import cv2
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge

from opensfm import dataset
from opensfm import features
from opensfm import matching
from opensfm import csfm


def plot_patches_comp(patch, pts, img):
    w, h = img.shape[1], img.shape[0]
    for i in range(10):
        x,y = pts[i,0], pts[i,1]
        plt.subplot(211)
        plt.imshow(patch[i], cmap='gray')
        plt.subplot(212)
        sz = pts[i, 2] * 7.5
        xs = max(0, int(x - sz))
        xm = min(w, int(x + sz + 0.5))
        ys = max(0, int(y - sz))
        ym = min(h, int(y + sz + 0.5))
        plt.imshow(img[ys:ym, xs:xm])
        plt.show()


def plot_patches(patches, nmax=100):
    patches = patches[:nmax]
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


def extract_patches(img, config, smin=0, smax=1000):
    assert len(img.shape) == 3
    resized_color_image = features.resized_image(img, config)
    ratio = float(img.shape[0]) / resized_color_image.shape[0]
    color_image = resized_color_image
    image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

    p, patches = csfm.hadetect(
        image.astype(np.float32) / 255,  # VlFeat expects pixel values between 0, 1
        peak_threshold=config['hahog_peak_threshold'],
        edge_threshold=config['hahog_edge_threshold'],
        target_num_features=config['feature_min_frames'],
        use_adaptive_suppression=config['feature_use_adaptive_suppression'],
        sigma=config.get('desc_patch_sigma', 0.5))

    valid = np.logical_and(p[:,2] >= smin, p[:,2] <= smax)
    p = p[valid]
    patches = patches[valid]
    plot_patches_comp(patches, p, resized_color_image)
    return p, patches


def extract_features(img, config, smin=0, smax=1000):
    p, f, c = features.extract_features(img, config, None)
    valid = np.logical_and(p[:,2] >= smin, p[:,2] <= smax)
    p = p[valid]
    f = f[valid]
    c = c[valid]
    order = np.argsort(p[:,2])
    p = p[order]
    f = f[order]
    c = c[order]
    print('feature [%.1f-%.1f] %d' %(smin, smax, p.shape[0]))
    return p, f, c


def draw_match(img1, img2, p1, p2, matches, rmatches):
    h1, w1, c = img1.shape
    h2, w2, c = img2.shape
    ha, wa = h1 + h2, max(w1, w2)
    dpi = plt.rcParams['figure.dpi']
    plt.figure()#figsize=(wa/dpi, ha/dpi))
    image = np.zeros((ha, wa, 3), dtype=img1.dtype)
    image[0:h1, 0:w1, :] = img1
    image[h1:h1 + h2, 0:w2, :] = img2
    plt.imshow(image)
    plt.axis('off')

    # draw features
    p1, sa1 = features.denormalized_image_coordinates(p1, w1, h1), p1[:, 2:4]
    p2, sa2 = features.denormalized_image_coordinates(p2, w2, h2), p2[:, 2:4]
    patches = []
    for p, sa in zip(p1, sa1):
        patches.append(Wedge(p, sa[0] * 2, sa[1] + 1, sa[1] - 1))
    for p, sa in zip(p2, sa2):
        patches.append(Wedge((p[0], p[1] + h1), sa[0] * 2, sa[1] + 1, sa[1] - 1))

    collection = PatchCollection(patches, alpha=0.4)
    plt.gca().add_collection(collection)

    if matches is not None:
        matches = np.array(list(set([(e[0], e[1]) for e in matches]) - set([(e[0], e[1]) for e in rmatches])), dtype=np.int32)
        matches = matches[:100]
        p1 = p1[matches[:, 0]]
        p2 = p2[matches[:, 1]]
        for a, b in zip(p1, p2):
            plt.plot([a[0], b[0]], [a[1], b[1] + h1], 'c')
    plt.show()


def onedir_match(f1, f2, reverse=False):
    mr = cv2.DescriptorMatcher_create('BruteForce')
    mts = mr.match(f2, f1)
    mts = [e for e in mts if e.distance<0.8]
    matches = [(e.queryIdx, e.trainIdx) if reverse else (e.trainIdx, e.queryIdx)  for e in mts]
    mdists = [e.distance for e in mts]
    return matches, mdists


def stripplot_multi(*datas):
    l = sum(len(d) for d in datas)
    ret = np.zeros((l,2), dtype=np.float32)
    cur = 0
    for idx, d in enumerate(datas):
        ret[cur:cur+len(d),0] = d
        ret[cur:cur+len(d),1] = idx
        cur=len(d)
    sns.stripplot(x=ret[:,1], y=ret[:,0], jitter=0.2)
    plt.show()

data = dataset.DataSet('data/desk')
cfg = data.config

img1 = data.image_as_array('IMG_20180606_104135.jpg')
img2 = data.image_as_array('IMG_20180606_104145.jpg')
#img2 = data.image_as_array('IMG_20180606_104150.jpg')
size_range = [3.0, 30]

p1, patches1 = extract_patches(img1, cfg, size_range[0], size_range[1])
#plot_patches(patches1)


# p1, f1, c1 = extract_features(img1, cfg, size_range[0], size_range[1])
# p2, f2, c2 = extract_features(img2, cfg, size_range[0], size_range[1])
#
# if False:
#     match12, dis12 = onedir_match(f1, f2)
#     match21, dis21 = onedir_match(f2, f1, reverse=True)
#     match_set = set(match12).intersection(match21)
#     matches = np.array(list(match_set), dtype=np.int32)
#     dists = np.array([d for i, d in enumerate(dis12) if match12[i] in match_set], dtype=np.float32)
#     rmatches = matching.robust_match_fundamental(p1, p2, matches, cfg)
#     print("match [%d : %d] 1->2: %d 2->1: %d symetric: %d robust: %d" % (
#         len(p1), len(p2), len(match12), len(match21), len(matches), len(rmatches)))
#     # sizes = np.concatenate([p1[matches[:, 0], 2], p2[matches[:, 1], 2]])
#     # rsizes = np.concatenate([p1[rmatches[:, 0], 2], p2[rmatches[:, 1], 2]])
#     # stripplot_multi(sizes, rsizes)
# else:
#     i1 = features.build_flann_index(f1, cfg)
#     i2 = features.build_flann_index(f2, cfg)
#     cfg['lowes_ratio'] = 1.0
#     matches = matching.match_symmetric(f1, i1, f2, i2, cfg)
#     rmatches = matching.robust_match_fundamental(p1, p2, matches, cfg)
#     print("match [%d : %d] symetric: %d robust: %d" % (
#         len(p1), len(p2), len(matches), len(rmatches)))
#
# draw_match(img1, img2, p1, p2, matches, rmatches)

# rmatchset = set((e[0], e[1]) for e in rmatches)
# succ = np.array([(e[0], e[1]) in rmatchset for e in matches])
# sns.stripplot(x=succ, y=dists, jitter=True)
# plt.show()
