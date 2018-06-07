#!/usr/bin/env python

import os.path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from itertools import combinations

import matplotlib.pyplot as pl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge
import networkx as nx
import numpy as np
from collections import defaultdict

from opensfm import dataset
from opensfm import features
from opensfm import io

class FStat:
    def __init__(self, p, f, c):
        self.p = p
        self.cnts = [0] * len(self.p)

def get_stats(ds):
    # Plot matches between images
    data = dataset.DataSet(ds)
    images = data.images()

    pairs = combinations(images, 2)

    match_counts = dict((i, FStat(*data.load_features(i))) for i in images)
    size_counts = defaultdict(lambda: [0,0])
    for im1, im2 in pairs:
        matches = data.find_matches(im1, im2)
        if len(matches) == 0:
            continue
        im1c, im2c = match_counts[im1], match_counts[im2]
        for i in range(len(im1c.p)):
            size_counts[int(im1c.p[i][2])][0] += 1
        for i in range(len(im2c.p)):
            size_counts[int(im2c.p[i][2])][0] += 1
        for e in matches:
            im1c.cnts[e[0]] += 1
            im2c.cnts[e[1]] += 1
            s1 = int(im1c.p[e[0]][2])
            s2 = int(im2c.p[e[1]][2])
            size_counts[s1][1] += 1
            size_counts[s2][1] += 1
    return size_counts, match_counts


if __name__ == '__main__':
    sc, mc = get_stats(sys.argv[1])
    [print(e, ': ', sc[e], '  ', float(sc[e][1]) / max(1, sc[e][0])) for e in sc]