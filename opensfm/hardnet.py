import torch
import torch.nn.init
import torch.nn as nn
from torch.autograd import Variable

import os
import sys
import cv2
import time
import numpy as np
import scipy.misc

from opensfm import features
from opensfm import csfm


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x


class HardNet(nn.Module):
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias = True))

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "HardNetPS.pth"))
model = HardNet()
checkpoint = torch.load(weights_path, map_location='cpu')
model.load_state_dict(checkpoint.get('state_dict') or checkpoint)
model.eval()
BATCH = 512


def extract_hardnet(img, config):
    p, patches = csfm.hadetect(
        img.astype(np.float32) / 255,  # VlFeat expects pixel values between 0, 1
        peak_threshold=config['hahog_peak_threshold'],
        edge_threshold=config['hahog_edge_threshold'],
        target_num_features=config['feature_min_frames'],
        use_adaptive_suppression=config['feature_use_adaptive_suppression'],
        sigma=config.get('desc_patch_sigma', 0.1),
        patch_size=30)

    N = len(patches)

    # resize to (32, 32)
    patches_resized = np.zeros((N, 1, 32, 32), dtype=np.float32)
    for i in range(len(patches)):
        patches_resized[i, 0, :, :] = scipy.misc.imresize(patches[i], (32, 32))

    descs = np.zeros((N, 128), dtype=np.float32)

    for i in range(0, N, BATCH):
        st = time.time()
        npinput = patches_resized[i:i+BATCH]
        input = torch.from_numpy(npinput)
        out = model(input)
        descs[i:i+BATCH] = out.data.numpy().reshape(-1, 128)
        print('calc hardnet feature: %d in %.1fs' % (len(npinput), time.time() - st))

    return p[:, :4], descs
