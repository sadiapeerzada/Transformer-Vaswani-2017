"""
Dataset: sklearn's bundled UCI 'digits' dataset (1797 real handwritten-digit scans, 8x8,
0-16 grayscale) -- used instead of full MNIST because this sandbox has no network access
to MNIST's hosting mirrors. Upsampled to 16x16 and scaled to [-1, 1], the standard DDPM
image range. Swapping this file for a real MNIST/CIFAR loader is a drop-in change; the
model/diffusion code is dataset-agnostic.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits


def get_dataset(img_size=16):
    digits = load_digits()
    imgs = digits.images.astype(np.float32) / 16.0  # -> [0, 1], shape (1797, 8, 8)
    imgs = torch.from_numpy(imgs).unsqueeze(1)       # (N, 1, 8, 8)
    imgs = F.interpolate(imgs, size=(img_size, img_size), mode="bicubic", align_corners=False)
    imgs = imgs.clamp(0, 1)
    imgs = imgs * 2 - 1                              # -> [-1, 1]
    return imgs
