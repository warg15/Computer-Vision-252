#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:40:46 2020

@author: femw90
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import imageio
from scipy.signal import convolve

def gaussian2d(filter_size=5, sig=1.0):
    """Creates a 2D Gaussian kernel with
    side length `filter_size` and a sigma of `sig`."""
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)


a = image = convolve(np.ones((5,5)), gaussian2d(), mode='same')
b = gaussian_filter(np.ones((5,5)), 1.0)


nms_rXY = np.zeros((1,3))
temp = np.array([[44, 2, 7]])
nms_rXY = np.concatenate((nms_rXY, temp), axis=0)
temp = np.array([[1, 44, 17]])
nms_rXY = np.concatenate((nms_rXY, temp), axis=0)

#ort = np.sort(nms_rXY, axis = 0)


result = np.where(nms_rXY == np.amax(nms_rXY))