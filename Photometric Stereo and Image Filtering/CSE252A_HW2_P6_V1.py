#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:56:41 2020

@author: femw90
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Open image as grayscale
shark_img = io.imread('shark.png', as_gray=True)

# Show image
plt.imshow(shark_img, cmap=cm.gray)
plt.show()


def gaussian2d(filter_size=5, sig=1.0): 
    """Creates a 2D Gaussian kernel with side length ‘filter_size‘ and a sigma of ‘sig‘."""
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

sharpening_kernel = np.array([[1, 4,     6,  4, 1],
                              [4, 16,   24, 16, 4],
                              [6, 24, -476, 24, 6],
                              [4, 16,   24, 16, 4],
                              [1,  4,    6,  4, 1],
                            ]) * -1.0 / 256.0

def plot_results(original, filtered):
    # Plot original image
    plt.subplot(2,2,1)
    plt.imshow(original, vmin=0.0, vmax=1.0)
    plt.title('Original')
    plt.axis('off')
    
    # Plot filtered image
    plt.subplot(2,2,2)
    plt.imshow(filtered, vmin=0.0, vmax=1.0)
    plt.title('Filtered')
    plt.axis('off')
    plt.show()
    
    

from scipy.signal import convolve

def filter1(img):
    """Convolve the image with a 5x5 Gaussian filter with sigma=5.""" 
    
    """ ==========YOUR CODE HERE========== """
    kerDim = 5
    ker = gaussian2d(kerDim,5)
    result = convolve(img, ker)
    return result

def filter2(img):
    """Convolve the image with a 31x31 Gaussian filter with sigma=5.""" 
    
    """ ==========YOUR CODE HERE========== """
    kerDim = 31
    ker = gaussian2d(kerDim,5)
    result = convolve(img, ker)
    return result

def filter3(img):
    """Convolve the image with the provided sharpening filter.""" 
    
    """ ==========YOUR CODE HERE========== """
    ker = sharpening_kernel
    result = convolve(img, ker)
    return result

for filter_name, filter_fn in [
    ('5x5 Gaussian filter, sigma=5', filter1),
    ('31x31 Gaussian filter, sigma=5', filter2),
    ('sharpening filter', filter3),
]:
    filtered = filter_fn(shark_img)
    print(filter_name)
    plot_results(shark_img, filtered)