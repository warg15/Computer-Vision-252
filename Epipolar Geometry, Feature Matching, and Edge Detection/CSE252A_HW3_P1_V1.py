#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:33:57 2020

@author: William Argus
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import convolve
#% matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = [5, 5]

def gaussian2d(filter_size=5, sig=1.0):
    """Creates a 2D Gaussian kernel with
    side length `filter_size` and a sigma of `sig`."""
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def smooth(image):
    """ ==========
    YOUR CODE HERE
    ========== """
    image = convolve(image, gaussian2d(), mode='same')
    return image

def gradient(image):
    """ ==========
    YOUR CODE HERE
    ========== """
    g_mag = np.zeros_like(image)
    g_theta = np.zeros_like(image)
    
    kh = np.array([[1,0,-1]])
    kv = np.array([[1],[0],[-1]])
    
    gX = convolve(image, kh, mode = 'same')
    gY = convolve(image, kv, mode = 'same')
    
    g_mag = abs(np.sqrt( np.square(gX) + np.square(gY) ))
    g_theta = (180/np.pi)*np.arctan(gY/gX)
    
    '''
    plt.figure(1)
    plt.imshow(gX, cmap='gray')
    plt.show()
    plt.figure(2)
    plt.imshow(gY, cmap='gray')
    plt.show()

    plt.figure(3)
    plt.imshow(g_mag, cmap='gray')
    plt.show()
    
    plt.figure(4)
    plt.imshow(g_theta, cmap='gray')
    plt.show()
    '''
    return g_mag, g_theta

def nms(g_mag, g_theta):
    """ ==========
    YOUR CODE HERE
    ========== """
    nms_response = np.array(g_mag)
    g_theta[(-22.5 <= g_theta) & (22.5 > g_theta)] = 0
    g_theta[(-67.5 <= g_theta) & (-22.5 > g_theta)] = -45
    g_theta[(67.5 > g_theta) & (22.5 <= g_theta)] = 45
    g_theta[-67.5 > g_theta] = 90
    g_theta[67.5 <= g_theta] = 90
    
    sz = np.shape(nms_response)
    pixel1,pixel2 = 0,0
    for i in range(1, sz[0]-2):
        for j in range(1,sz[1]-2):
            #get gradient direction to pick comparison pixels
           if g_theta[i,j] == 90:
               pixel1 = nms_response[i-1,j]
               pixel2 = nms_response[i+1,j]
           elif g_theta[i,j] == -45:
               pixel1 = nms_response[i-1,j-1]
               pixel2 = nms_response[i+1,j+1]
           elif g_theta[i,j] == 0:
               pixel1 = nms_response[i,j+1]
               pixel2 = nms_response[i,j-1]
           elif g_theta[i,j] == 45:
               pixel1 = nms_response[i-1,j+1]
               pixel2 = nms_response[i+1,j-1]
           else:
               print("ERROR!!")
           if (nms_response[i,j] >= pixel1) & (nms_response[i,j] >= pixel2):
               nothing=0
           else:
               nms_response[i,j]=0
    return nms_response



def hysteresis_threshold(image, g_theta):
    """ ==========
    YOUR CODE HERE
    ========== """
    threshImage = np.array(image)
    def checkPoint(x, y):
        x1,y1,x2,y2 = 0,0,0,0
        if g_theta[x,y] == 90:
            x1,y1 = x,y-1
            x2,y2 = x,y+1
        elif g_theta[x,y] == -45:
            x1,y1 = x-1,y+1
            x2,y2 = x+1,y-1
        elif g_theta[x,y] == 0:
            x1,y1 = x+1,y
            x2,y2 = x-1,y
        elif g_theta[x,y] == 45:
            x1,y1 = x-1,y-1
            x2,y2 = x+1,y+1
        else:
            print("ERROR!!")
        if (threshImage[x1,y1] > t_min) & (threshImage[x1,y1] < t_max):
            start_pts[x1,y1] = edgePoint
            threshImage[x1,y1] = edgePoint
            checkPoint(x1, y1)
            
        if (threshImage[x2,y2] > t_min) & (threshImage[x2,y2] < t_max):
            start_pts[x2,y2] = edgePoint
            threshImage[x2,y2] = edgePoint
            checkPoint(x2, y2)
        return
    
    t_min = .0001
    t_max = .2
    edgePoint = 1.0
    sz = np.shape(threshImage)
    start_pts = np.zeros((sz[0],sz[1]))
    #sz = np.shape(start_pts)
    for i in range(0, sz[0]-1):
        for j in range(0,sz[1]-1):
            if threshImage[i,j] >= t_max:
                start_pts[i,j] = edgePoint
                checkPoint(i, j)
    result = start_pts*image
    return result

def edge_detect(image):
    """Perform edge detection on the image."""
    smoothed = smooth(image)
    g_mag, g_theta = gradient(smoothed)
    nms_image = nms(g_mag, g_theta)
    thresholded = hysteresis_threshold(nms_image, g_theta)
    return smoothed, g_mag, nms_image, thresholded

# Load image in grayscale
image = io.imread('sio_pier.jpg', as_gray=True)
assert len(image.shape) == 2, 'image should be grayscale; check your Python/skimage versions'

# Perform edge detection
smoothed, g_mag, nms_image, thresholded = edge_detect(image)
'''
nms_edges = nms_image > 0
plt.imshow(nms_edges,cmap=cm.gray)
plt.show()

print('Original:')
plt.imshow(image, cmap=cm.gray)
plt.show()

print('Smoothed:')
plt.imshow(smoothed, cmap=cm.gray)
plt.show()
'''
print('Gradient magnitude:')
plt.imshow(g_mag, cmap=cm.gray)
plt.show()

print('NMS:')
plt.imshow(nms_image, cmap=cm.gray)
plt.show()

print('Thresholded:')
plt.imshow(thresholded, cmap=cm.gray)
plt.show()
