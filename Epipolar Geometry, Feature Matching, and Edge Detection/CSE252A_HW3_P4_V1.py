#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:51:42 2020

@author: femw90
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import maximum_filter
import imageio
from scipy.signal import convolve
from numpy.linalg import eig

def gaussian2d(filter_size=5, sig=1.0):
    """Creates a 2D Gaussian kernel with
    side length `filter_size` and a sigma of `sig`."""
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def smooth(image, sigma):
    """ ==========
    YOUR CODE HERE
    ========== """
    fltr = gaussian2d(filter_size=6*sigma, sig=sigma)
    image = convolve(image, fltr, mode='same')
    return image
def rgb2gray(rgb):
    """ Convert rgb image to grayscale.
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
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
    gX[:,0] = gX[:,1]
    gY[0,:] = gY[1,:]
    #g_mag = abs(np.sqrt( np.square(gX) + np.square(gY) ))
    #g_theta = (180/np.pi)*np.arctan(gY/gX)
    return gX, gY



def corner_detect(image, nCorners, smoothSTD, windowSize):
    """Detect corners on a given image.

    Args:
        image: Given a grayscale image on which to detect corners.
        nCorners: Total number of corners to be extracted.
        smoothSTD: Standard deviation of the Gaussian smoothing kernel.
        windowSize: Window size for corner detector and non-maximum suppression.

    Returns:
        Detected corners (in image coordinate) in a numpy array (n*2).

    """
    
    """ ==========
    YOUR CODE HERE
    ========== """
    corners = np.zeros((nCorners, 2))
    rXY = np.zeros_like(image)
    #windowSize=4
    #filter with Gaussian
    #plt.imshow(image)
    #plt.show()
    smoothedImage = gaussian_filter(image, smoothSTD)
    #plt.imshow(smoothedImage)
    #plt.show()
    #compute the gradient everywhere
    gX, gY = gradient(smoothedImage)
    #Move the window W over the image and construct C for each window location (x,y)
    sz = np.shape(smoothedImage)
    #sub = np.floor( (windowSize-1)/2 ).astype(numpy.int64)
    #add = np.ceil( (windowSize-1)/2 ).astype(numpy.int64)
    sub = windowSize//2
    add = windowSize//2
    for i in range(sub, sz[0]-1-add ):
        for j in range(sub, sz[1]-1-add ):
            #first move over the window for each pixel
            c = np.zeros((2,2))
            if windowSize%2 == 0:
                wX = np.array(gX[i-sub:i+add, j-sub:j+add])
                wY = np.array(gY[i-sub:i+add, j-sub:j+add])
            else:
                wX = np.array(gX[i-sub:i+add+1, j-sub:j+add+1])
                wY = np.array(gY[i-sub:i+add+1, j-sub:j+add+1])
            
            c[0,0] = np.sum(wX*wX)
            c[0,1] = np.sum(wX*wY)
            c[1,0] = np.sum(wX*wY)
            c[1,1] = np.sum(wY*wY)
            values, vectors = eig(c)
            rXY[i,j] = min(values)
    
    #non-maximal supression
    #nms_rXY = maximum_filter(rXY, size =windowSize)
    plt.imshow(rXY)
    plt.show()
    nms_rXY = np.zeros((1,3))
    for i in range(sub, sz[0]-1-add ):
        for j in range(sub, sz[1]-1-add ):
            if windowSize%2 == 0:
                window = np.array(rXY[i-sub:i+add, j-sub:j+add])
            else:
                window = np.array(rXY[i-sub:i+add+1, j-sub:j+add+1])
            mx = np.max(window)
            if rXY[i,j] == mx :
                #temp = np.array([rXY[i,j], i, j])
                temp = np.array([[rXY[i,j], i, j]])
                nms_rXY = np.concatenate((nms_rXY, temp), axis=0)
    
    plt.imshow(nms_rXY)
    plt.show()
    
    lambdas = nms_rXY[:,0]
    sort = np.sort(lambdas, axis = 0)
    for c in range(0,nCorners):
        sz = np.shape(sort)
        lam = sort[sz[0]-c-1]
        where = np.where(lambdas == lam)
        i = where[0]
        corners[c,0] = nms_rXY[i,1]
        corners[c,1] = nms_rXY[i,2]
        lambdas[i] =0
    a=1
    return corners






def show_corners_result(imgs, corners):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax1.imshow(imgs[0], cmap='gray')
    ax1.scatter(corners[0][:, 0], corners[0][:, 1], s=35, edgecolors='r', facecolors='none')

    ax2 = fig.add_subplot(222)
    ax2.imshow(imgs[1], cmap='gray')
    ax2.scatter(corners[1][:, 0], corners[1][:, 1], s=35, edgecolors='r', facecolors='none')
    plt.show()







for smoothSTD in (0.5, 1, 2, 4):
    windowSize = int(smoothSTD * 6)
    if windowSize % 2 == 0:
        windowSize += 1
        
    print('smooth stdev: %r' % smoothSTD)
    print('window size: %r' % windowSize)

    nCorners = 20

    # read images and detect corners on images      
    
    imgs_din = []
    crns_din = []
    imgs_mat = []
    crns_mat = []
    imgs_war = []
    crns_war = []
    
    downSize = 10
    
    for i in range(2):
        img_din = imageio.imread('p4/dino/dino' + str(i) + '.png')
        #imgs_din.append(rgb2gray(img_din))
        # downsize your image in case corner_detect runs slow in test
        imgs_din.append(rgb2gray(img_din)[::downSize, ::downSize])
        crns_din.append(corner_detect(imgs_din[i], nCorners, smoothSTD, windowSize))

        img_mat = imageio.imread('p4/matrix/matrix' + str(i) + '.png')
        #imgs_mat.append(rgb2gray(img_mat))
        # downsize your image in case corner_detect runs slow in test
        imgs_mat.append(rgb2gray(img_mat)[::downSize, ::downSize])
        crns_mat.append(corner_detect(imgs_mat[i], nCorners, smoothSTD, windowSize))

        img_war = imageio.imread('p4/warrior/warrior' + str(i) + '.png')
        #imgs_war.append(rgb2gray(img_war))
        # downsize your image in case corner_detect runs slow in test
        imgs_war.append(rgb2gray(img_war)[::downSize, ::downSize])
        crns_war.append(corner_detect(imgs_war[i], nCorners, smoothSTD, windowSize))

    show_corners_result(imgs_din, crns_din)
    show_corners_result(imgs_mat, crns_mat)
    show_corners_result(imgs_war, crns_war)
    

