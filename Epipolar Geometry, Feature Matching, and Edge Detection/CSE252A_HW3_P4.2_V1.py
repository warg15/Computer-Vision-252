#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:03:34 2020

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
    corners = np.zeros((nCorners, 2))
    rXY = np.zeros_like(image)
    smoothedImage = gaussian_filter(image, smoothSTD)
    gX, gY = gradient(smoothedImage)
    sz = np.shape(smoothedImage)
    sub = windowSize//2
    add = windowSize//2
    
    for i in range(sub, sz[0]-1-add ):
        for j in range(sub, sz[1]-1-add ):
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
    nms_rXY = np.zeros_like(rXY)
    plt.imshow(rXY)
    plt.show()
    for i in range(sub, sz[0]-1-add ):
        for j in range(sub, sz[1]-1-add ):
            if windowSize%2 == 0:
                window = np.array(rXY[i-sub:i+add, j-sub:j+add])
            else:
                window = np.array(rXY[i-sub:i+add+1, j-sub:j+add+1])
            mx = np.max(window)
            if rXY[i,j] == mx :
                nms_rXY[i,j] = rXY[i,j]
    plt.imshow(nms_rXY)
    plt.show()
    for c in range(0,nCorners):
        result = np.where(nms_rXY == np.amax(nms_rXY))
        i = result[0][0]
        j = result[1][0]
        corners[c,0]=j
        corners[c,1]=i
        nms_rXY[i,j] =0
    a=1
    return corners
def ncc_match(img1, img2, c1, c2, R):
    """Compute NCC given two windows.
    Args:
        img1: Image 1.
        img2: Image 2.
        c1: Center (in image coordinate) of the window in image 1.
        c2: Center (in image coordinate) of the window in image 2.
        R: R is the radius of the patch, 2 * R + 1 is the window size
    Returns:
        NCC matching score for two input windows.
    """
    matching_score = 0
    w1 = img1[ int(c1[1])-R:int(c1[1])+R+1, int(c1[0])-R:int(c1[0])+R+1 ]
    w2 = img2[ int(c2[1])-R:int(c2[1])+R+1, int(c2[0])-R:int(c2[0])+R+1 ]
    n1 = np.shape(w1)[0]*np.shape(w1)[1]
    n2 = np.shape(w2)[0]*np.shape(w2)[1]
    w1Bar = (np.sum(w1))/(n1)
    w2Bar = (np.sum(w2))/(n2)
    w1Sig = np.sqrt((np.sum( (w1-w1Bar)*(w1-w1Bar) ))/n1)
    w2Sig = np.sqrt((np.sum( (w2-w2Bar)*(w2-w2Bar) ))/n2)
    w1Sub = w1 - w1Bar
    w2Sub = w2 - w2Bar
    a=np.sum(w1Sub*w2Sub)/n1
    matching_score = a/(w1Sig*w2Sig)
    return matching_score



def naive_matching(img1, img2, corners1, corners2, R, NCCth):
    """Compute NCC given two windows.

    Args:
        img1: Image 1.
        img2: Image 2.
        corners1: Corners in image 1 (nx2)
        corners2: Corners in image 2 (nx2)
        R: NCC matching radius
        NCCth: NCC matching score threshold

    Returns:
        NCC matching result a list of tuple (c1, c2), 
        c1 is the 1x2 corner location in image 1, 
        c2 is the 1x2 corner location in image 2. 

    """
    
    """ ==========
    YOUR CODE HERE
    ========== """
    matching = []
    sz1 = np.shape(corners1)
    sz2 = np.shape(corners2)
    
    for i in range(0,sz1[0]):
        currentScore = 0
        
        for j in range(0,sz2[0]):
            newScore = ncc_match(img1, img2, corners1[i], corners2[j], R)
            if newScore > currentScore:
                currentScore = newScore
                matchTup = (corners1[i], corners2[j])
        if currentScore >= NCCth:
            matching.append(matchTup)
    
    
    
    return matching


# detect corners on warrior and matrix sets
# you are free to modify code here, create your helper functions, etc.

nCorners = 20  # do this for 10, 20 and 30 corners
smoothSTD = 1
windowSize = 17

downSize = 5

# read images and detect corners on images

imgs_mat = []
crns_mat = []
imgs_war = []
crns_war = []

for i in range(2):
    img_mat = imageio.imread('p4/matrix/matrix' + str(i) + '.png')
    #imgs_mat.append(rgb2gray(img_mat))
    # downsize your image in case corner_detect runs slow in test
    imgs_mat.append(rgb2gray(img_mat)[::downSize, ::downSize])
    crns_mat.append(corner_detect(imgs_mat[i], nCorners, smoothSTD, windowSize))
    
    img_war = imageio.imread('p4/warrior/warrior' + str(i) + '.png')
    #imgs_war.append(rgb2gray(img_war))
    imgs_war.append(rgb2gray(img_war)[::downSize, ::downSize])
    crns_war.append(corner_detect(imgs_war[i], nCorners, smoothSTD, windowSize))
    

# match corners
R = 12
NCCth = 0.6  # put your threshold here
matching_mat = naive_matching(imgs_mat[0]/255, imgs_mat[1]/255, crns_mat[0], crns_mat[1], R, NCCth)
matching_war = naive_matching(imgs_war[0]/255, imgs_war[1]/255, crns_war[0], crns_war[1], R, NCCth)

# plot matching result
def show_matching_result(img1, img2, matching):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(np.hstack((img1, img2)), cmap='gray') # two dino images are of different sizes, resize one before use
    for p1, p2 in matching:
        plt.scatter(p1[0], p1[1], s=35, edgecolors='r', facecolors='none')
        plt.scatter(p2[0] + img1.shape[1], p2[1], s=35, edgecolors='r', facecolors='none')
        plt.plot([p1[0], p2[0] + img1.shape[1]], [p1[1], p2[1]])
    plt.savefig('dino_matching.png')
    plt.show()

print("Number of Corners:", nCorners)
show_matching_result(imgs_mat[0], imgs_mat[1], matching_mat)
show_matching_result(imgs_war[0], imgs_war[1], matching_war)




'''
# test NCC match
img1 = np.array([[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 4]])
img2 = np.array([[1, 2, 1, 3], [6, 5, 4, 4], [9, 8, 7, 3]])

print (ncc_match(img1, img2, np.array([1, 1]), np.array([1, 1]), 1))
# should print 0.8546

print (ncc_match(img1, img2, np.array([2, 1]), np.array([2, 1]), 1))
# should print 0.8457

print (ncc_match(img1, img2, np.array([1, 1]), np.array([2, 1]), 1))
# should print 0.6258
'''

