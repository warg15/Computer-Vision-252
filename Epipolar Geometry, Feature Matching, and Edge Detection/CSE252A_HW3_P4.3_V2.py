#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:40:32 2020

@author: femw90
"""

import numpy as np
import imageio
from imageio import imread
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import svd, eig


import numpy as np
import matplotlib.pyplot as plt
import math
# convert points from euclidian to homogeneous
def to_homog(points):
    # write your code here
    check = np.shape(points)
    homog = np.ones((check[0]+1, check[1]))
    homog[0:check[0], 0:check[1]] = points[:,:]
    ones = np.ones(points.shape[1])
    homog = np.vstack((points, ones))
    return(homog)

# convert points from homogeneous to euclidian
def from_homog(points_homog):
    # write your code here
    check = np.shape(points_homog)
    last_row = points_homog[check[0]-1, :]
    euclid = points_homog/last_row[None,:]
    euclid = euclid[0:check[0]-1, :]
    points_homog /= points_homog[-1]
    points_homog = points_homog[:-1, :]
    #return(euclid)
    return(points_homog)

def compute_fundamental(x1, x2):
    """ Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Construct the A matrix according to lecture
        and solve the system of equations for the entries of the fundamental matrix.
        Returns:
        Fundamental Matrix (3x3)
    """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    A = np.array([[ x1[0,0]*x2[0,0], x1[0,0]*x2[1,0], x1[0,0], x1[1,0]*x2[0,0], 
                    x1[1,0]*x2[1,0], x1[1,0], x2[0,0], x2[1,0], 1 ]])
    
    for i in range(1,x1.shape[1]):
        temp = np.array([[ x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i], x1[1,i]*x2[0,i], 
                x1[1,i]*x2[1,i], x1[1,i], x2[0,i], x2[1,i], 1 ]])
        A = np.concatenate((A,temp), axis=0)
    u,s,v=svd(np.transpose(A)@A)
    F = v[-1]
    F = F.reshape(3,3)
    u, s, v = svd(F)
    s[2] = 0
    temp = np.zeros((3,3))
    temp[0,0], temp[1,1] = s[0],s[1]
    F = np.matmul( u, np.matmul(temp,v) )
    F = F/F[2,2]
    return F

def fundamental_matrix(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = np.dot(T1,x1)
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)
    F = compute_fundamental(x1,x2)
    F = np.dot(T1.T,np.dot(F,T2))
    return F/F[2,2]

def compute_epipole(F):
    """
    This function computes the epipoles for a given fundamental matrix.
    input:
      F --> fundamental matrix
    output:
      e1 --> corresponding epipole in image 1
      e2 --> epipole in image2
    """
    U,S,V = svd(F)
    e2 = V[-1]
    e2 = e2/e2[2]
    U,S,V = svd(np.transpose(F))
    e1 = V[-1]
    e1 = e1/e1[2]
    return e1, e2


def plot_epipolar_lines(img1, img2, cor1, cor2):
    """Plot epipolar lines on image given image, corners
    Args:
        img1: Image 1.
        img2: Image 2.
        cor1: Corners in homogeneous image coordinate in image 1 (3xn)
        cor2: Corners in homogeneous image coordinate in image 2 (3xn)
    """
    img1 = img1.astype(np.int)
    img2 = img2.astype(np.int)
    F = fundamental_matrix(cor1,cor2)
    e1, e2 = compute_epipole(F)
    fig= plt.figure(figsize = (6,6))
    yL, xL, trash = img1.shape
    cor1 = np.transpose(cor1)
    for corner in cor1:
        slope = - (e1[1] - corner[1])/(e1[0] - corner[0])
        yInt = corner[1] + slope*corner[0]
        yEnd = -slope*xL + yInt
        x = [0, xL]
        y = [yInt, yEnd]
        plt.plot(x, y, 'b-')
        plt.plot([corner[0]], [corner[1]], 'bo')
    plt.imshow(img1)
    plt.xlim([0, xL])
    plt.ylim([yL, 0]) 
    print('Image 1')
    plt.show() 
    
    fig= plt.figure(figsize = (6,6))
    yL, xL, trash = img2.shape
    cor2 = np.transpose(cor2)
    for corner in cor2:
        slope = - (e2[1] - corner[1])/(e2[0] - corner[0])
        
        yInt = corner[1] + slope*corner[0]
        yEnd = -slope*xL + yInt
        x = [0, xL]
        y = [yInt, yEnd]
        
        plt.plot(x, y, 'b-')
        plt.plot([corner[0]], [corner[1]], 'bo')
    plt.imshow(img2)
    plt.xlim([0, xL])
    plt.ylim([yL, 0]) 
    print('Image 2')
    plt.show() 
    return
    
    
    
def compute_matching_homographies(e2, F, im2, points1, points2):
    """This function computes the homographies to get the rectified images.
    
    input:
      e2 --> epipole in image 2
      F --> the fundamental matrix (think about what you should be passing: F or F.T!)
      im2 --> image2
      points1 --> corner points in image1
      points2 --> corresponding corner points in image2
      
    output:
      H1 --> homography for image 1
      H2 --> homography for image 2
    """
    # calculate H2
    width = im2.shape[1]
    height = im2.shape[0]

    T = np.identity(3)
    T[0][2] = -1.0 * width / 2
    T[1][2] = -1.0 * height / 2

    e = T.dot(e2)
    e1_prime = e[0]
    e2_prime = e[1]
    if e1_prime >= 0:
        alpha = 1.0
    else:
        alpha = -1.0

    R = np.identity(3)
    R[0][0] = alpha * e1_prime / np.sqrt(e1_prime**2 + e2_prime**2)
    R[0][1] = alpha * e2_prime / np.sqrt(e1_prime**2 + e2_prime**2)
    R[1][0] = - alpha * e2_prime / np.sqrt(e1_prime**2 + e2_prime**2)
    R[1][1] = alpha * e1_prime / np.sqrt(e1_prime**2 + e2_prime**2)

    f = R.dot(e)[0]
    G = np.identity(3)
    G[2][0] = - 1.0 / f

    H2 = np.linalg.inv(T).dot(G.dot(R.dot(T)))

    # calculate H1
    e_prime = np.zeros((3, 3))
    e_prime[0][1] = -e2[2]
    e_prime[0][2] = e2[1]
    e_prime[1][0] = e2[2]
    e_prime[1][2] = -e2[0]
    e_prime[2][0] = -e2[1]
    e_prime[2][1] = e2[0]

    v = np.array([1, 1, 1])
    M = e_prime.dot(F) + np.outer(e2, v)

    points1_hat = H2.dot(M.dot(points1.T)).T
    points2_hat = H2.dot(points2.T).T

    W = points1_hat / points1_hat[:, 2].reshape(-1, 1)
    b = (points2_hat / points2_hat[:, 2].reshape(-1, 1))[:, 0]

    # least square problem
    a1, a2, a3 = np.linalg.lstsq(W, b)[0]
    HA = np.identity(3)
    HA[0] = np.array([a1, a2, a3])

    H1 = HA.dot(H2).dot(M)
    return H1, H2

def warpWarped(img, H):
    sz = np.shape(img)
    target_img = 255*np.ones((sz[0], sz[1], 3))
    
    for u in range(sz[1]):
        for v in range(sz[0]):
            coords = np.array([[j], [i]])
            homogCoords = to_homog(coords)
            newhomogCoords =  np.matmul(H, homogCoords)
            newCoords = from_homog(newhomogCoords)
            
            #newCoords = from_homog(np.matmul(H, to_homog(np.array([[i], [j]]))))
            newCoords = np.rint(from_homog(np.matmul(H, to_homog(np.array([[j], [i]])))))
            
            #newCoords = np.rint(newCoords)
            
            xTarget, yTarget = int(newCoords[0]), int(newCoords[1])
            targetSizeX, targetSizeY = sz[0], sz[1]
            if(xTarget >= 0 and xTarget < targetSizeY and yTarget >= 0 and yTarget < targetSizeX):
                target_img[i, j, :] = img[yTarget, xTarget, :]
                #target_img = target_img.astype(np.int)
                #plt.imshow(target_img)
                #plt.show()
    target_img = target_img.astype(np.int)
    return target_img

def warp2(source_image, H):
    target_size = source_image.shape
    target_img = 255*np.ones(target_size)
    
    H_inv = np.linalg.inv(H)
    
    for i in range(target_size[0]):
        for j in range(target_size[1]):
            h_Mul = np.matmul(H_inv, to_homog(np.array([[i],[j]])))
            point = from_homog(h_Mul).T
            fin = list(map(int,np.floor(point)[0]))
            
            newCoords = np.rint(from_homog(np.matmul(np.linalg.inv(H), to_homog(np.array([[i], [j]])))))
            
            #newCoords = np.rint(newCoords)
            
            xTarget, yTarget = int(newCoords[0]), int(newCoords[1])
            
            if xTarget >= 0 and xTarget < source_image.shape[1] and yTarget>=0 and yTarget<source_image.shape[0]:
                target_img[j,i,:]=source_image[yTarget,xTarget,:]
    
    
    return target_img
    
def image_rectification(im1, im2, points1, points2):
    """This function provides the rectified images along with the new corner points as outputs for a given pair of 
    images with corner correspondences
    input:
    im1--> image1
    im2--> image2
    points1--> corner points in image1
    points2--> corner points in image2
    outpu:
    rectified_im1-->rectified image 1
    rectified_im2-->rectified image 2
    new_cor1--> new corners in the rectified image 1
    new_cor2--> new corners in the rectified image 2
    """
    
    """ ==========
    YOUR CODE HERE
    ========== """
    #need epipole and fundamental matrix
    F = fundamental_matrix(points1,points2)
    e1, e2 = compute_epipole(F)
    e1 = e1 / e1[-1]
    e2 = e2 / e2[-1]
    H1, H2 = compute_matching_homographies(e2, np.transpose(F), im2, points1.T, points2.T)
    
    check1 = H1@e1
    check2 = H2@e2
    
    rectified_im2 = im2
    rectified_im1 = im1
    
    new_cor1 = H1@points1
    new_cor2 = H2@points2
    
    #print(new_cor1)
    #print(new_cor2)
    new_cor1 = new_cor1/new_cor1[-1]
    new_cor2 = new_cor2/new_cor2[-1]
    #print(new_cor1)
    #print(new_cor2)
    
    #sz = np.shape(im1)
    #imCorners = np.ones((3,4))
    #imCorners = np.array([ [0, 0, sz[0], sz[0]], [0, sz[1], 0, sz[1]], [1,1,1,1] ])
    #convertCorners = from_homog(H1@imCorners)
    #targetSizeX, targetSizeY = target_size[0], target_size[1]
    #target_img = np.ones((targetSizeX,targetSizeY,3))
    rectified_im1 = warp2(im1, H1)
    rectified_im2 = warp2(im2, H2)
    
    
    
    return rectified_im1, rectified_im2, new_cor1, new_cor2

'''
# Plot the parallel epipolar lines using plot_epipolar_lines
for subj in ('matrix', 'warrior'):
    I1 = imread("./p4/%s/%s0.png" % (subj, subj))
    I2 = imread("./p4/%s/%s1.png" % (subj, subj))

    cor1 = np.load("./p4/%s/cor1.npy" % subj)
    cor2 = np.load("./p4/%s/cor2.npy" % subj)

    rectified_im1, rectified_im2, new_cor1, new_cor2 = image_rectification(I1, I2, cor1, cor2)
    plot_epipolar_lines(rectified_im1, rectified_im2, new_cor1, new_cor2)
'''
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
    
def display_correspondence(img1, img2, corrs):
    """Plot matching result on image pair given images and correspondences

    Args:
        img1: Image 1.
        img2: Image 2.
        corrs: Corner correspondence
    """
    
    """ ==========
    YOUR CODE HERE
    You may want to refer to the `show_matching_result` function.
    ========== """
    show_matching_result(img1, img2, corrs)

def correspondence_matching_epipole(img1, img2, corners1, F, R, NCCth):
    """Find corner correspondence along epipolar line.

    Args:
        img1: Image 1.
        img2: Image 2.
        corners1: Detected corners in image 1.
        F: Fundamental matrix calculated using given ground truth corner correspondences.
        R: NCC matching window radius.
        NCCth: NCC matching threshold.
    
    Returns:
        Matching result to be used in display_correspondence function
    """
    
    """ ==========
    YOUR CODE HERE
    ========== """
    matching = []
    for corner in corners1:
        vertConst = corner[0]
        currentScore = 0
        for j in range( R, np.shape(img2)[1]-R):
            checkPoint = np.zeros_like(corner)
            checkPoint[0], checkPoint[1] = j,vertConst
            newScore = ncc_match(img1, img2, corner, checkPoint, R)
            if newScore > currentScore:
                currentScore = newScore
                matchTup = (corner, checkPoint)
        if currentScore >= NCCth:
            matching.append(matchTup)
        a=1
    print(matching)
    return matching


I1 = imageio.imread("./p4/matrix/matrix0.png")
I2 = imageio.imread("./p4/matrix/matrix1.png")
cor1 = np.load("./p4/matrix/cor1_alt.npy")
cor2 = np.load("./p4/matrix/cor2_alt.npy")
I3 = imageio.imread("./p4/warrior/warrior0.png")
I4 = imageio.imread("./p4/warrior/warrior1.png")
cor3 = np.load("./p4/warrior/cor1.npy")
cor4 = np.load("./p4/warrior/cor2.npy")
'''
downSize = 1

cor1 = np.round(cor1/downSize)
cor2 = np.round(cor2/downSize)
cor3 = np.round(cor3/downSize)
cor4 = np.round(cor4/downSize)

I1 = I1[::downSize, ::downSize]
I2 = I2[::downSize, ::downSize]
I3 = I3[::downSize, ::downSize]
I4 = I4[::downSize, ::downSize]
'''
# For matrix
rectified_im1, rectified_im2, new_cor1, new_cor2 = image_rectification(I1, I2, cor1, cor2)
F_new = fundamental_matrix(new_cor1, new_cor2)

# replace black pixels with white pixels
_black_idxs = (rectified_im1[:, :, 0] == 0) & (rectified_im1[:, :, 1] == 0) & (rectified_im1[:, :, 2] == 0)
rectified_im1[:, :][_black_idxs] = [1.0, 1.0, 1.0]
_black_idxs = (rectified_im2[:, :, 0] == 0) & (rectified_im2[:, :, 1] == 0) & (rectified_im2[:, :, 2] == 0)
rectified_im2[:, :][_black_idxs] = [1.0, 1.0, 1.0]

nCorners = 10

# Choose your threshold and NCC matching window radius
NCCth = 0.6
R = 50

# detect corners using corner detector here, store in corners1
corners1 = corner_detect(rgb2gray(rectified_im1), nCorners, smoothSTD, windowSize)
corrs = correspondence_matching_epipole(rectified_im1, rectified_im2, corners1, F_new, R, NCCth)
display_correspondence(rectified_im1, rectified_im2, corrs)





# For warrior
rectified_im3, rectified_im4, new_cor3, new_cor4 = image_rectification(I3, I4, cor3, cor4)
F_new2 = fundamental_matrix(new_cor3, new_cor4)

# replace black pixels with white pixels
_black_idxs = (rectified_im3[:, :, 0] == 0) & (rectified_im3[:, :, 1] == 0) & (rectified_im3[:, :, 2] == 0)
rectified_im3[:, :][_black_idxs] = [1.0, 1.0, 1.0]
_black_idxs = (rectified_im4[:, :, 0] == 0) & (rectified_im4[:, :, 1] == 0) & (rectified_im4[:, :, 2] == 0)
rectified_im4[:, :][_black_idxs] = [1.0, 1.0, 1.0]

# You may wish to change your NCCth and R for warrior here.
NCCth = 0.6
R = 150

corners2 = corner_detect(rgb2gray(rectified_im3), nCorners, smoothSTD, windowSize)
corrs = correspondence_matching_epipole(rectified_im3, rectified_im4, corners2, F_new2, R, NCCth)
display_correspondence(rectified_im3, rectified_im4, corrs)

