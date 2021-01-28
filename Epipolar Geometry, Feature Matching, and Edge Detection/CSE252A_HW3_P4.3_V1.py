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


def compute_fundamental(x1, x2):
    """ Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        
        Construct the A matrix according to lecture
        and solve the system of equations for the entries of the fundamental matrix.

        Returns:
        Fundamental Matrix (3x3)
    """
    
    """ ==========
    YOUR CODE HERE
    ========== """
    #x1 and x2 passed into here are (3,14) matrices, (3,M) actually
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
        
    # return your F matrix
    pass
    #need to find A to take SVD
    #Start w/ A1, take x1 to be x' and x2 to be x
    A = np.array([[ x1[0,0]*x2[0,0], x1[0,0]*x2[1,0], x1[0,0], x1[1,0]*x2[0,0], 
                    x1[1,0]*x2[1,0], x1[1,0], x2[0,0], x2[1,0], 1 ]])
    
    for i in range(1,x1.shape[1]):
        temp = np.array([[ x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i], x1[1,i]*x2[0,i], 
                x1[1,i]*x2[1,i], x1[1,i], x2[0,i], x2[1,i], 1 ]])
        A = np.concatenate((A,temp), axis=0)
    '''
    Atest = np.zeros((n,9))
    for i in range(n):
        Atest[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
    for i in range(14):
        for j in range(9):
            print(A[i,j]-Atest[i,j])
    '''
    u,s,v=svd(np.transpose(A)@A)
    F = v[-1]
    F = F.reshape(3,3)
    #ut, st, vt = svd(A)
    #Ft = vt[-1].reshape(3,3)
    
    #make rank 2, rank = number of non-zero singular values of F
    #since F has 3 singular values, make last one zero to make rank 2
    u, s, v = svd(F)
    s[2] = 0
    #now reconstruct F with the 2 remaingin non-zero svd's
    temp = np.zeros((3,3))
    temp[0,0], temp[1,1] = s[0],s[1]
    #b = np.matmul(np.diag(s),v)
    F = np.matmul( u, np.matmul(temp,v) )
    F = F/F[2,2]
    #nms_rXY = np.concatenate((nms_rXY, temp), axis=0)
    return F


def fundamental_matrix(x1,x2):
    # Normalization of the corner points is handled here
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
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

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = np.dot(T1.T,np.dot(F,T2))

    return F/F[2,2]

#cor1 = np.load("./p4/matrix/cor1_alt.npy")
#cor2 = np.load("./p4/matrix/cor2_alt.npy")

#cor3 = np.load("./p4/warrior/cor1.npy")
#cor4 = np.load("./p4/warrior/cor2.npy")

#test = fundamental_matrix(cor1,cor2)


from numpy.linalg import svd, eig
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
    
    """ ==========
    YOUR CODE HERE
    ========== """
    #use previously defined functions
    F = fundamental_matrix(cor1,cor2)
    e1, e2 = compute_epipole(F)
    
    
    
    #plt.plot(x[i:i+2], y[i:i+2], 'ro-')
    fig= plt.figure(figsize = (6,6))

    y_lim, x_lim, z_lim = img1.shape
    cor1 = np.transpose(cor1)
    for corner in cor1:
        i = corner
        #x = [e1[0], corner[0]]
        #y = [e1[1], corner[1]]
        slope = - (e1[1] - corner[1])/(e1[0] - corner[0])
        #xP = -1000
        #yP = xP*slope
        yT = e1[0]*slope + e1[1]
        x = [e1[0], 0]
        y = [e1[1], yT]
        
        
        yInt = corner[1] + slope*corner[0]
        yEnd = -slope*x_lim + yInt
        x = [0, x_lim]
        y = [yInt, yEnd]
        
        plt.plot(x, y, 'b-')
        plt.plot([corner[0]], [corner[1]], 'bo')
        #plt.imshow(img1)
        #plt.show()
        #plt.plot(x[e1[1]:corner[1]], y[e1[2]:corner[2]], 'ro-')
        
        a=1
    plt.imshow(img1)
    plt.xlim([0, x_lim])
    plt.ylim([y_lim, 0]) 
    plt.show()
    return e1, e2
    '''
    m,n = im.shape[:2]
    line = dot(F,x)
    # epipolar line parameter and values
    t = linspace(0,n,100)
    lt = array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt>=0) & (lt<m) 
    plot(t[ndx],lt[ndx],linewidth=2)
    
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')
    '''
    
    
    
    
# replace images and corners with those of matrix and warrior
'''
imgids = ["dino", "matrix", "warrior"]
for imgid in imgids:
    I1 = imageio.imread("./p4/"+imgid+"/"+imgid+"0.png")
    I2 = imageio.imread("./p4/"+imgid+"/"+imgid+"1.png")

    cor1 = np.load("./p4/"+imgid+"/cor1.npy")
    cor2 = np.load("./p4/"+imgid+"/cor2.npy")
    e1,e2 = plot_epipolar_lines(I1,I2,cor1,cor2)
    print(e1)
    print(e2)
'''
    
    
    
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
    
    
    rectified_im2 = im2
    
    
    new_cor2 = H2@points2
    
    #H1, H2 = compute_matching_homographies(e1, np.transpose(F), im1, points1.T, points2.T)
    rectified_im1 = im1
    new_cor1 = H1@points1
    
    return rectified_im1, rectified_im2, new_cor1, new_cor2


# Plot the parallel epipolar lines using plot_epipolar_lines
for subj in ('matrix', 'warrior'):
    I1 = imread("./p4/%s/%s0.png" % (subj, subj))
    I2 = imread("./p4/%s/%s1.png" % (subj, subj))

    cor1 = np.load("./p4/%s/cor1.npy" % subj)
    cor2 = np.load("./p4/%s/cor2.npy" % subj)

    rectified_im1, rectified_im2, new_cor1, new_cor2 = image_rectification(I1, I2, cor1, cor2)
    plot_epipolar_lines(rectified_im1, rectified_im2, new_cor1, new_cor2)