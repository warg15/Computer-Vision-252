#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:16:09 2020

@author: femw90
"""

"""
Created on Tue Oct 27 15:22:35 2020

@author: femw90
"""

'''
Problem 3
'''

import numpy as np
import matplotlib.pyplot as plt
import math


import numpy as np
from PIL import Image

#from ECE252A_HW1_P1 import plot_points
# convert points from euclidian to homogeneous
def to_homog(points):
    check = np.shape(points)
    homog = np.ones((check[0]+1, check[1]))
    homog[0:check[0], 0:check[1]] = points[:,:]
    '''
    homog = np.array([ [0, 0, 0]]).T
    if check == 2:
        x=points[0,0]
        y=points[1,0]
        homog = np.array([ [x, y, 1]]).T
    elif check == 3:
        x=points[0,0]
        y=points[1,0]
        z=points[2,0]
        homog = np.array([ [x, y, z,1]]).T
    else: print("Error with \n", points, "\n in to_homog")
    '''
    return(homog)
    # write your code here
    
    
# convert points from homogeneous to euclidian
def from_homog(points_homog):
    # write your code here
    check = np.shape(points_homog)
    last_row = points_homog[check[0]-1, :]
    euclid = points_homog/last_row[None,:]
    euclid = euclid[0:check[0]-1, :]
    '''
    euclidian = np.array([ [0, 0, 0,0]]).T
    if check == 3:
        x=points_homog[0,0]/points_homog[2,0]
        y=points_homog[1,0]/points_homog[2,0]
        euclidian = np.array([ [x, y]]).T
    elif check == 4:
        x=points_homog[0,0]/points_homog[3,0]
        y=points_homog[1,0]/points_homog[3,0]
        z=points_homog[2,0]/points_homog[3,0]
        euclidian = np.array([ [x, y, z]]).T
    else: print("Error with \n", points_homog, "\n in from_homog")
    '''
    return(euclid)
    
    
# project 3D euclidian points to 2D euclidian
def project_points(P_int, P_ext, pts):
    # write your code here
    pts_homog = to_homog(pts)
    
    pts_final = np.matmul( P_int, np.matmul(P_ext, pts_homog))
    
    return from_homog(pts_final)
def plot_points(points, title='', style='.-r', axis=[]):
    inds = list(range(points.shape[1]))+[0]
    plt.plot(points[0,inds], points[1,inds],style)
    
    for i in range(len(points[0,inds])):
        plt.annotate(str("{0:.3f}".format(points[0,inds][i]))+","+str("{0:.3f}".format(points[1,inds][i])),(points[0,inds][i], points[1,inds][i]))
    
    if title:
        plt.title(title)
    if axis:
        plt.axis(axis)
        
    plt.tight_layout()
################

import matplotlib.pyplot as plt

# load image to be used - resize to make sure it's not too large
# You can use the given image as well
# A large image will make testing you code take longer; once you're satisfied with your result,
# you can, if you wish to, make the image larger (or till your computer memory allows you to)

source_image = np.array(Image.open("photo.jpg"))/255

#code to easily downsize without having to change any parameters other than "downsize"
from skimage.transform import rescale
downSize = 10
source_image = rescale(source_image, 1/downSize, anti_aliasing=False)



# display images
plt.imshow(source_image)

# Align the polygon such that the corners align with the document in your picture
# This polygon doesn't need to overlap with the edges perfectly, an approximation is fine
# The order of points is clockwise, starting from bottom left.

#Wrong x_coords = [1493,1154,122,211]  #assumes that X is vertical, since its (x,y)/(vertical/horoz)
#Wrong y_coords = [1944,102,511,1941]  #assumes that y is horoz
#correct for nomal sized y_coords = [1493,1154,122,211]
# x_coords = [1944,102,511,1941]

#resized coords
y_coords = [int(1493/downSize),int(1154/downSize),int(122/downSize),int(211/downSize)]
x_coords = [int(1944/downSize),int(102/downSize),int(511/downSize),int(1941/downSize)]


# Plot points from the previous problem is used to draw over your image 
# Note that your coordinates will change once you resize your image again
source_points = np.vstack((x_coords, y_coords))
plot_points(source_points)

plt.show()
print (source_image.shape)


def computeH(source_points, target_points):
    # returns the 3x3 homography matrix such that:
    # np.matmul(H, source_points) = target_points
    # where source_points and target_points are expected to be in homogeneous
    # make sure points are 3D homogeneous
    assert source_points.shape[0]==3 and target_points.shape[0]==3
    #Your code goes here
    
    #insert image of A here
    #see image inserted for what makes up the matrix A
    x1S,x2S,x3S,x4S = source_points[0,0],source_points[0,1],source_points[0,2],source_points[0,3]
    y1S,y2S,y3S,y4S = source_points[1,0],source_points[1,1],source_points[1,2],source_points[1,3]
    #source points ^^     
    x1T,x2T,x3T,x4T = target_points[0,0],target_points[0,1],target_points[0,2],target_points[0,3]
    y1T,y2T,y3T,y4T = target_points[1,0],target_points[1,1],target_points[1,2],target_points[1,3]
    #target points ^^     
    A = np.array([[0, 0, 0, -x1S, -y1S, -1, y1T*x1S, y1T*y1S, y1T],
                    [x1S, y1S, 1, 0, 0, 0, -x1T*x1S, -x1T*y1S, -x1T],
                    [0, 0, 0, -x2S, -y2S, -1, y2T*x2S, y2T*y2S, y2T],
                    [x2S, y2S, 1, 0, 0, 0, -x2T*x2S, -x2T*y2S, -x2T],
                    [0, 0, 0, -x3S, -y3S, -1, y3T*x3S, y3T*y3S, y3T],
                    [x3S, y3S, 1, 0, 0, 0, -x3T*x3S, -x3T*y3S, -x3T],
                    [0, 0, 0, -x4S, -y4S, -1, y4T*x4S, y4T*y4S, y4T],
                    [x4S, y4S, 1, 0, 0, 0, -x4T*x4S, -x4T*y4S, -x4T]])
    
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    
    vh = np.transpose(vh)
    
    H_mtx = np.zeros((3,3)) #Fill in the H_mtx with appropriate values.
    H_mtx[0,0:3] = vh[0:3,8]
    H_mtx[1,0:3] = vh[3:6,8]
    H_mtx[2,0:3] = vh[6:9,8]
    return  H_mtx
#######################################################
# test code. Do not modify
#######################################################
def test_computeH():
    source_points = np.array([[0,0.5],[1,0.5],[1,1.5],[0,1.5]]).T
    target_points = np.array([[0,0],[1,0],[2,1],[-1,1]]).T
    H = computeH(to_homog(source_points), to_homog(target_points))
    mapped_points = from_homog(np.matmul(H,to_homog(source_points)))
    print (from_homog(np.matmul(H,to_homog(source_points[:,1].reshape(2,1)))))

    plot_points(source_points,style='.-k')
    plot_points(target_points,style='*-b')
    plot_points(mapped_points,style='.:r')
    plt.show()
    print('The red and blue quadrilaterals should overlap if ComputeH is implemented correctly.')
#test_computeH()

def warp(source_img, source_points, target_size):
    
    #target image must be bigger than source image
    
    # Create a target image and select target points to create a homography from source image to target image,
    # in other words map all source points to target points and then create
    # a warped version of the image based on the homography by filling in the target image.
    # Make sure the new image (of size target_size) has the same number of color channels as source image
    assert target_size[2]==source_img.shape[2]
    #Your code goes here
    targetSizeX, targetSizeY = target_size[0], target_size[1]
    target_img = np.zeros((targetSizeX,targetSizeY,3))
    target_points = np.array([[0,targetSizeX-1],[0,0],[targetSizeY-1,0],[targetSizeY-1,targetSizeX-1]]).T
    H_mtx = computeH(to_homog(source_points), to_homog(target_points))
    
    
    for i in range(0, source_img.shape[1]):
        #print(i)
        for j in range(0, source_img.shape[0]):
            tempCoords = to_homog(np.array([[i,j]]).T)
            #tempCoords = to_homog(np.array([[194,21]]).T)
            targetCoords = np.matmul(H_mtx, tempCoords)
            targetCoords = from_homog(targetCoords)
            targetCoords = np.rint(targetCoords)
            xTarget = int(targetCoords[0])
            yTarget = int(targetCoords[1])
            if(yTarget >= 0 and yTarget < targetSizeX and xTarget >= 0 and xTarget < targetSizeY):
                target_img[yTarget, xTarget, :] = source_img[j, i, :]
            #target_img[xTarget, yTarget, :] = source_img[i, j, :]
    
    
    
    return target_img
'''

# Use the code below to plot your result
result = warp(source_image, source_points, (int(2200/downSize),int(1700/downSize),3)) #Choose appropriate target size

plt.subplot(1, 2, 1)
plt.imshow(source_image)
plt.subplot(1, 2, 2)
plt.imsave("myop.png",result)
plt.imshow(result)
plt.show()
'''

def warp2(source_img, source_points, target_size):
    # Create a target image and select target points to create a homography from target image to source image,
    # in other words map each target point to a source point, and then create a warped version
    # of the image based on the homography by filling in the target image.
    # Make sure the new image (of size target_size) has the same number of color channels as source image
    
    #Your code goes here
    targetSizeX, targetSizeY = target_size[0], target_size[1]
    target_img = np.zeros((targetSizeX,targetSizeY,3))
    target_points = np.array([[0,targetSizeX-1],[0,0],[targetSizeY-1,0],[targetSizeY-1,targetSizeX-1]]).T
    H_mtx = computeH(to_homog(target_points), to_homog(source_points))
    
    for i in range(0, targetSizeY):
        #print(i)
        for j in range(0, targetSizeX):
            tempCoords = to_homog(np.array([[i,j]]).T)
            sourceCoords = np.matmul(H_mtx, tempCoords)
            sourceCoords = from_homog(sourceCoords)
            sourceCoords = np.rint(sourceCoords)
            xSource = int(sourceCoords[0])
            ySource = int(sourceCoords[1])
            #if(yTarget >= 0 and yTarget < targetSizeX and xTarget >= 0 and xTarget < targetSizeY):
            target_img[j, i, :] = source_img[ySource, xSource, :]
            #source_img[ySource, xSource, :] = np.array([[0, 0, 0]])
    
    return target_img
'''
# Use the code below to plot your result
result = warp2(source_image, source_points, (int(2200/downSize),int(1700/downSize),3)) #Choose appropriate size
plt.subplot(1, 2, 1)
plt.imshow(source_image)
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.imsave("warp2.png",result)
plt.show()
'''


# Load the supplied source and target images here

source_image = np.array(Image.open("bear.png"))/255
target_image = np.array(Image.open("gallery.png"))/255

#code to easily downsize without having to change any parameters other than "downSize"
from skimage.transform import rescale
import cv2
downSize = 2
source_image = rescale(source_image, 1/downSize, anti_aliasing=False)
target_image = rescale(target_image, 1/downSize, anti_aliasing=False)

# display images
plt.imshow(source_image)

plt.show()
print (source_image.shape)

# display images
plt.imshow(target_image)

# Align the polygon such that the corners align with the document in your picture
# This polygon doesn't need to overlap with the edges perfectly, an approximation is fine
# The order of points is clockwise, starting from bottom left.
y_coords_frame = [int(2600/downSize),int(100/downSize),int(355/downSize),int(2600/downSize)]
x_coords_frame = [int(90/downSize),int(100/downSize),int(2000/downSize),int(1975/downSize)]

y_coords_ipad = [int(1290/downSize),int(760/downSize),int(820/downSize),int(1260/downSize)]
x_coords_ipad = [int(3080/downSize),int(3050/downSize),int(3480/downSize),int(3485/downSize)]


# Plot points from the previous problem is used to draw over your image 
# Note that your coordinates will change once you resize your image again
target_frame_points = np.vstack((x_coords_frame, y_coords_frame))
plot_points(target_frame_points)

target_ipad_points = np.vstack((x_coords_ipad, y_coords_ipad))
plot_points(target_ipad_points)

plt.show()
print (target_image.shape)

target_points = np.vstack((x_coords_frame, y_coords_frame, x_coords_ipad, y_coords_ipad))




def warp3(target_image, target_points, source_image):
    #Your code goes here
    #First the Frame
    frame_target_points = target_points[0:2,0:4]
    source_pointsX, source_pointsY = np.shape(source_image)[0], np.shape(source_image)[1]
    source_points = np.array([[0,source_pointsX-1],[0,0],[source_pointsY-1,0],[source_pointsY-1,source_pointsX-1]]).T
    H_mtx = computeH(to_homog(source_points), to_homog(frame_target_points))
    
    for i in range(0, source_pointsY):
        print(i)
        for j in range(0, source_pointsX):
            tempCoords = to_homog(np.array([[i,j]]).T)
            targetCoords = np.matmul(H_mtx, tempCoords)
            targetCoords = from_homog(targetCoords)
            targetCoords = np.rint(targetCoords)
            xTarget = int(targetCoords[0])
            yTarget = int(targetCoords[1])
            #if(yTarget >= 0 and yTarget < targetSizeX and xTarget >= 0 and xTarget < targetSizeY):
            target_image[yTarget, xTarget, :] = source_image[j, i, :]
            #source_img[ySource, xSource, :] = np.array([[0, 0, 0]])
    
    frame_target_points = target_points[2:4,0:4]
    source_pointsX, source_pointsY = np.shape(source_image)[0], np.shape(source_image)[1]
    source_points = np.array([[0,source_pointsX-1],[0,0],[source_pointsY-1,0],[source_pointsY-1,source_pointsX-1]]).T
    H_mtx = computeH(to_homog(source_points), to_homog(frame_target_points))
    
    for i in range(0, source_pointsY):
        print(i)
        for j in range(0, source_pointsX):
            tempCoords = to_homog(np.array([[i,j]]).T)
            targetCoords = np.matmul(H_mtx, tempCoords)
            targetCoords = from_homog(targetCoords)
            targetCoords = np.rint(targetCoords)
            xTarget = int(targetCoords[0])
            yTarget = int(targetCoords[1])
            #if(yTarget >= 0 and yTarget < targetSizeX and xTarget >= 0 and xTarget < targetSizeY):
            target_image[yTarget, xTarget, :] = source_image[j, i, :]
            #source_img[ySource, xSource, :] = np.array([[0, 0, 0]])
    
    return target_image


# Use the code below to plot your result
result1 = warp3(target_image, target_points, source_image)
plt.subplot(1, 2, 1)
plt.imshow(source_image)
plt.subplot(1, 2, 2)
plt.imshow(result1)
plt.imsave("warp3.png",result1)
plt.show()










