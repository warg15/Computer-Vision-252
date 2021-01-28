#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:22:35 2020

@author: femw90
"""

'''
Problem 2
'''

import numpy as np
import matplotlib.pyplot as plt
import math


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



# Change the three matrices for the four cases as described in the problem
# in the four camera functions geiven below. Make sure that we can see the formula
# (if one exists) being used to fill in the matrices. Feel free to document with
# comments any thing you feel the need to explain. 

def camera1():
    # write your code here
    '''
        Use transformation matrix as below
        [R t
         0 1]
        where R is 3x3 rotation matrix, t is 3x1 transformation matrix, 0 is 1x3 vector of 0's and 1 is a single 1 in the last cell.
    
        no rotation so R is 3x3 identity
        no translation so t is [0,0,0].T
    '''
    P_ext = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    
    '''
        intrinsic is given by: ([[f, 0, 0, 0],
                                 [0, f, 0, 0],
                                 [0, 0, 1, 0]])
        here f is focal length = 1
    '''
    f=1
    P_int_proj = np.array([[f, 0, 0, 0],
                           [0, f, 0, 0],
                           [0, 0, 1, 0]])
    return P_int_proj, P_ext

def camera2():
    # write your code here
    '''
        Use transformation matrix as below
        [R t
         0 1]
        where R is 3x3 rotation matrix, t is 3x1 transformation matrix, 0 is 1x3 vector of 0's and 1 is a single 1 in the last cell.
    
        no rotation so R is 3x3 identity
        translation is [1,-1,1].T, however since it is stated that the optical axis of the camera is aligned with the z-axis,
        the only way this is possible is if there is only z-axis translation. Therefore, translation is [0,0,1].T
    '''
    P_ext = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1]])
    
    '''
        intrinsic is given by: ([[f, 0, 0, 0],
                                 [0, f, 0, 0],
                                 [0, 0, 1, 0]])
        here f is focal length = 1
    '''
    f=1
    P_int_proj = np.array([[f, 0, 0, 0],
                           [0, f, 0, 0],
                           [0, 0, 1, 0]])
    
    return P_int_proj, P_ext

def camera3():
    # write your code here
    '''
        Use transformation matrix as below
        [R t
         0 1]
        where R is 3x3 rotation matrix, t is 3x1 transformation matrix, 0 is 1x3 vector of 0's and 1 is a single 1 in the last cell.
    
        rotation
        about x-axis: Rx = ([[1, 0, 0],
                             [0, cos(@), -sin(@)],
                             [0, sin(@), cos(@)]]), where @ is angle 20 degrees
                
        about y-axis: Ry = ([[cos(@), 0, sin(@)],
                             [0, 1, 0],
                             [-sin(@), 0, cos(@)]]) where @ is angle 45 degrees
        
        rotation is R = Rx*Ry (matrix mult), because the y rotation comes first in rotating from A to target B
        
        translation is [-1,0,1].T
    '''
    cos20 = np.cos(np.deg2rad(20))
    sin20 = np.sin(np.deg2rad(20))
    cos45 = np.cos(np.deg2rad(45))
    sin45 = np.sin(np.deg2rad(45))
    
    Rx = np.array([[1, 0, 0],
                   [0, cos20, -sin20],
                   [0, sin20, cos20]])
    Ry = np.array([[cos45, 0, sin45],
                    [0, 1, 0],
                    [-sin45, 0, cos45]])
    
    R = np.matmul(Rx, Ry)
    bOa = np.array([-1, 0, 1]).T
    P_ext = np.zeros((4,4))
    P_ext[0:3,0:3] = R
    P_ext[0:3, 3] = bOa
    P_ext[3,3] = 1
    
    
    
    '''
        intrinsic is given by: ([[f, 0, 0, 0],
                                 [0, f, 0, 0],
                                 [0, 0, 1, 0]])
        here f is focal length = 1
    '''
    f=1
    P_int_proj = np.array([[f, 0, 0, 0],
                           [0, f, 0, 0],
                           [0, 0, 1, 0]])

    return P_int_proj, P_ext

def camera4():    
    # write your code here
    '''
        Use transformation matrix as below
        [R t
         0 1]
        where R is 3x3 rotation matrix, t is 3x1 transformation matrix, 0 is 1x3 vector of 0's and 1 is a single 1 in the last cell.
    
        rotation
        about x-axis: Rx = ([[1, 0, 0],
                             [0, cos(@), -sin(@)],
                             [0, sin(@), cos(@)]]), where @ is angle 20 degrees
                
        about y-axis: Ry = ([[cos(@), 0, sin(@)],
                             [0, 1, 0],
                             [-sin(@), 0, cos(@)]]) where @ is angle 45 degrees
        
        rotation is R = Rx*Ry (matrix mult), because the y rotation comes first in rotating from A to target B
        
        translation is [-1,0,1].T
    '''
    cos20 = np.cos(np.deg2rad(20))
    sin20 = np.sin(np.deg2rad(20))
    cos45 = np.cos(np.deg2rad(45))
    sin45 = np.sin(np.deg2rad(45))
    
    Rx = np.array([[1, 0, 0],
                   [0, cos20, -sin20],
                   [0, sin20, cos20]])
    Ry = np.array([[cos45, 0, sin45],
                    [0, 1, 0],
                    [-sin45, 0, cos45]])
    
    R = np.matmul(Rx, Ry)
    bOa = np.array([-1, -1, 21]).T
    
    P_ext = np.zeros((4,4))
    P_ext[0:3,0:3] = R
    P_ext[0:3, 3] = bOa
    P_ext[3,3] = 1
    
    
    
    '''
        intrinsic is given by: ([[f, 0, 0, 0],
                                 [0, f, 0, 0],
                                 [0, 0, 1, 0]])
        here f is focal length = 1
    '''
    f=7
    P_int_proj = np.array([[f, 0, 0, 0],
                           [0, f, 0, 0],
                           [0, 0, 1, 0]])
    return P_int_proj, P_ext

# Use the following code to display your outputs
# You are free to change the axis parameters to better 
# display your quadrilateral but do not remove any annotations

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
        
def main():
    point1 = np.array([[-1,-.5,2]]).T
    point2 = np.array([[1,-.5,2]]).T
    point3 = np.array([[1,.5,2]]).T
    point4 = np.array([[-1,.5,2]]).T 
    points = np.hstack((point1,point2,point3,point4))
    
    for i, camera in enumerate([camera1, camera2, camera3, camera4]):
        P_int_proj, P_ext = camera()
        plt.subplot(2, 2, i+1)
        plot_points(project_points(P_int_proj, P_ext, points), title='Camera %d Projective'%(i+1), axis=[-0.6,2.5,-0.75,0.75])
        plt.show()

main()



'''
if __name__ == "__main__":
    point1 = np.array([[-1,-.5,2]]).T
    #homog = to_homog(point1)
    #euclid = from_homog(homog)
    
    
    point1 = np.array([[-1,-.5,2]]).T
    point2 = np.array([[1,-.5,2]]).T
    point3 = np.array([[1,.5,2]]).T
    point4 = np.array([[-1,.5,2]]).T 
    points = np.hstack((point1,point2,point3,point4))
    
    homog = to_homog(points)
    euclid = from_homog(homog)
'''

'''
    Use transformation matrix as below
    [R t
     0 1]
    where R is 3x3 rotation matrix, t is 3x1 transformation matrix, 0 is 1x3 vector of 0's and 1 is a single 1 in the last cell.
'''