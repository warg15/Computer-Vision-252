#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:49:59 2020

@author: femw90
"""

## Example: How to read and access data from a pickle
import pickle
import numpy as np
from time import time
from skimage import io
#%matplotlib inline
import matplotlib.pyplot as plt
import math

pickle_in = open('synthetic_data.pickle', 'rb')
data = pickle.load(pickle_in, encoding='latin1')
'''
# data is a dict which stores each element as a key-value pair. 
print('Keys: ' + str(data.keys()))

# To access the value of an entity, refer it by its key.
print('Image:')
plt.imshow(data['im1'], cmap = 'gray')
plt.show()

print('Light source direction: ' + str(data['l1']))
plt.imshow(data['im2'], cmap = 'gray')
plt.show()

print('Light source direction: ' + str(data['l2']))

plt.imshow(data['im3'], cmap = 'gray')
plt.show()

print('Light source direction: ' + str(data['l3']))

plt.imshow(data['im4'], cmap = 'gray')
plt.show()

print('Light source direction: ' + str(data['l4']))
'''



from scipy.signal import convolve
from numpy import linalg

def horn_integrate(gx, gy, mask, niter):
    """ horn_integrate recovers the function g from its partial derivatives gx and gy. 
        mask is a binary image which tells which pixels are involved in integration. 
        niter is the number of iterations (typically 100,000 or 200,000, 
        although the trend can be seen even after 1000 iterations).
    """    
    g = np.ones(np.shape(gx))
    
    gx = np.multiply(gx, mask)
    gy = np.multiply(gy, mask)
    
    A = np.array([[0,1,0],[0,0,0],[0,0,0]]) #y-1
    B = np.array([[0,0,0],[1,0,0],[0,0,0]]) #x-1
    C = np.array([[0,0,0],[0,0,1],[0,0,0]]) #x+1
    D = np.array([[0,0,0],[0,0,0],[0,1,0]]) #y+1
    
    d_mask = A + B + C + D
    
    den = np.multiply(convolve(mask, d_mask, mode='same'), mask)
    den[den == 0] = 1
    rden = 1.0 / den
    mask2 = np.multiply(rden, mask)
    
    m_a = convolve(mask, A, mode='same')
    m_b = convolve(mask, B, mode='same')
    m_c = convolve(mask, C, mode='same')
    m_d = convolve(mask, D, mode='same')
    
    term_right = np.multiply(m_c, gx) + np.multiply(m_d, gy)
    t_a = -1.0 * convolve(gx, B, mode='same')
    t_b = -1.0 * convolve(gy, A, mode='same')
    term_right = term_right + t_a + t_b
    term_right = np.multiply(mask2, term_right)
    
    for k in range(niter):
        g = np.multiply(mask2, convolve(g, d_mask, mode='same')) + term_right
    
    return g




def photometric_stereo(images, lights, mask, horn_niter=25000):

    """ ==========YOUR CODE HERE========== """
    '''Version to handle 4 images and lights'''
    #images = images/255
    #lightsInv = np.linalg.inv(lights.T)
    sz = np.shape(images)
    eMatrix = np.zeros((sz[1]*sz[2], sz[0]))
    for i in range(0,sz[0]):
        eMatrix[:,i] = np.reshape(images[i,:,:], (sz[1]*sz[2],1))[:,0]
    eMatrix = eMatrix.T
    s = lights
    sFind = np.matmul(np.linalg.inv(np.matmul(s.T, s)), s.T)
    
    B = np.matmul(sFind, eMatrix).T
    
    
    
    
    # note:
    # images : (n_ims, h, w) 
    # lights : (n_ims, 3)
    albedo = np.ones(images[0].shape)
    normals = np.dstack((np.zeros(images[0].shape),
                         np.zeros(images[0].shape),
                         np.ones(images[0].shape)))
    
    albedoIntermediate = np.reshape(albedo, (sz[1]*sz[2],1))
    albedoIntermediate[:,0] = B[:,0]*B[:,0] + B[:,1]*B[:,1] + B[:,2]*B[:,2]
    albedoVect = np.sqrt(albedoIntermediate)
    albedo = np.reshape(albedoVect, (sz[1],sz[2]))
    
    normals[:,:,0] = np.reshape(B[:,0]/albedoVect[:,0], (sz[1], sz[2]))
    normals[:,:,1] = np.reshape(B[:,1]/albedoVect[:,0], (sz[1], sz[2]))
    normals[:,:,2] = np.reshape(B[:,2]/albedoVect[:,0], (sz[1], sz[2]))
    
    #slant  = np.zeros((sz[1],sz[2]))
    slant = (-normals[:,:,0]/normals[:,:,2])*mask[:,:]
    tilt = (-normals[:,:,1]/normals[:,:,2])*mask[:,:]
    
    H = np.ones(images[0].shape)
    
    H[0,0] = slant[0,0]
    for j in range(1,sz[2]):
        H[0,j] = H[0,j-1] - slant[0,j]
    
    for i in range(1,sz[1]):
        for j in range(0, sz[2]):
            H[i,j] = H[i-1,j] - tilt[i,j]
    
    H_horn = np.ones(images[0].shape)
    H_horn = horn_integrate(-slant, -tilt, mask, horn_niter)
    return albedo, normals, H, H_horn




from mpl_toolkits.mplot3d import Axes3D

pickle_in = open('synthetic_data.pickle', 'rb')
data = pickle.load(pickle_in, encoding='latin1')

lights = np.vstack((data['l1'], data['l2'], data['l3'], data['l4'])) #

images = []
images.append(data['im1'])
images.append(data['im2'])
images.append(data['im3'])
images.append(data['im4'])
images = np.array(images)

mask = np.ones(data['im1'].shape)

albedo, normals, depth, horn = photometric_stereo(images, lights, mask)

'''
My test section

sz = np.shape(images)
eMatrix = np.zeros((sz[1]*sz[2], sz[0]))
for i in range(0,sz[0]):
    eMatrix[:,i] = np.reshape(images[i,:,:], (sz[1]*sz[2],1))[:,0]
ogMatrix = np.zeros((sz[0], sz[1], sz[2]))
for i in range(0,sz[0]):
    ogMatrix[i, :, :] = np.reshape(eMatrix[:,i], (sz[1], sz[2]))
vectIM1 = np.reshape(images[0,:,:], (sz[1]*sz[2],1))
origIM1 = np.reshape(vectIM1, (sz[1], sz[2]))
ogMatTest = np.reshape(eMatrix[:,i], (sz[1], sz[2]))[:,:]
'''


# --------------------------------------------------------------------------
# The following code is just a working example so you don't get stuck with any
# of the graphs required. You may want to write your own code to align the
# results in a better layout.
# --------------------------------------------------------------------------

def visualize(albedo, normals, depth, horn):

    # Stride in the plot, you may want to adjust it to different images
    stride = 15

    # showing albedo map
    fig = plt.figure()
    albedo_max = albedo.max()
    albedo = albedo / albedo_max
    plt.imshow(albedo, cmap='gray')
    plt.show()

    # showing normals as three separate channels
    figure = plt.figure()
    ax1 = figure.add_subplot(131)
    ax1.imshow(normals[..., 0])
    ax2 = figure.add_subplot(132)
    ax2.imshow(normals[..., 1])
    ax3 = figure.add_subplot(133)
    ax3.imshow(normals[..., 2])
    plt.show()

    # showing normals as quiver
    X, Y, _ = np.meshgrid(np.arange(0,np.shape(normals)[0], 15),
                          np.arange(0,np.shape(normals)[1], 15),
                          np.arange(1))
    X = X[..., 0]
    Y = Y[..., 0]
    Z = depth[::stride,::stride].T
    NX = normals[..., 0][::stride,::-stride].T
    NY = normals[..., 1][::-stride,::stride].T
    NZ = normals[..., 2][::stride,::stride].T
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    plt.quiver(X,Y,Z,NX,NY,NZ, length=15)
    plt.show()

    # plotting wireframe depth map
    H = depth[::stride,::stride]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y, H.T)
    plt.show()

    # plot horn output 
    H = horn[::stride,::stride]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y, H.T)
    plt.show()
    
visualize(albedo, normals, depth, horn)