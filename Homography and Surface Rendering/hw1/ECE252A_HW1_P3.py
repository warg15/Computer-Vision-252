#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:32:57 2020

@author: femw90
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_normals(diffuse_normals, original_normals):
    # Stride in the plot, you may want to adjust it to different images
    stride = 5
    
    normalss = diffuse_normals
    normalss1 = original_normals
    
    print("Normals:")
    print("Diffuse")
    # showing normals as three separate channels
    figure = plt.figure()
    ax1 = figure.add_subplot(131)
    ax1.imshow(normalss[..., 0])
    ax2 = figure.add_subplot(132)
    ax2.imshow(normalss[..., 1])
    ax3 = figure.add_subplot(133)
    ax3.imshow(normalss[..., 2])
    plt.show()
    print("Original")
    figure = plt.figure()
    ax1 = figure.add_subplot(131)
    ax1.imshow(normalss1[..., 0])
    ax2 = figure.add_subplot(132)
    ax2.imshow(normalss1[..., 1])
    ax3 = figure.add_subplot(133)
    ax3.imshow(normalss1[..., 2])
    plt.show()
    
    
    
    
    
    
#Plot the normals for the sphere and pear for both the normal and diffuse components.
#1 : Load the different normals
# LOAD HERE

file = open('normals.pkl', 'rb')
normals = pickle.load(file)
file.close()

sphereDiffuse = normals[0]
sphereND = normals[1]
pearDiffuse = normals[2]
pearND = normals[3]

#2 : Plot the normals using plot_normals
#What do you observe? What are the differences between the diffuse component and the original images shown?
''' The diffuse is smoother on the object itself, while the originals have a pimple like 
surface on them, as a result of the specularities where the light causes that 
part of the image to appear almost white, as discussed in class. It is also
worth noting that the background is different, with the diffuse versions having
a speckled background and the originals having a more uniform background, possibly 
a result of background specularities removed and replaced resulting in non-uniform background
'''
#PLOT HERE
#plot_normals(sphereDiffuse, sphereND)
#plot_normals(pearDiffuse, pearND)

#apply mask first to make background black, then normalize
def normalize(img):
    assert img.shape[2] == 3
    maxi = img.max()
    mini = img.min()
    return (img - mini)/(maxi-mini)

'''
def lambertian(normals, lights, color, intensity, mask):
    #Your implementation
    image = np.ones((normals.shape[0], normals.shape[1], 3))
    for i in range(0, normals.shape[0]):
        #print(i)
        for j in range(0, normals.shape[1]):
            #NC = np.matmul(normals[i,j,:].T, color)
            #Id = np.dot(lights, NC)
            nrml = np.array([normals[i,j,:]])
            LdotN = np.dot(lights, nrml)
            Id = np.matmul(LdotN, color)
            image[i,j,:] = Id[:,0]
    image[:,:,0] = image[:,:,0]*mask
    image[:,:,1] = image[:,:,1]*mask
    image[:,:,2] = image[:,:,2]*mask
    image = normalize(image)
    return (image)
'''

def lambertian(normals, lights, color, intensity, mask):
    '''Your implementation'''
    image = np.ones((normals.shape[0], normals.shape[1], 3))
    lights = lights.T
    for i in range(0, normals.shape[0]):
        print(i)
        for j in range(0, normals.shape[1]):
            #NC = np.matmul(normals[i,j,:].T, color)
            #Id = np.dot(lights, NC)
            nrml = np.array([normals[i,j,:]]).T
            LdotN = np.matmul(lights, nrml)[0,0]
            Id = LdotN * color
            image[i,j,:] = Id[:,0]
    image[:,:,0] = image[:,:,0]*mask
    image[:,:,1] = image[:,:,1]*mask
    image[:,:,2] = image[:,:,2]*mask
    image = normalize(image)
    return (image)

# Load the masks for the sphere and pear
# LOAD HERE

file = open('masks.pkl', 'rb')
masks = pickle.load(file)
file.close()

sphereMask = masks[0]
pearMask = masks[1]

# Output the rendering results for Pear
dirn1 = np.array([[1.0],[1.0],[0]])
color1 = np.array([[.75],[.75],[.5]])
dirn2 = np.array([[1.0/3],[-1.0/3],[1.0/2]])
color2 = np.array([[1],[1],[1]])

#test = lambertian(pearDiffuse, dirn1, color1, 1, pearMask)
#plt.imshow(test)
#plt.show()

shapesName = ['pear', 'sphere']
settingsName = ['original', 'diffuse']
lightingName = ['lighting1', 'lighting2']

'''
count = 1
for i in range(0,len(shapesName)):
    if (i == 0): mask = pearMask
    else: mask = sphereMask
    #print('i = ' + str(i))
    for j in range(0,len(settingsName)):
        if (j == 0): 
            if (i == 0): normals = pearDiffuse
            else: normals = sphereDiffuse
        if (j == 1): 
            if (i == 0): normals = pearND
            else: normals = sphereND
        #print('j = ' + str(j))
        for k in range(0,len(lightingName)):
            if (k == 0): 
                lights = dirn1 
                color = color1
            else: 
                lights = dirn2 
                color = color2
            print('k = ' + str(k))
            result = lambertian(normals, lights, color, 1, mask)
            
            plt.subplot(4, 2, count)
            plt.title(shapesName[i] + ' ' + settingsName[j] + ' ' + lightingName[k])
            plt.imshow(result)
            count = count + 1
plt.show()
'''

'''
#Display the rendering results for pear for both diffuse and for both the light sources
count = 1
i=0
if (i == 0): mask = pearMask
else: mask = sphereMask
for j in range(0,len(settingsName)):
    if (j == 0): 
        if (i == 0): normals = pearDiffuse
        else: normals = sphereDiffuse
    if (j == 1): 
        if (i == 0): normals = pearND
        else: normals = sphereND
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = dirn1 
            color = color1
        else: 
            lights = dirn2 
            color = color2
        print('k = ' + str(k))
        result = lambertian(normals, lights, color, 1, mask)
        
        plt.subplot(2, 2, count)
        plt.title(shapesName[i] + ' ' + settingsName[j] + ' ' + lightingName[k])
        plt.imshow(result)
        count = count + 1
plt.show()
'''

# Output the rendering results for Sphere
dirn1 = np.array([[1.0],[1.0],[0]])
color1 = np.array([[.75],[.75],[.5]])
dirn2 = np.array([[1.0/3],[-1.0/3],[1.0/2]])
color2 = np.array([[1],[1],[1]])
#Display the rendering results for sphere for both diffuse and for both the light sources

'''
count = 1
i=1
if (i == 0): mask = pearMask
else: mask = sphereMask
for j in range(0,len(settingsName)):
    if (j == 0): 
        if (i == 0): normals = pearDiffuse
        else: normals = sphereDiffuse
    if (j == 1): 
        if (i == 0): normals = pearND
        else: normals = sphereND
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = dirn1 
            color = color1
        else: 
            lights = dirn2 
            color = color2
        print('k = ' + str(k))
        result = lambertian(normals, lights, color, 1, mask)
        
        plt.subplot(2, 2, count)
        plt.title(shapesName[i] + ' ' + settingsName[j] + ' ' + lightingName[k])
        plt.imshow(result)
        count = count + 1
plt.show()
'''



##########
'''Phong Model'''
###########

def phong(normals, lights, color, material, view, mask):
    '''Your implementation'''
    image = np.ones((normals.shape[0], normals.shape[1], 3))
    
    numColors = np.shape(color)[1]  #find M
    ka = 0
    kd = material[0]
    ks = material[1]
    alpha = material[2]
    
    for i in range(0, normals.shape[0]):
        #print(i)
        for j in range(0, normals.shape[1]):
            Iphong = np.zeros((3,1))
            nrml = np.array([normals[i,j,:]]).T
            for k in range(0, numColors):
                L = np.zeros((1,3))
                L[0,:] = lights[:,k] #get the current light direction
                Im = np.zeros((3,1))
                Im[:,0] = color[:,k] #get the current light RGB color
                
                LdotN = np.matmul(L, nrml)[0,0]
                
                Rm = 2*nrml*LdotN - L.T
                
                firstTerm = kd*LdotN*Im
                secondTerm = ks*(np.matmul(Rm.T, view)**alpha)*Im
                SUM = firstTerm + secondTerm
                if (LdotN <= 0):
                    SUM = np.zeros((3,1))
                Iphong = np.add(Iphong, SUM)
            #now do the add to array thing
            image[i,j,:] = Iphong[:,0]
    image[:,:,0] = image[:,:,0]*mask
    image[:,:,1] = image[:,:,1]*mask
    image[:,:,2] = image[:,:,2]*mask
    image = normalize(image)
    
    
    
    return (image)

# Output the rendering results for sphere
view =  np.array([[0],[0],[1]])
material = np.array([[0.1,0.75,5],[0.5,0.1,5],[0.5,0.5,10]])
lightcol1 = np.array([[1,0.75],[1,0.75],[0,0.5]])
lightcol2 =  np.array([[1.0/3,1],[-1.0/3,1],[1.0/2,1]])
#Display rendered results for sphere for all materials and light sources and combination of light sources

material1 = material[0,:]
material2 = material[1,:]
material3 = material[2,:]
view = view
lights1 = np.array([lightcol1[:,0]]).T
color1 = np.array([lightcol1[:,1]]).T
lights2 = np.array([lightcol2[:,0]]).T
color2 = np.array([lightcol2[:,1]]).T
lights3 = np.concatenate((lights1, lights2), axis=1)
color3 = np.concatenate((color1, color2), axis=1)
#test = phong(sphereDiffuse, lights2, color2, material1, view, sphereMask)
#plt.imshow(test)
#plt.show()

# 2 - sphere and pear
# 2 - original and deffuse
# 3 - material 1,2 and 3
# 3 - light 1, 2, and 1&2


settingsName = ['original', 'diffuse']
lightingName = ['lighting1', 'lighting2', 'lighting3']
materialName = ['material1', 'material2', 'material3']

##sphere
mask = sphereMask
count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = sphereND
    if (j == 1): normals = sphereDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        h=0
        material = material1
        print(count)
        
        result = phong(normals, lights, color, material, view, mask)
        plt.subplot(9, 2, count)
        plt.title('sphere' + ' ' + settingsName[j] + ' ' + lightingName[k] + ' ' + materialName[h])
        plt.imshow(result)
        count = count + 1
plt.tight_layout(pad=1.0)
plt.show()


count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = sphereND
    if (j == 1): normals = sphereDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        h=1
        material = material2
        print(count)
        
        result = phong(normals, lights, color, material, view, mask)
        plt.subplot(9, 2, count)
        plt.title('sphere' + ' ' + settingsName[j] + ' ' + lightingName[k] + ' ' + materialName[h])
        plt.imshow(result)
        count = count + 1
plt.tight_layout(pad=1.0)
plt.show()


count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = sphereND
    if (j == 1): normals = sphereDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        h=2
        material = material3
        print(count)
        
        result = phong(normals, lights, color, material, view, mask)
        plt.subplot(9, 2, count)
        plt.title('sphere' + ' ' + settingsName[j] + ' ' + lightingName[k] + ' ' + materialName[h])
        plt.imshow(result)
        count = count + 1
plt.tight_layout(pad=1.0)
plt.show()

##pear
mask = pearMask
count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = pearND
    if (j == 1): normals = pearDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        
        for h in range(0,len(materialName)):
            if (h == 0): material = material1
            if (h == 1): material = material2
            if (h == 2): material = material3
            print(count)
            
            result = phong(normals, lights, color, material, view, mask)
            plt.subplot(9, 2, count)
            plt.title('pear' + ' ' + settingsName[j] + ' ' + lightingName[k] + ' ' + materialName[h])
            plt.imshow(result)
            count = count + 1
plt.tight_layout(pad=1.0)
plt.show()




'''
Scratch Shit
# Output the rendering results for sphere
view =  np.array([[0],[0],[1]])
material = np.array([[0.1,0.75,5],[0.5,0.1,5],[0.5,0.5,10]])
lightcol1 = np.array([[1,0.75],[1,0.75],[0,0.5]])
lightcol2 =  np.array([[1.0/3,1],[-1.0/3,1],[1.0/2,1]])
#Display rendered results for sphere for all materials and light sources and combination of light sources

material1 = material[0,:]
material2 = material[1,:]
material3 = material[2,:]
view = view
lights1 = np.array([lightcol1[:,0]]).T
color1 = np.array([lightcol1[:,1]]).T
lights2 = np.array([lightcol2[:,0]]).T
color2 = np.array([lightcol2[:,1]]).T
lights3 = np.concatenate((lights1, lights2), axis=1)
color3 = np.concatenate((color1, color2), axis=1)
#test = phong(sphereDiffuse, lights2, color2, material1, view, sphereMask)
#plt.imshow(test)
#plt.show()

# 2 - sphere and pear
# 2 - original and deffuse
# 3 - material 1,2 and 3
# 3 - light 1, 2, and 1&2


settingsName = ['original', 'diffuse']
lightingName = ['color1', 'color2', 'color3']
materialName = ['material1', 'material2', 'material3']

##sphere
mask = sphereMask
count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = sphereND
    if (j == 1): normals = sphereDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        h=0
        material = material1
        #print(count)
        result = phong(normals, lights, color, material, view, mask)
        plt.subplot(3, 2, count)
        plt.title(settingsName[j] + ' ' + lightingName[k] )
        plt.imshow(result)
        count = count + 1
plt.tight_layout(pad=0.2)
print('Sphere'  ', ' + materialName[h])
plt.show()


count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = sphereND
    if (j == 1): normals = sphereDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        h=1
        material = material2
        #print(count)
        
        result = phong(normals, lights, color, material, view, mask)
        plt.subplot(3, 2, count)
        plt.title(settingsName[j] + ' ' + lightingName[k] )
        plt.imshow(result)
        count = count + 1
plt.tight_layout(pad=0.3)
print('Sphere'  ', ' + materialName[h])
plt.show()


count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = sphereND
    if (j == 1): normals = sphereDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        h=2
        material = material3
        #print(count)
        
        result = phong(normals, lights, color, material, view, mask)
        plt.subplot(3, 2, count)
        plt.title(settingsName[j] + ' ' + lightingName[k] )
        plt.imshow(result)
        count = count + 1
plt.tight_layout(pad=0.1)
print('Sphere'  ', ' + materialName[h])
plt.show()

# Output the rendering results for the pear.
view =  np.array([[0],[0],[1]])
material = np.array([[0.1,0.75,5],[0.5,0.1,5],[0.5,0.5,10]])
lightcol1 = np.array([[1,0.75],[1,0.75],[0,0.5]])
lightcol2 =  np.array([[1.0/3,1],[-1.0/3,1],[1.0/2,1]])
#Display rendered results for pear for all materials and light sources and combination of light sources


material1 = material[0,:]
material2 = material[1,:]
material3 = material[2,:]
view = view
lights1 = np.array([lightcol1[:,0]]).T
color1 = np.array([lightcol1[:,1]]).T
lights2 = np.array([lightcol2[:,0]]).T
color2 = np.array([lightcol2[:,1]]).T
lights3 = np.concatenate((lights1, lights2), axis=1)
color3 = np.concatenate((color1, color2), axis=1)
#test = phong(sphereDiffuse, lights2, color2, material1, view, sphereMask)
#plt.imshow(test)
#plt.show()

# 2 - sphere and pear
# 2 - original and deffuse
# 3 - material 1,2 and 3
# 3 - light 1, 2, and 1&2


settingsName = ['original', 'diffuse']
lightingName = ['lighting1', 'lighting2', 'lighting3']
materialName = ['material1', 'material2', 'material3']



##pear
mask = pearMask
count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = pearND
    if (j == 1): normals = pearDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        h=0
        material = material1
        print(count)
        
        result = phong(normals, lights, color, material, view, mask)
        plt.subplot(3, 2, count)
        plt.title(settingsName[j] + ' ' + lightingName[k] )
        plt.imshow(result)
        count = count + 1
plt.tight_layout(pad=1.0)
plt.title('Pear'  ' ' + materialName[h])
plt.show()


count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = pearND
    if (j == 1): normals = pearDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        h=1
        material = material2
        print(count)
        
        result = phong(normals, lights, color, material, view, mask)
        plt.subplot(3, 2, count)
        plt.title(settingsName[j] + ' ' + lightingName[k] )
        plt.imshow(result)
        count = count + 1
plt.tight_layout(pad=1.0)
plt.title('Pear'  ' ' + materialName[h])
plt.show()


count = 1
for j in range(0,len(settingsName)):
    if (j == 0): normals = pearND
    if (j == 1): normals = pearDiffuse
    #print('j = ' + str(j))
    for k in range(0,len(lightingName)):
        if (k == 0): 
            lights = lights1 
            color = color1
        if (k == 1): 
            lights = lights2 
            color = color2
        if (k == 2): 
            lights = lights3 
            color = color2
        h=2
        material = material3
        print(count)
        
        result = phong(normals, lights, color, material, view, mask)
        plt.subplot(3, 2, count)
        plt.title(settingsName[j] + ' ' + lightingName[k] )
        plt.imshow(result)
        count = count + 1
plt.tight_layout(pad=1.0)
plt.title('Pear'  ' ' + materialName[h])
plt.show()








'''



