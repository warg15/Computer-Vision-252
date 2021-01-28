import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, convolve2d
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve
from skimage.transform import resize
from scipy.signal import convolve, convolve2d
# from tqdm import tqdm_notebook

def gradient(image):
    g_mag = np.zeros_like(image)
    g_theta = np.zeros_like(image)
    kh = np.array([[1,0,-1]])
    kv = np.array([[1],[0],[-1]])
    gX = convolve(image, kh, mode = 'same')
    gY = convolve(image, kv, mode = 'same')
    gX[:,0] = gX[:,1]
    gY[0,:] = gY[1,:]
    return gX, -gY

def grayscale(img):
    '''
    Converts RGB image to Grayscale
    '''
    gray=np.zeros((img.shape[0],img.shape[1]))
    gray=img[:,:,0]*0.2989+img[:,:,1]*0.5870+img[:,:,2]*0.1140
    return gray

def plot_optical_flow(img0,img1,U,V,titleStr, color=False):
    '''
    Plots optical flow given U,V and the images
    '''
    
    # Change t if required, affects the number of arrows
    # t should be between 1 and min(U.shape[0],U.shape[1])
    t=8
    
    
    ##########changed on recommendation of piazza
    ##########changed on recommendation of piazza
    #V=-V
    ##########changed on recommendation of piazza
    ##########changed on recommendation of piazza
    
    
    # Subsample U and V to get visually pleasing output
    U1 = U[::t,::t]
    V1 = V[::t,::t]
    
    # Create meshgrid of subsampled coordinates
    r, c = img0.shape[0],img0.shape[1]
    cols,rows = np.meshgrid(np.linspace(0,c-1,c), np.linspace(0,r-1,r))
    cols = cols[::t,::t]
    rows = rows[::t,::t]
    
    # Plot optical flow
    plt.figure(figsize=(20,20))
    plt.subplot(121)
    plt.imshow(img0, alpha=0.5)
    plt.imshow(img1, alpha=0.5)
    plt.title('Overlayed Images')
    plt.subplot(122)
    if color:
        plt.imshow(img0)
    else:
        plt.imshow(grayscale(img0), cmap='gray')
    plt.quiver(cols,rows,U1,V1)
    plt.title(titleStr)
    plt.show()
    

images=[]
for i in range(1,5):
    images.append(plt.imread('optical_flow_images/im'+str(i)+'.png')[:,:288,:])
# each image after converting to gray scale is of size -> 400x288
# each image after converting to gray scale is of size -> 400x288
    
# you can use interpolate from scipy
# You can implement 'upsample_flow' and 'OpticalFlowRefine' 
# as 2 building blocks in order to complete this.
    
# %%
import scipy.misc
from skimage.transform import resize
def upsample_flow(u_prev, v_prev):
    ''' You may implement this method to upsample optical flow from
    previous level
    u_prev, v_prev -> optical flow from prev level
    u, v -> upsampled optical flow to the current level
    '''
    if u_prev is None and v_prev is None:
        return u_prev, v_prev
    u = resize(u_prev,(u_prev.shape[0]*2,u_prev.shape[1]*2),order=1)
    v = resize(v_prev,(u_prev.shape[0]*2,u_prev.shape[1]*2),order=1)
    u = u*2;
    v = v*2;
    return u, v

def warp(im, u_prev, v_prev):
    warpedIm = np.zeros(im.shape)
    for i in range(0, im.shape[0]):
        for j in range(0,im.shape[1]):
            x = np.rint(i+v_prev[i,j]).astype(np.int)
            y = np.rint(j+u_prev[i,j]).astype(np.int)
            #x = np.rint(i+u_prev[i,j]).astype(int)
            #y = np.rint(j+v_prev[i,j]).astype(int)
            #if(x-i != 0): print('x: ', x-i, 'i,j = ', i,j)
            #if(y-j != 0): print('y: ', y-j, 'i,j = ', i,j)
            if (0<=x) and (x<im.shape[0]) and (0<=y) and (y<im.shape[1]):
                warpedIm[i,j] = im[x, y]
    return warpedIm

def It_patch(im2, im1, u_prev, v_prev, i, j, sub, add): 
    #lower and upper bounds
    #i and j are the placed it is cetnered on
    #shift entire patch by the u_prev and v_prev
    u_window = np.rint(u_prev[i,j]).astype(np.int)
    v_window = np.rint(v_prev[i,j]).astype(np.int)
    #if(u_window != 0): print('u_window: ', u_window, 'i,j = ', i,j)
    #if(v_window != 0): print('v_window: ', v_window, 'i,j = ', i,j)
    patch1 = im1[i-sub:i+add, j-sub:j+add]
    patch2 = im2[i-sub+v_window:i+add+v_window,
                 j-sub+u_window:j+add+u_window]
    #patch2 = im2[i-sub + u_window:i+add + u_window,
                 #j-sub + v_window:j+add + v_window]
    if patch2.shape != patch1.shape: 
        print('i: ',i,'j: ',j)
        print('u_window: ', u_window, 'v_window: ', v_window)
        print('im1 shape: ', im1.shape, 'im2 shape: ', im2.shape)
    It_patch = patch2 - patch1
    return It_patch
    
def OpticalFlowRefine(im1,im2,window, u_prev=None, v_prev=None):
    '''
    Inputs: the two images at current level and window size
    u_prev, v_prev - previous levels optical flow
    Return u,v - optical flow at current level
    '''
    """ ==========
    YOUR CODE HERE
    ========== """
    #apply pise-wise the sliding window, finding the u,v for each pixel
    #is that all form this function?
    #must solve with matrix eqn system
    #add u_prev and v_prev is they exist??
    u, v = np.zeros(im1.shape), np.zeros(im1.shape)
    if u_prev is None and v_prev is None:
        u_prev, v_prev = np.zeros(im1.shape), np.zeros(im1.shape)
    im1Warp = warp(im1, u_prev, v_prev)
    im2Warp = warp(im2, u_prev, v_prev)
    
    #Ix, Iy = gradient(im1Warp) 
    Tx, Ty = np.gradient(im1Warp)
    Ix, Iy = Tx, -Ty
    It = im2Warp - im1 #This has to be done over a window dumbass
    sub = window//2
    add = window//2
    sz = im1.shape
    for i in range(sub, sz[0]-1-add ):
        for j in range(sub, sz[1]-1-add ):
            c = np.zeros((2,2))
            oM = np.zeros((2,1))
            if window%2 == 0:
                wX = np.array(Ix[i-sub:i+add, j-sub:j+add])
                wY = np.array(Iy[i-sub:i+add, j-sub:j+add])
                #wT - np.array(It[i-sub:i+add, j-sub:j+add])
                
                wT = It_patch(im2, im1, u_prev, v_prev, i, j, sub, add)
                
            else:
                wX = np.array(Ix[i-sub:i+add+1, j-sub:j+add+1])
                wY = np.array(Iy[i-sub:i+add+1, j-sub:j+add+1])
                #wT = np.array(It[i-sub:i+add+1, j-sub:j+add+1])
                
                wT = It_patch(im2, im1, u_prev, v_prev, i, j, sub, add+1)
            
            c[0,0] = np.sum(wX*wX)
            c[0,1] = np.sum(wX*wY)
            c[1,0] = np.sum(wX*wY)
            c[1,1] = np.sum(wY*wY)
            #now to find the second matrix
            oM[0,0] = np.sum(-wX*wT)
            oM[1,0] = np.sum(-wY*wT)
            uv = np.matmul(np.linalg.pinv(c), oM) ##PROMBLE, C IS SINGULAR
            u[i,j] = np.rint(uv[0])
            v[i,j] = np.rint(uv[1])
    return u, v

# %%
def downSampleIm(im1, im2, level): #send it the level 1 through..5 (level 1 is fullsize)
    im1Down, im2Down = im1, im2
    for i in range(level-1):
        smoothedImage1 = gaussian_filter(im1Down, 1)
        smoothedImage2 = gaussian_filter(im2Down, 1)
        im1Down = resize(smoothedImage1, (smoothedImage1.shape[0] // 2, 
                                      smoothedImage1.shape[1] // 2), anti_aliasing=True)
        im2Down = resize(smoothedImage2, (smoothedImage2.shape[0] // 2, 
                                      smoothedImage2.shape[1] // 2), anti_aliasing=True)
        
    return im1Down, im2Down

def LucasKanadeMultiScale(im1,im2,window, numLevels=2):
    '''
    Implement the multi-resolution Lucas kanade algorithm
    Inputs: the two images, window size and number of levels
    if numLevels = 1, then compute optical flow at only the given image level.
    Returns: u, v - the optical flow
    '''
    
    """ ==========
    YOUR CODE HERE
    ========== """
    # You can call OpticalFlowRefine iteratively
    im1Down, im2Down = downSampleIm(im1, im2, numLevels)
    u, v = OpticalFlowRefine(im1Down, im2Down, window)
    #u_prev, v_prev = u, v
    for lvl in range(numLevels -1): #starts at 0, goes through running on numLevels-2
        #u_prev, v_prev = u, v
        u_prev, v_prev = upsample_flow(u, v)
        im1Down, im2Down = downSampleIm(im1, im2, numLevels-lvl-1)
        u_new, v_new = OpticalFlowRefine(im1Down, im2Down, window, u_prev, v_prev)
        u, v = u_prev+u_new, v_prev+v_new
    return u, v




# %%
#Call this shit?
window=13
im1, im2 = grayscale(images[0]), grayscale(images[1])
'''
#downsample im 1
level = 4
for i in range(level):
    smoothedImage1 = gaussian_filter(im1, 1)
    smoothedImage2 = gaussian_filter(im2, 1)
    im1 = resize(smoothedImage1, (smoothedImage1.shape[0] // 2, 
                                  smoothedImage1.shape[1] // 2), anti_aliasing=True)
    im2 = resize(smoothedImage2, (smoothedImage2.shape[0] // 2, 
                                  smoothedImage2.shape[1] // 2), anti_aliasing=True)
'''
#u, v = OpticalFlowRefine(im1, im2, window)
numLevels=5
u, v = LucasKanadeMultiScale(im1,im2,window,numLevels)









