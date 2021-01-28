import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, convolve2d
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve
from skimage.transform import resize
from scipy.signal import convolve, convolve2d
# from tqdm import tqdm_notebook

def gradient(image):
    kh = np.array([[1,0,-1]])
    kv = np.array([[1],[0],[-1]])
    gX = convolve2d(image, kh, mode = 'same')
    gY = convolve2d(image, kv, mode = 'same')
    return gX, gY

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
# %%
# you can use interpolate from scipy
# You can implement 'upsample_flow' and 'OpticalFlowRefine' 
# as 2 building blocks in order to complete this.
import scipy.misc
from scipy.signal import convolve2d
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

# %%
    
def It_patch(im2, im1, u_prev, v_prev, i, j, sub, add): 
    #lower and upper bounds
    #i and j are the placed it is cetnered on
    #shift entire patch by the u_prev and v_prev
    u_window = np.rint(u_prev[i,j]).astype(np.int)
    v_window = np.rint(v_prev[i,j]).astype(np.int)
    #Dims = [i-sub+v_window, i+add+v_window, j-sub+u_window, j+add+u_window]
    patch1 = im1[i-sub:i+add, j-sub:j+add]
    patch2 = im2[i-sub+v_window:i+add+v_window,
                 j-sub+u_window:j+add+u_window]
    It_patch = patch2 - patch1
    return It_patch

def Ix2(im, dim):
    gX, gY = gradient(im)
    result = gX*gX
    #each location in matrix has to be the summation over the entire window
    #convolve window at each location
    return gX, convolve2d(result, np.ones((dim,dim)), mode='same')
def Iy2(im, dim):
    gX, gY = gradient(im)
    result = gY*gY
    return gY, convolve2d(result, np.ones((dim,dim)), mode='same')
def IxIy(im, dim):
    gX, gY = gradient(im)
    result = gX*gY
    return convolve2d(result, np.ones((dim,dim)), mode='same')
    
def OpticalFlowRefine_RGB(im1,im2,window, u_prev=None, v_prev=None):
    '''
    Inputs: the two images at current level and window size
    u_prev, v_prev - previous levels optical flow
    Return u,v - optical flow at current level
    '''
    im1_0, im2_0 = im1[:,:,0], im2[:,:,0]
    im1_1, im2_1 = im1[:,:,1], im2[:,:,1]
    im1_2, im2_2 = im1[:,:,2], im2[:,:,2]
    ############################# get u_prev and v_prev
    u, v = np.zeros((im1_0.shape)), np.zeros((im1_0.shape))
    if u_prev is None and v_prev is None:
        u_prev, v_prev = np.zeros(im1_0.shape), np.zeros(im1_0.shape)
    u_prev = np.round(u_prev).astype(np.int)
    v_prev = np.round(v_prev).astype(np.int)
    
    ############################# get matrices of Ix^2, Iy^2, IxIy     
    Ix_0, Ix2Master_0 = Ix2(im1_0, window)
    Iy_0, Iy2Master_0 = Iy2(im1_0, window)
    IxIyMaster_0 = IxIy(im1_0, window)
    
    Ix_1, Ix2Master_1 = Ix2(im1_1, window)
    Iy_1, Iy2Master_1 = Iy2(im1_1, window)
    IxIyMaster_1 = IxIy(im1_1, window)
    
    Ix_2, Ix2Master_2 = Ix2(im1_2, window)
    Iy_2, Iy2Master_2 = Iy2(im1_2, window)
    IxIyMaster_2 = IxIy(im1_2, window)
    
    sub = window//2
    add = window//2
    sz = im1.shape
    for i in range(sub, sz[0]-add-1 ):
        for j in range(sub, sz[1]-add-1):
            c = np.zeros((2,2))
            d = np.zeros((2,2))
            oM = np.zeros((2,1))
            
            wX_0 = np.array(Ix_0[i-sub:i+add, j-sub:j+add])
            wY_0 = np.array(Iy_0[i-sub:i+add, j-sub:j+add])
            wT_0 = It_patch(im2_0, im1_0, u_prev, v_prev, i, j, sub, add)
            
            wX_1 = np.array(Ix_1[i-sub:i+add, j-sub:j+add])
            wY_1 = np.array(Iy_1[i-sub:i+add, j-sub:j+add])
            wT_1 = It_patch(im2_1, im1_1, u_prev, v_prev, i, j, sub, add)
            
            wX_2 = np.array(Ix_2[i-sub:i+add, j-sub:j+add])
            wY_2 = np.array(Iy_2[i-sub:i+add, j-sub:j+add])
            wT_2 = It_patch(im2_2, im1_2, u_prev, v_prev, i, j, sub, add)
            
            c[0,0] = Ix2Master_0[i,j] + Ix2Master_1[i,j] + Ix2Master_2[i,j]
            c[0,1] = IxIyMaster_0[i,j] + IxIyMaster_1[i,j] + IxIyMaster_2[i,j]
            c[1,0] = IxIyMaster_0[i,j] + IxIyMaster_1[i,j] + IxIyMaster_2[i,j]
            c[1,1] = Iy2Master_0[i,j] + Iy2Master_1[i,j] + Iy2Master_2[i,j]
            
            #now to find the second matrix
            oM[0,0] = np.sum(-wX_0*wT_0) + np.sum(-wX_1*wT_1) + np.sum(-wX_2*wT_2)
            oM[1,0] = np.sum(-wY_0*wT_0) + np.sum(-wY_1*wT_1) + np.sum(-wY_2*wT_2)
            #print(oM[0,0], oM[1,0])
            uv = np.matmul(np.linalg.pinv(c), oM) ##PROMBLE, C IS SINGULAR
            u[i,j] = uv[0]
            v[i,j] = uv[1]
            nothing =0
    u = u + u_prev
    v = v + v_prev
    return u, v

# %%
def gaussian2d(filter_size=5, sig=1.0):
    """Creates a 2D Gaussian kernel with
    side length `filter_size` and a sigma of `sig`."""
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def downSampleImRGB(im1, im2, level): #send it the level 1 through..5 (level 1 is fullsize)
    im1Down_0, im2Down_0 = im1[:,:,0], im2[:,:,0]
    im1Down_1, im2Down_1 = im1[:,:,1], im2[:,:,1]
    im1Down_2, im2Down_2 = im1[:,:,2], im2[:,:,2]
    if level == 1:
        im1DownF = im1
        im2DownF = im2
    
    for i in range(level-1):
        smooth1_0 = convolve2d(im1Down_0,gaussian2d(), mode='same')
        smooth2_0 = convolve2d(im2Down_0,gaussian2d(), mode='same')
        im1Down_0 = smooth1_0[::2,::2]
        im2Down_0 = smooth2_0[::2,::2]
        
        smooth1_1 = convolve2d(im1Down_1,gaussian2d(), mode='same')
        smooth2_1 = convolve2d(im2Down_1,gaussian2d(), mode='same')
        im1Down_1 = smooth1_1[::2,::2]
        im2Down_1 = smooth2_1[::2,::2]
        
        smooth1_2 = convolve2d(im1Down_2,gaussian2d(), mode='same')
        smooth2_2 = convolve2d(im2Down_2,gaussian2d(), mode='same')
        im1Down_2 = smooth1_2[::2,::2]
        im2Down_2 = smooth2_2[::2,::2]
        
        im1DownF = np.zeros((im1Down_0.shape[0], im1Down_0.shape[1], 3))
        im2DownF = np.zeros((im2Down_0.shape[0], im2Down_0.shape[1], 3))
        im1DownF[:,:,0] = im1Down_0
        im2DownF[:,:,0] = im2Down_0
        im1DownF[:,:,1] = im1Down_1
        im2DownF[:,:,1] = im2Down_1
        im1DownF[:,:,2] = im1Down_2
        im2DownF[:,:,2] = im2Down_2
    
    return im1DownF, im2DownF

def LucasKanadeMultiScaleRGB(im1,im2,window, numLevels=2):

    # You can call OpticalFlowRefine iteratively
    im1Down, im2Down = downSampleImRGB(im1, im2, numLevels)
    u, v = OpticalFlowRefine_RGB(im1Down, im2Down, window)
    #u_prev, v_prev = u, v
    for lvl in range(numLevels -1): #starts at 0, goes through running on numLevels-2
        #u_prev, v_prev = u, v
        u_prev, v_prev = upsample_flow(u, v)
        im1Down, im2Down = downSampleImRGB(im1, im2, numLevels-lvl-1)
        u, v = OpticalFlowRefine_RGB(im1Down, im2Down, window, u_prev, v_prev)
        #u, v = u_prev+u_new, v_prev+v_new
    return u, v
# %%
    
# Your code here
# use one fixed window and numLevels for all pairs
# Example code to generate output
window=13
numLevels=3
U,V=LucasKanadeMultiScaleRGB(images[0][:,:,:3],images[1][:,:,:3],\
                          window,numLevels)
plot_optical_flow(images[0],images[1],U,V, \
                  'levels = ' + str(numLevels) + ', window = '+str(window) + ', image 1 and image 2', color=True)

U,V=LucasKanadeMultiScale(grayscale(images[0]),grayscale(images[1]),\
                          window,numLevels)
plot_optical_flow(images[0],images[1],U,V, \
                  'levels = ' + str(numLevels) + ', window = '+str(window) + ', image 1 and image 2 grayscale')



U,V=LucasKanadeMultiScaleRGB(images[0][:,:,:3],images[2][:,:,:3],\
                          window,numLevels)
plot_optical_flow(images[0],images[2],U,V, \
                  'levels = ' + str(numLevels) + ', window = '+str(window) + ', image 1 and image 3', color=True)

U,V=LucasKanadeMultiScale(grayscale(images[0]),grayscale(images[2]),\
                          window,numLevels)
plot_optical_flow(images[0],images[2],U,V, \
                  'levels = ' + str(numLevels) + ', window = '+str(window) + ', image 1 and image 3 grayscale')




U,V=LucasKanadeMultiScaleRGB(images[0][:,:,:3],images[3][:,:,:3],\
                          window,numLevels)
plot_optical_flow(images[0],images[3],U,V, \
                  'levels = ' + str(numLevels) + ', window = '+str(window) + ', image 1 and image 4', color=True)

U,V=LucasKanadeMultiScale(grayscale(images[0]),grayscale(images[3]),\
                          window,numLevels)
plot_optical_flow(images[0],images[3],U,V, \
                  'levels = ' + str(numLevels) + ', window = '+str(window) + ', image 1 and image 4 grayscale')


