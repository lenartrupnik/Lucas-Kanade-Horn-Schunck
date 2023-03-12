import cv2
from ex1_utils import *
from scipy.ndimage import convolve
import numpy as np
import time

def lucaskanade(im1, im2, N = 3, Harris = False):
    assert im1.shape == im2.shape
    
    #Prepare kernel for convolution
    kernel = np.ones((N, N))
    
    #Get image spatial derivates
    Ix, Iy, It, dest = image_spatial_derivates(im1, im2)

    #Follow lecture procedure to calculate every component
    Ix_t = convolve(np.multiply(Ix, It), kernel)
    Iy_t = convolve(np.multiply(Iy, It), kernel)
    Ix_x = convolve(np.multiply(Ix, Ix), kernel)
    Iy_y = convolve(np.multiply(Iy, Iy), kernel)
    Ix_y = convolve(np.multiply(Ix, Iy), kernel)
    
    D = np.subtract(np.multiply(Ix_x, Iy_y), np.square(Ix_y))

    #Add a small value to D so that we don't divide by 0
    D += 0.000001
    
    U = -np.divide(
        np.subtract(
            np.multiply(Iy_y, Ix_t), 
            np.multiply(Ix_y, Iy_t)),
        D)
    
    V = -np.divide(
        np.subtract(
            np.multiply(Ix_x, Iy_t), 
            np.multiply(Ix_y, Ix_t)),
        D)
    # Harris improvement
    if Harris:
        U[dest< 0.001]=0
        V[dest< 0.001]=0
        
    return U, V
    