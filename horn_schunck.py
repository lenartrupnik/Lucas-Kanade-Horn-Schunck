from ex1_utils import *
from scipy.ndimage import convolve
from lucas_kanade import lucaskanade
from sklearn.metrics.pairwise import cosine_similarity

def horn_schunck(im1, im2, n_iters, lmbd, lucas_kanade= True):
    assert im1.shape == im2.shape, f'Image shapes are not the same.'
    #Get image spatial derivate
    Ix, Iy, It, _ = image_spatial_derivates(im1, im2)
    
    #Define residual Laplacian kernel
    L_d = np.matrix([[0, 0.25, 0],
                   [0.25, 0, 0.25],
                   [0, 0.25, 0]])
    
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    
    if  lucas_kanade:
        u, v = lucaskanade(im1, im2, 10)
    #Define initial estimates for u and v
    
    #Define iterative corrections
    u_a = np.ones(im1.shape)
    v_a = np.ones(im1.shape)
    
    D = np.sum([np.square(Ix), np.square(Iy), lmbd])
    
    for i in range(n_iters):
        
        #Update corrections
        u_a = convolve(u, L_d)
        v_a = convolve(v, L_d)
        
        #Calculate new P 
        P = sum([It, np.multiply(Ix, u_a), np.multiply(Iy, v_a)])

        P_D = np.divide(P, D)
        
        #Calculate u, v
        u = np.subtract(u_a, np.multiply(Ix, P_D))
        v = np.subtract(v_a, np.multiply(Iy, P_D))
        
    return u, v