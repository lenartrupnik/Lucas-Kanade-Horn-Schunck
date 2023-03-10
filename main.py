import numpy as np
import matplotlib.pyplot as plt
from ex1_utils import rotate_image, show_flow
from lucas_kanade import *
from horn_schunck import *
import time

def plot_scnthetic_image():
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)
    
    U_lk, V_lk = lucaskanade(im1, im2, 5)
    U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5, lucas_kanade=False)
    
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2,2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(U_lk, V_lk, ax1_21, type="angle")
    show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
    fig1.suptitle('Lucas-Kanade Optical Flow')
    
    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2,2)
    ax2_11.imshow(im1)
    ax2_12.imshow(im2)
    show_flow(U_hs, V_hs, ax2_21, type="angle")
    show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
    fig2.suptitle('Horn-Schunck Optical Flow')
    
    plt.show()
    

def plot_custom_images(im1, im2):
    
    U_lk, V_lk = lucaskanade(im1, im2, 10)
    U_hs, V_hs = horn_schunck(im1, im2, 500, 0.4, True)
    #U_hs_imp, V_hs_imp = horn_schunck(im1, im2, 350, 0.5, lucas_kanade=True)
    
    # fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2,2)
    # ax1_11.imshow(im1)
    # ax1_12.imshow(im2)
    # show_flow(U_lk, V_lk, ax1_21, type="angle")
    # show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
    # fig1.suptitle('Lucas-Kanade Optical Flow')
    
    # fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2,2)
    # ax2_11.imshow(im1)
    # ax2_12.imshow(im2)
    # show_flow(U_hs, V_hs, ax2_21, type="angle")
    # show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
    # fig2.suptitle('Horn-Schunck Optical Flow')
    
    # Plot images to get both images and methods on the same plot
    
    fig3, ((ax3_11, ax3_12), (ax3_21, ax3_22)) = plt.subplots(2,2)
    ax3_11.imshow(im1)
    ax3_12.imshow(im2)
    show_flow(U_lk, V_lk, ax3_21, type="field", set_aspect=True)
    show_flow(U_hs, V_hs, ax3_22, type='field', set_aspect=True)
    ax3_11.set_title("Frame t")
    ax3_12.set_title("Frame t + 1")
    ax3_21.set_title("Lucas-Kanade")
    ax3_22.set_title("Horn-Schunck")
    
    # Plot improvements on horn-schunck method
    # fig3, ((ax3_11, ax3_12), (ax3_21, ax3_22)) = plt.subplots(2,2)
    # ax3_11.imshow(im1)
    # ax3_12.imshow(im2)
    # show_flow(U_hs, V_hs, ax3_21, type='field', set_aspect=True)
    # show_flow(U_hs_imp, V_hs_imp, ax3_22, type="field", set_aspect=True)

    # ax3_11.set_title("Frame t")
    # ax3_12.set_title("Frame t + 1")
    # ax3_21.set_title("Basic Horn-Schunck, 1000 iterations")
    # ax3_22.set_title("Horn-Schunck with Lucas-Kanade, 350 iterations")
    
    plt.show()


def plot_harris_improvement(im1, im2):
    U_lk, V_lk = lucaskanade(im1, im2, 10, False)
    U_lk_h, V_lk_h = lucaskanade(im1, im2, 10, Harris=True)
    
    fig3, ((ax3_21, ax3_22)) = plt.subplots(2)
    show_flow(U_lk, V_lk, ax3_21, type="field", set_aspect=True)
    show_flow(U_lk_h, V_lk_h, ax3_22, type='field', set_aspect=True)
    ax3_21.set_title("Lucas-Kanade")
    ax3_22.set_title("Lucas-Kanade with harris")
    plt.show()

def plot_different_parameters(im1, im2):
    
    kernel_size = [5, 50, 100]
    lmbd = [0.1, 1, 10]
    iterations = [100, 1000, 10000]
    
    fig_0, axes_0 = plt.subplots(1,3)
    
    for k, ax in zip(kernel_size, axes_0):
        U_lk, V_lk = lucaskanade(im1, im2, k)
        show_flow(U_lk, V_lk, ax, type='field', set_aspect=True)
        ax.set_title(f'Kernel size: {k}x{k}')
    print(">>> Finished comparing kernel sizes")
    
    fig_1, axes_1 = plt.subplots(1,3)
    for l, ax in zip(lmbd, axes_1):
        U_hs, V_hs = horn_schunck(im1, im2, 1000, l)
        show_flow(U_hs, V_hs, ax, type='field', set_aspect=True)
        ax.set_title(f'Iterations: 1000, lambda: {l}')
    print(">>> Finished comparing lambdas")
    
    fig_2, axes_2 = plt.subplots(1,3)
    for iteration, ax in zip(iterations, axes_2):
        U_hs, V_hs = horn_schunck(im1, im2, iteration, 0.5)
        show_flow(U_hs, V_hs, ax, type='field', set_aspect=True)
        ax.set_title(f'Iterations: {iteration}, 0.5')
    print(">>> Finished comparing num. of iterations")

    plt.show()

def measure_time(im1, im2):
    kernel_size = [10, 100]
    iterations = [100, 1000]
    iterations = [35, 350]
    
    lucas_times = []
    horn_times = []
    print(">>> lucas vs horn")
    for kernel in kernel_size:
        start = time.process_time()
        lucaskanade(im1, im2, kernel)
        end= time.process_time()
        lucas_times.append(end-start)
        print(f'Time for lucas-Kanade with {kernel} kernel = {lucas_times[-1]}')
    
    for iteration in iterations:
        start = time.process_time()
        horn_schunck(im1, im2, iteration, 0.5)
        end= time.process_time()
        horn_times.append(end-start)
        print(f'Time for horn-Shunck with {iteration} iterations = {horn_times[-1]}')
    
    print(">>> Improved horn")
    for iteration in iterations:
        start = time.process_time()
        horn_schunck(im1, im2, iteration, 0.5, improved=True)
        end= time.process_time()
        horn_times.append(end-start)
        print(f'Time for horn-Shunck with {iteration} iterations = {horn_times[-1]}')
    
    
    
    #print(time_lucas, time_horn)
    
    
if __name__ == "__main__":
    #im1, im2 = prepare_images("custom images/traffic_0.png", "custom images/traffic_1.png")
    im1, im2 = prepare_images("custom images/ship_0.png", "custom images/ship_1.png")
    #im1, im2 = prepare_images("custom images/robot_0.png", "custom images/robot_1.png")

    #im1[dest > 0.01 * dest.max()]=[255]
    #show_img(im1)
    #plot_scnthetic_image()
    
    #plot_custom_images(im1, im2)
    plot_harris_improvement(im1, im2)
    #plot_different_parameters(im1, im2)
    #measure_time(im1, im2)