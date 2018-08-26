import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import numpy as np
import cv2


def cross_correlation_2d(img, kernel):
    #grey scale image
    if len(img.shape) == 2:
        return kernelMatrix(img, kernel)
    else:
        #image RGB
        R, G, B = img.shape
        channels = np.zeros((R, G, B))
        for i in range(B):
            channels[:, :, i] = kernelMatrix(img[:, :, i], kernel)
        return channels


def kernelMatrix(img, kernel):
    length, width = kernel.shape
    lHalf, wHalf = (length - 1) / 2, (width - 1) / 2 #half length and width of the kernel
    L, W = img.shape #length and width of the image
    padding = np.zeros((L + 2 * lHalf, W + 2 * wHalf))
    padding[lHalf : lHalf + L, wHalf : wHalf + W] = img
    # make a matrix of kernel for each pixel in the image
    matrix = np.zeros((L * W, length * width))
    counter = 0
    for i in range(L):
        for j in range(W):
            #a vector of neigbors of pixel itself
            pixel = padding[i : 2 * lHalf + i + 1, j : 2 * wHalf + j + 1].reshape((length * width,))
            matrix[counter] = pixel
            counter += 1
    kernelVec = kernel.reshape((length * width,))
    #Convolve the kernel with the picture
    return np.dot(matrix, kernelVec).reshape((L, W))


#Flip inverse a kernel
def convolve_2d(img, kernel):

    return cross_correlation_2d(img, kernel[::-1, ::-1])


def gaussian_blur_kernel_2d(sigma, width, height):
    xHalf = (width - 1) / 2 #radious of the kernel
    yHalf = (height - 1) / 2
    xSq = np.arange(-xHalf, xHalf + 1, 1.0) ** 2
    ySq = np.arange(-yHalf, yHalf + 1, 1.0) ** 2
    X = np.sqrt(1 / (2 * np.pi * sigma ** 2)) * np.exp(-xSq / (2 * sigma ** 2))
    Y = np.sqrt(1 / (2 * np.pi * sigma ** 2)) * np.exp(-ySq / (2 * sigma ** 2))
    kernel_blur = np.outer(X, Y) / (np.sum(X) * np.sum(Y))
    return kernel_blur

def low_pass(img, sigma, size):

    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):

    return img - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):

    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)



