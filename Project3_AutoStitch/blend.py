import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    corner1 = np.array((0,0,1))
    corner2 = np.array((0,img.shape[0]-1,1))
    corner3 = np.array((img.shape[1]-1,img.shape[0]-1,1))
    corner4 = np.array((img.shape[1]-1,0,1))

    p1 = np.dot(M, corner1)
    p2 = np.dot(M, corner2)
    p3 = np.dot(M, corner3)
    p4 = np.dot(M, corner4)
    
    p1 /= p1[2]
    p2 /= p2[2]
    p3 /= p3[2]
    p4 /= p4[2]
    
    minX = min(p1[0],p2[0],p3[0],p4[0])
    minY = min(p1[1],p2[1],p3[1],p4[1])
    maxX = max(p1[0],p2[0],p3[0],p4[0])
    maxY = max(p1[1],p2[1],p3[1],p4[1])
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function

       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights

    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    height, width, channel = img.shape

    #Normalization to remove brightness.
    for chan in range(3):
        c_max = np.max(img[:, :, chan])

        c_min = np.min(img[:, :, chan])
        ranging = c_max - c_min
        img[:, :, chan] = int((float(img[:, :, chan] - c_min) / float(ranging)) * 255)

    addon = np.zeros(acc.shape)

    alpha = np.ones((height ,width))
    for c in range(blendWidth):
        alpha[:, c] = [float(c) / blendWidth] * height
        alpha[:, width - c - 1] = [float(c) / blendWidth] * height


    feathering = np.zeros((height, width, channel + 1))
    for chan in range(3):
        feathering[:, :, chan] = img[:, :, chan] * alpha
    feathering[:, :, -1] = alpha

    for chan in range(4):
        addon[:, :, chan] = cv2.warpPerspective(feathering[:, :, chan], M, dsize=(acc.shape[1], acc.shape[0]), flags= cv2.INTER_LINEAR)

    acc += addon

#TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    img = np.zeros((acc.shape[0], acc.shape[1], 4), dtype=np.uint8)
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            for k in range(3):
                if acc[i,j,3] != 0:
                    img[i,j,k] = int(float(acc[i,j,k]) / float(acc[i,j,3]))
                else:
                    img[i,j,k] = 0
            img[i,j,3] = 1
    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        minX_tmp, minY_tmp, maxX_tmp, maxY_tmp = imageBoundingBox(img, M)
        minX = min(minX, minX_tmp)
        minY = min(minY, minY_tmp)
        maxX = max(maxX, maxX_tmp)
        maxY = max(maxY, maxY_tmp)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    # #extra credit cropout the dark margins
    # image = croppedImage
    #
    # imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # Set threshold
    # # th1 = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,1023,0)
    # _, th2 = cv2.threshold(imgray, 8, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Find with the largest rectangle
    # areas = [cv2.contourArea(contour) for contour in contours]
    # max_index = np.argmax(areas)
    # cnt = contours[max_index]
    # x, y, w, h = cv2.boundingRect(cnt)
    #
    # # Ensure bounding rect should be at least 16:9 or taller
    # if w / h > 16 / 9:
    #     # increase top and bottom margin
    #     newHeight = w / 16 * 9
    #     y = y - (newHeight - h) / 2
    #     h = newHeight
    #
    # # Crop with the largest rectangle
    # crop = image[y:y + h, x:x + w, :]


    return crop

