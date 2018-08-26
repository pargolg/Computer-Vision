import time
from math import floor
import numpy as np
import cv2
import scipy
dir(scipy)
from scipy.sparse import csr_matrix


def compute_photometric_stereo_impl(lights, images):
    """
    Part1:
    Compute Albedo and Normals.

    Input:
        lights -- 3 x N array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x channels image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    height = images[0].shape[0]
    width = images[0].shape[1]
    channels = images[0].shape[2]
    numImages = len(images)

    albedo = np.zeros((height, width, channels), dtype = np.float32)
    normals = np.zeros((height, width, 3), dtype = np.float32)

    for i in range(height):
        for j in range(width):
            for k in range(channels):
                image = np.array([im[i, j, k] for im in images])
                image.reshape((numImages, 1))
                pixel = np.dot(np.linalg.inv(np.dot(lights, lights.T)), np.dot(lights, image))
                K_d = np.linalg.norm(pixel)
                if K_d < 1e-7:
                    G = np.zeros((3, 1))
                    K_d = 0
                else:
                    G = pixel / np.linalg.norm(pixel)

                albedo[i, j, k] = K_d
                normals[i, j] = G.reshape((3,))
    return albedo, normals


def project_impl(K, Rt, points):
    """
    Part 2.1

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """

    height = points.shape[0]
    width = points.shape[1]
    projections = np.zeros((height, width, 2))

    # precompute K*Rt
    multiple = np.dot(K, Rt)

    for i in range(height):
        for j in range(width):
            dots = points[i, j]
            dots = np.append(dots, 1)
            proj = np.dot(multiple, dots)
            projections[i, j, 0] = proj[0] / proj[2]
            projections[i, j, 1] = proj[1] / proj[2]
    return projections


def preprocess_ncc_impl(image, ncc_size):
    """
    part 2.2 Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x112, x121, x122, x211, x212, x221, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height, width, channels = image.shape
    normalized = np.zeros((height, width, channels * ncc_size * ncc_size), dtype = np.float32)
    half = ncc_size / 2

    for i in range(height):
        for j in range(width):
            if i - half < 0 or i + half >= height or j - half < 0 or j + half >= width:
                continue
            holder = list()
            for k in range(channels):
                selected = image[i - half: i + half + 1, j - half: j + half + 1, k]
                selected = (selected - np.mean(selected)).flatten()
                holder.append(selected.T)
            arr = np.hstack(tuple(holder)).flatten()
            norms = np.linalg.norm(arr)
            if norms < 1e-6:
                arr.fill(0)
            else:
                arr = arr / norms
            normalized[i, j] = arr
    return normalized


def compute_ncc_impl(image1, image2):
    """
    part 2.3 output ncc
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    height = image1.shape[0]
    width = image1.shape[1]
    ncc = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            ncc[i, j] = np.correlate(image1[i, j], image2[i, j])[0]
    return ncc
