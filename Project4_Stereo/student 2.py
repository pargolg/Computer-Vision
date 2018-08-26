# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
# import util_sweep
# END IMPORTS
#import imagemagick

def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    L = lights.T
    L_T = L.T
    albedo = np.zeros((images[0].shape[0], images[0].shape[1], images[0].shape[2]), dtype=np.float32)
    normals = np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.float32)
    term1 = np.linalg.inv(L_T.dot(L)) # term1 of least squares solution
    for channel in range(images[0].shape[2]):
        for row in range(images[0].shape[0]):
            for col in range(images[0].shape[1]):
                I = [(images[i][row][col][channel]).T for i in range(len(images))]
                term2 = L_T.dot(I) # term2 of least squares solution
                G = term1.dot(term2) # least squares solution
                k = np.round(np.linalg.norm(G), 5)
                if k < 1e-7:
                    k = 0
                else:
                    normals[row][col] += G/k
                albedo[row][col][channel] = k
    normals /= images[0].shape[2]
    return albedo, normals




def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """

    # calculate projection matrix
    projection_matrix = K.dot(Rt)

    height, width = points.shape[:2] # extract original shape

    projections = np.zeros((height, width, 2)) # new 2D matrix

    curr_point = np.zeros(3)

    # apply transformation to every point
    for row_i, row in enumerate(points):
        for col_j, column in enumerate(row):
            curr_point = np.array(points[row_i, col_j]) # store 3vec of current point
            fourvec = np.array([curr_point[0], curr_point[1], curr_point[2], 1.0]) # construct 4vec of current point

            homogenous_pt = projection_matrix.dot(fourvec) # calculate new point in homogenous coords
            new_pt = np.array([homogenous_pt[0]/homogenous_pt[2], homogenous_pt[1]/homogenous_pt[2]]) # calculate project point

            projections[row_i, col_j] = new_pt # add it to matrix

    return projections



def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
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
    # get image shape
    height, width, num_channels = image.shape

    window_offset = int(ncc_size/2) # calculate window offset

    patch_vector = np.zeros((ncc_size**2)) # fill patch vector with zeros

    normalized = np.zeros((height, width, (num_channels * (ncc_size**2)))) # matrix to fill

    for row_i in range(window_offset, height-window_offset):
        for col_k in range(window_offset, width-window_offset):
            # grab window
            patch_vector = image[row_i - window_offset:row_i + window_offset + 1, col_k - window_offset:col_k + window_offset + 1,:]

            # subtract channel means
            mean_vec = np.mean(np.mean(patch_vector, axis=0), axis=0)
            patch_vector = patch_vector - mean_vec

            new_vec = np.zeros((num_channels * (ncc_size**2)))

            big_index = 0

            # rearrange in the order specified
            for channel in range(num_channels):
                for row in range(patch_vector.shape[0]):
                    for col in range(patch_vector.shape[1]):
                        new_vec[big_index] = patch_vector[row,col,channel]
                        big_index += 1

            # flatten, compute norm, and divide by norm
            patch_vector = new_vec
            if(np.linalg.norm(patch_vector) >= 1e-6):
                patch_vector /= np.linalg.norm(patch_vector)
            else:
                patch_vector = np.zeros((num_channels * ncc_size**2))

            # set correct position of normalized matrix
            normalized[row_i, col_k] = patch_vector

    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    height, width = image1.shape[:2] # extract shape
    ncc = np.zeros((height, width)) # new matrix

    # iterate through each pixel and compute cross correlation at each
    for row_i in range(height):
        for col_k in range(width):
            ncc[row_i, col_k] = np.correlate(image1[row_i, col_k], image2[row_i, col_k])

    return ncc
