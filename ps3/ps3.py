"""Problem Set 3: Window-based Stereo Matching."""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2

import os

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)


def get_patch(matrix, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x):
    return matrix[(y - y_up_offset):(y + y_down_offset + 1), (x - x_left_offset):(x + x_right_offset + 1)]

def disparity_ssd(L, R, window_size = 21):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L
    Returns: Disparity map, same size as L, R
    """


    # kernel_size = 5
    # sigma = 3
    # cv2.GaussianBlur(L.copy(), (kernel_size,kernel_size), sigma)
    # cv2.GaussianBlur(R.copy(), (kernel_size,kernel_size), sigma)

    D = np.zeros(L.shape, dtype=np.float)

    # subtract 1 due to the starting pixel
    offset = (window_size) / 2
    L = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
    R = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)

    shape = L.shape
    height = shape[0]
    width = shape[1]
    left_shift = False
    right_shift = False
    search_range = width / 3

    r_shape = (R.shape[0]-(window_size-1), R.shape[1]-(window_size-1), window_size, window_size)
    r_strides = (R.shape[1] * R.itemsize, R.itemsize, R.itemsize * R.shape[1], R.itemsize)
    r_strips = as_strided(R, r_shape, r_strides)

    for y in range(offset, height - offset):
        """ Compute Y Offsets """
        # y_up_offset = offset if y >= offset else y
        # y_down_offset = offset if y + offset < height else height - y - 1
        # print "y d off:", y_down_offset
        r_strip = r_strips[y-offset]
        for x in range(offset, width-offset):
            """ Compute X Offsets """
            # x_left_offset = offset if x >= offset else x
            # x_right_offset = offset if x + offset < width else width - x - 1

            # l_patch = get_patch(L, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x)
            l_patch = get_patch(L, offset, offset, offset, offset, y, x)

            copy_patch = np.copy(l_patch)
            l_strip = as_strided(copy_patch, r_strip.shape, (0, copy_patch.itemsize*window_size, copy_patch.itemsize))
            ssd = ((l_strip - r_strip)**2).sum((1, 2))

            x_prime = np.argmin(ssd)
            D[y-offset][x-offset] = x_prime - x

    print D.max()
    return D


def apply_disparity_ssd(l_image, r_image, problem, window_size = 21):
    L = cv2.imread(os.path.join('input', l_image), 0) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join('input', r_image), 0) * (1 / 255.0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = disparity_ssd(L, R, window_size)  # TODO: implemenet disparity_ssd()
    D_R = disparity_ssd(R, L, window_size)

    min = D_L.min()
    if min < 0:
        D_L += np.abs(min)
    max = D_L.max()
    D_L *= 255.0/max

    min = D_R.min()
    if min < 0:
        D_R += np.abs(min)
    D_R *= 255.0/D_R.max()

    cv2.imwrite(os.path.join("output", "ps3-" + problem + "-a-1.png"), np.clip(D_L, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join("output", "ps3-" + problem + "-a-2.png"), np.clip(D_R, 0, 255).astype(np.uint8))

def disparity_ncorr(L, R, window_size=19):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))

    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L
    Returns: Disparity map, same size as L, R
    """

    D = np.zeros(L.shape, dtype=np.float)

    # subtract 1 due to the starting pixel
    offset = (window_size - 1) / 2
    shape = L.shape
    height = shape[0]
    width = shape[1]
    for y in range(height):
        """ Compute Y Offsets """
        y_up_offset = offset if y >= offset else y
        y_down_offset = offset if y + offset < height else height - y - 1
        # print "y d off:", y_down_offset

        for x in range(width):
            """ Compute X Offsets """
            x_left_offset = offset if x >= offset else x
            x_right_offset = offset if x + offset < width else width - x - 1

            l_patch = get_patch(L, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x)


            strip = R[(y - y_up_offset):(y + y_down_offset + 1), :]
            print l_patch
            print strip
            result = cv2.matchTemplate(strip, l_patch, method=cv2.TM_CCOEFF_NORMED)
            upper_left = cv2.minMaxLoc(result)[3]
            # Add the left offset because that will get us the x coordinate in the patch
            x_prime = upper_left[1] + x_left_offset
            D[y][x] = x_prime - x

    return D

def apply_disparity_norm(l_image, r_image, problem, window_size = 21):
    L = cv2.imread(os.path.join('input', l_image), 0)
    R = cv2.imread(os.path.join('input', r_image), 0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D = disparity_ncorr(L, R, window_size)

    min = D.min()
    if min < 0:
        D += np.abs(min)
    max = D.max()
    D *= 255.0/max

    cv2.imwrite(os.path.join("output", "ps3-" + problem + "-a-1.png"), np.clip(D, 0, 255).astype(np.uint8))

def main():
    """Run code/call functions to solve problems."""

    # 1-a
    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    # TODO: Save output images (D_L as output/ps3-1-a-1.png and D_R as output/ps3-1-a-2.png)
    # Note: They may need to be scaled/shifted before saving to show results properly
    apply_disparity_ssd('pair0-L.png', 'pair0-R.png','1',21)  # TODO: implemenet disparity_ssd()

    # 2
    # TODO: Apply disparity_ssd() to pair1-L.png and pair1-R.png (in both directions)
    #apply_disparity_ssd('pair1-L.png', 'pair1-R.png','2',21)  # TODO: implemenet disparity_ssd()

    # 3
    # TODO: Apply disparity_ssd() to noisy versions of pair1 images
    #apply_disparity_ssd('pair1-D_L.png', 'pair1-D_R.png','3',21)  # TODO: implemenet disparity_ssd()

    # TODO: Boost contrast in one image and apply again

    # 4
    # TODO: Implement disparity_ncorr() and apply to pair1 images (original, noisy and contrast-boosted)

    # 5
    # TODO: Apply stereo matching to pair2 images, try pre-processing the images for best results


if __name__ == "__main__":
    main()
