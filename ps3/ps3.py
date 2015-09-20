"""Problem Set 3: Window-based Stereo Matching."""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
import time

import os

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)


def get_patch(matrix, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x):
    return matrix[(y - y_up_offset):(y + y_down_offset + 1), (x - x_left_offset):(x + x_right_offset + 1)]


# def disparity_ssd_2(L, R, window_size = 21):
#     """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
#     Params:
#     L: Grayscale left image, in range [0.0, 1.0]
#     R: Grayscale right image, same size as L
#     Returns: Disparity map, same size as L, R
#     """
#
#     D = np.zeros(L.shape, dtype=np.float)
#
#     # subtract 1 due to the starting pixel
#     offset = (window_size) / 2
#     L = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
#     R = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
#
#     shape = L.shape
#     height = shape[0]
#     width = shape[1]
#
#     r_shape = (R.shape[0]-(window_size-1), R.shape[1]-(window_size-1), window_size, window_size)
#     r_strides = (R.shape[1] * R.itemsize, R.itemsize, R.itemsize * R.shape[1], R.itemsize)
#     r_strips = as_strided(R, r_shape, r_strides)
#
#     l_shape = (L.shape[0]-(window_size-1), L.shape[1]-(window_size-1), window_size, window_size)
#     l_strides = (L.shape[1] * L.itemsize, L.itemsize, L.itemsize * L.shape[1], L.itemsize)
#     l_strips = as_strided(L, l_shape, l_strides)
#
#
#     for l_strip_dex in range(0, len(l_strips)):
#         for r_strip_dex in range(0, len(r_strips)):
#
#             l_strip = r_strips[l_strip_dex]
#             r_strip = r_strips[r_strip_dex]
#
#             #should be the same for both
#             filled_shape = (r_strip.shape[0], r_strip.shape[0], r_strip.shape[1], r_strip.shape[2]);
#             l_filled_strides = (0, l_strip.itemsize * l_strip.shape[1] * l_strip.shape[2], l_strip.shape[1]*l_strip.itemsize, l_strip.itemsize)
#             r_filled_strides = (l_strip.itemsize * l_strip.shape[1] * l_strip.shape[2], 0, l_strip.shape[1]*l_strip.itemsize, l_strip.itemsize)
#
#             l_filled = as_strided(l_strip, filled_shape, l_filled_strides)
#             r_filled = as_strided(r_strip, filled_shape, r_filled_strides)
#
#             ssd_res = (l_filled - r_filled**2).sum((2,3))
#             D[l_strip_dex][:] = ssd_res.argmin(0)
#
#     return D
#

def disparity_ssd(L, R, window_size = 21):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L
    Returns: Disparity map, same size as L, R
    """

    D = np.zeros(L.shape, dtype=np.float)

    # subtract 1 due to the starting pixel
    offset = (window_size) / 2
    L = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
    R = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)

    shape = L.shape
    height = shape[0]
    width = shape[1]

    r_shape = (R.shape[0]-(window_size-1), R.shape[1]-(window_size-1), window_size, window_size)
    r_strides = (R.shape[1] * R.itemsize, R.itemsize, R.itemsize * R.shape[1], R.itemsize)
    r_strips = as_strided(R, r_shape, r_strides)

    for y in range(offset, height - offset):
        r_strip = r_strips[y-offset]
        for x in range(offset, width-offset):
            l_patch = get_patch(L, offset, offset, offset, offset, y, x)

            copy_patch = np.copy(l_patch)
            l_strip = as_strided(copy_patch, r_strip.shape, (0, copy_patch.itemsize*window_size, copy_patch.itemsize))
            ssd = ((l_strip - r_strip)**2).sum((1, 2))

            x_prime = np.argmin(ssd)
            D[y-offset][x-offset] = x_prime - x

    #print D.max()
    return D

# def test_disparity_ssd2(l_image, r_image, problem, window_size = 21):
#     L = cv2.imread(os.path.join('input', l_image), 0) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
#     R = cv2.imread(os.path.join('input', r_image), 0) * (1 / 255.0)
#
#     # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
#     start = time.time()
#     D = disparity_ssd(L, R, window_size)  # TODO# : implemenet disparity_ssd()
#     print "first: " + str(time.time() - start)
#     start = time.time()
#     D2 = disparity_ssd_2(L, R, window_size)
#     print "second: " + str(time.time() - start)


    #print D == D2

    cv2.imwrite(os.path.join("output", "ps3-" + problem + ".png"), np.clip(D2, 0, 255).astype(np.uint8))

def apply_disparity_ssd(l_image, r_image, problem, window_size = 21):
    L = cv2.imread(os.path.join('input', l_image), 0) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join('input', r_image), 0) * (1 / 255.0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D = disparity_ssd(L, R, window_size)  # TODO: implemenet disparity_ssd()

    min = D.min()
    if min < 0:
        D += np.abs(min)
    max = D.max()
    D *= 255.0/max


    cv2.imwrite(os.path.join("output", "ps3-" + problem + ".png"), np.clip(D, 0, 255).astype(np.uint8))

def disparity_ncorr(L, R, window_size=19):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))

    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L
    Returns: Disparity map, same size as L, R
    """

    D = np.zeros(L.shape, dtype=np.float)

    # subtract 1 due to the starting pixel
    offset = (window_size) / 2
    L = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
    R = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)

    shape = L.shape
    height = shape[0]
    width = shape[1]

    r_shape = (R.shape[0]-(window_size-1), window_size, R.shape[1])
    r_strides = R.shape[1]*R.itemsize,R.shape[1]*R.itemsize,R.itemsize
    r_strips = as_strided(R, r_shape, r_strides)

    for y in range(offset, height - offset):
        r_strip = r_strips[y-offset]
        for x in range(offset, width-offset):
            l_patch = get_patch(L, offset, offset, offset, offset, y, x)

            result = cv2.matchTemplate(r_strip, l_patch, method=cv2.TM_CCOEFF_NORMED)
            upper_left = cv2.minMaxLoc(result)[3]

            # if(x%150 == 0):
            #     cv2.imshow("test", l_patch)
            #     cv2.imshow("strip", r_strip)
            #     print upper_left[0]
            #     input()
            # Add the left offset because that will get us the x coordinate in the patch
            x_prime = upper_left[0] + offset
            D[y-offset][x-offset] = x_prime - x

    #print D.max()
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

    cv2.imwrite(os.path.join("output", "ps3-" + problem + ".png"), np.clip(D, 0, 255).astype(np.uint8))

def create_noise_and_contrast_images():
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0)


    noiseParams=(5,11)

    L_noise = cv2.GaussianBlur(L.copy(), (noiseParams[0],noiseParams[0]), noiseParams[1])
    R_noise = cv2.GaussianBlur(R.copy(), (noiseParams[0],noiseParams[0]), noiseParams[1])

    L_contrast = L*1.1
    R_contrast = R*1.1

    cv2.imwrite(os.path.join('input', 'pair1-L-noise.png'), L_noise)
    cv2.imwrite(os.path.join('input', 'pair1-R-noise.png'), R_noise)
    cv2.imwrite(os.path.join('input', 'pair1-L-contrast.png'), L_contrast)
    cv2.imwrite(os.path.join('input', 'pair1-R-contrast.png'), R_contrast)


def getFile(name):
    return cv2.imread(os.path.join('input', name), 0)


def step5_2():
    R = cv2.imread(os.path.join('input', 'pair2-L.png'))
    L = cv2.imread(os.path.join('input', 'pair2-R.png'))

    #median filter both
    L = cv2.medianBlur(L, 3)
    R = cv2.medianBlur(R, 3)

    apply_disparity_norm(L, R, '5-a-1-ncorr', 7)


def step5():
    # R = cv2.imread(os.path.join('input', 'pair2-L.png'))
    # L = cv2.imread(os.path.join('input', 'pair2-R.png'))
    #
    #
    # L_b,L_g,L_r = cv2.split(L)
    # R_b,R_g,R_r = cv2.split(R)

    #
    # #median filter both
    # L = cv2.medianBlur(L, 5)
    # R = cv2.medianBlur(R, 5)
    #
    # cv2.imwrite(os.path.join('input', 'pair2-L-blur.png'), L)
    # cv2.imwrite(os.path.join('input', 'pair2-R-blur.png'), R)


    # cv2.imwrite(os.path.join('input', 'pair2-L-b.png'), L_b)
    # cv2.imwrite(os.path.join('input', 'pair2-R-b.png'), R_b)
    # cv2.imwrite(os.path.join('input', 'pair2-L-g.png'), L_g)
    # cv2.imwrite(os.path.join('input', 'pair2-R-g.png'), R_g)
    # cv2.imwrite(os.path.join('input', 'pair2-L-r.png'), L_r)
    # cv2.imwrite(os.path.join('input', 'pair2-R-r.png'), R_r)
    #
    #
    #
    # apply_disparity_norm('pair2-L-blur.png','pair2-R-blur.png','5-a-1', 7)
    # apply_disparity_norm('pair2-L-r.png','pair2-R-r.png','5-a-1-r', 7)
    # apply_disparity_norm('pair2-L-b.png','pair2-R-b.png','5-a-1-b', 7)
    # apply_disparity_norm('pair2-L-g.png','pair2-R-g.png','5-a-1-g', 7)

    L_b = getFile('pair2-L-b.png')
    R_b = getFile('pair2-R-b.png')
    L_g = getFile('pair2-L-g.png')
    R_g = getFile('pair2-R-g.png')
    L_r = getFile('pair2-L-r.png')
    R_r = getFile('pair2-R-r.png')

    window_size = 7
    D = np.zeros(L_b.shape, dtype=np.float)

    # subtract 1 due to the starting pixel
    offset = (window_size) / 2
    L_b = cv2.copyMakeBorder(L_b, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
    R_b = cv2.copyMakeBorder(R_b, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
    L_g = cv2.copyMakeBorder(L_g, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
    R_g = cv2.copyMakeBorder(R_g, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
    L_r = cv2.copyMakeBorder(L_r, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)
    R_r = cv2.copyMakeBorder(R_r, offset, offset, offset, offset, cv2.BORDER_CONSTANT,value=0)

    shape = L_b.shape
    height = shape[0]
    width = shape[1]

    r_shape = (R_b.shape[0]-(window_size-1), R_b.shape[1]-(window_size-1), window_size, window_size)
    r_strides = (R_b.shape[1] * R_b.itemsize, R_b.itemsize, R_b.itemsize * R_b.shape[1], R_b.itemsize)
    R_r_strips = as_strided(R_r, r_shape, r_strides)
    R_b_strips = as_strided(R_b, r_shape, r_strides)
    R_g_strips = as_strided(R_g, r_shape, r_strides)

    for y in range(offset, height - offset):
        R_r_strip = R_r_strips[y-offset]
        R_b_strip = R_b_strips[y-offset]
        R_g_strip = R_g_strips[y-offset]

        for x in range(offset, width-offset):
            L_r_patch = np.copy(get_patch(L_r, offset, offset, offset, offset, y, x))
            L_b_patch = np.copy(get_patch(L_b, offset, offset, offset, offset, y, x))
            L_g_patch = np.copy(get_patch(L_g, offset, offset, offset, offset, y, x))

            L_r_strip = as_strided(L_r_patch, R_r_strip.shape, (0, L_r_patch.itemsize*window_size, L_r_patch.itemsize))
            L_b_strip = as_strided(L_b_patch, R_b_strip.shape, (0, L_b_patch.itemsize*window_size, L_b_patch.itemsize))
            L_g_strip = as_strided(L_g_patch, R_g_strip.shape, (0, L_g_patch.itemsize*window_size, L_g_patch.itemsize))

            ssd_r = ((L_r_strip - R_r_strip)**2).sum((1, 2))
            ssd_b = ((L_b_strip - R_b_strip)**2).sum((1, 2))
            ssd_g = ((L_g_strip - R_g_strip)**2).sum((1, 2))


            ssd = ssd_b
            if(np.min(ssd_r) < np.min(ssd_g) and np.min(ssd_r) < np.min(ssd_b)):
                # print "r win"
                ssd = ssd_r
            if(np.min(ssd_b) < np.min(ssd_g) and np.min(ssd_b) < np.min(ssd_r)):
                ssd = ssd_b
                # print "b win"
            if(np.min(ssd_g) < np.min(ssd_b) and np.min(ssd_g) < np.min(ssd_r)):
                ssd = ssd_g
                # print "g win"

            x_prime = np.argmin(ssd)
            D[y-offset][x-offset] = x_prime - x

    min = D.min()
    if min < 0:
        D += np.abs(min)
    max = D.max()
    D *= 255.0/max

    cv2.imwrite(os.path.join("output", "ps3-5-a-1.png"), np.clip(D, 0, 255).astype(np.uint8))



def main():
    """Run code/call functions to solve problems."""
    # 1-a
    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    # TODO: Save output images (D_L as output/ps3-1-a-1.png and D_R as output/ps3-1-a-2.png)
    # Note: They may need to be scaled/shifted before saving to show results properly
    start = time.time()
    apply_disparity_ssd('pair0-L.png', 'pair0-R.png','1-a-1',21)
    apply_disparity_ssd('pair0-R.png', 'pair0-L.png','1-a-2',21)
    print (time.time() - start)

    # 2
    # TODO: Apply disparity_ssd() to pair1-L.png and pair1-R.png (in both directions)
    apply_disparity_ssd('pair1-L.png', 'pair1-R.png', '2-a-1', 7)
    apply_disparity_ssd('pair1-R.png', 'pair1-L.png', '2-a-2', 7)
    print (time.time() - start)



    # create noisy images:
    create_noise_and_contrast_images()


    # 3
    # TODO: Apply disparity_ssd() to noisy versions of pair1 images

    apply_disparity_ssd('pair1-L-noise.png', 'pair1-R-noise.png','3-a-1',7)
    apply_disparity_ssd('pair1-R-noise.png', 'pair1-L-noise.png','3-a-2',7)

    # TODO: Boost contrast in one image and apply again
    apply_disparity_ssd('pair1-L.png', 'pair1-R-contrast.png','3-b-1',7)
    apply_disparity_ssd('pair1-R-contrast.png', 'pair1-L.png','3-b-2',7)
    print (time.time() - start)

    #4
    #TODO: Implement disparity_ncorr() and apply to pair1 images (original, noisy and contrast-boosted)
    apply_disparity_norm('pair1-L.png', 'pair1-R.png','4-a-1',7)
    apply_disparity_norm('pair1-R.png', 'pair1-L.png','4-a-2',7)

    apply_disparity_norm('pair1-L-noise.png', 'pair1-R-noise.png','4-b-1',7)
    apply_disparity_norm('pair1-R-noise.png', 'pair1-L-noise.png','4-b-2',7)

    apply_disparity_norm('pair1-L.png', 'pair1-R-contrast.png','4-b-3',7)
    apply_disparity_norm('pair1-R-contrast.png', 'pair1-L.png','4-b-4',7)
    print (time.time() - start)


    # TODO: Apply stereo matching to pair2 images, try pre-processing the images for best results
    step5()
    print (time.time() - start)

if __name__ == "__main__":
    main()

