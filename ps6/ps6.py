"""Problem Set 6: Optic Flow."""

import numpy as np
import cv2
import scipy
from scipy import signal
import subprocess
import os

# I/O directories
input_dir = "input"
output_dir = "output"

# Assignment code
def gradientX(image, naive=False):
    """Compute image gradient in X direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
    """


    # TODO: Your code here

    if(naive):
        ret = np.zeros(image.shape)
        for i in range(0, len(image)):
            for j in range(0, len(image[0])-1):
                ret[i, j] = float(image[i, j+1]) - float(image[i, j])

        return ret

    else:
        kernel = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], dtype=np.float64)
        kernel /= 8
        ret = cv2.filter2D(image, -1, kernel)
        return ret


def gradientY(image, naive=False):
    """Compute image gradient in Y direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Iy: image gradient in Y direction, values in [-1.0, 1.0]
    """

    if naive:
        ret = np.zeros(image.shape)

        for i in range(0,len(image)-1):
            for j in range(0, len(image[0])):
                ret[i,j] = float(image[i+1, j]) - float(image[i, j])
        return ret


    else:
        kernel = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float64)
        kernel /= 8
        ret = cv2.filter2D(image, -1, kernel)

        return ret



def make_image_pair(image1, image2, margin=0):
    """Adjoin two images side-by-side to make a single new image.

    Parameters
    ----------
        image1: first image, could be grayscale or color (BGR)
        image2: second image, same type as first

    Returns
    -------
        image_pair: combination of both images, side-by-side, same type
    """

    # TODO: Your code here
    # Compute number of channels.
    num_channels = 1
    if len(image1.shape) == 3:
        num_channels = image1.shape[2]

    image_pair = np.zeros((max(image1.shape[0], image2.shape[0]),
                           image1.shape[1] + image2.shape[1] + margin,
                           3))
    if num_channels == 1:
        for channel_idx in range(3):
            image_pair[:image1.shape[0],
                       :image1.shape[1],
                       channel_idx] = image1
            image_pair[:image2.shape[0],
                       image1.shape[1] + margin:,
                       channel_idx] = image2
    else:
        image_pair[:image1.shape[0], :image1.shape[1]] = image1
        image_pair[:image2.shape[0], image1.shape[1] + margin:] = image2
    return image_pair


# Assignment code
def optic_flow_LK(img_A, img_B, sum_window=31, useNaive=False, blur_sigma=None):
    """Compute optic flow using the Lucas-Kanade method.

    Parameters
    ----------
        A: grayscale floating-point image, values in [0.0, 1.0]
        B: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
    """


    if blur_sigma is not None:
        kernelSize = (5,5)
        cv2.GaussianBlur(img_A,kernelSize, blur_sigma)
        cv2.GaussianBlur(img_B,kernelSize, blur_sigma)

    Ix = gradientX(img_A, useNaive)
    Iy = gradientY(img_A, useNaive)
    It = img_B-img_A

    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy
    Ixt = Ix*It
    Iyt = Iy*It


    kernel = np.ones((sum_window,sum_window), np.float32)/(sum_window * sum_window)
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    Sxt = cv2.filter2D(Ixt, -1, kernel)
    Syt = cv2.filter2D(Iyt, -1, kernel)

    # Sxx = cv2.blur(Ixx, kernel)
    # Syy = cv2.blur(Iyy, kernel)
    # Sxy = cv2.blur(Ixy, kernel)
    # Sxt = cv2.blur(Ixt, kernel)
    # Syt = cv2.blur(Iyt, kernel)

    U = np.zeros(img_A.shape).astype(np.float64)
    V = np.zeros(img_B.shape).astype(np.float64)

    for y, x in np.ndindex(img_A.shape):
        A = np.array([[Sxx[y][x], Sxy[y][x]],
                     [Sxy[y][x], Syy[y][x]]])
        B = np.array([-Sxt[y][x],
                     -Syt[y][x]])


        # print A
        # print B
        # print 'Solved'
        if np.linalg.det(A) < .000001:
            U[y][x] = 0
            V[y][x] = 0
        else:
            solution, residuals, rank, s = np.linalg.lstsq(A, B)
            U[y][x] = solution[0]
            V[y][x] = solution[1]

    return U, V


def reduce(image):
    """Reduce image to the next smaller level.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        reduced_image: same type as image, half size
    """

    # TODO: Your code here
    convolved = scipy.signal.convolve(image, generatingKernel(0.4), 'same')
    return convolved[0:len(convolved):2, 0:len(convolved[0]):2]

def gaussian_pyramid(image, levels):
    """Create a Gaussian pyramid of given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        levels: number of levels in the resulting pyramid

    Returns
    -------
        g_pyr: Gaussian pyramid, with g_pyr[0] = image
    """

    # TODO: Your code here
    output = [image]
    # WRITE YOUR CODE HERE.
    for i in range(0, levels-1):
        newIm = reduce(image)
        output.append(newIm)
        image = newIm

    return output


def expand(image):
    """Expand image to the next larger level.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        reduced_image: same type as image, double size
    """

    # TODO: Your code here
    retImage = np.zeros((len(image)*2, len(image[0])*2), dtype=float)
    for i in range(0,len(image)):
        for j in range(0, len(image[0])):
            retImage[2*i][2*j] = image[i][j]

    convolved = scipy.signal.convolve(retImage, generatingKernel(0.4), 'same')
    return convolved * 4


def laplacian_pyramid(g_pyr):
    """Create a Laplacian pyramid from a given Gaussian pyramid.

    Parameters
    ----------
        g_pyr: Gaussian pyramid, as returned by gaussian_pyramid()

    Returns
    -------
        l_pyr: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1]
    """

    # TODO: Your code here
    output = []
    for i in range(0, len(g_pyr)-1):
        print i
        current = g_pyr[i]
        next = g_pyr[i+1]
        expanded = expand(next)
        if expanded.shape[0] != current.shape[0]:
            expanded = expanded[:-1]
        if expanded.shape[1] != current.shape[1]:
            expanded = expanded[:, :-1]
        output.append(current-expanded)
    output.append(g_pyr[len(g_pyr)-1])
    return output

def warp(image, U, V):
    """Warp image using X and Y displacements (U and V).

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        warped: warped image, such that warped[y, x] = image[y + V[y, x], x + U[y, x]]

    """

    # TODO: Your code here
    warped = np.zeros(image.shape)

    warped[0] = image[0]
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            try:
                warped[y][x] = image[y + V[y, x]][x + U[y, x]]
            except:
                do = 1
               # print "whoops"
    return warped



def hierarchical_LK(A, B, sumWindow=31):
    """Compute optic flow using the Hierarchical Lucas-Kanade method.

    Parameters
    ----------
        A: grayscale floating-point image, values in [0.0, 1.0]
        B: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
    """
    max_level = 4
    k = max_level

    gy_pyr_a = gaussian_pyramid(A, k)
    gy_pyr_b = gaussian_pyramid(B, k)

    U = None
    V = None
    for i in reversed(range(0,k)):
        print i
        print k
        Ak = gy_pyr_a[i]
        Bk = gy_pyr_b[i]
        print Ak.shape
        print Bk.shape

        if i == k - 1:
            print "max level"
            U = np.zeros(Ak.shape)
            V = np.zeros(Ak.shape)
        else:
            print "at level" + str(i)
            print "init U shape: " + str(U.shape)
            print "init V shape: " + str(V.shape)

            U = 2*expand(U)
            V = 2*expand(V)
        Ck = warp(Bk, U, V)
        dx, dy = optic_flow_LK(Ak, Ck, sumWindow, True)
        print "dx, dy shapes:" + str(dx.shape) +  "  " + str(dy.shape)
        U = U + dx
        V = V + dy

        print "U shape: " + str(U.shape)
        print "V shape: " + str(V.shape)

    return U, V

    # TODO: Your code here
    # return U, V

def generatingKernel(parameter):
  kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                     0.25, 0.25 - parameter /2.0])
  return np.outer(kernel, kernel)

def norm_and_write_image(image, filename):
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    write_image(image, filename)

def write_image(image, filename):
    cv2.imwrite(os.path.join(output_dir, filename), image)

def make_pair_and_quiver(U, V, preFix, scale=5):
    cpU = np.copy(U)
    cpV = np.copy(V)

    pair = make_image_pair(cpU, cpV)
    cv2.normalize(pair, pair, 0, 255, cv2.NORM_MINMAX)
    pair = pair.astype(np.uint8)
    im_color = cv2.applyColorMap(pair, cv2.COLORMAP_JET)
    write_image(im_color, preFix + " - pair.png")

    stride = 15  # plot every so many rows, columns
    color = (0, 255, 0)  # green
    img_out = np.zeros((V.shape[0], U.shape[1], 3), dtype=np.uint8)
    print 'drawing'
    print U
    print V
    for y in xrange(0, V.shape[0], stride):
        for x in xrange(0, U.shape[1], stride):
            cv2.arrowedLine(img_out, (x, y), (x + int(U[y, x] * scale), y + int(V[y, x] * scale)), color, 1, tipLength=.3)

    write_image(img_out, preFix + " - quiver.png")
    return im_color

def pair_images(img_list):
    original = img_list[0]
    saved_shape = original.shape
    for dex in range(1, len(img_list)):
        print original.shape
        print img_list[dex].shape
        tmp = np.zeros((saved_shape[0], img_list[dex].shape[1]))
        tmp[0:img_list[dex].shape[0], 0:img_list[dex].shape[1]] = img_list[dex]
        original = np.concatenate((original, tmp), axis=1)

    return original
# Driver code
def main():
    # Note: Comment out parts of this code as necessary

    # #1a
    # Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    # ShiftR2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'), 0) / 255.0
    # U, V = optic_flow_LK(Shift0, ShiftR2, 21)  # TODO: implement this
    # make_pair_and_quiver(U,V, "pr1", 5)
    #
    #
    # # TODO: Similarly for Shift0 and ShiftR5U5
    # Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    # ShiftR5U5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0) / 255.0
    # U, V = optic_flow_LK(Shift0, ShiftR5U5, 61)  #
    # make_pair_and_quiver(U,V, "pr2", 5)

    # # 1b
    # # TODO: Similarly for ShiftR10, ShiftR20 and ShiftR40
    #

    # # 2a:
    #two_a()

    # # 3a
    #three_a()

    #testWarp()
    four_a()
    # # 4b
    # # TODO: Repeat for DataSeq1 (use yos_img_01.png as the original)
    #
    # # 4c
    # # TODO: Repeat for DataSeq1 (use 0.png as the original)

def two_a():
    # # 2a
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)
    # # TODO: Save pyramid images as a single side-by-side image (write a utility function?)
    img_out = pair_images(yos_img_01_g_pyr)
    norm_and_write_image(img_out, "2a.png")
    #
    # # 2b
    yos_img_01_l_pyr = laplacian_pyramid(yos_img_01_g_pyr)
    # # TODO: Save pyramid images as a single side-by-side image
    img_out = pair_images(yos_img_01_l_pyr)
    norm_and_write_image(img_out, "2b.png")

def testWarp():
    shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    shiftR2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'), 0) / 255.0
    U=np.zeros((240, 320),dtype=np.float32)
    V=np.zeros((240, 320),dtype=np.float32)
    U[68:113 + 68, 66:191 + 66] = 2 # or -2 depends which direction you warp
    warped = warp(shiftR2, U, V)

    norm_and_write_image(shift0, "test1.png")
    norm_and_write_image(warped, "test2.png")
    makeGif("test1.png", "test2.png", shift0.shape, "test_gif.gif")

def makeGif(image1_name, image2_name, image_shape, gif_name):
    convertcommand = ["convert", "-delay", "30", "-size", str(image_shape[1]) + "x" + str(image_shape[0])] \
                     + ["/Users/Jake/Programming/cs6476/ComputerVision/ps6/output/" + image1_name,
                        "/Users/Jake/Programming/cs6476/ComputerVision/ps6/output/" + image2_name] + [gif_name]
    subprocess.call(convertcommand)


def three_a():
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_img_02 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.0
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)
    yos_img_02_g_pyr = gaussian_pyramid(yos_img_02, 4)

    print 'yos size'
    print yos_img_02.shape
    # # TODO: Select appropriate pyramid *level* that leads to best optic flow estimation
    level = 2
    U, V = optic_flow_LK(yos_img_01_g_pyr[level], yos_img_02_g_pyr[level],21,True)
    # # TODO: Scale up U, V to original image size (note: don't forget to scale values as well!)
    for i in range(0, level):
        U = expand(U) * 2
        V = expand(V) * 2

    print 'uv size:'
    print U.shape
    print V.shape

    # # TODO: Save U, V as side-by-side false-color image or single quiver plot
    make_pair_and_quiver(U,V, "pr3", 5)
    #
    yos_img_02_warped = warp(yos_img_02, U, V)
    # # TODO: Save difference image between yos_img_02_warped and original yos_img_01
    paired_out = make_image_pair(yos_img_01, yos_img_02_warped)
    norm_and_write_image(yos_img_01, "pr3-img1.png")
    norm_and_write_image(yos_img_02_warped, "pr3-im2-warp.png")
    norm_and_write_image(paired_out, "pr3-pair.png")

    diff = yos_img_02_warped - yos_img_01
    norm_and_write_image(diff, "diff.png")


    makeGif("pr3-img1.png", "pr3-im2-warp.png", yos_img_01.shape, "anim.gif")

    # # Note: Scale values such that zero difference maps to neutral gray, max -ve to black and max +ve to white
    #
    # # Similarly, you can compute displacements for yos_img_02 and yos_img_03 (but no need to save images)
    #
    # # TODO: Repeat for DataSeq2 (save images)
    #
def four_a():
    # 4a
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.0
    ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.0
    ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.0
    U10, V10 = hierarchical_LK(Shift0, ShiftR10, 11)  # TODO: implement this
    U20, V20 = hierarchical_LK(Shift0, ShiftR20, 31)
    U40, V40 = hierarchical_LK(Shift0, ShiftR40, 61)
    # # TODO: Save displacement image pairs (U, V), stacked
    # # Hint: You can use np.concatenate()
    ShiftR10_warped = warp(ShiftR10, U10, V10)
    ShiftR20_warped = warp(ShiftR20, U20, V20)
    ShiftR40_warped = warp(ShiftR40, U40, V40)


    # img_out = np.vstack((np.hstack((U10, V10)), np.hstack((U20, V20)), np.hstack((U40, V40))))
    # cv2.normalize(img_out, img_out, 0, 255, cv2.NORM_MINMAX)
    # img_out = img_out.astype(np.uint8)
    # im_color = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    # write_image(im_color, "TEST.png")

    pair1 = make_pair_and_quiver(U10, V10, "4a_R10 ")
    pair2 = make_pair_and_quiver(U20, V20, "4a_R20 ")
    pair3 = make_pair_and_quiver(U40, V40, "4a_R40 ")
    write_image(np.vstack((pair1, pair2, pair3)), "4a-stacked-pairs.png")

    # # TODO: Save difference between each warped image and original image (Shift0), stacked
    diff_10 = ShiftR10_warped - Shift0
    diff_20 = ShiftR20_warped - Shift0
    diff_40 = ShiftR40_warped - Shift0



if __name__ == "__main__":
    main()
