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
    # tmp = cv2.getGaussianKernel(sum_window, 1)
    # kernel = np.dot(tmp, np.transpose(tmp))

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

    #Clip UV extremes
    # percentile = 3
    # U[np.where(U < np.percentile(U, percentile))] = 0
    # U[np.where(U > np.percentile(U, 100-percentile))] = 0
    # V[np.where(V < np.percentile(V, percentile))] = 0
    # V[np.where(V > np.percentile(V, 100-percentile))] = 0
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
        # print i
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
    max_level = 5
    k = max_level

    gy_pyr_a = gaussian_pyramid(A, k)
    gy_pyr_b = gaussian_pyramid(B, k)

    U = None
    V = None
    for i in reversed(range(0,k)):

        # print i
        # print k
        Ak = gy_pyr_a[i]
        Bk = gy_pyr_b[i]
        # print Ak.shape
        # print Bk.shape

        if i == k - 1:
            # print "max level"
            U = np.zeros(Ak.shape)
            V = np.zeros(Ak.shape)
        else:
            # print "at level" + str(i)
            # print "init U shape: " + str(U.shape)
            # print "init V shape: " + str(V.shape)

            U = 2*expand(U)
            V = 2*expand(V)
        Ck = warp(Bk, U, V)
        dx, dy = optic_flow_LK(Ak, Ck, 3* (k-i+1), False)
        # print "dx, dy shapes:" + str(dx.shape) +  "  " + str(dy.shape)

        U_pad = U.shape[0] - dx.shape[0]
        if U_pad > 0:
            dx = np.pad(dx,((0,U_pad),(0,U_pad)),mode='constant')

        V_pad = V.shape[0] - dy.shape[0]
        if V_pad > 0:
            dy = np.pad(dy,((0,V_pad),(0,V_pad)),mode='constant')

        U = U + dx
        V = V + dy
        #
        # print "U shape: " + str(U.shape)
        # print "V shape: " + str(V.shape)

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
    write_image(im_color, preFix + ".png")

    stride = 15  # plot every so many rows, columns
    color = (0, 255, 0)  # green
    img_out = np.zeros((V.shape[0], U.shape[1], 3), dtype=np.uint8)
    # print 'drawing'
    # print U
    # print V
    for y in xrange(0, V.shape[0], stride):
        for x in xrange(0, U.shape[1], stride):
            cv2.arrowedLine(img_out, (x, y), (x + int(U[y, x] * scale), y + int(V[y, x] * scale)), color, 1, tipLength=.3)

    write_image(img_out, preFix + "-quiver.png")
    return im_color

def pair_images(img_list):
    original = img_list[0]
    saved_shape = original.shape
    for dex in range(1, len(img_list)):
        # print original.shape
        # print img_list[dex].shape
        tmp = np.zeros((saved_shape[0], img_list[dex].shape[1]))
        tmp[0:img_list[dex].shape[0], 0:img_list[dex].shape[1]] = img_list[dex]
        original = np.concatenate((original, tmp), axis=1)

    return original

def one_a():
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    ShiftR2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'), 0) / 255.0
    U, V = optic_flow_LK(Shift0, ShiftR2, 31, False, 5)
    make_pair_and_quiver(U,V, "ps6-1-a-1", 5)

    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    ShiftR5U5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0) / 255.0
    U, V = optic_flow_LK(Shift0, ShiftR5U5, 61, False, 5)
    make_pair_and_quiver(U,V, "ps6-1-a-2", 5)


def one_b():
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.0
    U, V = optic_flow_LK(Shift0, ShiftR10, 91, False, 7)
    make_pair_and_quiver(U,V, "ps6-1-b-1", 9)

    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.0
    U, V = optic_flow_LK(Shift0, ShiftR20, 91, False, 9)
    make_pair_and_quiver(U,V, "ps6-1-b-2", 11)

    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.0
    U, V = optic_flow_LK(Shift0, ShiftR40, 91, False, 11)
    make_pair_and_quiver(U,V, "ps6-1-b-3", 15)




def two():
    # # 2a
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)
    # # TODO: Save pyramid images as a single side-by-side image (write a utility function?)
    img_out = pair_images(yos_img_01_g_pyr)
    norm_and_write_image(img_out, "ps6-2-a-1.png")
    #
    # # 2b
    yos_img_01_l_pyr = laplacian_pyramid(yos_img_01_g_pyr)
    # # TODO: Save pyramid images as a single side-by-side image
    img_out = pair_images(yos_img_01_l_pyr)
    norm_and_write_image(img_out, "ps6-2-b-1.png")

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


def three_a_1():
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_img_02 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.0
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)
    yos_img_02_g_pyr = gaussian_pyramid(yos_img_02, 4)

    level = 2
    U, V = optic_flow_LK(yos_img_01_g_pyr[level], yos_img_02_g_pyr[level],21,True)
    for i in range(0, level):
        U = expand(U) * 2
        V = expand(V) * 2

    make_pair_and_quiver(U,V, "ps6-3-a-1", 5)
    yos_img_02_warped = warp(yos_img_02, U, V)
    diff = yos_img_02_warped - yos_img_01
    norm_and_write_image(diff, "ps6-3-a-2.png")

    norm_and_write_image(yos_img_01, "pr3-img1.png")
    norm_and_write_image(yos_img_02_warped, "pr3-im2-warp.png")
    makeGif("pr3-img1.png", "pr3-im2-warp.png", yos_img_01.shape, "ps6-3-a-2.gif")


def three_a_2():
    img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '0.png'), 0) / 255.0
    img_02 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '1.png'), 0) / 255.0
    img_01_g_pyr = gaussian_pyramid(img_01, 5)
    img_02_g_pyr = gaussian_pyramid(img_02, 5)

    level = 2
    U, V = optic_flow_LK(img_01_g_pyr[level], img_02_g_pyr[level],7,False, 31)
    for i in range(0, level):
        U = expand(U) * 2
        V = expand(V) * 2

    make_pair_and_quiver(U,V, "ps6-3-a-3", 5)
    img_02_warped = warp(img_02, U, V)
    diff = img_02_warped - img_01
    norm_and_write_image(diff, "ps6-3-a-4.png")

    norm_and_write_image(img_01, "pr3-img1-2.png")
    norm_and_write_image(img_02_warped, "pr3-im2-warp-2.png")
    makeGif("pr3-img1-2.png", "pr3-im2-warp-2.png", img_01.shape, "ps6-3-a-4.gif")


def four_a():
    # 4a
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.0
    ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.0
    ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.0
    U10, V10 = hierarchical_LK(Shift0, ShiftR10, 9)  # TODO: implement this
    U20, V20 = hierarchical_LK(Shift0, ShiftR20, 9)
    U40, V40 = hierarchical_LK(Shift0, ShiftR40, 15)
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

    pair1 = make_pair_and_quiver(U10, V10, "4a_R10", 2)
    pair2 = make_pair_and_quiver(U20, V20, "4a_R20", 2)
    pair3 = make_pair_and_quiver(U40, V40, "4a_R40", 2)
    write_image(np.vstack((pair1, pair2, pair3)), "ps6-4-a-1.png")

    # # TODO: Save difference between each warped image and original image (Shift0), stacked
    diff_10 = ShiftR10_warped - Shift0
    diff_20 = ShiftR20_warped - Shift0
    diff_40 = ShiftR40_warped - Shift0

    stackedDiff = np.vstack((diff_10, diff_20, diff_40))
    cv2.normalize(stackedDiff, stackedDiff, 0, 255, cv2.NORM_MINMAX)
    stackedDiff = stackedDiff.astype(np.uint8)
    norm_and_write_image(stackedDiff, "ps6-4-a-2.png")


def four_b():
    # 4a
    yos_1 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_2 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.0
    yos_3 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.0
    U1, V1 = hierarchical_LK(yos_1, yos_2, 9)
    U2, V2 = hierarchical_LK(yos_1, yos_3, 9)
    warped_1 = warp(yos_1, U1, V1)
    warped_2 = warp(yos_2, U2, V2)


    pair1 = make_pair_and_quiver(U1, V1, "4b_yos_1", 2)
    pair2 = make_pair_and_quiver(U2, V2, "4b_yos_2", 2)
    write_image(np.vstack((pair1, pair2)), "ps6-4-b-1.png")

    # # TODO: Save difference between each warped image and original image (Shift0), stacked
    diff_10 = warped_1 - yos_1
    diff_20 = warped_2 - yos_1

    stackedDiff = np.vstack((diff_10, diff_20))
    cv2.normalize(stackedDiff, stackedDiff, 0, 255, cv2.NORM_MINMAX)
    stackedDiff = stackedDiff.astype(np.uint8)
    norm_and_write_image(stackedDiff, "ps6-4-b-2.png")

def four_c():
    # 4a
    im_1 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '0.png'), 0) / 255.0
    im_2 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '1.png'), 0) / 255.0
    im_3 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '2.png'), 0) / 255.0
    U1, V1 = hierarchical_LK(im_1, im_2, 9)  # TODO: implement this
    U2, V2 = hierarchical_LK(im_1, im_3, 9)
    # # TODO: Save displacement image pairs (U, V), stacked
    # # Hint: You can use np.concatenate()
    warped_1 = warp(im_1, U1, V1)
    warped_2 = warp(im_1, U2, V2)


    pair1 = make_pair_and_quiver(U1, V1, "4c_1", 2)
    pair2 = make_pair_and_quiver(U2, V2, "4c_2", 2)
    write_image(np.vstack((pair1, pair2)), "ps6-4-c-1.png")

    # # TODO: Save difference between each warped image and original image (Shift0), stacked
    diff_10 = warped_1 - im_1
    diff_20 = warped_2 - im_1

    stackedDiff = np.vstack((diff_10, diff_20))
    cv2.normalize(stackedDiff, stackedDiff, 0, 255, cv2.NORM_MINMAX)
    stackedDiff = stackedDiff.astype(np.uint8)
    norm_and_write_image(stackedDiff, "ps6-4-c-2.png")



def five():
    # 4a
    im_1 = cv2.imread(os.path.join(input_dir, 'Juggle', '1.png'), 0) / 255.0
    im_2 = cv2.imread(os.path.join(input_dir, 'Juggle', '2.png'), 0) / 255.0
    U1, V1 = hierarchical_LK(im_1, im_2, 9)  # TODO: implement this
    warped_1 = warp(im_2, U1, V1)

    make_pair_and_quiver(U1, V1, "ps6-5-a-1", 2)

    norm_and_write_image(im_1, "pr5-im1.png")
    norm_and_write_image(warped_1, "pr5-im2-warp.png")

    # # TODO: Save difference between each warped image and original image (Shift0), stacked
    diff = warped_1 - im_1
    norm_and_write_image(diff, "ps6-5-a-2.png")

    makeGif("pr5-im1.png", "pr5-im2-warp.png", im_1.shape, "juggle.gif")

# Driver code
def main():
    # Note: Comment out parts of this code as necessary
    one_a()
    one_b()


    #2:
    two()

    #3a
    three_a_1()
    three_a_2()

    #testWarp()
    four_a()
    # 4b
    four_b()
    # 4c
    four_c()
    five()


if __name__ == "__main__":
    main()
