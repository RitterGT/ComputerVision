"""Problem Set 2: Edges and Lines."""

import numpy as np
import cv2
import time
import os
from math import pi

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)

def hough_lines_acc(img_edges, rho_res=1, theta_res=pi/90):
    """Compute Hough Transform for lines on edge image.

    Parameters
    ----------
        img_edges: binary edge image
        rho_res: rho resolution (in pixels)
        theta_res: theta resolution (in radians)

    Returns
    -------
        H: Hough accumulator array
        rho: vector of rho values, one for each row of H
        theta: vector of theta values, one for each column of H
    """
    # TODO: Your code here
    rho_length = 2 * np.ceil(np.sqrt(np.power(img_edges.shape[0], 2) + np.power(img_edges.shape[1], 2))) / rho_res
    rho = np.arange(rho_length)

    theta = np.arange(180, dtype=np.float64)
    theta *= pi/180

    H = np.zeros((len(rho), len(theta)),dtype=np.int)


    for y,x in np.transpose(np.nonzero(img_edges)):
        for t_index in range(0, len(theta)):
            rhoVal = y * np.sin(theta[t_index]) + x * np.cos(theta[t_index])
            rhoVal = np.ceil(rhoVal + len(rho)/2)
            H[rhoVal, t_index] += 1

    return H, rho, theta


def hough_peaks(H, Q):
    """Find peaks (local maxima) in accumulator array.

    Parameters
    ----------
        H: Hough accumulator array
        Q: number of peaks to find (max)

    Returns
    -------
        peaks: Px2 matrix (P <= Q) where each row is a (rho_idx, theta_idx) pair
    """
    # TODO: Your code here
    H_copy = np.copy(H)
    indices = H_copy.ravel().argsort()[-Q:]
    indices = (np.unravel_index(i, H_copy.shape) for i in indices)
    peaks = [(i) for i in indices]

    return peaks


def hough_lines_draw(img_out, peaks, rho, theta):
    """Draw lines on an image corresponding to accumulator peaks.

    Parameters
    ----------
        img_out: 3-channel (color) image
        peaks: Px2 matrix where each row is a (rho_idx, theta_idx) index pair
        rho: vector of rho values, such that rho[rho_idx] is a valid rho value
        theta: vector of theta values, such that theta[theta_idx] is a valid theta value
    """
    for rho_idx, theta_idx in peaks:
        a = np.cos(theta[theta_idx])
        b = np.sin(theta[theta_idx])
        x0 = a*(rho[rho_idx] - len(rho)/2)
        y0 = b*(rho[rho_idx] - len(rho)/2)
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img_out,(x1,y1),(x2,y2),(0,0,255),2)

    pass  # TODO: Your code here (nothing to return, just draw on img_out directly)


def hough_circles_acc(img_edges, radiusMin, radiusMax=None):


    if radiusMax == None:
        radiusMax = radiusMin + 1

    radii_count = radiusMax - radiusMin

    theta = np.arange(90, dtype=np.float64)
    theta *= pi/180


    H = np.zeros((img_edges.shape[0], img_edges.shape[1], radii_count), dtype=np.int)

    for x,y in np.transpose(np.nonzero(img_edges)):
        for radius in range(radiusMin, radiusMax):
            a = x - radius * np.cos(theta)
            b = y + radius * np.sin(theta)
            a2 = x + radius * np.cos(theta)
            b2 = y - radius * np.sin(theta)
            for dex in range(0, len(a)):
                if a[dex] < H.shape[0] and b[dex] < H.shape[1] and a2[dex] < H.shape[0] and b2[dex] < H.shape[1]:
                    H[a[dex],b[dex],radius - radiusMin] += 1
                    H[a2[dex],b[dex],radius - radiusMin] += 1
                    H[a[dex],b2[dex],radius - radiusMin] += 1
                    H[a2[dex],b2[dex],radius - radiusMin] += 1

    return H


def boxesWork():
        # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale
    img_edges = cv2.Canny(img,100,200)
    # TODO: Compute edge image (img_edges)
    cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)  # save as ps2-1-a-1.png

    # 2-a
    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges)

    # TODO: Store accumulator array (H) as ps2-2-a-1.png
    # Note: Write a normalized uint8 version, mapping min value to 0 and max to 255
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-a-1.png'), H)


    # 2-b
    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, 6)  # TODO: implement this, try different parameters

    # TODO: Store a copy of accumulator array image (from 2-a), with peaks highlighted, as ps2-2-b-1.png
   # copiedH = np.copy(H).astype(np.uint8)
    copiedH = cv2.cvtColor(H.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    copiedH = copiedH.astype(np.uint8)
    cv2.line(copiedH, (0,0), (250,250),(0,0,0,255),2)
    for peak in peaks:
        cv2.circle(copiedH, (peak[1], peak[0]), 10, (0,0,255), 1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-b-1.png'), copiedH)

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)  # save as ps2-2-c-1.png

def boxesWork_noise():
    # 3-a
    # TODO: Read ps2-input0-noise.png, compute smoothed image using a Gaussian filter
    noise_img = cv2.imread(os.path.join(input_dir, 'ps2-input0-noise.png'), 0)  # flags=0 ensures grayscale
    blur = cv2.GaussianBlur(noise_img,(7,7),5)
    blur = cv2.medianBlur(blur, 7)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-a-1.png'), blur)

    # 3-b
    # TODO: Compute binary edge images for both original image and smoothed version
    noise_img_edges = cv2.Canny(noise_img, 100, 400)
    blur_img_edges = cv2.Canny(blur, 20, 100)

    # cv2.imshow("test", blur_img_edges)

    #Output: Two edge images: ps2-3-b-1.png (from original), ps2-3-b-2.png (from smoothed)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-1.png'), noise_img_edges)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-2.png'), blur_img_edges)


    # 3-c
    # TODO: Apply Hough methods to smoothed image, tweak parameters to find best lines
    H, rho, theta = hough_lines_acc(blur_img_edges)
    peaks = hough_peaks(H, 6)
    img_out = cv2.cvtColor(noise_img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)


    copiedH = cv2.cvtColor(H.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    copiedH = copiedH.astype(np.uint8)
    cv2.line(copiedH, (0,0), (250,250),(0,0,0,255),2)
    for peak in peaks:
        cv2.circle(copiedH, (peak[1], peak[0]), 10, (0,0,255), 1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-1.png'), copiedH)

#   Intensity image (original one with the noise) with lines drawn on them: ps2-3-c-2.png
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-2.png'), img_out)

def pen_lines_simple():
    # TODO: Like problem 3 above, but using ps2-input1.png
    noise_img = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)  # flags=0 ensures grayscale
    blur = cv2.GaussianBlur(noise_img,(7,7),11)
    #Output: Smoothed monochrome image: ps2-4-a-1.png
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-1.png'), blur)

    blur_img_edges = cv2.Canny(blur, 10, 50)
    #Output: Edge image: ps2-4-b-1.png
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1.png'), blur_img_edges)

    H, rho, theta = hough_lines_acc(blur_img_edges)
    peaks = hough_peaks(H, 4)

    #Hough accumulator array image with peaks highlighted: ps2-4-c-1.png
    copiedH = cv2.cvtColor(H.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    copiedH = copiedH.astype(np.uint8)
    cv2.line(copiedH, (0,0), (250,250),(0,0,0,255),2)
    for peak in peaks:
        cv2.circle(copiedH, (peak[1], peak[0]), 10, (0,0,255), 1)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-c-1.png'), copiedH)

    img_out = cv2.cvtColor(noise_img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)
    #Original monochrome image with lines drawn on it: ps2-4-c-2.png
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-c-2.png'), img_out)


def circles_simple():
    noise_img = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)  # flags=0 ensures grayscale
    blur = cv2.GaussianBlur(noise_img,(3,3),16)
    blur_img_edges = cv2.Canny(blur, 10, 600)
    # cv2.imshow("test", blur_img_edges)
    #
    # - Smoothed image: ps2-5-a-1.png (this may be identical to  ps2-4-a-1.png)
    # - Edge image: ps2-5-a-2.png (this may be identical to  ps2-4-b-1.png)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-1.png'), blur)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-2.png'), blur_img_edges)

    H = hough_circles_acc(blur_img_edges, 20)
    peaks = hough_peaks(H, 10)
    img_out = cv2.cvtColor(noise_img, cv2.COLOR_GRAY2BGR)
    for peak in peaks:
        cv2.circle(img_out, (peak[1], peak[0]), peak[2] + 20, (0,0,255), 2)
    #- Original monochrome image with the circles drawn in color:  ps2-5-a-3.png
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-3.png'), img_out)

    return

def circles_simple_enhanced():
    noise_img = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)  # flags=0 ensures grayscale
    blur = cv2.GaussianBlur(noise_img,(3,3),16)
    blur_img_edges = cv2.Canny(blur, 120, 500)
    # cv2.imshow("test", blur_img_edges)
    #
    # - Smoothed image: ps2-5-a-1.png (this may be identical to  ps2-4-a-1.png)
    # - Edge image: ps2-5-a-2.png (this may be identical to  ps2-4-b-1.png)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-1.png'), blur)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-2.png'), blur_img_edges)

    minRadius = 15
    maxRadius = 50
    H = hough_circles_acc(blur_img_edges, minRadius, maxRadius)
    peaks = hough_peaks(H, 20)
    img_out = cv2.cvtColor(noise_img, cv2.COLOR_GRAY2BGR)
    for peak in peaks:
        cv2.circle(img_out, (peak[1], peak[0]), peak[2] + minRadius, (0,0,255), 2)
    #- Original monochrome image with the circles drawn in color:  ps2-5-a-3.png
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-b-1.png'), img_out)

    return

def pen_lines_realistic():
    # TODO: Find lines a more realtistic image, ps2-input2.png
    noise_img = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)  # flags=0 ensures grayscale
    blur = cv2.GaussianBlur(noise_img,(3,3),7)
    blur_img_edges = cv2.Canny(blur, 10, 300)

    H, rho, theta = hough_lines_acc(blur_img_edges)
    peaks = hough_peaks(H, 10)
    img_out = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)
    #Output: Smoothed image you used with the Hough lines drawn on them: ps2-6-a-1.png
    cv2.imwrite(os.path.join(output_dir, 'ps2-6-a-1.png'), img_out)

def pen_lines_realistic_enhanced():
    # TODO: Find lines a more realtistic image, ps2-input2.png
    noise_img = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)  # flags=0 ensures grayscale
    blur = cv2.GaussianBlur(noise_img,(3,3),7)
    blur_img_edges = cv2.Canny(blur, 90, 250)
    # cv2.imshow("test", blur_img_edges)
    #input()
    H, rho, theta = hough_lines_acc(blur_img_edges)
    peaks = hough_peaks(H, 10)
    img_out = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    #for this, only draw lines with theta's that are not close to 90 or 0
    realPeaks = []
    for peak in peaks:
        if not (80<(theta[peak[1]] * 180/pi)<100 or -10<(theta[peak[1]] * 180/pi)<10 or 175<(theta[peak[1]] * 180/pi)<190):
            realPeaks.append(peak)

    hough_lines_draw(img_out, realPeaks, rho, theta)
    #Output: Smoothed image you used with the Hough lines drawn on them: ps2-6-a-1.png
    cv2.imwrite(os.path.join(output_dir, 'ps2-6-c-1.png'), img_out)
    cv2.imwrite("pen-lines-real-edges.png", blur_img_edges)

def circles_realistic():
    noise_img = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)  # flags=0 ensures grayscale

    blur = cv2.GaussianBlur(noise_img,(13,13),1)
    blur = cv2.medianBlur(blur, 5)
    blur_img_edges = cv2.Canny(blur, 130, 120)
    # cv2.imshow("test", blur_img_edges)


    minRadius = 15
    maxRadius = 50
    H = hough_circles_acc(blur_img_edges, minRadius, maxRadius)
    peaks = hough_peaks(H, 20)
    img_out = cv2.cvtColor(noise_img, cv2.COLOR_GRAY2BGR)
    for peak in peaks:
        cv2.circle(img_out, (peak[1], peak[0]), peak[2] + minRadius, (0,0,255), 2)
    cv2.imwrite(os.path.join(output_dir, 'ps2-7-a-1.png'), img_out)

    # cv2.imshow("test1", img_out)
    # input()
    return



def distorted_finds():
    noise_img = cv2.imread(os.path.join(input_dir, 'ps2-input3.png'), 0)  # flags=0 ensures grayscale

    blur = cv2.GaussianBlur(noise_img,(3,3),16)
    blur_circle_img_edges = cv2.Canny(blur, 80, 350)
    # cv2.imshow("circ", blur_circle_img_edges)
    H_circle = hough_circles_acc(blur_circle_img_edges, 20, 25)
    peaks_circles = hough_peaks(H_circle, 10)




    blur_lines = cv2.GaussianBlur(noise_img,(3,3),16)
    blur_lines = cv2.medianBlur(blur_lines, 5)
    blur_lines_img_edges = cv2.Canny(blur_lines, 10, 90)
    # cv2.imshow("lines", blur_lines_img_edges)

    #draw stuff
    H_lines, rho, theta = hough_lines_acc(blur_lines_img_edges)
    peaks_lines = hough_peaks(H_lines, 20)
    img_out = cv2.cvtColor(blur_lines, cv2.COLOR_GRAY2BGR)
    for peak in peaks_circles:
        cv2.circle(img_out, (peak[1], peak[0]), peak[2]+20, (0,0,255), 2)

    real_line_peaks = []
    for peak in peaks_lines:
        if not (80<(theta[peak[1]] * 180/pi)<100):
            real_line_peaks.append(peak)
    hough_lines_draw(img_out, real_line_peaks, rho, theta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-8-a-1.png'), img_out)

    # cv2.imshow("test1", img_out)
    # input()
    return

def main():
    """Run code/call functions to solve problems."""
    start = time.time()
    print "simple boxes start"
    boxesWork()
    boxesWork_noise()
    stop = time.time()
    print "simple boxes end, elapsed: " + str(stop - start)

    print "pen lines start"
    pen_lines_simple()
    pen_lines_realistic()
    pen_lines_realistic_enhanced()
    stop = time.time()
    print "pen lines end, elapsed: " + str(stop - start)

    print "circles start"
    circles_simple()
    circles_simple_enhanced()
    circles_realistic()
    stop = time.time()
    print "circles end, elapsed: " + str(stop - start)

    print "distorted start"
    distorted_finds()
    stop = time.time()
    print "distored end, elapsed: " + str(stop - start)


if __name__ == "__main__":
    main()

