import cv2
import numpy as np


def part1():
    cv2.imwrite("output/ps1-1-a-1.png", cv2.imread("input/Image1.png"))
    cv2.imwrite("output/ps1-1-a-2.png", cv2.imread("input/Image2.png"))
    return

def part2():
    image1 = cv2.imread("output/ps1-1-a-1.png")
    b,g,r = cv2.split(image1)

    #Swap red and blue
    cv2.imwrite("output/ps1-2-a-1.png", cv2.merge((r,g,b)))

    #store green channel
    cv2.imwrite("output/ps1-2-b-1.png", g)

    #store red channel
    cv2.imwrite("output/ps1-2-c-1.png", r)

    return

def part3():
    image1 = cv2.imread("output/ps1-1-a-1.png")
    image2 = cv2.imread("output/ps1-1-a-2.png")

    img1_blue, img1_green, img1_red = cv2.split(image1)
    img2_blue, img2_green, img2_red = cv2.split(image2)

    #get the center of each green channel
    img1_centerHeight = img1_green.shape[0] / 2
    img1_centerWidth = img1_green.shape[1] / 2

    img2_centerHeight = img2_green.shape[0] / 2
    img2_centerWidth = img2_green.shape[1] / 2

    centerSquare = img1_green[img1_centerHeight - 50 : img1_centerHeight + 50 : 1, img1_centerWidth-50 : img1_centerWidth+50 : 1]

    img2_green[img2_centerHeight - 50 : img2_centerHeight + 50 : 1, img2_centerWidth-50 : img2_centerWidth+50 : 1] = centerSquare

    cv2.imwrite("output/ps1-3-a-1.png", img2_green)

    return

def part4():
    image1 = cv2.imread("output/ps1-1-a-1.png")

    img1_blue, img1_green, img1_red = cv2.split(image1)

    min =  np.min(img1_green)
    max =  np.max(img1_green)
    mean =  np.mean(img1_green)
    std = np.std(img1_green)

    #Subtract the mean from all pixels,
    # then divide by standard deviation,
    # then multiply by 10 (if your image is 0 to 255)
    # or by 0.05 (if your image ranges from 0.0 to 1.0). Now add the mean back in.
    img1_green = img1_green.astype(np.float64)
    out = ((((img1_green - mean)/std) * 10) + mean)
    out = out.clip(0, 255).astype(np.uint8)
    cv2.imwrite("output/ps1-4-b-1.png", out)

    shift = np.copy(img1_green)
    shift = np.roll(shift, -2, axis=1)
    shift[:, -2] = 0
    shift[:, -1] = 0
    cv2.imwrite("output/ps1-4-c-1.png", shift)


#   Subtract the shifted version of img1_green from the original img1_green, and save the difference image.
    diff = np.clip((img1_green - shift), 0, 255).astype(np.uint8)
    cv2.imwrite("output/ps1-4-d-1.png", diff)




def part5():
    image1 = cv2.imread("output/ps1-1-a-1.png")

    img1_blue, img1_green, img1_red = cv2.split(image1)

    sigma = 16
    rand_num = np.random.randn(img1_green.shape[0], img1_green.shape[1]) * sigma

    img1_green_alter = np.copy(img1_green)
    img1_green_alter = img1_green_alter.astype(np.float64)
    img1_green_alter += rand_num
    img1_green_alter = img1_green_alter.clip(0, 255).astype(np.uint8)
    cv2.imwrite("output/ps1-5-a-1.png",cv2.merge((img1_blue, img1_green_alter, img1_red)))

    img1_blue_alter = np.copy(img1_blue)
    img1_blue_alter = img1_blue_alter.astype(np.float64)
    img1_blue_alter += rand_num
    img1_blue_alter = img1_blue_alter.clip(0,255).astype(np.uint8)
    cv2.imwrite("output/ps1-5-b-1.png",cv2.merge((img1_blue_alter, img1_green, img1_red)))

    return

part1()
part2()
part3()
part4()
part5()