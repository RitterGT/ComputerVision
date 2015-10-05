"""Problem Set 5: Harris, SIFT, RANSAC."""

import numpy as np
import cv2
import os
import itertools

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
def gradientX(image):
    """Compute image gradient in X direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
    """


    # TODO: Your code here

    ret = np.zeros(image.shape)


    for i in range(0, len(image)):
        for j in range(0, len(image[0])-1):
            ret[i, j] = float(image[i, j+1]) - float(image[i, j])

    return ret


def gradientY(image):
    """Compute image gradient in Y direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Iy: image gradient in Y direction, values in [-1.0, 1.0]
    """

    # TODO: Your code here
    ret = np.zeros(image.shape)

    for i in range(0,len(image)-1):
        for j in range(0, len(image[0])):
            ret[i,j] = float(image[i+1, j]) - float(image[i, j])

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


def harris_response(Ix, Iy, kernel, alpha):
    """Compute Harris reponse map using given image gradients.

    Parameters
    ----------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
        Iy: image gradient in Y direction, same size and type as Ix
        kernel: 2D windowing kernel with weights, typically square
        alpha: Harris detector parameter multiplied with square of trace

    Returns
    -------
        R: Harris response map, same size as inputs, floating-point
    """

    # TODO: Your code here

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    R = Sxx*Syy - alpha*((Sxx - Syy)**2)
    return R


def find_corners(R, threshold, radius):
    """Find corners in given response map.

    Parameters
    ----------
        R: floating-point response map, e.g. output from the Harris detector
        threshold: response values less than this should not be considered plausible corners
        radius: radius of circular region for non-maximal suppression (could be half the side of square instead)

    Returns
    -------
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates
    """


    # TODO: Your code here

    #first clip out below threshold:
    R[R < threshold] = 0

    #now grab all the windows and zero out eveyrthing but the localMax
    dX = radius
    dY = radius

    M, N = R.shape
    for x in range(0,M-dX+1):
        for y in range(0,N-dY+1):
            window = R[x:x+dX, y:y+dY]
            if np.sum(window)==0:
                localMax=0
            else:
                localMax = np.amax(window)
            maxCoord = np.argmax(window)
            # zero all but the localMax in the window
            window[:] = 0
            window.flat[maxCoord] = localMax

    return np.transpose(np.nonzero(R))
    #now get the non zero points


def draw_corners(image, corners):
    """Draw corners on (a copy of) given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates

    Returns
    -------
        image_out: copy of image with corners drawn on it, color (BGR), uint8, values in [0, 255]
    """


    image_out = np.copy(image)
    image_out *= 255
    image_out = image_out.astype(np.uint8)

    image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)

    for corner in corners:
        cv2.circle(image_out, (corner[1], corner[0]), 3, (255,0,0))

    # TODO: Your code here
    return image_out


def gradient_angle(Ix, Iy):
    """Compute angle (orientation) image given X and Y gradients.

    Parameters
    ----------
        Ix: image gradient in X direction
        Iy: image gradient in Y direction, same size and type as Ix

    Returns
    -------
        angle: gradient angle image, each value in degrees [0, 359)
    """

    # TODO: Your code here
    # Note: +ve X axis points to the right (0 degrees), +ve Y axis points down (90 degrees)
    angle = np.degrees(np.arctan2(Ix, Iy))

    return angle


def get_keypoints(points, R, angle, _size, _octave=0):
    """Create OpenCV KeyPoint objects given interest points, response and angle images.

    Parameters
    ----------
        points: interest points (e.g. corners), as a sequence (list) of (x, y) coordinates
        R: floating-point response map, e.g. output from the Harris detector
        angle: gradient angle (orientation) image, each value in degrees [0, 359)
        _size: fixed _size parameter to pass to cv2.KeyPoint() for all points
        _octave: fixed _octave parameter to pass to cv2.KeyPoint() for all points

    Returns
    -------
        keypoints: a sequence (list) of cv2.KeyPoint objects
    """

    # TODO: Your code here
    # Note: You should be able to plot the keypoints using cv2.drawKeypoints() in OpenCV 2.4.9+

    keypoints = []
    for point in points:
        keypoints.append(cv2.KeyPoint(point[1], point[0], _size, angle[point[0]][point[1]], R[point[0]][point[1]], _octave))

    return keypoints


def get_descriptors(image, keypoints):
    """Extract feature descriptors from image at each keypoint.

    Parameters
    ----------
        keypoints: a sequence (list) of cv2.KeyPoint objects

    Returns
    -------
        descriptors: 2D NumPy array of shape (len(keypoints), 128)
    """

    # TODO: Your code here
    # Note: You can use OpenCV's SIFT.compute() method to extract descriptors, or write your own!
    sift = cv2.SIFT()
    features, descriptors = sift.compute(image.astype(np.uint8), keypoints)

    return descriptors


def match_descriptors(desc1, desc2, maxResults = None):
    """Match feature descriptors obtained from two images.

    Parameters
    ----------
        desc1: descriptors from image 1, as returned by SIFT.compute()
        desc2: descriptors from image 2, same format as desc1

    Returns
    -------
        matches: a sequence (list) of cv2.DMatch objects containing corresponding descriptor indices
    """

    desc1 = desc1.astype(np.uint8)
    desc2 = desc2.astype(np.uint8)

 # Match descriptors.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
        #Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    if maxResults is not None:
        matches = matches[:maxResults]


    # TODO: Your code here
    # Note: You can use OpenCV's descriptor matchers, or roll your own!
    return matches


def draw_matches(image1, image2, kp1, kp2, matches):
    """Show matches by drawing lines connecting corresponding keypoints.

    Parameters
    ----------
        image1: first image
        image2: second image, same type as first
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matching keypoint index pairs (as cv2.DMatch objects)

    Returns
    -------
        image_out: image1 and image2 joined side-by-side with matching lines; color image (BGR), uint8, values in [0, 255]
    """

    # TODO: Your code here
    # Note: DO NOT use OpenCV's match drawing function(s)! Write your own :)
    margin = 0
    joined_image = make_image_pair(image1, image2, margin)

    for match in matches:
        image_1_point = (int(kp1[match.queryIdx].pt[0]),
                         int(kp1[match.queryIdx].pt[1]))
        image_2_point = (int(kp2[match.trainIdx].pt[0] + \
                             image1.shape[1] + margin),
                       int(kp2[match.trainIdx].pt[1]))

        cv2.circle(joined_image, image_1_point, 5, (0, 0, 255), thickness = -1)
        cv2.circle(joined_image, image_2_point, 5, (0, 255, 0), thickness = -1)
        cv2.line(joined_image, image_1_point, image_2_point, (255, 0, 0), \
                 thickness = 1)
    return joined_image


def compute_translation_RANSAC(kp1, kp2, matches, threshold = 20):
    """Compute best translation vector using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        translation: translation/offset vector <x, y>, NumPy array of shape (2, 1)
        good_matches: consensus set of matches that agree with this translation
    """

    # TODO: Your code here

    #gross - maintain a list, for every delta, loop the list check if it's within bounds, increment.
    buckets = {}


    maxKey = None
    maxValue = 0
    for match in matches:
        point1 = np.array(kp1[match.queryIdx].pt)
        point2 = np.array(kp2[match.trainIdx].pt)

        arrDiff = point2 - point1
        difference = tuple(arrDiff)

        #add each transaltion into the buckets
        if buckets.get(difference) is None:
            buckets[difference] = []
            buckets[difference].append(match)

    for match in matches:
        point1 = np.array(kp1[match.queryIdx].pt)
        point2 = np.array(kp2[match.trainIdx].pt)

        arrDiff = point2 - point1

        #vote for any key that's close.
        for key in buckets.keys():
            if np.linalg.norm(np.array(key) - arrDiff) < threshold:
                buckets[key].append(match)
                if len(buckets[key]) > maxValue:
                    maxValue = len(buckets[key])
                    maxKey = key

    #return translation, good_matches
    return maxKey, buckets[maxKey]


def compute_similarity_RANSAC(kp1, kp2, matches, threshold = 20, maxConsensus = 15):

    np.random.shuffle(matches)

    buckets = {}
    maxValue = 0
    maxKey = None
    maxTransform = None

    for pair in itertools.product(matches, repeat=2):
        if pair[0] == pair[1]:
            continue

        transform = calc_transform_similarity(pair, kp1, kp2)
        #print transform

        key = tuple(map(tuple, transform))
        buckets[key] = []
        consensusCount = 0
        for match in matches:
            query_pt = np.array(kp1[match.queryIdx].pt)
            train_pt = np.array(kp2[match.trainIdx].pt)

            new_pt = np.dot(transform, np.append(query_pt, 1))
            diff = np.linalg.norm(new_pt - train_pt)
            if diff < threshold:
                consensusCount += 1
                buckets[key].append(match)
                if consensusCount > maxValue:
                    maxValue = consensusCount
                    maxKey = key
                    maxTransform = transform
                    # print "New Max Sim: " + str(consensusCount)

        if consensusCount > maxConsensus:
            break

    # #return translation, good_matches
    # a = maxKey[0]
    # b = maxKey[1]
    # c = maxKey[2]
    # d = maxKey[3]
    # similarity_transform = np.array([[a, -b, c],[b, a, d]])
    matches = buckets[maxKey]
    # print "MATCHES:"
    # print matches
    # print len(matches)

    return maxTransform, matches


def calc_transform_similarity(pair, kp1, kp2):
    match1 = pair[0]
    match2 = pair[1]



    u,v = np.array(kp1[match1.queryIdx].pt)
    u_prime, v_prime = np.array(kp2[match1.trainIdx].pt)

    x,y = np.array(kp1[match2.queryIdx].pt)
    x_prime, y_prime = np.array(kp2[match2.trainIdx].pt)

    b = np.array([u_prime, v_prime, x_prime, y_prime])
    a = np.array([[u, -v, 1, 0], [v, u, 0, 1], [x, -y, 1, 0], [y, x, 0, 1]])

    a,b,c,d = np.linalg.solve(a, b)

    similarity_transform = np.array([[a, -b, c],[b, a, d]])

    return similarity_transform

def compute_affine_RANSAC(kp1, kp2, matches, threshold = 20, maxConsensus = 15):

    """Compute best similarity transform using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        transform: similarity transform matrix, NumPy array of shape (2, 3)
        good_matches: consensus set of matches that agree with this transform
    """

    # TODO: Your code here
    # for each pair of matches, compute a tranform, add to a dict


    # for each pair of matches, compute the transform - for each bucket, calculate the distance, increment if 'close'
    np.random.shuffle(matches)

    buckets = {}
    maxValue = 0
    maxKey = None

    for triple in itertools.product(matches, repeat=3):
        if triple[0] == triple[1] or triple[0] == triple[2] or triple[1] == triple[2]:
            continue


        transform = calc_transform_affine(triple, kp1, kp2)
        if transform is None:
            continue

       # print transform

        key = tuple(map(tuple, transform))
        buckets[key] = []
        consensusCount = 0
        for match in matches:
            query_pt = np.array(kp1[match.queryIdx].pt)
            train_pt = np.array(kp2[match.trainIdx].pt)

            new_pt = np.dot(transform, np.append(query_pt, 1))
            diff = np.linalg.norm(new_pt - train_pt)
            if diff < threshold:
                consensusCount += 1
                buckets[key].append(match)
                if consensusCount > maxValue:
                    maxValue = consensusCount
                    maxKey = key
                    maxTransform = transform
                    # print "New Max Affine: " + str(consensusCount)

        if consensusCount > maxConsensus:
            break

    good_matches = buckets[maxKey]

    return maxTransform, good_matches


def calc_transform_affine(triple, kp1, kp2):
    match1 = triple[0]
    match2 = triple[1]
    match3 = triple[2]



    u,v = np.array(kp1[match1.queryIdx].pt)
    u_prime, v_prime = np.array(kp2[match1.trainIdx].pt)

    x,y = np.array(kp1[match2.queryIdx].pt)
    x_prime, y_prime = np.array(kp2[match2.trainIdx].pt)

    p,q = np.array(kp1[match3.queryIdx].pt)
    p_prime, q_prime = np.array(kp2[match3.trainIdx].pt)

    b = np.array([u_prime, v_prime, x_prime, y_prime, p_prime, q_prime])
    a = np.array([[u, v, 1, 0, 0, 0],
                  [0, 0, 0, u, v, 1],
                  [x, y, 1, 0, 0, 0],
                  [0, 0, 0, x, y, 1],
                  [p, q, 1, 0, 0, 0],
                  [0, 0, 0, p, q, 1]])

    # a = np.array([[u, 0, x, 0, p, 0],
    #              [v, 0, y, 0, q, 0],
    #              [1, 0, 1, 0, 1, 0],
    #              [0, u, 0, x, 0, p],
    #              [0, v, 0, y, 0, q],
	# 	           [0, 1, 0, 1, 0, 1]])
    try:
        a, b, c, d, e, f = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        # print "Fail whale: " + str((match1, match2, match3))
        # print ((u,v), (u_prime, v_prime))
        # print ((x,y), (x_prime, y_prime))
        # print ((p,q), (p_prime, q_prime))
        return None
    transform = np.array([[a, b, c],
                         [d,  e, f]], dtype=np.float)
    return transform

def get_image(filename):
    return cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0

def write_image(image, filename):
    cv2.imwrite(os.path.join(output_dir, filename), image)

def norm_and_write_image(image, filename):
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    write_image(image, filename)

def get_image_gradients_paired(filename):
    image = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    cv2.GaussianBlur(image, (3,3), 7)
    image_Ix = gradientX(image)  
    image_Iy = gradientY(image) 
    image_pair = make_image_pair(image_Ix, image_Iy)  
    return image, image_Ix, image_Iy, image_pair

# Driver code
def main():
    # Note: Comment out parts of this code as necessary

    kernel1D = cv2.getGaussianKernel(3, 7)
    kernel = kernel1D * np.transpose(kernel1D)
    #kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 1a
    transA, transA_Ix, transA_Iy,transA_pair = get_image_gradients_paired("transA.jpg")
    norm_and_write_image(transA_pair, "ps5-1-a-1.png")

    # TODO: Similarly for simA.jpg
    simA, simA_Ix, simA_Iy,simA_pair = get_image_gradients_paired("simA.jpg")
    norm_and_write_image(simA_pair, "ps5-1-a-2.png")

    # # 1b
    transA_R = harris_response(transA_Ix, transA_Iy, kernel, 0.02)
    norm_and_write_image(transA_R, "ps5-1-b-1.png")

    transB, transB_Ix, transB_Iy,transB_pair = get_image_gradients_paired("transB.jpg")
    transB_R = harris_response(transB_Ix, transB_Iy, kernel, 0.02)
    norm_and_write_image(transB_R, "ps5-1-b-2.png")


    simA_R = harris_response(simA_Ix, simA_Iy, kernel, 0.02)
    norm_and_write_image(simA_R, "ps5-1-b-3.png")


    simB, simB_Ix, simB_Iy,simB_pair = get_image_gradients_paired("simB.jpg")
    simB_R = harris_response(simB_Ix, simB_Iy, kernel, 0.02)
    norm_and_write_image(simB_R, "ps5-1-b-4.png")


    # 1c
    transA_corners = find_corners(transA_R, 50, 10)
    print "transA corners " + str(len(transA_corners))
    transA_out = draw_corners(transA, transA_corners)
    norm_and_write_image(transA_out, "ps5-1-c-1.png")

    transB_corners = find_corners(transB_R, 50, 10)
    print "transB corners " + str(len(transB_corners))
    transB_out = draw_corners(transB, transB_corners)
    norm_and_write_image(transB_out, "ps5-1-c-2.png")

    simA_corners = find_corners(simA_R, 40, 5)
    print "simA corners " + str(len(simA_corners))

    simA_out = draw_corners(simA, simA_corners)
    norm_and_write_image(simA_out, "ps5-1-c-3.png")

    simB_corners = find_corners(simB_R, 40, 5)
    print "simB corners " + str(len(simB_corners))

    simB_out = draw_corners(simB, simB_corners)
    norm_and_write_image(simB_out, "ps5-1-c-4.png")



    # # 2a
    transA_angle = gradient_angle(transA_Ix, transA_Iy)
    transA_kp = get_keypoints(transA_corners, transA_R, transA_angle, _size=10.0, _octave=0)
    transA = cv2.imread(os.path.join(input_dir, "transA.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    transA_out = cv2.drawKeypoints(transA, transA_kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    transB_angle = gradient_angle(transB_Ix, transB_Iy)
    transB_kp = get_keypoints(transB_corners, transB_R, transB_angle, _size=10.0, _octave=0)
    transB = cv2.imread(os.path.join(input_dir, "transB.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    transB_out = cv2.drawKeypoints(transB, transB_kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    trans_paired = make_image_pair(transA_out, transB_out)
    write_image(trans_paired, "ps5-2-a-1.png")


    simA_angle = gradient_angle(simA_Ix, simA_Iy)
    simA_kp = get_keypoints(simA_corners, simA_R, simA_angle, _size=10.0, _octave=0)
    simA = cv2.imread(os.path.join(input_dir, "simA.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    simA_out = cv2.drawKeypoints(simA, simA_kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    simB_angle = gradient_angle(simB_Ix, simB_Iy)
    simB_kp = get_keypoints(simB_corners, simB_R, simB_angle, _size=10.0, _octave=0)
    simB = cv2.imread(os.path.join(input_dir, "simB.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    simB_out = cv2.drawKeypoints(simB, simB_kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    sim_paired = make_image_pair(simA_out, simB_out)
    write_image(sim_paired, "ps5-2-a-2.png")

    # TODO: Ditto for (simA, simB) pair

    # 2b
    transA_desc = get_descriptors(transA, transA_kp)
    transB_desc = get_descriptors(transB, transB_kp)
    trans_matches = match_descriptors(transA_desc, transB_desc)
    # print trans_matches
    trans_matched = draw_matches(transA, transB, transA_kp, transB_kp, trans_matches)
    write_image(trans_matched, "ps5-2-b-1.png")

    simA_desc = get_descriptors(simA, simA_kp)
    simB_desc = get_descriptors(simB, simB_kp)
    sim_matches = match_descriptors(simA_desc, simB_desc)
    sim_matched = draw_matches(simA, simB, simA_kp, simB_kp, sim_matches)
    write_image(sim_matched, "ps5-2-b-2.png")

     # 3a
    # TODO: Compute translation vector using RANSAC for (transA, transB) pair, draw biggest consensus set
    translation, matchSets = compute_translation_RANSAC(transA_kp, transB_kp, trans_matches, 20)
    print "translation"
    print translation
    print len(trans_matches)
    print len(matchSets)
    ransac_trans_match = draw_matches(transA, transB, transA_kp, transB_kp, matchSets)
    write_image(ransac_trans_match, "ps5-3-a-1.png")

    # 3b
    # TODO: Compute similarity transform for (simA, simB) pair, draw biggest consensus set
    sim_matrix, matchSets = compute_similarity_RANSAC(simA_kp, simB_kp, sim_matches, 10, 20)
    print "sim matrix"
    print sim_matrix
    print len(sim_matches)
    print len(matchSets)
    ransac_sim_match = draw_matches(simA, simB, simA_kp, simB_kp, matchSets)
    write_image(ransac_sim_match, "ps5-3-b-1.png")
    # # Extra credit: 3c, 3d, 3e


    #3c
    affine_matrix, matchSets = compute_affine_RANSAC(simA_kp, simB_kp, sim_matches, 10, 14)
    print "affine matrix"
    print affine_matrix
    print len(sim_matches)
    print len(matchSets)
    ransac_sim_match = draw_matches(simA, simB, simA_kp, simB_kp, matchSets)
    write_image(ransac_sim_match, "ps5-3-c-2.png")

    #3d

    # sim_matrix = np.array([[0.94355048,  -0.33337659,  56.75110304],
    #                        [0.33337659,   0.94355048, -67.81053724]])
    # simB = cv2.imread(os.path.join(input_dir, "simB.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    # simA = cv2.imread(os.path.join(input_dir, "simA.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    rand = np.zeros((simB.shape[0], simB.shape[1]))
    out_warp  = cv2.invertAffineTransform(sim_matrix)
    warped_image = cv2.warpAffine(simB.astype(np.uint8),out_warp, (simB.shape[1], simB.shape[0]), flags=cv2.INTER_LINEAR)
    write_image(warped_image, "ps5-3-d-1.png")
    merged = cv2.merge((rand.astype(np.uint8),warped_image.astype(np.uint8),simA.astype(np.uint8)))
    write_image(merged, "ps5-3-d-2.png")


    out_warp  = cv2.invertAffineTransform(affine_matrix)

    warped_image = cv2.warpAffine(simB.astype(np.uint8),out_warp, (simB.shape[1], simB.shape[0]), flags=cv2.INTER_LINEAR)
    write_image(warped_image, "ps5-3-e-1.png")
    merged = cv2.merge((rand.astype(np.uint8),warped_image.astype(np.uint8),simA.astype(np.uint8)))
    write_image(merged, "ps5-3-e-2.png")



if __name__ == "__main__":
    main()
