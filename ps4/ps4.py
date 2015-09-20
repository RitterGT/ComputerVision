"""Problem Set 4: Geometry."""

import numpy as np
import cv2
import random

import os

# I/O directories
input_dir = "input"
output_dir = "output"

# Input files
PIC_A = "pic_a.jpg"
PIC_A_2D = "pts2d-pic_a.txt"
PIC_A_2D_NORM = "pts2d-norm-pic_a.txt"
PIC_B = "pic_b.jpg"
PIC_B_2D = "pts2d-pic_b.txt"
SCENE = "pts3d.txt"
SCENE_NORM = "pts3d-norm.txt"

# Utility code
def read_points(filename):
    """Read point data from given file and return as NumPy array."""
    with open(filename) as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            pts.append(map(float, line.split()))
    return np.array(pts)


# Assignment code
def solve_least_squares(pts3d, pts2d):
    """Solve for transformation matrix M that maps each 3D point to corresponding 2D point using the least squares method.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        pts2d: corresponding 2D (image) points, NumPy array of shape (N, 2)

    Returns
    -------
        M: transformation matrix, NumPy array of shape (3, 4)
        error: sum of squared residuals of all points
    """
    N = pts2d.shape[0]
    A = np.zeros((2*N, 11))
    B = np.zeros((2*N, 1))


    for dex in range(0, N):
        u = pts2d[dex][0]
        v = pts2d[dex][1]
        X = pts3d[dex][0]
        Y = pts3d[dex][1]
        Z = pts3d[dex][2]
        A[2*dex] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z])
        B[2*dex] = u

        A[2*dex + 1] = np.array([ 0, 0, 0, 0,X, Y, Z, 1, -v*X, -v*Y, -v*Z])
        B[2*dex + 1] = v

    m, residuals, rank, singular_values = np.linalg.lstsq(A,B)

    error = ((B - (A * m.reshape((m.shape[1], m.shape[0])))) ** 2).sum()

    m_prime = np.zeros((m.shape[0]+1, m.shape[1]))
    m_prime[:-1] = m
    m_prime[-1] = 1
    M = m_prime.reshape((3, 4))

    return M, error


def project_points(pts3d, M):
    """Project each 3D point to 2D using matrix M.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        M: projection matrix, NumPy array of shape (3, 4)

    Returns
    -------
        pts2d_projected: projected 2D points, NumPy array of shape (N, 2)
    """

    # TODO: Your code here
    N = (pts3d.shape[0])
    pts2d_projected = np.zeros((N, 2))

    #tack on a column of ones
    lastColumn = np.ones((N,1))
    newPoints = np.concatenate((pts3d, lastColumn),1)

    #loop, dot product with M, normalize, set value in pointa2d
    for dex in range(0, N):
        homogenous_proj_point = np.dot(M, newPoints[dex])
        inhomogenous_proj = homogenous_proj_point / homogenous_proj_point[2]
        pts2d_projected[dex] = np.array([inhomogenous_proj[0], inhomogenous_proj[1]])

    return pts2d_projected    


def get_residuals(pts2d, pts2d_projected):
    """Compute residual error for each point.

    Parameters
    ----------
        pts2d: observed 2D (image) points, NumPy array of shape (N, 2)
        pts2d_projected: 3D (object) points projected to 2D, NumPy array of shape (N, 3) or (N, 2)

    Returns
    -------
        residuals: residual error for each point (L2 distance between observed and projected 2D points)
    """
    residuals = np.linalg.norm(pts2d - pts2d_projected, axis=1)
    print residuals
    # TODO: Your code here
    return residuals


def calibrate_camera(pts3d, pts2d):
    """Find the best camera projection matrix given corresponding 3D and 2D points.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        pts2d: corresponding 2D (image) points, NumPy array of shape (N, 2)

    Returns
    -------
        bestM: best transformation matrix, NumPy array of shape (3, 4)
        error: sum of squared residuals of all points for bestM
    """


    k_choices = [12,16,20]
    indexes = np.arange(pts3d.shape[0])

    best_m_error = None
    lowest_avg_residual = None
    for k in k_choices:
        lowest_avg_residual_per_k = None
        best_m_error_per_k = None
        for n in range(0,10):
            #choose k point indexes
            all_pt_dexes = random.sample(indexes, k)

            #first four points are for testing
            test_point_dexes = all_pt_dexes[:4]

            #rest are for doing the projection
            proj_point_dexes = all_pt_dexes[5:]

            proj_pts2d = pts2d[proj_point_dexes]
            proj_pts3d = pts3d[proj_point_dexes]
            M, error= solve_least_squares(proj_pts3d, proj_pts2d)

            avg_residual = np.average(get_residuals(proj_pts2d,project_points(proj_pts3d, M)))

            if lowest_avg_residual_per_k is None or avg_residual < lowest_avg_residual_per_k:
                lowest_avg_residual_per_k = avg_residual
                best_m_error_per_k = (M,error)

        if lowest_avg_residual is None or lowest_avg_residual_per_k  < lowest_avg_residual:
            lowest_avg_residual = lowest_avg_residual_per_k
            best_m_error = best_m_error_per_k


    # TODO: Your code here
    # NOTE: Use the camera calibration procedure in the problem set
    return best_m_error[0], best_m_error[1]


def compute_fundamental_matrix(pts2d_a, pts2d_b):
    """Compute fundamental matrix given corresponding points from 2 images of a scene.

    Parameters
    ----------
        pts2d_a: 2D points from image A, NumPy array of shape (N, 2)
        pts2d_b: corresponding 2D points from image B, NumPy array of shape (N, 2)

    Returns
    -------
        F: the fundamental matrix
    """

    N = pts2d_a.shape[0]
    A = np.zeros((N,8))
    B = np.ones((N,1)) * -1
    print B


    for dex in range(0, N):
        u = pts2d_a[dex][0]
        v = pts2d_a[dex][1]
        u_prime = pts2d_b[dex][0]
        v_prime = pts2d_b[dex][1]

        A[dex] = np.array([u*u_prime, v*u_prime, u_prime, u*v_prime, v*v_prime, v_prime, u, v])


    m, residuals, rank, singular_values = np.linalg.lstsq(A, B)

    m_prime = np.zeros((m.shape[0]+1, m.shape[1]))
    m_prime[:-1] = m
    m_prime[-1] = 1
    F = m_prime.reshape((3, 3))


    print F
    return F


# Driver code
def main():
    """Driver code."""

    # 1a
    # Read points
    pts3d_norm = read_points(os.path.join(input_dir, SCENE_NORM))
    pts2d_norm_pic_a = read_points(os.path.join(input_dir, PIC_A_2D_NORM))

    # Solve for transformation matrix using least squares
    M, error = solve_least_squares(pts3d_norm, pts2d_norm_pic_a)  # TODO: implement this

    # Project 3D points to 2D
    pts2d_projected = project_points(pts3d_norm, M)  # TODO: implement this

    # Compute residual error for each point
    residuals = get_residuals(pts2d_norm_pic_a, pts2d_projected)  # TODO: implement this

    # TODO: Print the <u, v> projection of the last point, and the corresponding residual
    print pts2d_projected[-1]
    print residuals[-1]

    # 1b
    # Read points
    pts3d = read_points(os.path.join(input_dir, SCENE))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))
    # NOTE: These points are not normalized

    # TODO: Use the functions from 1a to implement calibrate_camera() and find the best transform (bestM)
    bestM, error = calibrate_camera(pts3d, pts2d_pic_b)

    # 1c
    # TODO: Compute the camera location using bestM

    #split off the last column
    Q = bestM[:3,:3]
    m4 = bestM[:3, -1].reshape((3,1))
    center = -1 * np.dot(np.linalg.inv(Q), m4)
    print center

    # 2a
    # TODO: Implement compute_fundamental_matrix() to find the raw fundamental matrix
    pts2d_pic_a = read_points(os.path.join(input_dir, PIC_A_2D))
    F = compute_fundamental_matrix(pts2d_pic_a, pts2d_pic_b)

    # 2b
    # TODO: Reduce the rank of the fundamental matrix
    u,s,v = np.linalg.svd(F)
    s[s.argmin()] = 0
    s = np.diag(s)
    newF = np.dot(np.dot(u,s),v)


    print "RANK DOWN"
    print newF
    # 2c
    # TODO: Draw epipolar lines



    pic_b = cv2.imread("input/pic_b.jpg")
    pic_a = cv2.imread("input/pic_a.jpg")

    print pic_b.shape

    num_row = pic_b.shape[0]
    num_col = pic_b.shape[1]


    lastColumn = np.ones((pts2d_pic_a.shape[0], 1))
    newPoints_pic_a = np.concatenate((pts2d_pic_a, lastColumn), 1)

    for point in newPoints_pic_a:
        l_b = np.dot(newF, point)
        l_L = np.cross([0, 0, 1],[0, num_row, 1])
        l_R = np.cross([num_col,0, 1], [num_col, num_row, 1])
        P_i_L = np.cross(l_b, l_L)
        P_i_R = np.cross(l_b, l_R)

        print "edges"
        print l_L
        print l_R

        print P_i_L
        print P_i_R

        x1 = int(P_i_L[0] / P_i_L[2])
        y1 = int(P_i_L[1] / P_i_L[2])
        x2 = int(P_i_R[0] / P_i_R[2])
        y2 = int(P_i_R[1] / P_i_R[2])


        print "points"
        print ((x1, y1),(x2,y2))
        cv2.circle(pic_a, (int(point[0]), int(point[1])), 4, (0,255,0), 1)
        cv2.line(pic_b, (x1,y1),(x2,y2), (255,0,0), 1)
    cv2.imshow("foo", pic_b)
    cv2.imshow("bar", pic_a)
    input()

    # l_edge = np.cross(np.array([0,0]), np.array([0,pic_b.shape[0]]))
    # print l_edge
    # r_edge = np.cross(np.array((pic_b.shape[1], 0)), np.array((pic_b.shape[1],pic_b.shape[0])))
    # print r_edge
    #
    # print testpoint
    # li = np.dot(F,testpoint)
    # print li
    #
    # line = np.cross(li, l_edge)


if __name__ == '__main__':
    main()
