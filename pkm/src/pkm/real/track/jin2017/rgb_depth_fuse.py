#!/usr/bin/env python

import cv2
import numpy as np

from pkm.real.track.jin2017 import bayesplane
from pkm.real.track.jin2017 import LM_minimize as lm


def rigid_transform_3D(A, B):
    # print(A.shape, B.shape)
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.dot(np.transpose(AA), BB)

    U, S, Vt = np.linalg.svd(H)

    # R = Vt.T * U.T
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        # print "Reflection detected"
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_A.T) + centroid_B.T

    # print t

    return R, t


def normal_transfomation(init_normal, goal_normal):
    vector_init = init_normal
    vector_goal = goal_normal
    vector_cross = np.cross(vector_init, vector_goal)
    vector_sin = np.linalg.norm(vector_cross)
    vector_cos = np.dot(vector_init, vector_goal)
    vector_skew = np.array([[0, -vector_cross[2], vector_cross[1]],
                            [vector_cross[2], 0, -vector_cross[0]],
                            [-vector_cross[1], vector_cross[0], 0]])
    vector_eye = np.eye(3)
    R = vector_eye + vector_skew + np.linalg.matrix_power(
        vector_skew, 2) * (1 - vector_cos) / (vector_sin * vector_sin)
    [rvec, job] = cv2.Rodrigues(R)
    return rvec


def sample_depth_plane(depth_image, image_pts, K):
    # Generate the depth samples from the depth image
    fx = K[0][0]
    fy = K[1][1]
    px = K[0][2]
    py = K[1][2]

    rows = depth_image.shape[0]
    cols = depth_image.shape[1]
    if depth_image.ndim == 3:
        depth_image = depth_image.reshape(rows, cols)
    hull_pts = image_pts.reshape(4, 1, 2).astype(int)
    # print('hull_pts', hull_pts)
    rect = cv2.convexHull(hull_pts)
    # print('rect', rect)
    all_pts = []
    if False:
        xcoord = image_pts[:, 0]
        ycoord = image_pts[:, 1]
        xmin = int(np.amin(xcoord))
        xmax = int(np.amax(xcoord))
        ymin = int(np.amin(ycoord))
        ymax = int(np.amax(ycoord))
        for j in range(ymin, ymax):
            for i in range(xmin, xmax):
                if (cv2.pointPolygonTest(rect, (i, j), False) > 0):
                    depth = depth_image[j, i]
                    if (depth != 0):
                        x = (i - px) * depth / fx
                        y = (j - py) * depth / fy
                        all_pts.append([x, y, depth])
    else:
        xcoord = image_pts[:, 0]
        ycoord = image_pts[:, 1]
        xmin = max(0, int(np.amin(xcoord)))
        xmax = min(int(np.amax(xcoord)), cols-1)
        ymin = max(0, int(np.amin(ycoord)))
        ymax = min(int(np.amax(ycoord)), rows-1)
        # print(xmin, xmax, ymin, ymax)

        depth_roi = depth_image[ymin:ymax + 1, xmin:xmax + 1]
        y, x = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
        x = (x - px) * depth_roi / fx
        y = (y - py) * depth_roi / fy
        msk = (depth_roi > 0).astype(np.uint8)

        msk2 = np.zeros_like(msk)
        cv2.fillConvexPoly(msk2, rect - (xmin, ymin), 1)
        msk = msk.astype(bool) & msk2.astype(bool)
        # print(msk.shape, x.shape, y.shape, depth_image.shape)
        # raise ValueError('stop')
        # print(x.shape, y.shape, depth_roi.shape, msk.shape)
        all_pts = np.stack([x, y, depth_roi], axis=-1)[msk]

    sample_cov = 0.9
    samples_depth = np.array(all_pts)
    # print(samples_depth.shape)
    if samples_depth.shape[0] < 3:
        return None
    # print(samples_depth.shape)

    cov = np.asarray([sample_cov] * samples_depth.shape[0])
    depth_plane_est = bayesplane.fit_plane_bayes(samples_depth, cov)
    return depth_plane_est


def computeZ(n, d, x, y):
    sum = n[0] * x + n[1] * y
    z = (d - sum) / n[2]
    return z


def getDepthPoints(image_pts, depth_plane_est, depth_image, K):
    fx = K[0][0]
    fy = K[1][1]
    px = K[0][2]
    py = K[1][2]
    all_depth_points = []
    dim = image_pts.shape
    n = depth_plane_est.mean.n
    d = depth_plane_est.mean.d
    for i in range(dim[0]):
        x = image_pts[i, 0]
        y = image_pts[i, 1]
        depth = depth_image[int(y), int(x)]  # / 1000.0 + 0.00001
        if (depth != 0):
            X = (x - px) * depth / fx
            Y = (y - py) * depth / fy
            Z = computeZ(n, d, X, Y)
            all_depth_points = all_depth_points + [[X, Y, Z]]
    all_depth_points = np.array(all_depth_points)
    return all_depth_points


def computeExtrinsics(object_pts, image_pts, depth_points, K, D, verbose=0):

    rdepth, tdepth = rigid_transform_3D(object_pts, depth_points)
    if (verbose > 0):
        print(rdepth)
        print(tdepth)

    depthH = np.eye(4)
    depthH[0:3, 0:3] = rdepth
    depthH[0:3, 3:4] = tdepth.reshape(3, 1)
    if (verbose > 0):
        print(depthH)

    rvec_init, jacob = cv2.Rodrigues(rdepth)
    tvec_init = tdepth.reshape(3, 1)
    nrvec, ntvec = cv2.solvePnPRefineLM(object_pts,
                                        image_pts,
                                        K, D,
                                        rvec_init,
                                        tvec_init)
    # nrvec, ntvec = lm.PnPMin(rvec_init, tvec_init, object_pts, image_pts, K, D)
    nrvec = nrvec.reshape(3, 1)
    ntvec = ntvec.reshape(3, 1)
    return nrvec, ntvec


def solvePnP_RGBD(depth_image, object_pts,
                  image_pts, K, D, verbose=0):
    if (depth_image.ndim == 3):
        depth_image = depth_image[:, :, 0]
    depth_plane_est = sample_depth_plane(depth_image, image_pts, K)
    if depth_plane_est is None:
        print('depth_plane_est is None')
        return None
    depth_points = getDepthPoints(image_pts, depth_plane_est, depth_image, K)
    if len(depth_points) < 4:
        return None
    return computeExtrinsics(object_pts, image_pts,
                             depth_points, K, D, verbose)
