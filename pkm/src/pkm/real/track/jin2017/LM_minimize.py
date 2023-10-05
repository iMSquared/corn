
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from cho_util.math import transform as tx


def quatAngleDiff(rvec1, rvec2):
    rot1, jac1 = cv2.Rodrigues(rvec1)
    rot2, jac2 = cv2.Rodrigues(rvec2)
    quat1 = tx.rotation.quaternion.from_matrix(rot1)
    quat2 = tx.rotation.quaternion.from_matrix(rot2)

    dtheta = math.acos(2 * (np.dot(quat1, quat2)**2) - 1)
    return math.degrees(dtheta)


def model(x, K, D, object_pt):
    # rot1 = x[0]
    # rot2 = x[1]
    # rot3 = x[2]
    # trans1 = x[3]
    # trans2 = x[4]
    # trans3 = x[5]
    # rvec = np.array([rot1, rot2, rot3])
    rvec = x[0:3]
    tvec = x[3:6]
    # tvec = np.array([trans1, trans2, trans3])
    err, jacob = cv2.projectPoints(object_pt, rvec, tvec, K, D)
    # ret = np.array([err[0, 0], err[1, 0], err[2, 0], err[3, 0]])
    ret = err[:4, 0]
    return ret


def residual(x, K, D, object_pt, image_pt):
    diff = model(x, K, D, object_pt) - image_pt
    # print('diff', diff.shape)
    # sum_diff = np.concatenate([diff[0], diff[1], diff[2], diff[3]])
    sum_diff = diff[:4].ravel()
    return sum_diff


def jac(x, K, D, object_pt, image_pt):
    # rot1 = x[0]
    # rot2 = x[1]
    # rot3 = x[2]
    # trans1 = x[3]
    # trans2 = x[4]
    # trans3 = x[5]
    # rvec = np.array([rot1, rot2, rot3])
    # tvec = np.array([trans1, trans2, trans3])
    rvec = x[0:3]
    tvec = x[3:6]

    err, jacob = cv2.projectPoints(object_pt, rvec, tvec, K, D)
    jacob = jacob[:, :6]
    return jacob


def PnPMin(rvec, tvec, object_pt, image_pt, I, D):
    x0 = np.append(rvec, tvec)
    K = I
    D = np.zeros((5, 1))
    relaxation = 100
    bounds = ([x0[0] -
               relaxation, x0[1] -
               relaxation, x0[2] -
               relaxation, -
               np.inf, -
               np.inf, -
               np.inf], [x0[0] +
              relaxation, x0[1] +
              relaxation, x0[2] +
              relaxation, np.inf, np.inf, np.inf])
    res = least_squares(
        residual,
        x0,
        jac=jac,
        args=(K, D, object_pt, image_pt),
        verbose=0,
        bounds=bounds)
    # res = least_squares(residual, x0, args=(K, D, object_pt, image_pt), method='lm', verbose = 1)
    rvec_r = res.x[0:3]
    tvec_r = res.x[3:6]
    return rvec_r, tvec_r
