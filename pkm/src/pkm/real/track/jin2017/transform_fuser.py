#!/usr/bin/env python
import cv2
import numpy as np
from copy import deepcopy
from pkm.real.track.jin2017.rgb_depth_fuse import solvePnP_RGBD
from cho_util.math import transform as tx


def fuse_transform(tag_corners,
                   depth_image: np.ndarray,
                   K: np.ndarray,
                   tag_size: float = 0.015):
    # tag_corners = marker.corners2d
    tag_radius = tag_size / 2
    # corners0_r = tag_corners[0].x
    # corners0_c = tag_corners[0].y
    # corners1_r = tag_corners[1].x
    # corners1_c = tag_corners[1].y
    # corners2_r = tag_corners[2].x
    # corners2_c = tag_corners[2].y
    # corners3_r = tag_corners[3].x
    # corners3_c = tag_corners[3].y

    # image_pts = [
    #     [corners0_r, corners0_c],
    #     [corners1_r, corners1_c],
    #     [corners2_r, corners2_c],
    #     [corners3_r, corners3_c]
    # ]
    # image_pts = np.asarray(image_pts).reshape(4, 2)
    image_pts = np.asarray(tag_corners).reshape(4, 2)

    # FIXME: <3,2> order was switched.
    # is this ok?
    # old order: [--, +-, ++, -+]
    # new order: [-+, ++, +-, --]
    object_pts = [
        [-tag_radius, +tag_radius, 0.0],
        [+tag_radius, +tag_radius, 0.0],
        [+tag_radius, -tag_radius, 0.0],
        [-tag_radius, -tag_radius, 0.0],
    ]
    object_pts = np.asarray(object_pts).reshape(4, 3)

    D = np.zeros((5, 1))
    result = solvePnP_RGBD(depth_image, object_pts, image_pts, K, D)
    if result is None:
        return None
    
    rvec, tvec = result

    rotation, jacob = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[0:3, 0:3] = rotation
    T[0:3, 3] = tvec.reshape(3)
    return T
