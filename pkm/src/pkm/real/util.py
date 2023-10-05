#!/usr/bin/env python3

import numpy as np
import cv2
import open3d as o3d
import torch as th

def o3d2th(x):
    return th.utils.dlpack.from_dlpack(x.to_dlpack())

def th2o3d(x):
    return o3d.core.Tensor.from_dlpack(
        th.utils.dlpack.to_dlpack(x)
    )


def o3d2th(x):
    return th.utils.dlpack.from_dlpack(x.to_dlpack())


def th2o3d(x):
    return o3d.core.Tensor.from_dlpack(
        th.utils.dlpack.to_dlpack(x)
    )


def T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T_v = np.eye(4)
    T_v[:3, :3] = R
    T_v[:3, 3] = t.reshape(-1)
    return T_v


def draw_pose_axes(image: np.ndarray,
                   pose: np.ndarray,
                   K: np.ndarray,
                   length: float = 0.1) -> np.ndarray:
    axes = np.array([[0, 0, 0],
                     [length, 0, 0],
                     [0, length, 0],
                     [0, 0, length]],
                    dtype=np.float32)
    # X: Red, Y: Green, Z: Blue
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    axes_cam = (pose @
                np.vstack((axes.T, np.ones(4))))[:3, :]
    # Project 3D points to 2D
    axes_2d = (K @ axes_cam).T
    axes_2d = axes_2d[:, :2] / axes_2d[:, 2:]
    # Draw axes
    for i in range(1, 4):
        # Draw line from origin to each axis tip
        cv2.line(image,
                 tuple(axes_2d[0].astype(int)),
                 tuple(axes_2d[i].astype(int)),
                 colors[i - 1],
                 2)
    return image
    # Show output image
    # cv2.imshow('color', color_image)
