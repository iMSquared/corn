#!/usr/bin/env python3

from typing import Optional, Dict, Any, Iterable, Union
from dataclasses import dataclass
import numpy as np
import torch as th
import open3d as o3d
import pickle
from yourdfpy import URDF
from cho_util.math import transform as tx
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import cv2
import os
from pathlib import Path


from pkm.util.torch_util import dcn
from pkm.util.vis.img import digitize_image
from pkm.util.path import ensure_directory
from pkm.data.transforms.io_xfm import scene_to_mesh
from pkm.real.calib.optimize_transforms import (
    average_rotation_matrix,
    optimize_transforms,
    normalize_two_cloud,
)

from pkm.data.transforms.col import (
    IsInRobot,
    IsOnRobot,
    IsInRobotPV
)

from icecream import ic
from rt_cfg import RuntimeConfig


def T_from_tq(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Convert pose (translation, quaternion) into a 4x4 homogeneous matrix.
    """
    T = np.zeros(shape=(*t.shape[:-1], 4, 4),
                 dtype=t.dtype)
    tx.rotation.matrix.from_quaternion(q, out=T[..., :3, :3])
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def compute_transform(poses_file: Union[str, Iterable[str]],
                      default_T_ec: np.ndarray = np.eye(4)):
    base_from_ee = []
    robot_cam_from_tag = []
    table_cam_from_tag = []
    if not isinstance(poses_file, str):
        poses_files = poses_file
        for poses_file in poses_files:
            with open(poses_file, 'rb') as fp:
                pq_list = pickle.load(fp)
                print(len(pq_list))
                for data in pq_list:
                    (ee_pos, ee_quat,
                     cam1_pos, cam1_quat,
                     cam2_pos, cam2_quat,
                     *_) = data
                    # cam1_pos *= 0.155/0.1655
                    # cam2_pos *= 0.155/0.1655
                    base_from_ee.append(T_from_tq(ee_pos, ee_quat))
                    robot_cam_from_tag.append(T_from_tq(cam1_pos, cam1_quat))
                    table_cam_from_tag.append(T_from_tq(cam2_pos, cam2_quat))

    base_from_ee = np.stack(base_from_ee, axis=0)
    robot_cam_from_tag = np.stack(robot_cam_from_tag, axis=0)

    # For here, compute the average transform
    table_cam_from_tag = np.stack(table_cam_from_tag, axis=0)
    R = average_rotation_matrix(table_cam_from_tag[..., :3, :3])
    t = np.mean(table_cam_from_tag[..., :3, 3], axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    table_cam_from_tag = T

    if True:
        Ts_er = tx.invert(base_from_ee)
        Ts_co = robot_cam_from_tag

        # Initial guess for `T_ec`
        # set `fix_ec = True` if you're 100% sure about
        # the initial guess for `T_ec` and the dataset
        # doesn't have enough information to constrain
        # both T_ec/T_ro.
        T_ec_ = default_T_ec
        T_ec, T_ro = optimize_transforms(Ts_er,
                                         Ts_co,
                                         T_ec_,
                                         log_step=200,
                                         fix_ec=False)
        ee_from_robot_cam = T_ec
        base_from_tag = T_ro

        # == just for visualization ==
        if True:
            geoms = []
            cld0 = []
            cld1 = []
            for T_re, T_co in zip(base_from_ee, robot_cam_from_tag):
                axes_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1)
                axes_mesh.transform(T_re)
                geoms.append(axes_mesh)

                # base_from_tag'
                T = T_re @ T_ec @ T_co
                axes_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1)
                axes_mesh.transform(T)
                geoms.append(axes_mesh)

                t_re = T_re[:3, 3]
                t    = T[:3, 3]
                cld0.append(t_re)
                cld1.append(t)
            pcd0= o3d.geometry.PointCloud()
            pcd0.points=o3d.utility.Vector3dVector(np.asarray(cld0))
            
            pcd1= o3d.geometry.PointCloud()
            pcd1.points=o3d.utility.Vector3dVector(np.asarray(cld1))
            match = np.stack([np.arange(len(cld1)), np.arange(len(cld1))], axis=-1) # Nx2
            ls = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd0,pcd1,match)
            geoms.append(ls)
            

            


            o3d.visualization.draw(geoms)

    return {
        'base_from_ee': base_from_ee,
        'robot_cam_from_tag': robot_cam_from_tag,
        'table_cam_from_tag': table_cam_from_tag,
        'ee_from_robot_cam': ee_from_robot_cam,
        'base_from_tag': base_from_tag,
        'base_from_table_cam': (
            base_from_tag @ tx.invert(table_cam_from_tag)
        )
    }


def main():
    rt_cfg = RuntimeConfig()
    transforms = compute_transform([rt_cfg.calib_data_file],
                                   default_T_ec=rt_cfg.default_T_ec)

    ensure_directory(Path(rt_cfg.transforms_file).parent)
    with open(rt_cfg.transforms_file, 'wb') as fp:
        pickle.dump(transforms, fp)


if __name__ == '__main__':
    main()
