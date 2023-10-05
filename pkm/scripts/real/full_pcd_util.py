#!/usr/bin/env python3

from pkm.real.util import (
    o3d2th,
    th2o3d
)
import open3d as o3d
import torch as th
import numpy as np

from cho_util.math import transform as tx

from pkm.util.torch_util import dcn
import copy
from pose_gui_util import get_pose_and_coordinate, KeyCallbackViewer, get_pcd_and_coordinate

def _load_saved_pcd(object_name:str):
    if object_name == 'puppy':
        # pcd = o3d.t.io.read_point_cloud('/input/DGN/coacd/mujoco-Dog.obj')
        pcd = o3d.t.io.read_triangle_mesh('/input/DGN/coacd/mujoco-Dog.obj')
        pcd = pcd.to_legacy().sample_points_poisson_disk(512)
        pcd.scale(0.09, center=pcd.get_center())
    if object_name == 'lipton':
        # pcd = o3d.t.io.read_point_cloud('/input/DGN/coacd/mujoco-Dog.obj')
        pcd = o3d.t.io.read_triangle_mesh('/home/user/Downloads/lipton-aligned.obj')
        pcd = pcd.to_legacy().sample_points_poisson_disk(512)
    elif object_name == 'red_dripper':
        pcd = o3d.io.read_point_cloud('/tmp/dripper.ply')
        pcd = pcd.farthest_point_down_sample(512)
        # pcd = pcd.to_legacy().sample_points_poisson_disk(512)
    else:
        raise ValueError(F'unknown object {object_name}')
    pcd= th.as_tensor(pcd.points,
                 dtype=th.float32,
                 device='cuda:0')
    return pcd#.detach().cpu().numpy()

def load_full_pcd(object_name: str,
                  init_pose: th.Tensor,
                  init_pcd: th.Tensor):
    full_pcd = _load_saved_pcd(object_name)
    # T @ full_pcd = init_pcd
    viewer = KeyCallbackViewer()

    T = get_pose_and_coordinate(dcn(full_pcd),
                                dcn(init_pcd),
                                postprocess = False,
                                mode='normal')
    init_pcd_t, init_pose_mesh = get_pcd_and_coordinate(init_pcd, T.cpu().numpy())
    viewer.cur_cloud = init_pcd_t
    viewer.pose_mesh = init_pose_mesh
    canonical_pose = np.eye(4)
    canonical_pose[:3, 3] = dcn(full_pcd)[..., :3].mean(axis = -2)
    viewer.transform = T.cpu().numpy()
    viewer.canonical_pose = canonical_pose
    canonical_pcd, _ = get_pcd_and_coordinate(dcn(full_pcd),
                                              canonical_pose)
    viewer.canonical = canonical_pcd
    overlay_canonical_pcd = copy.deepcopy(canonical_pcd)
    overlay_canonical_pcd.paint_uniform_color([0.1, 0.1, 0.7])
    overlay = copy.deepcopy(init_pcd_t)
    T_v = np.eye(4)
    T_v[2, 3] = 0.2
    overlay.transform(T_v @ tx.invert(T.cpu().numpy()))
    viewer.overlay = overlay
    T_v[2, 3] = 0.21
    overlay_canonical_pcd.transform(T_v)
    for g in [overlay_canonical_pcd]:
        viewer.vis.add_geometry(g)
    viewer.draw()
    # print(type(init_pose), type(T))
    # dT = init_pose^{-1} @ T
    # # in other words, dT @ full_pcd = init_pose^{-1} init_pcd
    # = canonical_pcd
    # T = T @ tx.invert(canonical_pose)
    # T = th.as_tensor(T, dtype=th.float, device=full_pcd.device)
    T = th.as_tensor(viewer.transform, dtype=th.float, device=full_pcd.device)
    dT = th.as_tensor(tx.invert(init_pose), dtype=full_pcd.dtype,
                      device=full_pcd.device) @ T
    #@ o3d2th(T)
    return full_pcd @ dT[:3,:3].T + dT[:3,3]
