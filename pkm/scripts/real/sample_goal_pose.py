#!/usr/bin/env python3

import isaacgym

import json
import time
import copy
from dataclasses import dataclass
import numpy as np
import open3d as o3d

import torch as th
from yourdfpy import URDF
from cho_util.math import transform as tx
import trimesh
import glob
from pathlib import Path
import pickle
from scipy.spatial.transform import Rotation as R
from pkm.real.multi_perception_async import MultiPerception

from pkm.util.path import get_path
from pkm.data.transforms.col import DistanceToRobot, franka_link_transforms
from pkm.util.vis.win_o3d import AutoWindow
from icecream import ic
from typing import Dict, Tuple, Any, Optional, Iterable
from pkm.data.transforms.io_xfm import scene_to_mesh
from pkm.util.path import ensure_directory
from hydra_zen import (store, zen, hydrated_dataclass)
import subprocess

from rt_cfg import RuntimeConfig

# hardcoded base to origin transform
BASE_TO_ORIGIN_X = -0.5
BASE_TO_ORIGIN_Y = 0.
BASE_TO_ORIGIN_Z = 0.4
DEFAULT_KEY: str = 'core-can-9effd38015b7e5ecc34b900bb2492e-0.080'
DEFAULT_MASS: float = 0.2


def normalize_rotation(R):
    U, s, Vt = np.linalg.svd(R)
    return U @ Vt


def matrix_to_7d_pose(matrix):
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    rotation = R.from_matrix(rotation)
    quat = rotation.as_quat()

    pose = np.concatenate((translation, quat))

    return pose


def pose_to_matrix(pose):
    out = np.eye(4)
    out[:3, 3] = pose[:3]
    out[:3, :3] = tx.rotation.matrix.from_quaternion(pose[3:7])
    return out


def export_relative_pose(T_ref_goal, T_ref_initial):
    '''
    Print relative goal pose, euler angle and translation
    Given T_ref_goal and T_ref_initial
    '''
    rel_trans = T_ref_goal[:3, 3] - T_ref_initial[:3, 3]
    rel_rot = T_ref_goal[:3, :3] @ T_ref_initial[:3, :3].T
    rel_rot = normalize_rotation(rel_rot)
    print("Relative goal transformation: ", rel_trans)
    print("Relative goal rotation: ", rel_rot)

    # if dump:
    #     rt_cfg = RuntimeConfig()
    #     with open(rt_cfg.new_task_file, 'wb') as fp:
    #         task_cfg = {
    #             'translation': rel_trans,
    #             'rotation': rel_rot}
    #         pickle.dump(task_cfg, fp)
    return rel_trans, rel_rot


def dump_task_file(rel_trans, rel_rot):
    rt_cfg = RuntimeConfig()
    description = 'auto'#input('describe the task:')
    with open(rt_cfg.task_cfg_file, 'rb') as fp:
        task_cfg = pickle.load(fp)

    with open(rt_cfg.new_task_file, 'wb') as fp:
        task = {
            'translation': rel_trans,
            'rotation': rel_rot,
            'config': task_cfg,
            'config_file': rt_cfg.task_cfg_file,
            'description': description
        }
        pickle.dump(task, fp)


def base_to_origin(T):
    '''
        Transforms 4 by 4 pose in base frame into origin frame
    '''
    _T = copy.deepcopy(T)
    _T[:3, 3] += [BASE_TO_ORIGIN_X,
                  BASE_TO_ORIGIN_Y,
                  BASE_TO_ORIGIN_Z]
    return _T


def origin_to_base(T):
    '''
        Transforms 4 by 4 pose in base frame into origin frame
    '''
    _T = copy.deepcopy(T)
    _T[:3, 3] -= [BASE_TO_ORIGIN_X,
                  BASE_TO_ORIGIN_Y,
                  BASE_TO_ORIGIN_Z]
    return _T


def transform_pointcloud(initial_pcd=None, initial_pose=None,
                         init_pose=None,
                         goal_pose=None):
    '''
        GUI function that extracts task scenario
        args:
            initial_pcd: open3d geometry
            initial_pose
        return:
            rel_trans: relative transform, from initial to
                       user press Crtl^C
            rel_rot: relative rotation, from initial to
                       user press Crtl^C
            task_list: list of task, where the task is the
                       tuple of (init_pose, goal_pose,
                       rel_trans, rel_rot). Each task could be
                       added by using (SPACE, ENTER) pair

    '''

    path: str = '/tmp/docker/real-log/run-098/log/0.pkl'
    urdf_path = get_path('assets/franka_description/robots/franka_panda.urdf')
    cache = DistanceToRobot(urdf_path, device='cpu')
    mesh_list = cache._mesh_list
    mesh_list = [m.as_open3d for m in mesh_list]
    # Move robot mesh to home position
    home = np.array([-0.0050, -0.5170, 0.0065, -2.1196, 0.0004,
                     2.0273, 0.7912])
    q = home
    xfms = franka_link_transforms(th.as_tensor(q[:7]))
    ms = copy.deepcopy(mesh_list)
    [m.transform(x) for m, x in zip(ms, xfms)]
    robot_mesh = {k: v for k, v in zip(cache._chain, ms)}

    if initial_pcd is None:
        with open(path, 'rb') as fp:
            data = pickle.load(fp)

        data = {k: v[0] for (k, v) in data.items()}
        initial_pcd = o3d.geometry.PointCloud()
        x = data['partial_cloud']
        initial_pcd.points = o3d.utility.Vector3dVector(x)
        initial_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    if initial_pose is None:

        initial_pose = np.eye(4)
        initial_pose[:3, 3] = initial_pcd.get_center()

    task_list = []
    copied_pcd = None
    copied_pose_mesh = None
    copied_transform = None

    initial_pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2)
    initial_pose_mesh.transform(initial_pose)

    transformed_pose_mesh = copy.deepcopy(initial_pose_mesh)
    transormed_pcd = copy.deepcopy(initial_pcd)
    transform = initial_pose

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    table = trimesh.creation.box((0.4, 0.5, 0.4))
    table = table.apply_translation((0.5, 0.0, -0.2))
    table_ls = o3d.geometry.LineSet.create_from_triangle_mesh(
        table.as_open3d)
    table_ls.paint_uniform_color([0, 0, 0])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    geometries = [
        origin,
        initial_pcd,
        table_ls,
        initial_pose_mesh,
        transormed_pcd,
        transformed_pose_mesh] + ms
    for geometry in geometries:
        vis.add_geometry(geometry)

    def inc_x(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta = np.eye(4)
        delta[0, 3] = 0.005
        transform = delta @ transform
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def dec_x(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta = np.eye(4)
        delta[0, 3] = -0.005
        transform = delta @ transform
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def dec_y(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta = np.eye(4)
        delta[1, 3] = -0.005
        transform = delta @ transform
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def inc_y(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta = np.eye(4)
        delta[1, 3] = 0.005
        transform = delta @ transform
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def inc_z(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta = np.eye(4)
        delta[2, 3] = 0.005
        transform = delta @ transform
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def dec_z(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta = np.eye(4)
        delta[2, 3] = -0.005
        transform = delta @ transform
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def stamp(vis):
        nonlocal copied_pcd, copied_pose_mesh, copied_transform
        copied_pcd = copy.deepcopy(transormed_pcd)
        copied_pose_mesh = copy.deepcopy(transformed_pose_mesh)
        copied_pcd.paint_uniform_color([0, 1.0, 0])

        vis.add_geometry(copied_pcd)
        vis.add_geometry(copied_pose_mesh)

        # pose_list.append(copy.deepcopy(transform))
        # export_relative_pose(pose_list[-1],
        #                      pose_list[-2],
        #                      dump=False)
        copied_transform = copy.deepcopy(transform)
        ic(copied_transform)

    def remove_stamp(vis):
        nonlocal copied_transform
        vis.remove_geometry(copied_pcd)
        vis.remove_geometry(copied_pose_mesh)

        copied_transform = None

    def save_pair(vis):
        nonlocal copied_pcd, copied_pose_mesh, copied_transform, transormed_pcd, transformed_pose_mesh, transform

        # Assert initial pose exist
        assert (copied_transform is not None)

        transormed_pcd.paint_uniform_color([0, 0, 1.0])
        vis.update_geometry(transormed_pcd)
        vis.poll_events()
        time.sleep(1.0)

        # Remove geometry
        vis.remove_geometry(copied_pcd)
        vis.remove_geometry(copied_pose_mesh)
        vis.remove_geometry(transormed_pcd)
        vis.remove_geometry(transformed_pose_mesh)

        # Save (init_pose, goal_pose, rel_trans, rel_rot)
        rel_trans, rel_rot = export_relative_pose(transform, copied_transform)
        data = {
            'init_pose': matrix_to_7d_pose(base_to_origin(copied_transform)),
            'goal_pose': matrix_to_7d_pose(base_to_origin(transform)),
            'rel_trans': rel_trans,
            'rel_rot': rel_rot
        }
        ic(data)
        task_list.append(data)

        # Regenerate pcd
        transformed_pose_mesh = copy.deepcopy(initial_pose_mesh)
        transormed_pcd = copy.deepcopy(initial_pcd)
        transform = initial_pose
        vis.add_geometry(transformed_pose_mesh)
        vis.add_geometry(transormed_pcd)

    # Degrees

    def dec_roll(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('x', -5, degrees=True).as_matrix()

        # Transformation between global and child frame, child is
        # frame that is parallel wrt to global frame, translated only
        # to the centre of the object

        T_pc = np.eye(4)
        T_pc[:3, 3] = transform[:3, 3]
        delta = T_pc @ delta_child @ tx.invert(T_pc)

        transform = delta @ transform
        transform[:3, :3] = normalize_rotation(transform[:3, :3])
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def inc_roll(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('x', 5, degrees=True).as_matrix()

        T_pc = np.eye(4)
        T_pc[:3, 3] = transform[:3, 3]
        delta = T_pc @ delta_child @ tx.invert(T_pc)
        transform = delta @ transform
        transform[:3, :3] = normalize_rotation(transform[:3, :3])
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def dec_pitch(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('y', -5, degrees=True).as_matrix()

        T_pc = np.eye(4)
        T_pc[:3, 3] = transform[:3, 3]

        delta = T_pc @ delta_child @ tx.invert(T_pc)
        transform = delta @ transform
        transform[:3, :3] = normalize_rotation(transform[:3, :3])
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def inc_pitch(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('y', 5, degrees=True).as_matrix()

        T_pc = np.eye(4)
        T_pc[:3, 3] = transform[:3, 3]

        delta = T_pc @ delta_child @ tx.invert(T_pc)
        transform = delta @ transform
        transform[:3, :3] = normalize_rotation(transform[:3, :3])
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def dec_yaw(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('z', -5, degrees=True).as_matrix()

        T_pc = np.eye(4)
        T_pc[:3, 3] = transform[:3, 3]

        delta = T_pc @ delta_child @ tx.invert(T_pc)
        transform = delta @ transform
        transform[:3, :3] = normalize_rotation(transform[:3, :3])
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    def inc_yaw(vis):
        nonlocal transormed_pcd, transformed_pose_mesh, transform
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('z', 5, degrees=True).as_matrix()

        T_pc = np.eye(4)
        T_pc[:3, 3] = transform[:3, 3]

        delta = T_pc @ delta_child @ tx.invert(T_pc)
        transform = delta @ transform
        transform[:3, :3] = normalize_rotation(transform[:3, :3])
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)
        vis.update_geometry(transormed_pcd)
        vis.update_geometry(transformed_pose_mesh)
        # export_relative_pose(transform, initial_pose)

    vis.register_key_callback(68, inc_x)  # Deq
    vis.register_key_callback(65, dec_x)  # A
    vis.register_key_callback(87, inc_y)  # W
    vis.register_key_callback(83, dec_y)  # S
    vis.register_key_callback(90, inc_z)  # Z
    vis.register_key_callback(88, dec_z)  # X
    vis.register_key_callback(262, inc_roll)  # RIGHT
    vis.register_key_callback(263, dec_roll)  # LEFT
    vis.register_key_callback(265, inc_pitch)  # UP
    vis.register_key_callback(264, dec_pitch)  # DOWN
    vis.register_key_callback(69, inc_yaw)  # E
    vis.register_key_callback(81, dec_yaw)  # Q
    vis.register_key_callback(32, stamp)  # SPACE
    vis.register_key_callback(259, remove_stamp)  # BACKSPACE
    vis.register_key_callback(257, save_pair)  # ENTER

    if init_pose is not None:
        transform[...] = origin_to_base(pose_to_matrix(init_pose))
        delta = transform  # @ tx.invert(initial_pose)
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)

        stamp(vis)
        old_xfm = transform.copy()
        transform[...] = origin_to_base(pose_to_matrix(goal_pose))
        delta = transform @ tx.invert(old_xfm)
        transormed_pcd.transform(delta)
        transformed_pose_mesh.transform(delta)

    try:
        vis.run()
    finally:
        rel_trans, rel_rot = export_relative_pose(transform, initial_pose)
        final_pose = transform @ initial_pose
        ic(initial_pose)
        ic(final_pose)
        return rel_trans, rel_rot, task_list


def sample_goal_pose(**kwds):
    rt_cfg = RuntimeConfig()
    kwds.setdefault('debug', True)
    kwds.setdefault('ip', 'kim-MS-7C82')

    # TODO: deprecate task_cfg_file
    with open(rt_cfg.task_cfg_file, 'rb') as fp:
        tcfg = pickle.load(fp)
        drag = False
        for t, v in tcfg['task']:
            if 'drag' in t:
                drag = True
    mode = 'thin' if drag else 'normal'
    
    perception_cfg = MultiPerception.Config(**kwds,
                                            tracker_type='multi-april',
                                            # object='blue-holder',
                                            object='black-cup',
                                            mode=mode,
                                            skip_april=False
                                            )

    perception = MultiPerception(perception_cfg,
                                 rt_cfg.cam_ids,
                                 rt_cfg.extrinsics,
                                 rt_cfg=rt_cfg
                                 )
    if perception.cfg.ip is not None:
        q_home = np.asarray(
            [-0.0122, -0.1095, 0.0562, -2.5737, -0.0196, 2.4479, 0.0])
        perception.robot.move_to_joint_positions(q_home)
    else:
        perception.update_joint_states(
            np.asarray(
                [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0],
                dtype=np.float32))
    out = None
    while out is None:
        out = perception.get_observations()
        time.sleep(0.001)
    pcd_colored_tensor, pose = out

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        pcd_colored_tensor[..., :3].detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(
        pcd_colored_tensor[..., 3:].detach().cpu().numpy())
    rel_trans, rel_rot, _ = transform_pointcloud(pcd, pose)
    dump_task_file(rel_trans, rel_rot)
    subprocess.run(['pkill', '-f', 'from multiprocessing'])
    subprocess.run(['pkill', '-f', 'python3 sample_goal'])


@store(name="sample_from_mesh")
def sample_from_mesh(
        obj_key: str = F'{DEFAULT_KEY}', scale: float = 1.0,
        export_path: str = '/home/user/Documents/genom-task/{key}.json',
        abs_scale: Optional[float] = None, load_task: bool = False):
    '''
        Load object mesh, generates tasks list and export
    '''
    # sem-Rock-f7b2c1368b2c6f9a8b43e675a8d9c006-0.080
    obj_path: str = '/input/DGN/meta-v8/urdf/{obj_key}.urdf'.format(
        obj_key=obj_key)

    if load_task:
        with open(F'/home/user/Documents/genom-task/{obj_key}.json', 'rb') as fp:
            tasks = json.load(fp)
            tasks = tasks[obj_key]
        scale = tasks['scales'][0]

    # load object mesh
    urdf = URDF.load(obj_path)
    obj_mesh = scene_to_mesh(urdf.scene).as_open3d
    obj_mesh.paint_uniform_color([1, 0.0, 0])

    if abs_scale is not None:
        print('scale will be ignored by abs_scale')
        points = np.asarray(obj_mesh.vertices)
        diameter = 2.0 * np.linalg.norm(points, axis=0).max()
        scale = abs_scale / diameter

    obj_mesh.scale(scale, center=obj_mesh.get_center())

    init_pose = None
    goal_pose = None
    if load_task:
        init_pose = (tasks['pose']['value'][0]['init_pose'])
        goal_pose = (tasks['pose']['value'][0]['goal_pose'])

    _, _, task_list = transform_pointcloud(obj_mesh,
                                           init_pose=init_pose,
                                           goal_pose=goal_pose)

    ic(task_list)

    # Save object key, scale, and task list
    ensure_directory(Path(export_path).parent)
    key = Path(obj_path).stem

    print([{k: v.tolist() for (k, v) in t.items()} for t in task_list])
    tasks = {
        key: {
            'urdf': str(obj_path),
            'pose': {
                'mode': 'sampled',
                'path': None,
                'value': [
                    {k: v.tolist() for (k, v) in t.items()}
                    for t in task_list]
            },
            'mass': DEFAULT_MASS,
            'scales': [scale]
        }
    }
    if not load_task:
        json.dump(tasks, fp)

    subprocess.run(['pkill', '-f', 'from multiprocessing'])
    subprocess.run(['pkill', '-f', 'python3 sample_goal'])

if __name__ == '__main__':
    sample_goal_pose()
    # store.add_to_hydra_store()
    # zen(sample_from_mesh).hydra_main(config_name='sample_from_mesh',
    #                       version_base='1.1',
    #                       config_path=None)
