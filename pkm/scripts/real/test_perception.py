#!/usr/bin/env python3

import isaacgym

import time
import sys
import copy
import pickle

import numpy as np
import open3d as o3d
import torch as th
from cho_util.math import transform as tx
from pkm.real.multi_perception_async import MultiPerception
from typing import Dict, Tuple, Any, Optional, Iterable
from rt_cfg import RuntimeConfig
from pkm.util.vis.win_o3d import (
    AutoWindow
)
from tqdm.auto import tqdm
from pkm.data.transforms.aff import get_gripper_mesh
from pkm.util.path import get_path
from pkm.util.torch_util import dcn
from hydra_zen import (store, zen)


@store(name="test_perception")
def test_perception(
        object: str,
        mode: str = 'normal',
        skip_april: bool = True,
        use_kf: bool = True,
        ip: Optional[str] = 'kim-MS-7C82',
        export_dir: Optional[str] = None,
        debug: bool = True,
):
    rt_cfg = RuntimeConfig()
    set_dummy_joints = (ip is None)
    perception_cfg = MultiPerception.Config(fps=60,
                                            img_width=640,
                                            img_height=480,
                                            tracker_type='multi-april',
                                            object=object,
                                            use_kf=use_kf,
                                            mode=mode,
                                            skip_april=skip_april
                                            )
    perception_cfg.segmenter.table_color_file = rt_cfg.table_color_file
    perception = MultiPerception(perception_cfg,
                                 device_ids=rt_cfg.cam_ids,
                                 extrinsics=rt_cfg.extrinsics,
                                 rt_cfg=rt_cfg
                                 )

    win = AutoWindow()
    vis = win.vis

    urdf_path = get_path('assets/franka_description/robots/franka_panda.urdf')
    hand_mesh_default = get_gripper_mesh(cat=True,
                                         frame='panda_tool',
                                         urdf_path=urdf_path).as_open3d
    init_pcd=None
    for index in tqdm(range(1666)):
        if set_dummy_joints:
            perception.update_joint_states(
                np.asarray(
                    [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0],
                    dtype=np.float32))

        out = None
        while out is None:
            out = perception.get_observations()
            # time.sleep(0.001)
        pcd_colored, pose = out

        if export_dir is not None:
            with open(F'{export_dir}/{index:04d}.pkl', 'wb') as fp:
                pickle.dump({
                    'pcd': pcd_colored,
                    'pose': pose
                }, fp)

        if False:
            if init_pcd is None:
                #init_pose_inv = np.linalg.inv(obj_pose)
                #init_pcd = pcd0.copy(
                #) @ init_pose_inv[:3, :3].T + init_pose_inv[:3, 3]
                init_pcd = load_full_pcd('puppy',
                                            pose,
                                            pcd_colored)
                
        if True:
            if init_pcd is None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    pcd_colored[..., :3].cpu().numpy())

                if pcd_colored.shape[-1] >= 6:
                    pcd.colors =  o3d.utility.Vector3dVector(pcd_colored[..., 3:].cpu().numpy())
            else:
                # pcd = o3d.t.geometry.PointCloud()
                pcd = o3d.geometry.PointCloud()

                pose = th.as_tensor(pose, dtype=th.float32,
                                    device='cuda:0')
                print(init_pcd)
                print(pose)
                # pcd.point.positions = (
                #     th2o3d(init_pcd @ pose[:3,:3].T + pose[:3,3])
                # )
                pcd.points = o3d.utility.Vector3dVector(
                    dcn(init_pcd @ pose[:3,:3].T + pose[:3,3])
                )


            if pcd_colored.shape[-1] >= 6:
                pcd.colors = o3d.utility.Vector3dVector(
                    pcd_colored[..., 3:].cpu().numpy())

            vis.add_geometry('cloud', pcd, color=(1, 1, 1, 1))

        if False:
            ee_pos, ee_ori = perception.robot.get_ee_pose()
            ee_pos = ee_pos.detach().cpu().numpy()
            ee_ori = ee_ori.detach().cpu().numpy()
            franka_offset = (-0.5, 0.0, 0.4)

            hand_state_7 = np.concatenate([
                ee_pos,
                # + franka_offset,
                ee_ori
            ], axis=-1)
            h = hand_state_7
            h_T = np.eye(4)
            h_T[:3, :3] = tx.rotation.matrix.from_quaternion(h[3:7])
            h_T[:3, 3] = h[:3]

            hand_mesh = copy.deepcopy(hand_mesh_default)
            hand_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(
                hand_mesh)

            hand_mesh.transform(h_T)
            hand_mesh.paint_uniform_color([1, 1, 0])
            vis.add_geometry('hand', hand_mesh, color=(1, 1, 1, 1))

        if pose is not None:
            pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.2)
            pose_mesh.transform(pose)

            vis.add_geometry('pose', pose_mesh, color=(1, 1, 1, 1))

            # win.wait()
            win.tick()
            # time.sleep(0.1)

            # o3d.visualization.draw_geometries([pcd])
        # time.sleep(100000)
        print(pose)


if __name__ == '__main__':
    store.add_to_hydra_store()
    zen(test_perception).hydra_main(config_name='test_perception',
                                    version_base='1.1',
                                    config_path=None)
