import numpy as np
import open3d as o3d
import subprocess
from rt_cfg import RuntimeConfig
from scipy.spatial.transform import Rotation as R
from sample_goal_pose import normalize_rotation
from pkm.real.multi_perception_async import MultiPerception
from cho_util.math import transform as tx
import copy
import sys
import argparse
import torch as th

import tkinter as tk
from pathlib import Path
import glob
import time
import trimesh
import pickle

from pose_gui_util import KeyCallbackViewer, get_pcd_and_coordinate, get_pose_and_coordinate
# from sample_pcds import sample_pcd
from pkm.util.torch_util import dcn

def sample_yaw():
    h =-np.pi/2 +np.pi *np.random.rand(1)
    z = np.sin(h)
    w = np.cos(h)
    return np.stack([0 * z, 0 * z, z, w], axis=-1)

def goal_sample(mode,
                obj_name,
                rt_cfg,
                use_random_goal,
                use_random_init):
    np.random.seed(int(time.time()))
    canonical_file = rt_cfg.root+f'/object/{obj_name}/canonical.pkl'
    if not Path(canonical_file).is_file():
        raise FileNotFoundError("There is no canonical pcd file!")
    with open(canonical_file, 'rb') as fp:
        sampled_canonical_data= pickle.load(fp)
    sampled_canonical_pcd = dcn(sampled_canonical_data['pcd'])
    candidates = glob.glob(rt_cfg.root+f'/object/{obj_name}/*.pkl')
    candidates.remove(canonical_file)
    center = sampled_canonical_pcd[..., :3].mean(axis=-2)
    print(center)
    distances = np.linalg.norm(sampled_canonical_pcd[..., :3]
                            - center, axis=-1)
    radius = np.percentile(distances, 90)
    print(radius)

    pcds = []
    poses = []
    for candidate in candidates:
        with open(candidate, 'rb') as fp:
            data = pickle.load(fp)
            pcds.append(data['pcd'].to(th.float))
            poses.append(data['pose'].astype(np.float32))

    while True:
        index = np.random.choice(range(len(pcds)))
        selected = pcds[index].numpy()
        selected_pose = poses[index]

        if use_random_goal:
            
            rand_yaw = sample_yaw()
            center = selected[..., :3].mean(axis = -2)
            print(selected.shape, rand_yaw.shape, center.shape)
            selected[..., :3] = tx.rotation.quaternion.rotate(rand_yaw, 
                                        selected[..., :3]-center[None]) + center[None]
            selected_pose[:3, :3] = (tx.rotation.quaternion.to_matrix(rand_yaw) @
                                    selected_pose[:3, :3])
            x = 0.5 + np.random.uniform(
                max(-0.2 + radius, -0.2 + 0.05),
                min(+0.2 - radius, +0.2 - 0.1)
            )
            y = np.random.uniform(-0.25+radius, 0.25-radius)
            # print(x,y)
            # y = (-0.2 + radius +
            #      (0.3 - radius) *np.random.rand(1))
            diff = np.stack([x,y], axis=0)- center[..., :2]
            selected[..., 0] += diff[0]
            selected[..., 1] += diff[1]
            selected_pose[0, 3] = x
            selected_pose[1, 3] = y
                
            
        # win = AutoWindow()
        # vis = win.vis\
        canonical_pcd = o3d.geometry.PointCloud()
        canonical_pcd.points = o3d.utility.Vector3dVector(
            sampled_canonical_pcd[..., :3])
        canonical_pcd.colors = o3d.utility.Vector3dVector(
                    sampled_canonical_pcd[..., 3:])

        canonical_pose = np.eye(4)
        canonical_pose[:3, 3] = np.mean(sampled_canonical_pcd[..., :3], axis=0)
        canonical_pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2)
        canonical_pose_mesh.transform(canonical_pose)

        goal_pcd, goal_pose_mesh = get_pcd_and_coordinate(selected, selected_pose)

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

        table = trimesh.creation.box((0.4, 0.5, 0.4))
        table = table.apply_translation((0.5, 0.0, -0.2))
        table_ls = o3d.geometry.LineSet.create_from_triangle_mesh(
            table.as_open3d)
        table_ls.paint_uniform_color([0, 0, 0])
        geometries =[origin,
                    table_ls,
                    canonical_pcd,
                    canonical_pose_mesh,
                    goal_pose_mesh,
                    goal_pcd,
                    ]
        o3d.visualization.draw(geometries)
        text = input("type n if goal is NOT okay: ")
        if text != 'n':
            break
    
    if use_random_init:
        index = np.random.choice(range(len(pcds)))
        init = pcds[index]
        rand_yaw = sample_yaw()
        init[..., :3] = tx.rotation.quaternion.rotate(rand_yaw, 
                                    init[..., :3])
        center = init[..., :2].mean(axis = -2)
        x = 0.5 + np.random.uniform(-0.16, +0.16)
        y = np.random.uniform(-0.2, +0.2)
        diff = np.concatenate([x,y], axis=0)- center[..., :2]
        init[..., 0] += diff[0]
        init[..., 1] += diff[1]
        (init_from_base, 
         init_pcd, init_pose_mesh) = get_pose_and_coordinate(
                                    canonical=sampled_canonical_pcd,
                                    target=init,
                                    mode=mode)
        o3d.visualization.draw(geometries+[init_pcd, init_pose_mesh])

    else:
        ip = 'kim-MS-7C82'
        # ip = None
        perception_cfg = MultiPerception.Config(
                                            fps=60,
                                            img_width= 640,
                                            img_height = 480,
                                            tracker_type = 'multi-april',
                                            # object='blue-holder',
                                            object=obj_name,
                                            use_kf=False,

                                            # mode='thin',
                                            # mode='normal',
                                            mode=mode,
                                            skip_april=True,
                                            ip= ip,
                                            debug=True
                                            )
        perception_cfg.segmenter.table_color_file=rt_cfg.table_color_file
        perception = MultiPerception(perception_cfg,
                                    device_ids=rt_cfg.cam_ids,
                                    extrinsics=rt_cfg.extrinsics,
                                    rt_cfg=rt_cfg
                                    )
        viewer = KeyCallbackViewer(
            mode=mode
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
        perception.get_observations()
        out = None
        while out is None:
            out = perception.get_observations()
            time.sleep(0.001)
        pcd_colored_tensor, pose = out
        init = pcd_colored_tensor.cpu().numpy()
        (init_from_base, 
        init_pcd, init_pose_mesh) = get_pose_and_coordinate(
                                canonical=dcn(sampled_canonical_pcd),
                                target=init,
                                mode=mode)
        # o3d.visualization.draw(geometries+[init_pcd, init_pose_mesh])
        
        viewer.cur_cloud = init_pcd
        viewer.pose_mesh = init_pose_mesh
        viewer.transform = init_from_base
        viewer.canonical_pose = canonical_pose
        viewer.canonical = sampled_canonical_pcd
        overlay_canonical_pcd = copy.deepcopy(canonical_pcd)
        T_v = np.eye(4)
        T_v[2, 3] = 0.2
        overlay_canonical_pcd.transform(T_v)
        for g in [overlay_canonical_pcd, 
                    goal_pose_mesh, goal_pcd]:
            viewer.vis.add_geometry(g)
        viewer.draw()

            
    rel_trans = selected_pose[:3, 3] - viewer.transform[:3, 3]
    # rel_rot =  ((selected_pose 
                    # @ tx.invert(init_from_base))[:3, :3])
    rel_rot = selected_pose[:3,:3] @ viewer.transform[:3,:3].T
    rel_rot = normalize_rotation(rel_rot)

    
    description = 'auto'#
    # input('describe the task:')
    with open(rt_cfg.task_cfg_file, 'rb') as fp:
        task_cfg = pickle.load(fp)

    with open(rt_cfg.new_task_file, 'wb') as fp:
        task = {
            'goal_pose' : (selected_pose),
            'init_pose' : viewer.transform,

            'translation': rel_trans,
            'rotation': rel_rot,

            'config': task_cfg,
            'config_file': rt_cfg.task_cfg_file,
            'description': description,
            'obj_name': obj_name
        }
        pickle.dump(task, fp)


    subprocess.run(['pkill', '-f', 'from multiprocessing'])
    subprocess.run(['pkill', '-f', 'python3 sample_goal'])
    print('exit')
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Sample goal from pre_sampled data')
    parser.add_argument('-thin', 
                        action='store_true',
                        dest='thin_mode',
                    help='Turn on thin mode perception mode')
    parser.add_argument('-obj',
                        type=str, required=True,
                        dest='obj_name',
                        help='Name of the object to select')
    parser.add_argument('-rand_goal', 
                        action='store_true',
                        dest='rand_goal',
                    help='Turn on thin mode perception mode')
    parser.add_argument('-rand_init', 
                        action='store_true',
                        dest='rand_init',
                    help='Turn on thin mode perception mode')
    rt_cfg = RuntimeConfig()
    args = parser.parse_args()
    mode = 'thin' if args.thin_mode else 'normal'
    print('mode', mode)
    goal_sample(
        mode=mode,
        obj_name=args.obj_name,
        rt_cfg=rt_cfg,
        use_random_goal=args.rand_goal,
        use_random_init=args.rand_init
    )
    # subprocess.run(['pkill', '-f', 'from multiprocessing'])
    # subprocess.run(['pkill', '-f', 'python3 sample_goal'])
    print('exit')
    sys.exit(0)

if __name__ =="__main__":
    main()
