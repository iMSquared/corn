#!/usr/bin/env python3

import pickle
import trimesh
import numpy as np
import torch as th
from yourdfpy import URDF
from cho_util.math import transform as tx
from pathlib import Path

from pkm.data.transforms.aff import get_gripper_mesh

from icecream import ic

def print_radii():
    for frame in ['panda_tool', 'panda_hand']:
        m = get_gripper_mesh(cat=True, frame=frame)
        radius = np.linalg.norm(m.vertices, axis=-1).max()
        ic(frame, radius)

def print_stats():
    from tqdm.auto import tqdm
    from pkm.models.cloud.point_mae import get_group_module_v2
    group = get_group_module_v2('fps', 32)

    pos_count = 0
    all_count = 0
    files = list(Path('/tmp/col-10').glob('*.pkl'))
    try:
        with th.no_grad():
            for f in tqdm(files):
                with open(f, 'rb') as fp:
                    d = pickle.load(fp)
                # d['contact_flag']

                object_cloud = d['object_cloud']
                contact_flag = d['contact_flag']
                object_cloud = th.as_tensor(object_cloud,
                                            dtype=th.float32)
                contact_flag = th.as_tensor(contact_flag,
                                            dtype=th.float32)
                # Determine patch indices
                aux = {}
                _ = group(object_cloud, aux=aux)
                index = aux['patch_index']
                patch_label = contact_flag[index.reshape(-1)
                                           ].reshape(index.shape).any(dim=-1)
                pos_count += patch_label.sum()
                all_count += patch_label.shape[-1]
    finally:
        ic(pos_count / all_count, pos_count, all_count)


def main():
    for f in Path('/tmp/col-12-2048/').glob('*.pkl'):
        with open(f, 'rb') as fp:
            d = pickle.load(fp)
        object_pose = d['object_pose']
        hand_pose = d['hand_pose']
        contact_flag = d['contact_flag']
        object_cloud = d['object_cloud']

        pcd = trimesh.PointCloud(object_cloud)
        pcd.visual.vertex_colors = np.where(
            contact_flag[..., None], (255, 0, 0), (0, 0, 255))

        # urdf_path = '../../../src/pkm/data/assets/franka_description/robots/franka_panda.urdf'
        # urdf = URDF.load(urdf_path,
        #                  build_collision_scene_graph=True,
        #                  load_collision_meshes=True,
        #                  force_collision_mesh=False)
        # hand_mesh = urdf.collision_scene.subscene('panda_hand').dump(
        #     concatenate=True)
        hand_mesh = get_gripper_mesh(cat=True, frame='panda_tool')
        hand_xfm = np.eye(4)
        hand_xfm[: 3, : 3] = tx.rotation.matrix.from_quaternion(
            hand_pose[..., 3: 7])
        hand_xfm[:3, 3] = hand_pose[..., 0:3]
        hand_mesh.apply_transform(hand_xfm)

        table = trimesh.creation.box((0.4, 0.5, 0.4))
        table = table.apply_translation((0.0, 0.0, +0.2))
        trimesh.Scene([pcd, hand_mesh, table]).show()


if __name__ == '__main__':
    # print_radii()
    main()
