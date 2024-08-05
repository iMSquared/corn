#!/usr/bin/env python3

from typing import Optional, Tuple
import pickle
from pathlib import Path
from yourdfpy import URDF
from functools import partial
from tempfile import TemporaryDirectory
from tqdm.auto import tqdm
import torch as th
import copy
import trimesh
from icecream import ic
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import einops

from pkm.env.scene.dgn_object_set import DGNObjectSet
from pkm.env.scene.acronym_object_set import AcronymObjectSet
from pkm.env.scene.combine_object_set import CombinedObjectSet
from pkm.env.scene.filter_object_set import FilteredObjectSet, dataset_hasattr
from pkm.models.common import merge_shapes
from pkm.util.path import ensure_directory
from pkm.util.math_util import (
    apply_pose_tq,
    invert_pose_tq,
    compose_pose_tq,
    quat_rotate,
    matrix_from_pose
)
from pkm.util.torch_util import dcn
from pkm.data.transforms.sdf.sdf_cvx_set_th3 import SignedDistanceTransform
from pkm.data.transforms.sdf.sdf_cvx_set_th3 import IsInHulls
from pkm.data.transforms.df_th3 import DistanceTransform
from pkm.data.transforms.aff import CheckGripperCollisionV2


def sample_pose(size: Tuple[int, ...], bound: th.Tensor):
    pose = th.empty(merge_shapes(size, 7),
                    dtype=bound.dtype,
                    device=bound.device)
    scale = (bound[..., 1, :] - bound[..., 0, :])
    offset = bound[..., 0, :]
    pose[..., :3].uniform_().mul_(scale).add_(offset)
    pose[..., 3:7].normal_()
    pose[..., 3:7] /= th.linalg.norm(pose[..., 3:7],
                                     dim=-1, keepdim=True)
    return pose


class ObjectSetDataset(th.utils.data.Dataset):
    def __init__(self, meta,
                 min_radius: float = 0.04,
                 max_radius: float = 0.12):
        self.meta = meta
        self.keys = list(self.meta.keys())
        self.r_min = min_radius
        self.r_max = max_radius

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        cloud = self.meta.cloud(self.keys[i])
        radius = np.linalg.norm(cloud, axis=-1).max()
        # radius2 = np.linalg.norm(cloud - cloud.mean(axis=-1, keepdims=True), axis=-1).max()
        # print(radius, radius2)
        scale = np.random.uniform(self.r_min,
                                  self.r_max) / radius
        return {
            'cloud': cloud * scale,
            'key': self.keys[i],
            'scale': scale
        }


def main():
    # Configure generation.
    device: str = 'cuda:0'
    batch_size: int = 32
    repeat: int = 40
    sigma: float = 0.05
    delta: float = 0.03
    min_radius: float = 0.04
    max_radius: float = 0.12
    out_dir: str = ensure_directory('/tmp/col-12-2048')

    table_dim: Tuple[float, float, float] = (0.4, 0.5, 0.4)
    initial_count: int = 0
    export_data: bool = True
    show_result: bool = False

    # Allow restarting from where the process crashed last time.
    if Path(out_dir).is_dir():
        initial_count = len(list(Path(out_dir).glob('*.pkl')))

    # Prepare dataset.
    if True:
        meta = DGNObjectSet(DGNObjectSet.Config(
            data_path='/input/DGN/meta-v8/',
            cloud_path='/input/DGN/meta-v8/cloud-2048/'
        ))
        dataset = ObjectSetDataset(meta,
                                min_radius,
                                max_radius)
    else:
        dgn_object_set = DGNObjectSet(DGNObjectSet.Config(
            data_path='/input/DGN/meta-v8/',
            cloud_path='/input/DGN/meta-v8/cloud-2048/'
            ))
        acr_object_set = AcronymObjectSet(AcronymObjectSet.Config(
            data_path='/input/ACRONYM/meta-v1/',
            cloud_path='/input/ACRONYM/meta-v1/cloud-2048',
            ))
        all_object_set = CombinedObjectSet([dgn_object_set, acr_object_set])
        all_object_set = FilteredObjectSet(
            all_object_set, filter_fn=partial(
                dataset_hasattr, all_object_set, 'cloud'))

        dataset = ObjectSetDataset(all_object_set)
    loader = th.utils.data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      drop_last=False)

    # Prepare collision checker.
    collision_fn = CheckGripperCollisionV2(
        device=device)

    # Configure pose sampler.
    tx, ty, tz = table_dim
    workspace = th.as_tensor([
        [-0.6 * tx, -0.6 * ty, tz - 0.05],
        [+0.6 * tx, +0.6 * ty, tz + 0.25]
    ], dtype=th.float,
        device=device)

    # Repeat and generate !
    count: int = initial_count
    for repeat in tqdm(range(repeat)):
        for data in tqdm(loader):
            obj_cloud = data['cloud'].to(
                dtype=th.float,
                device=device)
            # obj_cloud = workspace[0] + th.rand((2048, 3),
            #                                    device=device) * (workspace[1] - workspace[0])
            # print(obj_cloud.min(dim=0))
            # print(obj_cloud.max(dim=0))

            # Sample scene configuration.
            hand_pose = sample_pose(
                obj_cloud.shape[0],
                workspace)  # world_from_hand
            objt_pose = sample_pose(
                obj_cloud.shape[0],
                workspace)  # world_from_obj

            # objt_pose = hand_pose
            hand_from_obj = compose_pose_tq(
                invert_pose_tq(hand_pose),
                objt_pose)

            obj_cloud_wrt_hand = apply_pose_tq(
                hand_from_obj[:, None, :],  # N,1,7
                obj_cloud)
            d, g = collision_fn(obj_cloud_wrt_hand)
            i = th.argmin(d, dim=-1, keepdim=True)
            d = th.take_along_dim(d, i, dim=-1).squeeze(dim=-1)
            g = th.take_along_dim(g, i[..., None], dim=-2).squeeze(dim=-2)

            # Gradient direction, in world frame
            u = quat_rotate(hand_pose[..., 3:7], g)

            # Move gripper near the surface of the object.
            # We additionally apply a bit of offset to
            # increase the proportion of patches' intersections.
            # d_ = d
            d_ = d + sigma * th.randn_like(d) + delta
            # d_ = d + th.rand_like(d)
            txn = -d_[..., None] * u
            objt_pose[..., :3].add_(txn)

            # Apply new transform and then recompute SDF values.
            hand_from_obj = compose_pose_tq(
                invert_pose_tq(hand_pose),
                objt_pose)
            obj_cloud_wrt_hand = apply_pose_tq(
                hand_from_obj[:, None, :],  # N,1,7
                obj_cloud)
            d, g = collision_fn(obj_cloud_wrt_hand)

            # Determine contact points.
            contact_flag = (d <= 0)
            # contact_flag = m
            # print('contact_flag', contact_flag.shape)
            # ic(th.mean(contact_flag.any(dim=-1).float()))

            if True:
                tmp = apply_pose_tq(
                    objt_pose[:, None, :],  # N,1,7
                    obj_cloud)
                zmin = tmp[..., 2].min(dim=-1).values
                target = table_dim[2] + 0.1 * th.rand(
                    size=(obj_cloud.shape[0],),
                    device=obj_cloud.device)
                dz = target - zmin
                objt_pose[..., 2] += dz
                hand_pose[..., 2] += dz

            # EXPORT DATA.
            if export_data:
                def __export(count):
                    objt_cloud = apply_pose_tq(objt_pose[..., None, :],
                                               obj_cloud)
                    p = dcn(hand_pose)
                    o = dcn(objt_pose)
                    c = dcn(objt_cloud)
                    f = dcn(contact_flag)
                    k = dcn(data['key'])
                    s = dcn(data['scale'])

                    for i in range(obj_cloud.shape[0]):
                        with open(F'{out_dir}/{count:06d}.pkl', 'wb') as fp:
                            pickle.dump({
                                'key': k[i],
                                'hand_pose': p[i],
                                'object_pose': o[i],
                                'object_cloud': c[i],
                                'contact_flag': f[i],
                                'rel_scale': s[i]
                            }, fp)
                        count += 1
                    return count
                count = __export(count)

            if show_result:
                def __visualize():
                    objt_matrix = matrix_from_pose(objt_pose[..., 0:3],
                                                   objt_pose[..., 3:7])
                    hand_matrix = matrix_from_pose(hand_pose[..., 0:3],
                                                   hand_pose[..., 3:7])
                    objt_cloud = apply_pose_tq(objt_pose[..., None, :],
                                               obj_cloud)

                    objt_matrix = dcn(objt_matrix)
                    hand_matrix = dcn(hand_matrix)
                    objt_cloud = dcn(objt_cloud)
                    ctct_flag = dcn(contact_flag)

                    for To, Th, Xo, Cf in zip(objt_matrix, hand_matrix,
                                              objt_cloud, ctct_flag):
                        hand_mesh = copy.deepcopy(collision_fn.mesh)
                        hand_mesh.apply_transform(Th)
                        objt_cloud = trimesh.PointCloud(Xo)
                        objt_cloud.visual.vertex_colors = np.where(
                            Cf[..., None], (255, 0, 0), (0, 0, 255))

                        hand_hull_mesh = copy.deepcopy(collision_fn.hull_mesh)
                        hand_hull_mesh.apply_transform(Th)
                        hand_hull_mesh.visual.vertex_colors = (0, 255, 0)

                        table_mesh = trimesh.creation.box((0.4, 0.5, 0.4))
                        table_xfm = np.eye(4)
                        table_xfm[2, 3] = +0.2
                        table_mesh.apply_transform(table_xfm)

                        trimesh.Scene([
                            # hand_mesh,
                            objt_cloud,
                            hand_hull_mesh,
                            trimesh.creation.axis(),
                            table_mesh
                        ]).show()
                        # return
                __visualize()


if __name__ == '__main__':
    main()
