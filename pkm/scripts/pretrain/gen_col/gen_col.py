#!/usr/bin/env python3

from typing import Optional, Tuple
import pickle
from pathlib import Path
from yourdfpy import URDF
from functools import partial
import torch as th

from pkm.env.scene.dgn_object_set import DGNObjectSet
from pkm.env.scene.acronym_object_set import AcronymObjectSet
from pkm.env.scene.combine_object_set import CombinedObjectSet
from pkm.env.scene.filter_object_set import FilteredObjectSet, dataset_hasattr

import pytorch_volumetric as pv
from tempfile import TemporaryDirectory
from tqdm.auto import tqdm
from pkm.util.path import ensure_directory
from pkm.models.common import merge_shapes
from pkm.util.math_util import (
    apply_pose_tq,
    invert_pose_tq,
    compose_pose_tq,
    quat_rotate,
    matrix_from_pose
)
from pkm.util.torch_util import dcn
import copy
import trimesh
from icecream import ic
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import einops

from pkm.data.transforms.sdf.sdf_cvx_set_th3 import SignedDistanceTransform

from pkm.data.transforms.sdf.sdf_cvx_set_th3 import IsInHulls
from pkm.data.transforms.df_th3 import DistanceTransform
from pkm.data.transforms.aff import CheckGripperCollisionV2


class CheckGripperCollision():
    """
    Check if a given point collides with the gripper.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device
        urdf = URDF.load(
            # '../../../src/pkm/data/assets/fe-gripper/robot.urdf',
            '../../../src/pkm/data/assets/franka_description/robots/franka_panda.urdf',
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_collision_mesh=False)

        cache_path = Path('~/.cache/pkm/gripper_sdf.pkl')
        ensure_directory(cache_path.expanduser().parent)
        # mesh = urdf.scene.dump(concatenate=True)
        # mesh = urdf.scene.subscene('panda_hand').dump(
        #     concatenate=True)
        mesh = urdf.collision_scene.subscene('panda_hand').dump(
            concatenate=True)
        # trimesh.Scene([mesh, trimesh.creation.axis()]).show()
        # mesh = urdf.collision_scene.dump(concatenate=True)

        # with TemporaryDirectory() as tmpdir:
        mesh_file: str = F'/tmp/gripper.obj'
        mesh.export(mesh_file)
        obj = pv.MeshObjectFactory(mesh_file,
                                   plausible_suboptimality=1e-4)
        sdf = pv.MeshSDF(obj)
        # csdf = pv.CachedSDF('gripper',
        #                    resolution=0.001,
        #                    range_per_dim=obj.bounding_box(padding=0.1),
        #                    gt_sdf=sdf,
        #                    device=device,
        #                    cache_path=str(cache_path),
        #                    debug_check_sdf=True)
        # query_range = np.array(obj.bounding_box(padding=0.1))
        # query_range[0] = query_range[0].mean(axis=0,
        #                                      keepdims=True)
        # pv.draw_sdf_slice(sdf, query_range)
        # plt.show()
        # self.__cache(obj, sdf, resolution=0.001)

        self.sdf = sdf
        self.mesh = mesh

    def __csdf(self, point: th.Tensor):
        pass

    def __cache(self, obj, sdf, resolution: float = 0.01):
        box = obj.bounding_box(padding=0.0)
        ptp = box[:, 1] - box[:, 0]
        num = np.ceil(ptp / resolution).astype(np.int32)
        xyz = box[:, 0] + resolution * np.stack(np.meshgrid(
            np.arange(num[0]),
            np.arange(num[1]),
            np.arange(num[2]),
            indexing='ij'), axis=-1)  # 264 405 339 3

        self.res2 = (xyz[-1, -1, -1] - xyz[0, 0, 0]) / (num - 1)
        # print(resolution)
        # print(self.res2)
        # print(xyz.reshape(-1, 3).min(axis=0),
        #       xyz.reshape(-1, 3).max(axis=0))
        d, g = sdf(th.as_tensor(xyz.reshape(-1, 3),
                                dtype=th.float,
                                device=self.device))  # 264 305 339
        # N, N3

        if True:
            # dg = th.cat([d[...,None], g], dim=-1)
            # dg = dg.view(*xyz.shape[:-1], 4)
            # dg = einops.rearrange(dg, '... c -> c ...') # C/HWD
            dg = th.cat([d[None], g.T], dim=0)
            dg = dg.view(4, *xyz.shape[:-1])
            # print('dg', dg.shape)

            # dg = val.reshape(*xyz.shape[:-1], 4)
            # gval = gval.reshape(*xyz.shape)
            # xyzv = th.cat([xyz, val[...,None]], dim=-1)
            # numel = np.prod(num-1)
            # stride0 = dg.stride()
            # dg_blocks = th.as_strided(dg,
            #                          size=(
            #                              # BLOCKS
            #                              num[0] - 2, num[1] - 2, num[2] - 2,
            #                              # CHANNELS
            #                              4,
            #                              # SUB_BLOCKS
            #                              3, 3, 3),
            #                          stride=(*stride0[1:], *stride0),
            #                          storage_offset=0)
            self.dg = dg  # 1,C,H,W,D
            # self.dg_blocks = dg_blocks
        else:
            stride0 = d.stride()
            d_blocks = th.as_strided(
                d,
                size=(
                    num[0] - 2,
                    num[1] - 2,
                    num[2] - 2,
                    3,
                    3,
                    3),
                stride=(
                    *stride0,
                    *stride0),
                storage_offset=0)
            self.d_blocks = d_blocks
        self.block_stride = th.as_tensor([(num[1] - 2) * (num[2] - 2),
                                          (num[2] - 2),
                                          1],
                                         dtype=th.long,
                                         device=self.device)

        # self.dg_blocks = dg_blocks.view(-1,3,3,3,4)
        self.cmin = th.as_tensor(xyz[0, 0, 0],
                                 dtype=th.float,
                                 device=self.device)
        self.cmax = th.as_tensor(xyz[-1, -1, -1],
                                 dtype=th.float,
                                 device=self.device)
        # print(self.cmin)
        # print(self.cmax)
        self.cptp = self.cmax - self.cmin
        self.resolution = resolution

        # self.__query_cache(
        #     th.zeros((8, 3), device=self.device,
        #              dtype=th.float))

        # raise ValueError('stop')
        # val = box[None, :, 0] + th.linspace(num[0])[:, None]
        # center = 0.5 * (box[:, 0] + box[:, 1])
        # # xyz  =
        # coord = th.cartesian_prod(
        #     th.arange(box[0, 0], box[0, 1]),
        #     th.arange(box[1, 0], box[1, 1]),
        #     th.arange(box[2, 0], box[2, 1]))

    def __query_cache(self, points: th.Tensor):
        # currently expects Nx3 inputs
        # block_index = ((points - self.cmin + 0.5*self.resolution) / self.resolution).to(
        #         dtype=th.long)
        # print(block_index.shape, self.block_stride.shape)
        # flat_index = (block_index * self.block_stride).sum(dim=-1)
        # src_block = self.dg_blocks[block_index[...,0],
        #                           block_index[...,1],
        #                           block_index[...,2]] # Nx3x3x3
        # print(src_block.shape)
        # local_point = (points - self.cmin) % self.resolution
        # offsets     = local_point # still (Nx3)
        # return F.grid_sample(src_block, offsets[:,None,None,None])[...,0,0,0]
        grid_dim = (th.as_tensor(
            self.dg.shape[-3:],
            dtype=points.dtype, device=points.device) - 1)
        loc = (
            (points - self.cmin).div_(self.resolution).mul_(
                2 / grid_dim).sub_(1))  # 0 ~ n-1
        # print(loc.min(dim=0), loc.max(dim=0))
        loc = loc[:, None, None, None, :]

        # s0, s1 = (0.0, 0.0, 0.0), np.subtract(self.dg.shape[-3:], 1)
        # t0, t1 = (-1.0, -1.0, -1.0), (+1.0, +1.0, +1.0)
        # t0 + (t1-t0) / (s1-s0) * x
        # x = t0 + (x - s0) * (t1-t0)/(s1-s0)
        # s0 = 0 so
        # x' = (t0 + x * (t1-t0)/s1)
        # x' = x*(t1-t0)/s1 + t0
        #    = x*(2)/s1 -1

        out = F.grid_sample(
            # 1,4,H,W,D
            self.dg[None].expand(points.shape[0], *self.dg.shape),
            # N,3
            # (points - self.cmin)[:,None,None,None,:],
            loc,
            align_corners=True,
            # align_corners=False,
            # mode='nearest'
            mode='bilinear'
        )
        return out[..., 0, 0, 0]  # Xx4

    def __call__(self, point: th.Tensor):
        # disable cache.
        return self.sdf(point)
        mask = th.logical_and(
            (point > self.cmin).all(dim=-1),
            (point < self.cmax).all(dim=-1))
        out = th.empty((*point.shape[:-1], 4),
                       dtype=point.dtype,
                       device=point.device)
        # Fill
        # print(out[mask].shape)
        out[mask] = self.__query_cache(point[mask])
        d, g = self.sdf(point[~mask])
        # print('d', d.shape)
        # print('g', g.shape)
        # print(out.shape)
        # print('mask', mask.shape)
        out[~mask] = th.cat([d[..., None], g], dim=-1)
        # out[~mask, ...,1:] = g
        return out[..., 0], out[..., 1:], mask


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
    device: str = 'cuda:1'
    batch_size: int = 64
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
            data_path='/input/ACRONYM/meta-v1/cloud-2048',
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
