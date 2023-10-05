#!/usr/bin/env python3

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Mapping, Any, Optional
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch as th

from pkm.util.math_util import (quat_multiply,
                                quat_inverse,
                                matrix_from_quaternion
                                )
from model_rot import rot6d_to_matrix
from data_common import glob_path, tv_split
from preprocess import Preprocess
from pkm.models.cloud.point_mae import (
    subsample
)


def subdict(d: Mapping[str, Any],
            keys: Iterable[str]):
    return {k: d[k] for k in keys}


class RotDataset:
    """
    Dataset gathered from synthetic (geometric) collision.
    """

    @dataclass
    class Config:
        data_root: str = ''
        pattern: str = '*.pkl'
        max_count: int = -1

        valid_ratio: float = 0.2
        seed: int = 0
        preload: bool = True

        device: Optional[str] = None

        cloud_size: int = 512

    def __init__(self, cfg: Config, split: str = 'train',
                 transform=None):
        self.cfg = cfg
        self.transform = transform
        # assert (self.transform is not None)

        paths = glob_path(cfg.data_root,
                          cfg.pattern,
                          cfg.max_count)
        self.paths = tv_split(paths, cfg.seed, cfg.valid_ratio,
                              split)

        self.data = None
        if cfg.preload:
            self.data = self.__load_all(self.paths,
                                        cfg.max_count)

    def __load_one(self, path: str):
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        data = subdict(data, ['object_cloud'])
        return data

    def __load_all(self,
                   paths: Iterable[str],
                   max_count: int = -1):
        data = defaultdict(list)
        count = 0
        for path in tqdm(paths, desc="RotDataset.preload"):
            entry = self.__load_one(path)
            for k in entry.keys():
                data[k].append(entry[k])
            count += 1
            if (max_count >= 0) and (count >= max_count):
                break
        data = {k: np.stack(v, axis=0) for (k, v) in data.items()}
        return data

    def __getitem__(self, index):
        if self.data is not None:
            data = self.data[index]
        else:
            data = self.__load_one(self.paths[index])

        # Format, just in case
        data = {k: th.as_tensor(v, device=self.cfg.device)
                for (k, v) in data.items()}
        data['object_cloud'] = subsample(
            data['object_cloud'], self.cfg.cloud_size
        )
        aux1 = {}
        aux2 = {}
        data1 = {k: v.clone() for (k, v) in data.items()}
        data1 = self.transform(data1, aux1)
        data2 = self.transform(data, aux2)

        quat1 = aux1['quat']
        quat2 = aux2['quat']
        quat_diff = quat_multiply(quat1,
                                  quat_inverse(quat2))
        rot_diff = matrix_from_quaternion(quat_diff)[..., :2, :].reshape(
            *quat_diff.shape[:-1], -1
        )
        # rot_diff = matrix_from_quaternion(quat_diff)

        data = {
            'object_cloud1': data1['object_cloud'],
            'object_cloud2': data2['object_cloud'],
            'rot_diff': rot_diff
        }

        return data

    def __len__(self):
        return len(self.paths)


def print_shapes():
    process: Preprocess.Config = Preprocess.Config(
        ref_cloud_keys=('object_cloud',),
        cloud_keys=(('object_cloud',)
                    ),
        noise_keys=[
            'object_cloud'
        ],

        rotate=True,
        translate=0.1,
        noise_scale=0.01,
        translate_2d=False,
        use_6d_rot=True
    )
    test_transform = Preprocess(process, 'cpu')
    dataset = RotDataset(RotDataset.Config('/tmp/col10',
                                           preload=False),
                                           transform=test_transform)
    for data in dataset:
        print({k: v.shape for k, v in data.items()})
        break

def show():
    import copy
    import open3d as o3d
    from cho_util.math import transform as tx
    from pkm.util.vis.win_o3d import (
        AutoWindow,
        o3d_frame_from_pose,
        o3d_cloud_from_cloud
    )
    from pkm.data.transforms.aff import get_gripper_mesh
    process: Preprocess.Config = Preprocess.Config(
        ref_cloud_keys=('object_cloud',),
        cloud_keys=(('object_cloud',)
                    ),
        noise_keys=[
            'object_cloud'
        ],

        rotate=True,
        translate=0.0,
        noise_scale=0.00,
        rotate_2d=False, 
        translate_2d=False,
        use_6d_rot=True,

    )
    test_transform = Preprocess(process, 'cpu')
    dataset = RotDataset(RotDataset.Config('/tmp/docker/col-12-2048/',
    # dataset = RotDataset(RotDataset.Config('/tmp/col10/',
                                           preload=False,
                                           max_count=128),
                                           transform=test_transform)

    win = AutoWindow()
    vis = win.vis

    for data in dataset:
        print({k: v.shape for k, v in data.items()})
        cloud1 = o3d_cloud_from_cloud(data['object_cloud1'], color= (1, 0, 0))
        cloud2 = o3d_cloud_from_cloud(data['object_cloud2'], color= (0, 1, 0))
        cloud3 = o3d_cloud_from_cloud(data['object_cloud2'], color= (0, 0, 1))
        pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2)
        # rot_mat = data['rot_diff'].cpu().numpy()
        rot_mat = rot6d_to_matrix(data['rot_diff']).cpu().numpy()
        T_pc = np.eye(4)
        T_pc[:3, 3] = cloud2.get_center()
        T = np.eye(4)
        T[:3, :3] = rot_mat
        transform = T_pc @ T @tx.invert(T_pc)
        # T[:3, -1] = cloud1.get_center()
        cloud3.transform(transform)
        vis.add_geometry('cloud1', cloud1, color=(1, 1, 1, 1))
        vis.add_geometry('cloud2', cloud2, color=(1, 1, 1, 1))
        vis.add_geometry('cloud3', cloud3, color=(1, 1, 1, 1))
        # vis.add_geometry('rot', pose_mesh, color=(1, 1, 1, 1))
        win.wait()

def main():
    # print_shapes()
    show()


if __name__ == '__main__':
    main()
