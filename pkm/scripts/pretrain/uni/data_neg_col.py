#!/usr/bin/env python3

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Mapping, Any, Optional
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch as th

from data_common import glob_path, tv_split


def subdict(d: Mapping[str, Any],
            keys: Iterable[str]):
    return {k: d[k] for k in keys}


class NegColDataset:
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
        preload: bool = False

        device: Optional[str] = None
        cache_pkl: Optional[str] = None

        def __post_init__(self):
            if self.cache_pkl is not None:
                self.cache_pkl = str(Path(self.cache_pkl).expanduser())

    def __init__(self, cfg: Config, split: str = 'train',
                 transform=None):
        self.cfg = cfg
        self.transform = transform

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
        data = subdict(data, ['hand_pose', 'object_pose',
                              'object_cloud', 'contact_flag',
                              'empty_cloud', 'empty_contact_flag',
                              ])
        return data

    def __load_all(self,
                   paths: Iterable[str],
                   max_count: int = -1):
        data = defaultdict(list)
        count = 0
        for path in tqdm(paths, desc="NegColDataset.preload"):
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
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.paths)


def print_shapes():
    dataset = NegColDataset(
        NegColDataset.Config(
            '/input/genom/col-with-empty/',
            preload=False))
    for data in dataset:
        print({k: v.shape for k, v in data.items()})
        break
    # OUTPUT:
    # 1422840it [00:06, 217269.30it/s]
    # {'hand_pose': torch.Size([7]), 'object_cloud': torch.Size([512, 3]),
    # 'contact_flag': torch.Size([512]), 'object_pose': torch.Size([7])}


def show():
    import copy
    from cho_util.math import transform as tx
    from pkm.util.vis.win_o3d import (
        AutoWindow,
        o3d_frame_from_pose,
        o3d_cloud_from_cloud
    )
    from pkm.data.transforms.aff import get_gripper_mesh

    from preprocess import Preprocess

    dataset = NegColDataset(
        NegColDataset.Config(
            '/input/genom/col-with-empty/',
            preload=False,
            max_count=128))

    win = AutoWindow()
    vis = win.vis

    mesh = get_gripper_mesh(cat=True, frame='panda_tool')
    hand_mesh_base = mesh.as_open3d
    process = Preprocess(Preprocess.Config(
        ref_cloud_keys=['object_cloud', 'empty_cloud'],
        cloud_keys=[
            ['cloud', 'object_cloud', 'cloud0', 'cloud1', 'contact_flag'],
            ['empty_contact_flag']
        ],
        noise_keys=[
            'object_cloud', 'empty_cloud'
        ]
    ))

    for data in dataset:
        print({k: v.shape for k, v in data.items()})
        data2 = process(data)
        print({k: v.shape for k, v in data2.items()})
        break

        cloud = o3d_cloud_from_cloud(data['object_cloud'], color=np.where(
            data['contact_flag'][..., None], (1, 0, 0), (0, 0, 1)))
        vis.add_geometry('cloud', cloud, color=(1, 1, 1, 1))

        empty_cloud = o3d_cloud_from_cloud(data['empty_cloud'], color=np.where(
            data['empty_contact_flag'][..., None], (1, 1, 0), (0, 1, 1)))
        vis.add_geometry('empty_cloud', empty_cloud, color=(1, 1, 1, 1))

        vis.add_geometry(
            'hand_axes', o3d_frame_from_pose(
                data['hand_pose']), color=(
                1, 1, 1, 1))

        hand_mesh = copy.deepcopy(hand_mesh_base)
        hand_xfm = np.eye(4)
        hand_xfm[: 3, : 3] = tx.rotation.matrix.from_quaternion(
            data['hand_pose'][..., 3: 7])
        hand_xfm[:3, 3] = data['hand_pose'][..., 0:3]
        hand_mesh.transform(hand_xfm)
        vis.add_geometry('hand_mesh', hand_mesh, color=(1, 1, 1, 0.3))
        vis.add_geometry(
            'object_pose',
            o3d_frame_from_pose(data['object_pose']), color=(1, 1, 1, 1))
        win.wait()
        # break


def main():
    show()


if __name__ == '__main__':
    main()
