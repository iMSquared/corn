#!/usr/bin/env python3


import pickle
from tqdm.auto import tqdm
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
import torch.utils.data

from pkm.util.config import ConfigBase
from cho_util.math import transform as tx

from icecream import ic

from data_common import glob_path, tv_split


class DynDataset:
    """
    Dataset gathered from dynamics simulation.
    """
    @dataclass
    class Config(ConfigBase):
        data_root: str = '/tmp/aff-w-col-toolframe-fixed'
        pattern: str = '*.pkl'
        max_count: int = -1

        valid_ratio: float = 0.2
        seed: int = 0
        preload: bool = False

        device: Optional[str] = None
        cache_pkl: Optional[str] = None

        # Empirically computed offset
        # from `panda_tool` to tip of finger
        contact_offset: float = 0.014

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
        self.data_path = tv_split(paths, cfg.seed, cfg.valid_ratio,
                                  split=split)

        if cfg.preload:
            (self.__keys, self.__dataset) = self._format_data(
                self.load_data(self.data_path,
                               max_count=cfg.max_count,
                               device=cfg.device),
                device=cfg.device)

    def load_data(self, paths, max_count: int = -1,
                  device: Optional[str] = None):
        cfg = self.cfg

        load_from_cache: bool = (cfg.cache_pkl is not None
                                 and Path(cfg.cache_pkl).exists())
        ic(cfg.cache_pkl)

        if load_from_cache:
            ic('load')
            with open(cfg.cache_pkl, 'rb') as fp:
                data = pickle.load(fp)
                return data

        ic('build')
        data = self.__load_all(paths, max_count, device)

        if cfg.cache_pkl is not None:
            # Cache to filesystem for future loads.
            ic('dump')
            with open(cfg.cache_pkl, 'wb') as fp:
                pickle.dump(data, fp)
            # By default, delete and reload from filesystem to ensure
            # data was exported correctly.
            del data
            return self.load_data(paths, max_count, device)
        return data

    def __load_one(self, path: str):
        cfg = self.cfg

        with open(path, "rb") as fp:
            data = pickle.load(fp)

        z = tx.rotation.quaternion.rotate(
            data['initial_hand_pose'][..., 3:7],
            np.asarray([0, 0, 1], dtype=np.float32)
        )
        affordance_point = (
            data['initial_hand_pose'][..., :3]
            + cfg.contact_offset * z
        )
        affordance_normal = z

        out = {
            'aff_pos': affordance_point.reshape(-1),
            'aff_vec': affordance_normal,
            'cloud0': data['initial_cloud'],
            'cloud1': data['final_cloud'],
            'hand0': data['initial_hand_pose'],
            'hand1': data['final_hand_pose'],
            # 'pose0': data['initial_object_pose'],
            # 'pose1': data['final_object_pose']
            'init': data['initial_object_pose'],
            'goal': data['final_object_pose']
        }
        if 'collides' in data:
            out['collide'] = data['collides']
        if 'coll_hand_pose' in data:
            out['hand'] = data['coll_hand_pose']
        if 'touch' in data:
            out['touch'] = data['touch']
        return out

    def __load_all(self, paths, max_count: int = -1,
                   device: Optional[str] = None):
        self.cfg
        count = 0
        dataset = {}
        for path in tqdm(paths, desc="loading data"):
            datum = self.__load_one(path)
            for (k, v) in datum.items():
                if k not in dataset:
                    dataset[k] = []
                dataset[k].append(v)
            count += 1
            if (max_count >= 0) and (count >= max_count):
                break
        dataset = {k: np.stack(v, 0) for k, v in dataset.items()}
        return dataset

    def _format_data(self, dataset, device):
        dtypes = {'collide': bool}
        dataset = {k: th.as_tensor(v, dtype=dtypes.get(k, th.float),
                                   device=device) for k, v in dataset.items()}
        keys = sorted(dataset.keys())
        dataset = th.utils.data.TensorDataset(*[dataset[k] for k in keys])
        return (keys, dataset)

    def __getitem__(self, index):
        cfg = self.cfg
        if cfg.preload:
            out = self.__dataset[index]
            out = {k: v for (k, v) in zip(self.__keys, out)}
        else:
            out = self.__load_one(self.data_path[index])

        out = {k: th.as_tensor(v, device=self.cfg.device)
               for (k, v) in out.items()}
        if self.transform is not None:
            out = self.transform(out)
        return out

    def __len__(self):
        cfg = self.cfg
        if cfg.preload:
            return len(self.__dataset)
        else:
            return len(self.data_path)


def main():
    dataset = DynDataset(DynDataset.Config(
        '/tmp/dyn-v12',
        preload=False,
        cache_pkl='~/.cache/pkm/aff-dyn.pkl'
    ), split='train')
    for data in dataset:
        print({k: v.shape for k, v in data.items()})
        break
    # 1048576it [00:04, 216229.52it/s]
    # {'aff_pos': (3,), 'aff_vec': (3,),
    # 'cloud0': (512, 3), 'cloud1': (512, 3),
    # 'hand0': (7,), 'hand1': (7,), 'init': (7,), 'goal': (7,),
    # 'collide': (512,), 'hand': (7,), 'touch': (1,)}


if __name__ == '__main__':
    main()
