#!/usr/bin/env python3

import pickle
from dataclasses import dataclass
from typing import Optional, Iterable
import numpy as np
import torch as th
from collections import defaultdict

from tqdm.auto import tqdm

from torch.utils.data import (
    TensorDataset,
)
from torchvision.transforms import (
    Compose
)

from pkm.data.shapenet import ShapeNetDataset
from pkm.data.transforms.common import WrapDict, SelectKeys
from pkm.data.transforms.io_xfm import LoadMesh
from pkm.data.transforms.sample_points import SampleSurfacePointsFromMesh
from pkm.util.torch_util import dcn

from data_common import glob_path, tv_split

# FIXME: this should _not_ be here
from pkm.models.cloud.point_mae import subsample


def ceildiv(x, y):
    return (x + y - 1) // y


class SubsampleCloud:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, x: np.ndarray):
        x = th.as_tensor(x)
        x = subsample(x, self.n)
        return dcn(x)


def load_single_pkl(path: str = '/input/ShapeNetCore.v2.cloud/full.pkl',
                    device: str = 'cpu'):
    with open(path, 'rb') as fp:
        d = pickle.load(fp)
    clouds = th.as_tensor(np.stack(d.values(), axis=0),
                          dtype=th.float,
                          device=device)
    return clouds


class OnlineSamplePcdDataset:
    def __init__(self,
                 mesh_dataset,
                 cloud_size: int,
                 preload: bool = False,
                 oversample_ratio: float = 2.0,
                 max_count: int = -1,
                 transform=None,
                 ):
        self.mesh_dataset = mesh_dataset
        sample_size: int = (
            int(cloud_size * oversample_ratio) if preload
            else
            cloud_size
        )
        self.sample = Compose([
            WrapDict(LoadMesh(as_mesh=True), 'filename', 'mesh'),
            SampleSurfacePointsFromMesh(
                sample_size,
                key='cloud'),
            SelectKeys(['cloud'])
        ])
        self.transform = transform

        self.data = None
        if preload:
            self.data = self.__load_all(max_count)

    def __load_one(self, index: int):
        mesh = self.mesh_dataset[index]
        out = self.sample({'filename': mesh['filename']})
        if self.transform is not None:
            out = self.transform(out)
        return out

    def __load_all(self, max_count: int = -1):
        data = defaultdict(list)
        count = 0
        for i in tqdm(range(len(self))):
            entry = self.__load_one(i)
            for k in entry.keys():
                data[k].append(entry[k])
            count += 1
            if (max_count >= 0) and (count >= max_count):
                break
        data = {k: np.stack(v, axis=0) for (k, v) in data.items()}
        return data

    def __getitem__(self, index):
        if self.data is not None:
            # data = self.data[index]
            data = {k: v[index] for k, v in self.data.items()}
        else:
            data = self.__load_one(index)
        # Format, just in case
        data = {k: th.as_tensor(v, device=self.cfg.device)
                for (k, v) in data.items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        n: int = len(self.mesh_dataset)
        return n


class OfflinePcdDataset:
    """
    Dataset of standalone point clouds.
    """

    @dataclass
    class Config:
        data_pkl: Optional[str] = None
        cloud_size: int = 512
        device: Optional[str] = 'cpu'
        valid_ratio: float = 0.2
        preload: bool = True
        shuffle: bool = True

    def __init__(self,
                 cfg: Config,
                 split: str = 'train',
                 transform=None,
                 batch_size=()
                 ):
        self.__batch_size = batch_size
        self.cfg = cfg
        assert (cfg.preload)
        self.transform = transform

        # all_dataset = TensorDataset(
        #     load_single_pkl(cfg.data_pkl, device=cfg.device)
        # )
        all_dataset = load_single_pkl(cfg.data_pkl, device=cfg.device)
        self.__data = all_dataset
        num_valid = int(len(all_dataset) * cfg.valid_ratio)
        num_train = len(all_dataset) - num_valid

        indices = th.randperm(len(all_dataset))
        train_indices, valid_indices = (
            indices[:num_train], indices[num_train:]
        )

        # train_dataset, valid_dataset = (
        #     random_split(all_dataset, [num_train, num_valid])
        # )

        if split == 'train':
            self.__indices = train_indices
        else:
            self.__indices = valid_indices

        subsample = WrapDict(
            SubsampleCloud(cfg.cloud_size),
            'cloud', 'cloud')
        if transform is None:
            transform = subsample
        else:
            transform = Compose([
                subsample,
                transform
            ])
        self.transform = transform

        self.__batched_indices = None
        if isinstance(self.__batch_size, int):
            self.__reset_batch_dataset()

    def __reset_batch_dataset(self):
        cfg = self.cfg

        if cfg.shuffle:
            perm = th.randperm(len(self.__indices))
            self.__indices = self.__indices[perm]

        num_chunks = ceildiv(len(self.__indices), self.__batch_size)
        self.__batched_indices = th.chunk(self.__indices,
                                          num_chunks)

    def __getitem__(self, index):
        cfg = self.cfg

        try:
            if self.__batch_size == ():
                out = self.__data[self.__indices[index]]
            else:
                out = self.__data[self.__batched_indices[index]]
        except KeyError:
            if isinstance(self.__batch_size, int):
                self.__reset_batch_dataset()
            raise

        out = {'cloud': out}

        if self.transform is not None:
            out = self.transform(out)

        return out

    def __len__(self):
        if self.__batch_size == ():
            out = len(self.__indices)
        else:
            out = len(self.__batched_indices)
        return out

        # return len(self.__dataset)
        # n: int = len(self.__dataset)
        # b = self.__batch_size
        # if b == ():
        #     return n
        # if isinstance(b, int):
        #     return (n + b - 1) // b
        # raise ValueError('nope')


class OfflineSplitPcdDataset:
    """
    Dataset of standalone point clouds.
    """

    @dataclass
    class Config:
        data_root: str = ''
        pattern: str = '*.pkl'
        max_count: int = -1
        preload: bool = False

        valid_ratio: float = 0.2
        seed: int = 0

    def __init__(self,
                 cfg: Config,
                 split: str = 'train',
                 transform=None):
        self.cfg = cfg
        self.transform = transform

        paths = glob_path(cfg.data_root,
                          cfg.pattern,
                          cfg.max_count)
        self.data_path = tv_split(paths,
                                  cfg.seed,
                                  cfg.valid_ratio,
                                  split)
        self.data = None
        if cfg.preload:
            self.data = self.__load_all(self.data_path, cfg.max_count)

    def __load_all(self,
                   paths: Iterable[str],
                   max_count: int = -1):
        data = defaultdict(list)
        count = 0
        for path in tqdm(paths, desc="OfflineSplitPcdDataset.preload"):
            entry = {'cloud': self.__load_one(path)}
            for k in entry.keys():
                data[k].append(entry[k])
            count += 1
            if (max_count >= 0) and (count >= max_count):
                break
        data = {k: np.stack(v, axis=0) for (k, v) in data.items()}
        return data

    def __load_one(self, path: str):
        return np.load(path)

    def __getitem__(self, index):
        cfg = self.cfg
        if cfg.preload:
            out = {k: v[index] for (k, v) in self.data.items()}
        else:
            out = {'cloud': self.__load_one(self.data_path[index])}

        if self.transform is not None:
            out = self.transform(out)

        return out

    def __len__(self):
        return len(self.paths)


class ShapeNetOnlineSamplePcdDataset(OnlineSamplePcdDataset):
    """
    Online sampling of points
    from ShapeNetCore mesh dataset.
    """
    @dataclass
    class Config:
        shapenet: ShapeNetDataset.Config = ShapeNetDataset.Config(
            data_dir='/input/ShapeNetCore.v2'
        )
        cloud_size: int = 512
        valid_ratio: float = 0.2
        preload: bool = False
        oversample_ratio: float = 2.0
        max_count: int = -1
        device: str = 'cpu'

    def __init__(self,
                 cfg: Config,
                 split: str = 'train',
                 transform=None):
        self.cfg = cfg
        mesh_dataset = ShapeNetDataset(cfg.shapenet,
                                       split=split,
                                       split_ratio=1.0 - cfg.valid_ratio,
                                       load_binvox=False,
                                       transform=None)
        subsample = WrapDict(
            SubsampleCloud(cfg.cloud_size),
            'cloud', 'cloud')
        if transform is None:
            transform = subsample
        else:
            transform = Compose([
                subsample,
                transform
            ])

        super().__init__(mesh_dataset,
                         cfg.cloud_size,
                         cfg.preload,
                         cfg.oversample_ratio,
                         cfg.max_count,
                         transform)


def test_shapenet_online():
    dataset = ShapeNetOnlineSamplePcdDataset(
        ShapeNetOnlineSamplePcdDataset.Config(
            preload=True,
            max_count=128))
    for data in dataset:
        print({k: v.shape for k, v in data.items()})
        break
    # {'cloud': (512, 3)}


def test_offline_monolith():
    for preload in (True,):
        dataset = OfflinePcdDataset(
            OfflinePcdDataset.Config(
                data_pkl='/input/ShapeNetCore.v2.cloud/full.pkl',
                preload=preload),
            batch_size=4)
        print(len(dataset))
        for data in dataset:
            print({k: v.shape for k, v in data.items()})
            break
    # {'cloud': (512, 3)}


def test_offline_split():
    for preload in (True, False):
        dataset = OfflineSplitPcdDataset(
            OfflineSplitPcdDataset.Config(
                data_root='/input/ACRONYM/meta-v0/cloud/',
                pattern='*.npy',
                max_count=128,
                preload=preload), split='train')
        for data in dataset:
            print({k: v.shape for k, v in data.items()})
            break
    # {'cloud': (512, 3)}


def main():
    test_offline_monolith()
    # test_offline_split()


if __name__ == '__main__':
    main()
