#!/usr/bin/env python3

from pathlib import Path
from dataclasses import dataclass
from pkm.util.config import ConfigBase

import torch as th
import numpy as np


class ProcessedShapenetSemOcc(th.utils.data.Dataset):
    """
    Preprocessed ShapeNetSem dataset with
    partial-view point clouds and feasible grasps.
    """
    @dataclass
    class Config(ConfigBase):
        data_dir: str = '/opt/datasets/occ/'

    def __init__(self, cfg: Config, split: str = 'train', transform=None):
        super().__init__()
        split_file = Path(cfg.data_dir) / F'{split}.txt'
        with open(split_file, 'r') as fp:
            self.data = fp.readlines()
        self.data = [d.rstrip() for d in self.data]
        # self.data = self.data[:16]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index: data index.

        Returns:
            data: Dictionary with the following entries:
                cloud:     Partial-view point cloud.
                query:     Query points on which sdf values are computed.
                label:     SDF labels for each query point.
                           Sign convention: positive indicates interior.
                path:      Label filename.
                mesh_file: Mesh filename.
                xfm:       Mesh transform to the same frame as `cloud`, `query`.
                           NOTE: the SDF values were computed after `xfm`
                           has been applied.

            NOTE:
                if `transform` is not None, returns transform(data).
        """

        path = str(self.data[index])
        data = dict(np.load(path))
        data['path'] = str(path)
        data['mesh_file'] = str(data['mesh_file'])

        if self.transform is not None:
            data = self.transform(data)
        return data


class ProcessedShapenetSemOccMultiView(ProcessedShapenetSemOcc):
    """
    Preprocessed GraspNet dataset with
    partial-view point clouds and feasible grasps.
    """

    def __init__(self, *args, **kwds):
        self.min_count: int = kwds.pop('min_count', 16)
        self.permute: bool = kwds.pop('permute', True)
        super().__init__(*args, **kwds)

        # Split by object.
        self.data = self._collect_same_object(self.data)

    def _collect_same_object(self, data):
        od = {}
        for f in self.data:
            obj = Path(f).name.rsplit('-', 1)[0]
            if obj not in od:
                od[obj] = []
            od[obj].append(f)
        out = list(od.values())
        # Which ones are not "full-frame"?
        # print(np.unique([len(o) for o in out],
        #                 return_counts=True))
        out = [v for v in out if len(v) >= self.min_count]
        return out

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index: data index.

        Returns:
            data: Dictionary with the following entries:
                cloud:     List of Point clouds from same object.
                query:     List of Query points on same object.
                label:     List of SDF labels for the query points.
                           Sign convention: positive indicates interior.
                path:      Label filename.
                mesh_file: Mesh filename.
                xfm:       Mesh transform to same frame as `cloud`, `query`.
                           NOTE: the SDF values were computed after `xfm`
                           has been applied.

            NOTE:
                if `transform` is not None, returns transform(data).
        """
        paths = self.data[index]
        if self.permute:
            np.random.shuffle(paths)

        # list of dicts
        data = [dict(np.load(path)) for path in paths]
        # dict of lists
        data = {k: [data[i][k] for i in range(len(data))]
                for k in data[0].keys()}

        # Sanitize some types
        data['path'] = [str(p) for p in paths]
        data['mesh_file'] = str(data['mesh_file'][0])

        if self.transform is not None:
            data = self.transform(data)
        return data


def test_multi():
    dataset = ProcessedShapenetSemOccMultiView(
        ProcessedShapenetSemOcc.Config(), 'train')
    for data in dataset:
        print(list(data.keys()))
        print([len(c) for c in data['cloud']])
        print([len(c) for c in data['query']])
        break


def main():
    for split in ('train', 'valid'):
        dataset = ProcessedShapenetSemOcc(ProcessedShapenetSemOcc.Config(),
                                          split)
        print(len(dataset))
        for data in dataset:
            print(list(data.keys()))
            print(data['cloud'].shape, data['cloud'].dtype)
            print(data['query'].shape, data['query'].dtype)
            print(data['label'].shape, data['label'].dtype)
            break


if __name__ == '__main__':
    test_multi()
