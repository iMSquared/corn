#!/usr/bin/env python3

from pathlib import Path
from dataclasses import dataclass
from pkm.util.config import ConfigBase

import torch as th
import numpy as np

from pkm.data.transforms.common import weights_to_splits
from pkm.data.shapenet_sem import load_mug_ids


def _create_splits(d1, d2, seed: int = 0):
    a = list(Path(d1).glob('*.npz'))
    b = list(Path(d2).glob('*.npz'))
    i = list(set([x.name for x in a]).intersection([x.name for x in b]))
    t, v = weights_to_splits(len(i), (0.8, 0.2))
    np.random.default_rng(seed).shuffle(i)
    split_t = i[:t]
    split_v = i[t:t + v]
    return (split_t, split_v)


def _apply_transform(T, x):
    """
    T: (4,4)
    x: (...,3)
    """
    return x @ T[:3, :3].T + T[:3, 3]


class PSS2(th.utils.data.Dataset):
    """
    Preprocessed ShapeNetSem dataset with
    partial-view point clouds and sdf labels.
    """
    @dataclass
    class Config(ConfigBase):
        sdf_dir: str = '/opt/datasets/ShapeNetSem/sdf3/'
        cloud_dir: str = '/opt/datasets/ShapeNetSem/cloud/'
        cloud_full_dir: str = '/opt/datasets/ShapeNetSem/cloud-full/'

    def __init__(self, cfg: Config, split: str = 'train', transform=None):
        super().__init__()
        self.cfg = cfg
        self.split = split

        split_file = Path(F'/home/user/.cache/pkm/pss2-sdf3-{split}.txt')
        if not split_file.exists():
            self._generate_splits(cfg)
        with open(split_file, 'r') as fp:
            self.data = fp.readlines()
        self.data = [d.rstrip() for d in self.data]

        # NOTE: filter by category
        # in practice, only using mugs for now.
        # mugs = load_mug_ids('/opt/datasets/ShapeNetSem/metadata.csv')
        # self.data = [d for d in self.data
        #              if d.split('.', maxsplit=1)[0] in mugs]
        self.transform = transform

    
    def _generate_splits(self, cfg:Config):
        t, v = _create_splits(cfg.sdf_dir, cfg.cloud_dir)
        for s, fs in [('train', t), ('valid', v)]:
            split_file = (F'/home/user/.cache/pkm/pss2-sdf3-{s}.txt')
            with open(split_file, 'w') as fp:
                fp.write('\n'.join(fs))

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
        cfg = self.cfg

        filename = str(self.data[index])
        data_sdf = dict(np.load(Path(cfg.sdf_dir) / filename,
                                allow_pickle=True))
        data_cloud = dict(np.load(Path(cfg.cloud_dir) / filename,
                                  allow_pickle=True))
        data_cloud_full = dict(np.load(Path(cfg.cloud_full_dir) / filename,
                                       allow_pickle=True))
        data = {**data_sdf, **data_cloud}
        data['mesh_file'] = str(data['mesh_file'])

        # Apply mesh transforms (used for the cloud) to
        # SDF-related quantities.
        # NOTE: sdf transform below is not valid unless
        # the scale was isotropic.
        data['query'] = _apply_transform(
            data['mesh_transform'], data['query'])
        data['label'] *= data['mesh_transform'][0, 0]
        data['cloud'] = data.pop('points')
        data['cloud_full'] = data_cloud_full['points']

        if self.transform is not None:
            data = self.transform(data)
        return data


def main():
    dataset = PSS2(PSS2.Config(), 'train')
    for data in dataset:
        print(list(data.keys()))
        return
    pass


if __name__ == '__main__':
    main()
