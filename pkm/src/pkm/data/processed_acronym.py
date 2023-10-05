#!/usr/bin/env python3

from pathlib import Path
from dataclasses import dataclass
from pkm.util.config import ConfigBase

import torch as th
import numpy as np


def _apply_mesh_transform_to_grasp(T: np.ndarray, x: np.ndarray,
                                   in_place: bool = True) -> np.ndarray:
    """
    Apply SE(3) transform to grasp parameters.

    T: (4,4)     homogeneous matrix (T@x convention)
    x: (..., 9)  grasp parameters, broadcastable over (...) dims.
                 format: (translation;quaternion;depth;width)
    """
    if not in_place:
        x = x.copy()

    # [1] update gripper center (I think?)
    x[..., :3] = x[..., :3] @ T[:3, :3].T + T[:3, 3]

    # [2] apply scale parameter, assuming isotropic scaling
    scale = np.cbrt(np.linalg.det(T[:3, :3]))
    x[..., -2:] *= scale
    return x


class ProcessedAcronym(th.utils.data.Dataset):
    """
    Preprocessed Acronym dataset with
    partial-view point clouds and grasp labels.
    """
    @dataclass
    class Config(ConfigBase):
        sdf_dir: str = '/opt/datasets/ShapeNetSem/sdf3/'
        cloud_dir: str = '/opt/datasets/ShapeNetSem/cloud/'
        grasp_dir: str = '/opt/datasets/acronym/processed2/'

    def __init__(self, cfg: Config, split: str = 'train', transform=None):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.cloud_dir = Path(cfg.cloud_dir)
        self.grasp_dir = Path(cfg.grasp_dir) / split

        # Filter by availability of the point cloud data.
        self.data = [f for f in self.grasp_dir.glob('*.npz') if
                     self._cloud_from_grasp(f).exists()]

        self.transform = transform

    def _cloud_from_grasp(self, grasp_file: str):
        cloud_file = Path(grasp_file).stem.split('_', 2)[1] + '.npz'
        return self.cloud_dir / cloud_file

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
        grasp_file = str(self.data[index])
        cloud_file = self._cloud_from_grasp(grasp_file)

        data_cloud = dict(np.load(self.cloud_dir / cloud_file,
                                  allow_pickle=True))
        data_grasp = dict(np.load(self.grasp_dir / grasp_file,
                                  allow_pickle=True))
        data = {**data_grasp, **data_cloud}
        if 'mesh_file' in data:
            data['mesh_file'] = str(data['mesh_file'])
        # data['cloud_file'] = str(data['cloud_file'])
        # data['grasp_file'] = str(data['grasp_file'])

        # Apply mesh transforms (used for the cloud) to
        # SDF-related quantities.
        # NOTE: sdf transform below is not valid unless
        # the scale was isotropic.
        data['grasp'] = _apply_mesh_transform_to_grasp(
            data['mesh_transform'],
            data.pop('grasps'))
        data['cloud'] = data.pop('points')

        if self.transform is not None:
            data = self.transform(data)
        return data


def main():
    dataset = ProcessedAcronym(ProcessedAcronym.Config(),
                               'train')
    for data in dataset:
        print(list(data.keys()))
        return
    pass


if __name__ == '__main__':
    main()
