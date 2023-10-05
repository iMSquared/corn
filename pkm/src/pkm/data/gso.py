#!/usr/bin/env python3

from typing import Iterable, Optional, Callable
import numpy as np
import torch as th
from pathlib import Path
from zipfile import ZipFile
from contextlib import redirect_stderr, redirect_stdout
import trimesh
import io

from pkm.data.util import (
    split_files,
    sample_occlusion
)
from pkm.data.binvox_rw import (
    read_as_3d_array
)


_OVERSAMPLE_RATIO: float = 4


class GSODataset(th.utils.data.Dataset):
    def __init__(self,
                 split: str = 'train',
                 split_ratio: float = 0.8,
                 n_full: int = 2048,
                 n_part: int = 1024,
                 path: str = '/opt/datasets/GSO/ScannedObjects',
                 load_binvox: bool = True,
                 transform: Optional[Callable] = None
                 ):
        self.rng = np.random.default_rng(0)
        self.root = Path(path)
        self.all_models = list(sorted([
            f for f in self.root.glob('*') if (f / 'meshes').is_dir()]))

        # TODO: splits should be cached.
        splits = {}
        splits['train'], splits['valid'] = split_files(
            self.all_models, self.rng, split_ratio)

        self.splits = splits
        self.split = split
        self.data = self.splits[self.split]

        self.n_full = n_full
        self.n_part = n_part
        self.transform = transform
        self.load_binvox = load_binvox

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        model_dir = self.data[index]

        # NOTE: either zip version or explicit ZipFile
        # version works. But neither version (currently) is able
        # to handle loading `model.mtl`.
        with open(F'{model_dir}/meshes/model.obj', 'r') as fp:
            mesh = trimesh.load_mesh(fp, file_type='obj')

        # NOTE: by default, center the mesh.
        mesh.apply_translation(-0.5 * (mesh.bounds[0] + mesh.bounds[1]))
        scale = 0.5 / max(mesh.bounds[1] - mesh.bounds[0])
        mesh.apply_scale(scale)

        # Construct output.
        out = {
            'mesh': mesh,
            # These are aliased to share memory,
            # so there's no harm in also returning these
            # for convenience' sake --> unless you're
            # doing stupid things like pickling...
            # 'verts': mesh.vertices,
            # 'faces': mesh.faces
        }

        # Optionally sample a point cloud on the surface of the mesh.
        if self.n_full > 0:
            # Sample full and partial point clouds.
            full, _ = trimesh.sample.sample_surface_even(
                mesh, _OVERSAMPLE_RATIO * self.n_full)

            ## Ensure number of sampled points
            ## match the output specification.
            if len(full) < self.n_full:
                indices = np.random.choice(
                    len(full), self.n_full, replace=True)
                full = full[indices]
            out['full'] = full

        if self.load_binvox:
            # FIXME:
            # Because we're loading a fixed-size voxelization, only
            # 64x64x64 voxelization is supported.
            binvox_file = (self.root / 'binvox_64'
                           / F'{model_dir.name}_64.binvox')
            with open(binvox_file, 'rb') as fp:
                voxel_grid = read_as_3d_array(fp)
            out['full/voxel'] = voxel_grid.data.astype(
                np.float32)
        if self.transform is not None:
            out = self.transform(out)
        return out


def main():
    dataset = GSODataset()
    for data in dataset:
        print(data['full'].shape)
        break


if __name__ == '__main__':
    main()
