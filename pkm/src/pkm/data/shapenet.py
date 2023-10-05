#!/usr/bin/env python3

from dataclasses import dataclass
from pkm.util.config import ConfigBase
from typing import List, Tuple, Dict, Union, Callable, Optional
from tqdm.auto import tqdm
from pathlib import Path
import json
import networkx as nx
import logging
import itertools
import open3d as o3d
import trimesh

import torch as th
import numpy as np
import einops

from pkm.data.util import (
    split_files,
    bbox_nd
)
from pkm.data.binvox_rw import (
    read_as_3d_array
)
from pkm.data.transforms.io_xfm import load_mesh


def _build_list(data_dir: str, pattern: str = '') -> Tuple[str, ...]:
    """build list of files matching a given pattern."""
    path = Path(data_dir).expanduser()
    out = tuple(tqdm(map(str, path.glob(pattern))))
    return out


def _build_class_map(filename: str):
    """Integer - valued Index mappings for synset tree structure."""
    # Parse class map taxonomy.
    with open(filename, 'r') as fp:
        taxonomy = json.load(fp)

    # Build union-find structure.
    u = nx.utils.UnionFind()
    for entry in taxonomy:
        # NOTE: self-union requires for synsets without children
        u.union(entry['synsetId'])
        for child in entry['children']:
            u.union(entry['synsetId'], child)

    # Format to dictionary mapping.
    class_map = {}
    num_class = 0
    for i, ss in enumerate(u.to_sets()):
        for s in ss:
            class_map[s] = i
        num_class += 1

    logging.debug(F'Coarse classes : {len(taxonomy)} -> {num_class}')
    return (class_map, num_class)


class ShapeNetDataset(th.utils.data.Dataset):
    @dataclass
    class Config(ConfigBase):
        data_dir: str = '/input/ShapeNetCore.v2'
        pattern: str = '*/*/models/model_normalized.obj'
        taxonomy_file: str = 'taxonomy.json'

    def __init__(self,
                 cfg: Config,
                 split: str = 'train',
                 split_ratio: float = 0.8,
                 load_binvox: bool = True,
                 transform: Optional[Callable] = None):
        super().__init__()
        self.rng = np.random.default_rng(0)
        self.cfg = cfg
        self.load_binvox = load_binvox
        self.transform = transform

        # Load list of model files.
        self.files = _build_list(cfg.data_dir, cfg.pattern)
        if self.load_binvox:
            self.files = [f for f in self.files
                          if Path(f).with_suffix('.surface.binvox').is_file()]
        self.files = np.asarray(self.files, dtype=str)

        # TODO: splits should be cached.
        splits = {}
        splits['train'], splits['valid'] = split_files(
            self.files, self.rng, split_ratio)

        self.splits = splits
        self.split = split
        self.data = self.splits[self.split]

        # Build class map taxonomy.
        m, n = _build_class_map(str(Path(cfg.data_dir) / cfg.taxonomy_file))

        self.class_map = m
        self.num_class = n

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Union[int, th.Tensor]):
        # FIXME: Is this a meaningful thing to do??
        if isinstance(index, th.Tensor):
            index: List[int] = index.tolist()
        entry = self.data[index]

        synset_id = Path(entry).parent.parent.parent.name
        cls_index = self.class_map[synset_id]
        out = {
            'filename': entry,
            'category': cls_index
        }
        if self.transform is not None:
            out = self.transform(out)
        return out

        # TODO: Does the following case need to be handled??
        # try:
        #    return [read_obj(x) for x in entry]
        # except TypeError:
        #    return read_obj(entry)

        # if use_pytorch_geometric:
        # verts, faces, aux = load_obj(entry, load_textures=False)
        # faces = faces.verts_idx
        # out = dict(pos=verts, face=faces, label=cls_index)
        verts, faces = load_mesh(entry)

        # Isotropic normalization so that the largest axis scales to 1.
        # TODO: perhaps need to be done as a transform,
        # rather than internally inside the dataset.
        p0 = np.min(verts, axis=0, keepdims=True)
        p1 = np.max(verts, axis=0, keepdims=True)
        center = 0.5 * (p0 + p1)
        scale = 0.5 * np.max(p1 - p0)
        verts -= (center / scale)

        out = {
            'verts': verts,
            'faces': faces
        }

        if self.load_binvox:
            binvox_file = Path(entry).with_suffix(
                '.surface.binvox')
            with open(str(binvox_file), 'rb') as fp:
                voxel_grid = read_as_3d_array(fp)
            voxels = voxel_grid.data.astype(np.float32)

            # (1) ShapeNet .binvox files are by default
            # aligned to the minimum corner. We should probably
            # center this, for consistency's sake.
            try:
                bbox = bbox_nd(voxels)  # first and last set elements
                bbox = np.asanyarray(bbox)
                shift = (np.subtract(voxels.shape,
                                     1) - bbox[:, 1] - bbox[:, 0] + 1) // 2
                voxels = np.roll(voxels, shift, range(len(shift)))
            except IndexError:
                # TODO: why would this ever be encountered?
                # when this happens, the _correct_ thing to do is
                # check what the hell is happening. :)
                pass
            # for axis, (i0, i1) in enumerate(bbox):
            #    print(axis, (i0, i1), voxels.shape[axis])
            #    np.roll(voxels, shift, axis

            # FIXME: temporary measure to
            # enforce 128x128x128 --> 64x64x64
            voxels = einops.reduce(voxels,
                                   '(x kx) (y ky) (z kz) -> x y z', 'max',
                                   kx=2, ky=2, kz=2)
            out['full/voxel'] = voxels

        if self.transform is not None:
            out = self.transform(out)
        return out


def main():
    # NOTE: only imported when used in __main__
    from cho_util.app.with_cli import with_cli

    def _main(cfg: ShapeNetDataset.Config):
        logging.basicConfig(level='DEBUG')
        dataset = ShapeNetDataset(cfg)
        print(len(dataset))
        out = dataset[0]
        print(out['filename'])
        print(out['category'])  # NUM_VERT X 3
        # print(out['face'].shape)  # NUM_FACE X 3
    return with_cli(_main)()


if __name__ == '__main__':
    main()
