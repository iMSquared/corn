#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from pkm.util.config import ConfigBase
from typing import List, Tuple, Dict, Union, Callable, Optional
from tqdm.auto import tqdm
from pathlib import Path
import trimesh

import json
import networkx as nx
import logging
import itertools
import open3d as o3d

import torch as th
import numpy as np
import einops


class ShapeNetOccDataset(th.utils.data.Dataset):
    """
    Preprocessed ShapeNet dataset with
    occupancy information for sampled points.
    See:
    * https://github.com/autonomousvision/occupancy_networks
    """
    @dataclass
    class Config(ConfigBase):
        data_dir: str = '/opt/datasets/ShapeNetOcc/'
        mesh_dir: str = '/opt/datasets/ShapeNetCore.v2/'

        # TODO: figure out why this is needed.
        unpack_bits: bool = True
        load_glb: bool = True

    def __init__(self,
                 cfg: Config,
                 split: str = 'train',
                 transform: Optional[Callable] = None):
        super().__init__()

        # Save input args as parameters, and initialize RNG.
        self.rng = np.random.default_rng(0)
        self.cfg = cfg
        self.transform = transform
        self.mesh_dir = Path(cfg.mesh_dir)

        # NOTE: convert between split identifier conventions.
        if split == 'valid':
            split_id = 'val'
        else:
            split_id = split

        # Load list of model files.
        self.data = []
        for d in sorted(Path(cfg.data_dir).iterdir()):
            files = d / F'{split_id}.lst'
            if not files.is_file():
                continue
            with open(files, 'r') as fp:
                models = fp.readlines()

            for m in models:
                m = m.strip()
                mesh_file = self._get_mesh_file(d / m)
                if not mesh_file.exists():
                    continue
                self.data.append(d / m)

        # Just for debugging: overfitting to a single object.
        # self.data = [self.data[0] for _ in range(len(self.data))]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        model_dir = self.data[index]

        mesh_file = self._get_mesh_file(model_dir)

        occs_file = model_dir / 'points.npz'
        occs_dict = np.load(occs_file)

        # points, normals, loc, scale
        cloud_file = model_dir / 'pointcloud.npz'
        cloud_dict = np.load(occs_file)

        # Load query points and deal with the strange conventions.
        query_points = occs_dict['points'].astype(np.float32)
        query_points = query_points[..., (2, 1, 0)]
        query_points[..., 2] = -query_points[..., 2]
        # query_points += center

        occ_labels = occs_dict['occupancies']
        if self.cfg.unpack_bits:
            occ_labels = np.unpackbits(occ_labels)[:query_points.shape[0]]
            occ_labels = occ_labels.astype(np.float32)

        mesh = trimesh.load_mesh(mesh_file,
                                 force='mesh',
                                 skip_texture=True,
                                 skip_materials=True)
        if isinstance(mesh, trimesh.Scene):
            if self.cfg.load_glb:
                # on GLB, we _know_ there's only one object.
                node_name = mesh.graph.nodes_geometry[0]
                (transform, geometry_name) = mesh.graph[node_name]
                mesh = mesh.geometry[geometry_name]
            else:
                # Else, we fallback to robust behavior.
                mesh = mesh.dump(concatenate=True)
        box = mesh.bounds
        center = 0.5 * (box[0] + box[1])
        # mesh.apply_translation(-center)

        xfm = np.eye(4)
        xfm[:, :3] = -center

        out = {
            # TODO: consider always returning filenames
            # from datasets, and adding a LoadMesh() type of transform.
            # 'mesh': trimesh.load_mesh(mesh_file),
            # 'mesh': mesh,
            'mesh_file' : mesh_file,

            # occupancy queries an labels.
            'query_points': cloud_dict['scale'] * query_points + cloud_dict['loc'],
            'occ_labels': occ_labels,
            'xfm' : xfm
            # we need to apply transforms to the mesh
            # in order to be "correct".
            # 'loc': cloud_dict['loc'],
            # 'scale': cloud_dict['scale']
        }
        if self.transform is not None:
            out = self.transform(out)
        return out

    def _get_mesh_file(self, model_dir: Path) -> Path:
        return (self.mesh_dir /
                model_dir.parent.name / model_dir.name
                / 'models' / 'model_normalized.glb')


def main():
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    logging.basicConfig(level=logging.ERROR)
    logging.root.setLevel(logging.ERROR)
    max_count: int = 4
    for split in ('train', 'valid'):
        dataset = ShapeNetOccDataset(ShapeNetOccDataset.Config(),
                                     split=split)
        loader = th.utils.data.DataLoader(dataset,
                                          batch_size=None,
                                          shuffle=True)
        print(F'split = {split} len dataset = {len(dataset)}')
        count: int = 0
        for data in loader:
            if count == 0:
                print(list(data.keys()))
            print(data['mesh'])

            print('queries')
            print(data['query_points'][data['occ_labels'] > 0].min(axis=0))
            print(data['query_points'][data['occ_labels'] > 0].max(axis=0))
            # m = o3d.io.read_triangle_mesh(str(data['mesh']))
            m = o3d.geometry.TriangleMesh()
            m.vertices = o3d.utility.Vector3dVector(data['mesh'].vertices)
            m.triangles = o3d.utility.Vector3iVector(data['mesh'].faces)
            print(m.get_axis_aligned_bounding_box())
            #m = m.translate(-m.get_axis_aligned_bounding_box().get_center(),
            #                relative=True)

            # try exporting...
            name = F'{split}-{count}'

            c = o3d.geometry.PointCloud()
            c.points = o3d.utility.Vector3dVector(
                data['query_points'][data['occ_labels'] > 0])
            o3d.io.write_triangle_mesh(F'/tmp/hmm/{name}-mesh.obj', m)
            o3d.io.write_point_cloud(F'/tmp/hmm/{name}-points.ply', c)

            count += 1
            if count >= max_count:
                break


if __name__ == '__main__':
    main()
