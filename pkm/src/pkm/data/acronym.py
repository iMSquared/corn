#!/usr/bin/env python3

import json
import h5py
import time
from dataclasses import dataclass
from pkm.util.config import ConfigBase
from typing import Iterable, Optional, Union, Dict
from pathlib import Path
import numpy as np
import torch as th
from pkm.data.util import (
    split_files,
)
from pkm.data.transforms.common import weights_to_splits
from pkm.data.transforms.common import SelectKeys
from cho_util.math import transform as tx
from tqdm.auto import tqdm


def _load_mesh(filename: str, mesh_root_dir: str,
               scale: Optional[float] = None):
    """
    [from NVlabs/acronym/acronym_tools/acronym.py]
    Load a mesh from a JSON or HDF5 file from the grasp dataset.
    The mesh will be scaled accordingly.
    Args:
        filename (str): JSON or HDF5 file name.
        scale (float, optional):
            If specified, use this as scale
            instead of value from the file.
            Defaults to None.
    Returns:
        Dictionary with the following entries:
            mesh_file:  mesh filename.
            mesh_scale: object scale.
    """
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        mesh_fname = data["object"].decode('utf-8')
        mesh_scale = data["object_scale"] if scale is None else scale
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()] if scale is None else scale
    else:
        raise RuntimeError("Unknown file ending:", filename)

    mesh_name = Path(mesh_fname).with_suffix('.obj').name
    return {
        'mesh_file': str(Path(mesh_root_dir) / mesh_name),
        'mesh_scale': mesh_scale
    }


def _load_grasps(filename: str):
    """
    [from NVlabs/acronym/acronym_tools/acronym.py]
    Load transformations and qualities of
    grasps from a JSON file from the dataset.
    Args:
        filename (str): HDF5 or JSON file name.
    Returns:
        np.ndarray: Homogenous matrices describing the grasp poses.
            2000 x 4 x 4.
        np.ndarray: List of binary values indicating
            grasp success in simulation.
    """
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        T = np.array(data["transforms"])
        success = np.array(data["quality_flex_object_in_gripper"])
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        T = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    else:
        raise RuntimeError("Unknown file ending:", filename)
    return {
        'grasp_pose': T,
        'grasp_suc': success.astype(bool),
        # temporary
        # 'friction': data['object/friction']
        # temporary
        'com': data['object/com'],
    }


class SelectPositive:
    def __call__(self, grasp: np.ndarray, grasp_suc: np.ndarray):
        return grasp[grasp_suc]


class ToGraspNet:
    def __call__(self, grasp: np.ndarray):
        origin = grasp[..., :3, 3]
        orientation = grasp[..., :3, :3]
        center = origin + orientation @ [0, 0, Acronym.OFFSET]
        orientation = (orientation @ tx.rotation.matrix.from_euler(
            [0.0, -np.pi / 2, np.pi / 2]))
        width: float = np.full(origin.shape[:-1], Acronym.WIDTH)
        depth: float = np.full(origin.shape[:-1], Acronym.DEPTH)
        # quat = tx.rotation.quaternion.from_matrix(orientatino)

        return (center, orientation, width, depth)


def split_labels_by_object(
        label_files: Iterable[str],
        rng: np.random.Generator,
        split_ratio: float = 0.8):
    # Group by objects
    objects = {}
    for f in label_files:
        key: str = Path(f).stem.rsplit('_', maxsplit=1)[0]
        if key not in objects:
            objects[key] = []
        objects[key].append(f)

    # Split keys
    num_train, num_valid = weights_to_splits(
        len(objects), (split_ratio, 1.0 - split_ratio))
    keys = list(objects.keys())
    rng.shuffle(keys)
    train_keys, valid_keys = np.array_split(keys,
                                            [num_train])

    # Dump to respective sets
    train_set = []
    for k in train_keys:
        train_set.extend(objects[k])
    valid_set = []
    for k in valid_keys:
        valid_set.extend(objects[k])

    # Output
    return (train_set, valid_set)


class Acronym(th.utils.data.Dataset):
    # distance from gripper origin to grasp point (finger)
    OFFSET: float = 0.11217
    # distance from gripper origin to clearance
    BASE: float = 0.066
    # DEPTH: float = (OFFSET - BASE)
    DEPTH: float = 0.017
    # width of the gripper
    WIDTH: float = 0.080

    @dataclass
    class Config(ConfigBase):
        data_dir: str = '/opt/datasets/acronym/grasps'
        mesh_dir: str = '/opt/datasets/ShapeNetSem/models/'
        cloud_dir: Optional[str] = '/opt/datasets/ShapeNetSem/cloud/'

        # NOTE:
        # rng for determining train-valid splits.
        seed: int = 0
        shuffle: bool = False
        split_ratio: float = 0.8

    def __init__(self, cfg: Config, split: str = 'train',
                 transform=None):
        super().__init__()

        self.root = Path(cfg.data_dir)
        self.rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg

        # QQ = 'Desktop_892af756b846e5f7397d0d738e1149_0.0076676976103658215.h5'
        # QQ = 'Refrigerator_4936e33b44153cd9d931a373c2cb01f_0.002237104609143189.h5'
        # all_files = list(sorted(list(self.root.glob(QQ))))
        all_files = list(sorted(list(self.root.glob('*.h5'))))

        # Filter by available mesh files.
        filtered_files = []
        for filename in all_files:
            data = h5py.File(filename, "r")
            mesh_base = data["object/file"][()].decode('utf-8')
            mesh_name = Path(mesh_base).with_suffix('.obj').name
            mesh_file = Path(self.cfg.mesh_dir) / mesh_name
            if not mesh_file.exists():
                continue

            # NOTE:
            # also filter by availability of the point cloud.
            if cfg.cloud_dir is not None:
                cloud_name = Path(mesh_name).with_suffix('.npz').name
                cloud_file = Path(cfg.cloud_dir) / cloud_name
                if not cloud_file.exists():
                    continue
            filtered_files.append(filename)

        # Is it reasonable to shuffle?
        if cfg.shuffle:
            self.rng.shuffle(filtered_files)

        # Internally determine the train-valid split.
        # TODO: try to avoid duplicate meshes?
        splits = {}
        splits['train'], splits['valid'] = split_labels_by_object(
            filtered_files, self.rng, cfg.split_ratio)
        self.splits = splits
        self.split = split
        self.files = splits[split]

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        """
        Returns:
            Dictionary with the following entries:
                mesh_file: absolute path of the mesh filename.
                mesh_scale: (float) mesh scaling factor on which grasps were computed.
                grasp_pose: Nx4x4 homogeneous matrix of grasp pose.
                grasp_suc: grasp success label.
        """
        f = self.files[index]
        mesh = _load_mesh(str(f), self.cfg.mesh_dir)
        grasp = _load_grasps(str(f))
        out = {**mesh, **grasp, 'label_file': str(f)}
        if self.transform is not None:
            out = self.transform(out)
        return out


def print_frictions():
    dataset = Acronym(
        Acronym.Config(),
        'valid')
    for data in dataset:
        print(data['friction'][()])


def count_grasps():
    transform = SelectKeys(['grasp_suc'])
    dataset = Acronym(
        Acronym.Config(),
        'train',
        transform=transform)
    loader = th.utils.data.DataLoader(
        dataset,
        shuffle=False,
        num_workers=8,
        batch_size=None
    )
    counts: List[int] = []
    for data in tqdm(loader):
        counts.append(data['grasp_suc'].shape[0])
    print(len(counts), np.mean(counts))  # mean = 2000


def check_duplicates_between_splits():
    transform = SelectKeys(['label_file'])
    train_set = Path('/opt/datasets/acronym/processed2/train').glob('*.npz')
    train_set = set([f.stem.rsplit('_', maxsplit=1)[0] for f in train_set])
    valid_set = Path('/opt/datasets/acronym/processed2/valid').glob('*.npz')
    valid_set = set([f.stem.rsplit('_', maxsplit=1)[0] for f in valid_set])
    print(len(train_set))
    print(len(valid_set))
    print(len(train_set.intersection(valid_set)))  # => 300


def main():
    from pkm.data.transforms.common import WrapDict
    from pkm.data.transforms.io_xfm import LoadMesh
    import open3d as o3d
    import open3d.visualization.gui as gui
    from pkm.util.vis import Window, gripper_mesh, gripper_mesh_acronym

    max_show: int = 1
    dataset = Acronym(
        Acronym.Config(),
        'valid',
        transform=WrapDict(
            LoadMesh(),
            'mesh_file',
            'mesh'))

    if True:
        # Setup visualizer.
        app = gui.Application.instance
        app.initialize()
        vis = Window()

        # Setup key callback.
        state = {'next': False}

        def on_key(key) -> bool:
            if key == gui.KeyName.SPACE:
                state['next'] = True
                return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED
        vis.set_on_key(on_key)

    for data in dataset:
        setup_camera: bool = True
        verts, faces = data['mesh']
        scale = data['mesh_scale']
        print(scale)
        verts *= scale  # will this work?
        grasp = data['grasp_pose']
        grasp = SelectPositive()(grasp, data['grasp_suc'])

        # print('bbox')
        # print(verts.min(axis=0))
        # print(verts.max(axis=0))

        vis.widget.scene.clear_geometry()

        # Create object trimesh geometry.
        # obj_mesh = o3d.geometry.TriangleMesh()
        # obj_mesh.vertices = o3d.utility.Vector3dVector(verts)
        # obj_mesh.triangles = o3d.utility.Vector3iVector(faces)
        obj_mesh = o3d.io.read_triangle_mesh(data['mesh_file'])
        obj_mesh.scale(scale, center=(0, 0, 0))
        vis.add_geometry('obj', obj_mesh,
                         setup_camera=setup_camera,
                         shader='defaultUnlit')

        bbox = obj_mesh.get_axis_aligned_bounding_box()
        vis.add_geometry('bbox', bbox, setup_camera=False,
                         shader='defaultUnlit')

        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(obj_mesh)
        vis.add_geometry(F'obj-wire', line_set,
                         color=(0, 0, 0, 1.0))
        setup_camera = False
        print(F'num grasps = {grasp.shape[0]}')

        gripper_center = grasp[..., :3, 3]
        gripper_orientation = grasp[..., :3, :3]

        setup_camera = False
        for i in range(min(len(grasp), max_show)):
            print(gripper_center[i], gripper_orientation[i])
            if True:
                vis_mesh = gripper_mesh(
                    # gripper center convention is different from graspnet.
                    gripper_center[i] + gripper_orientation[i] @ [0,
                                                                  0, Acronym.OFFSET],

                    gripper_orientation[i] @ tx.rotation.matrix.from_euler(
                        # [np.pi / 2, -np.pi / 2, 0]
                        [0, -np.pi / 2, np.pi / 2]),
                    width=Acronym.WIDTH,
                    depth=Acronym.DEPTH,
                    depth_base=Acronym.BASE,
                    offset=Acronym.OFFSET
                )
                vis.add_geometry(F'grasp-{i:02d}', vis_mesh,
                                 color=(0, 0, 1, 0.5),
                                 setup_camera=setup_camera,
                                 # shader='defaultUnlit'
                                 )

            if True:
                vm = gripper_mesh_acronym()
                transform = np.eye(4)
                scale = np.linalg.norm(gripper_orientation[i], axis=0)
                transform[:3, :3] = gripper_orientation[i]
                transform[:3, 3] = gripper_center[i]
                vm.apply_transform(transform)
                # vm.apply_rotation(gripper_orientation[i])
                # vm.apply_translation(gripper_center[i])

                vis_mesh = o3d.geometry.TriangleMesh()
                vis_mesh.vertices = o3d.utility.Vector3dVector(
                    np.asarray(vm.vertices))
                vis_mesh.triangles = o3d.utility.Vector3iVector(
                    np.asarray(vm.faces))

                vis.add_geometry(F'2grasp-{i:02d}', vis_mesh,
                                 color=(0, 1, 0, 0.5),
                                 setup_camera=setup_camera,
                                 # shader='defaultUnlit'
                                 )
            line_set = o3d.geometry.LineSet.create_from_triangle_mesh(vis_mesh)
            vis.add_geometry(F'grasp-{i:02d}-wire', line_set,
                             color=(0, 0, 0, 1.0))

        while not state['next']:
            gui.Application.instance.run_one_tick()
            time.sleep(2.0 / 128)
        state['next'] = False
        # break


if __name__ == '__main__':
    # print_frictions()
    # count_grasps()
    check_duplicates_between_splits()
