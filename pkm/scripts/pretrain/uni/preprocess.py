#!/usr/bin/env python3

import os
import pickle
from tqdm.auto import tqdm
from typing import Dict, Tuple, Optional, Iterable, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import math

import numpy as np
import torch as th
import torch.utils.data
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

from cho_util.math import transform as tx
from pkm.util.math_util import (
    quat_rotate,
    quat_multiply,
    quat_inverse,
    axa_from_quat,
    apply_pose_tq,
    invert_pose_tq,
    compose_pose_tq,
    random_yaw_quaternion,
    matrix_from_quaternion,
    random_quat
)

from pkm.util.torch_util import dcn
from pkm.models.common import merge_shapes
from pkm.models.cloud.point_mae import (
    subsample
)

from icecream import ic


def _to_torch(x):
    if isinstance(x, np.ndarray):
        return th.as_tensor(x)
    return x


class Normalize:
    def __init__(self, mean, std, eps: float = 1e-6,
                 device=None):
        self.mean = th.as_tensor(mean, dtype=th.float,
                                 device=device)
        self.std = th.as_tensor(std, dtype=th.float, device=device)
        self.istd = th.reciprocal(self.std + eps)

    def __call__(self, x: th.Tensor):
        return (x - self.mean) * self.istd

    def unnormalize(self, x: th.Tensor):
        return x * self.std + self.mean


# SCALAR = ('collide', 'touch')
# POINT = ('cloud', 'aff_pos')
# POSE = ('init', 'goal', 'hand')
# DIRECTION = ('aff_vec',)
# AUG_RXN = (POINT + POSE + DIRECTION)
# AUG_TXN = (POINT + POSE)


class SliceHelper:
    def __getitem__(self, s):
        return s


_S = SliceHelper()


AUG_TXN = {
    'collide': (None, None),
    'touch': (None, None),
    'contact_flag': (None, None),
    'empty_contact_flag': (None, None),

    'cloud': (_S[..., None, :3], _S[..., :3]),
    'object_cloud': (_S[..., None, :3], _S[..., :3]),
    'empty_cloud': (_S[..., None, :3], _S[..., :3]),

    'aff_pos': (_S[..., :3], _S[..., :3]),
    'init': (_S[..., :3], _S[..., :3]),
    'goal': (_S[..., :3], _S[..., :3]),
    'hand': (_S[..., :3], _S[..., :3]),
    'hand_pose': (_S[..., :3], _S[..., :3]),
    'object_pose': (_S[..., :3], _S[..., :3]),
    'aff_vec': (None, None),

    'hand0': (_S[..., :3], _S[..., :3]),
    'hand1': (_S[..., :3], _S[..., :3]),
    'cloud0': (_S[..., None, :3], _S[..., :3]),
    'cloud1': (_S[..., None, :3], _S[..., :3]),
}

AUG_RXN = {
    # source slice, rotate slice, compose slice
    'collide': (None, None, None),
    'touch': (None, None, None),
    'contact_flag': (None, None, None),
    'empty_contact_flag': (None, None, None),

    'cloud': (_S[..., None, :4], _S[..., :3], None),
    'object_cloud': (_S[..., None, :4], _S[..., :3], None),
    'empty_cloud': (_S[..., None, :4], _S[..., :3], None),

    'aff_pos': (_S[..., :4], _S[..., :3], None),
    'init': (_S[..., :4], _S[..., :3], _S[..., 3:7]),
    'goal': (_S[..., :4], _S[..., :3], _S[..., 3:7]),
    'hand': (_S[..., :4], _S[..., :3], _S[..., 3:7]),
    'hand_pose': (_S[..., :4], _S[..., :3], _S[..., 3:7]),
    'object_pose': (_S[..., :4], _S[..., :3], _S[..., 3:7]),
    'aff_vec': (_S[..., :4], _S[..., :3], None),

    'hand0': (_S[..., :4], _S[..., :3], _S[..., 3:7]),
    'hand1': (_S[..., :4], _S[..., :3], _S[..., 3:7]),
    'cloud0': (_S[..., None, :4], _S[..., :3], None),
    'cloud1': (_S[..., None, :4], _S[..., :3], None)
}

NORMALIZE_STAT = {
    "pose": {
        "mean": [0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0],
        "stddev": [0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0]
    },
    "pose_6drot": {
        "mean": [0.0, 0.0, 0.55,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "stddev": [0.4, 0.4, 0.4,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    },

    # NOTE:
    # `goal` is _relative_,
    # so subtracting by z_mean=0.55
    # is not necessary. Thus, we set it to zero here.
    "goal": {
        "mean": [0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0],
        "stddev": [0.4, 0.4, 0.4,
                   1.0, 1.0, 1.0, 1.0]
    },
    "goal_6drot": {
        "mean": [0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "stddev": [0.4, 0.4, 0.4,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    },

    "point": {
        "mean": [0.0, 0.0, 0.55],
        "stddev": [0.4, 0.4, 0.4]
    },
    "normal": {
        "mean": [0.0, 0.0, 0.0],
        "stddev": [0.4, 0.4, 0.4]
    },
    "object_state": {
        "mean": [0.0, 0.0, 0.55],
        "stddev": [0.4, 0.4, 0.4]
    }
}

NORMALIZE_TYPE = {
    'collide': None,
    'touch': None,
    'contact_flag': None,
    'empty_contact_flag': None,

    'cloud': 'point',
    'object_cloud': 'point',
    'empty_cloud': 'point',
    'aff_pos': 'point',
    # `goal` here is no longer the
    # absolute pose (it's the relative pose
    # between `init` and `goal`)
    'goal': 'goal',
    'hand': 'pose',
    'aff_vec': 'normal',

    'hand0': 'pose',
    'hand1': 'pose',
    'hand_pose': 'pose',
    'hand_state': 'pose',
    'object_pose': 'pose',
    'cloud0': 'point',
    'cloud1': 'point'
}


class Preprocess:
    @dataclass
    class Config:
        cloud_size: int = 512
        rotate: bool = False
        rotate_2d: bool = True
        translate: float = 0.0
        translate_2d: bool = True
        noise_scale: float = 0.0
        use_6d_rot: bool = True
        rotate_from_center:bool = True

        normalize_stat: Optional[Dict[str, Dict[str, List[float]]]] = field(
            default_factory=lambda: NORMALIZE_STAT)
        normalize_type: Optional[Dict[str, Optional[str]]] = field(
            default_factory=lambda: NORMALIZE_TYPE)

        load_collision: bool = True
        # unused, should be true though i guess
        post_normalize_goal: bool = True

        ref_cloud_keys: Tuple[str, ...] = ('cloud',)
        cloud_keys: Tuple[List[str], ...] = ((
            'cloud', 'object_cloud', 'cloud0', 'cloud1', 'contact_flag'),)
        noise_keys: Tuple[str, ...] = ('cloud',)

    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = device

        # Load normalization statistics.
        self.normalize = None
        if cfg.normalize_stat is not None:
            self.normalize = {
                k: Normalize(cfg.normalize_stat[k]["mean"],
                             cfg.normalize_stat[k]["stddev"],
                             device=self.device)
                for k in cfg.normalize_stat.keys()
            }

    def __rotate(self, data):
        cfg = self.cfg
        cloud = data[cfg.ref_cloud_keys[0]]

        if cfg.rotate_2d:
            aug_qxn = random_yaw_quaternion(cloud.shape[:-2],
                                            dtype=cloud.dtype,
                                            device=cloud.device)
        else:
            # raise ValueError('rotate_2d must be true for now')
            aug_qxn = random_quat(cloud.shape[:-2],
                                  dtype=cloud.dtype,
                                  device=cloud.device)

        if cfg.rotate_from_center:
            rotation_center = cloud.mean(dim=-2)

        for k, (src, dst_t, dst_r) in AUG_RXN.items():
            if k not in data:
                continue
            if dst_t is not None:
                if cfg.rotate_from_center:
                    data[k][dst_t] = (
                            quat_rotate(aug_qxn[src], data[k][dst_t] - rotation_center)
                            + rotation_center)
                else:
                    data[k][dst_t] = quat_rotate(aug_qxn[src],
                                                 data[k][dst_t])
            if dst_r is not None:
                data[k][dst_r] = quat_multiply(aug_qxn[src],
                                               data[k][dst_r])
        return aug_qxn

    def __translate(self, data):
        cfg = self.cfg
        cloud = data[cfg.ref_cloud_keys[0]]
        aug_txn = cfg.translate * th.randn(
            merge_shapes(cloud.shape[:-2], 3),
            dtype=cloud.dtype,
            device=cloud.device)
        if cfg.translate_2d:
            aug_txn[..., 2] = 0.0
        for k, (src, dst) in AUG_TXN.items():
            if k not in data:
                continue
            if src is None:
                continue
            data[k][dst] = data[k][dst] + aug_txn[src]
        return aug_txn

    def __call__(self, data: Dict[str, th.Tensor],
                 aux: Optional[Dict[str, th.Tensor]] = None):
        if aux is None:
            aux = {}
        cfg = self.cfg

        # Make a copy to avoid accidental overwrites.
        data = dict(data)
        # Move to target device.
        data = {k: _to_torch(v) for k, v in data.items()}
        data = {k: v.to(self.device) for k, v in data.items()}

        for (ref_cloud_key, cloud_keys) in zip(
                cfg.ref_cloud_keys,
                cfg.cloud_keys):

            if cfg.cloud_size != data[ref_cloud_key].shape[-2]:
                # First subsample the reference cloud...
                data[ref_cloud_key] = subsample(
                    data[ref_cloud_key], cfg.cloud_size, aux=aux)

                # Then subsample in the same order as `cloud`.
                index = aux['index']
                for k in cloud_keys:

                    if k not in data:
                        # skip missing
                        continue

                    if k == ref_cloud_key:
                        # skip already subsampled
                        continue

                    if len(index.shape) == len(data[k].shape):
                        # scalars
                        data[k] = th.take_along_dim(data[k],
                                                    index,
                                                    -1)
                    else:
                        # vectors
                        print(k, data[k].shape, index.shape)
                        data[k] = th.take_along_dim(data[k],
                                                    index[..., None],
                                                    -2)

        # Rotate
        if cfg.rotate:
            aug_qxn = self.__rotate(data)
            aux['quat'] = aug_qxn

        # Translate
        if cfg.translate > 0:
            aug_txn = self.__translate(data)

        # TODO: Scale?

        # Convert goal to relative pose
        # data['goal'] = compose_pose_tq(
        #     data.pop('goal'),
        #     invert_pose_tq(data.pop('init'))
        # )
        if 'init' in data and 'goal' in data:
            init_pose = data.pop('init')
            goal_pose = data.pop('goal')
            data['goal'] = th.cat([
                # T1 - T0
                goal_pose[..., :3] - init_pose[..., :3],
                # Q1 @ Q0^{-1}
                quat_multiply(goal_pose[..., 3:7],
                              quat_inverse(init_pose[..., 3:7]))
            ], dim=-1)

        # 6D ROT CONVERSION
        if cfg.use_6d_rot:
            for k, v in cfg.normalize_type.items():
                if v not in ('pose', 'pose_6drot'):
                    continue
                if k not in data:
                    continue
                data[k] = th.cat([
                    data[k][..., :3],
                    matrix_from_quaternion(data[k][..., 3:7])[..., :, :2].reshape(
                        *data[k].shape[:-1], 6)
                ], dim=-1)

        # Normalize
        if self.normalize is not None:
            def _map_kv(k, v):
                # print('===')
                # print(k, v.shape)
                # print(cfg.normalize_type[k])
                # print(self.normalize[cfg.normalize_type[k]].mean.shape)
                # print(self.normalize[cfg.normalize_type[k]].istd.shape)
                return (self.normalize[cfg.normalize_type[k]](v)
                        if cfg.normalize_type[k] is not None
                        else v)
            data = {k: _map_kv(k, v) for k, v in data.items()}

        # Add noise
        if cfg.noise_scale > 0:
            for key in cfg.noise_keys:
                if key not in data:
                    continue
                noise = (cfg.noise_scale *
                         th.randn_like(data[key]))
                data[key] = data[key] + noise

        return data

    def unnormalize(self, data):
        cfg = self.cfg
        if self.normalize is None:
            return data
        data = {self.normalize[cfg.normalize_type[k]].unnormalize(v)
                for k, v in data.items()}
        return data


def main():
    from dataset_v2 import ContactDataset
    dataset = ContactDataset(
        ContactDataset.Config(
            '/tmp/aff-w-col-toolframe-fixed',
            max_count=100,
            device='cpu'))
    with open('./normalize.json', 'r') as fp:
        normalize = json.load(fp)
    preprocess = Preprocess(Preprocess.Config(
        normalize=normalize),
        device='cpu')

    for data in dataset:
        data = preprocess(data)
        print(data.keys())
        break


if __name__ == '__main__':
    main()
