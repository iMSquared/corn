#!/usr/bin/env python3

import isaacgym

from dataclasses import dataclass
from tqdm.auto import tqdm
import torch as th
import open3d as o3d

from pkm.train.ckpt import (last_ckpt, load_ckpt, save_ckpt,
                            step_from_ckpt)
from pkm.util.torch_util import dcn
from pkm.util.vis.win_o3d import o3d_cloud_from_cloud
from pkm.util.config import recursive_replace_map

from train_dyn import (
    Config as TrainConfig,
    load_dataset
)
from model_dyn import DynModel


@dataclass
class Config(TrainConfig):
    # load_ckpt = '/tmp/corn-/dyn/run-037/ckpt/epoch-020.ckpt'
    # load_ckpt = '/tmp/corn-/dyn/run-040/ckpt/epoch-020.ckpt'
    load_ckpt: str = '/tmp/corn-/dyn/run-043/ckpt/epoch-092.ckpt'
    batch_size: int = 1


def main(cfg: Config):
    device = cfg.device
    model_cfg = recursive_replace_map(cfg.model, {
        'mask_ratio': 0.0,
        'encoder.p_drop': 0.0,
    })
    model = DynModel(model_cfg).to(device=device)

    aux = {}
    train_loader, valid_loader = load_dataset(cfg, aux=aux)
    # train_xfm = aux['train_transform']
    valid_xfm = aux['valid_transform']

    if cfg.load_ckpt is not None:
        print('-load-ckpt-')
        ckpt: str = last_ckpt(cfg.load_ckpt,
                              key=step_from_ckpt)
        load_ckpt(dict(model=model), ckpt, strict=True)
        # raise ValueError('stop')

    # model.eval()
    with th.inference_mode():
        for data in tqdm(valid_loader, desc='batch', leave=False):
            data = {k: v.to(cfg.device) for (k, v) in data.items()
                    if (v is not None)}
            aux = {}
            _ = model(data['cloud0'],
                      data['cloud1'],
                      data['hand0'],
                      data['hand1'],
                      aux=aux)

            pred_pcd1 = aux['pred_x1'].reshape(data['cloud1'].shape)

            # ===================================================================
            vis_pcd0 = valid_xfm.unnormalize(
                {'cloud0': data['cloud0']})['cloud0']
            vis_pcd1 = valid_xfm.unnormalize(
                {'cloud1': data['cloud1']})['cloud1']
            vis_pcd1_from_0 = valid_xfm.unnormalize(
                {'cloud1': pred_pcd1})['cloud1']

            table = o3d.geometry.TriangleMesh.create_box(0.4, 0.5, 0.4)
            table = table.translate([-0.2, -0.25, -0.2])
            table = table.translate([0, 0, +0.2])
            table_ls = o3d.geometry.LineSet.create_from_triangle_mesh(table)

            for c0, c1, c1_from_c0 in zip(
                    dcn(vis_pcd0),
                    dcn(vis_pcd1),
                    dcn(vis_pcd1_from_0)):
                o3d.visualization.draw([
                    o3d_cloud_from_cloud(c0, color=(1, 0, 0)),
                    o3d_cloud_from_cloud(c1, color=(0, 0, 1)),
                    o3d_cloud_from_cloud(c1_from_c0, color=(0, 1, 0)),
                    table_ls
                ])
            # break


if __name__ == '__main__':
    main(Config())
