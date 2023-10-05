#!/usr/bin/env python3
import isaacgym

from typing import Optional
from dataclasses import dataclass, replace
from omegaconf import OmegaConf
from functools import partial
from collections import Mapping

from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation
import open3d as o3d
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from pkm.models.common import grad_step
from pkm.train.wandb import with_wandb, WandbConfig
from pkm.train.hf_hub import (upload_ckpt, HfConfig)
from pkm.util.hydra_cli import hydra_cli
from pkm.util.config import ConfigBase, recursive_replace_map as rrm
from pkm.train.ckpt import (last_ckpt, load_ckpt, save_ckpt,
                            step_from_ckpt)
from pkm.data.cached_web_dataset import (wds, as_wds)
from pkm.util.torch_util import dcn
from pkm.util.vis.win_o3d import o3d_cloud_from_cloud
from pkm.env.util import set_seed
from pkm.util.path import RunPath
from pkm.util.math_util import matrix_from_quaternion

from data_dyn import DynDataset
from model_dyn import DynModel
from preprocess import Preprocess

from icecream import ic


@dataclass
class Config(WandbConfig, HfConfig, ConfigBase):

    # == wandb ==
    project: Optional[str] = 'corn-/dyn'
    sync_tensorboard: bool = True
    save_code: bool = True
    use_wandb: bool = True
    # It's technically "optional", but required if
    # `use_wandb`=True.
    group: Optional[str] = None
    name: Optional[str] = None
    seed: int = 0

    # == huggingface ==
    hf_repo_id: str = F'corn/{project}'
    use_hfhub: bool = True

    path: RunPath.Config = RunPath.Config(root=F'/tmp/{project}')
    process: Preprocess.Config = Preprocess.Config(
        ref_cloud_keys=['cloud0'],
        cloud_keys=[
            ['cloud0', 'cloud1'],
        ],
        noise_keys=[
            'cloud0',
            # `cloud1` should be clean
            # 'cloud1'
        ],
        rotate=True,
        translate=0.1,
        noise_scale=0.01,
        translate_2d=True,
        use_6d_rot=True
    )
    dataset: DynDataset.Config = DynDataset.Config(
        data_root='/tmp/dyn-v12',
        # FIXME: ONLY FOR TESTING
        # max_count=128,
    )

    use_wds: bool = True
    wds_cache: str = '/home/user/.cache/pkm/dyn-v12/'

    model: DynModel.Config = rrm(
        DynModel.Config(),
        {
            # == dynmodel ==
            'tau': 0.001,
            # 'mask_ratio': 0.25,
            'mask_ratio': 0.75,
            'average_targets': 2,
            'rec_coef': 0.1,
            'dyn_coef': 1.0,

            # == ICPNet canonical defaults ==
            # 'encoder.p_drop': 0.05,
            'encoder.p_drop': 0.01,

            # 'encoder.encoder.num_hidden_layers': 2,
            'encoder.encoder.num_hidden_layers': 4,
            'encoder.dim_in': (512, 3),
            'encoder.headers': [],
            'encoder.encoder_channel': 128,
            'encoder.num_query': 1,
            # 'encoder.keys': {'hand_state': 7},
            'encoder.keys': {'hand_state': 9},
            'encoder.pre_ln_bias': True,
            'encoder.patch_size': 32,
            'encoder.patch_encoder_type': 'mlp',
            'encoder.patch_overlap': 1.0,
            'encoder.group_type': 'fps',
            'encoder.patch_type': 'mlp',
        }
    )
    device: str = 'cuda:0'

    load_ckpt: Optional[str] = None

    # Training parameters
    num_epoch: int = 256
    batch_size: int = 1024

    # hmm
    valid_ratio: float = 0.2
    save_period: int = 4
    num_workers: int = 32

    # learning rate parameters
    base_lr: float = 2e-4  # or maybe go higher
    min_lr: float = 1e-6
    # Maybe this needs to be longer
    # cos_cycle_steps: int = 256
    cos_cycle_steps: int = num_epoch // 2
    # Maybe this needs to be longer
    cos_warmup_steps: int = 4
    cos_gamma: float = 0.75

    # optimizer regularization
    weight_decay: float = 5e-4
    max_grad_norm: float = 4.0

    def __post_init__(self):
        self.dataset.device = self.device
        self.name = F'{self.group}-{self.seed:06d}'
        # self.model.
        self.dataset.valid_ratio = self.valid_ratio

        # FIXME: hardcoded
        self.process.normalize_type['hand0'] = 'pose_6drot'
        self.process.normalize_type['hand1'] = 'pose_6drot'
        self.process.normalize_type['hand_pose'] = 'pose_6drot'
        self.process.normalize_type['object_pose'] = 'pose_6drot'


def drop___key__(x):
    x = dict(x)
    x.pop('__key__')
    return x


def collate_wds(x):
    return th.utils.data.default_collate([drop___key__(e) for e in x])


def test_permutation_preserved():
    device = 'cuda:0'
    transform = Preprocess(Preprocess.Config(
        ref_cloud_key=['cloud0']), device=device)
    dataset = DynDataset(
        DynDataset.Config(
            data_root='/tmp/dyn-v12',
            max_count=128,
            device=device),
        'train', transform=transform)

    for data in dataset:
        # print({k: v.shape for k, v in data.items()})
        data = {k: v.to(device) for (k, v) in data.items()}

        cld0 = dcn(data['cloud0'])
        cld1 = dcn(data['cloud1'])

        # Check whether permutation is preserved
        pcd0 = o3d_cloud_from_cloud(dcn(data['cloud0']), color=(1, 0, 0))
        pcd1 = o3d_cloud_from_cloud(dcn(data['cloud1']), color=(0, 0, 1))
        match = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pcd0, pcd1, list(zip(np.arange(512), np.arange(512))))

        # == apply kabsch assuming correct correspondence ==
        R, *_ = Rotation.align_vectors(cld0 - cld0.mean(axis=0),
                                       cld1 - cld1.mean(axis=0))
        R = R.as_matrix()
        pcd0_from_1 = dcn(data['cloud1']) @ R.T
        pcd0_from_1 += cld0.mean(axis=0) - pcd0_from_1.mean(axis=0)

        # == draw correspondences ==
        pcd0_from_1 = o3d_cloud_from_cloud(pcd0_from_1, color=(0, 1, 0))
        match0_from_1 = (o3d.geometry.LineSet.
                         create_from_point_cloud_correspondences(
                             pcd0,
                             pcd0_from_1,
                             list(zip(np.arange(512), np.arange(512)))
                         ))

        print(R)
        o3d.visualization.draw([pcd0, pcd1, pcd0_from_1, match,
                                match0_from_1])
        break


def token_std(tokens: th.Tensor) -> th.Tensor:
    return tokens.reshape(-1, tokens.shape[-1]).std(0).mean()


def train_epoch(cfg, model, train_loader, optimizer, writer, step: int):
    model.train()
    for data in tqdm(train_loader, desc='batch', leave=False):
        data = {k: v.to(cfg.device) for (k, v) in data.items()
                if (v is not None)}
        # == backprop ==
        aux = {}
        loss = model(data['cloud0'], data['cloud1'],
                     data['hand0'], data['hand1'],
                     aux=aux)
        loss = loss.mean()
        grad_step(loss, optimizer,
                  max_grad_norm=cfg.max_grad_norm)

        # == log ==
        with th.no_grad():
            writer.add_scalar('loss', loss.item(),
                              global_step=step)
            writer.add_scalar('z_pred_std', token_std(aux['z_pred']),
                              global_step=step)
            writer.add_scalar('z_full_std', token_std(aux['z_full']),
                              global_step=step)

            # Add individual loss terms for VIC Reg
            for key in ['varloss', 'invloss', 'covloss']:
                if key not in aux:
                    continue
                writer.add_scalar(key, aux[key].mean(),
                                  global_step=step)
        if 'loss' in aux:
            for k, v in aux['loss'].items():
                writer.add_scalar(F'loss/{k}',
                                  v,
                                  global_step=step)
        lr: float = next(iter(optimizer.param_groups))['lr']
        writer.add_scalar('learning_rate', lr, global_step=step)
        writer.add_scalar('global_step', step, global_step=step)

        step += 1
    return step


def valid_epoch(cfg, model, valid_loader):
    model.eval()
    val_loss: float = 0.0
    val_loss_count: int = 0
    for data in tqdm(valid_loader, desc='valid', leave=False):
        # data = {k: v.to(cfg.device) for (k, v) in data.items()}
        data = {k: v.to(cfg.device) for (k, v) in data.items()
                if (v is not None)}
        with th.no_grad():
            val_loss = model(data['cloud0'], data['cloud1'],
                             data['hand0'], data['hand1'])
        loss = val_loss
        val_loss += loss.item()
        val_loss_count += 1
    if val_loss_count <= 0:
        return 0.0
    return (val_loss / val_loss_count)


def load_dataset(cfg, aux=None):
    if cfg.use_wds:
        data_cfg = replace(cfg.dataset,
                           device='cpu',
                           cache_pkl=None,
                           preload=False)
        train_transform = Preprocess(cfg.process, 'cpu')

        train_dataset = (
            as_wds(F'{cfg.wds_cache}/train',
                   partial(DynDataset, data_cfg, split='train'),
                   shuffle=True, batch_size=cfg.batch_size)
            # .unbatched()
            .map(train_transform)
            # .batched(cfg.batch_size)
        )
        # train_loader = train_dataset
        train_loader = wds.WebLoader(
            train_dataset,
            #     # batch_size=cfg.batch_size,
            batch_size=None,
            num_workers=cfg.num_workers,
            #     collate_fn=collate_wds
            #     # Disabled with WDS
            #     # shuffle=True
        )
    else:
        train_transform = Preprocess(cfg.process, cfg.device)
        train_dataset = DynDataset(cfg.dataset,
                                   'train',
                                   transform=train_transform)
        train_loader = th.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers)

    if cfg.use_wds:
        valid_transform = Preprocess(replace(cfg.process,
                                             rotate=False,
                                             translate=0.0,
                                             noise_scale=0.0),
                                     'cpu')
        data_cfg = replace(cfg.dataset,
                           device='cpu',
                           cache_pkl=None,
                           preload=False)
        valid_dataset = as_wds(
            F'{cfg.wds_cache}/valid',
            partial(DynDataset, data_cfg, split='valid'),
            shuffle=False, batch_size=cfg.batch_size).map(valid_transform)
        valid_loader = wds.WebLoader(
            valid_dataset,
            batch_size=None,
            # batch_size=cfg.batch_size,
            num_workers=max(1, int(cfg.valid_ratio * cfg.num_workers)),
            # collate_fn=collate_wds
            # Disabled with WDS
            # shuffle=False
        )
    else:
        valid_transform = Preprocess(replace(cfg.process,
                                             rotate=False,
                                             translate=0.0,
                                             noise_scale=0.0),
                                     cfg.device)
        valid_dataset = DynDataset(cfg.dataset,
                                   'valid',
                                   transform=valid_transform)
        valid_loader = th.utils.data.DataLoader(
            valid_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=max(1, int(cfg.valid_ratio * cfg.num_workers)))
    if aux is not None:
        aux['train_transform'] = train_transform
        aux['valid_transform'] = valid_transform
        aux['train_dataset'] = train_dataset
        aux['valid_dataset'] = valid_dataset
    return (train_loader, valid_loader)


@with_wandb
def learn(cfg: Config):
    set_seed(cfg.seed)
    path = RunPath(cfg.path)

    OmegaConf.save(OmegaConf.structured(cfg),
                   path.dir / 'cfg.yaml')

    device: str = cfg.device
    model = DynModel(cfg.model).to(device=device)
    train_loader, valid_loader = load_dataset(cfg)

    ic(model)
    optimizer = th.optim.AdamW(model.parameters(), cfg.base_lr,
                               weight_decay=cfg.weight_decay
                               )
    if cfg.load_ckpt is not None:
        ckpt: str = last_ckpt(cfg.load_ckpt,
                              key=step_from_ckpt)
        load_ckpt(dict(model=model, optimizer=optimizer), ckpt,
                  strict=True)

    writer = SummaryWriter(log_dir=str(path.log))

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        cfg.cos_cycle_steps,
        max_lr=cfg.base_lr,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.cos_warmup_steps,
        gamma=cfg.cos_gamma)

    step: int = 0
    try:
        d = path.dir
        run_id = F'{d.parent.name}/{d.name}'
        for epoch in tqdm(range(cfg.num_epoch),
                          desc=F'[{run_id}]path.epoch'):

            # === TRAIN ===
            step = train_epoch(
                cfg,
                model,
                train_loader,
                optimizer,
                writer,
                step)

            # === VALID ===
            val_loss: float = valid_epoch(cfg, model, valid_loader)
            writer.add_scalar('val_loss', val_loss,
                              global_step=step)

            # writer.add_scalar('z_part_std', token_std(aux['z_part']),
            #                   global_step=step)
            # writer.add_scalar('z_full_std', token_std(aux['z_full']),
            #                   global_step=step)
            # Metadata I guess
            writer.add_scalar('epoch', epoch, global_step=step)

            scheduler.step()
            if epoch % cfg.save_period == 0:
                save_ckpt(dict(model=model, optimizer=optimizer),
                          F'{path.ckpt}/epoch-{epoch:03d}.ckpt')
    finally:
        save_ckpt(dict(model=model, optimizer=optimizer),
                  F'{path.ckpt}/last.ckpt')

        if cfg.use_hfhub and (cfg.hf_repo_id is not None):
            upload_ckpt(
                cfg.hf_repo_id,
                (path.ckpt / 'last.ckpt'),
                cfg.name)


# @hydra_cli(
#    config_path='../../../src/pkm/data/cfg/',
#    # config_path='/home/user/mambaforge/envs/genom/lib/python3.8/site-packages/pkm/data/cfg/',
#    config_name='train_rl')
def main(cfg: Config):
    if cfg.use_wandb:
        group = input('Group?')
        cfg = replace(cfg, group=group)
    learn(cfg)


if __name__ == '__main__':
    main(Config())
