#!/usr/bin/env python3

from dataclasses import dataclass, replace

from typing import Optional, Dict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops

from pytorch3d.loss import chamfer_distance

from timm.models.layers import DropPath
from pkm.models.rl.net.icp import ICPNet
from pkm.models.common import (
    MLP,
    transfer, soft_update, hard_update
)
from pkm.models.cloud.point_mae import (
    mask,
    PointMAEDecoder
)
from pkm.util.torch_util import masked_mean
from pkm.util.config import ConfigBase, recursive_replace_map
from pkm.util.math_util import apply_pose_tq

from icecream import ic
from pkm.models.rl.net.vicreg import VICRegLoss


class P2VModelWrapper(nn.Module):

    @dataclass
    class Config(ConfigBase):
        decoder: PointMAEDecoder.Config = PointMAEDecoder.Config(
            use_pred=False,
            pred_embed=True)
        tau: float = 0.0001
        mask_ratio: float = 0.0
        average_targets: int = 1
        share_patch_encoder: bool = True

    def __init__(self, cfg: Config, encoder: ICPNet):
        super().__init__()
        cfg = recursive_replace_map(cfg, {
            # 'decoder.hidden_size': encoder.cfg.encoder_channel,
            # 'decoder.decoder_hidden_size': encoder.cfg.encoder_channel,
            'decoder.embed_size': encoder.cfg.encoder_channel,
        })
        self.cfg = cfg

        self.encoder = encoder
        self.decoder = PointMAEDecoder(cfg.decoder)

        target_cfg = replace(encoder.cfg,
                             output_hidden=True)

        self.target_encoder = ICPNet(target_cfg)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self,
                x: th.Tensor,
                aux: Optional[Dict[str, th.Tensor]] = None):
        """
        Arg:
            x: cloud at t=0
            x1: cloud at t=1
            h0: hand pose at t=0
            h1: hand pose at t=1

        Return:
            loss
        """
        cfg = self.cfg

        # Compute target according to Point2Vec.
        with th.inference_mode():
            _aux = {}
            _, z_full = self.target_encoder(x, {}, aux=_aux)
            if aux is not None:
                aux.update(_aux)
            zs = th.stack(_aux['zs'][-cfg.average_targets:], dim=0)
            zs = F.layer_norm(zs, zs.shape[-1:])
            zs = zs.mean(dim=0)
            zs = F.layer_norm(zs, zs.shape[-1:])
            z_full = zs

        z, pe, _ = self.encoder._embed(_aux['patches'].clone(),
                                       _aux['centers'].clone())
        z_part, m, i = mask(z, cfg.mask_ratio)
        z_part, _, _ = self.encoder.encoder(z_part)
        z_part = self.encoder.layernorm(z_part)
        # For validation when mask_ratio=0.0; should be the same
        # z_part = th.take_along_dim(z_part, i[...,None], dim=-2)
        z_part, _, _ = self.decoder(z_part, i, pe)

        if aux is not None:
            aux['z_full'] = z_full.detach()
            aux['z_pred'] = z_part.detach()

        # Compute loss
        # ic(z_part.shape, z_full.shape)
        if aux is not None:
            aux['loss'] = {}
        loss = self.loss(z_part, z_full.clone().detach())
        loss = masked_mean(loss, m[..., None])
        if aux is not None:
            aux['loss']['p2v'] = loss

        # Update target encoder
        # target = (target * (1.0 - tau) + encoder * tau)
        soft_update(self.encoder,
                    self.target_encoder,
                    cfg.tau)

        if cfg.share_patch_encoder:
            hard_update(self.encoder.patch_encoder,
                        self.target_encoder.patch_encoder)
            hard_update(self.encoder.pos_embed,
                        self.target_encoder.pos_embed)
        return loss


class P2VModel(P2VModelWrapper):

    @dataclass
    class Config(P2VModelWrapper.Config):
        encoder: ICPNet.Config = recursive_replace_map(
            ICPNet.Config(), {
                'p_drop': 0.01,
                'encoder.num_hidden_layers': 2,
                'dim_in': (512, 3),
                'headers': [],
                'encoder_channel': 128,
                'num_query': 1,
                'keys': {},
                'pre_ln_bias': True,
                'encoder.num_hidden_layers': 2,
                'patch_size': 32,
                'patch_encoder_type': 'mlp',
                'patch_overlap': 1.0,
                'group_type': 'fps',
                'patch_type': 'mlp',
            })

        def __post_init__(self):
            if hasattr(P2VModelWrapper.Config, '__post_init__'):
                super().__post_init__()
            c = self.encoder.encoder_channel
            self.decoder = replace(self.decoder,
                                   hidden_size=c,
                                   decoder_hidden_size=c,
                                   embed_size=c)

    def __init__(self, cfg: Config):
        super().__init__(cfg, ICPNet(cfg.encoder))


def main():
    from torchview import draw_graph
    from pkm.models.common import grad_step
    device: str = 'cuda:0'
    model = P2VModel(P2VModel.Config(
        mask_ratio=0.5)).to(device)  # should be zero no?
    optimizer = th.optim.Adam(model.parameters(), 1e-3)
    for _ in range(1):
        x = th.randn((7, 512, 3),
                     dtype=th.float,
                     device=device)
        loss = model(x)
        ic(loss)
        grad_step(loss, optimizer)

    # model_graph = draw_graph(model,
    #                          input_data=(x,),
    #                          device=device)
    # model_graph.visual_graph.render('/tmp/docker/point2vec.svg')


if __name__ == '__main__':
    main()
