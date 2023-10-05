#!/usr/bin/env python3


from typing import Optional, Dict
from dataclasses import dataclass

import torch as th
import torch.nn as nn

from pkm.models.common import MLP, MultiHeadLinear, attention
from pkm.models.rl.net.icp import ICPNet
from pkm.util.config import ConfigBase, recursive_replace_map

import numpy as np


def rot6d_to_matrix(rot6d: th.Tensor) -> th.Tensor:
    x = rot6d[..., :3]
    y = rot6d[..., 3:6]
    x = x / th.norm(x, dim=-1, keepdim=True)
    y = y - th.sum(x*y, dim=-1, keepdim=True) * x
    y = y / th.norm(y, dim=-1, keepdim=True)
    z = th.cross(x,y, dim=-1)    
    R = th.stack((x, y, z), dim=-2)
    return R

class RotModelWrapper(nn.Module):

    @dataclass
    class Config(ConfigBase):
        num_query: int = 4
        loss: str= 'mse'

    def __init__(self, cfg: Config, encoder: ICPNet):
        super().__init__()
        self.cfg = cfg

        self.encoder = encoder
        e_cfg = encoder.cfg

        # FIXME: hardcoded decoder architectures
        # (copied from the original impl. in ICPNet)

        self.register_parameter(
                'query_token',
                nn.Parameter(
                    th.zeros(cfg.num_query, e_cfg.encoder.layer.hidden_size),
                    requires_grad=True
                ))
        num_patches = e_cfg.dim_in[0] // encoder.true_patch_size
        # dim = e_cfg.encoder.layer.hidden_size * cfg.num_query *2 # 4096
        dim = e_cfg.encoder.layer.hidden_size * num_patches *2 # 4096

        self.decoder_rot = MLP((dim,
                                dim // 8,
                                dim // 64,
                                6),
                               use_bn=False,
                               use_ln=True,
                               pre_ln_bias=e_cfg.pre_ln_bias)
        self.to_kv = MultiHeadLinear(e_cfg.encoder.layer.hidden_size,
                                     2,
                                     e_cfg.encoder.layer.hidden_size,
                                     unbind=True)
        self.loss = nn.MSELoss() if cfg.loss == 'mse' else self.arcsine_loss

    def arcsine_loss(self, pred_rotation, rotation):
        
        frobenius = th.sqrt(th.sum((pred_rotation - rotation) ** 2, dim=1))
        loss = 2 * \
            th.arcsin(th.minimum(th.ones_like(
                frobenius), frobenius / (2 * np.sqrt(2))))
        return loss.mean(dim=0)
    
    def geodesic_distance(self, m1, m2):
        m = th.bmm(m1, m2.transpose(-1, -2)) 
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1)/2
        cos = cos.clamp(-1, 1)
        theta = th.acos(cos).mean()
        return theta

    def forward(self,
                x1: th.Tensor,
                x2: th.Tensor,
                rot_diff: th.Tensor,
                aux: Optional[Dict[str, th.Tensor]] = None):
        """
        Arg:
            x1: cloud1
            x2: cloud2
            rot_diff: rotation difference
        Return:
            loss(es)
        """
        cfg = self.cfg
        s = x1.shape
        # Predict ...
        _aux = {}
        _, z1 = self.encoder(
            x1, ctx = {},
            aux = _aux
        )
        # k1, v1 = self.to_kv(z1)
        # z1 = attention(
        #     self.query_token,
        #     k1,
        #     v1,
        #     key_padding_mask = None
        # )
        _, z2 = self.encoder(
            x2, ctx = {},
            aux = _aux
        )
        # k2, v2 = self.to_kv(z2)
        # z2 = attention(
        #     self.query_token,
        #     k2,
        #     v2,
        #     key_padding_mask = None
        # )
        z1 = z1.reshape(*s[:-2], -1)
        z2 = z2.reshape(*s[:-2], -1)
        z = th.cat([z1, z2], dim = -1)

        if aux is not None:
            aux.update(_aux)

        pred_rot_diff = self.decoder_rot(z)
    
        # Compute losses ...
        # loss_rot = self.arcsine_loss(pred_rot_diff, rot_diff)
        if self.cfg.loss =='mse':
            loss_rot = self.loss(pred_rot_diff, rot_diff)
        else:
            rot_mat_pred = rot6d_to_matrix(pred_rot_diff).reshape(
                *s[:-2], -1
            )
            rot_mat_gt = rot6d_to_matrix(rot_diff).reshape(
                *s[:-2], -1
            )
            loss_rot = self.loss(rot_mat_pred, rot_mat_gt)
        geosedric_distance = self.geodesic_distance(
            rot6d_to_matrix(pred_rot_diff),
            rot6d_to_matrix(rot_diff)
        )
        if aux is not None:
            aux['loss'] = {
                'rot': loss_rot,
                'geosedric': geosedric_distance
            }

        return loss_rot


class RotModel(RotModelWrapper):

    @dataclass
    class Config(RotModelWrapper.Config):
        encoder: ICPNet.Config = recursive_replace_map(
            ICPNet.Config(), {
                'p_drop': 0.01,
                'encoder.num_hidden_layers': 2,
                'dim_in': (512, 3),
                'headers': [],
                'encoder_channel': 128,
                'num_query': 1,
                'keys': None,
                'pre_ln_bias': True,
                'encoder.num_hidden_layers': 2,
                'patch_size': 32,
                'patch_encoder_type': 'mlp',
                'patch_overlap': 1.0,
                'group_type': 'fps',
                'patch_type': 'mlp',
            })

        def __post_init__(self):
            if hasattr(RotModelWrapper.Config, '__post_init__'):
                super().__post_init__()

    def __init__(self, cfg: Config):
        super().__init__(cfg, ICPNet(cfg.encoder))


def main():
    batch_size: int = 3
    cloud_size: int = 512
    device: str = 'cuda:0'

    model = RotModel(RotModel.Config()).to(device)
    x1 = th.randn((batch_size, cloud_size, 3),
                 dtype=th.float,
                 device=device)
    x2 = th.randn((batch_size, cloud_size, 3),
                 dtype=th.float,
                 device=device)
    rot_diff = th.randn((batch_size, 6),
                   dtype=th.float,
                   device=device)
    loss = model(x1, x2, rot_diff)
    print(loss)


if __name__ == '__main__':
    main()
