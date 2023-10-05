#!/usr/bin/env python3


from typing import Optional, Dict
from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pkm.models.common import MLP
from pkm.models.rl.net.pointnet import (
    PointNetEncoder,
    FPModule)
from pkm.util.config import ConfigBase, recursive_replace_map


class ColModelWrapper(nn.Module):

    @dataclass
    class Config(ConfigBase):
        p_drop: float = 0.0

    def __init__(self, cfg: Config, encoder: PointNetEncoder):
        super().__init__()
        self.cfg = cfg

        self.encoder = encoder
        e_cfg = encoder.cfg
        sa_dims = e_cfg.sa_dims

        dim_0: int = e_cfg.dim_in[-1] + sum(encoder.keys.values(), 0)
        self.fp3_module = FPModule(1, MLP([sa_dims[2] + sa_dims[1], 256, 256]))
        self.fp2_module = FPModule(3, MLP([sa_dims[1] + sa_dims[0], 256, 128]))
        self.fp1_module = FPModule(3, MLP([sa_dims[0] + dim_0,
                                           128, 128, 128]))
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 1)
        self.loss_c_h = nn.BCEWithLogitsLoss()

    def forward(self,
                x: th.Tensor,
                h: th.Tensor,
                c_h: th.Tensor,
                aux: Optional[Dict[str, th.Tensor]] = None):
        """
        Arg:
            x: cloud
            h: hand pose
            c_h: (cloud<->hand) contact flag

        Return:
            loss(es)
        """

        cfg = self.cfg
        s = x.shape

        # == encode ==
        _aux = {}
        _ = self.encoder(x,
                         ctx={'hand_state': h},
                         aux=_aux)

        # == parse intermedaite features ==
        if aux is not None:
            aux.update(_aux)

        outputs = _aux['outputs']
        (sa0_out, sa1_out, sa2_out, sa3_out) = outputs

        # == decode ==
        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=cfg.p_drop, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=cfg.p_drop, training=self.training)
        x = self.lin3(x)

        # restore shape (full cloud + logits)
        x = x.reshape(*s[:-1], x.shape[-1])
        pred_c_h = x.squeeze(dim=-1)

        # Compute losses ...
        loss_c_h = self.loss_c_h(pred_c_h,
                                 c_h.float())

        if aux is not None:
            aux['loss'] = {
                'c_h': loss_c_h,
            }
            aux['log'] = {
                    # To compute binary accuracy,
                    # compare logits (>0) vs labels (> 0.5)
                    'c_h_acc': ((pred_c_h > 0) == (c_h.float() > 0.5)).float().mean().item()
            }
        return loss_c_h


class ColModel(ColModelWrapper):

    @dataclass
    class Config(ColModelWrapper.Config):
        encoder: PointNetEncoder.Config = recursive_replace_map(
            PointNetEncoder.Config(), {
                # 'p_drop': 0.01,
                'dim_in': (512, 3),
                'keys': {'hand_state': 9}
            })

        def __post_init__(self):
            if hasattr(ColModelWrapper.Config, '__post_init__'):
                super().__post_init__()

    def __init__(self, cfg: Config):
        super().__init__(cfg, PointNetEncoder(cfg.encoder))


def main():
    batch_size: int = 3
    cloud_size: int = 512
    device: str = 'cuda:0'

    model = ColModel(ColModel.Config()).to(device)
    x = th.randn((batch_size, cloud_size, 3),
                 dtype=th.float,
                 device=device)
    h = th.randn((batch_size, 7),
                 dtype=th.float,
                 device=device)
    c_h = th.randn((batch_size, cloud_size),
                   dtype=th.float,
                   device=device) > 0
    loss = model(x, h, c_h)
    print(loss)


if __name__ == '__main__':
    main()
