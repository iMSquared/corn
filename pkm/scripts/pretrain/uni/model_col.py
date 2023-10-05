#!/usr/bin/env python3


from typing import Optional, Dict
from dataclasses import dataclass

import torch as th
import torch.nn as nn

from pkm.models.common import MLP
from pkm.models.rl.net.icp import ICPNet
from pkm.util.config import ConfigBase, recursive_replace_map


class ColModelWrapper(nn.Module):

    @dataclass
    class Config(ConfigBase):
        pass

    def __init__(self, cfg: Config, encoder: ICPNet):
        super().__init__()
        self.cfg = cfg

        self.encoder = encoder
        e_cfg = encoder.cfg

        # FIXME: hardcoded decoder architectures
        # (copied from the original impl. in ICPNet)
        self.decoder_c_h = MLP((e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                1),
                               use_bn=False,
                               use_ln=True,
                               pre_ln_bias=e_cfg.pre_ln_bias)
        self.decoder_c_e = MLP((e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                1),
                               use_bn=False,
                               use_ln=True,
                               pre_ln_bias=e_cfg.pre_ln_bias)

        self.loss_c_h = nn.BCEWithLogitsLoss()
        self.loss_c_e = nn.BCEWithLogitsLoss()

    def _group(self,
               points: th.Tensor,
               indices: th.Tensor,
               patch_size: int):
        sorted_points = th.take_along_dim(
            points, indices.reshape(*points.shape[:-1], 1), dim=-2)
        grouped_points = sorted_points.reshape(
            *points.shape[:-2], -1, patch_size)
        return grouped_points

    def forward(self,
                x: th.Tensor,
                h: th.Tensor,
                c_h: th.Tensor,
                c_e: Optional[th.Tensor] = None,
                aux: Optional[Dict[str, th.Tensor]] = None):
        """
        Arg:
            x: cloud
            h: hand pose
            c_h: (cloud<->hand) contact flag
            c_e: (cloud<->env)  contact flag

        Return:
            loss(es)
        """

        cfg = self.cfg

        # Predict ...
        _aux = {}
        _, z = self.encoder(x, ctx={'hand_state': h},
                            aux=_aux)
        if aux is not None:
            aux.update(_aux)

        offset = len(self.encoder.keys)
        pred_c_h = self.decoder_c_h(z[..., offset:, :]).squeeze(dim=-1)
        pred_c_e = None
        if c_e is not None:
            pred_c_e = self.decoder_c_h(z[..., offset:, :]).squeeze(dim=-1)

        # Compute labels ...
        with th.no_grad():
            true_c_h = (self._group(c_h.reshape(*x.shape[:-1], 1),
                                    _aux['fps_nn_idx'],
                                    self.encoder.true_patch_size)
                        .squeeze(dim=-1)
                        .any(dim=-1))

            true_c_e = None
            if c_e is not None:
                true_c_e = (self._group(c_e.reshape(*x.shape[:-1], 1),
                                        _aux['fps_nn_idx'],
                                        self.encoder.true_patch_size)
                            .squeeze(dim=-1)
                            .any(dim=-1))

        # Compute losses ...
        loss_c_h = self.loss_c_h(pred_c_h, true_c_h.float())
        loss_c_e = None
        if true_c_e is not None:
            loss_c_e = self.loss_c_e(pred_c_e, true_c_e.float())

        if aux is not None:
            aux['loss'] = {
                'c_h': loss_c_h,
                'c_e': loss_c_e
            }
        if loss_c_e is None:
            return loss_c_h
        return loss_c_h + loss_c_e


class ColModel(ColModelWrapper):

    @dataclass
    class Config(ColModelWrapper.Config):
        encoder: ICPNet.Config = recursive_replace_map(
            ICPNet.Config(), {
                'p_drop': 0.01,
                'encoder.num_hidden_layers': 2,
                'dim_in': (512, 3),
                'headers': [],
                'encoder_channel': 128,
                'num_query': 1,
                'keys': {'hand_state': 7},
                'pre_ln_bias': True,
                'encoder.num_hidden_layers': 2,
                'patch_size': 32,
                'patch_encoder_type': 'mlp',
                'patch_overlap': 1.0,
                'group_type': 'fps',
                'patch_type': 'mlp',
            })

        def __post_init__(self):
            if hasattr(ColModelWrapper.Config, '__post_init__'):
                super().__post_init__()

    def __init__(self, cfg: Config):
        super().__init__(cfg, ICPNet(cfg.encoder))


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
    c_e = th.randn((batch_size, cloud_size),
                   dtype=th.float,
                   device=device) > 0
    loss = model(x, h, c_h, c_e=None)
    print(loss)


if __name__ == '__main__':
    main()
