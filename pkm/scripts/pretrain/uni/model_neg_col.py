
#!/usr/bin/env python3


from typing import Optional, Dict
from dataclasses import dataclass

import torch as th
import torch.nn as nn

from pkm.models.common import MLP
from pkm.models.rl.net.icp import ICPNet
from pkm.util.config import ConfigBase, recursive_replace_map
from pkm.models.cloud.point_mae import (
    mask,
    PointMAEDecoder
)


@dataclass
class TransformerDecoderConfig(ConfigBase):
    d_model: int = 128
    nhead: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    # activation: str = 'relu'
    activation: str = 'gelu'
    # layer_norm_eps: float = 1e-5
    layer_norm_eps: float = 1e-6
    # batch_first: bool = False
    batch_first: bool = True
    norm_first: bool = False
    # device=None
    # dtype=None


class NegColModelWrapper(nn.Module):
    """
    Train ICPNet based on collision.
    Unlike `ColModelWrapper`, accounts for "negative"
    points in the empty space outside of the object.
    """

    @dataclass
    class Config(ConfigBase):
        mask_ratio: float = 0.0
        decoder: TransformerDecoderConfig = TransformerDecoderConfig()

        coef_c_h: float = 1.0
        coef_c_e: float = 0.0
        coef_cls: float = 1.0

    def __init__(self, cfg: Config, encoder: ICPNet):
        super().__init__()

        cfg = recursive_replace_map(cfg, {
            'decoder.d_model': encoder.cfg.encoder_channel,
        })
        self.cfg = cfg

        self.encoder = encoder
        e_cfg = encoder.cfg
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=cfg.decoder.d_model,
                                       nhead=cfg.decoder.nhead,  # ??
                                       dim_feedforward=cfg.decoder.dim_feedforward,
                                       dropout=cfg.decoder.dropout,
                                       activation=nn.GELU(),
                                       layer_norm_eps=cfg.decoder.layer_norm_eps,
                                       batch_first=cfg.decoder.batch_first,
                                       norm_first=cfg.decoder.norm_first),
            2)

        # FIXME: hardcoded decoder architectures
        # (copied from the original impl. in ICPNet)
        self.predict_c_h = MLP((e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                1),
                               use_bn=False,
                               use_ln=True,
                               pre_ln_bias=e_cfg.pre_ln_bias)
        self.predict_c_e = MLP((e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                1),
                               use_bn=False,
                               use_ln=True,
                               pre_ln_bias=e_cfg.pre_ln_bias)
        # Predict whether patch is neg or pos
        self.predict_cls = MLP((e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                e_cfg.encoder.layer.hidden_size,
                                1),
                               use_bn=False,
                               use_ln=True,
                               pre_ln_bias=e_cfg.pre_ln_bias)

        self.loss_c_h = nn.BCEWithLogitsLoss()
        self.loss_c_e = nn.BCEWithLogitsLoss()
        self.loss_cls = nn.BCEWithLogitsLoss()

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
                x_neg: th.Tensor,

                c_h: th.Tensor,
                # c_e: Optional[th.Tensor] = None,
                c_neg_h: Optional[th.Tensor] = None,
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
        _aux_group = {}
        x_sorted, patch_center = self.encoder._group(x, aux=_aux_group)
        z_emb, pos_emb, z_pre_pos_emb = self.encoder._embed(
            x_sorted, patch_center)
        _aux_mask = {}
        z_msk, msk, idx = mask(z_emb, cfg.mask_ratio,
                               aux=_aux_mask)

        # FIXME: hardcoded z_ctx key
        z_ctx = [self.encoder.tokenizes['hand_state']
                 (h.reshape(*x.shape[:-2], -1))]
        z_inp = th.cat(z_ctx + [z_msk], dim=-2)
        z_enc, _, _ = self.encoder.encoder(z_inp)

        # == prediction targets
        _aux_group_neg = {}
        x_sorted_neg, patch_center_neg = self.encoder._group(
            x_neg, aux=_aux_group_neg)

        x_hid = th.take_along_dim(x_sorted,
                                  _aux_mask['ids_hide'][..., None, None],
                                  dim=-3)
        c_hid = th.take_along_dim(patch_center,
                                  _aux_mask['ids_hide'][..., None],
                                  dim=-2)
        x_query = th.cat([x_hid, x_sorted_neg], dim=-3)
        z_query_emb, _, _ = self.encoder._embed(
            x_query, th.cat([c_hid, patch_center_neg], dim=-2))
        z_out = self.decoder(z_query_emb, z_enc)

        pred_c_h = self.predict_c_h(z_out).squeeze(dim=-1)
        # pred_c_e = None
        # if c_e is not None:
        #    pred_c_e = self.predict_c_e(z_out).squeeze(dim=-1)

        if True:
            pred_cls = self.predict_cls(z_out).squeeze(dim=-1)
            true_cls = th.cat([
                th.ones_like(c_hid[..., 0]),
                th.zeros_like(patch_center_neg[..., 0]),
            ], dim=-1)

        # (ones * len(c_hid),
        #             zeros * len(patch_center_neg))

        # ... Compute labels ...
        with th.no_grad():
            true_c_h_obj = (self._group(c_h.reshape(*x.shape[:-1], 1),
                                        _aux_group['fps_nn_idx'],
                                        self.encoder.true_patch_size)
                            .squeeze(dim=-1)
                            .any(dim=-1))
            true_c_h_obj_hid = th.take_along_dim(true_c_h_obj,
                                                 _aux_mask['ids_hide'],
                                                 dim=-1)

            true_c_h_neg = (self._group(c_neg_h.reshape(*x_neg.shape[:-1], 1),
                                        _aux_group['fps_nn_idx'],
                                        self.encoder.true_patch_size)
                            .squeeze(dim=-1)
                            .any(dim=-1))
            true_c_h = th.cat([true_c_h_obj_hid, true_c_h_neg], dim=-1)
            # print(true_c_h.shape, pred_c_h.shape)

            # true_c_e = None
            # if c_e is not None:
            #    true_c_e = (self._group(c_e.reshape(*x.shape[:-1], 1),
            #                            _aux_group['fps_nn_idx'],
            #                            self.encoder.true_patch_size)
            #                .squeeze(dim=-1)
            #                .any(dim=-1))

        # Compute losses ...
        loss: float = 0.0

        loss_c_h = self.loss_c_h(pred_c_h, true_c_h.float())
        loss = loss + cfg.coef_c_h * loss_c_h

        loss_c_e = None
        # if true_c_e is not None:
        #    loss_c_e = self.loss_c_e(pred_c_e, true_c_e.float())
        # loss = loss + cfg.coef_c_e * loss_c_e

        loss_cls = self.loss_cls(pred_cls, true_cls)
        loss = loss + cfg.coef_cls * loss_cls

        if aux is not None:
            aux['loss'] = {
                'c_h': loss_c_h,
                'cls': loss_cls,
                # 'c_e': loss_c_e
            }
        return loss


class NegColModel(NegColModelWrapper):

    @dataclass
    class Config(NegColModelWrapper.Config):
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
            if hasattr(NegColModelWrapper.Config, '__post_init__'):
                super().__post_init__()

    def __init__(self, cfg: Config):
        super().__init__(cfg, ICPNet(cfg.encoder))


def main():
    batch_size: int = 3
    cloud_size: int = 512
    device: str = 'cuda:0'

    model = NegColModel(NegColModel.Config(mask_ratio=0.5)).to(device)
    x = th.randn((batch_size, cloud_size, 3),
                 dtype=th.float,
                 device=device)
    h = th.randn((batch_size, 7),
                 dtype=th.float,
                 device=device)
    x_neg = th.randn((batch_size, cloud_size, 3),
                     dtype=th.float,
                     device=device)
    c_h = th.randn((batch_size, cloud_size),
                   dtype=th.float,
                   device=device) > 0
    # c_e = th.randn((batch_size, cloud_size),
    #               dtype=th.float,
    #               device=device) > 0
    c_neg_h = th.randn((batch_size, cloud_size),
                       dtype=th.float,
                       device=device) > 0
    loss = model(x, h, x_neg, c_h,
                 # c_e=c_e,
                 c_neg_h=c_neg_h)
    print(loss)


if __name__ == '__main__':
    main()
