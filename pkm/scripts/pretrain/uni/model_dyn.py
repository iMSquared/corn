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
from pkm.models.common import transfer, soft_update, MLP
from pkm.models.cloud.point_mae import (
    mask,
    PointMAEDecoder
)
from pkm.util.torch_util import masked_mean
from pkm.util.config import ConfigBase, recursive_replace_map
from pkm.util.math_util import apply_pose_tq

from icecream import ic
from pkm.models.rl.net.vicreg import VICRegLoss


class CFMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        """
        z1: ...,NxD
        z2: ...,NxD
        """
        if True:
            # logits based on embedding difference ||x-y||
            # = as we all know, <x,x> + 2<x,y> + <y,y>
            dz = z2[..., None, :, :] - z1[..., :, None, :]
            logits = - th.einsum('...d, ...d -> ...', dz, dz)  # NxN
            labels = th.arange(z1.shape[-2],
                               dtype=th.int64,
                               device=z1.device).expand(z1.shape[:-1])
        else:
            # logits based on inner product <x, y>
            logits = th.einsum('... n d, ... m d -> ... n m',
                               z1, z2)  # NxDxD
            labels = th.arange(z1.shape[-2],
                               dtype=th.int64,
                               device=z1.device)[None].expand(z1.shape[:-1])
        return self.loss(logits, labels)


class DynModelWrapper(nn.Module):

    @dataclass
    class Config(ConfigBase):
        decoder: PointMAEDecoder.Config = PointMAEDecoder.Config(
            use_pred=False,
            pred_embed=True)
        tau: float = 0.0001
        mask_ratio: float = 0.0
        average_targets: int = 1
        loss_type: str = 'cfm'

        use_rec: bool = True
        rec_coef: float = 1.0
        dyn_coef: float = 1.0

        # def __post_init__(self):
        #    self.decoder.hidden_size = self.encoder.encoder_channel  # ?
        #    self.decoder.decoder_hidden_size = self.encoder.encoder_channel  # ?
        #    self.decoder.embed_size = self.encoder.encoder_channel

    def __init__(self, cfg: Config, encoder: ICPNet):
        super().__init__()
        cfg = recursive_replace_map(cfg, {
            'decoder.hidden_size': encoder.cfg.encoder_channel,
            'decoder.decoder_hidden_size': encoder.cfg.encoder_channel,
            'decoder.embed_size': encoder.cfg.encoder_channel,
        })
        self.cfg = cfg

        self.encoder = encoder
        self.decoder = PointMAEDecoder(cfg.decoder)

        # Use reconstruction loss ?
        if cfg.use_rec:
            self.pred_cloud = MLP(
                (
                    encoder.cfg.encoder_channel,
                    encoder.cfg.encoder_channel,
                    encoder.true_patch_size * 3
                ),
                use_bn=False,
                use_ln=True)
        self.tokenize_action = nn.Linear(
            9, encoder.cfg.encoder_channel)

        target_cfg = replace(encoder.cfg,
                             output_hidden=True)
        self.target_encoder = ICPNet(target_cfg)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

        if cfg.loss_type == 'vic':
            self.loss = VICRegLoss()
        elif cfg.loss_type == 'l1':
            self.loss = nn.SmoothL1Loss(reduction='none')
        elif cfg.loss_type == 'cfm':
            self.loss = CFMLoss()
        else:
            raise KeyError(F'Unknown loss_type={cfg.loss_type}')

    def forward(self,
                x0: th.Tensor,
                x1: th.Tensor,  # == T @ x0
                # action: th.Tensor,
                h0: th.Tensor,
                h1: th.Tensor,
                aux: Optional[Dict[str, th.Tensor]] = None):
        """
        Arg:
            x0: cloud at t=0
            x1: cloud at t=1
            h0: hand pose at t=0
            h1: hand pose at t=1

        Return:
            loss
        """
        cfg = self.cfg

        # Compute "posterior" target
        # similar to Point2Vec.
        _aux = {}
        with th.inference_mode():
            _, z_full = self.target_encoder(x1, {'hand_state': h1}, aux=_aux)
            if aux is not None:
                aux.update(_aux)

            # option 1
            zs = th.stack(_aux['zs'][-cfg.average_targets:], dim=0)
            zs = zs[..., 1:, :]
            zs = F.layer_norm(zs, zs.shape[-1:])
            zs = zs.mean(dim=0)
            zs = F.layer_norm(zs, zs.shape[-1:])
            z_full = zs

            # option 2
            # z_full = z_full[..., 1:, :]
            if aux is not None:
                aux['p1'] = aux['patches']

        # Encode x0 (==x_prev)
        p0 = th.take_along_dim(x0,
                               _aux['fps_nn_idx'].reshape(*x0.shape[:-1], 1),
                               dim=-2)
        p0 = p0.reshape(_aux['patches'].shape)
        # if aux is not None:
        #     aux['p0'] = p0

        # FIXME: slightly different `c0` definition...!!
        c0 = p0.mean(dim=-2)
        z0, pe, _ = self.encoder._embed(p0, c0)
        z_prev, m, i = mask(z0, cfg.mask_ratio)
        h0_tkn = self.encoder.tokenizes['hand_state'](
            h0.reshape(p0.shape[0], -1))
        z_prev = th.cat([h0_tkn, z_prev], dim=-2)
        z_prev, _, _ = self.encoder.encoder(z_prev)

        # Reorder `z_prev`
        if True:
            z_prev = th.cat([
                z_prev[..., :1, :],
                self.decoder.unshuffle(z_prev[..., 1:, :], i, pe)],
                dim=-2)
        else:
            # !! WRONG CODE PATH !!
            z_prev = th.cat([z_prev[..., :1, :],
                            th.take_along_dim(z_prev[..., 1:, :], i[..., None],
                                              dim=-2)],
                            dim=-2)
        # ic(z_prev.shape)
        # print('out', z_prev)
        # print('in', aux['z-pre-ln'])
        # print('(out)pre-ln', z_prev)
        # print( th.amax(aux['z-pre-ln']-z_prev, dim=-1))
        z_prev = self.encoder.layernorm(z_prev)

        # print(z_prev.shape) # 64,17,128 ?
        # print('out-z', z_prev[0, ..., 1:, :])

        # Concatenate "action" token
        z_actn = self.tokenize_action(h1)[..., None, :]
        if False:
            z_pred = z_prev[..., 1:, :]
        else:
            # print(z_actn.shape)
            # print(z_prev.shape)
            z_comb = th.cat([z_actn, z_prev], dim=-2)

            # Decode `z_prev` and only take the patch-related parts
            z_pred, p_preds, _ = self.decoder(z_comb, None, None,
                                              output_hidden_states=True)
            # starts from `2`, since `z_comb` looks like
            # z_actn: tok(h1)
            # z_prev: (tok(h0), patches...)
            z_pred = z_pred[..., 2:, :]
            p_pred = p_preds[-1][..., 2:, :]
            # ^ this means
            # [-1] = take the output from the last laster
            # ... = skip the batch dimensions
            # 2: = take the cloud-relevant patches
            # : = keep the feature dimension

        loss = 0.0

        # Compute point cloud reconstruction loss
        if aux is not None:
            aux['loss'] = {}
        if cfg.use_rec:
            # == pred ==
            pred_x1 = self.pred_cloud(p_pred)
            # pred_x0 = self.pred_cloud(z_prev[..., 1:, :])

            # == true ==
            p1 = th.take_along_dim(
                x1, _aux['fps_nn_idx'].reshape(*x1.shape[:-1], 1), dim=-2)
            p1 = p1.reshape(p0.shape)
            pred_x1 = pred_x1.reshape(p1.shape)
            if aux is not None:
                aux['pred_x1'] = pred_x1

            loss_cd = chamfer_distance(
                pred_x1.reshape(-1, *pred_x1.shape[-2:]),
                p1.reshape(-1, *pred_x1.shape[-2:]),
                norm=1)[0]
            if aux is not None:
                aux['loss']['cd'] = loss_cd
            loss = loss + cfg.rec_coef * loss_cd

        # ic(z_pred.shape, z_full.shape)

        # Compute latent-space forward dynamics loss
        if isinstance(self.loss, CFMLoss):
            loss_dyn = self.loss(
                z_pred,
                z_full.clone().detach()
            )
            # if aux is not None:
            #    aux['loss']['dyn'] = loss_dyn
            # loss = loss + loss_dyn
        else:
            loss_dyn = self.loss(
                z_pred.reshape(-1, z_pred.shape[-1]),
                z_full.clone().detach().reshape(-1, z_full.shape[-1])
            )
            # if aux is not None:
            #    aux['loss']['dyn'] = loss_dyn
            # loss = loss + loss_dyn

        if isinstance(self.loss, VICRegLoss):
            if aux is not None:
                aux.update({k: v.detach() for k, v in loss_dyn.items()})
            loss_dyn = sum([v.mean() for v in loss_dyn.values()])
        elif isinstance(self.loss, CFMLoss):
            pass
        else:
            if cfg.mask_ratio > 0.0:
                m = m.reshape(loss_dyn.shape[0])
                loss_dyn = masked_mean(loss_dyn, m[..., None])
            else:
                loss_dyn = loss_dyn.mean()

        if aux is not None:
            aux['loss']['dyn'] = loss_dyn
        loss = loss + cfg.dyn_coef * loss_dyn

        if aux is not None:
            aux['z_full'] = z_full.detach()
            aux['z_pred'] = z_pred.detach()

        # Update target encoder
        # target = (target * (1.0 - tau) + encoder * tau)
        soft_update(self.encoder,
                    self.target_encoder,
                    cfg.tau)
        return loss


class DynModel(DynModelWrapper):

    @dataclass
    class Config(DynModelWrapper.Config):
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
            if hasattr(DynModelWrapper.Config, '__post_init__'):
                super().__post_init__()
            c = self.encoder.encoder_channel
            self.decoder = replace(self.decoder,
                                   hidden_size=c,
                                   decoder_hidden_size=c,
                                   embed_size=c)

    def __init__(self, cfg: Config):
        super().__init__(cfg, ICPNet(cfg.encoder))


def main():
    import open3d as o3d
    from pkm.util.vis.win_o3d import AutoWindow

    from torchview import draw_graph
    from pkm.models.common import grad_step
    from pkm.util.torch_util import dcn
    device: str = 'cuda:0'
    model = DynModel(DynModel.Config(
        tau=0.001,
        mask_ratio=0.2,
    )).to(device)  # should be zero no?
    ic(model)
    optimizer = th.optim.Adam(model.parameters(), 1e-2)

    batch_size: int = 128
    cloud_size: int = 512
    action_size: int = 7

    for _ in range(1024):
        x = th.randn((batch_size, cloud_size, 3),
                     dtype=th.float,
                     device=device) + th.randn((1, cloud_size, 3),
                                               dtype=th.float,
                                               device=device)
        action = th.randn((batch_size, 7),
                          dtype=th.float,
                          device=device)
        action[..., 3:7] /= th.linalg.norm(action[...,
                                                  3:7],
                                           dim=-1,
                                           keepdim=True)
        zero_action = action * 0 + th.as_tensor([0, 0, 0, 0, 0, 0, 1],
                                                dtype=action.dtype,
                                                device=action.device)
        x1 = apply_pose_tq(action[..., None, :], x)
        aux = {}
        # loss = model(x, x, zero_action, zero_action, aux=aux)
        # print( th.amax(aux['p0'] - aux['p1']))
        loss = model(x, x1, zero_action, action, aux=aux)
        if False:
            ic(aux['p0'].shape)
            ic(aux['p1'].shape)
            p0 = dcn(aux['p0'][5])
            p1 = dcn(aux['p1'][5])

            cld0 = o3d.geometry.PointCloud()
            cld0.points = o3d.utility.Vector3dVector(dcn(p0).reshape(-1, 3))
            cld1 = o3d.geometry.PointCloud()
            cld1.points = o3d.utility.Vector3dVector(dcn(p1).reshape(-1, 3))

            line = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                cld0, cld1, list(zip(np.arange(cloud_size), np.arange(cloud_size))))

            o3d.visualization.draw([cld0, cld1, line])
            break

        ic(loss)
        ic(aux['z_full'].reshape(-1, 128).std(dim=0).mean())
        grad_step(loss, optimizer)

    # model_graph = draw_graph(model,
    #                          input_data=(x,),
    #                          device=device)
    # model_graph.visual_graph.render('/tmp/docker/point2vec.svg')


if __name__ == '__main__':
    main()
