#!/usr/bin/env python3

from typing import Tuple, Dict, Optional, List

from dataclasses import dataclass, fields
from functools import partial

import torch as th
import torch.nn as nn

from pkm.util.torch_util import dcn
from pkm.models.rl.net.base import FeatureBase
from pkm.models.common import (
    attention,
    MultiHeadLinear,
    grad_step,
    MLP,
    map_tensor
)
from pathlib import Path
from pkm.train.ckpt import save_ckpt, load_ckpt, last_ckpt

from pkm.util.path import ensure_directory
from pkm.models.cloud.point_mae import (
    MLPPatchEncoder,
    GroupFPSV2,
    get_pos_enc_module,
    PointMAEEncoder,
    get_group_module_v2
)
from pkm.util.config import recursive_replace_map
from pkm.env.env.wrap.normalize_env import NormalizeEnv
from icecream import ic
from pkm.train.losses import GaussianKLDivLoss


class SingleGRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 state_size: int):
        super().__init__()
        self.state_size = state_size
        self.cell = nn.GRUCell(input_size, state_size)

    def init_memory(self, batch_shape: Tuple[int, ...] = (),
                    **kwds):
        return th.zeros((*batch_shape, self.state_size), **kwds)

    def forward(self, x: th.Tensor, s: th.Tensor):
        out = self.cell(x, s)
        return (out, out)


class SingleLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 state_size: int):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, state_size)

    def init_memory(self, batch_shape: Tuple[int, ...] = (),
                    **kwds):
        h_0 = th.zeros((*batch_shape, self.state_size), **kwds)
        c_0 = th.zeros((*batch_shape, self.state_size), **kwds)
        return (h_0, c_0)

    def forward(self, x: th.Tensor, s: th.Tensor):
        out, hid = self.cell(x, s)
        return (out, hid)


class DeepGRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 state_size: int,
                 num_layer: int = 2):
        super().__init__()
        self.num_layer = num_layer
        self.state_size = state_size
        gru = nn.GRU(
            input_size=input_size,
            hidden_size=state_size,
            num_layers=num_layer,
            bias=True,
            batch_first=False,
            bidirectional=False)
        self.gru = gru

    def init_memory(self,
                    batch_shape: Tuple[int, ...] = (),
                    **kwds):
        return th.zeros(
            (self.num_layer, *batch_shape, self.state_size),
            **kwds)

    def forward(self, x: th.Tensor, s: th.Tensor):
        out, hid = self.gru(x[None], s)
        out = out.squeeze(dim=0)
        return (out, hid)

class StudentAgentRMA(nn.Module):
    @dataclass
    class StudentAgentRMAConfig(FeatureBase.Config):
        shapes: Optional[Dict[str, int]] = None
        num_query: int = 4
        # patch_size: int = 32,
        patch_size: Tuple[int, ...] = (32,)
        embed_size: int = 128
        encoder: PointMAEEncoder.Config = PointMAEEncoder.Config(
            num_hidden_layers=2
        )
        horizon: int = 1
        p_drop: float = 0.0
        rnn_arch: str = 'deep_gru'
        num_layer: int = 2
        pos_embed_type: Optional[str] = 'mlp'
        patch_type: str = 'mlp'  # mlp/knn/cnn
        batch_size: int = 4096
        state_size: int = 128
        action_size: int = 20
        # reset delay for student
        # student is reset after t ~ U(0, T) step
        max_delay_steps: int = 7
        estimate_level: str = 'state'  # or action but action is not implemented yet
        without_teacher: bool = True
        use_gpcd: bool = False
        pose_loss_coeff: float = 0.
        use_interim_goal: bool = True
        ckpt: Optional[str] = None
        pose_dim: int = 9
        use_triplet_loss: bool = False

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            p_drop = self.p_drop
            self.encoder = recursive_replace_map(self.encoder, {
                'layer.hidden_size': self.embed_size,
                'layer.attention.self_attn.attention_probs_dropout_prob': p_drop,
                'layer.attention.output.hidden_dropout_prob': p_drop,
                'layer.output.hidden_dropout_prob': p_drop
            })

    def __init__(self, cfg: StudentAgentRMAConfig,
                 writer, device):
        super().__init__()

        self.cfg = cfg
        self.writer = writer
        self.device = device
        self.input_keys = ['goal', 'hand_state',
                           'robot_state', 'previous_action']
        self.group = get_group_module_v2('fps', cfg.patch_size[0])
        if cfg.pos_embed_type is not None:
            self.pos_embed = get_pos_enc_module(cfg.pos_embed_type,
                                                cfg.embed_size)
        else:
            self.pos_embed = None
        self.patch_encoder = MLPPatchEncoder(
            MLPPatchEncoder.Config(
                pre_ln_bias=True
            ),
            patch_size=cfg.patch_size[0],
            encoder_channel=cfg.embed_size
        )
        self.encoder = PointMAEEncoder(cfg.encoder)
        if cfg.estimate_level in ['state', 'action']:
            tokenizer = {
                k: MLP((cfg.shapes[k], cfg.embed_size),
                       use_ln=True)
                for k in self.input_keys
            }
            self.tokenizer = nn.ModuleDict(tokenizer)
            self.register_parameter(
                'query_token',
                nn.Parameter(
                    th.zeros(cfg.num_query, cfg.embed_size),
                    requires_grad=True
                ))
            self.to_kv = MultiHeadLinear(cfg.embed_size,
                                         2, cfg.embed_size, unbind=True)
            self.state_size = cfg.state_size
        self.action_size = cfg.action_size

        if cfg.rnn_arch == 'deep_gru':
            agg_cls = partial(DeepGRU, num_layer=cfg.num_layer)
        elif cfg.rnn_arch == 'gru':
            # agg_cls = nn.GRUCell
            agg_cls = SingleGRU
        elif cfg.rnn_arch == 'lstm':
            agg_cls = SingleLSTM
        else:
            raise ValueError(F'Unknown rnn_arch = {cfg.rnn_arch}')
        self.aggregator = agg_cls(cfg.num_query * cfg.embed_size,
                                  self.state_size)

        if cfg.estimate_level == 'state':
            self.project = nn.Linear(self.state_size, self.state_size)
        elif cfg.estimate_level == 'action':
            self.project = nn.Linear(self.state_size,
                                     2 * self.action_size)

        self.pred_pose = nn.Linear(self.state_size, cfg.pose_dim)
        self.pose_loss = nn.MSELoss()

        if cfg.estimate_level == 'action':
            self.loss = GaussianKLDivLoss()
        else:
            if cfg.use_triplet_loss:
                self.loss = nn.TripletMarginLoss(margin=0.0)
            else:
                self.loss = nn.MSELoss()

        self.losses = 0
        self.pose_losses = 0

        self.optimizer = th.optim.Adam(
            # OK?
            self.parameters(),
            3e-4)
        self.need_goal = None

    def tokenize_cloud(self, x: th.Tensor):
        p, c = self.group(x)
        z = self.patch_encoder(p - c[..., None, :])
        pe = self.pos_embed(c)
        z = z + pe
        return z

    def reset(self, obs):
        cfg = self.cfg
        device = obs['partial_cloud'].device

        if not cfg.without_teacher:
            teacher_state = obs.get('teacher_state', None)
            if teacher_state is not None:
                teacher_state = teacher_state.detach().clone()

            teacher_action = obs.get('teacher_action', None)
            if teacher_action is not None:
                teacher_action = teacher_action.detach().clone()

            if 'neg_teacher_state' in obs:
                neg_teacher_state = obs.get(
                    'neg_teacher_state').detach().clone()

        if cfg.max_delay_steps > 0:
            self.delay_count = -th.randint(
                high=cfg.max_delay_steps,
                size=(cfg.batch_size,),
                device=device
            )

        self.memory = self.aggregator.init_memory((cfg.batch_size,),
                                                  dtype=th.float,
                                                  device=device)

        embed = self.encode(obs)
        embed = embed.reshape(*embed.shape[:-2], -1)

        # update only for the case where current timestep excced the delay
        if cfg.max_delay_steps > 0:
            step_indices = (self.delay_count >= 0).nonzero().flatten()
        else:
            step_indices = Ellipsis

        # state = self.hidden[step_indices] if cfg.rnn_arch == 'gru' \
        #     else (self.hidden[step_indices], self.cell[step_indices])
        assert (cfg.max_delay_steps <= 0)
        # memory = self.get_memory(self.memory, step_indices)
        state, self.memory = self.aggregator(embed, self.memory)
        # self.memory = self.set_memory(self.memory, memory, step_indices)

        # if isinstance(state, tuple):
        #     self.hidden[step_indices], self.cell[step_indices] = state
        # else:
        #     self.hidden[step_indices] = state

        if cfg.estimate_level == 'state':
            output = self.project(state)
            # Return hidden only for where the reset is ready for student
            # otherwise return teacher state
            if not cfg.without_teacher and cfg.max_delay_steps > 0:
                output = th.where(
                    self.delay_count[..., None] >= 0,
                    output, teacher_state
                )
        elif cfg.estimate_level == 'action':
            output = self.project(state)
            output = output.reshape(*output.shape[:-1], 2, -1)  # mu, ls
            # output = teacher_action
            if not cfg.without_teacher and cfg.max_delay_steps > 0:
                output = th.where(
                    self.delay_count[..., None, None] >= 0,
                    output,
                    teacher_action
                )

        if not cfg.use_interim_goal:
            self.need_goal = th.zeros(cfg.batch_size,
                                      dtype=bool,
                                      device=obs['partial_cloud'].device)
        return output.detach().clone()

    def encode(self, obs: Dict[str, th.Tensor],
               aux: Optional[Dict[str, th.Tensor]] = None,
               need_goal: th.Tensor = None):
        _aux = {}
        ctx_tokens = [
            self.tokenizer[k](obs[k].detach().clone())[:, None]
            for k in self.input_keys]
        # pcd_tokens = self.tokenize_cloud(obs['cloud'])

        # raise ValueError('stop')

        pcd_tokens = self.tokenize_cloud(obs['partial_cloud'])
        if self.cfg.use_gpcd:
            gpcd_tokens = self.tokenize_cloud(obs['gpcd'])
            tokens = th.cat(ctx_tokens + [pcd_tokens]
                            + [gpcd_tokens], dim=-2)
        else:
            tokens = th.cat(ctx_tokens + [pcd_tokens], dim=-2)

        if need_goal is None:
            key_padding_mask = None
        else:
            key_padding_mask = th.zeros(tokens.shape[:-1],
                                        dtype=bool,
                                        device=tokens.device)
            goal_token_pos = self.input_keys.index('goal')
            key_padding_mask[..., goal_token_pos] = ~need_goal
            key_padding_mask = key_padding_mask.reshape(
                -1, key_padding_mask.shape[-1]
            )

        out, _, _ = self.encoder(tokens)
        if self.cfg.estimate_level in ['state', 'action']:
            k, v = self.to_kv(out)
            out = attention(
                self.query_token,
                k,
                v,
                aux=_aux,
                key_padding_mask=key_padding_mask)
        return out

    def reset_state(self, done: th.Tensor):
        # reset memory
        cfg = self.cfg
        keep = (~done)[..., None]

        if cfg.rnn_arch == 'deep_gru':
            self.memory = self.memory * keep[None]
        elif cfg.rnn_arch == 'gru':
            self.memory = self.memory * keep
        else:
            self.memory = (self.memory[0] * keep, self.memory[1] * keep)

        # reset counts
        if cfg.max_delay_steps > 0:
            num_reset: int = done.sum()
            self.delay_count[done] = -th.randint(
                high=cfg.max_delay_steps,
                size=(num_reset,),
                device=self.delay_count.device
            )
        if not cfg.use_interim_goal:
            self.need_goal |= done

    def forward(self, obs, step, _, aux=None):
        # outputs (target)
        cfg = self.cfg
        # obs['goal'] = obs['goal'] * self.need_goal[..., None]

        if not cfg.without_teacher:
            teacher_state = obs.get('teacher_state', None)
            if teacher_state is not None:
                teacher_state = teacher_state.detach().clone()

            teacher_action = obs.get('teacher_action', None)
            if teacher_action is not None:
                teacher_action = teacher_action.detach().clone()

            if 'neg_teacher_state' in obs:
                neg_teacher_state = obs.get(
                    'neg_teacher_state').detach().clone()

        embed = self.encode(obs, need_goal=self.need_goal)
        embed = embed.reshape(*embed.shape[:-2], -1)
        if not cfg.use_interim_goal:
            self.need_goal.fill_(0)

        # Update only for the case where current timestep excced the delay
        if cfg.max_delay_steps > 0:
            step_indices = (self.delay_count >= 0).nonzero().flatten()
        else:
            # step_indices = th.arange(end=cfg.batch_size,
            #                          dtype=th.long,
            #                          device=obs['partial_cloud'].device)
            step_indices = Ellipsis

        assert (cfg.max_delay_steps <= 0)
        # state = self.hidden[step_indices] if cfg.rnn_arch == 'gru' \
        #     else (self.hidden[step_indices], self.cell[step_indices])
        # state = self.aggregator(embed[step_indices], state)
        state, self.memory = self.aggregator(embed, self.memory)

        # if isinstance(state, tuple):
        #     self.hidden[step_indices], self.cell[step_indices] = state
        # else:
        #     self.hidden[step_indices] = state

        output = self.project(state)

        # split action into logstd/mu
        if cfg.estimate_level == 'action':
            output = output.reshape(*output.shape[:-1], 2, -1)

        if cfg.pose_loss_coeff == 0.0:
            pose = self.pred_pose(state.detach())
        else:
            pose = self.pred_pose(state)

        if self.training:
            if cfg.estimate_level == 'state':
                if cfg.use_triplet_loss:
                    self.losses = self.losses + self.loss(
                        output[step_indices],
                        teacher_state[step_indices],
                        neg_teacher_state[step_indices])
                else:
                    self.losses = self.losses + self.loss(
                        output[step_indices], teacher_state[step_indices])
            else:
                self.losses = self.losses + self.loss(
                    # mu
                    output[step_indices][..., 0, :],
                    # ls
                    output[step_indices][..., 1, :],
                    # mu
                    teacher_action[step_indices][..., 0, :],
                    # ls
                    teacher_action[step_indices][..., 1, :],
                )

            self.pose_losses = (
                self.pose_losses +
                self.pose_loss(
                    pose[step_indices],
                    obs['goal'][step_indices].detach()))

            if (step + 1) % self.cfg.horizon == 0:
                if cfg.pose_loss_coeff == 0.0:
                    loss = self.losses
                else:
                    loss = self.losses + \
                        cfg.pose_loss_coeff * self.pose_losses
                grad_step(loss,
                          self.optimizer)
                if self.writer is not None:
                    with th.no_grad():
                        self.writer.add_scalar('loss/mse',
                                               self.losses / cfg.horizon,
                                               global_step=step)
                        self.writer.add_scalar('pose_loss/mse',
                                               self.pose_losses / cfg.horizon,
                                               global_step=step)
                self.losses = 0.
                self.pose_losses = 0.
                if cfg.rnn_arch in ['deep_gru', 'gru']:
                    self.memory.detach_()
                elif cfg.rnn_arch == 'lstm':
                    self.memory = (
                        self.memory[0].detach_(),
                        self.memory[1].detach_())
                else:
                    raise ValueError(F'Unknown `rnn_arch` = {cfg.rnn_arch}')

        if not cfg.without_teacher and cfg.max_delay_steps > 0:
            if cfg.estimate_level == 'state':
                output = th.where(
                    self.delay_count[..., None] >= 0,
                    output, teacher_state
                )
            elif cfg.estimate_level == 'action':
                output = th.where(
                    self.delay_count[..., None, None] >= 0,
                    output,
                    teacher_action
                )

        if cfg.max_delay_steps > 0:
            self.delay_count += 1
        if aux is not None:
            aux['pose'] = pose.clone()

        return output.detach().clone()

    def save(self, path: str):
        ensure_directory(Path(path).parent)
        save_ckpt(dict(self=self),
                  ckpt_file=path)

    def load(self, path: str, strict: bool = True):
        ckpt_path = last_ckpt(path)
        load_ckpt(dict(self=self),
                  ckpt_file=ckpt_path,
                  strict=strict)


# class MLPStudentAgentRMA(StudentAgentRMA):
#     def __init__(self, cfg: StudentAgentRMA.StudentAgentRMAConfig,
#                  writer):
#         super().__init__()
#         self.project = nn.Sequential(
#             nn.Linear(cfg.num_query * cfg.embed_size,
#                       self.state_size),
#             nn.ELU(),
#             nn.Linear(self.state_size, self.state_size)
#         )

#     def reset(self, obs):
#         cfg = self.cfg

#         embed = self.encode(obs)
#         embed = embed.reshape(*embed.shape[:-2], -1)

#         output = self.project(embed)
#         return output.detach().clone()

#     def forward(self, step, obs, done):
#         # outputs (target)
#         cfg = self.cfg

#         # keep = (~done)[..., None]
#         # map_tensor(self.hidden,
#         #             lambda src, _: src.mul_(keep))
#         # resets = done.nonzero(as_tuple=False).flatten()
#         if not cfg.without_teacher:
#             teacher_state = obs.pop('teacher_state').detach().clone()

#         embed = self.encode(obs)
#         embed = embed.reshape(*embed.shape[:-2], -1)

#         output = self.project(embed)

#         if self.training:
#             self.losses = self.losses + self.loss(output,
#                                                   teacher_state)
#             if (step + 1) % self.cfg.horizon == 0:
#                 grad_step(self.losses,
#                           self.optimizer)
#                 with th.no_grad():
#                     self.writer.add_scalar('loss/mse',
#                                            self.losses / cfg.horizon,
#                                            global_step=step)
#                 self.losses = 0.
#                 self.hidden = self.hidden.detach()
#                 if cfg.rnn_arch == 'lstm':
#                     self.cell = self.cell.detach()
#         return output.detach().clone()


def test_1():
    batch_size = 5
    cfg = StudentAgentRMA.StudentAgentRMAConfig(
        shapes={
            'goal': 7,
            'hand_state': 7,
            'robot_state': 14,
            'previous_action': 20
        },
        batch_size=batch_size,
        max_delay_steps=0,
        without_teacher=False,
        pose_dim=7
    )
    ic(cfg)
    student = StudentAgentRMA(cfg, None, None).to("cuda")
    ic(student)
    obs1 = {
        'goal': th.rand(batch_size, 7, device="cuda"),
        'hand_state': th.rand(batch_size, 7, device="cuda"),
        'robot_state': th.rand(batch_size, 14, device="cuda"),
        'previous_action': th.rand(batch_size, 20, device="cuda"),
        'teacher_state': th.rand(batch_size, 128, device="cuda"),
        'partial_cloud': th.rand(batch_size, 84, 3, device="cuda")

    }
    state = student.reset(obs1)
    print(state.shape)
    obs2 = {
        'goal': th.rand(batch_size, 7, device="cuda"),
        'hand_state': th.rand(batch_size, 7, device="cuda"),
        'robot_state': th.rand(batch_size, 14, device="cuda"),
        'previous_action': th.rand(batch_size, 20, device="cuda"),
        'teacher_state': th.rand(batch_size, 128, device="cuda"),
        'partial_cloud': th.rand(batch_size, 310, 3, device="cuda")
    }
    done = th.zeros(batch_size, dtype=th.bool, device="cuda")
    state2 = student(obs2, 1, done)
    print(state2.shape)


def test_deep_gru():
    B: int = 1
    D_X: int = 4
    D_S: int = 8
    N_L: int = 2

    gru_1 = DeepGRU(D_X, D_S, N_L)
    gru_2 = SingleGRU(D_X, D_S)

    x = th.zeros((B, D_X))
    h_1 = th.zeros((N_L, B, D_S))
    h_2 = th.zeros((B, D_S))

    y_1, h_1 = gru_1(x, h_1)
    y_2, h_2 = gru_2(x, h_2)
    print(h_1.shape)
    print(h_2.shape)


def main():
    test_1()
    # test_deep_gru()


if __name__ == "__main__":
    main()
