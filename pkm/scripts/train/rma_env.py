#!/usr/bin/env python3

from typing import Tuple
from dataclasses import dataclass

import torch as th
import einops
from gym import spaces

from pkm.env.env.wrap.base import (add_obs_field,
                                   WrapperEnv,
                                   ObservationWrapper)
from pkm.env.env.wrap.popdict import PopDict
from pkm.models.common import (
    map_tensor
)
from pkm.models.cloud.point_mae import (
    subsample
)


class CombineCloud(ObservationWrapper):
    @dataclass
    class Config:
        src_keys: Tuple[str, ...] = (
            'partial_cloud',
            'partial_cloud_1',
            'partial_cloud_2')
        dst_key: str = 'partial_cloud'
        cloud_size: int = 512

    def __init__(self, cfg, env):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        obs_space = env.observation_space
        cloud_space = spaces.Box(-float('inf'),
                                 +float('inf'),
                                 shape=(cfg.cloud_size, 3))
        self._obs_space, self._update_fn = add_obs_field(
            obs_space,
            cfg.dst_key,
            cloud_space)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg
        # combine
        cloud = th.cat([obs[k] for k in cfg.src_keys],
                       dim=-2)
        # reduce
        cloud = subsample(cloud, n=cfg.cloud_size)
        # update
        obs = self._update_fn(obs, cloud)
        return obs


class PerturbCloud(ObservationWrapper):
    @dataclass
    class Config:
        noise_mag: float = 0.005
        noise_type: str = 'additive'
        key: str = 'partial_cloud'

    def __init__(self, cfg: Config, env):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        self._obs_space = env.observation_space

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg

        # Make a shallow copy
        obs = dict(obs)

        cloud = obs.pop(cfg.key)

        # sample noise
        noise = cfg.noise_mag * th.randn(
            (*cloud.shape[:-1], 3),
            dtype=cloud.dtype, device=cloud.device
        )

        # add noise
        # random gaussian noise
        if self.cfg.noise_type == 'additive':
            cloud = cloud + noise
        # noise proportional to distance
        elif self.cfg.noise_type == 'scaling':
            cloud = cloud * (1 + noise)
        else:
            raise ValueError(
                f"{cfg.noise_type} is not a proper noise type")

        obs[cfg.key] = cloud
        return obs


class PerturbGoal(ObservationWrapper):
    @dataclass
    class Config:
        noise_mag: float = 0.005
        key: str = 'goal'

    def __init__(self, cfg: Config, env):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        self._obs_space = env.observation_space

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg

        # Make a shallow copy
        obs = dict(obs)
        goal = obs.pop(cfg.key)
        goal = goal + (cfg.noise_mag * th.randn_like(goal))
        obs[cfg.key] = goal
        return obs


class AddTeacherState(WrapperEnv):
    def __init__(self,
                 env,
                 teacher,
                 state_size: int,
                 key: str = 'teacher_state',
                 ):
        super().__init__(env)
        self.__key = key
        self.teacher = teacher
        state_space = spaces.Box(-float('inf'), +float('inf'),
                                 shape=(state_size,))
        self._obs_space, self._update_fn = add_obs_field(
            env.observation_space,
            key,
            state_space
        )
        self.__memory = None
        self.teacher.eval()

    @property
    def observation_space(self):
        return self._obs_space

    def reset(self):
        obs = super().reset()
        _, state, self.__memory, done = self.teacher.init(obs)
        obs = self._update_fn(obs, state)
        return obs

    def step(self, actions: th.Tensor):
        obs, rew, done, info = super().step(actions)
        with th.inference_mode():
            state, self.__memory = self.teacher.state_net(
                self.__memory,
                actions, obs)
        obs = self._update_fn(obs, state.clone())

        # Reset `memory` where done=True.
        # we reset memory after state.clone()
        # just in case it aliases the underlying
        # memory buffer.
        with th.inference_mode():
            keep = (~done)[..., None]
            map_tensor(self.__memory,
                       lambda src, _: src.mul_(keep))
        return obs, rew, done, info


class AddTeacherAction(WrapperEnv):
    def __init__(self,
                 env,
                 teacher,
                 key: str = 'teacher_action',
                 ):
        super().__init__(env)
        self.__key = key
        self.teacher = teacher

        # mu+ls
        num_act: int = env.action_space.shape[0]
        action_spaces = spaces.Box(-float('inf'), +float('inf'),
                                   shape=(2, num_act,))
        self._obs_space, self._update_fn = add_obs_field(
            env.observation_space,
            key,
            action_spaces
        )
        self.__memory = None
        self.teacher.eval()

    @property
    def observation_space(self):
        return self._obs_space

    def reset(self):
        obs = super().reset()
        _, state, self.__memory, done = self.teacher.init(obs)

        mu, ls = self.teacher.actor_net(state)
        ls = einops.repeat(ls, '... -> n ...',
                           n=mu.shape[0])
        muls = th.stack([mu, ls], dim=-2)
        obs = self._update_fn(obs, muls)
        return obs

    def step(self, actions: th.Tensor):
        obs, rew, done, info = super().step(actions)
        with th.inference_mode():
            state, self.__memory = self.teacher.state_net(
                self.__memory,
                actions, obs)
            mu, ls = self.teacher.actor_net(state)
            ls = einops.repeat(ls, '... -> n ...',
                               n=mu.shape[0])
            # muls = th.cat([mu, ls], dim=-1)
            muls = th.stack([mu, ls], dim=-2)

        obs = self._update_fn(
                obs, muls.clone())

        # Reset `memory` where done=True.
        # we reset memory after state.clone()
        # just in case it aliases the underlying
        # memory buffer.
        with th.inference_mode():
            keep = (~done)[..., None]
            map_tensor(self.__memory,
                       lambda src, _: src.mul_(keep))
        return obs, rew, done, info


class AddStudentState(WrapperEnv):
    def __init__(self, env, student,
                 state_size: int,
                 key: str = 'student_state'):
        super().__init__(env)
        self.__key = key
        self.student = student
        state_space = spaces.Box(-float('inf'), +float('inf'),
                                 shape=(state_size,))
        self._obs_space, self._update_fn = add_obs_field(
            env.observation_space,
            key,
            state_space)
        self.__memory = None
        self.student.eval()

    @property
    def observation_space(self):
        return self._obs_space

    def reset(self):
        obs = super().reset()
        state = self.student.reset(obs)

        # also update internal data
        done = th.zeros((self.num_env,),
                        dtype=bool,
                        device=self.device)
        self.student.reset_state(done)

        obs = self._update_fn(obs, state)
        return obs

    def step(self, actions: th.Tensor):
        obs, rew, done, info = super().step(actions)

        # Run inference and add state to `obs`.
        with th.inference_mode():
            state = self.student(obs, 0, done)
        obs = self._update_fn(obs, state.clone())

        # Reset `memory` where done=True.
        # we reset memory after state.clone()
        # just in case it aliases the underlying
        # memory buffer.
        with th.inference_mode():
            self.student.reset_state(done)

        return obs, rew, done, info


def setup_rma_env_v2(cfg, env, agent,
                     state_size: int,
                     is_student: bool = False,
                     dagger: bool = False):
    if not is_student:
        if dagger:
            env = AddTeacherAction(env, agent)
        else:
            env = AddTeacherState(env, agent, state_size)
    env = CombineCloud(cfg.combine_cloud, env)
    env = PerturbCloud(cfg.perturb_cloud, env)
    env = PerturbGoal(cfg.perturb_goal, env)
    env = PopDict(env, ['icp_emb'])
    if is_student:
        env = AddStudentState(env, agent, state_size)
    return env
