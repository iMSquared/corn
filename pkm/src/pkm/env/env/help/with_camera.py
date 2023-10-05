#!/usr/bin/env python3

import time

from isaacgym import gymtorch
from isaacgym import gymapi

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Iterable
from pkm.util.config import ConfigBase

import torch as th
import numpy as np

from pkm.env.env.iface import EnvIface
# from pkm.env.common import create_camera
from pkm.util.torch_util import dcn

from gym import spaces
import nvtx


def wrap_flow_tensor(gym_tensor, offsets=None, counts=None):
    data = gym_tensor.data_ptr
    device = gym_tensor.device
    dtype = int(gym_tensor.dtype)
    shape = gym_tensor.shape
    shape = tuple(shape) + (2,)
    if offsets is None:
        offsets = tuple([0] * len(shape))
    if counts is None:
        counts = shape
    return gymtorch.wrap_tensor_impl(
        data, device, dtype, shape, offsets, counts)


class WithCamera:
    """
    Isaac Gym helper class for adding
    image-based observations to envs.
    """
    KEYS: Tuple[str, ...] = ('color', 'depth', 'label', 'flow')

    @dataclass
    class Config(ConfigBase):
        height: int = 224
        width: int = 224
        fov: float = -1.
        use_collision_geometry: bool = False
        use_color: bool = True
        use_depth: bool = True
        use_label: bool = False
        use_flow: bool = False

        use_dr: bool = False

        pos: Tuple[float, float, float] = (0.1, 0.1, 5)
        target: Tuple[int, int, int] = (0, 0, 0)
        rot: Tuple[float,...] = (0, 0, 0, 1)
        use_transform: bool = False
        device: str = 'cuda:0'

    def __init__(self, cfg: Config):
        self.cfg = cfg
        prop = gymapi.CameraProperties()
        prop.height = cfg.height
        prop.width = cfg.width
        prop.enable_tensors = True
        prop.use_collision_geometry = cfg.use_collision_geometry
        # prop.near_plane = 0.01
        # prop.far_plane = 5.0
        if cfg.fov >0.:
            prop.horizontal_fov = cfg.fov

        self.prop = prop
        print(prop)
        self.buffers: Dict[str, th.Tensor] = {}
        self.sensors = []
        self.tensors = {}

        # FIXME: reduce code duplications
        h, w = cfg.height, cfg.width

        self.use_maps = {
            'color': cfg.use_color,
            'depth': cfg.use_depth,
            'label': cfg.use_label,
            'flow': cfg.use_flow
        }
        self.space_maps = {
            'color': spaces.Box(0, 255, (h, w, 4), np.uint8),
            'depth': spaces.Box(0, np.inf, (h, w), np.float),
            'label': spaces.Box(0, np.inf, (h, w), np.int32),

            # NOTE: for now we're going to assume
            # no conversion?
            'flow': spaces.Box(0, np.inf, (h, w), np.int16)
        }
        self.observation_space = spaces.Dict({
            k: self.space_maps[k] for k in self.KEYS if self.use_maps[k]
        })

    def setup(self, env):
        """
        * load assets.
        * allocate buffers related to {scene, robot, task}.
        """
        cfg = self.cfg

        self.num_env = env.num_env
        env.gym.step_graphics(env.sim)
        for e in env.envs:
            camera, tensors = self.create_camera(env.gym, env.sim, e, env)
            self.sensors.append(camera)
            for k, v in tensors.items():
                if k not in self.tensors:
                    self.tensors[k] = []
                self.tensors[k].append(v)

        h, w = cfg.height, cfg.width
        img_shape = (h, w)

        if cfg.use_color:
            self.buffers['color'] = (
                th.empty(
                    (self.num_env,) + img_shape + (4,),
                    device=cfg.device,
                    dtype=th.uint8))
        if cfg.use_depth:
            self.buffers['depth'] = (
                th.empty(
                    (self.num_env,) + img_shape,
                    device=cfg.device,
                    dtype=th.float32))
        if cfg.use_label:
            self.buffers['label'] = (
                th.empty(
                    (self.num_env,) + img_shape,
                    device=cfg.device,
                    dtype=th.int32))
        if cfg.use_flow:
            # NOTE: 2x 16-bit signed int??
            self.buffers['flow'] = (
                th.empty(
                    (self.num_env,) + img_shape + (2,),
                    device=cfg.device,
                    dtype=th.int16))

    def create_camera(self, gym, sim, env, envs):
        cfg = self.cfg
        camera = gym.create_camera_sensor(env, self.prop)
        x, y, z = cfg.pos
        z_offset = envs.scene.table_dims[0, 2]
        x_offset = (envs.scene.table_pos[0, 0] - 
                    0.5 * envs.scene.table_dims[0, 0]-
                    envs.robot.cfg.keepout_radius)
        if cfg.use_transform:
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(x+x_offset, y, z+z_offset)
            transform.r = gymapi.Quat(*cfg.rot)
            gym.set_camera_transform(camera, env, transform)
        else:
            gym.set_camera_location(camera,
                                    env,
                                    gymapi.Vec3(x, y, z),
                                    gymapi.Vec3(*cfg.target))
        # print(np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, camera))))
        USE_MAPS = {
            'color': cfg.use_color,
            'depth': cfg.use_depth,
            'label': cfg.use_label,
            'flow': cfg.use_flow}

        TYPE_MAPS = {'color': gymapi.IMAGE_COLOR,
                     'depth': gymapi.IMAGE_DEPTH,
                     'label': gymapi.IMAGE_SEGMENTATION,
                     'flow': gymapi.IMAGE_OPTICAL_FLOW}

        tensors = {}
        for key in self.KEYS:
            if not USE_MAPS.get(key, False):
                continue
            descriptor = gym.get_camera_image_gpu_tensor(
                sim, env, camera, TYPE_MAPS[key]
            )
            if key == 'flow':
                tensor = wrap_flow_tensor(descriptor)
            else:
                tensor = gymtorch.wrap_tensor(descriptor)
            # print(key, tensor)
            # # assert (tensor is not None)
            tensors[key] = tensor
        return (camera, tensors)

    @nvtx.annotate("WithCamera.reset()")
    def reset(self, gym, env,
              env_ids: Optional[Iterable[int]] = None):
        cfg = self.cfg

        if env_ids is None:
            env_ids = range(self.num_env)

        # [0] Camera pose randomization.
        # cameras = [env.sensors['scene'][i]['camera']
        #            for i in dcn(env_ids)]
        cameras = self.sensors

        # TODO: set camera position only during
        # the first reset?

        # TODO: figure out how to handle
        # camera-position DR...
        if cfg.use_dr:
            for env_id, camera in zip(env_ids, cameras):
                # Randomize sensor camera position.
                x = cfg.table_pos[0] + np.random.uniform(-1.0, 1.0)
                y = cfg.table_pos[1] + np.random.uniform(-1.0, 1.0)
                z = cfg.table_pos[2] + 0.5 + np.random.uniform(0.0, 1.0)
                x, y, z = cfg.pos
                if cfg.use_transform:
                    transform = gymapi.Transform()
                    transform.p = gymapi.Vec3(x, y, z)
                    transform.r = gymapi.Quat(*cfg.rot)
                    gym.set_camera_transform(camera, env.envs[env_id], transform)
                else:
                    gym.set_camera_location(camera,
                                            env.envs[env_id],
                                            gymapi.Vec3(x, y, z),
                                            gymapi.Vec3(*cfg.target))

    @nvtx.annotate("WithCamera.step()")
    def step(self, env):
        cfg = self.cfg
        # t0 = (time.time())

        with nvtx.annotate("step_graphics()"):
            env.gym.step_graphics(env.sim)
           # env.gym.fetch_results(env.sim, True)

        with nvtx.annotate("render_all()"):
            env.gym.render_all_camera_sensors(env.sim)

        with nvtx.annotate("access()"):
            # Access and convert all image-related tensors.
            # TODO: the image access and related rendering utilities
            # should only be conditioned on image-based environments.
            with nvtx.annotate("start()"):
                env.gym.start_access_image_tensors(env.sim)

            if cfg.use_color:
                color_tensors = self.tensors['color']
                th.stack(color_tensors, out=self.buffers['color'])

            if cfg.use_depth:
                depth_tensors = self.tensors['depth']
                th.stack(depth_tensors, out=self.buffers['depth'])

            if cfg.use_label:
                label_tensors = self.tensors['label']
                th.stack(label_tensors, out=self.buffers['label'])

            if cfg.use_flow:
                flow_tensors = self.tensors['flow']
                th.stack(flow_tensors, out=self.buffers['flow'])

            with nvtx.annotate("end()"):
                env.gym.end_access_image_tensors(env.sim)

        # t1 = (time.time())
        # dt = t1 - t0
        # print(F'dt={dt}')
        return self.buffers
