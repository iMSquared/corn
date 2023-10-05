import isaacgym


from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass
import time
import sys
import pickle
import copy
from pathlib import Path
import copy

import threading
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
import open3d as o3d

import numpy as np
import torch
import cv2
from cho_util.math import transform as tx


import pyrealsense2 as rs
from polymetis import RobotInterface, GripperInterface
import torchcontrol as toco

from pkm.models.cloud.point_mae import subsample
from pkm.util.config import recursive_replace_map

from pkm.util.path import RunPath, ensure_directory
from pkm.util.math_util import quat_from_axa

from pkm.real.rs_camera import RSCamera
try:
    from pkm.real.track.tracker import Tracker
except ImportError:
    pass
# from icp_tracker_colored import ICPTracker
from pkm.real.track.icp_tracker_tensor import ICPTracker
from pkm.real.track.april_tag_tracker import AprilTagTracker
from pkm.real.seg.segmenter import Segmenter, MultiSegmenter


def create_np_array_from_shared_mem(
        shared_mem: SharedMemory, shared_data_dtype: np.dtype,
        shared_data_shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.frombuffer(shared_mem.buf, dtype=shared_data_dtype)
    arr = arr.reshape(shared_data_shape)
    return arr


class Perception:
    @dataclass
    class Config:
        camera: RSCamera.Config = RSCamera.Config()
        segmenter: Segmenter.Config = Segmenter.Config(
            robot_file='/tmp/franka_panda_simple/robot.urdf',
            keep_last=3
        )
        tracker_type: str = 'april'  # april or bundlesdf or none
        # offset file path for april tag tracker
        offset_file: Optional[str] = '/tmp/tag_from_cube_3.pkl'
        # offset_file: Optional[str] = '/tmp/tag_from_spam_2.pkl'
        cloud_size: int = 512
        debug: bool = False
        save_images: bool = True
        ip: Optional[str] = None

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.camera = RSCamera(cfg.camera)

        K = self.camera.K

        if cfg.ip is not None:
            self.robot = RobotInterface(
                ip_address=cfg.ip
            )

        if cfg.tracker_type == 'bundlesdf':
            self.tracker = Tracker(
                cfg_path='/home/user/workspace/BundleSDF/BundleTrack/config_ho3d.yml',
                intrinsic_mat=K)
        elif cfg.tracker_type == 'april':
            self.tracker = AprilTagTracker(tag_size=0.034,
                                           K=K,
                                           offset_file=cfg.offset_file,
                                           debug=cfg.debug
                                           )
        elif cfg.tracker_type == 'none':
            self.tracker = None
        else:
            raise ValueError(
                f"no supporting tracker type for {cfg.tracker_type}")

        self.segmenter = Segmenter(cfg.segmenter)

        self.__object_pose_cam = None
        self.__q_current = None

    def update_joint_states(self, q: np.ndarray):
        self.__q_current = q

    def get_observations(self, aux=None):
        cfg = self.cfg
        if self.__q_current is None and cfg.ip is None:
            print(self.__q_current, cfg.ip)
            return None
        elif cfg.ip is not None:
            print('ip', cfg.ip)
            self.__q_current = self.robot.get_joint_positions().numpy()
            qd_current = self.robot.get_joint_velocities().numpy()
            aux['q'] = self.__q_current.copy()
            aux['qd'] = qd_current
            # ee_pos, ee_ori = self.robot.robot_model.forward_kinematics(
            #     self.__q_current, 'panda_tool'
            # )
            # aux['ee_pos'] = ee_pos
            # aux['ee_ori'] = ee_ori

        # if self.__object_pose_cam is None:
        #     return None

        images = self.camera.get_images()

        depth_image = images['depth']
        color_image = images['color']

        T_bc = self.segmenter.T_bc
        if self.__object_pose_cam is not None:
            object_pose_base = T_bc @ self.__object_pose_cam
        else:
            object_pose_base = None

        seg_mask, pcd = self.segmenter(
            self.__q_current, depth_image, color_image,
            object_pose=object_pose_base)

        # Process segmented point cloud.
        pcd = np.asarray(pcd.points)
        pcd = subsample(
            torch.as_tensor(pcd,
                            dtype=float,
                            device='cpu'),
            # TODO  is hardcoding ok?
            cfg.cloud_size
        )

        seg_mask_u8 = seg_mask.astype(np.uint8) * 255

        if cfg.debug:
            cv2.imshow('mask', seg_mask_u8)
            cv2.imshow('color', color_image)
            cv2.waitKey(1)
        if aux is not None:
            aux['color'] = color_image.copy()
            aux['depth'] = depth_image.copy()

        if self.tracker is not None:
            object_pose_cam = self.tracker(
                color_image,
                depth_image,
                seg_mask_u8)
            if object_pose_cam is not None:
                self.__object_pose_cam = object_pose_cam
            if self.__object_pose_cam is None:
                return None
            object_pose_base = T_bc @ self.__object_pose_cam
        else:
            object_pose_cam = None
        return (pcd, object_pose_base)

    @classmethod
    def _start_inner_thread(cls,
                            cfg,
                            shared_values: Dict[str, Any],
                            lock_joint: threading.Lock,
                            lock_cloud: threading.Lock):

        instance = cls(cfg)

        while True:
            with lock_joint:
                if 'joint_state' in shared_values:
                    q_current = np.copy(shared_values['joint_state'])
                    instance.update_joint_states(q_current)
            out = instance.get_observations()
            if out is None:
                continue
            (pcd, obj_pose) = out
            with lock_cloud:
                shared_values['cloud'] = pcd
                shared_values['object_pose'] = obj_pose

    @classmethod
    def start_mt(cls, cfg):
        lock_joint = threading.Lock()
        lock_cloud = threading.Lock()
        values = {}

        thread = threading.Thread(
            target=cls._start_inner_thread,
            args=(cfg,
                  values,
                  lock_joint,
                  lock_cloud),
            daemon=True
        )
        thread.start()
        return (thread, values, lock_joint, lock_cloud, None)

    @classmethod
    def _start_inner_process(cls,
                             cfg,
                             smems: Dict[str, Any],
                             lock_joint: threading.Lock,
                             lock_cloud: threading.Lock):
        smm = SharedMemoryManager(address=('127.0.0.1', 50000), authkey=b'abc')
        smm.connect()

        instance = cls(cfg)

        schema = dict(
            joint_state=np.zeros(7, dtype=np.float32),
            cloud=np.zeros((512, 3), dtype=np.float32),
            object_pose=np.zeros((4, 4), dtype=np.float32),

            has_joint_state=np.zeros((), dtype=bool),
            has_cloud=np.zeros((), dtype=bool),
        )
        if cfg.save_images:
            schema['color'] = np.zeros(
                (cfg.camera.img_height, cfg.camera.img_width, 3),
                dtype=np.uint8)
            schema['depth'] = np.zeros(
                (cfg.camera.img_height, cfg.camera.img_width),
                dtype=np.float32)
        if cfg.tracker_type == 'none':
            schema.pop('object_pose')
        if cfg.ip is not None:
            schema['has_joint_state'] = np.ones((), dtype=bool)
            schema['joint_state'] = np.zeros(14, dtype=np.float32)
        shared_values = {
            k: create_np_array_from_shared_mem(
                smems[k], schema[k].dtype, schema[k].shape)
            for k in smems.keys()
        }
        c_t = time.time()
        o_t = c_t
        while True:
            c_t = time.time()
            if c_t - o_t > 1 / 30:
                o_t = c_t
                if cfg.ip is None:
                    with lock_joint:
                        if shared_values['has_joint_state']:
                            instance.update_joint_states(
                                shared_values['joint_state'])
                aux = None
                if cfg.save_images or cfg.ip is not None:
                    aux = {}
                out = instance.get_observations(aux=aux)
                if 'q' in aux:
                    with lock_joint:
                        shared_values['joint_state'][...] = np.concatenate(
                            [aux['q'], aux['qd']], axis=-1
                        )

                if out is None:
                    continue
                (pcd, obj_pose) = out
                with lock_cloud:
                    shared_values['cloud'][...] = pcd
                    shared_values['has_cloud'][...] = True
                    if cfg.tracker_type != 'none':
                        shared_values['object_pose'][...] = obj_pose
                    if cfg.save_images:
                        # cv2.namedWindow('color-in',cv2.WINDOW_GUI_NORMAL)
                        # cv2.imshow('color-in', aux['color'])
                        # cv2.waitKey(1)

                        shared_values['color'][...] = aux['color']
                        shared_values['depth'][...] = aux['depth']
            time.sleep(1 / 2000)

    @classmethod
    def start_mp(cls, cfg):
        # Data to be shared:
        schema = dict(
            joint_state=np.zeros(7, dtype=np.float32),
            cloud=np.zeros((512, 3), dtype=np.float32),
            object_pose=np.zeros((4, 4), dtype=np.float32),
            has_joint_state=np.zeros((), dtype=bool),
            has_cloud=np.zeros((), dtype=bool),
        )
        if cfg.save_images:
            schema['color'] = np.zeros(
                (cfg.camera.img_height, cfg.camera.img_width, 3),
                dtype=np.uint8)
            schema['depth'] = np.zeros(
                (cfg.camera.img_height, cfg.camera.img_width),
                dtype=np.float32)
        if cfg.tracker_type == 'none':
            schema.pop('object_pose')
        if cfg.ip is not None:
            schema['has_joint_state'] = np.ones((), dtype=bool)
            schema['joint_state'] = np.zeros(14, dtype=np.float32)
        smm = SharedMemoryManager(address=('127.0.0.1', 50000), authkey=b'abc')
        smm.start()

        smems = {
            k: smm.SharedMemory(v.nbytes)
            for (k, v) in schema.items()
        }
        arrs = {
            k: create_np_array_from_shared_mem(
                smems[k], schema[k].dtype, schema[k].shape)
            for k in smems.keys()
        }

        ctx = mp.get_context('spawn')
        lock_joint = ctx.Lock()
        lock_cloud = ctx.Lock()
        proc = ctx.Process(
            target=Perception._start_inner_process,
            args=(cfg, smems, lock_joint, lock_cloud),
            daemon=False
        )
        proc.start()
        return (proc, arrs, lock_joint, lock_cloud, (smm, smems))


class TestLidarPerception:
    @dataclass
    class Config:
        camera: RSCamera.Config = RSCamera.Config()
        segmenter: Segmenter.Config = Segmenter.Config(
            robot_file='/tmp/franka_panda_simple/robot.urdf',
            # keep_last=3
            device='cuda:0'
        )
        # tracker_type: str = 'april'  # april or bundlesdf or none
        tracker_type: str = 'icp'
        # offset file path for april tag tracker
        offset_file: Optional[str] = '/tmp/tag_from_cube_3.pkl'
        # offset_file: Optional[str] = '/tmp/tag_from_spam_2.pkl'
        cloud_size: int = 512
        debug: bool = False
        save_images: bool = True

        export_dir: Optional[str] = None

        ip: Optional[str] = None

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.__count = 0

        # self.camera = RSCamera(cfg.camera)
        cfg_ref = RSCamera.Config()
        # left wrt robot
        cfg_cam1 = recursive_replace_map(
            RSCamera.Config(),
            {'device_id': '101622072564'})
        cfg_cam2 = recursive_replace_map(
            RSCamera.Config(),
            {'device_id': '233622070987'})
        cfg_lidar = recursive_replace_map(
            RSCamera.Config(),
            {'device_id': 'f0172012'})

        # === multiple camera setting ===
        self.base_segmenter = Segmenter(cfg.segmenter)

        self.ctx = rs.context()
        if False:
            for dev in ['233622074125', '101622072564', '233622070987']:
                is_master = (dev == '233622074125')
                mode = 1 if is_master else 2
                for device in self.ctx.query_devices():
                    # print(device.get_info(rs.camera_info.serial_number))
                    if device.get_info(rs.camera_info.serial_number) == dev:
                        s = device.first_depth_sensor()
                        s.set_option(rs.option.global_time_enabled, 1)
                        s.set_option(rs.option.inter_cam_sync_mode, mode)
                        s.set_option(rs.option.output_trigger_enabled, 1)
                        s.set_option(rs.option.frames_queue_size, 2)

        self.ref_cam = RSCamera(cfg_ref, ctx=self.ctx, is_master=True)
        self.T_base_ref = self.base_segmenter.T_bc
        # Segmenter for the 1st camera
        self.cam1 = RSCamera(cfg_cam1, ctx=self.ctx, is_master=False)
        self.T_base_cam1 = np.asarray(
            [[-0.86639853, -0.26092146, 0.42576237, 0.20532958],
             [-0.49916346, 0.47604191, -0.72403037, 0.53421551],
             [-0.01376569, -0.83982387, -0.54268444, 0.24451794],
             [0., 0., 0., 1.]])
        self.cam1_segmenter = Segmenter(cfg.segmenter, self.T_base_cam1)

        # Segmenter for the 2nd camera
        self.cam2 = RSCamera(cfg_cam2, ctx=self.ctx, is_master=False)
        self.T_base_cam2 = np.asarray(
            [[0.78543368, -0.20161266, 0.58518912, 0.0722901],
             [-0.6167283, -0.33488763, 0.7123879, -0.4837098],
             [0.05234621, -0.92043612, -0.38737224, 0.20618753],
             [0., 0., 0., 1.]])
        self.cam2_segmenter = Segmenter(cfg.segmenter, self.T_base_cam2)

        self.cam_lidar = RSCamera(cfg_lidar, ctx=None)
        self.T_base_cam_lidar = np.asarray(
            [[0.42261142, 0.26561838, -0.86651399, 1.0719011],
             [0.90446933, -0.18452612, 0.38455869, -0.05505368],
             [-0.05774861, -0.94625419, -0.31822655, 0.38457731],
             [0., 0., 0., 1.]])

        # T_bc_list = [self.T_base_ref, self.T_base_cam1, self.T_base_cam2]
        T_bc_list = [self.T_base_cam_lidar]
        self.multi_segmenter = MultiSegmenter(cfg.segmenter, T_bc_list,
                                              #   intrinsics=[self.ref_cam.K, self.cam1.K, self.cam2.K]
                                              intrinsics=[self.cam_lidar.K]
                                              )
        # === end of multiple camera setting ===

        # K = self.camera.K
        K = self.ref_cam.K

        if cfg.ip is not None:
            self.robot = RobotInterface(
                ip_address=cfg.ip
            )

        if cfg.tracker_type == 'bundlesdf':
            self.tracker = Tracker(
                cfg_path='/home/user/workspace/BundleSDF/BundleTrack/config_ho3d.yml',
                intrinsic_mat=K)
        elif cfg.tracker_type == 'april':
            self.tracker = AprilTagTracker(tag_size=0.034,
                                           K=K,
                                           offset_file=cfg.offset_file,
                                           debug=cfg.debug
                                           )
        elif cfg.tracker_type == 'icp':
            self.tracker = ICPTracker()
        elif cfg.tracker_type == 'none':
            self.tracker = None
        else:
            raise ValueError(
                f"no supporting tracker type for {cfg.tracker_type}")

        self.segmenter = Segmenter(cfg.segmenter)

        self.__object_pose_cam = None
        self.__q_current = None
        self.__pcd_init = None
        self._count = 0

        self.__delay = 10

    def update_joint_states(self, q: np.ndarray):
        self.__q_current = q

    def get_observations(self, aux=None):
        cfg = self.cfg
        if self.__q_current is None and cfg.ip is None:
            return None
        elif cfg.ip is not None:
            self.__q_current = self.robot.get_joint_positions().numpy()
            qd_current = self.robot.get_joint_velocities().numpy()
            aux['q'] = self.__q_current.copy()
            aux['qd'] = qd_current
            # ee_pos, ee_ori = self.robot.robot_model.forward_kinematics(
            #     self.__q_current, 'panda_tool'
            # )
            # aux['ee_pos'] = ee_pos
            # aux['ee_ori'] = ee_ori

        # if self.__object_pose_cam is None:
        #     return None

        '''
        images = self.camera.get_images()

        depth_image = images['depth']
        color_image = images['color']

        T_bc = self.segmenter.T_bc
        if self.__object_pose_cam is not None:
            object_pose_base = T_bc @ self.__object_pose_cam
        else:
            object_pose_base = None

        seg_mask, pcd = self.segmenter(
            self.__q_current, depth_image, color_image,
            object_pose=object_pose_base)
        '''

        if True:
            if self.__delay > 0:
                for i in range(self.__delay):
                    images_ref = self.ref_cam.get_images()
                    images_cam1 = self.cam1.get_images()
                    images_cam2 = self.cam2.get_images()
                    images_cam_lidar = self.cam_lidar.get_images()
                    time.sleep(0.1)
                self.__delay = 0
            else:
                images_ref = self.ref_cam.get_images()
                images_cam1 = self.cam1.get_images()
                images_cam2 = self.cam2.get_images()
                images_cam_lidar = self.cam_lidar.get_images()

        else:
            frames_ref = None
            frames_cam1 = None
            frames_cam2 = None

            while (frames_ref is None or frames_cam1 is None or frames_cam2 is None):
                if frames_ref is None:
                    frames_ref = self.ref_cam.poll_images()
                    if not frames_ref.get_depth_frame().is_depth_frame():
                        frames_ref = None
                if frames_cam1 is None:
                    frames_cam1 = self.cam1.poll_images()
                    if not frames_cam1.get_depth_frame().is_depth_frame():
                        frames_cam1 = None
                if frames_cam2 is None:
                    frames_cam2 = self.cam2.poll_images()
                    if not frames_cam2.get_depth_frame().is_depth_frame():
                        frames_cam2 = None

            images_ref = self.ref_cam.proc_images(frames_ref)
            images_cam1 = self.cam1.proc_images(frames_cam1)
            images_cam2 = self.cam2.proc_images(frames_cam2)

        # Process images from reference camera
        depth_image_ref = images_ref['depth']
        color_image_ref = images_ref['color']

        # Process images from 1st camera
        depth_image_cam1 = images_cam1['depth']
        color_image_cam1 = images_cam1['color']

        # Process images from 2nd camera
        depth_image_cam2 = images_cam2['depth']
        color_image_cam2 = images_cam2['color']

        # different depth unit
        depth_image_cam_lidar = images_cam_lidar['depth'] * 0.25
        color_image_cam_lidar = images_cam_lidar['color']

        color_images = [color_image_cam_lidar]
        depth_images = [depth_image_cam_lidar]

        pcd = self.multi_segmenter(
            self.__q_current, depth_images, color_images,
            object_pose=None)
        seg_mask = None

        # ===================================
        # if self.cfg.debug:
        #     o3d.visualization.draw_geometries([pcd])
        #     time.sleep(100000)

        # Process segmented point cloud.

        pcd_points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        pcd_colored = np.concatenate((pcd_points, colors), axis=-1)

        pcd_full_colored = pcd_colored
        pcd_colored = subsample(
            torch.as_tensor(pcd_colored,
                            dtype=float,
                            device='cpu'),
            # TODO  is hardcoding ok?
            # cfg.cloud_size
            3072
        )
        pcd_colored = torch.as_tensor(pcd_colored,
                                      dtype=float,
                                      device='cpu')

        pcd = pcd_colored[..., :3]

        # Export colored point cloud
        if cfg.export_dir is not None:
            log_path = ensure_directory(cfg.export_dir)
            with open(f'{log_path}/{self.__count}.pkl', 'wb') as fp:
                # pickle.dump(pcd_full_colored, fp)
                pickle.dump({
                    'cloud': pcd_full_colored,
                    'colors': color_images,
                    'depths': depth_images,
                    'extrinsics': [self.T_base_ref, self.T_base_cam1, self.T_base_cam2],
                    'intrinsics': [self.ref_cam.K, self.cam1.K, self.cam2.K]
                }, fp)
                self.__count += 1

        if seg_mask is None:
            seg_mask_u8 = None
        else:
            seg_mask_u8 = seg_mask.astype(np.uint8) * 255

        if cfg.debug:
            # cv2.imshow('mask', seg_mask_u8)
            # cv2.imshow('color', color_image_ref)
            # cv2.waitKey(1)
            pass
        if aux is not None:
            if color_image_ref is not None:
                aux['color'] = color_image_ref.copy()
            aux['depth'] = depth_image_ref.copy()
            if color_image_cam1 is not None:
                aux['color_cam1'] = color_image_cam1.copy()
            aux['depth_cam1'] = depth_image_cam1.copy()
            if color_image_cam2 is not None:
                aux['color_cam2'] = color_image_cam2.copy()
            aux['depth_cam2'] = depth_image_cam2.copy()

        if self.tracker in ['april', 'bundlesdf']:
            object_pose_cam = self.tracker(
                color_image_ref,
                depth_image_ref,
                seg_mask_u8)
            if object_pose_cam is not None:
                self.__object_pose_cam = object_pose_cam
            if self.__object_pose_cam is None:
                return None
            object_pose_base = self.T_base_ref @ self.__object_pose_cam
        elif cfg.tracker_type == 'icp':
            # base_from_object
            # object_pose_base = self.tracker(pcd)
            object_pose_base = self.tracker(pcd_colored[..., :3],
                                            pcd_colored[..., 3:],
                                            torch.as_tensor(pcd_full_colored,
                                                            dtype=float,
                                                            device='cpu')
                                            )
            print("*********************")
            # ======= subsample here ! ==========
            pcd = subsample(pcd, cfg.cloud_size)
        else:
            object_pose_base = None
            print("-----------------------------")

        if cfg.debug:
            # Return concat of pcd+color
            return (pcd_colored, object_pose_base)
        else:
            return (pcd, object_pose_base)

    @classmethod
    def _start_inner_thread(cls,
                            cfg,
                            shared_values: Dict[str, Any],
                            lock_joint: threading.Lock,
                            lock_cloud: threading.Lock):

        instance = cls(cfg)

        while True:
            with lock_joint:
                if 'joint_state' in shared_values:
                    q_current = np.copy(shared_values['joint_state'])
                    instance.update_joint_states(q_current)
            out = instance.get_observations()
            if out is None:
                continue
            (pcd, obj_pose) = out
            with lock_cloud:
                shared_values['cloud'] = pcd
                shared_values['object_pose'] = obj_pose

    @classmethod
    def start_mt(cls, cfg):
        lock_joint = threading.Lock()
        lock_cloud = threading.Lock()
        values = {}

        thread = threading.Thread(
            target=MultiPerception._start_inner_thread,
            args=(cfg,
                  values,
                  lock_joint,
                  lock_cloud),
            daemon=True
        )
        thread.start()
        return (thread, values, lock_joint, lock_cloud, None)

    @classmethod
    def _start_inner_process(cls,
                             cfg,
                             smems: Dict[str, Any],
                             lock_joint: threading.Lock,
                             lock_cloud: threading.Lock):
        smm = SharedMemoryManager(address=('127.0.0.1', 50000), authkey=b'abc')
        smm.connect()

        instance = cls(cfg)

        schema = dict(
            joint_state=np.zeros(7, dtype=np.float32),
            cloud=np.zeros((512, 3), dtype=np.float32),
            object_pose=np.zeros((4, 4), dtype=np.float32),

            has_joint_state=np.zeros((), dtype=bool),
            has_cloud=np.zeros((), dtype=bool),
        )
        if cfg.save_images:
            schema['color'] = np.zeros(
                (cfg.camera.img_height, cfg.camera.img_width, 3),
                dtype=np.uint8)
            schema['depth'] = np.zeros(
                (cfg.camera.img_height, cfg.camera.img_width),
                dtype=np.float32)
        if cfg.tracker_type == 'none':
            schema.pop('object_pose')
        if cfg.ip is not None:
            schema['has_joint_state'] = np.ones((), dtype=bool)
            schema['joint_state'] = np.zeros(14, dtype=np.float32)
        shared_values = {
            k: create_np_array_from_shared_mem(
                smems[k], schema[k].dtype, schema[k].shape)
            for k in smems.keys()
        }
        c_t = time.time()
        o_t = c_t
        while True:
            c_t = time.time()
            if c_t - o_t > 1 / 30:
                o_t = c_t
                if cfg.ip is None:
                    with lock_joint:
                        if shared_values['has_joint_state']:
                            instance.update_joint_states(
                                shared_values['joint_state'])
                aux = None
                if cfg.save_images or cfg.ip is not None:
                    aux = {}
                out = instance.get_observations(aux=aux)
                if 'q' in aux:
                    with lock_joint:
                        shared_values['joint_state'][...] = np.concatenate(
                            [aux['q'], aux['qd']], axis=-1
                        )

                if out is None:
                    continue
                (pcd, obj_pose) = out
                with lock_cloud:
                    shared_values['cloud'][...] = pcd
                    shared_values['has_cloud'][...] = True
                    if cfg.tracker_type != 'none':
                        shared_values['object_pose'][...] = obj_pose
                    if cfg.save_images:
                        # cv2.namedWindow('color-in',cv2.WINDOW_GUI_NORMAL)
                        # cv2.imshow('color-in', aux['color'])
                        # cv2.waitKey(1)

                        shared_values['color'][...] = aux['color']
                        shared_values['depth'][...] = aux['depth']
            time.sleep(1 / 2000)

    @classmethod
    def start_mp(cls, cfg):
        # Data to be shared:
        schema = dict(
            joint_state=np.zeros(7, dtype=np.float32),
            cloud=np.zeros((512, 3), dtype=np.float32),
            object_pose=np.zeros((4, 4), dtype=np.float32),
            has_joint_state=np.zeros((), dtype=bool),
            has_cloud=np.zeros((), dtype=bool),
        )
        if cfg.save_images:
            schema['color'] = np.zeros(
                (cfg.camera.img_height, cfg.camera.img_width, 3),
                dtype=np.uint8)
            schema['depth'] = np.zeros(
                (cfg.camera.img_height, cfg.camera.img_width),
                dtype=np.float32)
        if cfg.tracker_type == 'none':
            schema.pop('object_pose')
        if cfg.ip is not None:
            schema['has_joint_state'] = np.ones((), dtype=bool)
            schema['joint_state'] = np.zeros(14, dtype=np.float32)
        smm = SharedMemoryManager(address=('127.0.0.1', 50000), authkey=b'abc')
        smm.start()

        smems = {
            k: smm.SharedMemory(v.nbytes)
            for (k, v) in schema.items()
        }
        arrs = {
            k: create_np_array_from_shared_mem(
                smems[k], schema[k].dtype, schema[k].shape)
            for k in smems.keys()
        }

        ctx = mp.get_context('spawn')
        lock_joint = ctx.Lock()
        lock_cloud = ctx.Lock()
        proc = ctx.Process(
            target=TestLidarPerception._start_inner_process,
            args=(cfg, smems, lock_joint, lock_cloud),
            daemon=False
        )
        proc.start()
        return (proc, arrs, lock_joint, lock_cloud, (smm, smems))


def test_perception(**kwds):
    kwds.setdefault('debug', True)
    kwds.setdefault('ip', "kim-MS-7C82")
    # perception = Perception(Perception.Config(debug=True))
    perception = MultiPerception(MultiPerception.Config(**kwds))
    for _ in range(16666):
        # perception.update_joint_states(
        #     np.asarray(
        #         [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0],
        #         dtype=np.float32))
        out = None
        while out is None:
            out = perception.get_observations()
            time.sleep(0.001)
        pcd_colored, pose = out
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_colored[..., :3].numpy())
        # pcd.colors =  o3d.utility.Vector3dVector(pcd_colored[..., 3:].numpy())

        # o3d.visualization.draw_geometries([pcd])
        # time.sleep(100000)
        print(pose)


if __name__ == '__main__':
    from hydra_zen import (store, zen, hydrated_dataclass)

    @store(name="test_perception")
    def main(export_dir: str):
        test_perception(export_dir=export_dir)

    store.add_to_hydra_store()
    zen(main).hydra_main(config_name='test_perception',
                         version_base='1.1',
                         config_path=None)
