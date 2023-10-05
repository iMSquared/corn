
from typing import Dict, Tuple, Any, Optional, Iterable
from dataclasses import dataclass, replace
import time
import pickle
import concurrent.futures as cf

import threading
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
import open3d as o3d
import numpy as np
import torch

from cho_util.math import transform as tx

import pyrealsense2 as rs
from polymetis import RobotInterface

from pkm.models.cloud.point_mae import subsample
from pkm.util.path import ensure_directory

from pkm.real.rs_camera import RSCamera
from pkm.real.seg.segmenter import Segmenter, MultiSegmenter
# try:
#     from pkm.real.track.tracker import Tracker
# except ImportError:
#     print('Tracker disabled')
from pkm.real.track.icp_tracker_tensor import ICPTracker
# from pkm.real.track.icp_tracker_colored import ICPTracker
from pkm.real.track.april_tag_tracker import AprilTagTracker
from pkm.real.track.multicam_april_tag_tracker import MulticamAprilTagTracker
from pkm.real.kalman_filter_v3 import KalmanFilter6D


def pose7d_from_matrix(m):
    t = m[..., :3, 3]
    q = tx.rotation.quaternion.from_matrix(m[..., :3, :3])
    return np.concatenate([t, q], axis=-1)


def compose_pose(tq0: np.ndarray, tq1: np.ndarray):
    t0, q0 = tq0[..., 0:3], tq0[..., 3:7]
    t1, q1 = tq1[..., 0:3], tq1[..., 3:7]
    return np.concatenate([
        tx.rotation.quaternion.rotate(q0, t1) + t0,
        tx.rotation.quaternion.multiply(q0, q1)
    ], axis=-1)


def invert_pose(tq: np.ndarray) -> np.ndarray:
    out = np.empty_like(tq)
    t, q = tq[..., 0:3], tq[..., 3:7]
    out[..., 3:7] = tx.rotation.quaternion.inverse(q)
    out[..., 0:3] = -tx.rotation.quaternion.rotate(out[..., 3:7], t)
    return out


def create_np_array_from_shared_mem(
        shared_mem: SharedMemory, shared_data_dtype: np.dtype,
        shared_data_shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.frombuffer(shared_mem.buf, dtype=shared_data_dtype)
    arr = arr.reshape(shared_data_shape)
    return arr


# This is an ad-hoc alternative of Perception class that
# returns combined point clouds from multiple cameras
class MultiPerception:
    @dataclass
    class Config:
        camera: RSCamera.Config = RSCamera.Config()
        # NOTE: possible (resolution, frame rate)
        # ((640 X 480), 60), ((424 X 240), 60)
        # img_width:int = 424
        # img_height:int = 240
        img_width: int = 640
        img_height: int = 480
        fps: float = 30
        # mode:str = 'thin'
        mode: str = 'normal'
        # fps: float = 60

        segmenter: MultiSegmenter.Config = MultiSegmenter.Config(
            robot_file='/tmp/franka_panda_simple/robot.urdf',
            # keep_last=3
            device='cuda:0',
            stride=2
        )
        # tracker_type: str = 'april'  # april or bundlesdf or none
        tracker_type: str = 'icp'
        # offset file path for april tag tracker
        tracker_tag_size: float = 0.015
        object: str = 'ceramic-cup'
        offset_file: Optional[str] = '/tmp/tag_from_cube_3.pkl'
        # offset_file: Optional[str] = '/tmp/tag_from_spam_2.pkl'
        cloud_size: int = 512
        tracker_cloud_size: int = 2560
        # tracker_cloud_size: int = 512
        ip: Optional[str] = None
        try_sync: bool = False

        debug: bool = False
        save_images: bool = True
        export_dir: Optional[str] = None
        return_img_only: bool = False
        export_img_only_dir: Optional[str] = None
        period: float = (1.0 / 30.0)

        def __post_init__(self):
            self.segmenter = replace(self.segmenter, mode=self.mode)

    def __init__(self, cfg: Config,
                 device_ids: Iterable[str],
                 extrinsics: Iterable[np.ndarray],
                 rt_cfg=None):
        self.cfg = cfg
        self.__count = 0

        # self.camera = RSCamera(cfg.camera)
        cam_cfgs = [replace(cfg.camera,
                            device_id=device_id,
                            img_width=cfg.img_width,
                            img_height=cfg.img_height,
                            fps=cfg.fps)
                    for device_id in device_ids]

        # === multiple camera setting ===
        master_id = device_ids[0]
        self.ctx = rs.context()
        if cfg.try_sync:
            for dev in device_ids:
                is_master = (dev == master_id)
                mode = 1 if is_master else 2
                for device in self.ctx.query_devices():
                    # print(device.get_info(rs.camera_info.serial_number))
                    if device.get_info(rs.camera_info.serial_number) == dev:
                        s = device.first_depth_sensor()
                        s.set_option(rs.option.global_time_enabled, 1)
                        s.set_option(rs.option.inter_cam_sync_mode, mode)
                        s.set_option(rs.option.output_trigger_enabled, 1)
                        s.set_option(rs.option.frames_queue_size, 2)
        self.cams = [RSCamera(cam_cfg, ctx=self.ctx,
                              is_master=(cam_cfg.device_id == master_id))
                     for cam_cfg in cam_cfgs]
        self.extrinsics = extrinsics
        self.intrinsics = [cam.K for cam in self.cams]
        self.multi_segmenter = MultiSegmenter(
            cfg.segmenter,
            extrinsics,
            intrinsics=self.intrinsics
        )
        # === end of multiple camera setting ===

        K = self.cams[0].K

        # == robot ==
        if cfg.ip is not None:
            self.robot = RobotInterface(
                ip_address=cfg.ip
            )

        # == tracker ==
        if cfg.tracker_type == 'bundlesdf':
            raise ValueError('BundleSDF disabled for now.')
            self.tracker = Tracker(
                cfg_path='/home/user/workspace/BundleSDF/BundleTrack/config_ho3d.yml',
                intrinsic_mat=K)
        elif cfg.tracker_type == 'april':
            self.tracker = AprilTagTracker(
                tag_size=cfg.tracker_tag_size,
                K=K,
                offset_file=cfg.offset_file,
                debug=cfg.debug
            )
        elif cfg.tracker_type == 'multi-april':
            self.base_tracker = ICPTracker(mode=cfg.mode,
                                           reinit=True)
            self.tracker = MulticamAprilTagTracker(
                # rt_cfg=rt_cfg,
                tag_size=cfg.tracker_tag_size,
                # object=cfg.object,
                april_offset_file=rt_cfg.april_offset_file(cfg.object),
                max_angle_between_frames=np.deg2rad(30),
                debug=False,
                extrinsics=rt_cfg.extrinsics,
                intrinsics=[c.K for c in self.cams],
            )
            self.kf = KalmanFilter6D(
                q_pos=1e-2**2,
                q_orn=1e-2**2,
                q_vel=1e-4**2,
                q_ang_vel=1e-4**2,
                r_pos=0.05**2,
                r_orn=0.5**2)
        elif cfg.tracker_type == 'icp':
            self.tracker = ICPTracker(mode=cfg.mode)
        elif cfg.tracker_type == 'none':
            self.tracker = None
        else:
            raise ValueError(
                f"No known tracker type={cfg.tracker_type}")

        # == segmenter ==
        # self.segmenter = Segmenter(cfg.segmenter)

        self.__object_pose_cam = None
        self.__q_current = None
        self.__pcd_init = None
        self.__delay = 1
        self.__require_reset: bool = True
        self.__offset = None
        self.__prev_pose = None
        self.__step = 0

    def update_joint_states(self, q: np.ndarray):
        self.__q_current = q

    def get_observations(self, aux=None):
        cfg = self.cfg
        if self.__q_current is None and cfg.ip is None:
            # print(self.__q_current, cfg.ip)
            return None
        elif cfg.ip is not None:
            # print(cfg.ip)
            self.__q_current = self.robot.get_joint_positions().numpy()
            qd_current = self.robot.get_joint_velocities().numpy()
            if aux is not None:
                aux['q'] = self.__q_current.copy()
                aux['qd'] = qd_current
            # ee_pos, ee_ori = self.robot.robot_model.forward_kinematics(
            #     self.__q_current, 'panda_tool'
            # )
            # aux['ee_pos'] = ee_pos
            # aux['ee_ori'] = ee_ori

        # if self.__object_pose_cam is None:
        #     return None

        if True:
            if self.__delay > 0:
                for i in range(self.__delay):
                    images_ref = self.cams[0].get_images()
                    images_cam1 = self.cams[1].get_images()
                    images_cam2 = self.cams[2].get_images()
                    time.sleep(0.1)
                self.__delay = 0
            else:
                images = [None for _ in range(3)]
                for i in range(3):
                    if images[i] is not None:
                        continue
                    images[i] = self.cams[i].get_images(poll=False)
                # images_cam1 = self.cams[1].get_images(poll=True)
                # images_cam2 = self.cams[2].get_images(poll=True)
                (images_ref, images_cam1, images_cam2) = images

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

        color_images = [color_image_ref, color_image_cam1, color_image_cam2]
        depth_images = [depth_image_ref, depth_image_cam1, depth_image_cam2]

        # Return list of color images immediately
        if cfg.return_img_only:
            assert aux is not None

            aux['colors'] = color_images.copy()
            # aux['pre_rm_robot'] = seg_aux['pre_rm_robot']
            return (None, np.eye(4))

        # Export images only and immediately return, without seg/tracking
        if cfg.export_img_only_dir is not None and cfg.export_dir is None:
            log_path = ensure_directory(cfg.export_img_only_dir)
            with open(f'{log_path}/{self.__count}.pkl', 'wb') as fp:
                # pickle.dump(pcd_full_colored, fp)
                pickle.dump({
                    'joint_state': self.__q_current,
                    'colors': color_images,
                    'depths': depth_images,
                    'extrinsics': self.extrinsics,
                    'intrinsics': self.intrinsics
                }, fp)
                self.__count += 1
            return (None, np.eye(4))

        seg_aux = {}
        pcd = self.multi_segmenter(
            self.__q_current, depth_images, color_images,
            object_pose=None, aux=seg_aux)
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
            cfg.tracker_cloud_size
        )
        pcd = pcd_colored[..., :3]

        # Export colored point cloud
        if cfg.export_dir is not None:
            log_path = ensure_directory(cfg.export_dir)
            with open(f'{log_path}/{self.__count}.pkl', 'wb') as fp:
                # pickle.dump(pcd_full_colored, fp)
                pickle.dump({
                    'pre_rm_robot': seg_aux['pre_rm_robot'],
                    # 'pre_seg_color': seg_aux['pre_seg_color'],
                    'joint_state': self.__q_current,
                    'cloud': pcd_full_colored,
                    'colors': color_images,
                    'depths': depth_images,
                    'extrinsics': self.extrinsics,
                    'intrinsics': self.intrinsics
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

        if cfg.tracker_type in ['april', 'bundlesdf']:
            object_pose_cam = self.tracker(
                color_image_ref,
                depth_image_ref,
                seg_mask_u8)
            if object_pose_cam is not None:
                self.__object_pose_cam = object_pose_cam
            if self.__object_pose_cam is None:
                return None
            object_pose_base = self.extrinsics[0] @ self.__object_pose_cam
        elif cfg.tracker_type == 'multi-april':

            # if self.__offset is not None and self.__prev_pose is not None:
            #     prev_pose = self.__prev_pose @ tx.invert(self.__offset)
            # else:

            prev_pose = None

            object_pose_base_april = self.tracker([color_image_ref,
                                                   color_image_cam1,
                                                   color_image_cam2],
                                                  prev_pose=prev_pose,
                                                  # blocklist=set([0,1,2]).difference([self.__step%3])
                                                  )
            self.__step += 1
            # object_pose_base = object_pose_base_april
            if True:
                # if object_pose_base_april is not None:
                #     object_pose_base_april = pose7d_from_matrix(object_pose_base_april)
                # if object_pose_base is None:
                object_pose_base = self.base_tracker(
                    pcd_colored[..., : 3],
                    pcd_colored[..., 3:],
                    torch.as_tensor(
                        pcd_colored, dtype=float, device='cpu'))
                if self.__require_reset:
                    xx = np.concatenate(
                        [object_pose_base[: 3, 3],
                         tx.rotation.quaternion.from_matrix(
                             object_pose_base[: 3, : 3]),
                         np.zeros(6)],
                        axis=0)
                    self.kf.reset(xx)

                    if True and (object_pose_base_april is not None):
                        # world0_from_canonical = object_pose_base_april
                        # canonical_from_world0 @ world0_from_icp0
                        # self.__offset = canonical_from_icp0

                        # self.__old_april = object_pose_base_april
                        self.__offset = (
                            tx.invert(object_pose_base_april)
                            @ object_pose_base
                        )
                        # self.__offset = np.eye(4, dtype=np.float32)
                        # self.__offset = (
                        #     object_pose_base @
                        #     tx.invert(object_pose_base_april)
                        # )
                        # @ invert_pose(
                        # object_pose_base)
                        self.__require_reset = False
                        self.base_tracker.reinit = False

                # == correction via apriltag man ==
                r_scale = 1.0
                if True and (object_pose_base_april is not None):
                    # print('_APRIL_')
                    object_pose_base = (object_pose_base_april @
                                        self.__offset)
                    # object_pose_base = (self.__offset @ object_pose_base_april)
                    r_scale = 0.1

                # == filter ==
                xx = np.concatenate(
                    [object_pose_base[: 3, 3],
                     tx.rotation.quaternion.from_matrix(
                         object_pose_base[: 3, : 3])],
                    axis=0)
                xyz_euler = self.kf(xx, r_scale=r_scale)
                # print('xyz_euler', xx, xyz_euler)
                object_pose_base[:3, 3] = xyz_euler[0:3]
                object_pose_base[: 3, : 3] = tx.rotation.matrix.from_quaternion(
                    xyz_euler[3: 7])
                self.__prev_pose = object_pose_base
                # == save ==
                entry = torch.as_tensor(object_pose_base,
                                        dtype=torch.float,
                                        device='cuda')
                entry = o3d.core.Tensor.from_dlpack(
                    torch.utils.dlpack.to_dlpack(entry))
                self.base_tracker._short_history[-1] = (
                    self.base_tracker._short_history[-1][0],
                    entry
                )

        elif cfg.tracker_type == 'icp':
            object_pose_base = self.tracker(pcd_colored[..., :3],
                                            pcd_colored[..., 3:],
                                            torch.as_tensor(pcd_colored,
                                                            dtype=float,
                                                            device='cpu')
                                            )
            # print("*********************")
        else:
            object_pose_base = None
            print("-----------------------------")

        # ======= subsample here ! ==========
        pcd = subsample(pcd, cfg.cloud_size)

        if cfg.debug:
            # Return concat of pcd+color
            return (pcd_colored, object_pose_base)
        else:
            return (pcd, object_pose_base)

    @classmethod
    def _start_inner_thread(cls,
                            cfg,
                            device_ids,
                            extrinsics,
                            shared_values: Dict[str, Any],
                            lock_joint: threading.Lock,
                            lock_cloud: threading.Lock):
        instance = cls(cfg, device_ids, extrinsics)

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
    def start_mt(cls, cfg,
                 device_ids,
                 extrinsics,
                 ):
        lock_joint = threading.Lock()
        lock_cloud = threading.Lock()
        values = {}

        thread = threading.Thread(
            target=cls._start_inner_thread,
            args=(cfg,
                  device_ids,
                  extrinsics,
                  values,
                  lock_joint,
                  lock_cloud),
            daemon=True
        )
        thread.start()
        return (thread, values, lock_joint, lock_cloud, None)

    @classmethod
    def _start_inner_process(cls,
                             cfg: Config,
                             device_ids,
                             extrinsics,
                             rt_cfg,
                             smems: Dict[str, Any],
                             lock_joint: threading.Lock,
                             lock_cloud: threading.Lock):
        smm = SharedMemoryManager(address=('127.0.0.1', 50000), authkey=b'abc')
        smm.connect()

        instance = cls(cfg, device_ids, extrinsics, rt_cfg)

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
            if c_t - o_t > cfg.period:
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
    def start_mp(cls, cfg, device_ids, extrinsics, rt_cfg):
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
            target=cls._start_inner_process,
            args=(cfg, device_ids, extrinsics, rt_cfg,
                  smems, lock_joint, lock_cloud),
            daemon=False
        )
        proc.start()
        return (proc, arrs, lock_joint, lock_cloud, (smm, smems))
