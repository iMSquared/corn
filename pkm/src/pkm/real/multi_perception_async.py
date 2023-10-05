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
import torch as th
from cho_util.math import transform as tx

import pyrealsense2 as rs
from polymetis import RobotInterface

from pkm.models.cloud.point_mae import subsample
from pkm.util.path import ensure_directory

from pkm.real.rs_camera import RSCamera
from pkm.real.seg.segmenter import MultiSegmenter
# try:
#     from pkm.real.track.tracker import Tracker
# except ImportError:
#     print('Tracker disabled')
from pkm.real.track.icp_tracker_tensor import ICPTracker
# from pkm.real.track.icp_tracker_colored import ICPTracker
from pkm.real.track.april_tag_tracker import AprilTagTracker
from pkm.real.track.multicam_april_tag_tracker import (
    MulticamAprilTagTracker, update_best
)
from pkm.real.kalman_filter_v3 import KalmanFilter6D
from pkm.real.util import o3d2th, th2o3d

CAM_THREAD:bool = True

def create_np_array_from_shared_mem(
        shared_mem: SharedMemory, shared_data_dtype: np.dtype,
        shared_data_shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.frombuffer(shared_mem.buf, dtype=shared_data_dtype)
    arr = arr.reshape(shared_data_shape)
    return arr


def angle_from_dcm(R):
    cos_angle = 0.5 * (np.trace(R) - 1)
    angle = np.abs(
        np.arccos(np.clip(cos_angle, -1, +1))
    )
    return angle

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
        icp_reinit:bool=True
        segmenter: MultiSegmenter.Config = MultiSegmenter.Config(
            robot_file='/tmp/franka_panda_simple/robot.urdf',
            # keep_last=3
            device='cuda:0',
            stride=2
        )
        # tracker_type: str = 'april'  # april or bundlesdf or none
        tracker_type: str = 'multi-april'
        skip_april: bool = False
        # offset file path for april tag tracker
        tracker_tag_size: float = 0.015
        object: str = 'ceramic-cup'
        offset_file: Optional[str] = '/tmp/tag_from_cube_3.pkl'
        april_max_dist: float = 0.05
        april_max_ang: float = float(np.deg2rad(30))
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
        use_kf: bool = True

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
                            fps=cfg.fps,
                            poll=False)
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

        # table_colors
        with open(rt_cfg.table_color_file, 'rb') as fp:
            table_colors = pickle.load(fp)

        self.multi_segmenter = MultiSegmenter(
            cfg.segmenter,
            extrinsics,
            intrinsics=self.intrinsics,
            table_colors=[table_colors[dev_id] for dev_id in
                          device_ids]
        )
        # === end of multiple camera setting ===

        K = self.cams[0].K

        # == robot ==
        if cfg.ip is not None:
            self.robot = RobotInterface(
                ip_address=cfg.ip
            )

        # == tracker ==
        assert (cfg.tracker_type == 'multi-april')
        if cfg.tracker_type == 'bundlesdf':
            raise ValueError('BundleSDF disabled for now.')
            # self.tracker = Tracker(
            #     cfg_path='/home/user/workspace/BundleSDF/BundleTrack/config_ho3d.yml',
            #     intrinsic_mat=K)
        elif cfg.tracker_type == 'april':
            self.tracker = AprilTagTracker(
                tag_size=cfg.tracker_tag_size,
                K=K,
                offset_file=cfg.offset_file,
                debug=cfg.debug
            )
        elif cfg.tracker_type == 'multi-april':
            self.base_tracker = ICPTracker(mode=cfg.mode,
                                           reinit=cfg.icp_reinit)

            if False:
                self.tracker = [None for _ in range(3)]
                for i, (T, K) in enumerate(
                        zip(self.extrinsics, [c.K for c in self.cams])
                ):
                    self.tracker[i] = MulticamAprilTagTracker(
                        tag_size=cfg.tracker_tag_size,
                        april_offset_file=rt_cfg.april_offset_file(cfg.object),
                        # max_angle_between_frames=cfg.april_max_ang,
                        # max_dist_between_frames=cfg.april_max_dist,
                        max_angle_between_frames=cfg.april_max_ang,
                        max_dist_between_frames=cfg.april_max_dist,
                        debug=False,
                        extrinsics=[T],
                        intrinsics=[K],
                    )
            else:
                april_offset_file=None
                if not cfg.skip_april:
                    april_offset_file=rt_cfg.april_offset_file(cfg.object)
                self.tracker = MulticamAprilTagTracker(
                    tag_size=cfg.tracker_tag_size,
                    april_offset_file=april_offset_file,
                    max_angle_between_frames=cfg.april_max_ang,
                    max_dist_between_frames=cfg.april_max_dist,
                    debug=False,
                    extrinsics=self.extrinsics,
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
        self.__offset = None
        self.__prev_pose = None

        self.__executor = cf.ThreadPoolExecutor(max_workers=3 + 3 + 3+1)

        self.__first = True
        self.__stable_count = 0

        if CAM_THREAD:
            self.__img_locks = [threading.Lock() for _ in range(3)]
            self.__images = [None for _ in range(3)]

            if False:
                # == buffers ==
                self.cam_threads = []
                

                # == start thread ==
                for i in range(3):
                    thread = threading.Thread(target=self.get_images,
                                            args=(i,),
                                            daemon=True) 
                    thread.start()
                self.cam_threads.append(thread)
            else:
                for i in range(3):
                    self.__executor.submit(self.get_images, i)
            # wait until self.__images is populated at least once.

            done = False
            while (not done):
                done = True
                for i in range(3):
                    with self.__img_locks[i]:
                        if self.__images[i] is None:
                            done = False
                time.sleep(0.1)

    def get_images(self, index: int):
        while True:
            images = self.cams[index].get_images()
            if images is not None:
                with self.__img_locks[index]:
                    self.__images[index] = images
            time.sleep(1.0 / 60.0)

    def get_image(self, i: int):
        while True:
            images = self.cams[i].get_images()
            if images is None:
                continue
            return images

    def update_joint_states(self, q: np.ndarray):
        self.__q_current = q

    def run_joints(self):
        q_current = self.robot.get_joint_positions().numpy()
        qd_current = self.robot.get_joint_velocities().numpy()
        return (q_current, qd_current)

    def run_camera(self):
        with cf.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_img = {executor.submit(self.cams[i].get_images): i
                             for i in range(3)}
            images = [None for _ in range(3)]
            for future in cf.as_completed(future_to_img):
                index = future_to_img[future]
                images[index] = future.result()
        return images

    def get_images_and_tags_and_segmentation(self):
        APRIL_ASYNC: bool = True
        SEG_ASYNC: bool = False
        cfg = self.cfg
        # with cf.ThreadPoolExecutor(max_workers=3 + 3 + 1) as executor:
        if True:
            executor = self.__executor
            # future_to_img = {executor.submit(self.cams[i].get_images): i
            #                  for i in range(3)}
            
            images = [None for _ in range(3)]
            if CAM_THREAD:
                for i in range(3):
                    with self.__img_locks[i]:
                        images[i] = dict(self.__images[i])
            else:
                future_to_img = {executor.submit(self.get_image, i): i
                                for i in range(3)}
                for future in cf.as_completed(future_to_img):
                    index = future_to_img[future]
                    images[index] = future.result()
            
            future_to_tag = {}

            if True:
                if not cfg.skip_april:
                    future_to_tag = executor.submit(
                        self.tracker,
                        [images[i]['color'] for i in range(3)],
                        depth_images=[images[i]['depth'] for i in range(3)]
                    )
            else:
                for index in range(3):  # images:
                    # Additionally submit tag processing jobs for
                    # received images.
                    # print( type(images[index]) )
                    if not cfg.skip_april:
                        if APRIL_ASYNC:
                            future_to_tag[executor.submit(
                                self.tracker[index],
                                [images[index]['color']],
                                depth_images=[images[index]['depth']]
                            )] = index

            # All images are available now; launch segmentation
            color_images = [e['color'] for e in images]
            depth_images = [e['depth'] for e in images]
            if SEG_ASYNC:
                future_to_seg = executor.submit(self.multi_segmenter,
                                                self.__q_current,
                                                depth_images,
                                                color_images,
                                                # self.__prev_pose ?
                                                object_pose=None)
            else:
                pcd = self.multi_segmenter(self.__q_current,
                                           depth_images,
                                           color_images,
                                           object_pose=None)

            # == get apriltag results ==
            tags = [None for _ in range(3)]
            if not cfg.skip_april:
                if False:
                    if APRIL_ASYNC:
                        for future in cf.as_completed(future_to_tag):
                            index = future_to_tag[future]
                            tags[index] = future.result()
                    else:
                        tags = [self.tracker[index](
                                [images[index]['color']],
                                depth_images=[images[index]['depth']])
                                for index in range(3)]
                else:
                    tags = [future_to_tag.result()]

            # == get segmentation results
            if SEG_ASYNC:
                pcd = future_to_seg.result()

        return images, tags, pcd

    def _get_observations(self, aux=None):
        cfg = self.cfg
        if self.__q_current is None and cfg.ip is None:
            # print(self.__q_current, cfg.ip)
            return None
        elif cfg.ip is not None:
            # print(cfg.ip)
            q, qd = self.run_joints()
            self.__q_current = q
            if aux is not None:
                aux['q'] = self.__q_current.copy()
                aux['qd'] = qd

        # [1] Run camera, apriltag, and segmentation routines in parallel.
        if self.__first:
            for _ in range(3):
                images, tags, pcd = self.get_images_and_tags_and_segmentation()
                time.sleep(0.1)
            self.__first = False
        else:
            images, tags, pcd = self.get_images_and_tags_and_segmentation()
        # Process segmented point cloud.
        pcd_points = o3d2th(pcd.point.positions)
        colors = o3d2th(pcd.point.colors)
        pcd_colored = th.cat((pcd_points, colors), dim=-1)

        pcd_full_colored = pcd_colored
        pcd_colored = subsample(
            pcd_colored,
            cfg.tracker_cloud_size
        )
        pcd = pcd_colored[..., :3]

        # Populate auxiliary outputs (passthrough from camera).
        if aux is not None:
            color_keys = ['color', 'color_cam1', 'color_cam2']
            depth_keys = ['depth', 'depth_cam1', 'depth_cam2']
            for ck, dk, e in zip(color_keys, depth_keys, images):
                aux[ck] = e['color'].copy()
                aux[dk] = e['depth'].copy()

        # Select the 'best' among april tag detections.
        best_dist = float('inf')
        best_ang = float('inf')
        best_T = None
        for tag in tags:
            if tag is None:
                continue
            if self.__prev_pose is None or self.__offset is None:
                best_T = tag
                continue
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            prev = self.__prev_pose @ tx.invert(self.__offset)
            best_dist, best_ang, best_T = update_best(
                best_dist, best_ang, best_T,
                prev, tag, float('inf'),
                float('inf'))
            # print('>>', tag, best_T)
        object_pose_base_april = best_T
        # print('opba', object_pose_base_april)

        try:
            object_pose_base = self.base_tracker(
                pcd_colored[..., :3],
                pcd_colored[..., 3:],
                pcd_colored,
            )
        except RuntimeError:
            object_pose_base = self.__prev_pose

        if self.__offset is None:
            curr_pose7 = np.concatenate(
                [object_pose_base[: 3, 3],
                    tx.rotation.quaternion.from_matrix(
                        object_pose_base[: 3, : 3]),
                    np.zeros(6)],
                axis=0)
            self.kf.reset(curr_pose7)

            if object_pose_base_april is not None:
                # print('opba', object_pose_base_april)
                T_init = np.eye(4)
                T_init[..., :3, 3] = (pcd_colored[..., :3].mean(
                    dim=0)).detach().cpu().numpy()
                object_pose_base = T_init

                self.__offset = (
                    tx.invert(object_pose_base_april)
                    @ T_init
                )

                # self.__require_reset = False
                # Do not use default reinit facility;
                # rely on april tag.
                # print('--no-reinit--')
                self.base_tracker.reinit = False

        # == drift correction via apriltag ==
        r_scale = 1.0
        if object_pose_base_april is not None:
            object_pose_base = (object_pose_base_april @ self.__offset)
            r_scale = 0.1

        # == ad-hoc drift correction by annealing ==
        if False:
            com = th.median(pcd[..., :3], dim=0)[0].detach().cpu().numpy()
            object_pose_base[:3, 3] = lerp(
                object_pose_base[:3, 3], com, 0.1
            )

        # == filter by kalman ==
        if cfg.use_kf:
            curr_pose7 = np.concatenate(
                [object_pose_base[: 3, 3],
                    tx.rotation.quaternion.from_matrix(
                        object_pose_base[: 3, : 3])],
                axis=0)
            curr_pose7_kf = self.kf(curr_pose7, r_scale=r_scale)

            object_pose_base[:3, 3] = curr_pose7_kf[0:3]
            object_pose_base[:3, :3] = tx.rotation.matrix.from_quaternion(
                curr_pose7_kf[3: 7])
        self.__prev_pose = object_pose_base

        # == Overwrite tracker's reference pose ==
        entry = torch.as_tensor(object_pose_base,
                                dtype=torch.float,
                                device='cuda')
        entry = o3d.core.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(entry))
        self.base_tracker._short_history[-1] = (
            self.base_tracker._short_history[-1][0],
            entry
        )

        # ======= subsample here ! ==========
        pcd = subsample(pcd, cfg.cloud_size)

        if cfg.debug:
            # Return concat of pcd+color
            return (pcd_colored, object_pose_base)
        else:
            return (pcd, object_pose_base)

    def get_observations(self, *args, **kwds):
        if not self.cfg.skip_april:
            while self.__stable_count < 10:
                _, object_pose_base = self._get_observations(*args, **kwds)
                print(self.__stable_count)
                if angle_from_dcm(object_pose_base[:3, :3]) > np.deg2rad(10):
                    self.__stable_count = 0
                    self.__offset = None
                    continue

                self.__stable_count += 1
        return self._get_observations(*args, **kwds)

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
                    shared_values['cloud'][...] = pcd.detach().cpu().numpy()
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
