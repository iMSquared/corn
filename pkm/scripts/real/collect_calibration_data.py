#!/usr/bin/env python3

import numpy as np
import pickle
import time
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from pathlib import Path
from tqdm.auto import tqdm
import cv2

from dt_apriltags import Detector
import pyrealsense2 as rs

from polymetis import RobotInterface

from pkm.util.path import ensure_directory
from pkm.real.rs_camera import RSCamera

from rt_cfg import RuntimeConfig


@dataclass
class Config:
    # hand_cam_id: str = '234322070242'
    hand_cam_id: str = '233622074736'
    hand_cam_fps: int = 60
    hand_cam_width: int = 424
    hand_cam_height: int = 240

    table_cam_id: str = '233622074125'
    table_cam_fps: int = 60
    table_cam_width: int = 640
    table_cam_height: int = 480

    tag_family: str = 'tag36h11'

    num_segments: int = 2
    move_delay: float = 2.0
    wait_delay: float = 3.0

    robot_ip: str = 'kim-MS-7C82'


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    return gray
    # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # return gray


def get_rgb_depth(pipeline):
    frames_1 = pipeline.wait_for_frames()
    align_to = rs.stream.color
    align = rs.align(align_to)
    frames_1 = align.process(frames_1)
    color_frame_1 = frames_1.get_color_frame()
    depth_frame_1 = frames_1.get_depth_frame()
    color_image_1 = np.asanyarray(color_frame_1.get_data())
    depth_image_1 = np.asanyarray(depth_frame_1.get_data()) / 1000.
    return color_image_1, depth_image_1


def get_camera_pose(at_detector, rgb, cam_param):
    gray = rgb2gray(rgb)
    tags = at_detector.detect(gray.astype(np.uint8), True, cam_param, 0.0835)
    tags_dict = {tag.tag_id: (tag.pose_R, tag.pose_t) for tag in tags}

    if len(tags_dict) != 1:
        return None, None, False

    camera_from_tag_rotation = tags_dict[0][0]
    camera_from_tag_translation = tags_dict[0][1].reshape(-1)
    camera_from_tag_quaternion = R.from_matrix(
        camera_from_tag_rotation).as_quat()
    return (camera_from_tag_translation, camera_from_tag_quaternion, True)


def get_cam_ee_poses(robot, at_detector, pipelines,
                     camera_param_1,
                     camera_param_2):
    # robot end-effector extrinsics
    ee_pos, ee_quat = robot.pose_ee()

    # 1st camera extrinsics
    color_image_1, depth_image_1 = get_rgb_depth(pipelines[0])
    
    cv2.namedWindow('color1', cv2.WINDOW_NORMAL)
    cv2.imshow('color1', color_image_1)
    cv2.waitKey(1)

    cam1_pos, cam1_quat, valid1 = get_camera_pose(
        at_detector,
        color_image_1,
        camera_param_1)
    if not valid1:
        return None
    cam1_R = R.from_quat(cam1_quat)
    T_a1 = np.identity(4)
    T_a1[:3, :3] = cam1_R.as_matrix()
    T_a1[:3, 3] = cam1_pos

    # 2nd camera extrinsics
    color_image_2, depth_image_2 = get_rgb_depth(pipelines[1])

    cam2_pos, cam2_quat, valid2 = get_camera_pose(
        at_detector, color_image_2, camera_param_2)
    if not valid2:
        return None
    cam2_R = R.from_quat(cam2_quat)
    T_a2 = np.identity(4)
    T_a2[:3, :3] = cam2_R.as_matrix()
    T_a2[:3, 3] = cam2_pos

    if not (valid1 and valid2):
        return None

    return (
        np.asarray(ee_pos),
        np.asarray(ee_quat),

        np.asarray(cam1_pos),
        np.asarray(cam1_quat),

        np.asarray(cam2_pos),
        np.asarray(cam2_quat),
        (valid1 and valid2)
    )


def main():
    # Configure.
    cfg: Config = Config()
    rt_cfg = RuntimeConfig()
    at_detector = Detector(families=cfg.tag_family,
                           quad_decimate=1)

    # Hand camera
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device(cfg.hand_cam_id)
    config_1.enable_stream(rs.stream.depth,
                           cfg.hand_cam_width,
                           cfg.hand_cam_height,
                           rs.format.z16,
                           cfg.hand_cam_fps)
    config_1.enable_stream(rs.stream.color,
                           cfg.hand_cam_width,
                           cfg.hand_cam_height,
                           rs.format.bgr8,
                           cfg.hand_cam_fps)

    # far camera
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device(cfg.table_cam_id)
    config_2.enable_stream(rs.stream.depth,
                           cfg.table_cam_width,
                           cfg.table_cam_height,
                           rs.format.z16,
                           cfg.table_cam_fps)
    config_2.enable_stream(rs.stream.color,
                           cfg.table_cam_width,
                           cfg.table_cam_height,
                           rs.format.bgr8,
                           cfg.table_cam_fps)

    # Start streaming from both cameras
    cfg_1 = pipeline_1.start(config_1)
    cfg_2 = pipeline_2.start(config_2)

    pipelines = [pipeline_1, pipeline_2]

    # Fetch stream 1 profile & intrinsics for color stream
    profile_1 = cfg_1.get_stream(rs.stream.color)
    intr_1 = profile_1.as_video_stream_profile().get_intrinsics()
    camera_param_1 = (intr_1.fx, intr_1.fy, intr_1.ppx, intr_1.ppy)

    # Fetch stream 2 profile & intrinsics for color stream
    profile_2 = cfg_2.get_stream(rs.stream.color)
    intr_2 = profile_2.as_video_stream_profile().get_intrinsics()
    camera_param_2 = (intr_2.fx, intr_2.fy, intr_2.ppx, intr_2.ppy)

    align_to = rs.stream.color
    rs.align(align_to)

    # Camera 1
    color_image_1, depth_image_1 = get_rgb_depth(pipelines[0])
    color_image_2, depth_image_2 = get_rgb_depth(pipelines[1])

    if False:
        pqs = np.array([
            # [0.1 + 0.4999, +0.0809, 0.5431-0.1, -0.7295, 0.6433, 0.0069, -0.2322],
            # [0.1 + 0.2864, +0.0344, 0.4827-0.1, -0.8524, 0.4876, 0.1839, -0.0423],
            # [0.1 + 0.4733, +0.1887, 0.3829-0.1, -0.9581, 0.1054, -0.0133, -0.2662],
            # [0.1 + 0.4555, -0.2514, 0.3062-0.1, -0.9433, 0.2720, -0.0842, 0.1709],
            # [0.1 + 0.5533, +0.0711, 0.4472-0.1, -0.8143, 0.5108, -0.0560, -0.2699],
            # [0.1 + 0.2827, +0.1715, 0.4474-0.1, -0.9568, 0.1676, 0.1861, -0.1474],
            # [0.1 + 0.4105, -0.2035, 0.4603-0.1, -0.9632, 0.2495, 0.0238, 0.0975],
            # [0.1 + 0.3027, -0.0190, 0.6465-0.1, -0.9695, -0.0309, 0.2352, -0.0609],

            [0.4999, -0.0809, 0.5431-0.1, 0.7295, 0.6433, 0.0069, -0.2322],
            [0.2864, -0.0344, 0.4827-0.1, 0.8524, 0.4876, 0.1839, -0.0423],
            [0.4733, -0.1887, 0.3829-0.1, 0.9581, 0.1054, -0.0133, -0.2662],
            [0.5533, -0.0711, 0.4472-0.1, 0.8143, 0.5108, -0.0560, -0.2699],
            [0.2827, -0.1715, 0.4474-0.1, 0.9568, 0.1676, 0.1861, -0.1474],

            [0.4555, +0.2514, 0.3062-0.1, 0.9433, 0.2720, -0.0842, 0.1709],
            [0.4105, +0.2035, 0.4603-0.1, 0.9632, 0.2495, 0.0238, 0.0975],
            [0.3027, +0.0190, 0.6465-0.1, 0.9695, -0.0309, 0.2352, -0.0609]

        ])
    else:
        pqs = np.array([[ 4.5448e-01, -7.3933e-02,  4.4853e-01, -9.3632e-01, -3.4630e-01,
          4.7871e-03,  5.7977e-02],
        [ 5.5774e-01, -5.2052e-02,  3.9902e-01, -9.0410e-01,  4.1635e-01,
         -6.8642e-02,  6.7411e-02],
        [ 4.3137e-01,  1.5570e-02,  3.2997e-01, -9.4590e-01,  2.7188e-01,
         -6.2418e-02,  1.6570e-01],
        [ 4.7878e-01, -3.6743e-02,  4.9835e-01,  9.5014e-01, -2.8621e-01,
          1.2279e-01,  1.5665e-02],
        [ 5.1438e-01,  8.6358e-02,  3.2250e-01, -8.5754e-01,  5.1411e-01,
         -1.7783e-02,  2.8096e-03],
        [ 4.1574e-01,  3.3657e-03,  3.0532e-01, -9.0914e-01,  4.0072e-01,
          2.4846e-02,  1.1078e-01],
        [ 5.6391e-01, -8.6524e-02,  3.6547e-01, -9.8304e-01,  1.0314e-01,
          9.6831e-02,  1.1670e-01],
        [ 4.7830e-01, -5.3533e-02,  3.1095e-01, -7.6190e-01,  6.0918e-01,
         -1.0195e-01,  1.9501e-01],
        [ 5.1071e-01,  4.7827e-02,  4.4776e-01,  9.7461e-01,  2.0473e-01,
          2.5998e-02,  8.6842e-02],
        [ 5.5576e-01, -5.1598e-02,  3.3257e-01,  7.6714e-01, -6.1905e-01,
         -1.6380e-01,  3.8013e-02],
        [ 5.6430e-01, -9.2989e-02,  4.8695e-01,  9.3104e-01,  3.6489e-01,
          4.2924e-03,  8.1399e-06],
        [ 5.5721e-01,  1.5400e-02,  3.5114e-01,  9.4590e-01,  3.0888e-01,
         -9.6162e-02,  2.4924e-02],
        [ 5.4888e-01, -7.2319e-02,  3.6730e-01, -9.0482e-01,  3.5873e-01,
          2.2852e-01,  1.9564e-02],
        [ 4.9721e-01,  7.7352e-02,  4.9563e-01,  9.7838e-01, -7.8229e-02,
          1.0166e-01,  1.6221e-01],
        [ 4.1052e-01, -9.4584e-02,  3.7305e-01, -9.9454e-01, -4.4068e-02,
         -9.3139e-02,  1.6769e-02],
        [ 5.7408e-01,  7.6109e-02,  3.2593e-01,  8.8270e-01, -4.6387e-01,
         -7.1047e-02,  2.5074e-02],
        [ 4.5593e-01,  7.4321e-02,  4.5879e-01,  9.2987e-01,  2.7575e-01,
          1.4605e-02,  2.4307e-01],
        [ 4.9195e-01, -6.0578e-02,  4.8696e-01, -9.8833e-01, -1.3417e-01,
         -3.5563e-02,  6.2691e-02],
        [ 4.7033e-01, -8.2023e-02,  3.8877e-01, -9.1455e-01, -3.2490e-01,
         -1.5283e-01,  1.8623e-01],
        [ 4.3898e-01, -5.1771e-02,  4.0955e-01,  9.9689e-01,  7.3944e-02,
          6.6221e-03,  2.6447e-02],
        [ 4.1924e-01, -2.6053e-02,  4.5599e-01, -9.6842e-01, -2.4322e-01,
          5.3980e-02,  9.1184e-03]])
    # generate_lookat_pqs()

    robot = RobotInterface(
        enforce_version=False,
        ip_address=cfg.robot_ip,
    )

    # == INITIALIZE ==
    print('Initialize robot.')
    q_home = np.asarray(
        [-0.0122, -0.1095, 0.0562, -2.5737, -0.0196, 2.4479, 0.0])
    robot.move_to_joint_positions(q_home)

    init_pos = pqs[0, :3]
    init_quat = pqs[0, 3:]
    robot.move_to_ee_pose(
        position=init_pos,
        orientation=init_quat,
        time_to_go=cfg.move_delay
    )

    # == COLLET DATA ==
    print('Start collecting data.')
    pq_list = []
    for i in tqdm(range(1, pqs.shape[0])):
        pos_prev = pqs[i - 1, :3]
        pos_curr = pqs[i, :3]
        slerp = Slerp([0, 1], R.from_quat(pqs[i - 1:i + 1, 3:]))

        for j in range(cfg.num_segments):
            # Interpolate to target.
            ratio = j / cfg.num_segments
            target_pos = pos_prev * (1. - ratio) + pos_curr * ratio
            target_quat = slerp(ratio)
            target_quat = target_quat.as_quat()

            # Move and wait.
            robot.move_to_ee_pose(
                position=target_pos,
                orientation=target_quat,
                time_to_go=cfg.move_delay)
            time.sleep(cfg.wait_delay)

            # Get camera poses.
            out = get_cam_ee_poses(robot, at_detector, pipelines,
                                   camera_param_1,
                                   camera_param_2)

            # Parse output.
            if out is None:
                continue
            (ee_pos, ee_quat,
             cam1_pos, cam1_quat,
             cam2_pos, cam2_quat,
             valid) = out

            # Save output.
            if valid:
                pq_list.append(
                    (ee_pos, ee_quat,
                     cam1_pos, cam1_quat,
                     cam2_pos, cam2_quat))

    # == EXPORT DATA ==
    print('Export data.')
    ensure_directory(Path(rt_cfg.calib_data_file).parent)
    with open(rt_cfg.calib_data_file, 'wb') as f:
        pickle.dump(pq_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
