#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
import pickle
import time
from pathlib import Path

import open3d as o3d
import cv2
from cho_util.math import transform as tx
from dt_apriltags import Detector
from polymetis import RobotInterface

from pkm.util.path import ensure_directory
from pkm.util.config import recursive_replace_map

from pkm.real.util import (T_from_Rt, draw_pose_axes)
from pkm.real.rs_camera import RSCamera

from rt_cfg import RuntimeConfig


def get_transform_between_cam(
        cam1: RSCamera,
        cam2: RSCamera,
        tag_family: str = 'tag36h11',
        # FIXME: super dangerous hardcoding
        tag_size: float = 0.0835,
        **kwds):
    '''
        Calculates transformation between two given camera objects
        args:
            cam1: first camera object
            cam2: second camera object
            tag_family
            tag_size: size of the april tag used for calibration

        return:
            cam2_from_cam1: transformation from cam1 to cam2

    '''
    # images = cam1.get_images()['color']

    # cv2.imshow('color', images)
    # cv2.waitKey(1000000)

    detector = Detector(families=tag_family)

    # For cam1
    color_img_1 = cam1.get_images()['color']
    gray_img_1 = cv2.cvtColor(color_img_1, cv2.COLOR_RGB2GRAY)
    # print(gray_img_1.min(), gray_img_1.max())
    # Detect
    
    # print(cam1.cam_param)
    # cv2.imshow('gray1', gray_img_1)
    # cv2.waitKey(0)
    tag1 = detector.detect(gray_img_1,
                           True,
                           cam1.cam_param,
                           tag_size)
    tag1_Rt = (tag1[0].pose_R, tag1[0].pose_t)
    cam1_from_tag = T_from_Rt(*tag1_Rt)

    # For cam2
    color_img_2 = cam2.get_images()['color']
    gray_img_2 = cv2.cvtColor(color_img_2, cv2.COLOR_RGB2GRAY)

    # Detect
    tag2 = detector.detect(gray_img_2,
                           True,
                           cam2.cam_param,
                           tag_size)
    tag2_Rt = (tag2[0].pose_R, tag2[0].pose_t)
    cam2_from_tag = T_from_Rt(*tag2_Rt)

    cam2_from_cam1 = cam2_from_tag @ tx.invert(cam1_from_tag)
    # ic(cam2_from_cam1)

    if kwds.pop('debug', False):
        draw_pose_axes(color_img_1, cam1_from_tag,
                       K=cam1.K)

        # Show output image
        cv2.imshow('color', color_img_1)
        cv2.waitKey(3000)

        draw_pose_axes(color_img_2, cam2_from_tag,
                       K=cam2.K)

        # Show output image
        cv2.imshow('color', color_img_2)
        cv2.waitKey(3000)

    return cam2_from_cam1


def test():
    rt_cfg = RuntimeConfig()
    cfg = RSCamera.Config()
    cfg_l = recursive_replace_map(
        RSCamera.Config(),
        {'device_id': rt_cfg.left_cam_device_id})
    cfg_r = recursive_replace_map(
        RSCamera.Config(),
        {'device_id': rt_cfg.right_cam_device_id})

    # Load multiple cameras
    base_camera = RSCamera(cfg)
    left_camera = RSCamera(cfg_l)
    right_camera = RSCamera(cfg_r)

    print(base_camera.K)

    for _ in range(128):
        # print(cfg.device_id)
        # images = base_camera.get_images()
        # images = left_camera.get_images()
        images = right_camera.get_images()

        cv2.imshow('color', images['color'])
        cv2.waitKey(1)


def main():
    rt_cfg = RuntimeConfig()

    cfg = RSCamera.Config(device_id=rt_cfg.table_cam_device_id,
                          poll=False)
    cfg_l = RSCamera.Config(device_id=rt_cfg.left_cam_device_id,
                            poll=False)
    cfg_r = RSCamera.Config(device_id=rt_cfg.right_cam_device_id,
                            poll=False)

    # Load multiple cameras
    ref_camera = RSCamera(cfg)
    left_camera = RSCamera(cfg_l)
    right_camera = RSCamera(cfg_r)

    with open(rt_cfg.transforms_file, 'rb') as fp:
        transforms = pickle.load(fp)
    T_bc = transforms['base_from_table_cam']
    T_bl = transforms['base_from_table_cam'] @ get_transform_between_cam(
        left_camera, ref_camera,
        tag_family=rt_cfg.tag_family,
        # tag_size=rt_cfg.tag_size
        # FIXME: super dangerous hardcoding
        tag_size=0.0835
        )
    T_br = transforms['base_from_table_cam'] @ get_transform_between_cam(
        right_camera, ref_camera,
        tag_family=rt_cfg.tag_family,
        # tag_size=rt_cfg.tag_size
        # FIXME: super dangerous hardcoding
        tag_size=0.0835
        )
        

    ensure_directory(Path(rt_cfg.extrinsics_file).parent)
    with open(rt_cfg.extrinsics_file, 'wb') as fp:
        extrinsics = {
            rt_cfg.table_cam_device_id: T_bc,
            rt_cfg.left_cam_device_id: T_bl,
            rt_cfg.right_cam_device_id: T_br
        }
        pickle.dump(extrinsics, fp)


if __name__ == '__main__':
    main()
    # test()
