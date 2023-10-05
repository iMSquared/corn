
#!/usr/bin/env python3
from pathlib import Path
import pickle
from typing import Optional, List, Iterable, Tuple
import networkx as nx
import numpy as np
import cv2
from dt_apriltags import Detector
from itertools import combinations
import open3d as o3d
from cho_util.math import transform as tx
import concurrent.futures as cf


from pkm.real.util import (T_from_Rt, draw_pose_axes)
from pkm.real.track.jin2017.transform_fuser import fuse_transform

DEPTH_PNP:bool = True
def angle_from_dcm(R):
    cos_angle = 0.5 * (np.trace(R) - 1)
    angle = np.abs(
        np.arccos(np.clip(cos_angle, -1, +1))
    )
    return angle


def update_best(
        best_dist,
        best_ang,
        best_T,

        prev_T,
        curr_T,

        max_dist: float = 0.05,
        max_ang: float = np.deg2rad(60)
):

    delta_txn = prev_T[..., :3, 3] - curr_T[..., :3, 3]

    dist = np.linalg.norm(delta_txn)
    if dist >= max_dist:
        return (best_dist, best_ang, best_T)

    # if dist >= best_dist:
    #     return (best_dist, best_ang)

    delta_rot = (
        prev_T[..., :3, :3].T @
        curr_T[..., :3, :3])
    angle = angle_from_dcm(delta_rot)
    if angle >= max_ang:
        return (best_dist, best_ang, best_T)

    if angle >= best_ang:
        return (best_dist, best_ang, best_T)

    best_dist, best_ang, best_T = (dist, angle, curr_T)

    return best_dist, best_ang, best_T


class MulticamAprilTagTracker:
    def __init__(self,
                 # rt_cfg: 'RuntimeConfig',
                 tag_size: float = 0.0357,  # should not be hardcoded
                 april_offset_file: str = '',
                 max_angle_between_frames: float = np.deg2rad(60.0),
                 max_dist_between_frames: float = 0.05,

                 extrinsics: Tuple[np.ndarray, ...] = (),
                 intrinsics: Tuple[np.ndarray, ...] = (),
                 debug: bool = False,
                 **kwds
                 ):
        self.__tag_size = tag_size
        if True:
            self.detector = [Detector(
                families='tag36h11',
                quad_decimate=1.0,
                decode_sharpening=0.5,
                nthreads=4,
            ) for _ in range(len(intrinsics))]
        else:
            self.detector = Detector(
                families='tag36h11',
                quad_decimate=2.0,
                decode_sharpening=0.5,
                nthreads=1,
            )  # should not be hardcoded

        # Load tag-from-object offset
        # Which basically determines where the
        # object pose is relative to the mounted tag.
        self.offset_file = april_offset_file
        if april_offset_file is not None:
            if Path(self.offset_file).exists():
                with open(self.offset_file, 'rb') as fp:
                    self.tag_from_object = pickle.load(fp)
                    self.tag_from_object = {k:v.astype(np.float32) for (k,v)
                                            in self.tag_from_object.items()}
            else:
                print("April tag offset file not found!")
                # raise NotImplementedError

        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.__cam_params = [(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
                             for K in self.intrinsics]
        self.num_cameras = len(self.extrinsics)

        self.__prev_pose = None
        self.__debug = debug
        self.__max_angle_between_frames = max_angle_between_frames
        self.__max_dist_between_frames = max_dist_between_frames
        self.__executor = cf.ThreadPoolExecutor(max_workers=24)

    def _make_detections_async(self,
                         color_images: np.ndarray,
                         depth_images: np.ndarray,
                         blocklist=None):
        """
        Detect april tags and process the transforms for one time-step
        Copy-pasted from calibrate_multicam_april_tag_tacker.py

        Arg:
            color_image: The image from which to detect april tags.
        """
        # gray_images = [cv2.cvtColor(color_images[i], cv2.COLOR_RGB2GRAY) for i in range(3)]

        if True:
            # with cf.ThreadPoolExecutor(max_workers=24) as executor:
            if True:
                executor = self.__executor
                future_to_tag = {}
                for i in range(len(color_images)):
                    if (blocklist is not None) and (i in blocklist):
                        continue
                    gray = cv2.cvtColor(color_images[i],
                                        cv2.COLOR_RGB2GRAY)
                    gray = gray.astype(np.uint8)


                    # Detect april tags
                    future_to_tag[executor.submit(self.detector[i].detect,
                                    gray,
                                    False if DEPTH_PNP else True,
                                    self.__cam_params[i],
                                    self.__tag_size)] = i
                future_to_xfm = {}
                for future in cf.as_completed(future_to_tag):
                    index = future_to_tag[future]
                    tags = future.result()
                    
                        # tags_dict_local = {
                        # tag.tag_id: T_from_Rt(tag.pose_R, tag.pose_t)
                        # for tag in tags}
                    if DEPTH_PNP:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                    100,
                                    0.001)

                        for tag in tags:
                            # print(tag.corners)
                            # tag.corners = cv2.cornerSubPix(color_images[index],
                            #                  tag.corners,
                            #                  (5,5),
                            #                  (-1, -1),
                            #                  criteria)
                            future_to_xfm[executor.submit(
                                fuse_transform,
                                tag.corners,
                                depth_images[index],
                                self.intrinsics[index],
                                self.__tag_size    
                                )] = (index, tag.tag_id)
                    else:
                        for tag in tags:
                            future_to_xfm[executor.submit(
                                T_from_Rt,
                                tag.pose_R,
                                tag.pose_t,
                                )] = (index, tag.tag_id)
                        
                tags_dict = {}
                for future in cf.as_completed(future_to_xfm):
                    index, tag_id = future_to_xfm[future]
                    T = future.result()
                    if T is not None:
                        tags_dict[tag_id] = self.extrinsics[index] @ T
        else:
            tagss = [None for _ in range(3)]
            for i in range(len(color_images)):
                gray = cv2.cvtColor(color_images[i],
                    cv2.COLOR_RGB2GRAY)
                gray = gray.astype(np.uint8)

                tagss[i] = self.detector.detect(gray, 
                                    False,
                                    self.__cam_params[i],
                                    self.__tag_size)
            assert(depth_images is not None)

            future_to_xfm = {}
            # for future in cf.as_completed(future_to_tag):
            #     index = future_to_tag[future]
            #     tags = future.result()
            with cf.ThreadPoolExecutor(max_workers=24) as executor:

                for index, tags in enumerate(tagss):
                    for tag in tags:
                        future_to_xfm[executor.submit(
                            fuse_transform,
                            tag.corners,
                            depth_images[index],
                            self.intrinsics[index],
                            self.__tag_size      
                            )] = (index, tag.tag_id)
                        
                tags_dict = {}
                for future in cf.as_completed(future_to_xfm):
                    index, tag_id = future_to_xfm[future]
                    T = future.result()
                    if T is not None:
                        tags_dict[tag_id] = self.extrinsics[index] @ T
        
        return tags_dict
    

    def _make_detections(self,
                         color_images: np.ndarray,
                         depth_images: np.ndarray,
                         blocklist=None):
        """
        Detect april tags and process the transforms for one time-step
        Copy-pasted from calibrate_multicam_april_tag_tacker.py

        Arg:
            color_image: The image from which to detect april tags.
        """
        tags_dict = {}
        for i in range(len(color_images)):
            if (blocklist is not None) and (i in blocklist):
                continue
            gray = cv2.cvtColor(color_images[i], cv2.COLOR_RGB2GRAY)
            # Detect
            tags = self.detector.detect(gray.astype(np.uint8),
                                        False,
                                        self.__cam_params[i],
                                        self.__tag_size)

            if depth_images[i] is not None:
                tags_dict_local = {}
                for tag in tags:
                    try:
                        T = fuse_transform(tag.corners,
                                            depth_images[i],
                                            self.intrinsics[i],
                                            self.__tag_size)
                    except IndexError:
                        continue
                    if T is not None:
                        tags_dict_local[tag.tag_id] = T
                        # break
            else:
                # z = tag.pose_R @ [0,0,1] # << z
                # is_correct = np.dot(z, (com - tag.pose_t)) > 0
                tags_dict_local = {
                    tag.tag_id: T_from_Rt(tag.pose_R, tag.pose_t)
                    for tag in tags}

            tags = [tag for tag in tags if
                    tag.decision_margin > 20]

            # tags dict in the LOCAL frame
            # print([tag for tag in tags])
            if False:
                colors = [
                    (0, 0, 255),
                    (0, 255, 0),
                    (255, 0, 0),
                    (0, 255, 255)
                ]
                for tag in tags:
                    for j in range(4):
                        cv2.line(color_images[i],
                                 tag.corners[j].astype(np.int32),
                                 tag.corners[(j + 1) % 4].astype(np.int32),
                                 color=colors[j])

            # tags dict in the BASE frame
            tags_dict_global = {tag: self.extrinsics[i] @ tag_T
                                for tag, tag_T in tags_dict_local.items()}

            tags_dict.update(tags_dict_global)

        return tags_dict

    def __call__(self,
                 color_images: np.ndarray,
                 prev_pose=None,
                 blocklist=None,
                 depth_images: Optional[np.ndarray] = None
                 ):
        # print(color_images)
        tags_dict = self._make_detections_async(
            color_images,
            depth_images,
            blocklist=blocklist)
        # tags_dict, color_image = self.collect_step()
        # if len(tags_dict) <= 0:
        #     return None

        if prev_pose is None:
            prev_pose = self.__prev_pose

        best_dist = float('inf')
        best_ang = float('inf')

        # Convert
        curr_pose = None
        # print(tags_dict)
        # print(self.tag_from_object)
        for k, base_from_tag in tags_dict.items():
            if k not in self.tag_from_object:
                continue
            if np.isnan(base_from_tag).any():
                continue

            base_from_object = (
                base_from_tag @ self.tag_from_object[k]
            )
            # base cam
            # maybe_camera_from_object = (
            #     tx.invert(self.extrinsics[0]) @ base_from_object
            # )

            if prev_pose is None:
                # TODO : voting
                self.__prev_pose = base_from_object
                return base_from_object
            else:
                best_dist, best_ang, curr_pose = update_best(
                    best_dist, best_ang, curr_pose,
                    prev_pose,
                    base_from_object,
                    self.__max_dist_between_frames,
                    self.__max_angle_between_frames)

        # if curr_pose is not None:
        self.__prev_pose = curr_pose

        if self.__debug:
            if self.__prev_pose is not None:
                draw_pose_axes(
                    color_images[0],
                    tx.invert(
                        self.extrinsics[0]) @ self.__prev_pose,
                    K=self.intrinsics[0])
                # Show output image
                cv2.imshow('color', color_images[0])
                cv2.waitKey(1)
        return curr_pose
