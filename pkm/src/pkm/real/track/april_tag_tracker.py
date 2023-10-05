
#!/usr/bin/env python3
from pathlib import Path
import pickle
from typing import Optional, List, Iterable
import networkx as nx
import numpy as np
import cv2
from dt_apriltags import Detector
from itertools import combinations
import open3d as o3d
from cho_util.math import transform as tx

from pkm.real.util import T_from_Rt


class AprilTagTracker:
    def __init__(self,
                 tag_size: float = 0.1655,
                 K: Optional[np.ndarray] = None,
                 offset_file: str = '/tmp/perception/tag_from_object.pkl',
                 exclude_tags: Optional[Iterable[int]] = None,
                 debug: bool = False,
                 max_angle_between_frames: float = np.deg2rad(60.0)
                 ):
        self.__K = K
        self.__tag_size = tag_size
        self.detector = Detector(families='tag36h11')

        # Store tag-from-object offset
        # Which basically determines where the
        # object pose is relative to the mounted tag.
        if Path(offset_file).exists():
            with open(offset_file, 'rb') as fp:
                self.tag_from_object = pickle.load(fp)
            if exclude_tags is not None:
                for k in exclude_tags:
                    self.tag_from_object.pop(k, None)
        else:
            self.tag_from_object = {0: np.eye(4)}

        # FIXME: check if this is ok.
        if K is None:
            self.__cam_param = (608.916, 608.687, 317.557, 253.814)
        else:
            self.__cam_param = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self.__prev_pose = None
        self.__debug = debug
        self.__max_angle_between_frames = max_angle_between_frames

    def __draw(self, image, pose):
        length = 0.1
        axes = np.array(
            [[0, 0, 0],
                [length, 0, 0],
                [0, length, 0],
                [0, 0, length]],
            dtype=np.float32)
        # X: Red, Y: Green, Z: Blue
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        axes_cam = (pose @
                    np.vstack((axes.T, np.ones(4))))[:3, :]
        # Project 3D points to 2D
        axes_2d = (self.__K @ axes_cam).T
        axes_2d = axes_2d[:, :2] / axes_2d[:, 2:]
        # Draw axes
        for i in range(1, 4):
            # Draw line from origin to each axis tip
            cv2.line(image,
                     tuple(axes_2d[0].astype(int)),
                     tuple(axes_2d[i].astype(int)),
                     colors[i - 1],
                     2)
        return image
        # Show output image
        # cv2.imshow('color', color_image)

    def __call__(self, color_image: np.ndarray, *args, **kwds):
        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

        # Detect
        tags = self.detector.detect(gray.astype(np.uint8), True,
                                    self.__cam_param,
                                    self.__tag_size)

        # Format
        tags_dict = {tag.tag_id: (tag.pose_R, tag.pose_t) for tag in tags}
        if len(tags_dict) <= 0:
            return None

        # Convert
        for k, v in tags_dict.items():
            if k not in self.tag_from_object:
                continue

            camera_from_tag = T_from_Rt(*v)
            maybe_camera_from_object = (
                camera_from_tag @ self.tag_from_object[k]
            )

            # draw axes
            if self.__debug:
                vis = self.__draw(color_image.copy(), maybe_camera_from_object)
                cv2.imshow('tags', vis)
                cv2.waitKey(1)

            if self.__prev_pose is None:
                self.__prev_pose = maybe_camera_from_object
                return maybe_camera_from_object
            else:
                delta_rot = (
                    self.__prev_pose[..., :3, :3].T @
                    maybe_camera_from_object[..., :3, :3]
                )
                axa = tx.rotation.axis_angle.from_matrix(delta_rot)
                if axa.shape[-1] == 4:
                    angle = axa[..., -1]
                else:
                    angle = np.linalg.norm(axa, axis=-1)

                if np.abs(angle) < self.__max_angle_between_frames:
                    self.__prev_pose = maybe_camera_from_object
                    return maybe_camera_from_object
        return None


def main():
    from rs_camera import RSCamera
    camera = RSCamera(RSCamera.Config())
    tracker = AprilTagTracker(tag_size=0.034, K=camera.K,
                              debug= True,
offset_file=                              '/tmp/tag_from_cube_3.pkl'
           )
    for _ in range(1000):
        color_image = camera.get_images()['color']
        tracker(color_image)


if __name__ == '__main__':
    main()
