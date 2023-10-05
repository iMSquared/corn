#!/usr/bin/env python3

from typing import Optional, List, Iterable
from dataclasses import dataclass, replace

import time
from pathlib import Path
import pickle
import networkx as nx
import numpy as np
import cv2
from itertools import combinations

from dt_apriltags import Detector
import open3d as o3d
from cho_util.math import transform as tx
from pkm.util.config import ConfigBase
from matplotlib import pyplot as plt

from common import T_from_Rt, draw_pose_axes
from april_tag_tracker import AprilTagTracker


class AprilTagCalibrator:

    @dataclass
    class Config(ConfigBase):
        tag_size: float = 0.1655
        K: Optional[np.ndarray] = None
        data_cache: str = '/tmp/cube_april_tag.pkl'
        offset_file: str = '/tmp/tag_from_object.pkl'
        tag_family: str = 'tag36h11'
        debug: bool = False
        rel_pose_weight: float = 100.0

    def __init__(self, cfg: Config = None,
                 **kwds):

        if cfg is None:
            cfg = AprilTagCalibrator.Config(**kwds)
        else:
            cfg = replace(cfg, **kwds)

        self.cfg=cfg

        self.detector = Detector(families=cfg.tag_family)

        # FIXME: check if this is ok.
        if cfg.K is None:
            self.__cam_param = (608.916, 608.687, 317.557, 253.814)
        else:
            K = cfg.K
            self.__cam_param = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        # Whether to take a picture of the tag in the current frame
        self.capture = False
        # Whether the transforms are technically all defined
        self.finish = False

    def collect_step(self, color_image: np.ndarray):
        """
        Detect april tags and process the transforms for one time-step (one image).

        Arg:
            color_image: The image from which to detect april tags.
        """
        cfg = self.cfg
        cv2.imshow('color', color_image)

        k = cv2.waitKey(1)

        if k == ord('c'):
            self.capture = True

        if k == ord('f'):
            self.finish = True

        if not self.capture:
            return

        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

        # Detect
        tags = self.detector.detect(gray.astype(np.uint8),
                                    True,
                                    self.__cam_param,
                                    cfg.tag_size)
        # Format
        tags_dict = {tag.tag_id: (tag.pose_R, tag.pose_t) for tag in tags}
        tags_dict['color_image'] = color_image

        if cfg.debug:
            for tag_id, tag_Rt in tags_dict.items():
                if not isinstance(tag_id, int):
                    continue
                camera_from_tag = T_from_Rt(*tag_Rt)
                draw_pose_axes(color_image, camera_from_tag,
                               K = self.cfg.K)

            # Show output image
            cv2.imshow('color', color_image)
            cv2.waitKey(1)
        return tags_dict

    def collect(self, get_image_fn,
                target_tags: Iterable[int]):
        cfg = self.cfg
        G = nx.Graph()
        done = False
        data = []
        really_done = False
        while not really_done:
            tags = self.collect_step(get_image_fn())
            if tags is None:
                continue

            # Filter tags by "usefulness"
            tags = {k: v for k, v in tags.items() if k in target_tags}

            if len(tags) < 2:
                continue

            # self.capture = False

            data.append(tags)

            for tag_A, tag_B in combinations(tags.keys(), r=2):
                G.add_edge(tag_A, tag_B)

            done = (
                G.number_of_nodes() == len(target_tags) and
                nx.number_connected_components(G) == 1
            )
            print(done)
            really_done = (done and self.finish)
            time.sleep(0.5)
        return data

    def __initialize_nodes(self, data, idx_from_tag,
                           target_tags, pose_graph):
        cfg = self.cfg
        edges = {}
        G = nx.Graph()

        for tags in data:
            for tag_A, tag_B in combinations(tags.keys(), r=2):
                idx_A = idx_from_tag[tag_A]
                idx_B = idx_from_tag[tag_B]
                cam_from_A = tags[tag_A]
                cam_from_B = tags[tag_B]

                T_CA = T_from_Rt(*cam_from_A)
                T_CB = T_from_Rt(*cam_from_B)

                T_BA = tx.invert(T_CB) @ T_CA

                edges[(idx_B, idx_A)] = T_BA
                G.add_edge(idx_B, idx_A)

        show_graph = False
        if show_graph:
            nx.draw(G)
            plt.show()

        nodes = {}
        nodes[0] = np.eye(4)

        # nodes[tag_X] = tag_0_from_tag_X
        visited = set()
        seeds = [0]
        while len(seeds) > 0:
            new_seeds = set()
            for src in seeds:
                visited.add(src)
                for dst in idx_from_tag.values():
                    if dst in visited:
                        continue

                    con = False
                    if (src, dst) in edges:
                        con = True
                        T_sd = edges[(src, dst)]
                        nodes[dst] = nodes[src] @ T_sd

                    if (dst, src) in edges:
                        con = True
                        T_ds = edges[(dst, src)]
                        T_sd = tx.invert(T_ds)
                        nodes[dst] = nodes[src] @ T_sd

                    if con:
                        new_seeds.add(dst)

            seeds = new_seeds

        # Initialize node transforms.
        for t in target_tags:
            T0 = nodes[idx_from_tag[t]]
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(T0))

    def __initialize_edges(self, data,
                           idx_from_tag, information_matrix, pose_graph):
        cfg = self.cfg
        # build edges (is this OK???)
        for tags in data:
            for tag_A, tag_B in combinations(tags.keys(), r=2):
                idx_A = idx_from_tag[tag_A]
                idx_B = idx_from_tag[tag_B]

                cam_from_A = tags[tag_A]
                cam_from_B = tags[tag_B]

                T_CA = T_from_Rt(*cam_from_A)
                T_CB = T_from_Rt(*cam_from_B)

                T_BA = tx.invert(T_CB) @ T_CA

                # tag_B_from_tag_A
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        idx_A, idx_B, T_BA, information_matrix, uncertain=True))

    def __optimize_pose_graph(self, pose_graph):
        cfg = self.cfg

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.03,
            edge_prune_threshold=0.0001,
            preference_loop_closure=0.1,
            reference_node=0)

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            crit = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
            crit.max_iteration = 256
            crit.min_residual = 0
            crit.max_iteration_lm = 64
            # crit.min_right_term = 1e-8
            # crit.min_relative_increment = 1e-8
            # crit.min_relative_residual_increment = 1e-8
            print(crit)
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                crit,
                option)

    def calibrate(self, get_image_fn, target_tags: Iterable[int]):
        cfg = self.cfg
        target_tags = list(target_tags)

        idx_from_tag = {k: i for i, k in enumerate(target_tags)}
        print(idx_from_tag)
        # tag_from_idx = {i:k for (k,i) in idx_from_tag.items()}

        # Collect of retrieve tags data from cache.
        if Path(cfg.data_cache).exists():
            with open(cfg.data_cache, 'rb') as fp:
                data = pickle.load(fp)
        else:
            data = self.collect(get_image_fn=get_image_fn,
                                target_tags=target_tags)
            with open(cfg.data_cache, 'wb') as fp:
                pickle.dump(data, fp)

        # == build pose graph ==
        information_matrix = cfg.rel_pose_weight * np.eye(6)
        pose_graph = o3d.pipelines.registration.PoseGraph()
        self.__initialize_nodes(data,
                                idx_from_tag,
                                target_tags,
                                pose_graph)
        self.__initialize_edges(data, idx_from_tag,
                                information_matrix, pose_graph)

        # == optimize pose graph ==
        if True:
            self.__optimize_pose_graph(pose_graph)

        # == store and print result ==
        output_poses = {}
        for tag in target_tags:
            idx = idx_from_tag[tag]
            pose = (pose_graph.nodes[idx].pose)
            output_poses[tag] = pose
        # print({k: np.array2string(v, separator=',') for (k,v) in output_poses})

        # nodes[tag_X] = tag_0_from_tag_X
        Ts = np.stack(output_poses.values(), axis=0)
        center = Ts[..., :3, 3].mean(axis=0)
        tag0_from_object = np.eye(4)
        tag0_from_object[:3, 3] = center

        # tag_X_from_tag_0 @ tag_0_from_object
        tag_from_object = {k: (tx.invert(v.copy()) @ tag0_from_object)
                           for k, v in output_poses.items()}

        if cfg.offset_file is not None:
            with open(cfg.offset_file, 'wb') as fp:
                pickle.dump(tag_from_object, fp)

        return output_poses


def calibrate():
    from rs_camera import RSCamera

    camera = RSCamera(RSCamera.Config())

    def get_image_fn():
        return camera.get_images()['color']

    intrinsics = camera.intrinsics
    K = np.asarray([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]])

    calibrator = AprilTagCalibrator(
        AprilTagCalibrator.Config(tag_size=0.018, K=K,
                                    # data_cache = '/tmp/spam_data.pkl',
                                    # offset_file='/tmp/tag_from_spam.pkl'

                                    data_cache='/tmp/gudu_data_3.pkl',
                                    offset_file='/tmp/tag_from_gudu_4.pkl'
                                    ))
    # target_tags = list(range(1, 25)) # cube
    target_tags = list(range(1, 11))  # spam, gudu
    # target_tags.remove(4)
    return calibrator.calibrate(get_image_fn, target_tags=target_tags)


def plot_poses(poses):
    geoms = []
    for k, v in poses.items():
        if k == 4:
            continue
        goal_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        v = np.asarray(v, dtype=np.float32)
        goal_mesh.transform(v)
        geoms.append(goal_mesh)
    o3d.visualization.draw(geoms)


def main():
    poses = calibrate()
    plot_poses(poses)


if __name__ == '__main__':
    main()
