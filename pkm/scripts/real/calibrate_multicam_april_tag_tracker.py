#!/usr/bin/env python3

from typing import Optional, List, Iterable, Tuple
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

# from common import T_from_Rt, draw_pose_axes
from pkm.real.util import (T_from_Rt, draw_pose_axes)
from pkm.real.multi_perception_async import MultiPerception
from pkm.util.vis.win_o3d import AutoWindow

from rt_cfg import RuntimeConfig
from icecream import ic
from pkm.real.rs_camera import RSCamera
from pkm.real.track.jin2017.transform_fuser import fuse_transform

from hydra_zen import (store, zen, hydrated_dataclass)

DEPTH_PNP: bool = True


class MultiAprilTagCalibrator:
    '''
        Generates offset_file, which is the list of
        transformations of each tags from the object
        center
    '''
    @dataclass
    class Config(ConfigBase):
        tag_size: float = 0.015
        rt_cfg = RuntimeConfig()
        tag_family: str = 'tag36h11'
        set_dummy_joints: bool = True
        debug: bool = False
        rel_pose_weight: float = 100.0
        object: str = 'ceramic-cup'
        data_cache: str = F'/tmp/{object}.pkl'

        def __post_init__(self):
            self.data_cache = F'/tmp/{self.object}.pkl'

    def __init__(self, cfg: Config = None,
                 object: str = None,
                 **kwds):
        if cfg is None:
            cfg = MultiAprilTagCalibrator.Config(**kwds)
        else:
            cfg = replace(cfg, **kwds)

        self.cfg = cfg

        # perception stack
        kwds.setdefault('return_img_only', True)
        self.cams = [
            RSCamera(RSCamera.Config(device_id=dev_id,
                                     poll=False))
            for dev_id in cfg.rt_cfg.cam_ids
        ]

        self.detector = Detector(families=cfg.tag_family)

        self.extrinsics = cfg.rt_cfg.extrinsics
        self.intrinsics = [cam.K for cam in self.cams]

        self.__cam_params = [(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
                             for K in self.intrinsics]
        self.num_cameras = len(self.extrinsics)
        self.object = object
        self.offset_file = cfg.rt_cfg.april_offset_file(cfg.object)
        self.finish = False

    def collect_step(self):
        """
        Detect april tags and process the transforms for one time-step

        Arg:
            color_image: The image from which to detect april tags.
        """
        cfg = self.cfg

        aux = {}

        images = [cam.get_images() for cam in self.cams]
        color_images = [img['color'] for img in images]
        depth_images = [img['depth'] for img in images]

        tags_dict = {}
        for i in range(self.num_cameras):
            gray = cv2.cvtColor(color_images[i], cv2.COLOR_RGB2GRAY)

            use_dpnp = (DEPTH_PNP and (depth_images[i] is not None))
            # Detect
            tags = self.detector.detect(gray.astype(np.uint8),
                                        (not use_dpnp),
                                        self.__cam_params[i],
                                        cfg.tag_size)

            # FIXME: consider reviving
            # thresholding logic
            # tags = [tag for tag in tags if
            #        tag.decision_margin > 30]

            # tags dict in the LOCAL frame
            tags_dict_local = {}
            if use_dpnp:
                for tag in tags:
                    try:
                        T = fuse_transform(tag.corners,
                                           depth_images[i],
                                           self.intrinsics[i],
                                           cfg.tag_size)
                    except IndexError as e:
                        print(e)
                        continue
                    if T is not None:
                        tags_dict_local[tag.tag_id] = T
            else:
                tags_dict_local = {
                    tag.tag_id: T_from_Rt(tag.pose_R, tag.pose_t)
                    for tag in tags}

            # tags dict in the BASE frame
            tags_dict_global = {tag: self.extrinsics[i] @ tag_T
                                for tag, tag_T in tags_dict_local.items()}
            tags_dict.update(tags_dict_global)

        if cfg.debug:
            # win = AutoWindow()
            # vis = win.vis
            # if aux['pre_rm_robot'] is not None:
            #     pcd = aux['pre_rm_robot'].cpu().to_legacy()
            #     vis.add_geometry('cloud', pcd, color=(1,1,1,1))

            for tag_id, tag_T in tags_dict.items():
                # if not isinstance(tag_id,` int):
                #     continue
                # camera_from_tag = T_from_Rt(*tag_Rt)
                draw_pose_axes(
                    color_images[0],
                    tx.invert(
                        self.extrinsics[0]) @ tag_T,
                    K=self.intrinsics[0])
                pose = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1)
                pose.transform(tag_T)
                # vis.add_geometry(f'{tag_id}_pose', pose, color=(1,1,1,1))
            cv2.imshow('color', color_images[0])
            cv2.waitKey(1)
            # win.wait()
        return tags_dict

    def collect(self, target_tags: Iterable[int]):
        cfg = self.cfg
        G = nx.Graph()
        done = False
        data = []
        really_done = False
        try:
            while not really_done:
                tags = self.collect_step()
                if tags is None:
                    continue

                # Filter tags by "usefulness"
                tags = {k: v for k, v in tags.items() if k in target_tags}
                if len(tags) < 2:
                    continue

                data.append(tags)

                for tag_A, tag_B in combinations(tags.keys(), r=2):
                    G.add_edge(tag_A, tag_B)

                done = (
                    G.number_of_nodes() == len(target_tags) and
                    nx.number_connected_components(G) == 1
                )
                print(G.nodes, done)
                really_done = (done and self.finish)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
            # break
        print(data)
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

                T_CA = cam_from_A
                T_CB = cam_from_B
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

                T_CA = cam_from_A
                T_CB = cam_from_B

                T_BA = tx.invert(T_CB) @ T_CA

                # tag_B_from_tag_A
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        idx_A, idx_B, T_BA, information_matrix, uncertain=True))

    def __optimize_pose_graph(self, pose_graph):
        cfg = self.cfg

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.03,
            edge_prune_threshold=0.02,
            preference_loop_closure=0.1,
            reference_node=0)

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            crit = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
            crit.max_iteration = 1024
            crit.min_residual = 0
            crit.max_iteration_lm = 256
            # crit.min_right_term = 1e-8
            # crit.min_relative_increment = 1e-8
            # crit.min_relative_residual_increment = 1e-8
            print(crit)
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                crit,
                option)

    def calibrate(self, target_tags: Iterable[int],
                  pop: Iterable[int] = None,
                  use_cache: bool = True):
        cfg = self.cfg
        target_tags = list(target_tags)

        idx_from_tag = {k: i for i, k in enumerate(target_tags)}
        print(idx_from_tag)
        # tag_from_idx = {i:k for (k,i) in idx_from_tag.items()}

        # Collect of retrieve tags data from cache.
        if use_cache and Path(cfg.data_cache).exists():
            with open(cfg.data_cache, 'rb') as fp:
                data = pickle.load(fp)
        else:
            data = self.collect(target_tags=target_tags)
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

        if pop is not None:
            for t in pop:
                tag_from_object.pop(t)
                output_poses.pop(t)

        if self.offset_file is not None:
            with open(self.offset_file, 'wb') as fp:
                pickle.dump(tag_from_object, fp)

        return output_poses


def plot_poses(poses):
    geoms = []
    ks = sorted(poses.keys(), key=lambda s: int(s))

    # for k, v in poses.items():
    for k in ks:
        print(k, type(k))
        v = poses[k]
        # if k in [10, 18]:
        #     continue
        goal_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        v = np.asarray(v, dtype=np.float32)
        goal_mesh.transform(v)
        geoms.append(goal_mesh)
    o3d.visualization.draw(geoms)


@store(name='calib_april')
def _main(object: str,
          tag_size: float = 0.015,
          debug: bool = True,
          #   pop: Optional[Tuple[int,...]] = None,
          plot: bool = True,
          use_cache: bool = True):
    print(object, tag_size, debug, plot)

    with open(F'/tmp/{object}_tag_ids.pkl', 'rb') as fp:
        tags = pickle.load(fp)

    # FIXME: Why is it not possible to
    # configure `pop` as an input argument?
    # pop = [list(tags)[13]]
    pop = []

    calibrator = MultiAprilTagCalibrator(
        MultiAprilTagCalibrator.Config(debug=debug,
                                       object=object,
                                       tag_size=tag_size))

    poses = calibrator.calibrate(
        target_tags=tags, pop=pop,
        use_cache=use_cache)

    if plot:
        plot_poses(poses)


def main():
    store.add_to_hydra_store()
    zen(_main).hydra_main(config_name='calib_april',
                          version_base='1.1',
                          config_path=None)


if __name__ == '__main__':
    main()
