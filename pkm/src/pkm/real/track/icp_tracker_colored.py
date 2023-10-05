from typing import Any
import open3d as o3d
import pygicp
import numpy as np
import time
from cho_util.math import transform as tx
import sys
sys.path.append("/home/user/workspace/corn/pkm/scripts/real/posetracker")
from track_object_pose import normalize_two_cloud_v2
import pickle
import copy
from pkm.models.cloud.point_mae import subsample

from collections import deque


class ICPTracker:
    def __init__(self,
                 method: str = "GICP",
                 k_correspondences: int = 31,
                 max_correspondence_distance: float = 0.015,
                 neighbor_search_method: str = "DIRECT_1",
                 neighbor_search_radius: float = 0.03):

        self.initial_obj_pose = np.eye(4)
        self.method = method
        self.k_correspondences = k_correspondences
        self.max_correspondence_distance = max_correspondence_distance
        self.neighbor_search_method = neighbor_search_method
        self.neighbor_search_radius = neighbor_search_radius

        self.prev_cloud = None
        self.prev_pose = None
        self._pcd_list = []

        self._long_history = []
        self._short_history = []

    def __call__(self, curr_cloud, curr_color=None,
                 full_cloud_colored=None) -> np.ndarray:
        '''
            Given current pointcloud, return the current object pose
            args:
                curr_cloud: open3d point cloud object
        '''
        prev_cloud = self.prev_cloud

        if prev_cloud is None:
            self.prev_cloud = full_cloud_colored[..., :3]
            self.prev_pose = np.eye(4)
            self.prev_pose[...,
                           :3,
                           3] = full_cloud_colored[...,
                                                   :3].mean(dim=0)
            self.initial_obj_pose = self.prev_pose

            curr = o3d.geometry.PointCloud()
            curr.points = o3d.utility.Vector3dVector(
                full_cloud_colored[..., :3].numpy())

            if curr_color is not None:
                curr.colors = o3d.utility.Vector3dVector(
                    full_cloud_colored[..., 3:].numpy())

            self._pcd_list.append(curr)
            self._long_history.append((curr, self.prev_pose))
            self._short_history.append((curr, self.prev_pose))

            return self.prev_pose

        else:
            curr = o3d.geometry.PointCloud()
            curr.points = o3d.utility.Vector3dVector(curr_cloud.numpy())

            prev = self._short_history[-1][0]

            if curr_color is not None:
                curr.colors = o3d.utility.Vector3dVector(curr_color.numpy())

            initial_cloud, initial_pose = self._short_history[0]

            if True:
                guess_transform = self._short_history[-1][1] @ tx.invert(
                    initial_pose)
            else:
                # curr_pose @ prev^{-1}
                # T @ prev==curr
                # prev_pose^{-1}@prev == curr_pose^{-1}@curr
                # curr_pose @ prev^{-1} @ prev
                # world(t1)_from_curr @ obj_ @ prev
                prev_from_curr = pairwise_registration(
                    prev, curr, np.eye(4)
                )
                guess_transform = (
                    prev_from_curr @
                    self._short_history[-1][1] @
                    tx.invert(initial_pose)
                )

            if False:
                # initial transformation
                # Disable colored icp
                transform, _ = pairwise_normalized_registration(
                    initial_cloud,
                    curr,
                    guess_transform
                )
            elif True:
                # Use colored icp
                transform, _ = pairwise_colored_registration(
                    initial_cloud,
                    curr,
                    guess_transform
                )
            else:  # consecutive transformation
                transform, _ = pairwise_normalized_registration(
                    self._short_history[-1][0],
                    curr)
                curr_pose = transform @ self._short_history[-1][1]
                self.prev_cloud = curr_cloud
                self.prev_pose = curr_pose
                # maintain long and short list
                self._short_history.append((curr, curr_pose))
                return curr_pose

            curr_pose = transform @ initial_pose
            self.prev_cloud = curr_cloud
            self.prev_pose = curr_pose

            # maintain long and short list
            self._short_history.append((curr, curr_pose))

            return curr_pose


# ================= Pose graph optimization =======================


def pairwise_registration(source, target, trans_init=np.eye(4)):

    # source: prev, target: curr
    # print("Apply point-to-plane ICP")
    # Apply ICP
    threshold = 0.02  # Set this to an appropriate value depending on your data
    # trans_init = np.eye(4)
    relative_fitness = 0.000001
    relative_rmse = 0.000001
    max_iteration = 30
    # Estimate normal
    target.estimate_normals()

    reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
        source, target, threshold, trans_init,
        # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness,
                                                          relative_rmse,
                                                          max_iteration))
    transformation_icp = reg_p2p.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, threshold, reg_p2p.transformation)

    return transformation_icp, information_icp


def pairwise_normalized_registration(source, target, trans_init=np.eye(4)):
    '''
        Perform normalization(to center) and scailing before ICP matching
    '''

    # Perform normalization
    norm_transform, target_norm, source_norm = \
        normalize_two_cloud_v2(target, source)

    # Apply ICP
    threshold = 0.015  # Set this to an appropriate value depending on your data
    # trans_init = np.eye(4)
    relative_fitness = 0.000001
    relative_rmse = 0.000001
    max_iteration = 30
    # Estimate normal
    target.estimate_normals()

    trans_init = norm_transform @ trans_init @ tx.invert(norm_transform)

    # scailing
    SCALE = 50.0
    threshold *= SCALE
    source_norm.scale(scale=SCALE, center=np.array([0., 0., 0.]))
    target_norm.scale(scale=SCALE, center=np.array([0., 0., 0.]))
    trans_init[:3, 3] *= SCALE

    reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
        source_norm, target_norm, threshold, trans_init,
        # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness,
                                                          relative_rmse,
                                                          max_iteration))

    transformation_icp = copy.deepcopy(reg_p2p.transformation)
    # unscailing
    transformation_icp[:3, 3] *= (1 / SCALE)

    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, threshold, transformation_icp)

    transform = tx.invert(norm_transform) @ transformation_icp @ norm_transform

    return transform, information_icp


def pairwise_colored_registration(source, target, trans_init=np.eye(4)):
    '''
        Perform normalization, scailing, and colored ICP matching
        Currently not working
    '''
    # Perform normalization
    norm_transform, target_norm, source_norm = \
        normalize_two_cloud_v2(target, source)
    trans_init = norm_transform @ trans_init @ tx.invert(norm_transform)

    threshold = 0.02  # Set this to an appropriate value depending on your data
    relative_fitness = 0.000001
    relative_rmse = 0.000001
    max_iteration = 30

    # scailing
    SCALE = 50.0
    threshold *= SCALE
    source_norm.scale(scale=SCALE, center=np.array([0., 0., 0.]))
    target_norm.scale(scale=SCALE, center=np.array([0., 0., 0.]))
    trans_init[:3, 3] *= SCALE

    source_norm.estimate_normals()
    target_norm.estimate_normals()
    # Applying colored point cloud registration
    reg_p2p = o3d.pipelines.registration.registration_colored_icp(
        source_norm, target_norm, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(
            lambda_geometric=0.968
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness,
                                                          relative_rmse,
                                                          max_iteration))

    transformation_icp = copy.deepcopy(reg_p2p.transformation)
    # unscailing
    transformation_icp[:3, 3] *= (1 / SCALE)

    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, threshold, transformation_icp)

    transform = tx.invert(norm_transform) @ transformation_icp @ norm_transform

    return transform, information_icp


def index_key(k):
    from pathlib import Path
    try:
        return int(Path(k).stem.split('-')[-1])
    except Exception:
        return np.inf


def load_cloud_sequence_from_log():
    import torch as th
    from pathlib import Path

    path = '/tmp/real-log/run-441/log'
    datas = Path(path).glob('*.pkl')

    datas = sorted(datas, key=index_key)
    clouds = []
    for d in datas:
        try:
            with open(d, 'rb') as fp:
                data = pickle.load(fp)
        except Exception:
            print(F'Decided to skip {d} that we failed to read')
            continue
        clouds.append(data['partial_cloud'])
    return clouds


def load_cloud_sequence_from_log_v2():
    from pathlib import Path
    # datas = list(Path('/home/user/Documents/rolling').glob('*.pkl'))
    # datas = list(Path('/home/user/Documents/textureless_cup').glob('*.pkl'))
    # datas = list(Path('/home/user/Documents/rapid_motion').glob('*.pkl'))
    # datas = list(Path('/tmp/rapid_motion_textured').glob('*.pkl'))
    # datas = list(Path('/home/user/Documents/colored-cup').glob('*.pkl'))
    datas = list(Path('/home/user/Documents/textured_cup').glob('*.pkl'))   
    # datas = list(Path('/home/user/Documents/thin-board').glob('*.pkl'))
    # datas = list(Path('/home/user/Documents/pig-allviews-lowest').glob('*.pkl'))
    datas = sorted(datas, key=index_key)
    clouds = []
    for d in datas:
        # print(d)
        try:
            with open(d, 'rb') as fp:
                data = pickle.load(fp)
                if 'cloud' in data:
                    data = data['cloud']
        except Exception:
            print(F'Decided to skip {d} that we failed to read')
            continue
        clouds.append(data)
    return clouds


def main():
    from pkm.util.vis.win_o3d import AutoWindow

    import torch as th
    clouds = load_cloud_sequence_from_log_v2()
    tracker = ICPTracker()
    if True:
        win = AutoWindow()
        vis = win.vis
    import copy
    for cloud in clouds:
        # cloud_cpy = copy.deepcopy(cloud)

        # subsample cloud
        cloud = cloud[np.random.randint(cloud.shape[0], size=2048), :]

        curr_pose = tracker(
            th.as_tensor(cloud[..., 0:3]),
            th.as_tensor(cloud[..., 3:6]),
            th.as_tensor(cloud))
        # # visualize()
        # print(curr_pose)

        if True:
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(cloud[..., :3])
            o3d_cloud.colors = o3d.utility.Vector3dVector(cloud[..., 3:6])

            # Fix axes(transform cloud)
            if False:
                o3d_cloud.transform(np.linalg.inv(curr_pose))
                vis.add_axes('origin', size=0.1, origin=np.zeros(3))
                vis.add_geometry('cloud', o3d_cloud, color=(1, 1, 1, 1))
            else:
                pose = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1)
                pose.transform(curr_pose)
                vis.add_geometry('pose', pose, color=(1, 1, 1, 1))
                vis.add_geometry('cloud', o3d_cloud, color=(1, 1, 1, 1))

            # vis.add_
            # win.wait()
            win.tick()


if __name__ == '__main__':
    main()
