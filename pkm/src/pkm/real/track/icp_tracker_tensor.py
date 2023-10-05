from typing import Any
import time
from collections import deque
import pickle
import copy

import open3d as o3d
import torch as th
import numpy as np
from icecream import ic
import pygicp
from cho_util.math import transform as tx

from pkm.models.cloud.point_mae import subsample
from pkm.real.track.track_object_pose import normalize_two_cloud_v2, normalize_two_cloud_v3

class ICPTracker:
    def __init__(self, 
                 method:str = "GICP",
                 k_correspondences:int = 31,
                 max_correspondence_distance:float = 0.015,
                 neighbor_search_method:str = "DIRECT_1",
                 neighbor_search_radius:float = 0.03,
                 mode:str = 'normal',
                 reinit:bool = True
                 ):
        
        self.initial_obj_pose = np.eye(4)
        self.method = method
        self.k_correspondences = k_correspondences
        self.max_correspondence_distance = max_correspondence_distance
        self.neighbor_search_method = neighbor_search_method
        self.neighbor_search_radius = neighbor_search_radius
        self.mode = mode
        # self.prev_cloud = None
        self.prev_pose = None
        self.reinit = reinit
        self._short_history = []


    def __call__(self, curr_cloud, curr_color=None,
                 full_cloud_colored=None,
                 device='cuda:0') -> np.ndarray:
        '''
            Given current pointcloud, return the current object pose
            args:
                curr_cloud: open3d point cloud object
        '''
        # prev_cloud = self.prev_cloud

        curr = o3d.t.geometry.PointCloud(o3d.core.Device(device))
        full_cloud_colored = full_cloud_colored.float().to(device)

        if len(self._short_history) <= 0:#prev_cloud is None:
            
            self.initial_obj_pose = o3d.core.Tensor.eye(4, 
                                        device=o3d.core.Device("cuda:0"))
        
            curr = o3d.t.geometry.PointCloud(o3d.core.Device(device))
            curr.point.positions = o3d.core.Tensor.from_dlpack(
                                    th.utils.dlpack.to_dlpack(
                                        full_cloud_colored[..., :3]))
            curr.point.colors = o3d.core.Tensor.from_dlpack(
                                    th.utils.dlpack.to_dlpack(  
                                        full_cloud_colored[..., 3:]))
            
            # curr.voxel_down_sample(voxel_size=0.01)
            self.initial_obj_pose[:3, 3] = curr.get_center()
            self._short_history.append((curr, self.initial_obj_pose))

            # self.prev_cloud = curr
            
            return th.utils.dlpack.from_dlpack(
                        self.initial_obj_pose.to_dlpack()).cpu().numpy()
        
        else:
            curr = o3d.t.geometry.PointCloud(o3d.core.Device(device))            

            curr.point.positions = o3d.core.Tensor.from_dlpack(
                                    th.utils.dlpack.to_dlpack(
                                        full_cloud_colored[..., :3]))
            curr.point.colors = o3d.core.Tensor.from_dlpack(
                                    th.utils.dlpack.to_dlpack(  
                                        full_cloud_colored[..., 3:]))
            # curr.voxel_down_sample(voxel_size=0.001)
            if self.reinit:
                initial_cloud, initial_pose = self._short_history[0]
                guess_initial_transform = self._short_history[-1][1] @ o3d.core.inv(initial_pose)
                    
                # point cloud registration with initial point cloud
            
                transform_init, fitness_init, _ = pairwise_normalized_t_plane_registration(
                        initial_cloud,
                        curr,
                        guess_initial_transform,
                        mode=self.mode
                        )
                curr_pose_init = transform_init @ initial_pose

            # Heuristically select registration pair wrt fitness score
            # of initial <-> current matching
            
            if self.reinit and (fitness_init > 0.6):
                curr_pose = curr_pose_init # match initial cloud to reduce drift
            else:
                # point cloud registration with previous point cloud
                transform_consec, fitness_consec, _ = pairwise_normalized_t_plane_registration(
                        self._short_history[-1][0],
                        curr,
                        mode=self.mode)
                curr_pose_consec = transform_consec @ self._short_history[-1][1]
                curr_pose = curr_pose_consec
            
            self.prev_cloud = curr_cloud#copy.deepcopy(curr_cloud)
            self.prev_pose = curr_pose

            # maintain history list
            self._short_history.append((curr, curr_pose))
            
            return th.utils.dlpack.from_dlpack(curr_pose.to_dlpack()).cpu().numpy()
            


def pairwise_normalized_t_plane_registration(source, target, 
                                            trans_init =None,
                                            mode = 'normal',
                                            threshold = 0.015,
                                            relative_fitness = 0.000001,
                                            relative_rmse = 0.000001,
                                            max_iteration = 30
                                            ):

    # Perform normalization
    # norm_transform, source_norm, target_norm = \
    #             normalize_two_cloud_v2(source, target)

    if True:
        # print(source.get_center())
        # print(target.get_center())
        norm_transform, source_norm, target_norm = \
                    normalize_two_cloud_v3(source, target)
        # print(source_norm.get_center())
        # print(target_norm.get_center())
    
        # o3d.visualization.draw([source_norm, target_norm])
        if trans_init is not None:
            trans_init = norm_transform @ trans_init @ norm_transform.inv()
            # trans_init = norm_transform.inv() @ trans_init @ norm_transform
    else:
        target_norm = copy.deepcopy(target)
        source_norm = copy.deepcopy(source)
        norm_transform = o3d.core.Tensor.eye(4, device=source.device)
    
    if trans_init is None:
        trans_init= o3d.core.Tensor.eye(4, device=source.device)
    
    
    # Estimate normal
    
    try:
        target_norm.point.normals
    except KeyError:
        target_norm.estimate_normals()

    try:
        source_norm.point.normals
    except KeyError:
        source_norm.estimate_normals()

    if mode == 'normal':
        registration = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    elif mode == 'thin':
        registration = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
        # print("=======================================================")

    reg_p2p = o3d.t.pipelines.registration.icp(
            source_norm, target_norm, threshold, trans_init,
            registration, # For Point-to-point metric
            o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness,
                                       relative_rmse,
                                       max_iteration))
    fitness = reg_p2p.fitness
    transform = reg_p2p.transformation.to(
        o3d.core.Dtype.Float32).cuda(0)
    # T @ N @ src = N @ dst
    # N^{-1} T @ N
    # Q = N^{-1} T @ N
    # T = N Q N^{-1}
    
    # print(norm_transform.device,
    #       norm_transform.dtype,
    #       type(norm_transform))
    # print(norm_transform)
    # print(transform.device,
    #       transform.dtype,
    #       type(transform)
    #       )

    transform = norm_transform.inv() @ transform @ norm_transform
    transform = transform.cuda(0)

    return transform, fitness, None


def pairwise_normalized_t_colored_registration(source, target, 
                                             trans_init = o3d.core.Tensor.eye(4, 
                                                            device=o3d.core.Device("cuda:0"))):

    # Perform normalization
    # norm_transform, source_norm, target_norm = \
    #             normalize_two_cloud_v2(source, target)
    target_norm = copy.deepcopy(target)
    source_norm = copy.deepcopy(source)
    
    threshold = 0.015  # Set this to an appropriate value depending on your data
    # trans_init = np.eye(4)
    relative_fitness = 0.000001
    relative_rmse = 0.000001
    max_iteration = 30
    # Estimate normal
    
    
    # scailing
    SCALE = 50.0
    threshold *= SCALE
    '''
        Somehow scale method does not work for tensor pointcloud
    '''
    # _source_norm = copy.deepcopy(source_norm.scale(scale=SCALE, center=np.array([0., 0., 0.])))
    _source_norm = o3d.t.geometry.PointCloud(o3d.core.Device("cuda:0"))
    _source_norm.point["positions"] = source_norm.point["positions"] * SCALE
    _source_norm.point["colors"] = source_norm.point["colors"]

    # _target_norm = copy.deepcopy(target_norm.scale(scale=SCALE, center=np.array([0., 0., 0.])))
    _target_norm = o3d.t.geometry.PointCloud(o3d.core.Device("cuda:0"))
    _target_norm.point["positions"] = target_norm.point["positions"] * SCALE
    _target_norm.point["colors"] = target_norm.point["colors"]


    trans_init[:3, 3] *= SCALE
    trans_init.to(o3d.core.Dtype.Float64)
    # print(threshold)
    


    
    _target_norm.estimate_normals()
    reg_p2p = o3d.t.pipelines.registration.icp(
            _source_norm, _target_norm, threshold, trans_init,
            # o3d.t.pipelines.registration.TransformationEstimationPointToPoint(), # For Point-to-point metric
            # o3d.t.pipelines.registration.TransformationEstimationPointToPlane(), # For Point-to-plane metric
            o3d.t.pipelines.registration.TransformationEstimationForColoredICP(), # For Point-to-plane metric
            o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness,
                                       relative_rmse,
                                       max_iteration))

    
    transform = copy.deepcopy(reg_p2p.transformation.to(o3d.core.Dtype.Float32).cuda(0))

    # unscale
    transform[:3, 3] *= (1/SCALE)
    
    fitness = reg_p2p.fitness


    return transform, fitness, None







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
    # datas = list(Path('/home/user/Documents/textured_cup').glob('*.pkl'))   
    # datas = list(Path('/home/user/Documents/pig-allviews-lowest').glob('*.pkl'))
    # datas = list(Path('/home/user/Documents/thin-board').glob('*.pkl'))
    datas = list(Path('/home/user/corn_runtime//2023-09-08/run/run-003/log').glob('*.pkl'))
    datas = sorted(datas, key = index_key)
    clouds = []
    for d in datas:
        # print(d)
        try:
            with open(d, 'rb') as fp:
                data = pickle.load(fp)
                print(list(data.keys()))
                if 'cloud' in data:
                    data = data['cloud']
                elif 'partial_cloud' in data:
                    data = data['partial_cloud'].squeeze(0)
        except Exception:
            print(F'Decided to skip {d} that we failed to read')
            continue
        clouds.append(data)
    return clouds

def main():
    from pkm.util.vis.win_o3d import AutoWindow

    import torch as th
    clouds = load_cloud_sequence_from_log_v2()
    tracker = ICPTracker(reinit=True)
    if True:
        win = AutoWindow()
        vis = win.vis
    import copy
    for cloud in clouds:
        # cloud_cpy = copy.deepcopy(cloud)
        
        # subsample cloud
        cloud = cloud[np.random.randint(cloud.shape[0], size=2048), :]
        print(cloud.shape)
        # print(cloud.mean(axis=0))

        curr_pose = tracker(
                th.as_tensor(cloud[..., 0:3]),
                th.as_tensor(cloud[..., 3:6]),
                th.as_tensor(cloud))
        # # visualize()
        # print(curr_pose)

        if True:
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(cloud[...,:3])
            if cloud.shape[-1] == 6:
                o3d_cloud.colors = o3d.utility.Vector3dVector(cloud[...,3:6])
            
            
            # Fix axes(transform cloud)
            if False:
                o3d_cloud.transform(np.linalg.inv(curr_pose))
                vis.add_axes('origin', size=0.1, origin=np.zeros(3))
                vis.add_geometry('cloud', o3d_cloud, color=(1,1,1,1))   
            else:
                pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                pose.transform(curr_pose)
                vis.add_geometry('pose', pose, color=(1,1,1,1))
                vis.add_geometry('cloud', o3d_cloud, color=(1,1,1,1))   



            # vis.add_
            # win.wait()
            win.tick()
            time.sleep(0.05)


if __name__ == '__main__':
    main()
