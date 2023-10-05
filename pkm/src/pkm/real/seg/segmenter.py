#!/usr/bin/env python3

from matplotlib import pyplot as plt
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import numpy as np
import torch as th
import open3d as o3d
import pickle
from icecream import ic

import cuml
import cupy as cp

from pkm.util.torch_util import dcn
from pkm.data.transforms.io_xfm import scene_to_mesh

from pkm.data.transforms.col import (
    IsInRobot
)
import concurrent.futures as cf
cuml.set_global_output_type('cupy')
from kornia.color import rgb_to_hsv

# NOT SURE WHICH OPTION WILL BE FASTEST
PROC_EACH: bool = True

from pkm.real.util import o3d2th, th2o3d


def fit_table(pcd,
              eps: float,
              n_ransac: int = 3,
              n_iter: int = 1000):
    """ fit table params """
    plane, inliers = pcd.segment_plane(
        distance_threshold=eps,
        ransac_n=n_ransac,
        num_iterations=n_iter)
    return plane


def remove_table(pcd, **kwds):
    """Remove table from point cloud via plane fitting."""
    plane_eps: float = kwds.pop('plane_eps', 0.01)
    eps: float = kwds.pop('eps', 0.01)

    num_points: int = kwds.pop('num_points', 3)
    num_iter: int = kwds.pop('num_iter', 1000)
    aux: Optional[Dict[str, np.ndarray]] = kwds.pop('aux', None)
    use_normal = kwds.pop('use_normal', True)

    if pcd.is_empty():
        return pcd

    if (aux is not None) and ('plane' in aux) and (
            aux['plane'] is not None):
        # == use cached plane parameters ==
        plane = aux['plane']
        # print(pcd.point.positions.dtype)
        # print(pcd.point.positions.dtype)

        # if use_normal:
        #     try:
        #         pcd.estimate_normals()
        #     except RuntimeError:
        #         use_normal = False
        points = pcd.point.positions
        points = o3d2th(points).to(dtype=th.float32)

        # points = pcd.point.positions
        # plane = o3d2th(plane).to(dtype=th.float32)
        dist = (points @ plane[:3] + plane[3])
        mask = dist <= eps

        if use_normal:
            normals = pcd.point.normals
            normals = o3d2th(normals).to(dtype=th.float32)
            mask_normal = (normals[..., 2].abs() >=
                           float(np.cos(np.deg2rad(20))))
            if not kwds.pop('thin', False):
                mask = th.logical_or(dist <= 0.005,
                                    th.logical_and(mask,
                                    mask_normal))
                

        inliers = th.argwhere(mask)
        inliers = th2o3d(inliers)
        
    else:
        # == compute plane params with ransac ==
        plane, inliers = pcd.segment_plane(
            distance_threshold=plane_eps,
            ransac_n=num_points,
            num_iterations=num_iter)
        print("calculated plane", plane)
        if aux is not None:
            aux['plane'] = o3d2th(plane).to(dtype=th.float32)
    pcd = pcd.select_by_index(inliers, invert=True)
    return pcd


def remove_by_color(pcd, **kwds):
    '''
    Heuristically remove the background points using color
    Requires the lower/upper bounds of the HSV value
    '''
    # Range for the background(green's hue: ~180)
    bounds = None
    if 'bounds' in kwds:
        bounds = kwds.pop('bounds')

    # Parse bounds inputs.
    lo2 = None
    hi2 = None
    if bounds is not None:
        lo1 = bounds[..., 0]
        hi1 = bounds[..., 1]
        if bounds.shape[-1] > 2:
            lo2 = bounds[..., 2]
            hi2 = bounds[..., 3]
    else:
        lo1 = np.asarray([180.0, 0.59, 0.1])
        hi1 = np.asarray([204.0, 1.0, 0.69])

    rgb_colors = pcd.point["colors"]
    rgb_colors = o3d2th(rgb_colors)

    hsv_colors = rgb_to_hsv(rgb_colors[..., None, None])[..., 0, 0]
    hsv_colors[..., 0] = th.rad2deg(hsv_colors[..., 0])

    lower = th.as_tensor(lo1, dtype=hsv_colors.dtype,
                         device=hsv_colors.device)
    upper = th.as_tensor(hi1, dtype=hsv_colors.dtype,
                         device=hsv_colors.device)

    # Eliminate the background using hue
    mask = th.logical_and(hsv_colors >= lower,
                          hsv_colors <= upper).all(dim=-1)
    if lo2 is not None:
        lower = th.as_tensor(lo2, dtype=hsv_colors.dtype,
                             device=hsv_colors.device)
        upper = th.as_tensor(hi2, dtype=hsv_colors.dtype,
                             device=hsv_colors.device)
        mask2 = th.logical_and(hsv_colors >= lower,
                               hsv_colors <= upper).all(dim=-1)
        mask |= mask2

    points = pcd.point["positions"]
    points = o3d2th(points)

    mask &= points[..., 2] <= 0.03

    mask = th.logical_not(mask)
    index = th.argwhere(mask)
    index = index.squeeze(dim=-1)

    hsv = hsv_colors[index]

    index = th2o3d(index)

    pcd = pcd.select_by_index(index)

    return pcd, hsv


def remove_outside_bounds(pcd, **kwds):
    """Heuristically remove points below the table by height threshold.

    Requires that the extrinsics calibration is correct.
    """
    if pcd.is_empty():
        return pcd
    lower = kwds.pop('lower', [
        -0.25 + 0.5,
        -0.3,
        -0.03
    ])
    upper = kwds.pop('upper', [
        +0.25 + 0.5,
        +0.3,
        +0.15
    ])
    if True:
        points = pcd.point.positions
        points = o3d2th(points)
        # print(points.min(dim=0),
        #       points.max(dim=0),
        #       points.std(dim=0))

        # lower = np.asarray(lower, dtype=np.float32)
        # upper = np.asarray(upper, dtype=np.float32)
        # lower = o3d.core.Tensor(lower, dtype=o3d.core.Dtype.Float32,
        #                         device=o3d.core.Device.CUDA)
        # upper = o3d.core.Tensor(upper, dtype=o3d.core.Dtype.Float32,
        #                          device=o3d.core.Device.CUDA)

        lower = th.as_tensor(lower, dtype=points.dtype,
                             device=points.device)
        upper = th.as_tensor(upper, dtype=points.dtype,
                             device=points.device)
        # points = th.as_tensor(pcd.point.positions)
        # print(points.shape, lower.shape, upper.shape)
        mask = th.logical_and(points > lower,
                              points < upper).all(dim=-1)
        # print(mask.shape)
        index = th.argwhere(mask)
        # print(index.shape)
        # print(points.shape, lower.shape)
        index = index.squeeze(dim=-1)

        index = th2o3d(index)

    else:
        points = np.asarray(pcd.points)
        index = np.argwhere(
            np.logical_and(points > lower, points < upper).all(axis=-1)
        )

    pcd = pcd.select_by_index(index)
    # print(pcd.point.positions.shape)
    return pcd


def place_robot(robot, joint_state):
    """Run forward kinematics to determine the robot geometry at a particular
    joint state."""
    q = np.zeros(7)
    q[:7] = joint_state[:7]
    robot.update_cfg(q)
    robot_mesh = scene_to_mesh(robot.scene).as_open3d
    return robot_mesh


def remove_robot_raycast(pcd,
                         robot,
                         joint_state,
                         aux: Optional[Dict[str, Any]] = None,
                         **kwds):
    """Remove the robot from the point cloud, based on analytic model of where
    the robot _should_ be at a given kinematic state.

    Backend based on raycasting mesh<->point distances (~embree)
    """
    max_distance = kwds.pop('max_distance', 0.02)

    # Compute robot configuration + corresponding geometry.
    robot_mesh = place_robot(robot, joint_state)
    aux['robot_mesh'] = robot_mesh

    # Generate accelerated spatial data structure for distance computation.
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(robot_mesh))

    # Compute distance to mesh, and prune by max distance.
    dis = scene.compute_distance(
        np.asarray(pcd.points,
                   dtype=np.float32)
    )
    dis = np.asarray(dis)
    pcd = pcd.select_by_index(
        np.argwhere(dis > max_distance)
    )
    return pcd


def remove_robot_chull(pcd, hsv,
                       is_in_robot,
                       joint_state,
                       aux: Optional[Dict[str, Any]] = None,
                       **kwds):
    """Remove the robot from the point cloud, based on analytic model of where
    the robot _should_ be at a given kinematic state.

    Backend based on Convex Hull Intersection Queries.
    """
    if pcd.is_empty():
        return pcd
    max_distance = kwds.pop('max_distance', 0.02)

    # Compute distance to mesh, and prune by max distance.
    q = joint_state

    # x = np.asarray(pcd.points,
    #                dtype=np.float32)

    points = o3d2th(pcd.point.positions)
    x = points

    occ = is_in_robot(th.as_tensor(q, device=is_in_robot._device),
                      x,
                      tol=max_distance).bool()

    index = th.argwhere(~occ)
    index = index.squeeze(dim=-1)
    # print(x.shape, index.shape)
    index = th2o3d(index)
    pcd = pcd.select_by_index(index)
    return pcd


def remove_robot_surface(pcd,
                         is_on_robot,
                         joint_state,
                         aux: Optional[Dict[str, Any]] = None,
                         **kwds):
    """Remove the robot from the point cloud, based on analytic model of where
    the robot _should_ be at a given kinematic state.

    Backend based on raycasting mesh<->point distances (~embree)
    difference with remove_robot_raycast is that we try to cache the
    robot geometry for faster processing.
    """
    max_distance = kwds.pop('max_distance', 0.02)

    # Compute distance to mesh, and prune by max distance.
    q = joint_state
    x = np.asarray(pcd.points,
                   dtype=np.float32)
    occ = np.argwhere(~dcn(is_on_robot(q, x, tol=max_distance)))
    pcd = pcd.select_by_index(occ)
    return pcd


def remove_noise(pcd, **kwds):
    """Remove noise from the point cloud based on a heuristic radius-based
    outlier removal logic.

    Basically, clusters within a radius with fewer points than threshold
    are discarded from the source point cloud.
    """
    min_num_points = kwds.pop('min_num_points', 64)
    radius = kwds.pop('radius', 0.02)

    pcd, inliers = pcd.remove_radius_outliers(
        nb_points=min_num_points,
        search_radius=radius)
    return pcd


def dbscan_cuml(pcd: o3d.t.geometry.PointCloud,
                eps: float = 0.01,
                min_points: int = 4):
    if not isinstance(pcd, th.Tensor):
        pcd = o3d2th(pcd.point.positions)
    pcd = pcd.to(dtype=th.float32)

    with cuml.using_output_type('cupy'):
        dbscan_float = cuml.DBSCAN(eps=eps, min_samples=min_points,
                                   calc_core_sample_indices=False)
        dbscan_float.fit(pcd)
        labels = dbscan_float.labels_
    labels = th.utils.dlpack.from_dlpack(labels.toDlpack())
    # labels = o3d2th(labels)
    return labels


def select_largest(pcd, eps: float = 0.01,
                   min_points: int = 8):
    # SELECT THE LARGEST CLUSTER.
    if False:
        # Open3D DBSCAN (CPU)
        cluster_labels = pcd.cluster_dbscan(eps=0.01,
                                            min_points=4)
        cluster_labels = o3d2th(cluster_labels)
    else:
        # Open3D DBSCAN (GPU)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        pcd_th = o3d2th(pcd.point.positions)
        cluster_labels = dbscan_cuml(pcd_th,
                                     eps=eps,
                                     min_points=min_points)

    if False:
        # counts = np.bincount(cluster_labels)
        labels, counts = np.unique(cluster_labels, return_counts=True)
        counts[labels < 0] = 0
        largest_label = labels[np.argmax(counts)]
        cluster_indices = np.argwhere(cluster_labels == largest_label)
    else:
        labels, counts = th.unique(cluster_labels, return_counts=True)
        counts[labels < 0] = 0
        ordered_labels = labels[th.argsort(counts, descending=True)]

        best_label = ordered_labels[0]
        for label in ordered_labels:
            zmin = pcd_th[cluster_labels == label][..., 2].amin()
            # print(zmin,)
            if zmin > eps:
                continue
            best_label = label
            break

        # largest_label = labels[th.argmax(counts)]
        cluster_indices = th.argwhere(cluster_labels == best_label)
        cluster_indices = cluster_indices.squeeze(dim=-1)
        cluster_indices = th2o3d(cluster_indices)

    pcd = pcd.select_by_index(cluster_indices)
    return pcd


def cloud_from_depth(depth_image,
                     mask,
                     fx, fy, cx, cy,
                     color_image: Optional[np.ndarray] = None):
    """Convert a depth image and a mask into a point cloud.

    Parameters:
    - depth_image: A 2D numpy array containing depth values in meters.
    - mask: A 2D numpy array of the same size as depth_image.
      It should contain 1 for pixels that belong to the object,
      and 0 for all other pixels.
    - fx, fy: The focal lengths of the camera in the x and y directions.
    - cx, cy: The optical centers of the camera in the x and y directions.

    Returns:
    - A PointCloud object containing the 3D points of the object.
    """
    depth_image = depth_image * mask

    # Create an Open3D Image from the depth image
    depth_o3d = o3d.t.geometry.Image(
        depth_image.astype(np.float32)).cuda()

    # Create an Open3D PinholeCameraIntrinsic object
    # This contains the camera's intrinsic parameters
    if True:
        camera_intrinsic = np.eye(3, dtype=np.float32)
        camera_intrinsic[0, 0] = fx
        camera_intrinsic[1, 1] = fy
        camera_intrinsic[0, 2] = cx
        camera_intrinsic[1, 2] = cy
        # print(camera_intrinsic)

        camera_intrinsic = o3d.core.Tensor(camera_intrinsic,
                                           dtype=o3d.core.Dtype.Float32,
                                           #    device=o3d.core.Device.CUDA
                                           )
    else:
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth_image.shape[1], depth_image.shape[0], fx, fy, cx, cy)

    if color_image is None:
        # Convert the depth image to a point cloud
        point_cloud = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth_o3d, camera_intrinsic, depth_scale=1.0)
    else:
        # depths = depth_image.reshape(-1)
        # point_cloud.colors = o3d.utility.Vector3dVector(
        #     color_image.reshape(-1, 3)[depths > 0] / 255.0
        # )
        color_o3d = o3d.t.geometry.Image(
            color_image.astype(np.uint8)).cuda()
        rgbd_o3d = o3d.t.geometry.RGBDImage(color_o3d, depth_o3d)

        point_cloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_o3d, camera_intrinsic, depth_scale=1.0,
            with_normals=True)
        # print(point_cloud.point.colors.min(axis=0),
        #       point_cloud.point.colors.max(axis=0),
        #       )

    return point_cloud


def load_cloud(fx: float,
               fy: float,
               cx: float,
               cy: float,
               depth_image: np.ndarray,
               color_image: np.ndarray,
               mask: Optional[np.ndarray] = None):
    if mask is None:
        mask = np.ones_like(depth_image)
    point_cloud = cloud_from_depth(depth_image, mask,
                                   fx, fy, cx, cy,
                                   color_image=color_image)
    return point_cloud


def project_point(img_shape: Tuple[int, ...],
                  K: np.ndarray,
                  point: np.ndarray):
    h, w, *_ = img_shape
    # Compute Projection
    point_2d_h = np.einsum('ij, ...j -> ...i', K, point)
    point_2d = point_2d_h[..., :2] / point_2d_h[..., 2:]

    # Create an empty image
    image_shape = (h, w)
    point_2d = point_2d.astype(np.int32)

    # Clip points that fall outside the image
    point_2d[..., 0] = np.clip(point_2d[..., 0],
                               0, image_shape[1] - 1)
    point_2d[..., 1] = np.clip(point_2d[..., 1],
                               0, image_shape[0] - 1)
    return point_2d


class MultiSegmenter:
    '''
        Segmenter that gets list of depth and color images
        And returns combined point cloud
    '''
    @dataclass
    class Config:
        # FIXME: currently unused
        win_radius: int = 96
        robot_file: str = (
            '/home/user/workspace/corn/pkm/src/pkm/data/assets/' +
            'franka_description/robots/franka_panda_2.urdf'
        )
        device: str = 'cpu'
        hand_only: bool = True
        keep_last: Optional[int] = None
        dilate_robot: float = 0.02
        stride: int = 1
        mode: str = 'normal'
        table_eps: float = 0.02
        noise_eps: float = 0.05
        use_roi: bool = True

    def __init__(self,
                 cfg: Config,
                 T_bc_list,
                 intrinsics,
                 table_colors):
        self.cfg = cfg
        # FIXME: `keep_last=4` only works
        # if using franka_2
        keep_last = None
        if cfg.hand_only:
            keep_last = 4
        if cfg.keep_last is not None:
            keep_last = cfg.keep_last
        self.is_in_robot = IsInRobot(cfg.robot_file, device=cfg.device,
                                     # `prune` does not currently work :(
                                     keep_last=keep_last,
                                     prune=False)

        self.num_cameras = len(T_bc_list)
        self.intrinsics = intrinsics

        if T_bc_list is None:
            raise (NotImplementedError)
        else:
            self.__T_bc_list = T_bc_list

        self.__aux = {}
        self.have_table: bool = False

        self.__table_color = []
        for table_color in table_colors:
            table_color1 = [
                np.asarray(table_color['h1']) * 2,
                np.asarray(table_color['s']) / 255,
                np.asarray(table_color['v']) / 255,
            ]  # 6x2
            table_color1 = np.stack(table_color1,
                                    axis=0)  # 3x2
            table_color2 = [
                np.asarray(table_color['h2']) * 2,
                np.asarray(table_color['s']) / 255,
                np.asarray(table_color['v']) / 255,
            ]  # 6x2
            table_color2 = np.stack(table_color2,
                                    axis=0)  # 3x2
            table_color = np.concatenate([table_color1,
                                          table_color2],
                                         axis=-1)  # 3x4
            self.__table_color.append(table_color)
        self.__plane = None
        self.__prev_center = None
        self.__executor = cf.ThreadPoolExecutor(max_workers=3)

    @property
    def T_bc(self):
        # return self.__T_bc
        return self.__T_bc_list[0]

    def fit_table(self, pcds):
        cfg = self.cfg
        # Combine
        pcd = pcds[0].clone()
        for c in pcds[1:]:
            pcd.append(c)
        plane = fit_table(pcd, cfg.table_eps)
        return plane

    def proc_1(self, pcd, joint_state, bounds):
        cfg = self.cfg
        # pcd=pcd.clone()
        if cfg.mode == 'normal':
            # center = (0.5, 0.0)
            # +-4
            pcd = remove_outside_bounds(pcd,
                                        upper=[0.82, 0.4, 0.2],
                                        # TODO: or z=0.01?
                                        lower=[0.18, -0.4, 0.00])
        elif cfg.mode == 'thin':
            pcd = remove_outside_bounds(pcd,
                                        upper=[0.75, 0.3, 0.03],
                                        lower=[0.25, -0.3, 0.00])
        if True:
            if cfg.mode == 'normal':
                pcd = remove_table(pcd,
                                   aux={'plane': self.__plane},
                                   plane_eps=cfg.table_eps,
                                   # plane_eps=cfg.noise_eps,
                                   eps=cfg.noise_eps,
                                   # eps=cfg.table_eps
                                   thin=(cfg.mode == 'thin')
                                   )
        if False:
            # if not pcd.is_empty():
            pcd = pcd.voxel_down_sample(voxel_size=0.005)

        if True:
            # if not pcd.is_empty():
            # print('from')
            # print(pcd.point.positions.shape)
            pcd, hsv = remove_by_color(pcd,
                                       bounds=bounds)
            # print('to')
            # print(pcd.point.positions.shape)
            # o3d.visualization.draw(pcd)
        
        if cfg.mode == 'normal':#not pcd.is_empty():
            pcd = remove_robot_chull(pcd, None,
                                     self.is_in_robot,
                                     joint_state,
                                     max_distance=0.02)
        return pcd

    def filter(self, pcd, joint_state,
               ws: bool = True,
               table: bool = True):
        cfg = self.cfg
        if ws:
            # [1] remove by workspace
            if cfg.mode == 'normal':
                pcd = remove_outside_bounds(pcd,
                                            upper=[0.75, 0.3, 0.2],
                                            # TODO: or z=0.01?
                                            lower=[0.25, -0.3, 0.00])
            elif cfg.mode == 'thin':
                pcd = remove_outside_bounds(pcd,
                                            upper=[0.75, 0.3, 0.02],
                                            lower=[0.25, -0.3, -0.01])

        # [2] remove table
        if table:
            if cfg.mode == 'normal':
                pcd = remove_table(pcd,
                                   aux={'plane': self.__plane},
                                   plane_eps=cfg.table_eps,
                                   eps=cfg.noise_eps)

        pcd = remove_robot_chull(pcd,
                                 self.is_in_robot,
                                 joint_state,
                                 max_distance=0.01)
        return pcd

    def __call__(self,
                 joint_state: np.ndarray,
                 #  depth_image: np.ndarray,
                 depth_images: List[np.ndarray],
                 #  color_image: np.ndarray,
                 color_images: List[np.ndarray],
                 object_pose: Optional[np.ndarray] = None,
                 aux=None
                 ) -> np.ndarray:

        cfg = self.cfg
        # pcds = []
        pcds= [None for _ in range(self.num_cameras)]

        def __load_cloud(i:int):
            stride: int = cfg.stride
            inv_s = (1.0 / stride)
            K = inv_s * self.intrinsics[i]
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            ref_points = None

            if cfg.use_roi and (self.__prev_center is not None):
                Ti = np.linalg.inv(self.__T_bc_list[i])
                center = self.__prev_center @ Ti[:3, :3].T + Ti[:3, 3]
                radius = 0.1

                ref_points = np.stack([
                    center - (radius, radius, 0),
                    center + (radius, radius, 0)], axis=0)

                ref_points = project_point(
                    depth_images[i].shape,
                    self.intrinsics[i],
                    ref_points)
            if ref_points is None:
                pcd = load_cloud(fx, fy, cx, cy,
                                 depth_images[i][::stride, ::stride],
                                 # color_image=None,
                                 color_images[i][::stride, ::stride],
                                 mask=None)
            else:
                # print('stride',stride)
                j0, i0 = ref_points[0]
                j1, i1 = ref_points[1]
                # cvt 2 int
                j0 = int(j0)
                i0 = int(i0)
                j1 = int(j1)
                i1 = int(i1)
                # print(i0, i1, j0, j1)

                pcd = load_cloud(fx, fy,
                                 -inv_s * j0 + cx,
                                 -inv_s * i0 + cy,
                                 (
                                     depth_images[i][i0:i1 + 1:stride, j0:j1 + 1:stride]),
                                 # color_image=None,
                                 (
                                     color_images[i][i0:i1 + 1:stride, j0:j1 + 1:stride]),
                                 mask=None)
            # downsample the interim point clouds.
            # pcd = pcd.voxel_down_sample(voxel_size = 0.005)
            pcd = pcd.transform(self.__T_bc_list[i])
            pcds[i] = pcd
        if False:
            for i in range(self.num_cameras):
                __load_cloud(i)            
        else:
            executor = self.__executor
            future_to_proc = {executor.submit(__load_cloud, i) : i
                            for i in range(3)}
            for future in cf.as_completed(future_to_proc):
                future.result()
                    
                    
        # o3d.visualization.draw(pcds)
        if self.__plane is None:
            if False:
                p = o3d2th(pcds[0].point.positions)
                with open('/tmp/points.pkl', 'wb') as fp:
                    pickle.dump(p ,fp)
                plt.hist(dcn(p[...,2]), bins=64)
                plt.savefig('/tmp/hist.png')
            self.__plane = self.fit_table(pcds)
            self.__plane = o3d2th(self.__plane).to(dtype=th.float32)

            # print(F'>>>>self.plane = {self.__plane}')

        # preprocess + merge
        pcd_combined = None

        if False:
            for i in range(self.num_cameras):
                pcd = pcds[i]

                if PROC_EACH:
                    if True:
                        # [1] remove by workspace
                        if cfg.mode == 'normal':
                            pcd = remove_outside_bounds(pcd,
                                                        upper=[0.75, 0.3, 0.2],
                                                        # TODO: or
                                                        # z=0.01?
                                                        lower=[0.25, -0.3, 0.00])
                        elif cfg.mode == 'thin':
                            pcd = remove_outside_bounds(
                                pcd, upper=[0.75, 0.3, 0.02],
                                lower=[0.25, -0.3, 0.00])
                    if cfg.mode == 'normal':
                        pcd = remove_table(pcd,
                                           aux={'plane': self.__plane},
                                           plane_eps=cfg.table_eps,
                                           eps=cfg.noise_eps)

                # [1] remove by color
                if len(self.__table_color) > 1:
                    pcd = remove_by_color(pcd,
                                          bounds=self.__table_color[i])
                else:
                    pcd = remove_by_color(pcd,
                                          bounds=self.__table_color[0])

                # [2] apply other processes
                if PROC_EACH:
                    pcd = self.filter(pcd, joint_state,
                                      ws=False,
                                      table=False)

                if pcd_combined is None:
                    pcd_combined = pcd
                else:
                    pcd_combined = pcd_combined.append(pcd)
        else:
            if True:
                # with cf.ThreadPoolExecutor(max_workers=3) as executor:
                if True:
                    executor = self.__executor
                    future_to_proc = {executor.submit(self.proc_1, pcds[i], joint_state,
                                    self.__table_color[i]) : i
                                    for i in range(3)}
                    pcds = [None for _ in range(3)]
                    for future in cf.as_completed(future_to_proc):
                        index = future_to_proc[future]
                        pcds[index] = future.result()
                    pcd_combined = pcds[0]
                    for pcd in pcds[1:]:
                        if pcd.is_empty():
                            continue
                        pcd_combined = pcd_combined.append(pcd)
            else:
                pcds = [self.proc_1(pcds[i], joint_state,
                                    self.__table_color[i]) for i in range(3)]
                pcd_combined = pcds[0]
                for pcd in pcds[1:]:
                    if pcd.point.positions.shape[0] <= 0:
                        continue

                    # if pcd.is_empty():
                    #     continue
                    pcd_combined = pcd_combined.append(pcd)

        # TODO consider
        # downsampling the final point cloud.
        pcd = pcd_combined

        if not PROC_EACH:
            pcd = self.filter(pcd, joint_state)

        # pcd = pcd.voxel_down_sample(voxel_size = 0.005)
        # o3d.visualization.draw(pcd)

        pcd = remove_noise(pcd, radius=0.01,
                           min_num_points=16)
        pcd = select_largest(pcd, eps=0.025,
                             min_points=64  # works well for convex-ish objects
                             # min_points=16
                             )
        # o3d.visualization.draw(pcd)

        out = pcd  # .cpu().to_legacy()
        self.__prev_center = np.asarray(out.get_center().cpu().numpy()).ravel()
        # print('..', self.__prev_center)
        return out


def main():

    # with open('/tmp/test_color_seg/001.pkl', 'rb') as fp:
    with open('/home/user/Documents/color_table/0.pkl', 'rb') as fp:
        data = pickle.load(fp)
        # pcd = data['pcd']
        pcd = data['pre_seg_color']

    # pcd = remove_outside_bounds(pcd, upper=[0.68, 0.3, 0.01],
    #                         lower=[0.29, -0.28, -0.03])
    pcd = remove_by_color(pcd)

    # pcd = remove_outside_bounds(pcd)
    o3d.visualization.draw_geometries([pcd.cpu().to_legacy()])


if __name__ == '__main__':
    main()
