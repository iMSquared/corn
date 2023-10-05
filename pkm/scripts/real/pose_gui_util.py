import open3d as o3d
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from cho_util.math import transform as tx
import copy 
from sample_goal_pose import normalize_rotation
from pkm.real.track.icp_tracker_tensor import pairwise_normalized_t_plane_registration
import torch as th


def get_pcd_and_coordinate(cloud, pose):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        cloud[..., :3])
    if cloud.shape[-1]>3:
        pcd.colors = o3d.utility.Vector3dVector(
                    cloud[..., 3:])
    pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2)
    pose_mesh.transform(pose)
    return pcd, pose_mesh

def make_o3dt_cloud(cloud, device="cuda:0"):
    o3dt_cloud = o3d.t.geometry.PointCloud(o3d.core.Device(device))
    o3dt_cloud.point.positions = o3d.core.Tensor.from_numpy(
                                cloud[..., :3]).to(o3d.core.Device(device)).to(o3d.core.Dtype.Float64)
    if cloud.shape[-1] == 6:
        o3dt_cloud.point.colors = o3d.core.Tensor.from_numpy(
                                cloud[..., 3:]).to(o3d.core.Device(device)).to(o3d.core.Dtype.Float64)
    return o3dt_cloud


def get_pose_and_coordinate(canonical, target, mode,
                            device="cuda:0",
                            postprocess:bool=True):
    
    canonical_pcd = make_o3dt_cloud(canonical)
    target_pcd = make_o3dt_cloud(target)

    canonical_pose = np.eye(4)
    canonical_pose[:3, 3] = canonical[..., :3].mean(axis = -2)

    pos_offset = (
        target[..., :3].mean(axis = -2)
        - canonical[..., :3].mean(axis = -2)
    )
    
    best = 0.
    best_transform = None
    for _ in range(100):
        trans_init = np.eye(4)

        rotmat = R.random().as_matrix()
        trans_init[:3, :3] = rotmat
        # trans_init[:3, 3] = pos_offset
        trans_init[:3, 3] = (
        
            target[..., :3].mean(axis = -2)
            - rotmat @ canonical[...,:3].mean(axis=-2)
        )
        # T(add_offset) T(rotate) T(subtract_mean)
        # d-Rc
        
        # o3d.visualization.draw([canonical_pcd.clone().transform(trans_init), target_pcd])
        # o3d.visualization.draw([
        #     canonical_pcd.to_legacy().transform(trans_init),
        #                         target_pcd.to_legacy()
        #                         ])
        
        trans_init = o3d.core.Tensor(trans_init,
                                dtype=o3d.core.Dtype.Float32,
                            device=o3d.core.Device(device))

        transform, fitness, _ = pairwise_normalized_t_plane_registration(
                                            canonical_pcd,
                                            target_pcd,
                                            trans_init,
                                            mode = mode,
                                            threshold = 0.01,
                                            max_iteration = 300
                                            )
        if fitness > best:
            best_transform = transform
            best = fitness
            print(fitness)
    if not postprocess:
        return best_transform
    pose_from_canonical = best_transform.cpu().numpy()
    pose_from_base = pose_from_canonical @ canonical_pose
    pcd, pose_mesh = get_pcd_and_coordinate(target, pose_from_base)
    return pose_from_base, pcd, pose_mesh

class KeyCallbackViewer:
    def __init__(self,
                 mode='normal',
                 device="cuda:0") -> None:
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.cur_cloud = None
        self.pose_mesh = None
        self.transform = None

        self.quit = False
        self.T_v = None
        self.overlay = None
        self.canonical = None 
        self.canonical_pose = None

        self.poses = None
        self.index = None

        self.device = device
        self.mode = mode

        table = trimesh.creation.box((0.4, 0.5, 0.4))
        table = table.apply_translation((0.5, 0.0, -0.2))
        table_ls = o3d.geometry.LineSet.create_from_triangle_mesh(
            table.as_open3d)
        table_ls.paint_uniform_color([0, 0, 0])
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.vis.add_geometry(table_ls)
        self.vis.add_geometry(origin)    

        self.vis.register_key_callback(68, self.inc_x)  # Deq
        self.vis.register_key_callback(65, self.dec_x)  # A
        self.vis.register_key_callback(87, self.inc_y)  # W
        self.vis.register_key_callback(83, self.dec_y)  # S
        self.vis.register_key_callback(90, self.inc_z)  # Z
        self.vis.register_key_callback(88, self.dec_z)  # X
        self.vis.register_key_callback(262, self.inc_roll)  # RIGHT
        self.vis.register_key_callback(263, self.dec_roll)  # LEFT
        self.vis.register_key_callback(265, self.inc_pitch)  # UP
        self.vis.register_key_callback(264, self.dec_pitch)  # DOWN
        self.vis.register_key_callback(69, self.inc_yaw)  # E
        self.vis.register_key_callback(81, self.dec_yaw)  # Q
        self.vis.register_key_callback(79, self.out) # O to out 
        self.vis.register_key_callback(85, self.run_icp)

    def out(self, vis):
        self.quit = True

    def run_icp(self, vis):

        print(self.transform, self.cur_cloud)
        if self.transform is None or self.cur_cloud is None:
            return
        print("Runicp")
        if th.is_tensor(self.canonical):
            canonical = self.canonical.numpy()
        else:
            canonical=self.canonical

        target_points = np.asarray(self.cur_cloud.points).astype(np.float32)
        target_color = np.asarray(self.cur_cloud.colors)
        target = np.concatenate([target_points, target_color], -1).astype(np.float32)

        canonical_pcd = make_o3dt_cloud(canonical)
        target_pcd = make_o3dt_cloud(target)

        pose_from_canonical = self.transform @ tx.invert(self.canonical_pose)
        
        
        pose_from_canonical = o3d.core.Tensor(pose_from_canonical,
                                dtype=o3d.core.Dtype.Float32,
                            device=o3d.core.Device(self.device))

        transform, fitness, _ = pairwise_normalized_t_plane_registration(
                                            canonical_pcd,
                                            target_pcd,
                                            pose_from_canonical,
                                            mode = self.mode,
                                            threshold = 0.01,
                                            max_iteration = 300
                                            )
        
        print(fitness)
        pose_from_canonical = transform.cpu().numpy()
        pose_from_base = pose_from_canonical @ self.canonical_pose
        pcd, pose_mesh = get_pcd_and_coordinate(target, pose_from_base)
        
        self.vis.remove_geometry(self.overlay,
                                     reset_bounding_box=False)
        self.vis.remove_geometry(self.cur_cloud,
                                     reset_bounding_box=False)
        self.vis.remove_geometry(self.pose_mesh,
                                     reset_bounding_box=False)
        
        self.transform = pose_from_base
        self.cur_cloud = pcd
        self.pose_mesh = pose_mesh
        
        self.vis.add_geometry(self.cur_cloud,
                                     reset_bounding_box=False)
        self.vis.add_geometry(self.pose_mesh,
                                     reset_bounding_box=False)
        self.overlay = None
        if self.poses is not None:
            self.poses[self.index] = self.transform.copy()
        self.update_overay(None)
        

    def inc_x(self, vis):
        delta = np.eye(4)
        delta[0, 3] = 0.005
        if self.transform is not None:
            self.update_pose(delta, vis)

    def dec_x(self, vis):
        delta = np.eye(4)
        delta[0, 3] = -0.005
        if self.transform is not None:
            self.update_pose(delta, vis)

    def dec_y(self, vis):
        delta = np.eye(4)
        delta[1, 3] = -0.005
        if self.transform is not None:
            self.update_pose(delta, vis)

    def inc_y(self, vis):
        delta = np.eye(4)
        delta[1, 3] = 0.005
        if self.transform is not None:
            self.update_pose(delta, vis)
    def inc_z(self, vis):
        delta = np.eye(4)
        delta[2, 3] = 0.005
        if self.transform is not None:
            self.update_pose(delta, vis)

    def dec_z(self, vis):
        delta = np.eye(4)
        delta[2, 3] = -0.005
        if self.transform is not None:
            self.update_pose(delta, vis)
    
    def dec_roll(self, vis):
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('x', -5, degrees=True).as_matrix()
        
        # Transformation between global and child frame, child is 
        # frame that is parallel wrt to global frame, translated only
        # to the centre of the object
        if self.transform is not None:
            # T_pc = np.eye(4)
            # T_pc[:3, 3] = self.transform[:3, 3]
            # delta = T_pc @ delta_child @ tx.invert(T_pc)
            # self.update_pose(delta, vis)
            self.update_pose(delta_child, vis)
        

    def inc_roll(self, vis):
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('x', 5, degrees=True).as_matrix()

        if self.transform is not None:
        #     T_pc = np.eye(4)
        #     T_pc[:3, 3] = self.transform[:3, 3]
        #     delta = T_pc @ delta_child @ tx.invert(T_pc)
        #     self.update_pose(delta, vis)
            self.update_pose(delta_child, vis)
        

    def dec_pitch(self, vis):
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('y', -5, degrees=True).as_matrix()
        if self.transform is not None:
            # T_pc = np.eye(4)
            # T_pc[:3, 3] = self.transform[:3, 3]

            # delta = T_pc @ delta_child @ tx.invert(T_pc)
            # self.update_pose(delta, vis)
            self.update_pose(delta_child, vis)
        

    def inc_pitch(self, vis):
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('y', 5, degrees=True).as_matrix()

        if self.transform is not None:
            # T_pc = np.eye(4)
            # T_pc[:3, 3] = self.transform[:3, 3]

            # delta = T_pc @ delta_child @ tx.invert(T_pc)
            # self.update_pose(delta, vis)
            self.update_pose(delta_child, vis)
        


    def dec_yaw(self, vis):
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('z', -5, degrees=True).as_matrix()

        if self.transform is not None:
            # T_pc = np.eye(4)
            # T_pc[:3, 3] = self.transform[:3, 3]

            # delta = T_pc @ delta_child @ tx.invert(T_pc)
            # self.update_pose(delta, vis)
            self.update_pose(delta_child, vis)
        

    def inc_yaw(self, vis):
        delta_child = np.eye(4)
        delta_child[:3, :3] = R.from_euler('z', 5, degrees=True).as_matrix()

        if self.transform is not None:
            # T_pc = np.eye(4)
            # T_pc[:3, 3] = self.transform[:3, 3]

            # delta = T_pc @ delta_child @ tx.invert(T_pc)
            # self.update_pose(delta, vis)
            self.update_pose(delta_child, vis)
        

    def update_pose(self, delta, vis):
        update_frame = (self.transform @ 
                        tx.invert(delta) @ tx.invert(self.transform))
        update_frame[:3, :3] = normalize_rotation(update_frame[:3, :3])
        self.transform = self.transform @ tx.invert(delta)
        self.transform[:3, :3] = normalize_rotation(self.transform[:3, :3])
        self.pose_mesh = self.pose_mesh.transform(update_frame)
        vis.update_geometry(self.pose_mesh)
        self.update_overay(delta)

    def update_overay(self, delta=None):
        # new_pose = self.canonical_pose.copy()
        # new_pose[2, 3] += 0.2
        # pose_diff = new_pose @ tx.invert(self.transform)
        if self.overlay is not None:
            self.vis.remove_geometry(self.overlay,
                                     reset_bounding_box=False)
            if delta is not None:
                T_v= np.eye(4)
                T_v[2, 3] = 0.2
                can_pose = self.canonical_pose.copy()
                T_vc = T_v @ can_pose 
                true_move = T_vc @ delta @ tx.invert(T_vc)
                self.overlay.transform(true_move)
        else:
            self.overlay = copy.deepcopy(self.cur_cloud)
            can_pose = self.canonical_pose.copy()
            T_v= np.eye(4)
            T_v[2, 3] = 0.2
            can_pose = T_v @ can_pose
            pose_diff = can_pose @ tx.invert(self.transform)
            
            self.overlay.transform(pose_diff)
            # move_up = np.eye(4)
            # move_up[2, 3] += 0.2
            # self.overlay.transform(move_up)
            self.overlay_pose = pose_diff
            self.overlay_pose[:3, :3] = normalize_rotation(self.overlay_pose[:3, :3])
        # print("delta")
        # print(delta)
        # print("pose diff")
        # print(pose_diff)
        # self.overlay.transform(pose_diff)

        self.vis.add_geometry(self.overlay,
                              reset_bounding_box=False)

    def draw(self):
        self.vis.add_geometry(self.cur_cloud)
        self.vis.add_geometry(self.pose_mesh)
        self.update_overay(None)
        while not self.quit:
            self.vis.poll_events()
            self.vis.update_renderer()
        self.vis.destroy_window()

    def run(self):
        while not self.quit:
            self.vis.poll_events()
            self.vis.update_renderer()
