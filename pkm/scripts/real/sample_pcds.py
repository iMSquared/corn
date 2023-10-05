import numpy as np
import open3d as o3d

import argparse
import copy
import torch as th
import pickle
import trimesh
import tkinter as tk
from pathlib import Path
import time 
import copy
from cho_util.math import transform as tx

from pkm.util.path import ensure_directory
from pkm.real.multi_perception_async import MultiPerception

from rt_cfg import RuntimeConfig
from sample_goal_pose import normalize_rotation
from pose_gui_util import KeyCallbackViewer, get_pose_and_coordinate

class sample_pcd(KeyCallbackViewer):
    def __init__(self,
                 mode,
                 obj_name,
                 rt_config):
        super().__init__()
    
        self.mode = mode
        self.obj_name = obj_name
        self.rt_cfg = rt_config


        # ip = 'kim-MS-7C82'
        ip=None
        perception_cfg = MultiPerception.Config(
                                            fps=60,
                                            img_width= 640,
                                            img_height = 480,
                                            tracker_type = 'multi-april',
                                            object='blue-holder',
                                            use_kf=False,
                                            # mode='thin',
                                            mode=mode,
                                            skip_april=True,
                                            ip= ip,
                                            debug=True
                                            )
        perception_cfg.segmenter.table_color_file=rt_config.table_color_file
        self.perception = MultiPerception(perception_cfg,
                                    device_ids=rt_config.cam_ids,
                                    extrinsics=rt_config.extrinsics,
                                    rt_cfg=rt_config
                                    )
        
        if self.perception.cfg.ip is not None:
            q_home = np.asarray(
                    [-0.0122, -0.1095, 0.0562, -2.5737, -0.0196, 2.4479, 0.0])
            self.perception.robot.move_to_joint_positions(q_home)
        else:
            self.perception.update_joint_states(
                np.asarray(
                    [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0],
                    dtype=np.float32))

       
        self.win = tk.Tk()
        self.listbox = tk.Listbox(self.win, selectmode=tk.SINGLE)
        self.listbox.pack()
        self.listbox.bind('<Double-Button-1>', self.__show_on_o3d)
        self.text = tk.Text(self.win)
        self.text.insert('end', 'canonical')
        self.text.pack()

        add_button = tk.Button(self.win,
                            text="Add Point Cloud",
                            command=self.__add_point_cloud)
        add_button.pack(pady=5)

        # Button to remove selected point cloud
        remove_button = tk.Button(self.win,
                                text="Remove Selected",
                                command=self.__remove_point_cloud)
        remove_button.pack(pady=5)

        save_button = tk.Button(self.win,
                                text="Save all pcds",
                                command=self.__save_point_cloud)
        save_button.pack(pady=5)

        canonical_button = tk.Button(self.win,
                                    text="Save canonical",
                                    command=self.__make_canonical)
        canonical_button.pack(pady=5)

        self.__pcds = []
        self.__keys = []
        self.poses = []
        self.overlay = None
        self.overlay_pose = None
        self.counter = 0
        self.is_canonical = True
        self.canonical = None
        self.index = None
    
    def run(self):
        while not self.quit:
            self.win.update()
            self.vis.poll_events()
            self.vis.update_renderer()

    def __make_canonical(self):
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "canonical")
        self.is_canonical = True

    def __save_point_cloud(self):
        path = self.rt_cfg.root+f'/object/{self.obj_name}/canonical.pkl'
        ensure_directory(Path(path).parent)

        with open(path, 'wb') as fp:
            canonical = {
                'pcd': self.canonical,
                'pose': self.canonical_pose
            }
            pickle.dump(canonical, fp)
        for i, pcd in enumerate(self.__pcds):
            path = self.rt_cfg.root+f'/object/{self.obj_name}/{i}.pkl'
            with open(path, 'wb') as fp:
                data ={
                    'pcd': pcd,
                    'pose': self.poses[i]
                } 
                pickle.dump(data, fp)
        self.quit=True
        self.win.destroy()
        self.vis.destroy_window()

    def __show_on_o3d(self, event):
        cs = self.listbox.curselection()
        if cs:
            index = cs[0]
            if self.cur_cloud is not None:
                self.vis.remove_geometry(self.cur_cloud)
                self.vis.remove_geometry(self.pose_mesh)
                self.vis.remove_geometry(self.overlay)
            self.index = index
            selectd = self.__pcds[index]
            pcd =  o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                selectd[..., :3].numpy())
            pcd.colors = o3d.utility.Vector3dVector(
                selectd[..., 3:].numpy())
            self.cur_cloud = pcd
            self.transform = self.poses[index]
            


            self.vis.add_geometry(self.cur_cloud)
            pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.2)
            pose_mesh.transform(self.transform)
            self.pose_mesh = pose_mesh
            self.vis.add_geometry(self.pose_mesh)
            
            self.update_overay(None)
            self.vis.poll_events()
            self.vis.update_renderer()

    def __add_point_cloud(self):
        out = None
        while out is None:
            out = self.perception.get_observations()
            time.sleep(0.001)
        pcd_colored_tensor, pose = out
        pcd_colored_tensor = pcd_colored_tensor.detach().cpu()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            pcd_colored_tensor[..., :3].numpy())
        pcd.colors = o3d.utility.Vector3dVector(
            pcd_colored_tensor[..., 3:].numpy())
        
        if self.is_canonical:
            self.is_canonical = False
            self.canonical = pcd_colored_tensor.clone()
            pose =  np.eye(4)
            pose[:3, 3] = pcd.get_center()
            self.canonical_pose = pose.copy()
            transform = np.eye(4)
            transform[2, 3] = 0.2 
            pcd.transform(transform)
            self.vis.add_geometry(pcd)

            transform[:3, 3] = pcd.get_center()
            pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2)
            pose_mesh.transform(transform)
            self.vis.add_geometry(pose_mesh)
        else:
            if self.cur_cloud is not None:
                self.vis.remove_geometry(self.cur_cloud)
                self.vis.remove_geometry(self.pose_mesh)
                self.vis.remove_geometry(self.overlay)
            self.__pcds.append(pcd_colored_tensor)
            self.__keys.append(self.counter)
            self.counter += 1
            (pose, 
            init_pcd, init_pose_mesh) = get_pose_and_coordinate(
                                        canonical=self.canonical.numpy(),
                                        target=pcd_colored_tensor.numpy(),
                                        mode=self.mode)
            self.pose_mesh = init_pose_mesh
            print(F'got pose = {pose}')
            self.poses.append(pose)
            
           
            self.cur_cloud = pcd
            self.transform = pose #<<
            self.vis.add_geometry(self.cur_cloud)
            self.vis.add_geometry(self.pose_mesh) # inconsistent with `pose`
            self.overlay = None
            self.update_overay(None)
            self.index = self.listbox.size()
        self.__update_listbox()

    def __remove_point_cloud(self):
        cs = self.listbox.curselection()
        if cs:
            index = cs[0]
            del self.__pcds[index]
            del self.__keys[index]
            self.__update_listbox()
        if self.cur_cloud is not None:
            self.vis.remove_geometry(self.cur_cloud)
            self.vis.remove_geometry(self.pose_mesh)
            self.vis.remove_geometry(self.overlay)
        self.cur_cloud = None
        self.pose_mesh = None
        self.index = None
        self.overlay = None
        self.overlay_pose = None

    def __update_listbox(self):
        # Clear the Listbox
        self.listbox.delete(0, tk.END)

        # Add point clouds to the Listbox
        for key in self.__keys:
            self.listbox.insert(tk.END, key)

        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "normal")
    
    def update_pose(self, delta, vis):
        if self.index is not None:
            update_frame = (self.transform @ 
                            tx.invert(delta) @ tx.invert(self.transform))
            update_frame[:3, :3] = normalize_rotation(update_frame[:3, :3])
            self.transform = self.transform @ tx.invert(delta)
            self.transform[:3, :3] = normalize_rotation(self.transform[:3, :3])
            self.poses[self.index] = self.transform.copy()
            self.pose_mesh = self.pose_mesh.transform(update_frame)
            vis.update_geometry(self.pose_mesh)
            self.update_overay(delta)


        
def main():
    parser = argparse.ArgumentParser(description='Collect pcd')
    parser.add_argument('-thin', 
                        action='store_true',
                        dest='thin_mode',
                    help='Turn on thin mode perception mode')
    parser.add_argument('-obj',
                        type=str, required=True,
                        dest='obj_name',
                        help='Name of the object to select')
    rt_cfg = RuntimeConfig()
    args = parser.parse_args()
    mode = 'thin' if args.thin_mode else 'normal'
    sampler = sample_pcd(
        mode=mode,
        obj_name=args.obj_name,
        rt_config=rt_cfg
    )
    sampler.run()

if __name__ == "__main__":
    main()
