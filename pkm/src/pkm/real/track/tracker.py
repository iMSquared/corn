
import sys
import numpy as np
import cv2
# sys.path.append("/home/user/workspace/corn/pkm/scripts/real/seg")
# from segmenter import Segmenter

import time
import pyrealsense2 as rs
import ruamel.yaml
from polymetis import RobotInterface, GripperInterface
yaml = ruamel.yaml.YAML()

from pkm.util.vis.img import digitize_image
sys.path.append("/home/user/workspace/BundleSDF")
from bundlesdf import BundleSdf

class Tracker:
  def __init__(self, 
               cfg_path,
               intrinsic_mat, 
               max_BA_frames=3,
               resize=400,
               max_trans_neighbor=0.10,
               max_rot_deg_neighbor=60):
        
        cfg_bundletrack = yaml.load(open(cfg_path))

        cfg_bundletrack['SPDLOG'] = '0'#int(args.debug_level)
        cfg_bundletrack['depth_processing']["zfar"] = 1
        cfg_bundletrack['depth_processing']["percentile"] = 95
        # cfg_bundletrack['erode_mask'] = 3
        cfg_bundletrack['erode_mask'] = 0
        # cfg_bundletrack['debug_dir'] = out_folder+'/'
        # cfg_bundletrack['bundle']['max_BA_frames'] = 10
        cfg_bundletrack['bundle']['max_BA_frames'] = 3
        cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03
        # cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
        cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.10
        cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
        cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
        cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20
        cfg_bundletrack['feature_corres']['map_points'] = True
        # cfg_bundletrack['feature_corres']['resize'] = 240
        cfg_bundletrack['feature_corres']['resize'] = 400
        # cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
        cfg_bundletrack['feature_corres']['rematch_after_nerf'] = False
        cfg_bundletrack['keyframe']['min_rot'] = 5
        cfg_bundletrack['ransac']['inlier_dist'] = 0.01
        cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
        # cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
        cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.10
        # cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
        cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 60
        # cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
        cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 999
        # cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10
        cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 180
        cfg_bundletrack['ransac']['min_match_after_ransac'] = 5
        cfg_bundletrack['p2p']['max_dist'] = 0.02
        cfg_bundletrack['p2p']['max_normal_angle'] = 45
        cfg_bundletrack['SPDLOG'] = 0




        cfg_bundletrack['bundle']['max_BA_frames'] = max_BA_frames
        cfg_bundletrack['feature_corres']['resize'] = resize
        cfg_bundletrack['ransac']['max_trans_neighbor'] = max_trans_neighbor
        cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = max_rot_deg_neighbor

        # These values need to be set sufficiently large
        cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 60
        cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 999

        # Eroded images already given
        cfg_bundletrack['erode_mask'] = 0

        # We don't use NERF
        cfg_bundletrack['feature_corres']['rematch_after_nerf'] = False

        cfg_nerf = yaml.load(open(f"/home/user/workspace/BundleSDF/config.yml",'r'))
        
        cfg_nerf['continual'] = True
        cfg_nerf['trunc_start'] = 0.01
        cfg_nerf['trunc'] = 0.01
        cfg_nerf['mesh_resolution'] = 0.005
        cfg_nerf['down_scale_ratio'] = 1
        cfg_nerf['fs_sdf'] = 0.1
        cfg_nerf['far'] = cfg_bundletrack['depth_processing']["zfar"]
        cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
        cfg_nerf['notes'] = ''
        cfg_nerf['expname'] = 'nerf_with_bundletrack_online'
        cfg_nerf['save_dir'] = cfg_nerf['datadir']

        # Temporary folder that saves bundlesdf configuration
        cfg_track_dir = "/tmp/last_bundlesdf_experiment/config_ho3d.yml"
        yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))
        cfg_nerf_dir = "/tmp/last_bundlesdf_experiment/config.yml"
        yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))

        # Define tracker
        self.tracker = BundleSdf(cfg_track_dir=cfg_track_dir,
                                 cfg_nerf_dir=cfg_nerf_dir,
                                 start_nerf_keyframes=1e8,
                                 use_gui=False)
        self.count = 0
        self.K = intrinsic_mat



  def __call__(self, color_image:np.ndarray, depth_image:np.ndarray, mask_image:np.ndarray) -> np.ndarray:
    """
    Arg:
      color_image: array(H,W,3), uint8
      depth_image: array(H,W), float32
      mask_image: array(H,W), bool
    Return:
      current_pose: array(4,4), homogeneous transform matrix; object pose in camera frame.
    """
    self.count += 1
    return self.tracker.run_fast(color_image, 
                            depth_image, 
                            self.K, 
                            f"Img{self.count}",
                            mask=mask_image,
                            occ_mask=None,
                            pose_in_model=np.eye(4))
  
def main():
    from pkm.real.seg.segmenter import Segmenter

    robot = RobotInterface(
        ip_address="kim-MS-7C82",
    )
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device('233622074125')
    config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_1.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    cfg_1 = pipeline_1.start(config_1)
    s = cfg_1.get_device().query_sensors()[1]
    s.set_option(rs.option.exposure, 50)

    profile_1 = cfg_1.get_stream(rs.stream.color) # Fetch stream profile for depth stream
    intr_1 = profile_1.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    print(intr_1)
    intr_mat = np.array([[608.916000, 0.000000, 317.557000],
                        [0.000000, 608.687000, 253.814000],
                        [0.000000, 0.000000, 1.000000]])

    align_to = rs.stream.color
    align = rs.align(align_to)

    tracker = Tracker(cfg_path='/home/user/workspace/BundleSDF/BundleTrack/config_ho3d.yml',
                      intrinsic_mat=intr_mat)
    segmenter = Segmenter(Segmenter.Config())
    cur_time=time.time()
    old_time = cur_time
    count = 0
    length = 0.1
    axes = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=np.float32)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X: Red, Y: Green, Z: Blue
    while True:
        
        cur_time=time.time()
        if cur_time-old_time>1/10:
            old_time=cur_time  
            frames_1 = pipeline_1.wait_for_frames()
            frames_1 = align.process(frames_1)
            depth_image_1 = np.array(frames_1.get_depth_frame().get_data())/1.
            color_frame_1 = frames_1.get_color_frame()
            color_image_1 = np.asanyarray(color_frame_1.get_data())
            q_current = robot.get_joint_positions()
            seg_mask = segmenter(q_current, depth_image_1/1000, color_image_1)
            seg_mask_u8 = seg_mask.astype(np.uint8)*255
            # seg_mask_u8.fill(255)

            # cv2.imshow('color', color_image_1)
            # cv2.imshow('depth', digitize_image(depth_image_1).astype(np.uint8))
            cv2.imshow('mask', seg_mask_u8)
            cv2.waitKey(1)

            ob_c = tracker(color_image_1, depth_image_1/1000.0, seg_mask_u8)
            axes_cam = (ob_c @ np.vstack((axes.T, np.ones(4))))[:3, :]
            # Project 3D points to 2D
            axes_2d = (tracker.K @ axes_cam).T
            axes_2d = axes_2d[:, :2] / axes_2d[:, 2:]
            # Draw axes
            for i in range(1, 4):
                # Draw line from origin to each axis tip
                cv2.line(color_image_1, tuple(axes_2d[0].astype(int)), tuple(axes_2d[i].astype(int)), colors[i-1], 2)
            # Save output image
            cv2.imshow('color', color_image_1)
            cv2.waitKey(1)

            print(f"Current pose in {count} frame: {ob_c }")
            count += 1
        time.sleep(1 / 2000)

   
if __name__ == "__main__":
    main()
  

  
            
