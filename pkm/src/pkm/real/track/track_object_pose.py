import numpy as np
import open3d as o3d
import pickle
from scipy.spatial.transform import Rotation as R
import torch
import matplotlib.pyplot as plt
import os
import time
import pygicp
import pandas as pd
import copy


# Need to scale so that it fits almost 3*3 meter(in case of using GeoTrnsformer)
# i.e. 3.0 for geotransformer
SCALE:float = 1.0
# METHOD = "GENICP"
# METHOD = "ICP"
METHOD = "FGICP"
EPISODE:int = 77
FREQ_LIST = [60, 30, 20, 15, 10]
VISUALIZE = False
# FREQ_LIST = [30]

# If we use heuristic, in case of the match fitness is under the threshold,
# It uses current point cloud and previous point cloud from two timesteps 
# before to calculate the transformation
USE_HEURISTIC = False
FITNESS_THRESHOLD = 0.8

if METHOD == "GEO":
    '''
        WARNING: If you want to use geotransformer in order to find 
        the registration, you should clone the repository:
        https://github.com/qinzheng93/GeoTransformer.git
    '''
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda
    from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
    from geotransformer.utils.registration import compute_registration_error
    from config import make_cfg
    from model import create_model

    

def pose_to_transform_matrix(pose):
    '''
        Converts 7d vector pose into 4 by 4 transformation matrix

        Args:
            pose: 7-dim numpy array, first 3 is position vector,
            and last 4 is the quaternion
    '''
    # Extract position and quaternion
    position = np.array(pose[0:3])
    quaternion = np.array(pose[3:7])

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Create 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = position

    return transformation_matrix

def find_transform_geo(model, cfg, ref_points, src_points):
    '''
        Find transformation between given reference and source point cloud,
        using the GeoTransformer
        Args:
            model: GeoTransformer model
            cfg: GeoTransformer configuration
            ref_points: reference point clouds, in numpy ndarray
            src_points: source point clouds, in numpy ndarray
        Returns:
            transformation: 4 by 4 transformation matrix, from src to ref
    '''

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])
    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "transform": np.eye(4).astype(np.float32)
    }
    # prepare data
    # neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    neighbor_limits = [8,8,8,8]  

    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )
    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)
    return output_dict["estimated_transform"]


def find_transform_fgicp(prev_cloud, curr_cloud):
    '''
        Find transformation between given reference and source point cloud,
        using the fast-GICP
        Args:
            prev_cloud: previous point clouds, in numpy ndarray
            curr_cloud: current point clouds, in numpy ndarray
        Returns:
            transformation: 4 by 4 transformation matrix, from prev to curr
    '''
    # Using the function interface
    matrix = pygicp.align_points(target= curr_cloud, 
                                 source= prev_cloud,
                                #  method = "GICP",
                                 method = "VGICP_CUDA",
                                 k_correspondences = 15,
                                 max_correspondence_distance = 0.05,
                                 neighbor_search_method = "DIRECT_1",
                                 neighbor_search_radius = 0.1)

    # Using the class interface
    # gicp = pygicp.FastGICP()
    # gicp.set_input_target(curr_cloud)
    # gicp.set_input_source(prev_cloud)
    # gicp.set_correspondence_randomness(15)
    # gicp.set_max_correspondence_distance(0.1)
    # # gicp.set_rotation_epsilon(2e-3)

    # matrix = gicp.align()
    

    # Return the transformation matrix
    return matrix

def find_transform_icp(prev_cloud, curr_cloud):
    '''
        Find transformation between given reference and source point cloud,
        using the ICP
        Args:
            prev_cloud: previous point clouds, in numpy ndarray
            curr_cloud: current point clouds, in numpy ndarray
        Returns:
            transformation: 4 by 4 transformation matrix, from prev to curr
            fitness score: overlapping area (# of inlier correspondences 
                / # of points in source)
            rmse: RMSE of all inlier correspondences
    '''
    # Convert numpy arrays to open3d point cloud objects
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(prev_cloud)
    target.points = o3d.utility.Vector3dVector(curr_cloud)

    # Apply ICP
    threshold = 0.05  # Set this to an appropriate value depending on your data
    trans_init = np.eye(4)
    relative_fitness = 0.000001
    relative_rmse = 0.000001
    max_iteration = 30

    # Estimate normal
    target.estimate_normals()
   
    reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), # For Point-to-point metric
            # o3d.pipelines.registration.TransformationEstimationPointToPlane(), # For Point-to-plane metric
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness,
                                       relative_rmse,
                                       max_iteration))
    

    # Return the transformation matrix
    return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

def find_transform_genicp(prev_cloud, curr_cloud):
    '''
        Find transformation between given reference and source point cloud,
        using the generalized ICP
        Args:
            prev_cloud: previous point clouds, in numpy ndarray
            curr_cloud: current point clouds, in numpy ndarray
        Returns:
            transformation: 4 by 4 transformation matrix, from prev to curr
            fitness score: overlapping area (# of inlier correspondences 
                / # of points in source)
            rmse: RMSE of all inlier correspondences
    '''
    # Convert numpy arrays to open3d point cloud objects
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(prev_cloud)
    target.points = o3d.utility.Vector3dVector(curr_cloud)

    # Apply ICP
    threshold = 0.05  # Set this to an appropriate value depending on your data
    trans_init = np.eye(4)
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

    # Return the transformation matrix
    return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

    

def normalize_two_cloud(reference_cloud, target_cloud):
    '''
        Performs normalization, given pair of the two cloud
        Args:
            reference_cloud: the reference cloud to be normalized
            target_cloud: target cloud to be normalized
        Return:
            norm_transform: 4 by 4 transformation matrix, that does
                normalization transform(shifting to the center)
            ref_cloud_norm: normalized reference cloud
            tar_cloud_norm: normalized target cloud
    '''
    mean_point = reference_cloud.mean()
    ref_cloud_norm = reference_cloud - mean_point
    tar_cloud_norm = target_cloud - mean_point
    # norm_transform = np.eye(4)
    norm_transform = np.eye(4)
    norm_transform[:3, 3] = -mean_point
    return norm_transform, ref_cloud_norm, tar_cloud_norm

def normalize_two_cloud_v2(reference_cloud, target_cloud):
    '''
        Performs normalization, given pair of the two cloud
        Args:
            reference_cloud: the reference cloud to be normalized(o3d pointcloud)
            target_cloud: target cloud to be normalized(o3d pointcloud)
        Return:
            norm_transform: 4 by 4 transformation matrix, that does
                normalization transform(shifting to the center)
            ref_cloud_norm: normalized reference cloud
            tar_cloud_norm: normalized target cloud
    '''
    # Scaling
    # SCALE = 10.0
    reference_cloud = copy.deepcopy(reference_cloud)
    target_cloud = copy.deepcopy(target_cloud)

    # reference_cloud.scale(scale=SCALE, center=np.array([0., 0., 0.]))
    # target_cloud.scale(scale=SCALE, center=np.array([0., 0., 0.]))

    # mean_point, _ = reference_cloud.compute_mean_and_covariance()
    mean_point = reference_cloud.get_center()
    ref_cloud_norm = reference_cloud.translate(-mean_point)
    tar_cloud_norm = target_cloud.translate(-mean_point)
    # norm_transform = np.eye(4)
    norm_transform = np.eye(4)
    norm_transform[:3, 3] = -mean_point 
    
    # scailing
    # norm_transform[3, 3] *= SCALE
    

    return norm_transform, ref_cloud_norm, tar_cloud_norm


def normalize_two_cloud_v3(reference_cloud, target_cloud):
    '''
        Performs normalization, given pair of the two cloud
        Args:
            reference_cloud: the reference cloud to be normalized(o3d tensor pointcloud)
            target_cloud: target cloud to be normalized(o3d tensor pointcloud)
        Return:
            norm_transform: 4 by 4 transformation matrix, that does
                normalization transform(shifting to the center)
            ref_cloud_norm: normalized reference cloud
            tar_cloud_norm: normalized target cloud
    '''
    # Scaling
    # SCALE = 10.0
    reference_cloud = reference_cloud.clone()#copy.deepcopy(reference_cloud)
    target_cloud = target_cloud.clone()#copy.deepcopy(target_cloud)

    # reference_cloud.scale(scale=SCALE, center=np.array([0., 0., 0.]))
    # target_cloud.scale(scale=SCALE, center=np.array([0., 0., 0.]))

    mean_point = reference_cloud.get_center()
    ref_cloud_norm = reference_cloud.translate(-mean_point)
    tar_cloud_norm = target_cloud.translate(-mean_point)
    # norm_transform = np.eye(4)
    norm_transform = o3d.core.Tensor.eye(4,device=reference_cloud.device)
    norm_transform[:3, 3] = -mean_point 
    
    # scailing
    # norm_transform[3, 3] *= SCALE
    
    

    return norm_transform, ref_cloud_norm, tar_cloud_norm
    # return norm_transform, reference_cloud, target_cloud
    


def predict_poses(cloud_trajectory, initial_pose = np.eye(4), timestep = 1):
    '''
        Given the sequence of the point cloud and the initial pose,
        Returns the list of the estimated pose.
        Args:
            cloud trajectory: sequences of the point cloud, which is 
                4 by 4 transformation matrix(numpy ndarray)
            initial pose: Initial pose of the object that want to track,
                which is 4 by 4 transformation matrix(numpy ndarray)
            timestep(optional): timestep that wants to calculate transformation.
                Default value is set to 1
        Return:
            estimated_trajectory: sequences of the estimated poses,
                which is 4 by 4 transformation matrix(numpy ndarray)
            fitness_list: List of the fitness of each transformation
                (Only returns for the ICP or GENICP)
            rmse_list: List of the RMSE of each transformation
                (Only returns for the ICP or GENICP)
    '''
    # print(cloud_trajectory.shape)
    if METHOD == "GEO":
        # prepare model
        cfg = make_cfg()
        # path of the pretrained model
        model_path = "/home/user/corn/pkm/scripts/toy/pose_tracking/GeoTransformer/weights/geotransformer-3dmatch.pth.tar"
        model = create_model(cfg).cuda()
        model.eval()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict["model"])

    # Estimate the trajectory from the cloud trajectory
    estimated_trajectory = []
    estimated_trajectory.append(initial_pose)

    # List of fitness and rmse
    fitness_list = [1.0]
    rmse_list = [0]

    # Measuring time consumption
    start_time = time.time()

    for i in range(timestep, len(cloud_trajectory), timestep):
        
        # Perform normalization
        norm_transform, prev_cloud_norm, curr_cloud_norm = \
            normalize_two_cloud(cloud_trajectory[i-timestep], cloud_trajectory[i])

        if METHOD == "GEO":
            transform = find_transform_geo(model, cfg,
                                    curr_cloud_norm,
                                    prev_cloud_norm)
        elif METHOD == "ICP":
            (transform, fitness, rmse) = find_transform_icp(prev_cloud_norm,curr_cloud_norm)
            fitness_list.append(fitness)
            rmse_list.append(rmse)
        elif METHOD == "GENICP":
            (transform, fitness, rmse) = find_transform_genicp(prev_cloud_norm,curr_cloud_norm)
            fitness_list.append(fitness)
            rmse_list.append(rmse)
        elif METHOD == "FGICP":
            # (transform, fitness, rmse) = find_transform_fgicp(prev_cloud_norm,curr_cloud_norm)
            # fitness_list.append(fitness)
            # rmse_list.append(rmse)
            transform = find_transform_fgicp(prev_cloud_norm,curr_cloud_norm)

        if(USE_HEURISTIC and fitness_list[-1] < FITNESS_THRESHOLD):
            # calculate transformation with before 2 timestep
            norm_transform, prev_cloud_norm, curr_cloud_norm = \
                normalize_two_cloud(cloud_trajectory[i- 2* timestep], cloud_trajectory[i])

            if METHOD == "ICP":
                (transform, fitness, rmse) = find_transform_icp(prev_cloud_norm,curr_cloud_norm)
            elif METHOD == "GENICP":
                (transform, fitness, rmse) = find_transform_genicp(prev_cloud_norm,curr_cloud_norm)
            elif METHOD == "FGICP":
                transform = find_transform_fgicp(prev_cloud_norm,curr_cloud_norm)
            
            # update fitness and rmse
            fitness_list[-1] = fitness
            rmse_list[-1] = rmse
            # using before 2 timestep
            transform = np.linalg.inv(norm_transform)@transform@norm_transform
            estimated_trajectory.append(transform@estimated_trajectory[-1])

        else:
            transform = np.linalg.inv(norm_transform)@transform@norm_transform
            estimated_trajectory.append(transform@estimated_trajectory[-1])
    
    # print time consumption
    end_time = time.time()
    average_time = (end_time - start_time) * timestep / len(cloud_trajectory)
    print(f"Average time per iteration: {average_time} seconds, for timestep: {timestep}")



        
    estimated_trajectory = np.array(estimated_trajectory)

    return estimated_trajectory, fitness_list, rmse_list


def find_trans_rot_dist(estimated_trajectory, gt_poses, timestep = 1):
    '''
        Finds the translational error and the rotational error, given
        the sequences of the estimated poses and the ground-truth poses

        Args:
            estimated_trajectory: sequence of the estimated poses(numpy ndarray)
            gt_poses: sequence of the ground-truth poses(numpy ndarray)
            timestep: the timestep used for calculation of the estimated poses
        Return:
            translation_error: list of the translation error(in meter)
            rotation_error: list of the rotational error(in degree)
        

    '''
    # Compute the translation error
    estimated_translation = estimated_trajectory[:, :3, 3]
    gt_translation = gt_poses[::timestep, :3, 3]

    assert estimated_translation.shape[0] == gt_translation.shape[0]

    translation_error = np.linalg.norm(estimated_translation - gt_translation, axis=-1)

    # Compute the rotation error
    estimated_rotation = estimated_trajectory[:, :3, :3]
    gt_rotation = gt_poses[::timestep, :3, :3]
    rotation_error = []
    for est_rot, gt_rot in zip(estimated_rotation, gt_rotation):
        rotation_diff = np.dot(est_rot, gt_rot.T)
        angle_diff = np.arccos((np.trace(rotation_diff) - 1) / 2)
        rotation_error.append(np.degrees(angle_diff))  # Convert to degrees
    
    return translation_error, rotation_error

def plot_errors(errors, timesteps, title, yrange = None):
    '''
        Saves the error plot, in the real-time
    '''
    # Check if errors and timesteps have the same length
    if len(errors) != len(timesteps):
        raise ValueError("Errors and timesteps must have the same length")

    # Create a new figure
    plt.figure(figsize=(10, 5))
    plt.title(title)
    if yrange is not None:
        ax = plt.gca()
        ax.set_ylim(yrange)

    # Plot each error list
    for i in range(len(errors)):
        # Create an array of times for the x-axis
        times = [t * timesteps[i] * (1/60) for t in range(1, len(errors[i]) + 1)]

        # Plot the errors for this timestep
        plt.plot(times, errors[i], label=f"Frequency {60/timesteps[i]}hz")


    # Add legend and labels
    plt.legend()
    plt.xlabel('Timestep')

    # Save the figures
    directory = "./plot/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + title + '.png')  # Save the plot


def load_from_simulation(episode_num):
    # Load point clouds and gt poses from the simulated data
    filename = "/home/user/corn/pkm/scripts/toy/pose_tracking/real_data/dataset/Sim_data/log-episodes/" \
        + "/fail/" + f"{episode_num:05}" + ".pkl"

    with open(str(filename), 'rb') as fp:
        d = pickle.load(fp)
    cloud_trajectory = []
    gt_poses = []

    for step, data in enumerate(d):
        action, observation, reward = data
        # object_pose = observation['object_state'][..., 0:7]
        # partial_cloud = observation['partial_cloud']

        # Rescailing
        object_pose = observation['object_state'][..., 0:7]
        object_pose[0:3] *= SCALE
        partial_cloud = observation['partial_cloud'] * SCALE
        
        # Generate pcd object
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(partial_cloud)
        # # print(object_pose.shape)  # (512, 3)
        # # print(partial_cloud.shape)  # (7, )
        # cloud_trajectory.append(pcd)

        # convert numpy array to pandas dataframe
        df = pd.DataFrame(partial_cloud, columns=['x', 'y', 'z'])

        # remove duplicates
        df = df.drop_duplicates()

        # convert dataframe back to numpy array
        partial_cloud = df.values
        

        cloud_trajectory.append(partial_cloud)
        gt_poses.append(pose_to_transform_matrix(object_pose))
    
    # print(len(cloud_trajectory))
    # For point cloud visualization
    # original_vis = [cloud for i, cloud in enumerate(cloud_trajectory) if i%30 == 0]
    # o3d.visualization.draw_plotly(original_vis)
    gt_poses= np.array(gt_poses)
    # cloud_trajectory= np.array(cloud_trajectory)

    # assert(np.isnan(cloud_trajectory).any()== False and np.isinf(cloud_trajectory).any() == False)
    
    return cloud_trajectory, gt_poses

def load_from_real_robot():
    # pose_trajectory = np.load('pose_trajectory.npy')
    # with open('point_clouds.pkl', 'rb') as f:
    #     cloud_trajectory = pickle.load(f)
    descriptor = "dryer"
    file_path = "/home/user/corn/pkm/scripts/toy/pose_tracking/real_data/dataset/Real_robot_data/" 
    
    cloud_trajectory = []
    for i in range(50):
        # Create the filename
        # filename = f'./point_clouds/point_cloud_{i}.pcd'
        filename = file_path + descriptor + f'/point_cloud_{i}.pcd'

        # Load the point cloud from the file
        pc = o3d.io.read_point_cloud(filename)

        # Add the point cloud to the list
        cloud_trajectory.append(pc)
    # o3d.visualization.draw_plotly(cloud_trajectory[0:40])
    return cloud_trajectory

def track_trajectory(cloud_trajectory, 
                          gt_poses, 
                          freq_list,
                          default_freq):
    '''
        Track the trajectory of the object, with multiple frequencies
        then returns the list of errors, and matching scores
        Args:
            cloud_trajectory: sequence of point cloud(numpy ndarray)
            gt_poses: Sequences of an object pose, each pose is 4 by 4
                transformation matrix(numpy ndarray)
            freq_list: list of frequencies that wants to investigate
                It should be divisor of the default frequency
            default_freq: Frequency that the cloud trajectory has
                been recorded
        Returns:
            trans_errors: list of translational errors
            rot_erros: list of rotational errors
            fitness_list: list of fitness results
            rmse_list: list of rmse results
            timestep_list: list of timesteps, calculated by freq_list

    '''
    timestep_list = [int(default_freq/f) for f in freq_list]
    # error_list
    trans_errors = []
    rot_errors = []

    estimated_trajectories = []
    fitness_list = []
    rmse_list = []

    for t in timestep_list:
        if METHOD == "GENICP" or METHOD == "ICP":
            (estimated_trajectory, fitness, rmse) = predict_poses(cloud_trajectory,
                                                        initial_pose= gt_poses[0],
                                                        timestep = t)
            estimated_trajectories.append(estimated_trajectory)
            fitness_list.append(fitness)
            rmse_list.append(rmse)
        else:
            estimated_trajectories.append(predict_poses(cloud_trajectory,
                                                        initial_pose= gt_poses[0],
                                                        timestep = t)[0])

        trans_error, rot_error = find_trans_rot_dist(estimated_trajectories[-1],
                                                     gt_poses,
                                                     timestep= t)
        trans_errors.append(trans_error)
        rot_errors.append(rot_error)

    for error in trans_errors: # due to scailing
        error *= (1.0/SCALE)
    
    # finding the "error jump"
    for i in range(1, len(trans_errors[0])):
        if(trans_errors[0][i] - trans_errors[0][i-1] > 0.1):
            print(f"jumping timestep: {i-1}")

    if USE_HEURISTIC:
        for i in range(1, len(trans_errors[0])):
            if(fitness_list[0][i] < 0.90 ):
                print(f"low-fitness timestep: {i-1}")
    
    
    # Visualization for "jumping"
    # Generate list of pcd
    if VISUALIZE:
        original_point_clouds = []
        for cloud in cloud_trajectory:
            point_cloud= o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(cloud)
            original_point_clouds.append(point_cloud)
        transformed_point_clouds = transform_point_clouds(cloud_trajectory[0], estimated_trajectories[0])

        vis_list = [142, 143]
        transformed_vis = [cloud for i, cloud in enumerate(transformed_point_clouds) if i in vis_list]
        original_vis = [cloud for i, cloud in enumerate(original_point_clouds) if i in vis_list]
        
        o3d.visualization.draw_plotly(original_vis)
        o3d.visualization.draw_plotly(transformed_vis)
    
    return trans_errors, rot_errors, fitness_list, rmse_list, timestep_list



    





def transform_point_clouds(initial_point_cloud, poses):
    """
    Transform an initial point cloud by a sequence of poses.

    Args:
        initial_point_cloud: An Open3D PointCloud object.
        poses: A sequence of 4x4 homogeneous transformation matrices.

    Returns:
        A list of PointCloud objects, each transformed by a pose from the sequence.
    """
    transformed_point_clouds = []

    for pose in poses:
        # Make a copy of the initial point cloud
        point_cloud= o3d.geometry.PointCloud()

        # point_cloud.points = o3d.utility.Vector3dVector(initial_point_cloud.points)
        point_cloud.points = o3d.utility.Vector3dVector(initial_point_cloud)

        # Transform the point cloud by the pose
        point_cloud.transform(np.linalg.inv(poses[0]))
        point_cloud.transform(pose)

        # Change the color of the point cloud
        point_cloud.paint_uniform_color([1, 0.706, 0])

        # Add the transformed point cloud to the list
        transformed_point_clouds.append(point_cloud)

    return transformed_point_clouds


def main():
    # Loading the dataset
    # cloud_trajectory = load_from_real_robot()
    cloud_trajectory, gt_poses = load_from_simulation(EPISODE)
    # cloud_trajectory = np.array(cloud_trajectory)
    # cloud_trajectory = torch.from_numpy(cloud_trajectory).cuda()
    # gt_poses = torch.from_numpy(gt_poses).cuda()
    # gt_poses.to('cuda')


    # Frequency list
    freq_list = FREQ_LIST


    (trans_errors, rot_errors, fitness_list, rmse_list, timestep_list) = \
        track_trajectory(cloud_trajectory,
                            gt_poses,
                            freq_list,
                            default_freq = 60)    

    plot_errors(trans_errors, timestep_list, str(METHOD) + "_translation_error_episode_" + str(EPISODE),
                yrange= [0, 0.5])
    plot_errors(rot_errors, timestep_list, str(METHOD) + "_rotation_error_episode_" + str(EPISODE),
                yrange=[0, 90])
    # plot_errors(fitness_list, timestep_list, str(METHOD) + "_fitness_episode_" + str(EPISODE))
    # plot_errors(rmse_list, timestep_list, str(METHOD) + "_rmse_episode_" + str(EPISODE))


 

if __name__ == '__main__':
    main()
