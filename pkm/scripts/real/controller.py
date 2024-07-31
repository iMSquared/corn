#!/usr/bin/env python3

import isaacgym


from typing import Dict, Tuple, List, Optional
import time
import sys
import pickle
import copy
import subprocess
import open3d as o3d

import numpy as np
import torch
import torch as th
import cv2
from cho_util.math import transform as tx

from polymetis import RobotInterface, GripperInterface
import torchcontrol as toco

# from pkm.real.perception import Perception
# from pkm.real.multi_perception import MultiPerception
from pkm.real.multi_perception_async import MultiPerception

from pkm.models.rl.env_normalizer import EnvNormalizer
from pkm.models.rl.nets import PiNet

from pkm.util.path import RunPath, get_path
from pkm.util.math_util import (
    quat_from_axa,
    matrix_from_quaternion,
    invert_transform
)
from pkm.models.rl.v6.ppo import (
    STATE_KEY
)

TRAIN_PATH = get_path('../../../scripts/train')
sys.path.append(TRAIN_PATH)
from distill import StudentAgentRMA
sys.path.pop(-1)

from rt_cfg import RuntimeConfig
from load_policy import (load_configs,
                         load_checkpoints,
                         load_checkpoints_from_dir)

import argparse

EXPORT: bool = True
EE_FRAME = 'panda_tool'
AUG_PCD: bool = False
HACK_CLIP_STATE: bool = False


from full_pcd_util import (load_full_pcd)

@torch.jit.script
def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out.copy_(q)
    out[..., 3] = -out[..., 3]
    return out


@torch.jit.script
def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = torch.unbind(q1, dim=-1)
    x2, y2, z2, w2 = torch.unbind(q2, dim=-1)
    out = torch.empty_like(q1)
    out[...] = torch.stack([
        x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
        -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
        x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2,
        -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2], dim=-1)
    return out


@torch.jit.script
def axis_angle_from_quat(quat):
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference:
    # https://github.com/facebookresearch/pytorch3d/blob/bee31c48d3d36a8ea268f9835663c52ff4a476ec/pytorch3d/transforms/rotation_conversions.py#L516-L544

    axis = torch.nn.functional.normalize(quat[..., 0:3])

    half_angle = torch.acos(
        torch.clamp(quat[..., 3:], -1.0, 1.0))
    angle = (2.0 * half_angle + torch.pi) % (2 * torch.pi) - torch.pi
    return axis * angle


class GenomController(toco.PolicyModule):
    """
    Custom policy that executes a sine trajectory on joint 6
    (magnitude = 0.5 radian, frequency = 1 second)
    """
    logging: List[torch.Tensor]

    def __init__(self, time_horizon, hz,
                 robot_model: torch.nn.Module, dof_pose, kp, kd, **kwds):
        """
        Controller now can be used in three different mode
        1. Directly update the target dof pose from policy
        2. Update residual ee pose
            from policy and target dof pose is update by controller
        3. Update absolute ee target
            from policy and target dof pose is update by controller
        """
        self.ee_frame = kwds.pop('ee_frame', 'panda_tool')
        self.lerp = kwds.pop('lerp', False)

        super().__init__(**kwds)

        self.hz = hz
        self.time_horizon = time_horizon
        self.robot_model = robot_model

        # Initialize variables
        self.steps = 0
        self.timecount = 0

        self.source_dof_pose = torch.nn.Parameter(dof_pose)
        self.source_timecount = 0
        self.target_dof_pose = torch.nn.Parameter(dof_pose)

        self.res_ee = torch.nn.Parameter(
            torch.zeros(6)
        )

        self.kp = torch.nn.Parameter(kp)
        self.kd = torch.nn.Parameter(kd)
        self.u = torch.zeros(7)

        self.res_ee = torch.nn.Parameter(
            torch.zeros(6)
        )
        self.target_ee = torch.nn.Parameter(
            torch.zeros(7)
        )
        self.updated_res = torch.nn.Parameter(
            data=torch.zeros(1)
        )
        self.updated_abs = torch.nn.Parameter(
            data=torch.zeros(1)
        )

        self.logging = []

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]
        # mm = state_dict['mm'].reshape(7, 7).T

        if self.updated_res > 0:
            self.compute_dof_target_from_res(
                self.res_ee, q_current
            )
            self.updated_res[:] = 0.
            self.source_timecount = self.timecount
            self.source_dof_pose[...] = q_current
        elif self.updated_abs > 0:
            pos, quat = self.robot_model.forward_kinematics(q_current,
                                                            self.ee_frame)
            res = torch.zeros(6)
            res[:3] = self.target_ee[:3] - pos
            dq = quat_multiply(self.target_ee[..., 3:7],
                               quat_inverse(quat))
            res[3:] = axis_angle_from_quat(dq[None]).squeeze(0)
            self.compute_dof_target_from_res(
                res, q_current
            )
            self.updated_abs[:] = 0.
            self.source_timecount = self.timecount
            self.source_dof_pose[...] = q_current
            print(
                "***************************udpated*************************************")

        if self.timecount % 1 == 0:
            # mm = self.robot_model.compute_mm(q_current)
            # mm_offset = 0.2
            # mm[4][4] += mm_offset
            # mm[5][5] += mm_offset
            # mm[6][6] += mm_offset
            if self.lerp:
                weight = (self.timecount - self.source_timecount) / 200.0 + 0.7
                target = torch.lerp(self.source_dof_pose,
                                    self.target_dof_pose,
                                    min(max(weight, 0), 1))
            else:
                target = self.target_dof_pose.clone()
            # target[-1] = 0.
            # u=mm@(self.kp*(target-q_current)-self.kd*qd_current).unsqueeze(-1)
            # self.u = u.squeeze(-1)
            kp = self.kp.clone()
            # kp[..., 5] *= 1.2
            self.u = kp * (target - q_current) - self.kd * qd_current
            print("detla q7", (target - q_current)[-1])
            # print("tau j7: ", self.u[-1])
            # print(self.kp, target, self.kd)
            # print("-------------------------------------------------------")
            # print(target - q_current)
            # print(self.kd * qd_current, self.kp * (target - q_current))
            # print("-------------------------------------------------------")
            # print(self.u)
            # print("********************************************************")
            # self.u *= 0.0
        # print(
        #     "q_curent: ", q_current
        # )
        # print(
        #     "qd_curent: ", qd_current
        # )
        # Check termination
        if self.timecount > self.time_horizon:
            self.set_terminated()

        self.timecount += 1
        return {"joint_torques": self.u}

    def compute_dof_target_from_res(self, res_ee: torch.Tensor,
                                    q_current: torch.Tensor):
        jacobian = self.robot_model.compute_jacobian(
            q_current, self.ee_frame)
        lambda_val = 0.1
        jacobian_T = jacobian.T
        lambda_matrix = (lambda_val ** 2) * torch.eye(n=6)
        delta_dof_pos = jacobian_T @ torch.linalg.solve(
            jacobian @ jacobian_T + lambda_matrix,
            res_ee.unsqueeze(-1))
        delta_dof_pos = delta_dof_pos.squeeze(-1)
        # print("delta j7: ", delta_dof_pos[-1])
        # print("delta: ", delta_dof_pos)

        # == Format and send data. ==
        self.target_dof_pose = q_current + delta_dof_pos

# @torch.jit.script


def to_rot6d(x: torch.Tensor):
    R = matrix_from_quaternion(x)
    return R[..., :, :2].reshape(*R.shape[:-2], -1)

# @torch.jit.script


def to_rot6d_pose(x: torch.Tensor):
    return torch.cat([
        x[..., :3],
        to_rot6d(x[..., 3:7])
    ], dim=-1)

# @torch.jit.script


def convert_hand(x: torch.Tensor):
    return to_rot6d_pose(x)




class Policy:
    def __init__(self,
                 rt_cfg: RuntimeConfig,
                 cfgs,
                 update_hand_post_norm,
                 dagger:bool=False,
                 obj_name:Optional[str] = None,
                 thin:Optional[bool] = None,
                 april:Optional[bool] = True,
                 icp_reinit:bool=True,
                 device: str = "cuda:0") -> None:
        """
        Teacher config
        teacher ckpt
        student config
        student ckpt

        """
        self.rt_cfg = rt_cfg
        self.device = device
        self.dagger = dagger
        self.obs_shape = dict(cfgs['student'].shapes)
        self.obs_shape['partial_cloud'] = (3,)
        self.update_hand_post_norm = update_hand_post_norm
        # NOTE: HACK
        # For student with this setup
        # Normalizer get 7 dim obs for hand state
        if self.update_hand_post_norm:
            self.obs_shape['hand_state'] = (7,)

        self.update_6d_goal = (self.obs_shape['goal'] == 9)
        self.update_6d_hand = (self.obs_shape['hand_state'] == 9)
        print("Setup!!")
        print(self.obs_shape)
        print(self.update_6d_goal, self.update_6d_hand)

        self.joint_update_from_percepton = True
        self.control_mode = 'ee_abs'  # ee_abs, ee_rel, dof_target

        # == robot interface ==
        self.robot = RobotInterface(
            ip_address=rt_cfg.robot_ip,
        )

        self.gripper = GripperInterface(
            ip_address=rt_cfg.robot_ip,
        )

        # == policy stack ==
        self.normalizer = EnvNormalizer(
            cfgs['normalizer'],
            1,
            self.obs_shape)
        self.actor = None
        if not self.dagger:
            self.actor = PiNet(cfgs['policy'])
        print(cfgs['student'])
        self.rma = StudentAgentRMA(cfgs['student'], None, 'cuda:0')

        # == perception stack ==
        ip_address = (rt_cfg.robot_ip
                      if self.joint_update_from_percepton
                      else None)


        if thin is None: # read tcfg
            with open(rt_cfg.task_cfg_file, 'rb') as fp:
                tcfg = pickle.load(fp)
                drag = False
                for t, v in tcfg['task']:
                    if 'drag' in t:
                        drag = True
            mode = 'thin' if drag else 'normal'
        else: # CLI override
            mode = 'thin' if thin else 'normal'

        perception_cfg = MultiPerception.Config(
            ip=ip_address,
            fps=30,
            mode=mode,
            icp_reinit=icp_reinit,
            tracker_type='multi-april',
            # tracker_type='icp',
            # object='dispenser',
            # object='ceramic-cup',
            # object='chapa-bottle',
            # object='blue-holder',
            # object='black-cup',
            # object='tape',
            # object='gray_basket',
            object=obj_name,
            # object='coaster_holder',
            # object='wood_holder',
            # object='ham',
            # object = 'lipton',
            # object='candy',
            # object= 'toy_table',
            use_kf=False,
            skip_april=True
        )

        (self.perception_handle,
            self.shared_values,
            self.lock_joint,
            self.lock_cloud,
            # FIXME: optionally support single perception
            self.share_manager) = MultiPerception.start_mp(
            perception_cfg,
            rt_cfg.cam_ids,
            rt_cfg.extrinsics,
            rt_cfg
        )

    def setup(self, rotation, translation, ckpt_path):
        # load ckpt
        ckpts = load_checkpoints_from_dir(
            ckpt_path,
            is_dagger = self.dagger
        )

        # load env
        self.normalizer.load_state_dict(ckpts['normalizer'], strict=False)
        self.normalizer.to(self.device)
        self.normalizer.eval()

        # [optional] load state
        self.state = None
        if 'state' in ckpts:
            self.state=ckpts['state']
            self.state.to(self.device)
            self.state.eval()


        # load actor
        if not self.dagger:
            self.actor.load_state_dict(ckpts['policy'])
            self.actor.to(self.device)
            self.actor.eval()

        # load student
        self.rma.load_state_dict(ckpts['student'], strict=True)
        self.rma.to(self.device)
        self.rma.eval()


        # Prepare all tensors
        self.franka_offset = torch.as_tensor([-0.5, 0., 0.4],
                                             device=self.device)
        # this is right?
        self.franka_offset_pcd = torch.as_tensor([-0.5, 0., 0.4],
                                             device=self.device)
        
        self._init_tensor()

        self.goal_offset = np.eye(4)
        self.goal_offset[:3, :3] = rotation
        self.goal_offset[:3, -1] = translation

        self.rel_goal = np.eye(4)

        self.done = torch.ones(1, 1, dtype=bool,
                               device=self.device)
        self.goal = None

        # q_home = np.array([   -0.0050,     -0.5170,      0.0065,     -2.1196,      0.0004,
        #         2.0273,      0.7912])
        # q_home = np.array([0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -1.5708])
        # q_home = np.asarray([0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0])
        # q_home = np.asarray(
        #     [-0.0122, -0.1095, 0.0562, -2.5737, -0.0196, 2.4479, np.pi/2])
        q_home = np.asarray(
            [-0.0122, -0.1095, 0.0562, -2.5737, -0.0196, 2.4479, 0])

        self.gripper.goto(width=0., speed=0.05, force=1)
        self.robot.move_to_joint_positions(q_home)

        time.sleep(3)
        self.kp = torch.as_tensor([200.] * 4 + [100.] * 3)
        self.kd = torch.as_tensor([0.5] * 7) * torch.sqrt(self.kp)
        hz = self.robot.metadata.hz

        # setup controller
        self.controller = GenomController(
            time_horizon=5000 * hz,
            hz=hz,
            robot_model=self.robot.robot_model,
            dof_pose=self.robot.get_joint_positions(),
            kp=self.kp,
            kd=self.kd,
            ee_frame=EE_FRAME
        )

    def _init_tensor(self):
        # assume we have dimension in cfg
        # TODO have to be properly changed to right config at some time
        self.observation = {}
        for k, v in self.obs_shape.items():
            if not isinstance(v, Tuple):
                v = (v, )
            if k == 'partial_cloud':
                v = (512, 3)
            self.observation[k] = torch.zeros(1, *v, device=self.device)
        self.action = torch.zeros(20, device=self.device)
        self.prev_action = torch.zeros(20, device=self.device)

    def get_current_rel_goal(self, object_pose=None):
        if object_pose is None:
            object_pose = np.eye(4)

        if self.goal is None:
            self.goal = np.eye(4)
            self.goal[..., :3, :3] = self.goal_offset[..., :3, :3] @ \
                object_pose[..., :3, :3]
            self.goal[..., :3, -1] = self.goal_offset[..., :3, -1] +\
                object_pose[..., :3, -1]
            self.rel_goal = self.goal_offset.copy()

        if True:
            T0 = object_pose
            T1 = self.goal

            q0 = tx.rotation.quaternion.from_matrix(T0[:3, :3])
            q1 = tx.rotation.quaternion.from_matrix(T1[:3, :3])

            quat_multiply = tx.rotation.quaternion.multiply
            quat_inverse = tx.rotation.quaternion.inverse

            dq = quat_multiply(q1, quat_inverse(q0))
            # FIXME: hack to ensure sign consistency of quaternion
            if not self.update_6d_goal and dq[..., -1] > 0:
                dq *= -1
            dt = T1[..., 0:3, 3] - T0[..., 0:3, 3]
            self.rel_goal = np.concatenate([dt, dq])
        return self.rel_goal

    def run_policy(self, path=None):

        state_log = self.robot.send_torch_policy(
            self.controller, blocking=False)

        cur_time = time.time()
        old_time = cur_time
        is_first = True
        pcd0 = None
        cnt = 0

        fetch_obj_pose = True  # where to get this?
        obj_pose = None

        color = None
        depth = None
        success = False

        init_pcd = None
        # init_pose = None
        cur_step = 0
        while True:
            cur_time = time.time()
            # q_current = self.robot.get_joint_positions()
            # if self.async_perception:
            #     with self.lock_joint:
            #         self.shared_values['joint_state'][...] = q_current

            if cur_time - old_time > 1 / 10:
                old_time = cur_time
                if not self.joint_update_from_percepton:
                    q_current = self.robot.get_joint_positions()
                    # print('q_current', q_current)
                    qd_current = self.robot.get_joint_velocities()
                # q_current = self.robot.get_joint_positions()
                # print('q_current', q_current)
                # qd_current = self.robot.get_joint_velocities()
                # self.robot.get_ee_pose()

                # == Send joint positions ==
                # == Recv cloud / pose (possibly outdated) ==
                with self.lock_joint:
                    if not self.joint_update_from_percepton:
                        self.shared_values['joint_state'][...] = q_current
                        self.shared_values['has_joint_state'][...] = True
                    else:
                        q_current = self.shared_values['joint_state'][..., :7].copy(
                        )
                        qd_current = self.shared_values['joint_state'][..., 7:].copy(
                        )
                        q_current = torch.as_tensor(q_current)
                        qd_current = torch.as_tensor(qd_current)
                with self.lock_cloud:
                    if not self.shared_values['has_cloud']:
                        continue
                    pcd0 = np.copy(self.shared_values['cloud'])
                    if fetch_obj_pose:
                        obj_pose = np.copy(
                            self.shared_values['object_pose'])
                    if 'color' in self.shared_values:
                        # print('update color :)')
                        color = self.shared_values['color'].copy()
                        depth = self.shared_values['depth'].copy()

                if AUG_PCD:
                    if init_pcd is None:
                        init_pose_inv = np.linalg.inv(obj_pose)
                        init_pcd = pcd0.copy(
                        ) @ init_pose_inv[:3, :3].T + init_pose_inv[:3, 3]
                current_EE_pos, current_EE_ori = \
                    self.robot.robot_model.forward_kinematics(q_current, EE_FRAME)
                # == Postprocess pcd and goal ==
                if (not AUG_PCD) or (init_pcd is None):
                    pcd = torch.as_tensor(
                        pcd0, dtype=torch.float, device=self.device) + self.franka_offset_pcd[..., None, :]
                else:
                    # == augment PCD with extra points ==
                    aug_pcd = (
                        init_pcd @ obj_pose[:3, :3].T + obj_pose[:3, 3]
                    )
                    merged_pcd = np.concatenate([pcd0, aug_pcd], axis=0)
                    # FIXME: hardcoded `512`
                    subsample_indices = np.random.choice(
                        len(merged_pcd),
                        size=(512,),
                        replace=False)
                    merged_pcd = merged_pcd[subsample_indices]
                    pcd = torch.as_tensor(
                        merged_pcd, dtype=torch.float,
                        device=self.device) + self.franka_offset[..., None, :]

                if fetch_obj_pose:
                    rel_goal = self.get_current_rel_goal(obj_pose)
                else:
                    rel_goal = self.get_current_rel_goal()
                if fetch_obj_pose:
                    delta_angle = tx.rotation.axis_angle.from_quaternion(
                        rel_goal[..., 3:7])[..., -1]
                    true_angle = np.abs(
                        (delta_angle + np.pi) % (2 * np.pi) - np.pi
                    )
                    if (np.linalg.norm(rel_goal[..., :3]) < 0.05 and
                            np.abs(true_angle) < np.deg2rad(10)):
                        print('success!')
                        success = True
                        break

                # == Format observations & send to device ==
                self.observation['partial_cloud'][..., :] = torch.as_tensor(
                    pcd[None], dtype=torch.float, device=self.device)

                # NOTE: (q0, v0, q1, v1... formatting, from isaac gym)
                # self.observation['robot_state'][:,:7] = q_current.to(self.device)
                # self.observation['robot_state'][:,7:] = qd_current.to(self.device)
                self.observation['robot_state'][
                    :, :: 2] = q_current.to(self.device)
                # FIXME: hack to ensure J6 input=0
                # self.observation['robot_state'][..., 12] = 0
                self.observation['robot_state'][
                    :, 1:: 2] = qd_current.to(self.device)
                # HACK for some policy
                # self.observation['robot_state'][:, 9] = 0.
                # self.observation['hand_state'][:, :3] = current_EE_pos.to(
                #     self.device) + self.franka_offset[..., :]
                # self.observation['hand_state'][
                #     :, 3:7] = current_EE_ori.to(
                #     self.device)
                hand_state_7 = torch.cat([
                    current_EE_pos.to(
                        self.device) + self.franka_offset[..., :],
                    current_EE_ori.to(
                        self.device)
                ], dim=-1)[None]

                rel_goal_7 = torch.as_tensor(
                    rel_goal, dtype=torch.float, device=self.device)[None]

                self.observation['previous_action'][
                    :, :] = self.prev_action.to(
                    self.device)

                # print(self.observation['goal'].shape,
                #       to_rot6d_pose(rel_goal_7).shape,
                #       self.observation['hand_state'].shape,
                #       convert_hand(self.observation['hand_state']).shape
                #       )

                if self.update_6d_goal:
                    self.observation['goal'][...] = to_rot6d_pose(rel_goal_7)
                else:
                    self.observation['goal'][...] = rel_goal_7

                if self.update_6d_hand:
                    self.observation['hand_state'][...] = convert_hand(
                        hand_state_7)
                else:
                    self.observation['hand_state'][...] = hand_state_7

                # Get action from policy.
                with torch.inference_mode():
                    student_aux = {}
                    norm_obs = self.normalizer.normalize_obs(self.observation)
                    if self.update_hand_post_norm:  # HACK
                        norm_obs['hand_state'] = convert_hand(hand_state_7)
                    if HACK_CLIP_STATE:
                        norm_obs['robot_state'][
                            :, 1:: 2].clip_(-1.0, +1.0)

                    if is_first:
                    #or (cur_step % 8 == 0):
                    # if is_first:
                        print('==reset==')
                        _ = self.rma.reset(norm_obs)
                        self.rma.reset_state(self.done.squeeze(dim=-1))
                        state = self.rma(norm_obs, 0,
                                         self.done.squeeze(dim=-1),
                                         aux=student_aux)
                        if self.state is not None:
                            init_hidden = self.state.init_hidden
                            hidden = init_hidden(
                                batch_shape=1,
                                dtype=torch.float,
                                device=self.device)
                            _, hidden = self.state(hidden, None, {'student_state':state})
                            state = hidden[STATE_KEY]
                        # print('state', state.shape)

                        if self.dagger:
                            action, log_std = state.unbind(dim=-2)
                        else:
                            action, log_std = self.actor(state)
                        # action += torch.exp(log_std) * torch.randn_like(action)
                        is_first = False
                    else:
                        # self.rma.need_goal.fill_(True)
                        state = self.rma(
                            norm_obs, 0, self.done.squeeze(
                                dim=-1),
                            aux=student_aux)

                        if self.state is not None:
                            _, hidden = self.state(hidden, None, {'student_state':state})
                            state = hidden[STATE_KEY]

                        if self.dagger:
                            action, log_std = state.unbind(dim=-2)
                        else:
                            action, log_std = self.actor(state)
                        # action += torch.exp(log_std) * torch.randn_like(action)

                action = action.clone()
                action.clip_(-1.0, +1.0)

                lower = torch.as_tensor(
                    [-0.02] * 3 + [-0.03] * 3
                    + [10] * 7
                    + [0.3] * 7,
                    device=action.device)
                upper = torch.as_tensor(
                    [0.02] * 3 + [0.03] * 3
                    + [200] * 4 + [200] * 3
                    + [2.0] * 7,
                    device=action.device)

                unnormalized_action = self.unscale_transform(
                    action[0],
                    lower=lower,
                    upper=upper).cpu()
                self.prev_action[:] = unnormalized_action

                if False:
                    hand_txn = current_EE_pos[..., :3]
                    txn = hand_txn + unnormalized_action[..., :3]
                    # stage 1
                    txn.clamp_(
                        torch.as_tensor(obj_pose[..., :3, 3] - 0.24),
                        torch.as_tensor(obj_pose[..., :3, 3] + 0.24)
                    )
                    # stage 2
                    txn -= hand_txn
                    txn.clamp_(-0.02, +0.02)
                    unnormalized_action[..., :3] = txn

                if EXPORT:
                    log_path = path.log
                    with open(f'{log_path}/{cnt}.pkl', 'wb') as fp:
                        export = {}
                        for k, v in self.observation.items():
                            export[k] = v.detach().cpu().numpy()
                        export['action'] = unnormalized_action.numpy()[None]
                        export['goal'] = rel_goal_7
                        export['hand_state'] = hand_state_7
                        if color is not None:
                            if False:
                                cv2.namedWindow('color', cv2.WINDOW_GUI_NORMAL)
                                cv2.imshow('color', color)
                                cv2.waitKey(1)
                            export['color'] = color
                            # NOTE: Depth is not exported to save sotrage
                            # export['depth'] = depth
                        if (student_aux is not None and
                                'pose' in student_aux):
                            export['pred_goal'] = self.normalizer.unnormalize_obs(
                                {'goal': student_aux['pose']})['goal'].detach().cpu().numpy()
                        pickle.dump(export, fp)
                        cnt += 1

                self.kp = unnormalized_action[6:13]
                self.kd = unnormalized_action[13:] * torch.sqrt(self.kp)
                to_send = {"kp": self.kp.cpu(), "kd": self.kd.cpu()}
                if self.control_mode == 'dof_target':
                    # == Compute control target from action. ==
                    jacobian = self.robot.robot_model.compute_jacobian(
                        q_current, EE_FRAME)
                    lambda_val = 0.1
                    # lambda_val=0.07
                    jacobian_T = jacobian.T
                    lambda_matrix = (lambda_val ** 2) * torch.eye(n=6)
                    delta_dof_pos = jacobian_T @ torch.linalg.solve(
                        jacobian @ jacobian_T + lambda_matrix,
                        unnormalized_action[:6].unsqueeze(-1))
                    delta_dof_pos = delta_dof_pos.squeeze(-1)

                    # == Format and send data. ==
                    target_dof_pose = q_current + delta_dof_pos
                    to_send['target_dof_pose'] = target_dof_pose
                elif self.control_mode == 'ee_abs':
                    to_send['updated_abs'] = torch.ones(1)
                    ee_target = torch.empty(7)
                    ee_target[:3] = current_EE_pos + unnormalized_action[:3]
                    quat = quat_from_axa(unnormalized_action[3:6])
                    ee_target[3:] = quat_multiply(
                        quat, current_EE_ori)
                    to_send['target_ee'] = ee_target
                elif self.control_mode == 'ee_rel':
                    to_send['res_ee'] = unnormalized_action[:6]
                    to_send['updated_res'] = torch.ones(1)
                else:
                    raise ValueError(
                        f'not available model for {self.control_mode}')
                self.robot.update_current_policy(to_send)

            time.sleep(1 / 300)
            cur_step += 1

        # return to home after success
        q_home = np.asarray(
            [-0.0122, -0.1095, 0.0562, -2.5737, -0.0196, 2.4479, 0.0])
        self.robot.move_to_joint_positions(q_home)

        return success


    def unscale_transform(self,
                          x: torch.Tensor,
                          lower: torch.Tensor,
                          upper: torch.Tensor) -> torch.Tensor:
        # default value of center
        offset = (lower + upper) * 0.5
        # return normalized tensor
        return x * (upper - lower) * 0.5 + offset


def main():
    parser = argparse.ArgumentParser(description='Real robot setup')
    parser.add_argument(
        '--hacky-hand-update', '-hack', action='store_true',
        dest='update_hand_post_norm',
        help='Trun on hack for student trained with hand without normalization')
    parser.add_argument(
        '--ckpt-path',
        '-ckpt',
        type=str,
        required=True,
        dest='ckpt_path',
        help='Directory containing yaml and ckpt files for policy and studnet')
    parser.add_argument(
        '--dagger',
        '-dagger',
        action='store_true',
        dest='dagger',
        help='Use DAgger-style agent')
    parser.add_argument('-thin', 
                        action='store_true',
                        dest='thin_mode',
                    help='Turn on thin mode perception mode')
    parser.add_argument('-obj',
                        type=str, required=True,
                        dest='obj_name',
                        help='Name of the object to select')
    
    parser.add_argument('-april',
                    type=int,
                    required=False,
                    dest='april',
                    help='Use april',
                    default=1)
    parser.add_argument('-icp_reinit',
                type=int,
                required=False,
                dest='icp_reinit',
                help='Use icp_reinit',
                default=1)
    
    rt_cfg = RuntimeConfig()
    args = parser.parse_args()

    update_hand_post_norm: bool = args.update_hand_post_norm
    ckpt_path = args.ckpt_path

    # cfgs = load_configs('/home/user/corn_runtime//2023-08-28')
    cfgs = load_configs(ckpt_path)

    path = RunPath(RunPath.Config(root=F'{rt_cfg.root}/run'))
    # print(path.log)
    # return

    suc = False
    try:
        policy = Policy(rt_cfg, cfgs,
                        update_hand_post_norm = update_hand_post_norm,
                        dagger=args.dagger,
                        obj_name=args.obj_name,
                        thin=args.thin_mode,
                        april=args.april,
                        icp_reinit=args.icp_reinit
                        )

        if True:
            # from goal gui
            task_file = rt_cfg.task_file
            with open(task_file, 'rb') as fp:
                task = pickle.load(fp)
            if task['rotation'].shape == (3, 3):
                rotation = task['rotation'].astype(np.float32)
            else:
                rotation = tx.rotation.matrix.from_euler(
                    task['rotation']).astype(
                    dtype=np.float32)
            translation = task['translation']
            print(rotation)
            print(translation)

        else:
            # Hard-coded
            # rotation = tx.rotation.matrix.from_euler(
            #     [1.54434716, -0.69494293, 0.07300239]).astype(dtype=np.float32)
            # translation = np.asarray([0.13, 0.165, 0.045],
            #                          dtype=np.float32)
            rotation = tx.rotation.matrix.from_euler(
                [-np.pi / 2, 0, 0]).astype(dtype=np.float32)
            translation = np.asarray([0, 0, 0],
                                     dtype=np.float32)
        policy.setup(rotation, translation, ckpt_path)
        suc = policy.run_policy(path=path)
    finally:
        with open(F'{path.log}/task.pkl', 'wb') as fp:
            pickle.dump(task, fp)

        run_desc = 'auto' #input('run desc?')
        with open(F'{path.log}/description.txt', 'w') as fp:
            fp.write(run_desc)
            fp.write(F' success = {suc}')
        if False:
            state_log = policy.robot.terminate_current_policy()
            # print(state_log)
            torque = []
            joint = []
            for log in state_log:
                torque.append(np.array(log.prev_joint_torques_computed))
                joint.append(np.array(log.joint_positions))
            np.save('/home/user/Documents/torques.npy', np.stack(torque, 0))
            np.save('/home/user/Documents/joints.npy', np.stack(joint, 0))
        # kill perception process
        policy.perception_handle.terminate()
        policy.perception_handle.join()

        # kill shared manager
        smm, smems = policy.share_manager
        smm.shutdown()

        subprocess.run(['pkill', '-f', 'from multiprocessing'])
        subprocess.run(['pkill', '-f', 'python3 controller.py'])



if __name__ == "__main__":
    main()
