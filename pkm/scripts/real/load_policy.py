#!/usr/bin/env python3

# from isaacgym import gymtorch

# from pkm.models.cloud.point_mae import PointMAE
from dataclasses import dataclass, replace, asdict
from omegaconf import OmegaConf
from typing import Optional, Dict, Mapping, Any, Tuple
import hydra
from hydra.types import ConvertMode
import torch
import pickle
from pathlib import Path

from pkm.models.rl.nets import PiNet
from pkm.models.rl.net.icp import ICPNet
from pkm.models.rl.net.base import FeatureBase
from pkm.models.cloud.point_mae import PointMAEEncoder
from pkm.models.rl.env_normalizer import EnvNormalizer
from pkm.models.rl.generic_state_encoder import MLPStateEncoder

import sys
sys.path.append('/home/user/workspace/corn/pkm/scripts/toy/push_ppo/')
from distill import StudentAgentRMA

from pkm.train.ckpt import last_ckpt
from pkm.util.config import ConfigBase, recursive_replace_map
from pkm.util.hydra_cli import hydra_cli

from icecream import ic

ROOT = '/home/user/workspace/data'


def load_configs(root: Optional[str] = None,
                is_student = True):
    root = ROOT if root is None else root

    normalizer_cfg_base = OmegaConf.structured(EnvNormalizer.Config())
    normalizer_cfg = OmegaConf.load(F'{root}/normalizer.yaml')
    normalizer_cfg = OmegaConf.to_object(OmegaConf.structured(
        OmegaConf.merge(normalizer_cfg_base,
                        normalizer_cfg)
    ))

    policy_cfg_base = OmegaConf.structured(PiNet.Config())

    policy_cfg = None    
    if Path(F'{root}/policy.yaml').exists():
        policy_cfg = OmegaConf.load(F'{root}/policy.yaml')
        policy_cfg = OmegaConf.to_object(OmegaConf.structured(OmegaConf.merge(
            policy_cfg_base,
            policy_cfg)))
    
    cfgs={
        'normalizer': normalizer_cfg,
        'policy': policy_cfg
        }

    if is_student:
        student_cfg_base = OmegaConf.structured(
            StudentAgentRMA.StudentAgentRMAConfig())
        student_cfg = OmegaConf.load(F'{root}/student.yaml')
        student_cfg = OmegaConf.to_object(
            OmegaConf.structured(
                OmegaConf.merge(student_cfg_base, student_cfg)
            ))
        cfgs['student'] = student_cfg
    else:
        state_cfg_base = OmegaConf.structured(
            MLPStateEncoder.Config())
        ic(state_cfg_base)
        # state_cfg = OmegaConf.load(F'{root}/state.yaml')
        # ic(state_cfg)
        # state_cfg = OmegaConf.to_object(
        #     OmegaConf.structured(
        #         OmegaConf.merge(state_cfg_base, state_cfg)
        #     ))
        # state_cfg = OmgaConf.to_object(
        with open(F'{root}/state.pkl', 'rb') as fp:
            state_cfg = pickle.load(fp)
        ic(state_cfg)
        cfgs['state'] = state_cfg
        
        if Path(F'{root}/icp.yaml').is_file():
            icp_cfg_base = OmegaConf.structured(
            ICPNet.Config())
            icp_cfg = OmegaConf.load(F'{root}/icp.yaml')
            icp_cfg = OmegaConf.to_object(
            OmegaConf.structured(
                OmegaConf.merge(icp_cfg_base, icp_cfg)
            ))
        cfgs['icp'] = icp_cfg
    return cfgs

def load_objects(cfgs: Mapping[str, ConfigBase]):
    policy = PiNet(cfgs['policy'])
    # obs_shape = {'cloud': [3], 'goal': [7], 'partial_cloud': [3]}
    obs_shape = dict(cfgs['student'].shapes)
    obs_shape['partial_cloud'] = (3,)
    obs_shape['goal'] = (9,)
    normalizer = EnvNormalizer(cfgs['normalizer'], 1, obs_shape)
    student = StudentAgentRMA(cfgs['student'], None, "cuda")
    return {
        'student': student,
        'normalizer': normalizer,
        'policy': policy
    }


def load_checkpoints():
    teacher_ckpt = torch.load(last_ckpt(
        # 'corn/corn-/arm:jh-arm_div_cloud_v8_dgn-cloud_arm_pretrained_v5-lf-centering-dgn-with-scaling-reduced-fric-phase2-high-kd-000937'
        # 'corn/corn-/arm:jh-arm_div_cloud_v8_dgn-cloud_arm_pretrained_v5-lf-centering-dgn-pretrained-phase2-000937'
        # 'corn/corn-/arm:jh-arm_div_cloud_v8_dgn-cloud_arm_pretrained_v5-lf-centering-dgn-with-fixed-scaling-reduced-fric-new-lambda-phase2-000937'
        # 'corn/corn-/arm:jh-arm_div_cloud_v8_dgn-cloud_arm_pretrained_v8-icp-wrapper-new-set-phase2-000937'
        'corn/corn-/arm:rroom-arm_div_cloud_v8_new_dgn_6d-cloud_arm_pretrained_v10-icp-wrapper-new-set-6d-goal-phase2-000937'
    ), map_location='cpu')
        # 'corn/corn-/arm:gcp-arm_div_cloud_v5_dgn_phase2-cloud_arm_pretrained_v1-phase2-state-debug-wall-no-whole-longer-epi-high-th-000937'))
    student_ckpt = torch.load(last_ckpt(
        # 'corn/corn-/arm:xai-arm_div_cloud_v8_dgn_phase2_rma_mc-cloud_arm_pretrained_v5-partial-gru-8-phase2-multi-cam-reduced-fric-phase2-high-kd-000937'
        # 'corn/corn-/arm:jh-arm_div_cloud_v8_dgn_phase2_rma_mc-cloud_arm_pretrained_v5-partial-gru-8-phase2-multi-cam-high-goal-noise-000937'
        # 'corn/corn-/arm:jh-arm_div_cloud_v8_dgn_phase2_rma_mc-cloud_arm_pretrained_v5-partial-gru-8-phase2-multi-cam-new-lambda-000937'
        # 'corn/corn-/arm:jh-arm_div_cloud_v8_dgn_phase2_rma_mc-cloud_arm_pretrained_v8-partial-gru-8-phase2-new-set-mc-000937'
        # 'corn/corn-/arm:rroom-arm_div_cloud_v8_new_dgn_phase2_rma_mc-cloud_arm_pretrained_v10-partial-gru-8-phase2-new-set-6d-000937'
        'corn/corn-/arm:rroom-arm_div_cloud_v8_new_dgn_phase2_rma_mc-cloud_arm_pretrained_v10-partial-gru-8-phase2-new-set-6d-more-env-000937'

    ), map_location='cpu')
    env_ckpt = torch.load(last_ckpt(
        # 'corn/corn-/arm:xai-arm_div_cloud_v8_dgn_phase2_rma_mc-cloud_arm_pretrained_v5-partial-gru-8-phase2-multi-cam-reduced-fric-phase2-high-kd-000937_stat'
        # 'corn/corn-/arm:jh-arm_div_cloud_v8_dgn_phase2_rma_mc-cloud_arm_pretrained_v5-partial-gru-8-phase2-multi-cam-new-lambda-000937_stat'
        # 'corn/corn-/arm:jh-arm_div_cloud_v8_dgn_phase2_rma_mc-cloud_arm_pretrained_v8-partial-gru-8-phase2-new-set-mc-000937_stat'
        # 'corn/corn-/arm:rroom-arm_div_cloud_v8_new_dgn_phase2_rma_mc-cloud_arm_pretrained_v10-partial-gru-8-phase2-new-set-6d-000937_stat'
        'corn/corn-/arm:rroom-arm_div_cloud_v8_new_dgn_phase2_rma_mc-cloud_arm_pretrained_v10-partial-gru-8-phase2-new-set-6d-more-env-000937_stat'
    ), map_location='cpu')

    # student_ckpt = torch.load(F'{ROOT}/dgn-pose/ckpt/last.ckpt')
    # env_ckpt = torch.load(F'{ROOT}/dgn-pose/stat/env-last.ckpt')
    # student_ckpt = torch.load('/tmp/pkm/ppo-a/run-016/ckpt/last.ckpt')
    # env_ckpt = torch.load('/tmp/pkm/ppo-a/run-016/stat/env-last.ckpt')
    # student_ckpt = torch.load(last_ckpt(
    #     'corn/corn-/arm:jh-arm_div_cloud_v8_cube_phase2-cloud_arm_pretrained_v5-partial-gru-8-phas2-teacher-cube-000937'
    # ))
    # env_ckpt = torch.load(last_ckpt(
    #     'corn/corn-/arm:jh-arm_div_cloud_v8_cube_phase2-cloud_arm_pretrained_v5-partial-gru-8-phas2-teacher-cube-000937_stat'
    # ))
    # env_ckpt = torch.load(F'{ROOT}/new-student/stat/env-last.ckpt')

    
    policy = {
        k.split('actor_net.')[-1]: v for (k, v) in teacher_ckpt['self'].items()
        if 'actor_net' in k}
    student = student_ckpt['self']
    normalizer = env_ckpt['normalizer']
    return {
        'student': student,
        'normalizer': normalizer,
        'policy': policy
    }

def load_checkpoints_from_dir(root: Optional[str] = None,
                              is_student = True,
                              is_dagger:bool = False):
    root = ROOT if root is None else root

    policy = None
    if not is_dagger:
        policy = torch.load(F'{root}/policy.ckpt')['policy']

    normalizer = torch.load(F'{root}/normalize.ckpt')['normalizer']

    ckpts = {
            'normalizer': normalizer,
            'policy': policy
        }

    if is_student:
        student = torch.load(F'{root}/student.ckpt')['student']
        ckpts['student'] = student

        if Path(F'{root}/state.pkl').exists():
            with open(F'{root}/state.pkl', 'rb') as fp:
                state = pickle.load(fp)
            ckpts['state'] = state
    else:
        state = torch.load(F'{root}/state.ckpt')['state']
        ckpts['state'] = state
         
    return ckpts

def main():
    # root = '/tmp/rma-test'
    root = None
    cfgs = load_configs(root)
    objs = load_objects(cfgs)
    ckpts = load_checkpoints_from_dir(root)
    verbose: bool = True

    for k in objs.keys():
        src = ckpts[k].keys()
        dst = objs[k].state_dict().keys()
        if verbose:
            print(k)
            print('checkpoint has ... ')
            print(src)
            print('object wants ... ')
            print(dst)
            print('extra  is ...')
            print(set(dst).symmetric_difference(src))
        try:
            objs[k].load_state_dict(ckpts[k], strict=True)
        except:
            print(f"failed to load checkpoint of {k} in strict mode")
            print(objs[k].load_state_dict(ckpts[k], strict=False))

if __name__ == '__main__':
    main()
