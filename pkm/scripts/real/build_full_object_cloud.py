#!/usr/bin/env python3
import isaacgym

import pickle
from pathlib import Path
import os
from pkm.util.torch_util import dcn
from utill import get_pose_and_coordinate
import numpy as np
import open3d as o3d
from cho_util.math import transform as tx

srcs = ['/tmp/drp1', '/tmp/drp2', '/tmp/drp3']
pkls = []
for src in srcs:
    src = sorted(list(Path(src).glob('*.pkl')),
    key=os.path.getmtime)
    pkls.append(src[-1])

pcds = []
for pkl in pkls:
    with open(str(pkl), 'rb') as fp:
        data = pickle.load(fp)
    pcd = dcn(data['pcd'])
    print(pcd.shape)
    pcds.append(pcd)

pcds[1][..., :3] = pcds[1][..., :3] @ tx.rotation.matrix.from_euler([np.pi,0,0])

# for i in range(len(pcds)):
aligned_pcds = [pcds[0]]
for j in range(1, len(pcds)):
    pcd0 = pcds[0]
    pcd1 = pcds[j]
    T = get_pose_and_coordinate(pcd1, pcd0,
                                postprocess = False,
                                mode='normal')
    T = T.cpu().numpy()
    pcd1[..., :3] = pcd1[..., :3] @ T[:3,:3].T + T[:3,3]
    aligned_pcds.append(pcd1)

def to_o3d(x):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x[..., 0:3])
    pcd.colors = o3d.utility.Vector3dVector(x[..., 3:6])
    return pcd

seq = [to_o3d(x) for x in aligned_pcds]

aligned_pcds = np.concatenate(
    [aligned_pcds[0], aligned_pcds[2]], axis=0)
print(aligned_pcds.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(aligned_pcds[..., 0:3])
pcd.colors = o3d.utility.Vector3dVector(aligned_pcds[..., 3:6])
o3d.visualization.draw(seq+[pcd])

o3d.io.write_point_cloud('/tmp/dripper.ply', pcd)