#!/usr/bin/env python3

from typing import Optional

import numpy as np
import copy

import torch as th
import torch.nn as nn
import torch.nn.functional as F
# from pkm.util.math_util import matrix
from pkm.util.torch_util import dcn
from cho_util.math import transform as tx
# from pkm.util.math_util import invert_transform

from icecream import ic


def average_rotation_matrix(Rs):
    """
    Compute the unweighted "average" rotation matrix
    based on singular-value decomposition.
    """
    R_sum = np.sum(Rs, axis=0)
    U, s, Vh = np.linalg.svd(R_sum,
                             full_matrices=True, compute_uv=True)
    R_avg = U @ Vh
    return R_avg


def H_from_x(x: th.Tensor) -> th.Tensor:
    """
    Unpack the homogeneous matrix
    from optimization-friendly
    dual-vector parameterization.
    """
    txn = x[..., :3]
    rxn1 = F.normalize(x[..., 3:6], dim=-1)
    rxn3 = F.normalize(th.cross(rxn1, x[..., 6:9]), dim=-1)
    rxn2 = th.cross(rxn3, rxn1)
    T = th.zeros(*x.shape[:-1], 4, 4,
                 dtype=x.dtype,
                 device=x.device)
    T[..., :3, :4] = th.stack([rxn1, rxn2, rxn3, txn], dim=-1)
    T[..., 3, 3] = 1
    return T


def x_from_H(H: th.Tensor):
    """
    Pack the homogeneous matrix
    into optimization-friendly
    dual-vector parameterization.
    """
    txn = H[..., :3, 3]
    rxn1 = H[..., :3, 0]
    rxn2 = H[..., :3, 1]
    x = th.cat([txn, rxn1, rxn2],
               dim=-1)
    return x


def normalize_two_cloud(reference_cloud, target_cloud):
    '''
        Performs normalization, given pair of the two cloud
        Args:
            reference_cloud: the reference cloud to be normalized
            (Point cloud object)
            target_cloud: target cloud to be normalized
            (Point cloud object)
        Return:
            norm_transform: 4 by 4 transformation matrix, that does
                normalization transform(shifting to the center)
            ref_cloud_norm: normalized reference cloud
            tar_cloud_norm: normalized target cloud
    '''
    dx = -reference_cloud.get_center()
    norm_transform = np.eye(4)
    norm_transform[:3, 3] = dx
    return (norm_transform,
            copy.deepcopy(reference_cloud).translate(dx),
            copy.deepcopy(target_cloud).translate(dx))


def optimize_transforms(
        Ts_er: np.ndarray,
        Ts_co: np.ndarray,
        T_ec: Optional[np.ndarray] = None,
        T_ro: Optional[np.ndarray] = None,

        num_iter: int = 4096,
        sigma0: float = 0.1,
        decay_step: int = 128,
        gamma: float = 0.5,
        log_step: int = 128,
        fix_ec: bool = False
):
    """
    Solve a system of form:
        T_er @ T_ro? == T_ec? @ T_co

    Args:
        Ts_er: [hand_from_robot_base] transforms, Nx4x4
        Ts_co: [hand_camera_from_tag] transforms, Nx4x4
        T_ec: Initial guess for [hand_from_hand_camera] transform, 4x4
        T_ro: Initial guess for [robot_base_from_tag] transform, 4x4

        num_iter: Number of optimization iteration.
        sigma0: Initial variance for additive noise on transform params.
        decay_step: How often to decay sigma.
        gamma: The factor by which to decay sigma.
        log_step: How often to print logs.
    """
    Ts_er = th.as_tensor(Ts_er, dtype=th.float64)
    Ts_co = th.as_tensor(Ts_co, dtype=th.float64)
    x_ec = nn.Parameter(x_from_H(th.as_tensor(np.asarray(T_ec),
                                              dtype=th.float64)),
                        requires_grad=(not fix_ec))

    if T_ec is None and T_ro is None:
        T_ec = np.eye(4)
        T_ro = np.eye(4)

    if T_ro is None:
        Ts_re = tx.invert(Ts_er)
        Ts_ro = np.einsum('...ij, jk, ...kl -> ...il',
                          Ts_re, T_ec, Ts_co)
        txn_ro = Ts_ro[..., :3, 3].mean(axis=0)
        rxn_ro = average_rotation_matrix(Ts_ro[..., :3, :3])
        T_ro = np.eye(4)
        T_ro[:3, :3] = rxn_ro
        T_ro[:3, 3] = txn_ro

    x_ro = nn.Parameter(x_from_H(th.as_tensor(np.asarray(T_ro),
                                              dtype=th.float64)),
                        requires_grad=True)

    optimizer = th.optim.Adam([x_ec, x_ro], lr=1e-1)

    sigma = sigma0
    for ii in range(num_iter):
        if ii % decay_step == 0:
            sigma *= gamma
        Ts_eo1 = H_from_x(x_ec)[None] @ Ts_co
        Ts_eo2 = Ts_er @ H_from_x(x_ro)[None]
        loss = F.mse_loss(Ts_eo1, Ts_eo2)
        # loss = F.huber_loss(Ts_eo1, Ts_eo2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with th.no_grad():
            if x_ro.requires_grad:
                x_ro += sigma * th.randn_like(x_ro)
            if x_ec.requires_grad:
                x_ec += sigma * th.randn_like(x_ec)
        if (log_step > 0) and (ii % log_step == 0):
            print(loss)

    T_ec = (dcn(H_from_x(x_ec)))
    T_ro = (dcn(H_from_x(x_ro)))
    if log_step > 0:
        print(np.array2string(T_ec, separator=','))
        print(np.array2string(T_ro, separator=','))
    return (T_ec, T_ro)


def test():
    x_er = th.randn(9)
    T_er = H_from_x(x_er)
    x_co = th.randn(9)
    T_co = H_from_x(x_co)
    x_ec = th.randn(9)
    T_ec = H_from_x(x_ec)
    T_ro = tx.invert(T_er) @ T_ec @ T_co
    x_ro = x_from_H(T_ro)

    xs_er = x_er[None] + 0.1 * th.randn(size=(128, 9))
    # xs_co = x_co[None] + 0.001 * th.randn(128, 9)
    Ts_er = H_from_x(xs_er)
    Ts_co = tx.invert(T_ec)[None] @ Ts_er @ T_ro[None]
    xs_co = x_from_H(Ts_co)
    xs_co += 0.01 * th.randn(size=(128, 9))
    Ts_co = H_from_x(xs_co)

    sol = (optimize_transforms(Ts_er, Ts_co,
                               # T_ec,
                               np.eye(4),
                               sigma0=0.1,
                               log_step=-1))

    ic(sol)

    ic(T_ec, T_ro)


def test_cycle_conversions():
    x = th.randn(9)
    H = H_from_x(x)
    x2 = x_from_H(H)
    H2 = H_from_x(x2)
    print(H, H2)


def main():
    test()


if __name__ == '__main__':
    main()
