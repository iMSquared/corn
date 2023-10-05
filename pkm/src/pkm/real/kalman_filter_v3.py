#!/usr/bin/env python3

from typing import Optional
import numpy as np
from scipy.linalg import block_diag

from cho_util.math import transform as tx

DIM_X: int = 13
DIM_Z: int = 7
NUM_SIGMA_POINT: int = 5


def normalize(x, eps: float = 1e-6):
    # print(x)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.where(n <= eps, 1, n)
    return x / denom
    # return np.where(n <= eps, x, x/n)


# vec2quat = tx.rotation.quaternion.from_axis_angle
quat_multiply = tx.rotation.quaternion.multiply
quat_normalize = normalize
quat_inverse = tx.rotation.quaternion.inverse
# quat2vec = tx.rotation.axis_angle.from_quaternion


def normalize_angle(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def vec2quat(x, out=None, eps: float = 1e-6):
    angle = np.linalg.norm(x, axis=-1)
    # angle = normalize_angle(angle)
    half_angle = 0.5 * angle

    denom = np.where(angle > 0, angle, 1)
    axis = x / denom[..., None]
    if out is None:
        out = np.zeros((*x.shape[:-1], 4),
                       dtype=x.dtype)

    np.concatenate([
        np.sin(half_angle)[..., None] * axis,
        np.cos(half_angle)[..., None]
    ], axis=-1, out=out)
    # out[angle <= eps] = np.asarray([[0, 0, 0, 1]])
    out[...] = np.where(angle[...,None] > eps,
                   out,
                   [0,0,0,1])
    return out



def quat2vec(x, out=None, eps: float = 1e-6):
    x = np.asarray(x)
    if out is None:
        out = np.empty(shape=np.shape(x)[:-1] + (3,))

    sin_half = np.linalg.norm(x[..., :3], axis=-1)
    denom = np.where(sin_half > 0, sin_half, 1)
    scale = np.where(
        sin_half > eps,
        (2.0 * np.arcsin(np.clip(sin_half, -1.0, 1.0))) / denom,
        0.0)

    out[...] = x[..., :3] * scale[..., None]
    return out


def quat_err(q0, q1):
    q_err = quat_normalize(
        quat_multiply(q0,
                      quat_inverse(q1)))
    return q_err


def quat_avg_new(q_set, qt,
             max_err: float = 1e-3,
             eps: float = 1e-6,
             ):
    n = q_set.shape[0]

    max_iter = 1000
    for t in range(max_iter):
        err_vec = np.zeros((n, 3))
        qti = quat_inverse(qt)

        # Calc error quaternion and transform to error vector>
        q_err = quat_normalize( quat_err(q_set, qti[None]) )
        v_err = quat2vec(q_err)
        v_nrm = np.linalg.norm(v_err, axis=-1, keepdims=True)

        # Restrict vector angles to (-pi, pi].  
        err_vec = np.where(
            (v_nrm > eps),
            v_err * (normalize_angle(v_nrm) / v_nrm),
            0.0)
        # print('err_vec', err_vec.shape)
        # measure derivation between estimate and real, then update estimate
        err = np.mean(err_vec, axis=0)
        qt = quat_normalize(quat_multiply(vec2quat(err), qt))

        if np.linalg.norm(err) < max_err:
            break

    return qt, err_vec

def quat_avg(q_set, qt,
             max_err: float = 1e-3,
             eps: float = 1e-6,
             ):
    n = q_set.shape[0]

    epsilon = 1E-3
    max_iter = 1000
    for t in range(max_iter):
        err_vec = np.zeros((n, 3))
        for i in range(n):
            # calc error quaternion and transform to error vector
            qi_err = quat_normalize(quat_multiply(q_set[i, :], quat_inverse(qt)))
            vi_err = quat2vec(qi_err)

            # restrict vector angles to (-pi, pi]
            vi_norm = np.linalg.norm(vi_err)
            if vi_norm == 0:
                err_vec[i,:] = np.zeros(3)
            else:
                err_vec[i,:] = (-np.pi + np.mod(vi_norm + np.pi, 2 * np.pi)) / vi_norm * vi_err

        # measure derivation between estimate and real, then update estimate
        err = np.mean(err_vec, axis=0)
        qt = quat_normalize(quat_multiply(vec2quat(err), qt))

        if np.linalg.norm(err) < epsilon:
            break

    return qt, err_vec

def compute_sigma_pts(q, P, Q, n: int = None):
    if n is None:
        n = 2 * P.shape[0]
    print('n', n)

    # compute distribution around zero, apply noise before process model
    S = np.linalg.cholesky(P + Q)
    Xpos = S * np.sqrt(n)
    Xneg = -S * np.sqrt(n)
    W = np.hstack((Xpos, Xneg))
    print('W', W.shape)

    # shift mean and transform to quaternions
    X = np.zeros((2 * n, 4))
    for i in range(2 * n):
        qW = vec2quat(W[:, i])
        X[i, :] = quat_multiply(q, qW)

    # add mean, 2n+1 sigma points in total
    X = np.vstack((q, X))

    return X


def process_model(X, gyro, dt):
    n = X.shape[0]
    Y = np.zeros((n, 4))

    # compute delta quaternion
    qdelta = vec2quat(gyro * dt)

    for i in range(n):
        # project sigma points by process model
        q = X[i, :]
        Y[i, :] = quat_multiply(q, qdelta)

    return Y


def prediction(Y, qk):
    n = Y.shape[0]
    # compute mean (in quaternion)
    q_pred, W = quat_avg(Y, qk)

    # compute covariance (in vector)
    P_pred = np.zeros((3, 3))
    for i in range(n):
        P_pred += np.outer(W[i, :], W[i, :])
    P_pred /= n

    return q_pred, P_pred, W


def measurement_model(Y, z, W, R, x):
    n = Y.shape[0]

    Z = Y[..., :7]

    # measurement mean
    zk = np.mean(Z, axis=0)
    zk[3:7], q_errs = quat_avg(Z[..., 3:7], x[3:7])

    zk = np.mean(Z, axis=0)
    zk /= np.linalg.norm(zk)

    # measurement cov and correlation
    Pzz = np.zeros((7, 7))
    Pxz = np.zeros((13, 7))
    Z_err = Z - zk
    for i in range(n):
        Pzz += np.outer(Z_err[i, :], Z_err[i, :])
        Pxz += np.outer(W[i, :], Z_err[i, :])
    Pzz /= n
    Pxz /= n

    # innovation
    vk = z - zk
    Pvv = Pzz + R

    return vk, Pvv, Pxz


def update(q_pred, P_pred, vk, Pvv, K):
    # note: q_pred, P_pred are in quaternion, while vk, Pvv in vector
    q_gain = vec2quat(K.dot(vk))
    q_update = quat_multiply(q_gain, q_pred)
    P_update = P_pred - K.dot(Pvv).dot(K.T)
    return q_update, P_update


def fx(x, dt: float):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0
    p = x[0:3]
    q = x[3:7]
    v = x[7:10]
    w = x[10:13]

    if True:
        angle = np.linalg.norm(w)
        if angle <= 0:
            dq = np.asarray([0, 0, 0, 1], dtype=q.dtype)
        else:
            dq = tx.rotation.quaternion.from_axis_angle(w * dt)
    else:
        dq = tx.rotation.quaternion.from_axis_angle(w * 0.5 * dt)

        # reset NaNs to identities
        # which _do_ occasionally occur... I guess
        dq[np.isnan(dq).any()] = (0, 0, 0, 1)

    q1 = tx.rotation.quaternion.multiply(dq, q)
    print(np.linalg.norm(q1))
    p1 = p + v * dt
    out = np.concatenate([p1, q1, v, w])

    norm = np.linalg.norm(q1)
    if norm > 0:
        q1 /= norm

    # print(out)
    return out


class KalmanFilter6D:
    def __init__(self,
                 dtype=np.float32,
                 q_pos: float = 1e-3,
                 q_orn: float = 1e-3,
                 q_vel: float = 1e-3,
                 q_ang_vel: float = 1e-3,
                 r_pos: float = 1e-1,
                 r_orn: float = 1e-1,
                 dt: float = 1.0 / 30.0):
        self.dtype = dtype
        self.q = np.asarray(
            [q_pos] * 3 +
            [q_orn] * 3 +  # QUAT
            [q_vel] * 3 +
            [q_ang_vel] * 3).astype(dtype=dtype)
        self.r = np.asarray([r_pos] * 3
                            + [r_orn] * 3).astype(dtype=dtype)

    def reset(self, x0, P0: Optional[np.ndarray] = None):
        self.x = x0.astype(self.dtype)
        if P0 is None:
            P0 = np.eye(12)
        self.P = 1e-6 * P0

        # qk = np.array([1,0,0,0]) # last mean in quaternion
        # Pk = np.identity(3) * 0.1 # last cov in vector
        # Q = np.identity(3) * 2 # process noise cov
        # R = np.identity(3) * 2 # measurement noise cov

        self.Q = np.diag(self.q)
        self.R = np.diag(self.r)

    def process_model(self, X, dt):
        n = X.shape[0]
        Y = np.copy(X)
        # 0:3, 3:7, 7:10, 10:13
        w = X[..., 10:13]
        dq = vec2quat(w * dt)

        Y[..., 0:3] = X[..., 0:3] + X[..., 7:10] * dt
        # Y[..., 3:7] = quat_multiply(X[..., 3:7], dq)
        Y[..., 3:7] = quat_multiply(dq, X[..., 3:7])
        return Y

    def project_local(self, Y, prev_x):
        n = Y.shape[0]

        y = np.mean(Y, axis=0)
        y[3:7], Wq = quat_avg(Y[..., 3:7], prev_x[3:7])

        # W = Y - y[None]
        W = np.zeros((n, 12), dtype=self.dtype)
        W[..., 0:3] = Y[..., 0:3] - y[None, 0:3]
        W[..., 3:6] = Wq
        W[..., 6:12] = Y[..., 7:] - y[None, 7:]

        P = np.cov(W, rowvar=False, ddof=0)  # is this right?
        return y, P, W

    def measure_model(self, Y, z, W, R, x):
        n = Y.shape[0]

        Z = Y[..., :7]

        err_xyz = z[None, 0:3] - Z[..., 0:3]  # Z - z
        # err_vec = quat2vec(quat_err(Z[..., 3:7], z[None, 3:7])) # q0q1^{-1}
        z0 = np.broadcast_to(z[None, 3:7], Z[..., 3:7].shape)
        # err_vec = quat2vec(quat_err(z0, Z[..., 3:7])) # q0q1^{-1}
        err_vec = quat2vec(quat_err(z0, Z[..., 3:7]))
        err = np.concatenate([err_xyz, err_vec], axis=-1)
        # print(err)
        vk = err.mean(axis=0)
        cov = np.cov(W, -err, rowvar=False, ddof=0)
        # Pxx = cov[:12, :12]
        Pxz = cov[:12, 12:]
        Pzz = cov[12:, 12:]
        Pvv = Pzz + R

        return vk, Pvv, Pxz

    def update(self, x, P, vk, Pvv, K):
        gain = K.dot(vk)  # (12,12)@12->12

        q_gain = vec2quat(gain[3:6])
        x_post = np.concatenate([
            x[..., 0:3] + gain[0:3],
            quat_multiply(q_gain, x[..., 3:7]),
            x[..., 7:13] + gain[6:12]
        ], axis=-1)

        P_post = P - K.dot(Pvv).dot(K.T)
        return (x_post, P_post)

    def sigma_points(self, x, P, Q):
        n = 2 * P.shape[0]
        S = np.linalg.cholesky(P + Q)
        Xpos = S * np.sqrt(n)
        Xneg = -S * np.sqrt(n)
        WT = np.hstack((Xpos, Xneg))
        # print('W', W.shape) # 12 x 24
        W = WT.T  # 24 x 12

        X = np.zeros(shape=(n, 13),
                     dtype=x.dtype)
        X[..., 0:3] = x[None, 0:3] + W[..., 0:3]

        q0 = np.broadcast_to(x[None, 3:7], (n, 4))
        dq = vec2quat(W[..., 3:6])
        # X[..., 3:7] = quat_multiply(q0, dq)
        X[..., 3:7] = quat_multiply(dq, q0)

        X[..., 7:13] = x[None, 7:13] + W[..., 6:12]
        X = np.concatenate([X, x[None]], axis=0)
        # print('X', X)
        return X

        # X = np.zeros((2*n, 13))
        # vec2quat(W[3:6])
        # for i in range(2*n):
        #     qW = vec2quat(W[:, i])
        #     X[i, :] = quat_multiply(x[3:7], qW)

    # def convert_

    def __call__(self, z,
                 dt: Optional[float] = (1.0 / 30.0),
                 r_scale: Optional[float] = None):

        if r_scale is not None:
            R = np.diag(r_scale * self.r).astype(dtype=self.dtype)
            self.R = R

        if True:
            pts = self.sigma_points(self.x, self.P, self.Q)
        else:

            spts = self.sigma_points.sigma_points(
                np.r_[self.x[0:3], self.x[7:13]],
                block_diag(self.P[:3, :3], self.P[6:, 6:])
            )  # 19

            pts = np.zeros((spts.shape[0], 13),
                           dtype=spts.dtype)
            pts[..., 0:3] = spts[..., 0:3]
            pts[..., 7:13] = spts[..., 3:9]
            pts[..., 3:7] = compute_sigma_pts(self.x[3:7],
                                              self.P[3:6, 3:6],
                                              self.Q[3:6, 3:6],
                                              n=(spts.shape[0] // 2 - 1)
                                              )  # 7

        Y = self.process_model(pts, dt)
        y, P, W = self.project_local(Y, self.x)
        # print(y.shape)
        # self.x[3:7], self.P[3:7, 3:7], W = prediction(Y, self.x)
        vk, Pvv, Pxz = self.measure_model(Y, z, W, self.R, self.x)
        # Update
        K = np.dot(Pxz, np.linalg.inv(Pvv))  # Kalman gain
        self.x, self.P = self.update(y, P, vk, Pvv, K)

        qnorm = np.linalg.norm(self.x[3:7])
        # print(qnorm)
        if qnorm > 0:
            self.x[3:7] /= qnorm

        return self.x


def main():
    kf = KalmanFilter6D(dtype=np.float32)
    x = np.zeros(13, dtype=np.float32)
    x[..., 3:7] = (0, 0, 0, 1)
    kf.reset(x)
    z = np.zeros(7, dtype=np.float32)
    z[:3] = 1
    print(kf(z, dt=1.0 / 30.0))


if __name__ == '__main__':
    main()
