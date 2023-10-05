#!/usr/bin/env python
# Author: Siddhartha Srinivasa

import numpy as np
from pkm.real.track.jin2017.plane import Plane

# BayesPlane helper class with the helper.


class BayesPlane(object):
    '''
    Mean and covariance of an N-D plane in Hesse normal form
    '''

    def __init__(self, mean, cov):
        '''
        @param mean - mean Plane, in Plane class
        @param cov  - covariance of plane, (n,n) matrix
        '''
        self.mean = mean
        self.cov = cov

    def __repr__(self):
        return 'BayesPlane({0.mean!r}, {0.cov!r})'.format(self)

    def sample(self, M):
        '''
        Samples M planes from the distribution
        @param M - number of samples
        @return list of Planes
        '''
        psample = np.random.multivariate_normal(
            self.mean.vectorize(), self.cov, M)
        return [Plane(ps[0:-1], ps[-1]) for ps in psample]

    def point_probability(self, points, cov, numSamples=100):
        '''
        Computes the probability of the points to lie on the uncertain plane
        based on radial noise covariance by marginalizing over planes
        via Monte Carlo sampling
        @param points - (X,N) matrix of points
        @param cov - radial covariance of each point, (X,) vector
        @param numSamples - number of Monte Carlo samples, defaults to 100
        @return (X,) vector of probabilities
        '''
        ps = self.sample(numSamples)
        return np.mean(
            np.asarray([p.point_probability(points, cov) for p in ps]),
            axis=0)

    def diff(self, sample):
        '''
        Returns the difference from the mean to the samples
        @param sample, list of Plane samples
        '''
        if self.mean.dim() == 3:
            # Define a single reference
            reference = np.cross(self.mean.n, sample[0].n)
        else:
            reference = None
        return np.asarray([self.mean.diff(s, reference) for s in sample])

    def plot(self, M, center=None, scale=1.0, color='r', ax=None):
        '''
        2D or 3D render of M sampled plane boxes centered and scaled
        @param M - number of samples
        @param center - center of plane box, (2,) or (3,) vector
                        defaults to origin
        @param scale - scale of box, defaults to 1.0
        @param color - color of box, defaults to red
        @param alpha - alpha of box, defaults to 1.0
        @param ax - axis handle, defaults to creating a new one
        @return axis handle
        '''

        import matplotlib.pyplot as plt
        myax = self.mean.plot(center=center,
                              scale=scale, color=color, alpha=1.0, ax=ax)
        psample = self.sample(M)
        for ps in psample:
            ps.plot(center=center,
                    scale=scale, color=color, alpha=0.2, ax=myax)
        plt.show()
        return myax


def compute_cov(pts, cov=None):
    '''
    A simple constant cov noise model for illustration
    '''
    return np.asarray([cov] * pts.shape[0])


def fit_plane_bayes(data, cov, nancheck:bool=False):
    '''
    Fits a plane and its covariance, based on a radial noise model
    @param data - points to fit plane to, (X,N) matrix
    @param cov - radial covariance of each point, (X,) vector
    @return BayesPlane class with mean and covariance
    '''
    if nancheck:
        nans = np.isnan(data[:, 0] * data[:, 1])
        data = data[~nans, :]
        cov = cov[~nans]
    if data.size == 0:
        raise ValueError('Data matrix is empty')
    w = 1 / cov

    # pc = np.sum(w * data) / np.sum(w)
    pc = np.average(data, weights = w, axis=0)
    # sum([_w * _d for _w, _d in zip(w, data)]) / sum(w)
    # pc = np.asarray(sum([_w * _d for _w, _d in zip(w, data)]) / sum(w))
    x = data - pc
    # wx = np.asarray([_w * _x for _w, _x in zip(w, x)])
    wx = w[..., None] * x
    M = np.dot(wx.T, x)
    n = np.linalg.svd(M)[0][:, -1]
    d = np.dot(n.T, pc)
    plane = Plane(n, d)
    H = np.eye(plane.dim() + 1)
    H[-1, -1] = -np.sum(w)
    H[0:-1, -1] = -H[-1, -1] * pc
    H[-1, 0:-1] = -H[-1, -1] * pc
    H[0:-1, 0:-1] = -M + H[-1, -1] * np.dot(pc, pc.T) + \
        np.dot(n.T, np.dot(M, n)) * np.eye(data.shape[1])
    return BayesPlane(plane, -np.linalg.inv(H))
