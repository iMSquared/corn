#!/usr/bin/env python
# Author: Siddhartha Srinivasa

import numpy as np


class Plane(object):
    '''
    N-D plane class and helper functions
    '''
    def __init__(self, n, d):
        '''
        Hesse normal form of plane in arbitrary dimensions
        @param n - plane normal, (N,) vector
        @param d - distance from origin, scalar
        '''
        self.n = n / np.linalg.norm(n)
        self.d = d

    def __repr__(self):
        np.set_printoptions(precision=3, suppress=True)
        return 'Plane({0.n!r}, {0.d!r})'.format(self)

    def vectorize(self):
        '''
        Vectorizes the Plane class
        @return (N+1,) vector of n and d
        '''
        return np.hstack((self.n, self.d))

    def dim(self):
        '''
        Dimension of plane
        '''
        return len(self.n)

    def basis(self):
        '''
        Returns a basis set on the plane via QR decomposition
        @return Plane basis, (N-1,N) matrix
        '''
        A = np.vstack((self.n, np.eye(len(self.n)))).T
        Q, R = np.linalg.qr(A)
        return Q[:, 1:]

    def project(self, points):
        '''
        Projects a set of points onto the plane
        @param points - (X,N) matrix of points
        @return projected points, (X,N) matrix
        '''
        n = self.n[:, np.newaxis]
        A = np.eye(len(n)) - np.dot(n, n.T)
        return np.dot(A, points.T).T + np.squeeze(self.d * n)

    def distance(self, points):
        '''
        Computes the unsigned distance of points to the plane
        @param points - (X,N) matrix of points
        @return unsigned distance, (X,) vector
        '''
        n = self.n[:, np.newaxis]
        return np.squeeze(np.absolute(np.dot(points, n) - self.d))

    def point_probability(self, points, cov):
        '''
        Computes the probability that the point could be drawn from the
        [point, infinity] interval [Error function]
        based on radial noise covariance
        @param points - (X,N) matrix of points
        @param cov - radial covariance of each point, (X,) vector
        @return (X,) vector of probabilities
        '''
        import scipy.special
        k = self.d / np.dot(points, self.n)
        dr = np.linalg.norm(points, axis=1) * np.absolute(1 - k)
        return 1 - scipy.special.erf(dr / (np.sqrt(2 * cov)))

    def diff(self, plane, reference_3d=None):
        '''
        Computes the solid angle (radians) and distance (m) difference
        to another plane.
        Note that the solid angle has ambiguous sign in 3D without a reference
        vector.
        @param - plane, Plane object
        @param - reference_3d (3,) vector to disambiguate sign,
                 defaults to positive. Only works in 3D
        '''
        dn = np.arccos(np.dot(self.n.T, plane.n))
        numd = self.dim()
        dcross = np.cross(self.n, plane.n)
        if numd == 2:
            dn = np.sign(dcross) * dn
        if (numd == 3) and (reference_3d is not None):
            dn = np.sign(np.dot(self.n.T, dcross)) * dn
        dd = self.d - plane.d
        return np.asarray([dn, dd])

    def sample(self, M):
        '''
        Samples M points from the plane uniformly in unit interval
        @param M - number of points
        @return sampled points, (M,N) matrix
        '''
        basis = self.basis()
        samples = np.dot(basis, np.random.uniform(size=(basis.shape[1], M)))
        samples = samples + self.d * self.n[:, np.newaxis]
        return samples.T

    def box(self, scale=1.0, center=None):
        '''
        Computes corners of a scale*unit box centered at center on the plane
        @param scale - scale of box
        @param center - center of box
        @return (2*(N-1),N) matrix of corners
        '''
        basis = self.basis()
        samples = np.hstack((basis, -basis))
        if center is None:
            samples = scale * samples + self.d * self.n[:, np.newaxis]
        else:
            samples = scale * samples + self.project(center)[:, np.newaxis]
        return samples.T

    def plot(self, center=None, scale=1.0, color='r', alpha=1.0, ax=None):
        '''
        2D or 3D render of plane box centered and scaled
        @param center - center of plane box, (2,) or (3,) vector
                        defaults to origin
        @param scale - scale of box, defaults to 1.0
        @param color - color of box, defaults to red
        @param alpha - alpha of box, defaults to 1.0
        @param ax - axis handle, defaults to creating a new one
        @return axis handle
        '''
        import matplotlib.pyplot as plt

        numd = self.dim()
        if not ((numd == 2) or (numd == 3)):
            raise TypeError('Plotting restricted to 2D or 3D')
        if numd == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from mpl_toolkits.mplot3d import Axes3D
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            if numd == 2:
                ax = fig.gca()
            if numd == 3:
                ax = fig.gca(projection='3d')
                ax.set_zlabel('Z')
            plt.xlabel('X')
            plt.ylabel('Y')
        if center is None:
            center = np.zeros(numd)
        box = self.box(scale, center)
        if numd == 2:
            ax.plot(box[:, 0], box[:, 1], color, alpha=alpha)
        if numd == 3:
            tri = Poly3DCollection([box])
            tri.set_color(color)
            tri.set_edgecolor('k')
            tri.set_alpha(alpha)
            ax.add_collection3d(tri)
        plt.draw()
        return ax
