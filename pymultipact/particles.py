import copy

from matplotlib import cm
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy

q0 = 1.60217663e-19
m0 = 9.1093837e-31
mu0 = 4 * pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458


class Particles:
    def __init__(self, xrange, init_v, bounds, phi, cmap='jet', step=None):
        self.cmap = cmap
        M = len(phi)

        self.bounds = np.array(bounds)

        self.x = self.bounds[(self.bounds[:, 0] > xrange[0]) & (self.bounds[:, 0] < xrange[1])]
        if step:
            self.x = self._select_values_with_step(self.x, step)

        # check if any particle in range
        if len(self.x) == 0:
            print('No surface emission point in selected range.')
            exit()

        shape = self.x.shape
        self.len = len(self.x)

        # get normal pointing inwards of emission points
        self.pt_normals = np.ones(self.x.shape)
        # get neighbouring points
        res, idxs = self.distance(1)
        for nn, idx in enumerate(idxs):
            n12 = self.get_point_normal(idx)
            self.pt_normals[nn] = n12

        # repeat into multidimensional array
        self.x = np.array(self.x.tolist() * M)
        self.pt_normals = np.array(self.pt_normals.tolist() * M)

        # convert velocity from eV to m/s
        self.init_v = np.sqrt(2 * init_v * q0 / m0)
        self.u = self.pt_normals * self.init_v

        self.phi = np.atleast_2d(np.repeat(phi, self.len)).T

        cmap = self.get_cmap(len(self.x), self.cmap)
        self.colors = np.array([cmap(i) for i in range(len(self.x))])

        #         self.len = self.x[M,S,:]

        self.x_old, self.u_old, self.phi_old = np.zeros(self.x.shape), np.zeros(self.u.shape), np.zeros(
            self.phi.shape)  # <- hold solution from prev time step
        self.x_temp, self.u_temp, self.phi_temp = np.zeros(self.x.shape), np.zeros(self.u.shape), np.zeros(
            self.phi.shape)  # <- hold tentative change in position

        # record initial particles state
        self.x_init = copy.deepcopy(self.x)
        self.u_init = copy.deepcopy(self.u)
        self.phi_init = copy.deepcopy(self.phi)

        # particles path
        self.paths = copy.deepcopy(np.hstack((self.x, self.phi)))
        self.paths_count = 1

        # hit energy
        self.E = [[] for ii in range(len(self.x))]
        self.n_secondaries = [[] for ii in range(len(self.x))]
        self.df_n = [[] for ii in range(len(self.x))]

        self.record = [self.x]
        self.lost_particles = []
        self.nhit = np.zeros(len(self.x))

        self.bright_set = []
        self.shadow_set = []

    def save_old(self):
        self.x_old = copy.deepcopy(self.x)
        self.u_old = copy.deepcopy(self.u)
        self.phi_old = copy.deepcopy(self.phi)

        self.x_temp = copy.deepcopy(self.x)
        self.u_temp = copy.deepcopy(self.u)
        self.phi_temp = copy.deepcopy(self.phi)

    def distance(self, n):
        norms = np.linalg.norm(self.x[:, None] - self.bounds, axis=-1)
        closest_dist_to_surface = norms.min(axis=1)
        indx = norms.argsort(axis=1).T[0:n, :]

        return np.atleast_2d(closest_dist_to_surface).T, indx.T.tolist()

    def remove(self, ind, bright='no'):
        if bright != 'yes':
            # add to shadow set before removal from main set
            self.shadow_set.append(self.paths[[ii * len(self.x) + np.array(ind) for ii in range(self.paths_count)]])

        self.paths = np.delete(self.paths, [ii * len(self.x) + np.array(ind) for ii in range(self.paths_count)], axis=0)

        self.x = np.delete(self.x, ind, axis=0)
        self.u = np.delete(self.u, ind, axis=0)

        self.x_old = np.delete(self.x_old, ind, axis=0)
        self.u_old = np.delete(self.u_old, ind, axis=0)

        self.phi = np.delete(self.phi, ind, axis=0)

        self.x_temp = np.delete(self.x_temp, ind, axis=0)
        self.u_temp = np.delete(self.u_temp, ind, axis=0)

        self.x_init = np.delete(self.x_init, ind, axis=0)
        self.u_init = np.delete(self.u_init, ind, axis=0)
        self.phi_init = np.delete(self.phi_init, ind, axis=0)

        self.colors = np.delete(self.colors, ind, axis=0)

        self.len = len(self.x)

        # print number of hits of particle before deleting
        self.nhit = np.delete(self.nhit, ind, axis=0)

        if bright != 'yes':
            # delete hit energy of lost particle
            indicesList = sorted(ind, reverse=True)
            for indx in indicesList:
                if indx < len(self.E):
                    # removing element by index using pop() function
                    self.E.pop(indx)
                    self.n_secondaries.pop(indx)

    def colors(self):
        return self.colors

    def set_cmap(self, cmap):
        self.cmap = cmap
        cmap = self.get_cmap(len(self.x), cmap)
        self.colors = np.array([cmap(i) for i in range(len(self.x))])

    def get_point_normal(self, idx):
        # calculate normal as average of connecting edge normals
        # assumpution is that the surface points are ordered in increasing x
        x0, y0 = self.bounds[idx[0] - 1]
        x1, y1 = self.bounds[idx[0]]
        x2, y2 = self.bounds[idx[0] + 1]

        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1

        n1 = -np.array([-dy1, dx1])
        n2 = -np.array([-dy2, dx2])
        n12 = n1 + n2

        return n12 / np.linalg.norm(n12)

    def update_record(self):
        self.record.append(self.x)

    def update_hit_count(self, inds):
        removed_inds = []
        self.nhit[inds] = self.nhit[inds] + 1
        # sort inds to start deleting from largest index
        inds.sort(reverse=True)
        # check if nhit = 20 and remove from main set and add to bright set
        for ind in inds:
            if self.nhit[ind] == 20:
                self.bright_set.append(self.paths[[ii * len(self.x) + np.array(ind) for ii in range(self.paths_count)]])
                # remove the index from main set
                self.remove([ind], bright='yes')
                removed_inds.append(ind)
        return removed_inds

    @staticmethod
    def get_cmap(n, name='jet'):
        '''
        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.
        '''
        return cm.get_cmap(name, n)

    def trace(self, ax):
        for xx_old, xx in zip(self.x_old, self.x):
            ax.plot([xx[0], xx_old[0]], [xx[1], xx_old[1]], color='g', marker='o', ms=1, zorder=10000)

    @staticmethod
    def _select_values_with_step(values, step):
        selected_values = []
        last_value = values[0][0] - step
        for value in values:
            if value[0] >= last_value + step:
                selected_values.append(value)
                last_value = value[0]

        return np.array(selected_values)
