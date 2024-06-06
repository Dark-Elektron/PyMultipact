import copy
import itertools
import os.path
import time
import ngsolve as ng
from ngsolve.webgui import Draw
import netgen.occ as ngocc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from ipywidgets import IntSlider, interact, Layout
import multiprocessing as mp
import pickle
import pymultipact.geometry_writer as geometry_writer
from pymultipact.integrators import Integrators
from pymultipact.particles import Particles

q0 = 1.60217663e-19
m0 = 9.1093837e-31
mu0 = 4 * np.pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458


class Domain:
    def __init__(self, project, boundary_file=None, field=None):
        """

        Parameters
        ----------
        project: str
            Project directory
        boundary_file: str
            Boundary file path
        field: bytearray
            Field to be loaded
        """
        self.cn_c0 = None
        self.bounding_rect = None
        self.project_folder = project.folder

        self.Epk = None
        self.n_init_particles = None
        self.phi_v, self.epks_v = None, None
        self.particles_left = None
        self.particles_objects = None
        self.zmin, self.zmax, self.rmin, self.rmax = None, None, None, None
        self.eigen_freq = None
        self.K = None
        self.M = None
        self.precond = None
        self.boundary = None
        self.mesh = None
        self.domain = None

        if field is None:
            self.field = None
        else:
            self.field = field

        self.sey = None
        # set default sey
        self.set_sey(r'../sample_seys/sey')

        self.bc_zmin, self.bc_zmax, self.bc_rmin, self.bc_rmax = [0, 0, 0, 0]
        self.gfu_E = None
        self.gfu_H = None
        self.eigenvals, self.eigenvecs = None, None

        # define domain
        self.load_boundary('sample_domains/tesla_mid_cell.n')

    def load_boundary(self, geopath):
        if geopath is None:
            print("Please enter geometry path.")
            return
        try:
            # read geometry
            cav_geom = pd.read_csv(geopath, header=None, skiprows=3, skipfooter=1,
                                   sep='\s+', engine='python')[[1, 0]]
            self.boundary = np.array(list(cav_geom.itertuples(index=False, name=None)))
            self.mesh_domain()
        except Exception as e:
            print("Please enter valid geometry path.", e)

    def define_boundary(self, kind='cavity', name='geodata', **kwargs):
        # Implement entering dimensions using kwargs

        if self.project_folder is None:
            print("Something is wrong. Project folder is not defined.")
            return
        try:
            if kind == 'cavity':
                mid_cell, lend_cell, rend_cell, beampipe = None, None, None, None
                keys = kwargs.keys()
                if 'mid_cell' in keys:
                    mid_cell = kwargs['mid_cell']
                if 'lend_cell' in keys:
                    lend_cell = kwargs['lend_cell']
                if 'rend_cell' in keys:
                    rend_cell = kwargs['rend_cell']
                if 'beampipe' in keys:
                    beampipe = kwargs['beampipe']

                # write geometry
                geometry_writer.write_ell_cavity(self.project_folder, mid_cell, lend_cell, rend_cell, beampipe,
                                                 name=name)

                # read geometry
                cav_geom = pd.read_csv(f'{self.project_folder}/{name}.n', header=None,
                                       sep='\s+', engine='python')[[1, 0]]

                self.boundary = np.array(list(cav_geom.itertuples(index=False, name=None)))

            self.mesh_domain()
        except Exception as e:
            print("Please enter valid geometry path.", e)

    def show_initial_points(self, xrange, step=None):
        pts = self.boundary[(self.boundary[:, 0] > xrange[0]) & (self.boundary[:, 0] < xrange[1])]

        if step:
            pts = self._select_values_with_step(pts, step)

        fig, ax = plt.subplots()
        ax.plot(self.boundary[:, 0], self.boundary[:, 1])
        ax.scatter(pts[:, 0], pts[:, 1], fc='None', ec='k', s=50)
        plt.show()

    def define_elliptical_cavity(self, mid_cell=None, lend_cell=None, rend_cell=None, beampipe='None'):
        kwargs = {
            'mid_cell': mid_cell,
            'lend_cell': lend_cell,
            'rend_cell': rend_cell,
            'beampipe': None
        }
        self.define_boundary(kind='cavity', **kwargs)

    def set_boundary_conditions(self, zmin='PMC', zmax='PMC', rmin='PEC', rmax='PEC'):
        self.bc_zmin, self.bc_zmax, self.bc_rmin, self.bc_rmax = [zmin, zmax, rmin, rmax]

    def mesh_domain(self, maxh=0.00577):
        wp = ngocc.WorkPlane()
        wp.MoveTo(*self.boundary[0])
        for p in self.boundary[1:]:
            wp.LineTo(*p)
        wp.Close().Reverse()
        self.domain = wp.Face()

        # name the boundaries
        self.domain.edges.Max(ngocc.X).name = "zmax"
        self.domain.edges.Max(ngocc.X).col = (1, 0, 0)
        self.domain.edges.Min(ngocc.X).name = "zmin"
        self.domain.edges.Min(ngocc.X).col = (1, 0, 0)
        self.domain.edges.Min(ngocc.Y).name = "rmin"
        self.domain.edges.Min(ngocc.Y).col = (1, 0, 0)

        # get xmin, xmax, ymin
        self.zmin = self.domain.vertices.Min(ngocc.X).p[0]
        self.zmax = self.domain.vertices.Max(ngocc.X).p[0]
        self.rmin = self.domain.vertices.Min(ngocc.Y).p[1]
        self.rmax = self.domain.vertices.Max(ngocc.Y).p[1]

        self.bounding_rect = [self.zmin, self.zmax, self.rmin, self.rmax]

        geo = ngocc.OCCGeometry(self.domain, dim=2)

        # mesh
        ngmesh = geo.GenerateMesh(maxh=maxh)
        self.mesh = ng.Mesh(ngmesh)

        # save mesh
        with open(f"{self.project_folder}/mesh.pkl", "wb") as f:
            pickle.dump(self.mesh, f)

    def compute_fields(self):
        # define finite element space
        fes = ng.HCurl(self.mesh, order=1, dirichlet='default')
        u, v = fes.TnT()

        a = ng.BilinearForm(ng.y * ng.curl(u) * ng.curl(v) * ng.dx).Assemble()
        m = ng.BilinearForm(ng.y * u * v * ng.dx).Assemble()

        apre = ng.BilinearForm(ng.y * ng.curl(u) * ng.curl(v) * ng.dx + ng.y * u * v * ng.dx)
        pre = ng.Preconditioner(apre, "direct", inverse="sparsecholesky")

        with ng.TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()

            # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
            gradmat, fesh1 = fes.CreateGradient()
            gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
            math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
            math1[0, 0] += 1  # fix the 1-dim kernel
            invh1 = math1.Inverse(inverse="sparsecholesky", freedofs=fesh1.FreeDofs())
            # build the Poisson projector with operator Algebra:
            proj = ng.IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
            projpre = proj @ pre.mat

            self.K = a.mat
            self.M = m.mat
            self.precond = pre.mat
            self.eigenvals, self.eigenvecs = ng.solvers.PINVIT(a.mat, m.mat, pre=projpre, num=3, maxit=20,
                                                               printrates=False)

        # print out eigenvalues
        self.eigen_freq = []
        for i, lam in enumerate(self.eigenvals):
            self.eigen_freq.append(c0 * np.sqrt(lam) / (2 * np.pi) * 1e-6)
            print(i, lam, 'freq: ', c0 * np.sqrt(lam) / (2 * np.pi) * 1e-6, "MHz")

        # plot results
        self.gfu_E = []
        self.gfu_H = []
        for i in range(len(self.eigenvecs)):
            w = 2 * ng.pi * self.eigen_freq[i] * 1e6
            gfu = ng.GridFunction(fes)
            gfu.vec.data = self.eigenvecs[i]

            self.gfu_E.append(gfu)

            self.gfu_H.append(1j / (mu0 * w) * ng.curl(gfu))

    def draw(self):
        """
        Function to draw the deomain

        Returns
        -------

        """
        Draw(self.domain)

    def draw_mesh(self):
        Draw(self.mesh)

    def draw_fields(self, mode=1, which='E'):
        if which == 'E':
            Draw(ng.Norm(self.gfu_E[mode]), self.mesh, order=2)  # , vectors={"grid_size":300};#
        else:
            Draw(ng.Norm(self.gfu_H[mode]), self.mesh, order=2)  # , vectors={"grid_size":150}

    def add_particles(self, particles):
        pass

    def define_field(self, field):
        pass

    def track_particles(self, mode=1, integrator='rk4'):
        pass

    def analyse_multipacting_parallel(self, proc_count, mode=1, init_pos=None, epks=None, phis=None,
                                      v_init=2, init_points=None, integrator='rk4', parallel=False):

        # save mode fields
        with open(f"{self.project_folder}/gfu_EH.pkl", "wb") as f:
            pickle.dump([self.gfu_E[mode], self.gfu_H[mode]], f)

        processes = []
        self.fig, self.ax = plt.subplots()
        lmbda = c0 / (self.eigen_freq[mode] * 1e6)
        if self.sey is None:
            print("Secondary emission yield not defined, using default sey.")

        xpnts_surf = self.boundary[(self.boundary[:, 1] > 0) & (self.boundary[:, 0] > min(self.boundary[:, 0])) & (
                self.boundary[:, 0] < max(self.boundary[:, 0]))]
        self.ax.plot(xpnts_surf[:, 0], xpnts_surf[:, 1])
        Esurf = [ng.Norm(self.gfu_E[mode])(self.mesh(xi, yi)) for (xi, yi) in xpnts_surf]
        self.Epk = (max(Esurf))

        if epks is None:
            self.epks_v = 1 / self.Epk * 1e6 * np.linspace(30, 80, 4)
        else:
            self.epks_v = epks

        if phis is None:
            phi_v = np.linspace(0, 2 * np.pi, 20)  # <- initial phase
        else:
            phi_v = phis

        if init_pos is None:
            init_pos = [-0.006, 0.000]

        # get surface points
        pec_boundary = self.mesh.Boundaries("default")
        bel = [xx.vertices for xx in pec_boundary.Elements()]
        bel_unique = list(set(itertools.chain(*bel)))
        xpnts_surf = sorted([self.mesh.vertices[xy.nr].point for xy in bel_unique])
        xsurf = np.array(xpnts_surf)

        self.em = EMField(copy.deepcopy(self.gfu_E[mode]), copy.deepcopy(self.gfu_H[mode]))

        # define one internal point
        xin = [0, 0]
        no_of_remaining_particles = []

        # calculate time for 10 cycles, 20 alternations
        T = 1 / (self.eigen_freq[mode] * 1e6) * 10
        lmbda = c0 / (self.eigen_freq[mode] * 1e6)

        self.particles_left = []
        self.particles_nhits = []
        self.particles_objects = []
        start = time.time()

        epks_len = len(self.epks_v)
        share = round(epks_len / proc_count)

        for p in range(proc_count):
            # try:
            if p < proc_count - 1:
                proc_epks_list = self.epks_v[p * share:p * share + share]
            else:
                proc_epks_list = self.epks_v[p * share:]

            service = mp.Process(target=self._analyse_multipacting, args=(p, self.project_folder, self.eigen_freq,
                                                                          mode, init_pos, proc_epks_list, phi_v,
                                                                          v_init, self.sey, self.Epk,
                                                                          self.bounding_rect))
            service.start()
            processes.append(service)

        # Wait for all processes to complete
        for service in processes:
            service.join()

        # compile results
        self.particles_left = []
        self.particles_objects = []
        self.cn_c0 = []
        for p in range(proc_count):
            # Saving model to pickle file
            with open(f"{self.project_folder}/mresults_{p}", "rb") as file:
                m_result = pickle.load(file)

            self.cn_c0.extend(m_result['cn/c0'])
            self.particles_objects.extend(m_result['particles_objects'])

            if p == 0:
                self.n_init_particles = m_result['n_init_particles']

    @staticmethod
    def _analyse_multipacting(proc_id, folder, eigen_freq, mode, init_pos, procs_epks, phis,
                              v_init, sey, Epk, bounding_rect):

        # pickle mesh and fields
        with open(f'{folder}/mesh.pkl', 'rb') as f:
            mesh = pickle.load(f)
        with open(f'{folder}/gfu_EH.pkl', "rb") as f:
            gfu_E, gfu_H = pickle.load(f)

        # get surface points
        pec_boundary = mesh.Boundaries("default")
        bel = [xx.vertices for xx in pec_boundary.Elements()]
        bel_unique = list(set(itertools.chain(*bel)))
        xpnts_surf = sorted([mesh.vertices[xy.nr].point for xy in bel_unique])
        xsurf = np.array(xpnts_surf)

        # calculate time for 10 cycles, 20 alternations
        # T = 1 / (eigen_freq[mode] * 1e6) * 10
        # lmbda = c0 / (eigen_freq[mode] * 1e6)

        w = 2 * np.pi * eigen_freq[mode] * 1e6
        integrator = Integrators(mesh, w, bounding_rect=bounding_rect)

        particles_left = []
        particles_nhits = []
        particles_objects = []
        start = time.time()
        for epk in procs_epks:
            sub_start = time.time()
            t = 0
            dt = 1e-11
            counter = 0

            particles = Particles(init_pos, v_init, xsurf, phis, cmap='jet')

            em = EMField(gfu_E, gfu_H)
            n_init_particles = len(particles.x)
            print(f'{proc_id}: Initial number of particles: ', n_init_particles)

            # move particles with initial velocity. ensure all initial positions after first move lie inside the bounds
            particles.x = particles.x + particles.u * dt  # remove later

            record = {}
            scale = epk  # <- scale Epk to 1 MV/m and multiply by sweep value
            while t < 1000e-10:
                if particles.len != 0:
                    particles.save_old()
                    integrator.rk4(particles, t, dt, em, scale, sey)
                    particles.update_record()
                counter += 1
                t += dt

            # self.calculate_distance_function(particles, lmbda)
            particles_objects.append(particles)

            if len(particles.nhit) == 0:
                particles_nhits.append(0)
            else:
                particles_nhits.append(particles.nhit[0])

            particles_left.append(len(particles.bright_set))
            print(
                f"Epk: {epk * Epk * 1e-6} MV/m, particles in bright set: {len(particles.bright_set)}, time: {time.time() - sub_start}")

        print("Total runtime:: ", time.time() - start)
        cn_c0 = np.array(particles_left) / n_init_particles
        # results
        mresult = {'cn/c0': cn_c0,
                   'particles_objects': particles_objects,
                   'n_init_particles': n_init_particles,
                   'epks': procs_epks,
                   'phis_v': phis}

        # Saving model to pickle file
        with open(f"{folder}/mresults_{proc_id}", "wb") as file:
            pickle.dump(mresult, file)

        print(f"Proc {proc_id} done with multipacting analysis.")

    def analyse_multipacting(self, mode=1, xrange=None, epks=None, phis=None,
                             v_init=2, init_points=None, integrator='rk4', step=None):
        """

        Parameters
        ----------
        mode: int
            Eigenmode index
        xrange: list, ndarray
            range of x
        epks
        phis
        v_init
        init_points
        integrator
        step

        Returns
        -------

        """

        self.fig, self.ax = plt.subplots()
        lmbda = c0 / (self.eigen_freq[mode] * 1e6)
        if self.sey is None:
            print("Secondary emission yield not defined, using default sey.")

        xpnts_surf = self.boundary[(self.boundary[:, 1] > 0) & (self.boundary[:, 0] > min(self.boundary[:, 0])) & (
                self.boundary[:, 0] < max(self.boundary[:, 0]))]
        self.ax.plot(xpnts_surf[:, 0], xpnts_surf[:, 1])
        Esurf = [ng.Norm(self.gfu_E[mode])(self.mesh(xi, yi)) for (xi, yi) in xpnts_surf]
        self.Epk = (max(Esurf))

        if epks is None:
            self.epks_v = 1 / self.Epk * 1e6 * np.linspace(0, 80, 192)
        else:
            self.epks_v = 1 / self.Epk * epks

        if phis is None:
            phi_v = np.linspace(0, 2 * np.pi, 72)  # <- initial phase
        else:
            phi_v = phis

        if xrange is None:
            xrange = [-0.00025, -0.000]

        # get surface points
        pec_boundary = self.mesh.Boundaries("default")
        bel = [xx.vertices for xx in pec_boundary.Elements()]
        bel_unique = list(set(itertools.chain(*bel)))
        xpnts_surf = sorted([self.mesh.vertices[xy.nr].point for xy in bel_unique])
        xsurf = np.array(xpnts_surf)

        # define one internal point
        xin = [0, 0]

        w = 2 * np.pi * self.eigen_freq[mode] * 1e6
        integrators = Integrators(self.mesh, w, bounding_rect=self.bounding_rect)
        no_of_remaining_particles = []

        # calculate time for 10 cycles, 20 alternations
        T = 1 / (self.eigen_freq[mode] * 1e6) * 10
        lmbda = c0 / (self.eigen_freq[mode] * 1e6)

        self.particles_left = []
        self.particles_nhits = []
        self.particles_objects = []
        start = time.time()
        for epk in self.epks_v:
            sub_start = time.time()
            t = 0
            dt = 1e-11
            PLOT = False
            error = False
            counter = 0

            particles = Particles(xrange, v_init, xsurf, phi_v, cmap='jet', step=step)

            self.n_init_particles = len(particles.x)
            print('Initial number of particles: ', self.n_init_particles)
            em = EMField(copy.deepcopy(self.gfu_E[mode]), copy.deepcopy(self.gfu_H[mode]))

            # move particles with initial velocity. ensure all initial positions after first move lie inside the bounds
            particles.x = particles.x + particles.u * dt  # remove later

            record = {}
            scale = epk  # <- scale Epk to 1 MV/m and multiply by sweep value
            #     while t < T:
            while t < 1000e-10:
                if particles.len != 0:
                    particles.save_old()
                    integrators.rk4(particles, t, dt, em, scale, self.sey)
                    particles.update_record()
                counter += 1
                t += dt

            self.calculate_distance_function(particles, lmbda)
            self.particles_objects.append(particles)

            if len(particles.nhit) == 0:
                self.particles_nhits.append(0)
            else:
                self.particles_nhits.append(particles.nhit[0])

            self.particles_left.append(len(particles.bright_set))
            print(
                f"Epk: {epk * self.Epk * 1e-6} MV/m, particles in bright set: {len(particles.bright_set)}, time: {time.time() - sub_start}")

        print("Total runtime:: ", time.time() - start)

        self.cn_c0 = np.array(self.particles_left) / self.n_init_particles
        # results
        mresult = {'cn/c0': np.array(self.particles_left) / self.n_init_particles,
                   'particles_objects': self.particles_objects,
                   'n_init_particles': self.n_init_particles,
                   'Epk': self.Epk,
                   'epks': self.epks_v,
                   'phis_v': phi_v}

        # Saving model to pickle file
        with open(f"{self.project_folder}/mresults.pkl", "wb") as file:
            pickle.dump(mresult,
                        file)  # Dump function is used to write the object into the created file in byte format.

        print("Done with multipacting analysis.")
        plt.show()

    def set_sey(self, sey_filepath):
        self.sey = SEY(sey_filepath)

    def load_multipacting_result(self, filepath=None):
        if filepath is None:
            print("Please enter a filepath.")

        try:
            # Opening saved model
            with open("mresults.pkl", "rb") as file:
                mresult_loaded = pickle.load(file)

            self.particles_left = mresult_loaded['cn/c0']
            self.particles_objects = mresult_loaded['particles_objects']
            self.n_init_particles = mresult_loaded['n_init_particles']
            self.Epk = mresult_loaded['Epk']
            self.epks_v = mresult_loaded['epks']
            self.phi_v = mresult_loaded['phis_v']
        except FileNotFoundError as e:
            print("Please enter valid file path. ", e)

    def calculate_distance_function(self, particles, lmbda):
        kappa = lmbda / (2 * np.pi)
        # calculate distance function
        for path_i in range(len(particles.bright_set)):
            x_n, phi_n = particles.bright_set[path_i][:, 0:2], particles.bright_set[path_i][:, 2]
            df = np.sqrt(np.linalg.norm(x_n[-1] - x_n[0]) ** 2 + kappa * np.linalg.norm(
                np.exp(1j * phi_n[-1]) - np.exp(1j * phi_n[0])) ** 2)
            particles.df_n[path_i].append(df)

    def calculate_Ef(self):
        self.Ef = []
        for particles in self.particles_objects:
            #     print(particles.nhit)
            Ef_p = [particle_energy[-1] if len(particle_energy) != 0 else 0 for particle_energy in particles.E]
            #     print(np.sum(Ef_p)/len(Ef_p) if len(Ef_p) > 0 else 0)
            #     print(len(Ef_p), Ef_p)
            self.Ef.append(np.sum(Ef_p) / len(Ef_p) if len(Ef_p) > 0 else 0)

        return self.Ef

    def save_fields(self):
        pass

    def plot_cf(self):
        fig, ax = plt.subplots()
        ax.plot(self.epks_v * self.Epk * 1e-6, self.cn_c0)
        ax.set_ylim(bottom=0)
        plt.show()

    def plot_Ef(self):
        if len(self.Ef) == 0:
            self.calculate_Ef()
        fig, ax = plt.subplots()
        ax.plot(self.epks_v * self.Epk * 1e-6, self.Ef)
        ax.axhline(50, c='r')
        ax.set_ylim(0, 100)
        plt.show()

    def plot_ef(self):
        secondaries = [(sum([np.prod(nn) for nn in particles.n_secondaries])) for particles in self.particles_objects]
        if len(secondaries) > 0:
            fig, ax = plt.subplots()
            ax.plot(self.epks_v * self.Epk * 1e-6, 2 * (np.array(secondaries) + 1) / self.n_init_particles)
            ax.axhline(1, c='r')
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-3)
            plt.show()
        else:
            print('No secondaries to plot!')

    def get_sey(self):
        return self.sey

    def plot_sey(self):
        fig, ax = plt.subplots()
        ax.plot(self.sey.data['E'][:-1], self.sey.data['sey'][:-1])
        ax.axhline(1, 0, color='r')
        plt.show()

    def plot_trajectories(self):

        # create plot
        # fig, axs = plt.subplot_mosaic([[0, 1, 2]], figsize=(11, 4), layout='constrained')
        fig, axs = plt.subplot_mosaic([[0]], figsize=(6, 4), layout='constrained')
        # p1 = particles.paths.reshape(particles.paths_count, *particles.x.shape)
        # path_i = 0
        # line, = ax.plot(p1[:, path_i, :][:, 0], p1[:, path_i, :][:, 1])#, lw=0, marker='o', ms=2)

        path_i = 0
        Epk_indx = 1

        line_surf, = axs[0].plot(np.array(self.boundary)[:, 0] * 1e3, np.array(self.boundary)[:, 1] * 1e3, lw=3)
        line, = axs[0].plot([], [], c='k', label='PyMultipact')  # , lw=0, marker='o', ms=2)
        line_init, = axs[0].plot([], [], c='k', marker='o', zorder=10)  # plot initial point
        line_end, = axs[0].plot([], [], c='b', marker='o', zorder=10)  # plot initial point

        # line2, = axs[1].plot([], [])  #, lw=0, marker='o', ms=2)
        # line3, = axs[2].plot([], [])  #, lw=0, marker='o', ms=2)

        # Define the function to update the maximum value of slider w based on the value of slider epk_i
        def update_w_max(epk_i):
            if isinstance(epk_i, int):
                w_slider.max = len(self.particles_objects[epk_i].bright_set) - 1
            else:
                w_slider.max = len(self.particles_objects[epk_i.new].bright_set) - 1

        # Create slider widgets
        epk_i_slider = IntSlider(min=0, max=len(self.particles_objects), step=1, description='epk_i:',
                                 layout=Layout(width='50%'), value=83)
        print(len(self.particles_objects))
        # Observe changes in the value attribute of epk_i_slider and update w_slider accordingly
        epk_i_slider.observe(update_w_max, names='value')
        w_slider = IntSlider(min=-1, max=len(self.particles_objects[epk_i_slider.value].bright_set) - 1,
                             description='w:',
                             layout=Layout(width='50%'), value=28)

        axs[0].set_xlabel('z [mm]')
        axs[0].set_ylabel('r [mm]')

        # plot multipac results
        # plot_path(r"D:\Dropbox\multipacting\MPGUI21", loc='left', ax=axs[0], label='MultiPac: 42.5 MV/m')

        def update(epk_i, w):

            particles = self.particles_objects[epk_i]
            if len(particles.bright_set) != 0:
                line.set_data(-particles.bright_set[w][:, 0] * 1e3, particles.bright_set[w][:, 1] * 1e3)
                line.set_label(f'PyMultipact: {self.epks_v[epk_i] * self.Epk * 1e-6} MV/m')

                line_init.set_data(-particles.bright_set[w][:, 0][0] * 1e3,
                                   particles.bright_set[w][:, 1][0] * 1e3)  # plot initial point
                #         line_end.set_data(-particles.bright_set[w][:, 0][-1]*1e3, particles.bright_set[w][:, 1][-1]*1e3) # plot end point
                #         line2.set_data(particles.bright_set[w][:, 2]*1e3, particles.bright_set[w][:, 0]*1e3)
                #         line2.set_label(w)
                #         line3.set_data(particles.bright_set[w][:, 2]*1e3, particles.bright_set[w][:, 1]*1e3)
                #         line3.set_label(w)

                # Set limits for the data
                x_min, x_max = min(particles.bright_set[w][:, 0] * 1e3), max(particles.bright_set[w][:, 0] * 1e3)
                y_min, y_max = min(particles.bright_set[w][:, 1] * 1e3), max(particles.bright_set[w][:, 1] * 1e3)

                # Calculate padding dynamically based on the range of data
                padding_factor = 0.1  # adjust this factor as needed
                x_padding = (x_max - x_min) * padding_factor
                y_padding = (y_max - y_min) * padding_factor

                # Add padding around the plot
                #         axs[0].set_xlim(x_min - x_padding, x_max + x_padding)
                axs[0].set_xlim(-(x_max + x_padding), -(x_min - x_padding))
                axs[0].set_ylim(y_min - y_padding, y_max + y_padding)

                axs[0].set_aspect('equal', 'box')
                for ii in axs:
                    axs[ii].legend(loc='lower right')

            #     plt.autoscale()
            fig.canvas.draw_idle()
            # plt.savefig("trajectory_comparison.png", dpi=150)

        interact(update, epk_i=epk_i_slider, w=w_slider);

    @staticmethod
    def _select_values_with_step(values, step):
        selected_values = []
        last_value = values[0][0] - step

        for value in values:
            if value[0] >= last_value + step:
                selected_values.append(value)
                last_value = value[0]

        return np.array(selected_values)


class SEY:
    def __init__(self, sey_filepath):
        self.data = pd.read_csv(sey_filepath, delim_whitespace=True, header=None, names=["E", "sey"])
        self.Emax = max(self.data['E'])
        self.Emin = min(self.data['E'])
        self.sey = CubicSpline(self.data['E'], self.data['sey'])


class EMField:
    def __init__(self, e, h):
        self.e = e
        self.h = h


class Project:
    def __init__(self):
        self.default_folder = '.'
        self.folder = '.'

    def create_project(self, folder_path):
        # check if path exists
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        self.folder = folder_path

    def load_project(self, folder_path):
        pass
