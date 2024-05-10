import copy
import itertools

from matplotlib import cm
import ngsolve as ng
from ngsolve.webgui import Draw
import netgen.occ as ngocc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import scipy

from integrators import Integrators
from particles import Particles

q0 = 1.60217663e-19
m0 = 9.1093837e-31
mu0 = 4 * np.pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458


class Domain:
    def __init__(self, boundary_file=None, field=None):
        self.mesh = None
        self.domain = None
        cav_geom = pd.read_csv('sample_domains/tesla_mid_cell.n',
                               header=None, skiprows=3, skipfooter=1, sep='\s+', engine='python')[[1, 0]]

        self.boundary = np.array(list(cav_geom.itertuples(index=False, name=None)))

        self.particles = np.array([])
        if field is None:
            self.field = None
        else:
            self.field = field

        self.sey = None
        # set default sey
        self.set_sey(r'sample_seys/sey')

        self.zmin, self.zmax, self.rmin, self.rmax = [0, 0, 0, 0]
        self.gfu_E = None
        self.gfu_H = None
        self.eigenvals, self.eigenvecs = None, None

        # define domain
        self.define_domain()

    def set_boundary_conditions(self, zmin='PMC', zmax='PMC', rmin='PEC', rmax='PEC'):
        self.zmin, self.zmax, self.rmin, self.rmax = [zmin, zmax, rmin, rmax]

    def define_domain(self, maxh=0.000577):
        wp = ngocc.WorkPlane()
        wp.MoveTo(*self.boundary[0])
        for p in self.boundary[1:]:
            try:
                wp.LineTo(*p)
            except:
                pass
        wp.Close().Reverse()
        self.domain = wp.Face()

        # name the boundaries
        self.domain.edges.Max(ngocc.X).name = "zmax"
        self.domain.edges.Max(ngocc.X).col = (1, 0, 0)
        self.domain.edges.Min(ngocc.X).name = "zmin"
        self.domain.edges.Min(ngocc.X).col = (1, 0, 0)
        self.domain.edges.Min(ngocc.Y).name = "rmin"
        self.domain.edges.Min(ngocc.Y).col = (1, 0, 0)

        geo = ngocc.OCCGeometry(self.domain, dim=2)

        # mesh
        ngmesh = geo.GenerateMesh()
        # ngmesh = geo.GenerateMesh(maxh=maxh)
        self.mesh = ng.Mesh(ngmesh)

    def compute_fields(self):
        # define finite element space
        fes = ng.HCurl(self.mesh, order=1, dirichlet='default')
        u, v = fes.TnT()

        self.a = ng.BilinearForm(ng.y * ng.curl(u) * ng.curl(v) * ng.dx).Assemble()
        self.m = ng.BilinearForm(ng.y * u * v * ng.dx).Assemble()

        apre = ng.BilinearForm(ng.y * ng.curl(u) * ng.curl(v) * ng.dx + ng.y * u * v * ng.dx)
        pre = ng.Preconditioner(apre, "direct", inverse="sparsecholesky")

        with ng.TaskManager():
            self.a.Assemble()
            self.m.Assemble()
            apre.Assemble()

            # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
            gradmat, fesh1 = fes.CreateGradient()
            gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
            math1 = gradmattrans @ self.m.mat @ gradmat  # multiply matrices
            math1[0, 0] += 1  # fix the 1-dim kernel
            invh1 = math1.Inverse(inverse="sparsecholesky", freedofs=fesh1.FreeDofs())
            # build the Poisson projector with operator Algebra:
            proj = ng.IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ self.m.mat
            projpre = proj @ pre.mat

            self.eigenvals, self.eigenvecs = ng.solvers.PINVIT(self.a.mat, self.m.mat, pre=projpre, num=3, maxit=20,
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

    def analyse_multipacting(self, mode=1, init_pos=None, epks=None, phis=None, v_init=2, init_points=None,
                             integrator='rk4'):
        if self.sey is None:
            print("Secondary emission yield not defined, using default sey.")

        # epks_v = 1/Epk*1e6*np.linspace(1, 60, 119)
        # phi_v = np.linspace(0, 2*np.pi, 72)  # <- initial phase
        # epks_v = 1 / Epk * 1e6 * np.linspace(30, 60, 1)
        print(self.boundary)
        xpnts_surf = self.boundary[(self.boundary[:, 1] > 0) & (self.boundary[:, 0] > min(self.boundary[:, 0])) & (
                    self.boundary[:, 0] < max(self.boundary[:, 0]))]
        plt.plot(xpnts_surf[:, 0], xpnts_surf[:, 1])
        plt.show()
        Esurf = [ng.Norm(self.gfu_E[mode])(self.mesh(xi, yi)) for (xi, yi) in xpnts_surf]
        Epk = (max(Esurf))

        if epks is None:
            epks_v = 1 / Epk * 1e6 * np.linspace(47, 80, 1)
        else:
            epks_v = epks

        if phis is None:
            phi_v = np.linspace(0, 2 * np.pi, 10)  # <- initial phase
        else:
            phi_v = phis

        if init_pos is None:
            init_pos = [-0.00025, 0.00025]

        # get surface points
        pec_boundary = self.mesh.Boundaries("default")
        bel = [xx.vertices for xx in pec_boundary.Elements()]
        bel_unique = list(set(itertools.chain(*bel)))
        xpnts_surf = sorted([self.mesh.vertices[xy.nr].point for xy in bel_unique])
        xsurf = np.array(xpnts_surf)
        print(xsurf)

        # define one internal point
        xin = [0, 0]
        integrators = Integrators(self, mode)
        no_of_remaining_particles = []

        # calculate time for 10 cycles, 20 alternations
        T = 1 / (self.eigen_freq[mode] * 1e6) * 10
        lmbda = c0 / (self.eigen_freq[mode] * 1e6)

        particles_left = []
        particles_nhits = []
        particles_objects = []

        for epk in epks_v:
            t = 0
            dt = 1e-11
            PLOT = False
            error = False
            counter = 0

            particles = Particles(init_pos, v_init, xsurf, phi_v, cmap='jet')

            print('Initial number of particles: ', len(particles.x))
            em = EMField(copy.deepcopy(self.gfu_E[mode]), copy.deepcopy(self.gfu_H[mode]))

            # move particles with initial velocity. ensure all initial positions after first move lie inside the bounds
            particles.x = particles.x + particles.u * dt

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
            particles_objects.append(particles)

            if len(particles.nhit) == 0:
                particles_nhits.append(0)
            else:
                particles_nhits.append(particles.nhit[0])

            particles_left.append(len(particles.bright_set))
            print(f"Epk: {epk * Epk * 1e-6} MV/m, particles in bright set: {len(particles.bright_set)}")

        print("Done with multipacting analysis.")

    def set_sey(self, sey_filepath):
        self.sey = SEY(sey_filepath)

    def get_sey(self):
        return self.sey

    def plot_sey(self):
        fig, ax = plt.subplots()
        ax.plot(self.sey.data['E'][:-1], self.sey.data['sey'][:-1])
        ax.axhline(1, 0, color='r')
        plt.show()


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
