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

# repository root (parent of the package) -- used to resolve the bundled
# sample data regardless of the caller's working directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sample_path(relpath):
    """Resolve a bundled sample-data path (sample_seys/, sample_domains/).

    Tries the path as given (so explicit/relative user paths keep working),
    then falls back to the repository root, so notebooks and scripts can run
    from any directory once the package is installed (pip install -e .)."""
    if os.path.exists(relpath):
        return relpath
    candidate = os.path.join(_REPO_ROOT, relpath)
    return candidate if os.path.exists(candidate) else relpath


class Domain:
    def __init__(self, project, boundary_file=None, field=None,
                 n_boundary_points=250, **kwargs):
        """

        Parameters
        ----------
        project: str
            Project directory
        boundary_file: str
            Boundary file path
        field: bytearray
            Field to be loaded
        n_boundary_points: int or None
            Target number of boundary polyline points. Dense boundary files
            (e.g. tesla_mid_cell.n with ~1830 points; the original is kept as
            tesla_mid_cell_fine.n) force netgen to anchor a mesh node at every
            point, producing a needlessly fine surface mesh. With 3rd-order
            field elements (see compute_fields) ~250 points retain accuracy
            while shrinking the mesh and the collision surface considerably.
            None keeps the file's full resolution.
        """

        self.cn_c0 = None
        self.bounding_rect = None
        self.project_folder = project.folder
        self.n_boundary_points = n_boundary_points

        self.fig, self.ax = plt.subplots()
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
        self.order = 3

        if field is None:
            self.field = None
        else:
            self.field = field

        self.sey = None
        # set default sey
        self.set_sey(_sample_path('sample_seys/sey'))

        self.bc_zmin, self.bc_zmax, self.bc_rmin, self.bc_rmax = [0, 0, 0, 0]
        self.gfu_E = None
        self.gfu_H = None
        self.eigenvals, self.eigenvecs = None, None

        self.define_boundary(kwargs=kwargs)
        # define domain
        self.load_boundary(_sample_path('sample_domains/tesla_mid_cell.n'))

    def load_boundary(self, geopath):
        """

        Parameters
        ----------
        geopath: str
            Path to geometry file

        Returns
        -------

        """
        if geopath is None:
            print("Please enter geometry path.")
            return
        try:
            # read geometry
            cav_geom = pd.read_csv(geopath, header=None, skiprows=3, skipfooter=1,
                                   sep=r'\s+', engine='python')[[1, 0]]
            self.boundary = self._resample_boundary(
                np.array(list(cav_geom.itertuples(index=False, name=None))),
                self.n_boundary_points)
            self.mesh_domain()
        except Exception as e:
            print("Please enter valid geometry path.", e)

    @staticmethod
    def _resample_boundary(boundary, n_points):
        """Decimate a dense boundary polyline to ~n_points, keeping ORIGINAL
        points at uniform arc-length spacing (no new points are invented, so
        every kept point lies exactly on the design profile). The endpoints and
        the extreme-coordinate points (z/r min and max, which define the
        bounding rectangle) are always kept. n_points=None disables."""
        boundary = np.asarray(boundary)
        if n_points is None or len(boundary) <= n_points:
            return boundary
        seg = np.linalg.norm(np.diff(boundary, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        targets = np.linspace(0.0, s[-1], int(n_points))
        keep = set(np.clip(np.searchsorted(s, targets), 0, len(boundary) - 1).tolist())
        keep.update([0, len(boundary) - 1,
                     int(np.argmin(boundary[:, 0])), int(np.argmax(boundary[:, 0])),
                     int(np.argmin(boundary[:, 1])), int(np.argmax(boundary[:, 1]))])
        resampled = boundary[sorted(keep)]
        print(f"Boundary polyline resampled: {len(boundary)} -> {len(resampled)} points "
              f"(n_boundary_points=None keeps the full resolution)")
        return resampled

    def define_boundary(self, kind='cavity', name='geodata', **kwargs):
        """

        Parameters
        ----------
        kind: ['cavity']
            Type of geometry
        name: str
            Name of the geometry
        kwargs: dict
            Extra parameters depending on the geometry kind

        Returns
        -------

        """

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
                                       sep=r'\s+', engine='python')[[1, 0]]

                self.boundary = self._resample_boundary(
                    np.array(list(cav_geom.itertuples(index=False, name=None))),
                    self.n_boundary_points)

            self.mesh_domain()
        except Exception as e:
            print("Please enter valid geometry path.", e)

    def show_initial_points(self, xrange, step=None):
        """

        Parameters
        ----------
        xrange: list, ndarray
            Interval of initial surface points
        step: int
            Minimum distance between initial surface points

        Returns
        -------

        """
        pts = self.boundary[(self.boundary[:, 0] > xrange[0]) & (self.boundary[:, 0] < xrange[1])]

        if step:
            pts = self._select_values_with_step(pts, step)

        fig, ax = plt.subplots()
        ax.plot(self.boundary[:, 0], self.boundary[:, 1])
        ax.scatter(pts[:, 0], pts[:, 1], fc='None', ec='k', s=50)
        plt.show()

    def define_elliptical_cavity(self, mid_cell=None, lend_cell=None, rend_cell=None, beampipe='None'):
        """

        Parameters
        ----------
        mid_cell: list, ndarray
            Array of cavity middle cells' geometric parameters
        lend_cell: list, ndarray
            Array of cavity left end cell's geometric parameters
        rend_cell: list, ndarray
            Array of cavity left end cell's geometric parameters
        beampipe: str {"left", "right", "both", "none"}
            Specify if beam pipe is on one or both ends or at no end at all

        Returns
        -------

        """
        kwargs = {
            'mid_cell': mid_cell,
            'lend_cell': lend_cell,
            'rend_cell': rend_cell,
            'beampipe': None
        }
        self.define_boundary(kind='cavity', **kwargs)

    def set_boundary_conditions(self, zmin='PMC', zmax='PMC', rmin='PEC', rmax='PEC'):
        """

        Parameters
        ----------
        zmin: str
            ['PEC', 'PMC']
        zmax str
            ['PEC', 'PMC']
        rmin str
            ['PEC', 'PMC']
        rmax str
            ['PEC', 'PMC']

        Returns
        -------

        """
        self.bc_zmin, self.bc_zmax, self.bc_rmin, self.bc_rmax = [zmin, zmax, rmin, rmax]

    def mesh_domain(self, maxh=0.00577):
        """

        Parameters
        ----------
        maxh: float
            Mesh resolution

        Returns
        -------

        """
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
        self.mesh.Curve(self.order)

        # save mesh
        with open(f"{self.project_folder}/mesh.pkl", "wb") as f:
            pickle.dump(self.mesh, f)

    def compute_fields(self):
        """Solve the eigenmodes.

        Parameters
        ----------
        order: int
            Finite element order. 3 (default) matches MultiPac's third-order
            elements and keeps field accuracy on the coarser resampled
            boundary/mesh; the original code used order=1 on a very dense
            surface mesh.
        """
        # define finite element space
        fes = ng.HCurl(self.mesh, order=self.order, dirichlet='default')
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
        Function to draw the domain

        Returns
        -------

        """
        Draw(self.domain)

    def draw_mesh(self):
        Draw(self.mesh)

    def draw_fields(self, mode=1, which='E'):
        """

        Parameters
        ----------
        mode: int
            Mode number
        which: ['E', 'H']
            E for electric field or H for magnetic field

        Returns
        -------

        """
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

    def analyse_multipacting(self, mode=1, xrange=None, epks=None, phis=None,
                             v_init=2, integrator='rk4', step=None, proc_count=None,
                             loss_model='field'):
        """
        Analyse multipacting. The Epk sweep is run in parallel by default.

        Parameters
        ----------
        mode: int
            Eigenmode index
        xrange: list, ndarray
            Range of surface emission sites (z interval)
        epks: list, ndarray
            Peak surface electric field sweep values [V/m]
        phis: list, ndarray
            Initial phases
        v_init: float, int
            Particle emission energy [eV]
        integrator: str
            Numerical integration scheme
        step: float
            Minimum distance between emission sites
        proc_count: int or None
            Number of worker processes for the Epk sweep. None (default)
            chooses automatically from the machine's CPU count and the number
            of sweep points; 1 runs in-process without spawning workers.
            Note for Windows scripts: guard the call with
            ``if __name__ == '__main__':`` (multiprocessing spawn requirement);
            notebooks are fine as-is.
        loss_model: str
            What happens to an electron impacting the wall while the surface
            field is unfavourable (E.n < 0): 'field' absorbs it (paper
            behaviour, default); 'wait' re-emits it uncounted until the RF
            phase turns favourable (MultiPac-style delayed re-emission);
            'always' re-emits and counts the impact (upper bound).

        Returns
        -------

        """
        lmbda = c0 / (self.eigen_freq[mode] * 1e6)
        if self.sey is None:
            print("Secondary emission yield not defined, using default sey.")

        xpnts_surf_ = self.boundary[(self.boundary[:, 1] > 0) & (self.boundary[:, 0] > min(self.boundary[:, 0])) & (
                self.boundary[:, 0] < max(self.boundary[:, 0]))]
        Esurf = [ng.Norm(self.gfu_E[mode])(self.mesh(xi, yi)) for (xi, yi) in xpnts_surf_]
        self.Epk = (max(Esurf))

        if epks is None:
            self.epks_v = 1 / self.Epk * 1e6 * np.linspace(0, 80, 192)
        else:
            # user-passed peak fields are in V/m; normalise by 1/Epk
            self.epks_v = 1 / self.Epk * np.asarray(epks)

        if phis is None:
            phi_v = np.linspace(0, 2 * np.pi, 72)  # <- initial phase
        else:
            phi_v = phis

        if xrange is None:
            xrange = [-0.00025, -0.000]

        # worker count: leave one core free, never more workers than sweep points
        if proc_count is None:
            proc_count = max(1, min(mp.cpu_count() - 1, len(self.epks_v)))
        proc_count = max(1, int(proc_count))
        print(f"Running Epk sweep ({len(self.epks_v)} points) on {proc_count} "
              f"process{'es' if proc_count > 1 else ''}.")

        # save mode fields for the workers
        with open(f"{self.project_folder}/gfu_EH.pkl", "wb") as f:
            pickle.dump([self.gfu_E[mode], self.gfu_H[mode]], f)

        # round-robin split of the sweep, remembering original indices so the
        # results can be re-assembled in epks_v order
        divided_lists = [[] for _ in range(proc_count)]
        divided_idx = [[] for _ in range(proc_count)]
        for idx, value in enumerate(self.epks_v):
            divided_lists[idx % proc_count].append(value)
            divided_idx[idx % proc_count].append(idx)
        proc_epks_list = [np.array(lst) for lst in divided_lists]

        # remove stale worker outputs so a crashed worker cannot be silently
        # replaced by results from a previous run
        for p in range(proc_count):
            stale = f"{self.project_folder}/mresults_{p}"
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

        start = time.time()
        if proc_count == 1:
            # in-process, same code path as the workers, without a subprocess
            self._analyse_multipacting(0, self.project_folder, self.eigen_freq,
                                       mode, xrange, proc_epks_list[0], phi_v,
                                       v_init, self.sey, self.Epk, step,
                                       self.bounding_rect, loss_model)
        else:
            processes = []
            for p in range(proc_count):
                service = mp.Process(target=self._analyse_multipacting,
                                     args=(p, self.project_folder, self.eigen_freq,
                                           mode, xrange, proc_epks_list[p], phi_v,
                                           v_init, self.sey, self.Epk, step,
                                           self.bounding_rect, loss_model))
                service.start()
                processes.append(service)

            # Wait for all processes to complete
            for service in processes:
                service.join()

        # compile results, restoring the original epks_v order (the round-robin
        # split would otherwise leave cn_c0 interleaved relative to epks_v and
        # plot_cf would pair wrong values)
        cf_by_idx = {}
        po_by_idx = {}
        for p in range(proc_count):
            result_file = f"{self.project_folder}/mresults_{p}"
            if not os.path.exists(result_file):
                raise RuntimeError(f"Worker {p} produced no result file "
                                   f"({result_file}) -- it probably crashed; "
                                   f"check its console output.")
            with open(result_file, "rb") as file:
                m_result = pickle.load(file)

            for local_i, global_i in enumerate(divided_idx[p]):
                cf_by_idx[global_i] = m_result['cn/c0'][local_i]
                po_by_idx[global_i] = m_result['particles_objects'][local_i]

            if p == 0:
                self.n_init_particles = m_result['n_init_particles']

        self.cn_c0 = np.array([cf_by_idx[i] for i in range(len(self.epks_v))])
        self.particles_objects = [po_by_idx[i] for i in range(len(self.epks_v))]
        self.particles_left = [len(po.bright_set) for po in self.particles_objects]
        self.particles_nhits = [po.nhit[0] if len(po.nhit) else 0 for po in self.particles_objects]

        # distance function of each surviving (bright) trajectory
        for po in self.particles_objects:
            self.calculate_distance_function(po, lmbda)

        # persist combined results
        mresult = {'cn/c0': self.cn_c0,
                   'particles_objects': self.particles_objects,
                   'n_init_particles': self.n_init_particles,
                   'Epk': self.Epk,
                   'epks': self.epks_v,
                   'phis_v': phi_v}
        with open(f"{self.project_folder}/mresults.pkl", "wb") as file:
            pickle.dump(mresult, file)

        print("Total runtime:: ", time.time() - start)
        print("Done with multipacting analysis.")

    def analyse_multipacting_parallel(self, proc_count=1, mode=1, xrange=None, epks=None, phis=None,
                                      v_init=2, integrator='rk4', step=None):
        """Deprecated alias -- analyse_multipacting is parallel by default now."""
        print("analyse_multipacting_parallel is deprecated; use analyse_multipacting "
              "(parallel by default, proc_count=... to override).")
        return self.analyse_multipacting(mode=mode, xrange=xrange, epks=epks, phis=phis,
                                         v_init=v_init, integrator=integrator, step=step,
                                         proc_count=proc_count)

    @staticmethod
    def _analyse_multipacting(proc_id, folder, eigen_freq, mode, xrange, procs_epks, phis,
                              v_init, sey, Epk, step, bounding_rect, loss_model='field'):
        n_init_particles = 1
        # pickle mesh and fields
        with open(f'{folder}/mesh.pkl', 'rb') as f:
            mesh = pickle.load(f)
        with open(f'{folder}/gfu_EH.pkl', "rb") as f:
            [gfu_E, gfu_H] = pickle.load(f)

        # get surface points
        pec_boundary = mesh.Boundaries("default")
        bel = [xx.vertices for xx in pec_boundary.Elements()]
        bel_unique = list(set(itertools.chain(*bel)))
        xpnts_surf = sorted([mesh.vertices[xy.nr].point for xy in bel_unique])
        xsurf = np.array(xpnts_surf)

        # calculate time for 10 cycles, 20 alternations
        # T = 1 / (eigen_freq[mode] * 1e6) * 10
        dt = 1 / (eigen_freq[mode] * 1e6 * 20 * 6)
        # lmbda = c0 / (eigen_freq[mode] * 1e6)

        w = 2 * np.pi * eigen_freq[mode] * 1e6
        integrator = Integrators(mesh, w, bounding_rect=bounding_rect,
                                 loss_model=loss_model)

        # Field object built once (independent of the Epk sweep value).
        em = EMField(gfu_E, gfu_H)

        particles_left = []
        particles_nhits = []
        particles_objects = []

        start = time.time()
        for epk in procs_epks:
            sub_start = time.time()
            t = 0
            counter = 0

            particles = Particles(xrange, v_init, xsurf, phis, cmap='jet', step=step)

            n_init_particles = len(particles.x)
            print(f'\t{proc_id}: Initial number of particles: ', n_init_particles)

            # # move particles with initial velocity. ensure all initial positions after first move lie inside the bounds
            # particles.x = particles.x + particles.u * dt  # remove later

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
                f"\tEpk: {epk * Epk * 1e-6} MV/m, particles in bright set: {len(particles.bright_set)}, time: {time.time() - sub_start}")

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

        print(f"\tProc {proc_id} done with multipacting analysis. Time: ", time.time() - start)

    def set_sey(self, sey_filepath):
        """
        Set custom secondary emission yield

        Parameters
        ----------
        sey_filepath: str, Path
            Secondary emission yield file path

        Returns
        -------

        """
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
        """Distance function d_20 of each bright (20-hit) trajectory,
        Yla-Oijala Eq. (3.2.3): the distance in (position, phase) space between
        the initial point (emission site, launch phase) and the 20th impact
        point. Minima locate the fixed points of resonant multipacting orbits.
        Stored in particles.df20, aligned with bright_set."""
        kappa = lmbda / (2 * np.pi)
        particles.df20 = []
        use_exact = (hasattr(particles, 'bright_impact_x')
                     and len(getattr(particles, 'bright_impact_x', []))
                     == len(particles.bright_set))
        for path_i in range(len(particles.bright_set)):
            if use_exact and len(particles.bright_impact_x[path_i]) > 0:
                # exact n-th impact point and RF phase (thesis definition)
                x_0 = np.asarray(particles.bright_init_x[path_i])
                phi_0 = particles.bright_init_phi[path_i]
                x_n = np.asarray(particles.bright_impact_x[path_i][-1])
                phi_n = particles.bright_impact_phi[path_i][-1]
            else:
                # fallback for results predating the impact archive: use the
                # first/last recorded path rows
                path = particles.bright_set[path_i]
                x_0, x_n = path[0, 0:2], path[-1, 0:2]
                phi_0, phi_n = path[0, 2], path[-1, 2]
            df = np.sqrt(np.linalg.norm(x_n - x_0) ** 2
                         + kappa * abs(np.exp(1j * phi_n) - np.exp(1j * phi_0)) ** 2)
            particles.df20.append(float(df))

    def calculate_Ef(self):
        """Mean FINAL impact energy of the electrons that reached 20 hits (the
        bright set) -- zero wherever nothing survived to 20 hits, exactly like
        the paper's Ef_20. Computing this over the leftover particles instead
        produced spurious out-of-band spikes from runaway lost particles."""
        self.Ef = []
        for particles in self.particles_objects:
            bright_E = getattr(particles, 'bright_E', None)
            if bright_E is None:
                # backward compatibility with result pickles from before the
                # bright-history archive existed
                Ef_p = [pe[-1] for pe in particles.E if len(pe) != 0]
            else:
                Ef_p = [be[-1] for be in bright_E if len(be) != 0]
            self.Ef.append(np.sum(Ef_p) / len(Ef_p) if len(Ef_p) > 0 else 0)

        return self.Ef

    def save_fields(self):
        pass

    def launchable_fraction(self, mode=1):
        """Fraction of the launched (site, phase) combinations whose surface
        field at emission allows the electron to leave the wall (E.n >= 0).
        For a sinusoidal field this is ~0.5: half of all initial phases die on
        their first impacts. MultiPac's counter function effectively counts
        only launchable electrons in c0, so to compare against MultiPac divide
        cn_c0 by this fraction (see plot_cf(launchable_norm=True))."""
        p0 = (self.particles_objects or [None])[0]
        gfu_E, mesh = self.gfu_E, self.mesh
        if p0 is None or gfu_E is None or mesh is None or not hasattr(p0, 'sites_init'):
            return 0.5  # sinusoidal-field default
        sites = np.asarray(p0.sites_init)
        normals = np.asarray(p0.pt_normals[:p0.n_sites])
        phis = np.asarray(p0.phis_v)
        fav, tot = 0, 0
        for sx, nrm in zip(sites, normals):
            Ec = np.asarray(gfu_E[mode](mesh(float(sx[0]), float(sx[1]))),
                            dtype=complex).ravel()[:2]
            e_at_phis = np.real(np.outer(np.exp(1j * phis), Ec))  # (n_phis, 2)
            fav += int(np.sum(e_at_phis @ nrm >= 0))
            tot += len(phis)
        return fav / tot if tot else 0.5

    def plot_cf(self, launchable_norm=False):
        """Counter function. launchable_norm=True divides by the fraction of
        launchable initial electrons (E.n >= 0 at emission, ~0.5), which is the
        normalisation MultiPac's c20/c0 effectively uses."""
        cf = np.asarray(self.cn_c0, dtype=float)
        label = '$c_\\mathrm{20}/c_\\mathrm{0}$'
        if launchable_norm:
            frac = self.launchable_fraction()
            cf = cf / frac
            label += f'  (launchable norm, /{frac:.2f})'
        fig, ax = plt.subplots()
        ax.plot(self.epks_v * self.Epk * 1e-6, cf)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$E_\mathrm{pk}$ [MV/m]')
        ax.set_ylabel(label)
        plt.show()

    def plot_Ef(self):
        """
        Plot average final impact energy for peak field values

        Returns
        -------

        """
        if not getattr(self, 'Ef', None):
            self.calculate_Ef()
        fig, ax = plt.subplots()
        ax.plot(self.epks_v * self.Epk * 1e-6, self.Ef)

        # SEY reference lines: first/second crossover energies (sey = 1,
        # solid) and the peak-sey energy (dashed), from the loaded SEY table.
        if self.sey is not None:
            sey_E = np.asarray(self.sey.data['E'], dtype=float)
            sey_v = np.asarray(self.sey.data['sey'], dtype=float)
            above = sey_v > 1
            crossings = []
            for i in np.nonzero(np.diff(above.astype(int)) != 0)[0]:
                # linear interpolation of the sey = 1 crossing in [E_i, E_i+1]
                crossings.append(sey_E[i] + (1 - sey_v[i]) * (sey_E[i + 1] - sey_E[i])
                                 / (sey_v[i + 1] - sey_v[i]))
            for E_cross in crossings:
                ax.axhline(E_cross, c='r')
            if np.any(above):
                ax.axhline(sey_E[np.argmax(sey_v)], c='r', ls='--')

        ax.set_yscale('log')
        ax.set_xlabel(r'$E_\mathrm{pk}$ [MV/m]')
        ax.set_ylabel(r'$E_\mathrm{f, 20}$ [eV]')
        plt.show()

    def plot_ef(self):
        # e20/c0 from the archived impact histories of the 20-hit (bright)
        # electrons. Each archived list has at most ~20 entries; using the
        # live (previously misaligned) lists let entries from many different
        # particles pile into one list and the product blow up astronomically.
        secondaries = [
            sum(np.prod(nn) for nn in getattr(particles, 'bright_n_secondaries',
                                              particles.n_secondaries))
            for particles in (self.particles_objects or [])]
        if len(secondaries) > 0:
            fig, ax = plt.subplots()
            ax.plot(self.epks_v * self.Epk * 1e-6, 2 * (np.array(secondaries) + 1) / self.n_init_particles)
            ax.axhline(1, c='r')
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-3)
            ax.set_xlabel(r'$E_\mathrm{pk}$ [MV/m]')
            ax.set_ylabel(r'$e_\mathrm{20}/c_\mathrm{0}$')
            plt.show()
        else:
            print('No secondaries to plot!')

    def plot_df(self, epk_i, metric='d20', vmax=None):
        """MultiPac-style distance map over (emission site, initial phase) for
        the epk_i-th field level of the sweep. Grey cells = no electron
        survived to 20 impacts from that (site, phase).

        Parameters
        ----------
        epk_i: int
            Index into the Epk sweep (self.epks_v).
        metric: str
            'd20' (default): distance between the initial (site, phase) and
            the nearest of the last two impacts. For two-point multipacting
            the orbit returns to the launch side only on every other impact,
            so taking the closer of impacts 19 and 20 removes the arbitrary
            branch parity: launches at the resonant phase give d ~ 0 (dark
            core) and off-core launches grow with their phase offset --
            the same graded structure as MultiPac's d20 map.
            'd20_strict': the literal Yla-Oijala Eq. 3.2.3 (20th impact
            only). Carries a ~pi phase offset whenever the 20th impact lands
            on the opposite branch.
            'closure': distance between the 20th and 18th impacts (closure of
            the two-impact map); zero for any phase-locked orbit.
        vmax: float, None or 'kappa'
            Colour scale maximum. 'kappa' (default for 'd20') clips at
            lambda/(2 pi) like MultiPac's d20 display; None autoscales.
        """
        particles_objects = self.particles_objects
        if not particles_objects:
            print("No results to plot -- run analyse_multipacting first.")
            return
        particles = particles_objects[epk_i]
        if not hasattr(particles, 'bright_init_x') or not hasattr(particles, 'sites_init'):
            print("Result predates the bright-identity archive; re-run the analysis.")
            return
        eigen_freq = self.eigen_freq if self.eigen_freq is not None else [0, 1300.0]
        lmbda = c0 / (eigen_freq[1] * 1e6)
        kappa = lmbda / (2 * np.pi)
        if not hasattr(particles, 'df20'):
            self.calculate_distance_function(particles, lmbda)
        epks_v = np.asarray(self.epks_v)
        boundary = np.asarray(self.boundary)

        # per-bright metric values
        def _dist(x_a, phi_a, x_b, phi_b):
            return float(np.sqrt(
                np.linalg.norm(np.asarray(x_a) - np.asarray(x_b)) ** 2
                + kappa * abs(np.exp(1j * phi_a) - np.exp(1j * phi_b)) ** 2))

        if metric == 'closure':
            values = []
            for xs, ps in zip(particles.bright_impact_x, particles.bright_impact_phi):
                values.append(_dist(xs[-1], ps[-1], xs[-3], ps[-3])
                              if len(xs) >= 3 else np.nan)
            label = 'closure $d(x_{20}, x_{18})$'
        elif metric == 'd20' and not hasattr(particles, 'bright_impact_x'):
            # results predating the impact archive
            values = particles.df20
            label = '$d_\\mathrm{20}$'
        elif metric == 'd20':
            # parity-robust: closer of the last two impacts to the launch point
            values = []
            for x0, p0, xs, ps in zip(particles.bright_init_x,
                                      particles.bright_init_phi,
                                      particles.bright_impact_x,
                                      particles.bright_impact_phi):
                if len(xs) >= 2:
                    values.append(min(_dist(x0, p0, xs[-1], ps[-1]),
                                      _dist(x0, p0, xs[-2], ps[-2])))
                elif len(xs) == 1:
                    values.append(_dist(x0, p0, xs[-1], ps[-1]))
                else:
                    values.append(np.nan)
            label = '$d_\\mathrm{20}$'
            if vmax is None:
                vmax = 'kappa'   # MultiPac-style display by default
        elif metric == 'd20_strict':
            values = particles.df20
            label = '$d_\\mathrm{20}$ (strict)'
        else:
            raise ValueError(f"metric must be 'd20', 'd20_strict' or 'closure', "
                             f"got {metric!r}")

        sites = np.asarray(particles.sites_init)
        phis_v = np.asarray(particles.phis_v)
        dmap = np.full((len(phis_v), len(sites)), np.nan)
        for bx, bphi, df in zip(particles.bright_init_x, particles.bright_init_phi,
                                values):
            si = int(np.argmin(np.linalg.norm(sites - np.asarray(bx), axis=1)))
            pi = int(np.argmin(np.abs(phis_v - bphi)))
            dmap[pi, si] = df

        fig, axs = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[2, 1.2])
        cmap = plt.get_cmap('hot').copy()
        cmap.set_bad('0.85')   # non-surviving cells: light grey, not white
        if vmax == 'kappa':
            vmax = kappa       # MultiPac-style display (~0.04 at 1.3 GHz)
        im = axs[0].pcolormesh(np.arange(1, len(sites) + 1), np.degrees(phis_v),
                               dmap, cmap=cmap, shading='nearest',
                               vmin=0.0, vmax=vmax)
        fig.colorbar(im, ax=axs[0], label=label,
                     extend='max' if vmax is not None else 'neither')
        axs[0].set_xlabel('Place referring to picture below')
        axs[0].set_ylabel('Initial phase [deg]')
        axs[0].set_title(f'Distance map ({metric})   '
                         f'$E_\\mathrm{{pk}}$ = {epks_v[epk_i] * self.Epk * 1e-6:.1f} MV/m')

        axs[1].plot(boundary[:, 0], boundary[:, 1], 'r', lw=1)
        axs[1].plot(sites[:, 0], sites[:, 1], 'o', mfc='none', mec='b', ms=5)
        for k, (sz, sr) in enumerate(sites):
            axs[1].annotate(str(k + 1), (sz, sr), fontsize=7)
        axs[1].set_xlabel('z [m]')
        axs[1].set_ylabel('r [m]')
        axs[1].set_aspect('equal', 'box')
        axs[1].set_title('Initial points')
        fig.tight_layout()
        plt.show()

    def get_sey(self):
        return self.sey

    def plot_sey(self):
        fig, ax = plt.subplots()
        ax.plot(self.sey.data['E'][:-1], self.sey.data['sey'][:-1])
        ax.axhline(1, 0, color='r')
        ax.set_xlabel('$#delta$')
        ax.set_ylabel('Impact Energy [eV]')
        plt.show()

    def plot_trajectories(self):
        if not self.particles_objects:
            print("No results to plot -- run analyse_multipacting first "
                  "(or load results with load_multipacting_result).")
            return

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

        # Create slider widgets. max is the last valid INDEX (was off by one and
        # raised IndexError at the top of the slider); start values are clamped
        # to the actual result size instead of being hardcoded.
        n_epks = len(self.particles_objects)
        epk_i_slider = IntSlider(min=0, max=n_epks - 1, step=1, description='epk_i:',
                                 layout=Layout(width='50%'),
                                 value=min(83, n_epks - 1))
        # Observe changes in the value attribute of epk_i_slider and update w_slider accordingly
        epk_i_slider.observe(update_w_max, names='value')
        n_bright0 = len(self.particles_objects[epk_i_slider.value].bright_set)
        w_slider = IntSlider(min=-1, max=n_bright0 - 1,
                             description='w:',
                             layout=Layout(width='50%'), value=min(28, n_bright0 - 1))

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


class _TableInterp:
    """Picklable linear table interpolation (multiprocessing passes SEY through
    process arguments, so a lambda/closure would break the parallel path)."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def __call__(self, xq):
        return np.interp(xq, self.x, self.y)


class SEY:
    def __init__(self, sey_filepath):
        self.data = pd.read_csv(sey_filepath, sep=r'\s+', engine='python', header=None, names=["E", "sey"])
        self.Emax = max(self.data['E'])
        self.Emin = min(self.data['E'])
        # LINEAR table interpolation (as MultiPac treats secy files). The
        # previous CubicSpline oscillated to ~1e6 inside the huge gap between
        # the last dense data point (~1.9 keV) and the 1e12 eV sentinel row,
        # poisoning the recorded secondary yields (and hence e20/c0) for any
        # impact above ~1.9 keV. sey values are recorded diagnostics only --
        # they never feed back into the particle dynamics.
        self.sey = _TableInterp(self.data['E'], self.data['sey'])


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
