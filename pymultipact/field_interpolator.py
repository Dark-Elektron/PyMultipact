"""
Field access objects for PyMultipact.

Two implementations share one interface so they are interchangeable in the
integrator::

    .e(points)        -> (N, 2) complex   bulk E field; RAISES if any point is
                                          outside the domain (this drives the
                                          exception-based boundary detection)
    .h(points)        -> (N, 1) complex   bulk H field (same raise behaviour)
    .e_exact(point)   -> (1, 2) complex   exact FEM E at a single point
    .h_exact(point)   -> (1, 1) complex   exact FEM H at a single point
    .is_inside(points)-> (N,)  bool       true-polygon inside test

- EMFieldDirect evaluates the FEM solution directly everywhere (accurate,
  matches the paper; slow because ngsolve must locate every point in the
  unstructured mesh).
- FastFieldCache precomputes E/H on a regular grid once and does O(1) bilinear
  lookups for the bulk force. The field query itself is ~13x faster than exact
  FEM. HOWEVER it is NOT viable for multipacting and is kept only for reference:
  the surviving electrons live in a sub-millimetre layer at the wall and bounce
  ~20 times, so they need (a) a boundary criterion that matches the *mesh*
  exactly -- the polygon is slightly larger than the meshed region, so a grid
  raise lets particles penetrate the wall and be lost (wrong counter function),
  and (b) near-exact near-wall fields -- small grid errors compound over the
  bounces and throw the impact energies off by several fold. Using an exact
  mesh-based raise fixes the counter function but removes the speed-up (mesh
  point-location is back) and the energies are still wrong. In short: a global
  grid cannot cheaply resolve the near-surface physics; accurate fields here
  require ngsolve's mesh point-location (which is the real cost, and why a
  compiled code like Multipac is faster -- it is largely a language, not an
  algorithm, gap for the field query).
"""

import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from matplotlib.path import Path


class _OutsideDomain(Exception):
    """Raised when a bulk field lookup hits a point outside the domain; caught
    by the integrator to trigger boundary handling (mirrors ngsolve raising
    'Meshpoint not in mesh')."""


class EMFieldDirect:
    """Direct FEM point evaluation (accurate, slow). Interface as above."""

    def __init__(self, gfu_E, gfu_H, mesh, boundary, bounding_rect=None,
                 resolution=None):
        self.gfu_E = gfu_E
        self.gfu_H = gfu_H
        self.mesh = mesh
        self._path = Path(np.asarray(boundary))

    def e(self, points_zr):
        pts = np.atleast_2d(points_zr)
        ev = np.asarray(self.gfu_E(self.mesh(pts[:, 0], pts[:, 1])), dtype=complex)
        if ev.ndim == 1:
            ev = ev.reshape(1, -1)
        return ev

    def h(self, points_zr):
        pts = np.atleast_2d(points_zr)
        hv = np.asarray(self.gfu_H(self.mesh(pts[:, 0], pts[:, 1])),
                        dtype=complex).ravel()
        return hv[:, np.newaxis]

    def eh(self, points_zr):
        pts = np.atleast_2d(points_zr)
        mps = self.mesh(pts[:, 0], pts[:, 1])
        ev = np.asarray(self.gfu_E(mps), dtype=complex)
        if ev.ndim == 1:
            ev = ev.reshape(1, -1)
        hv = np.asarray(self.gfu_H(mps), dtype=complex).ravel()
        return ev, hv[:, np.newaxis]

    e_exact = e
    h_exact = h

    def is_inside(self, points_zr):
        return self._path.contains_points(np.atleast_2d(points_zr))


class FastFieldCache:
    """Structured-grid field lookup for the bulk force + exact FEM at impacts."""

    def __init__(self, gfu_E, gfu_H, mesh, boundary, bounding_rect,
                 resolution=500):
        t0 = time.time()
        self.gfu_E = gfu_E
        self.gfu_H = gfu_H
        self.mesh = mesh
        self._path = Path(np.asarray(boundary))

        zmin, zmax, rmin, rmax = bounding_rect
        self._z = np.linspace(zmin, zmax, resolution)
        self._r = np.linspace(rmin, rmax, resolution)
        Z, R = np.meshgrid(self._z, self._r, indexing='ij')
        pts = np.column_stack([Z.ravel(), R.ravel()])
        n_total = len(pts)

        inside = self._path.contains_points(pts)
        idx_in = np.nonzero(inside)[0]

        E_flat = np.zeros((n_total, 2), dtype=complex)
        H_flat = np.zeros(n_total, dtype=complex)
        ok = np.zeros(n_total, dtype=bool)

        # Exact FEM at inside grid nodes, batched with a per-point fallback for
        # nodes that are inside the polygon but outside the mesh.
        batch = 4000
        for s in range(0, len(idx_in), batch):
            bidx = idx_in[s:s + batch]
            bpts = pts[bidx]
            try:
                mps = mesh(bpts[:, 0], bpts[:, 1])
                ev = np.asarray(gfu_E(mps), dtype=complex)
                hv = np.asarray(gfu_H(mps), dtype=complex)
                if ev.ndim == 1:
                    ev = ev.reshape(1, -1)
                E_flat[bidx] = ev
                H_flat[bidx] = hv.ravel()
                ok[bidx] = True
            except Exception:
                for j, gi in enumerate(bidx):
                    try:
                        mp = mesh(float(bpts[j, 0]), float(bpts[j, 1]))
                        E_flat[gi] = np.asarray(gfu_E(mp), dtype=complex).ravel()[:2]
                        H_flat[gi] = np.asarray(gfu_H(mp), dtype=complex).ravel()[0]
                        ok[gi] = True
                    except Exception:
                        pass

        # Fill every un-evaluated node (outside domain, or mesh miss) with the
        # nearest valid node's value so bilinear interpolation next to the wall
        # never collapses toward zero.
        good = np.nonzero(ok)[0]
        bad = np.nonzero(~ok)[0]
        if len(bad) and len(good):
            tree = cKDTree(pts[good])
            _, nn = tree.query(pts[bad], k=1)
            E_flat[bad] = E_flat[good[nn]]
            H_flat[bad] = H_flat[good[nn]]

        # Grids for hand-rolled bilinear interpolation (pure numpy, no scipy
        # object dispatch -- that dispatch, not the O(1) lookup, was what made
        # RegularGridInterpolator no faster than ngsolve's mesh search).
        self._res = resolution
        self._z0, self._r0 = self._z[0], self._r[0]
        self._dz = self._z[1] - self._z[0]
        self._dr = self._r[1] - self._r[0]
        self._E = E_flat.reshape(resolution, resolution, 2)
        self._H = H_flat.reshape(resolution, resolution)
        ig = inside.reshape(resolution, resolution)
        self._inside_grid = ig
        # Per-cell classification for a fast + exact inside test: 0 exterior
        # (all 4 corners outside), 2 interior (all 4 inside), 1 boundary (mixed).
        corners = (ig[:-1, :-1].astype(np.int8) + ig[1:, :-1] + ig[:-1, 1:]
                   + ig[1:, 1:])
        self._cell_type = np.where(corners == 0, 0,
                                   np.where(corners == 4, 2, 1)).astype(np.int8)
        self._build_s = time.time() - t0

    def _outside_mask(self, x):
        """Fast + exact: interior/exterior cells resolved by O(1) lookup; only
        points in boundary-straddling cells fall back to the polygon test."""
        z = x[:, 0]; r = x[:, 1]
        i = np.clip(((z - self._z0) / self._dz).astype(np.intp), 0, self._res - 2)
        j = np.clip(((r - self._r0) / self._dr).astype(np.intp), 0, self._res - 2)
        ct = self._cell_type[i, j]
        outside = (ct == 0)
        bnd = (ct == 1)
        if bnd.any():
            outside[bnd] = ~self._path.contains_points(x[bnd])
        return outside

    def _bilinear(self, x):
        """Vectorised bilinear interpolation; returns (E (N,2), H (N,), outside
        (N,) bool). outside is true where the nearest grid node is outside the
        domain (drives boundary detection)."""
        z = x[:, 0]; r = x[:, 1]
        fi = (z - self._z0) / self._dz
        fj = (r - self._r0) / self._dr
        i = np.clip(fi.astype(np.intp), 0, self._res - 2)
        j = np.clip(fj.astype(np.intp), 0, self._res - 2)
        tz = (fi - i)[:, None]
        tr = (fj - j)[:, None]
        E = self._E
        e00 = E[i, j]; e10 = E[i + 1, j]; e01 = E[i, j + 1]; e11 = E[i + 1, j + 1]
        Ev = ((1 - tz) * (1 - tr) * e00 + tz * (1 - tr) * e10
              + (1 - tz) * tr * e01 + tz * tr * e11)
        H = self._H
        h00 = H[i, j]; h10 = H[i + 1, j]; h01 = H[i, j + 1]; h11 = H[i + 1, j + 1]
        tz0 = tz[:, 0]; tr0 = tr[:, 0]
        Hv = ((1 - tz0) * (1 - tr0) * h00 + tz0 * (1 - tr0) * h10
              + (1 - tz0) * tr0 * h01 + tz0 * tr0 * h11)
        # nearest-node inside flag
        ii = np.clip(np.round(fi).astype(np.intp), 0, self._res - 1)
        jj = np.clip(np.round(fj).astype(np.intp), 0, self._res - 1)
        outside = ~self._inside_grid[ii, jj]
        return Ev, Hv, outside

    # ---- bulk lookups (raise if any point outside the domain) ----
    def eh(self, points_zr):
        """Combined E,H lookup with a single inside test (used by the force)."""
        pts = np.atleast_2d(points_zr)
        if self._outside_mask(pts).any():
            raise _OutsideDomain()
        Ev, Hv, _ = self._bilinear(pts)
        return Ev, Hv[:, np.newaxis]

    def e(self, points_zr):
        return self.eh(points_zr)[0]

    def h(self, points_zr):
        return self.eh(points_zr)[1]

    # ---- exact FEM (impact energy, boundary field) ----
    def e_exact(self, points_zr):
        pts = np.atleast_2d(points_zr)
        ev = np.asarray(self.gfu_E(self.mesh(pts[:, 0], pts[:, 1])), dtype=complex)
        if ev.ndim == 1:
            ev = ev.reshape(1, -1)
        return ev

    def h_exact(self, points_zr):
        pts = np.atleast_2d(points_zr)
        hv = np.asarray(self.gfu_H(self.mesh(pts[:, 0], pts[:, 1])),
                        dtype=complex).ravel()
        return hv[:, np.newaxis]

    def is_inside(self, points_zr):
        return self._path.contains_points(np.atleast_2d(points_zr))
