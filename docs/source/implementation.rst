Implementation
==============

Workflow
--------
A multipacting analysis proceeds through three objects:

.. code-block:: python

    from pymultipact.domain import Project, Domain

    proj = Project()
    proj.create_project('TESLA')

    domain = Domain(proj)          # geometry + mesh
    domain.compute_fields()        # Maxwell eigenvalue problem
    domain.analyse_multipacting()  # particle tracking sweep (parallel)

    domain.plot_cf(launchable_norm=True)   # counter function
    domain.plot_Ef()                       # final impact energy + SEY lines
    domain.plot_ef()                       # enhanced counter function
    domain.plot_df(epk_i=42)               # d20 map at one field level

Geometry and mesh
-----------------
The cavity contour is a polyline read from a ``.n`` geometry file (or written
by ``geometry_writer`` for parametrised elliptical cavities). Because netgen
anchors a mesh node to every polyline point, dense contours would force a
needlessly fine surface mesh: the contour is therefore resampled to
``Domain(n_boundary_points=250)`` points by arc-length decimation. Only
original profile points are kept (none are invented), and the endpoints plus
the extreme-coordinate points that define the bounding box always survive.
``n_boundary_points=None`` keeps the file's full resolution; the original
TESLA mid-cell contour is preserved in ``sample_domains/tesla_mid_cell_fine.n``.

MEVP solver
-----------
``Domain.compute_fields(order=3)`` assembles the r-weighted curl-curl and mass
forms (see :doc:`theory`) on an :math:`H(\operatorname{curl})` space of
third-order Nédélec elements and solves the generalised eigenproblem with
NGSolve's preconditioned inverse iteration (PINVIT), wrapped in a
divergence-free projector built from the discrete gradient to suppress the
gradient kernel. The magnetic field follows from Faraday's law,
:math:`\mathbf{H} = \frac{j}{\mu_0\,\omega}\nabla\times\mathbf{E}`.

Particle tracking
-----------------
``Integrators.rk4`` advances all particles simultaneously (vectorised over the
population) with the classical RK4 scheme. Field values along trajectories are
exact FEM point evaluations — a structured-grid field cache was investigated
and rejected because surviving electrons live in a sub-millimetre layer at the
wall where interpolation errors compound over the ~20 impacts.

Boundary handling is exception-driven: evaluating the field at a point outside
the meshed domain raises, which triggers the collision handler for the
affected sub-step. A particle that nevertheless escapes (e.g. re-emitted
marginally outside) is dropped as lost by a guard at the next step instead of
aborting the run.

Collision detection
-------------------
For every particle within one light-step of the wall, the segment between its
previous and current position is intersected with candidate wall segments
reconstructed from the ``HIT_NEIGHBOURS`` (default 100) nearest surface points,
found with a k-d tree over the mesh boundary vertices. On intersection, the
impact time, surface normal and impact energy are computed and the particle is
either re-emitted or absorbed according to the ``loss_model``
(:doc:`theory`, *Collision Detection and Resolution*).

When a particle reaches 20 counted impacts it is archived to the *bright set*
together with its full impact history (energies, secondary yields, initial
site and phase), from which all multipacting metrics are computed.

Parallel field-level sweep
--------------------------
The Epk sweep is embarrassingly parallel and ``analyse_multipacting`` runs it
on multiple processes by default (``proc_count=None`` picks
``min(cpu_count - 1, n_epks)``; ``proc_count=1`` runs in-process). Workers
receive the mesh and modal fields via pickle and return per-field-level
results that are reassembled in sweep order. On Windows, scripts (not
notebooks) must call it under ``if __name__ == '__main__':``.

Testing
-------
The test suite lives in ``tests/`` (``pytest``; full tracking regressions are
marked ``slow``):

.. code-block:: bash

    python -m pytest -q            # everything (~2 min)
    python -m pytest -q -m "not slow"   # fast unit tests only (~2 s)
