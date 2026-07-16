Multipacting
=============

Multipacting is a resonant phenomenon arising from the emission and subsequent
multiplication of charged particles in accelerating radiofrequency (RF)
cavities, which can limit the achievable RF power. An electron emitted from the
cavity wall is accelerated by the RF field, strikes the wall again and, if the
impact energy falls in the range where the surface's secondary emission yield
(SEY) exceeds unity and the field phase is synchronous, releases more than one
secondary electron. Repeating over many RF periods, this avalanche absorbs RF
power and can quench superconducting cavities. Predicting the field levels at
which multipacting occurs is therefore crucial when optimising cavity
geometries.

PyMultipact analyses multipacting in 2D axisymmetric structures in three
stages, following the workflow common to multipacting codes
[YlaOijala1999]_:

1. solve the Maxwell eigenvalue problem (MEVP) for the resonant modes of the
   structure (NGSolve finite element framework),
2. integrate the relativistic Lorentz force equation to track electrons
   emitted from the cavity wall through the modal fields,
3. detect and resolve wall collisions, applying the SEY data to compute
   multipacting metrics over a sweep of field levels.

Maxwell Eigenvalue Problem
==========================
For the solution domain :math:`\Omega`, the MEVP reads

.. math::
    \begin{equation}
        \begin{array}{rl}
        \nabla \times \left(\nabla \times {\bf E}\right) - \lambda {\bf E}= 0 & {\bf E}\in \Omega, \\
        \nabla \cdot {\bf E} = 0 & {\bf E}\in \Omega,\\
        {\bf n} \times {\bf E} = 0 & {\bf E} \in \partial \Omega_\mathrm{1},\\
        {\bf n} \times ({\bf {\mu^{-1}}\, \nabla \times {\bf E}}) = 0 & {\bf E} \in \partial \Omega_\mathrm{2},
        \end{array}
    \end{equation}

where :math:`\mathbf{n}` is the surface normal, :math:`\mathbf{E}` is the
electric field, :math:`\lambda = (\omega/c)^2` are the eigenvalues,
:math:`\omega` is the angular frequency, :math:`c` is the speed of light and
:math:`\partial \Omega := \partial \Omega_\mathrm{1}\cup \partial
\Omega_\mathrm{2}` is the boundary of :math:`\Omega`. A similar equation could
be written for the :math:`\mathbf{H}` field. The domain :math:`\Omega` is
assumed to be vacuum.

The unknown field is expanded in basis functions of a finite element space,

.. math::
    \begin{equation}
        \mathbf{E} = \sum_{i=1}^N e_i \mathbf{u}_i,
    \end{equation}

where the natural functional space for the
:math:`\operatorname{curl}\operatorname{curl}` problem is
:math:`H(\operatorname{curl}, \Omega)`; the :math:`\mathbf{u}_i` are the edge
(Nédélec) basis functions.

For an **axisymmetric** domain, the volume integrals of the weak (Galerkin)
form pick up the radial coordinate :math:`r` from the cylindrical volume
element :math:`\mathrm{d}V = r\,\mathrm{d}r\,\mathrm{d}z\,\mathrm{d}\varphi`,
so the weak form solved by PyMultipact on the 2D :math:`(z, r)` cross-section
is the *r-weighted* problem

.. math::
    \begin{equation}
        \left[\int_\Omega r\, \operatorname{curl}\mathbf{u} \cdot \operatorname{curl}\mathbf{v}\, \mathrm{d}V\right]\mathbf{e}
        = \lambda \left[\int_\Omega r\, \mathbf{u}\cdot\mathbf{v}\, \mathrm{d}V \right]\mathbf{e},
    \end{equation}

where :math:`\mathbf{v}` is the test function (equal to the basis functions).
In matrix form this is the generalised eigenvalue problem

.. math::
    \begin{equation}
        \mathbf{K}\,\mathbf{e} = \lambda\, \mathbf{M}\,\mathbf{e}.
    \end{equation}

PyMultipact assembles both forms with NGSolve and solves the eigenproblem with
the preconditioned inverse iteration (PINVIT) solver. Because the
:math:`\operatorname{curl}` operator has a large kernel (all gradient fields),
a divergence-free projector built from the discrete gradient matrix is applied
inside the preconditioner to suppress the spurious zero-frequency space and
keep the iteration on the physical modes.

Third-order Nédélec elements are used by default
(``Domain.compute_fields(order=3)``), matching the element order of the
MultiPac field solver [YlaOijala2001]_; this retains field accuracy on a
comparatively coarse mesh. The boundary polyline of dense geometry files is
resampled to ~250 points by default (``Domain(n_boundary_points=...)``) so the
mesh size follows the element order rather than the input file resolution.

.. note::
    The eigenmode solver currently supports only perfect magnetic conductor
    (PMC) boundary conditions on the left and right edges and the axisymmetry
    axis, with perfect electric conductor (PEC) boundary conditions elsewhere.
    Efforts are ongoing to provide more flexibility in specifying boundary
    conditions, including the addition of waveguide and open boundary
    conditions.

Relativistic Lorentz Force
==========================
The motion of charged particles in electromagnetic fields is described by the
Lorentz force equation. Since the emitted electrons are relativistic
(:math:`\beta = 1`) or near relativistic (:math:`\beta \approx 1`), the
relativistic form [YlaOijala1999]_ is integrated:

.. math::
    \begin{equation}
        \begin{array}{l}
            \dfrac{\mathrm{d} \mathbf{v}}{\mathrm{d} t}=-\dfrac{q}{m}\left(1-\left(\dfrac{||\mathbf{v}||}{c}\right)^2\right)^{1 / 2}\left(\mathbf{E}+\mathbf{v} \times \mathbf{B}-\dfrac{1}{c^2}(\mathbf{v} \cdot \mathbf{E}) \mathbf{v}\right), \\[2ex]
            \dfrac{\mathrm{d} \mathbf{x}}{\mathrm{d} t}=\mathbf{v},
        \end{array}
    \end{equation}

where :math:`q` and :math:`m` are the charge and mass of the electron.

The launch ensemble is the phase space of surface position and RF phase,

.. math::
    \begin{equation}
        X := \partial\Omega \times \Psi, \qquad
        \Psi = \{\psi : \psi \in [0, 2\pi]\},
    \end{equation}

i.e. each initial particle is a pair :math:`(\mathbf{x}_i, \psi_i)` of an
emission site on the wall and an initial field phase. Particles are released
with a small initial velocity :math:`v_0\,\mathbf{n}` along the inward surface
normal (:math:`v_0` corresponding to e.g. 2 eV) and tracked for a fixed number
of RF periods at every field level of the sweep.

Integration scheme
==================
Classic Runge-Kutta
+++++++++++++++++++

With the right-hand side

.. math::
    \mathbf{f}(\mathbf{v}, \mathbf{x}, t) = -\frac{q}{m} \left(1 - \left(\frac{\|\mathbf{v}\|}{c}\right)^2\right)^{1/2} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} - \frac{1}{c^2} (\mathbf{v} \cdot \mathbf{E}) \mathbf{v} \right),

the classical fourth-order Runge-Kutta scheme is used. Let the values at time
:math:`t_n` be :math:`\mathbf{v}_n`, :math:`\mathbf{x}_n` and the time step
:math:`h`:

.. math::
    \begin{aligned}
    \mathbf{k}_1^v &= h \cdot \mathbf{f}(\mathbf{v}_n, \mathbf{x}_n, t_n), &
    \mathbf{k}_1^x &= h \cdot \mathbf{v}_n,\\
    \mathbf{k}_2^v &= h \cdot \mathbf{f}\left(\mathbf{v}_n + \tfrac{\mathbf{k}_1^v}{2}, \mathbf{x}_n + \tfrac{\mathbf{k}_1^x}{2}, t_n + \tfrac{h}{2}\right), &
    \mathbf{k}_2^x &= h \cdot \left(\mathbf{v}_n + \tfrac{\mathbf{k}_1^v}{2}\right),\\
    \mathbf{k}_3^v &= h \cdot \mathbf{f}\left(\mathbf{v}_n + \tfrac{\mathbf{k}_2^v}{2}, \mathbf{x}_n + \tfrac{\mathbf{k}_2^x}{2}, t_n + \tfrac{h}{2}\right), &
    \mathbf{k}_3^x &= h \cdot \left(\mathbf{v}_n + \tfrac{\mathbf{k}_2^v}{2}\right),\\
    \mathbf{k}_4^v &= h \cdot \mathbf{f}(\mathbf{v}_n + \mathbf{k}_3^v, \mathbf{x}_n + \mathbf{k}_3^x, t_n + h), &
    \mathbf{k}_4^x &= h \cdot (\mathbf{v}_n + \mathbf{k}_3^v),
    \end{aligned}

.. math::
    \begin{aligned}
        \mathbf{v}_{n+1} &= \mathbf{v}_n + \frac{1}{6} (\mathbf{k}_1^v + 2\mathbf{k}_2^v + 2\mathbf{k}_3^v + \mathbf{k}_4^v), \\
        \mathbf{x}_{n+1} &= \mathbf{x}_n + \frac{1}{6} (\mathbf{k}_1^x + 2\mathbf{k}_2^x + 2\mathbf{k}_3^x + \mathbf{k}_4^x).
    \end{aligned}

If an intermediate stage carries a particle outside the domain, the stage is
resolved by the collision handler (below) before the update is completed.

Collision Detection and Resolution
==================================
Let

.. math::
    \begin{equation}
        L_k = \{(x, y) \mid x = x_k + t(x_{k+1} - x_k),\;
                          y = y_k + t(y_{k+1} - y_k),\; 0 \le t \le 1\}
    \end{equation}

be the line segment connecting the particle positions at time steps :math:`k`
and :math:`k+1`, and write the discretised boundary as
:math:`\partial\Omega := \bigcup_i \partial_i\Omega` of straight wall
segments. A collision occurs iff :math:`|L_k \cap \partial_i\Omega| > 0` and
odd. The candidate wall segments are found with a k-d tree over the boundary
points; the intersection point :math:`\mathbf{x}_\mathrm{c}` and the exact
impact time within the step are recovered from the crossing.

At the moment of impact the electric field at :math:`\mathbf{x}_\mathrm{c}` is
evaluated (from the FEM solution). If the field pulls electrons away from the
wall, i.e. :math:`\mathbf{n}\cdot\mathbf{E} > 0` with :math:`\mathbf{n}` the
inward normal, a secondary is released with the same initial speed
:math:`v_0\,\mathbf{n}` and tracking continues; otherwise the particle is
counted as absorbed (``loss_model='field'``, the default and the behaviour
benchmarked in the paper). Two alternative loss models can be selected in
``Domain.analyse_multipacting(loss_model=...)`` for numerical experiments:
``'wait'`` re-emits the electron uncounted until the RF phase turns favourable
(similar in spirit to delayed re-emission), and ``'always'`` re-emits and
counts every impact.

Multipacting Metrics
====================
The counter function (CF) and enhanced counter function (ECF) of the Helsinki
group [YlaOijala1999]_ are implemented.

**Counter function.** :math:`c_n/c_0` is the ratio of the number of particles
that survive at least :math:`n` impacts (by default :math:`n=20`) to the
number of particles released. It identifies surface sites and field levels
that geometrically support multipacting.

**Impact energy.** The kinetic energy at the :math:`k`-th impact is

.. math::
    \begin{equation}
        E_k = \frac{1}{q_0}\,(\gamma_k - 1)\, m_0 c^2,
    \end{equation}

with :math:`\gamma_k` the Lorentz factor at impact and :math:`q_0` the
elementary charge. The **final impact energy** :math:`Ef_{20}` reported by
``Domain.plot_Ef`` is the mean of the last (20th) impact energies of the
surviving particles; multipacting is dangerous where :math:`Ef_{20}` falls in
the region where the SEY exceeds unity (between the crossover energies of the
SEY curve, drawn as reference lines).

**Enhanced counter function.** The number of secondaries generated by one
particle after :math:`n` impacts is

.. math::
    \begin{equation}
        N_n := \prod_{k=1}^{n} \delta(\mathbf{x}_k, E_k),
    \end{equation}

where :math:`\delta` is the SEY of the surface (interpolated linearly from
tabulated data, e.g. measurements). For :math:`P` initial particles the ECF is

.. math::
    \begin{equation}
        e_n := \sum_{j=1}^{P} N_n^j.
    \end{equation}

:math:`e_{20}/c_0 > 1` indicates that the electron population actually grows,
i.e. genuinely dangerous multipacting rather than merely resonant trajectories.

**Distance function.** For each surviving trajectory the distance in
(position, phase) space between the final and initial state,

.. math::
    \begin{equation}
        d_{20} = \sqrt{\left\|\mathbf{x}_{20} - \mathbf{x}_0\right\|^2
        + \kappa \left| e^{j\varphi_{20}} - e^{j\varphi_0} \right|^2},
        \qquad \kappa = \frac{\lambda_\mathrm{RF}}{2\pi},
    \end{equation}

measures how exactly the electron returns to its starting point: small
:math:`d_{20}` identifies stable (fixed-point) multipacting orbits.
``Domain.plot_df`` renders it as a map over emission site and initial phase.

For two-point multipacting the orbit returns to the launch side only on every
*other* impact, so the plotted default compares the launch point with the
nearest of the last two impacts — this removes the arbitrary branch parity of
the impact count and reproduces MultiPac's graded map (dark core at the
resonant fixed-point phase, growing with the launch offset). The literal
20th-impact definition (``metric='d20_strict'``) and the closure of the
two-impact map (``metric='closure'``, zero for any phase-locked orbit) are
also available.

.. note::
    **Comparing counter functions with MultiPac.** By symmetry of the RF
    oscillation, exactly half of the launched :math:`(\mathbf{x}_i, \psi_i)`
    ensemble sees :math:`\mathbf{n}\cdot\mathbf{E} < 0` at emission and is
    absorbed on its first impacts. MultiPac's :math:`c_0` effectively counts
    only launchable electrons, so its counter function is a factor ~2 above
    PyMultipact's for the same physics.
    ``Domain.plot_cf(launchable_norm=True)`` divides by the launchable
    fraction (``Domain.launchable_fraction()``) to plot in MultiPac's
    convention.

References
==========
.. [YlaOijala1999] P. Ylä-Oijala, *Multipacting Analysis and Electromagnetic
   Field Computation by the Boundary Integral Equation Method in RF Cavities
   and Waveguides*, PhD thesis, Rolf Nevanlinna Institute, Helsinki, 1999.
.. [YlaOijala2001] P. Ylä-Oijala and D. Proch, "MultiPac - Multipacting
   Simulation Package with 2D FEM Field Solver", Proc. SRF'01, Tsukuba, 2001.
