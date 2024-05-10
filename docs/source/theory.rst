Multipacting
=============

Multipacting is a phenomenon arising from the emission and subsequent multiplication of charged particles in
accelerating radiofrequency (RF) cavities, which can limit the achievable RF power. Predicting field levels at
which multipacting occurs is crucial for optimising cavity geometries. This paper presents a new open-source Python
code for analysing multipacting in 2D axisymmetric cavity structures.

The code leverages the NGSolve framework to solve the Maxwell Eigenvalue Problem (MEVP) to obtain the cavity's
resonant modes' electromagnetic (EM) fields. The relativistic Lorentz force equation governing the motion of charged
particles is then integrated using the fields within the cavity. Benchmarking against existing multipacting analysis
tools is performed to validate the code's accuracy and efficiency.

Maxwell Eigenvalue Problem
=========================
The MEVP introduced in \ref{} can be written in a variational form as follows

.. math::
    \begin{equation}
        \begin{array}{rl}
        \nabla \times \left(\nabla \times {\bf E}\right) - \lambda {\bf E}= 0 & {\bf E}\in \Omega,\nonumber \\
        \nabla \cdot {\bf E} = 0 & {\bf E}\in \Omega,\nonumber\\
        {\bf n} \times {\bf E} = 0 & {\bf E} \in \partial \Omega_\mathrm{1},\nonumber\\
        {\bf n} \times ({\bf {\mu^{-1}}\, \nabla \times {\bf E}}) = 0 & {\bf E} \in \partial \Omega_\mathrm{2},
        \end{array}
    \end{equation}
where :math:`\mathbf{E}` is the electric field, :math:`\lambda = (\omega/c)^2`, :math:`\partial \Omega :=
\partial \Omega_\mathrm{1}\cup \partial \Omega_\mathrm{2}` is the boundary of :math:`\Omega`.
A similar equation could also be written for the :math:`\mathbf{H}` field. The assumption already intrinsic in the
equation is that the domain :math:`\Omega` is vacuum.

In variational form, we expand the unknown field as the linear combination of several basis function of a finite element space (FES)

.. math::
    \begin{equation}
        \mathbf{E} = \sum_{i=1}^N e_i \mathbf{u}_i,
        \label{eqn: basis function expansion}
    \end{equation}
where :math:`\mathbf{u}_i` represents the basis functions and :math:`e_i` are the coeeficients which are to be calculated.
The natural basis functions for the :math:`\textbf{curl}\textbf{curl}` problem is the :math:`H(\operatorname{curl}, \Omega)`
functional space. Therefore, :math:`\mathbf{u}_i` here represents the edge or Nedelec basis functions.

Following the Galerkin method, we use a test function :math:`\mathbf{e}_j` as the basis function, integrate and
equate to zero to get the weak form of the equation.

.. math::
    \begin{equation}
        \int_{\Omega}\mathrm{curl} \mathbf{u}_i\cdot \mathrm{curl} \textbf{u}_j \text{d}V
        + \int_{\partial \Omega} (\textbf{u}_i \times ( \mathrm{curl} \textbf{u}_j)) \cdot  \textbf{n}\text{d}S
        = \int_{\Omega} \lambda\textbf{u}_i \cdot \textbf{u}_j \text{d}S
        \label{eqn: weak form}
    \end{equation}

which is the weak formulation of the original problem with :math:`\mathbf{e}\in H(\mathbf{\operatorname{curl}};\Omega)`
and :math:`\mathbf{e}_j\in H_0(\mathbf{\operatorname{curl}};\Omega)` as
Substituting (\ref{eqn: basis function expansion}) in (\ref{eqn: weak form}) and applying boundary
conditions (:math:`\int_{\partial \Omega_\mathrm{1}} \textbf{u}_i\cdot (\textbf{n} \times(  \operatorname{curl} \textbf{u}_j)) \text{d}S = 0`,
:math:`\int_{ \partial \Omega_\mathrm{2}} (\textbf{n} \times \textbf{e}_i) \cdot
( \operatorname{curl} \textbf{u}_j))\text{d}S = 0`), we get

.. math::
    \begin{equation}
        \left[\int_{\Omega}\operatorname{curl} \textbf{u}_i \cdot \operatorname{curl} \textbf{u}_j \text{d}V \right] e_i = \left[\int_{\Omega} \lambda\textbf{u}_i \cdot \textbf{u}_j \text{d}S\right] e_i
        \label{eqn: weak form2}
    \end{equation}
which can be written in matrix form as

.. math::
    \begin{equation}
        k_{ij} e_i = \lambda m_{ij} e_i
    \end{equation}

on each element where

.. math::
    \begin{equation*}
        \begin{array}{cc}
            k_{ij} = \int_{\Omega}(\operatorname{curl} \textbf{u}_i)\cdot ( \operatorname{curl} \textbf{u}_j) \text{d}V, & m_{ij} = \int_{\Omega} \textbf{u}_i \cdot \textbf{u}_j \text{d}S. \\
        \end{array}
    \end{equation*}

For the entire elements making up the domain, it is written as

.. math::
    \begin{equation}
        \textbf{K} \mathbf{e} = \lambda \textbf{M} \mathbf{e}
    \end{equation}

which is the generalised eigenvalue problem.



Relativistic Lorentz Force Integration
======================================
The geometry was also written in such a way that there are lot of surface points from which particles can be emitted.
To motion of charged particles in electromagnetic fields can be described using the Lorentz equation.
Since we are dealing with relativsitic :math:`\beta=1` or near relativistic :math:`\beta \approx 1`, we consider the
relativistic Lorentz force equation~\cite{yla1999multipacting}

.. math::
    \begin{equation}
        \begin{array}{l}
            \dfrac{\mathrm{d} \mathbf{v}}{\mathrm{d} t}=-\dfrac{q}{m}\left(1-\left(\dfrac{||\mathbf{v}||}{c}\right)^2\right)^{1 / 2}\left(\mathbf{E}+\mathbf{v} \times \mathbf{B}-\dfrac{1}{c^2}(\mathbf{v} \cdot \mathbf{E}) \mathbf{v}\right) \\
            \dfrac{\mathrm{d} \mathbf{r}}{\mathrm{d} t}=\mathbf{v}
        \end{array}
    \end{equation}

where :math:`q` and :math:`m` are the charge and mass of the charged particle (in this case, electron), respectively.
Different methods can be used to solve the initial value problem. Multiple step and multiple stage methods exist to solve the problem. The code includes the following multisteps and multi stage.

Integration scheme
==================
Classic Runge-Kutta
+++++++++++++++

Adam-Bashforth
++++++++++++++
