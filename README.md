# PyMultipact

Multipacting is a phenomenon arising from the emission and subsequent multiplication of charged 
particles in accelerating radiofrequency (RF) cavities, which can limit the achievable RF power. 
Predicting field levels at which multipacting occurs is crucial for optimising cavity geometries. 
This paper presents an open-source Python code (PyMultipact) for analysing multipacting 
in 2D axisymmetric cavity structures. The code leverages the NGSolve framework to solve the 
Maxwell eigenvalue problem (MEVP) for the electromagnetic (EM) fields in axisymmetric RF structures.
The relativistic Lorentz force equation governing the motion of charged particles is then integrated 
using the calculated fields within the domain to describe the motion of charged particles. 
Benchmarking against existing multipacting analysis tools is performed to validate the code's accuracy.

# Workflow

```mermaid
graph LR
    A[Define Domain]:::defineDomain --> B[Solve MEVP]:::solveMevp --> C[Calculate Trajectory]:::calculateTrajectory
    C --> D[Detect and Resolve Collision]:::detectResolveCollision
    D --> E[Define Multipacting Metric]:::defineMultipactingMetric

    classDef defineDomain fill:#FFDAB9,stroke:#333,stroke-width:2px;
    classDef solveMevp fill:#FFE4B5,stroke:#333,stroke-width:2px;
    classDef calculateTrajectory fill:#FFB6C1,stroke:#333,stroke-width:2px;
    classDef detectResolveCollision fill:#FFCCCB,stroke:#333,stroke-width:2px;
    classDef defineMultipactingMetric fill:#FFEFD5,stroke:#333,stroke-width:2px;
```

Certainly! Here’s a refined version:

The workflow begins by defining the domain using `geometry_writer.py`. 
Next, the Maxwell eigenvalue problem (MEVP) is solved with the NGSolve finite element method (FEM) 
framework via `domain.compute_field`. Collision detection and handling are performed in Python. 
The multipacting metrics currently defined are the counter and enhanced counter functions. 
To analyse the domain for multipacting, `domain.analyse_multipacting` is used.