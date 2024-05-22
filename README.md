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
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```

<html>
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
  
  <div style="flex: 1; text-align: center; background-color: #FFDAB9; padding: 10px; border-radius: 5px; margin: 5px;">
    
   <strong>Define Domain</strong>

  </div>

  <div style="flex: 1; text-align: center; background-color: #FFE4B5; padding: 10px; border-radius: 5px; margin: 5px;">
    <strong>Solve MEVP</strong>
  </div>

  <div style="flex: 1; text-align: center; background-color: #FFB6C1; padding: 10px; border-radius: 5px; margin: 5px;">
    <strong>Calculate Trajectory</strong>
  </div>

  <div style="flex: 1; text-align: center; background-color: #FFCCCB; padding: 10px; border-radius: 5px; margin: 5px;">
    <strong>Detect and Resolve Collision</strong>
  </div>

  <div style="flex: 1; text-align: center; background-color: #FFEFD5; padding: 10px; border-radius: 5px; margin: 5px;">
    <strong>Define Multipacting Metric</strong>
  </div>

</div>
</html>

```mermaid
graph LR
    A[Define Domain]:::defineDomain --> B[Solve MEVP]:::solveMevp
    B --> C[Calculate Trajectory]:::calculateTrajectory
    C --> D[Detect and Resolve Collision]:::detectResolveCollision
    D --> E[Define Multipacting Metric]:::defineMultipactingMetric

    classDef defineDomain fill:#FFDAB9,stroke:#333,stroke-width:2px;
    classDef solveMevp fill:#FFE4B5,stroke:#333,stroke-width:2px;
    classDef calculateTrajectory fill:#FFB6C1,stroke:#333,stroke-width:2px;
    classDef detectResolveCollision fill:#FFCCCB,stroke:#333,stroke-width:2px;
    classDef defineMultipactingMetric fill:#FFEFD5,stroke:#333,stroke-width:2px;
```