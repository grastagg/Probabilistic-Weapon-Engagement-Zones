# DMC

This folder contains deterministic maneuvering-constraint models and related
path-planning experiments built on top of the engagement-zone formulations in
`PEZ/`.

Core modules:

- `dmc.py` implements the analytic deterministic maneuvering constraint (DMC)
  and helper plotting routines for a single pursuer.
- `rect_dmc.py` adapts the DMC idea to rectangular pursuer launch sets and
  includes solver and visualization helpers for the box-based construction.
- `dmc_path_planner.py` couples the DMC constraint to B-spline trajectory
  optimization.

Supporting files:

- `pdmc.py` is currently empty and appears to be a placeholder for a future
  probabilistic DMC extension.
