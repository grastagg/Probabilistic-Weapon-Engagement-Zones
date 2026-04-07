# OUQ_BEZ

This folder contains scripts for optimal uncertainty quantification (OUQ)
approximations to engagement-zone and path-planning problems, along with the
precomputed grids and parameter files those scripts use.

Core scripts:

- `OUQ_range_only.py` models range-only pursuer uncertainty and compares the
  resulting OUQ safety contours and paths.
- `OUQ_pez_pursuer_position.py` studies OUQ bounds for uncertain pursuer
  position inside a rectangular support set and derives inner-rectangle path
  approximations.
- `plan_path_from_grid.py` loads precomputed OUQ/PEZ grids, interpolates them
  over `(x, y, psi)`, and uses those interpolants inside spline path planning.


