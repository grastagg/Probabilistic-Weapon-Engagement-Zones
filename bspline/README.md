# `bspline`

Shared B-spline helpers used by the path-planning code in `PEZ/`, `CSPEZ/`, `GEOMETRIC_BEZ/`, `DMC/`, and `OUQ_BEZ/`.

## Files

- `helper_functions.py`: shape, knot, and control-point utility functions.
- `matrix_evaluation.py`: low-level matrix-based spline and derivative evaluation.
- `spline_opt_tools.py`: planner-facing helpers for headings, curvature, turn rate, and endpoint constraints.

## Conventions

- Low-level spline evaluation functions generally expect control points in shape `(dimension, num_control_points)`.
- Several planner-facing helpers flatten 2D control points to a single vector of shape `(2 * num_control_points,)` before reshaping internally.
- Most planning code assumes 2D trajectories.
