# `CSPEZ`

Chance-stochastic engagement-zone code built around Dubins-style pursuer kinematics, associated plotting utilities, path planners, and neural-network approximation experiments.

## Files

- `csbez.py`: core deterministic Dubins-style BEZ geometry and distance calculations.
- `cspez.py`: stochastic and approximate PEZ calculations built on `csbez.py`.
- `csbez_plotting.py`: plotting utilities for the deterministic Dubins-style BEZ model.
- `cspez_plotting.py`: plotting utilities for the stochastic PEZ approximations.
- `csbez_path_planner.py`: path planning with deterministic Dubins-style BEZ constraints.
- `cspez_path_planner.py`: path planning with stochastic PEZ constraints and surrogate-model support.
- `nueral_network_cspez.py`: neural-network training and surrogate-model helpers used by some CSPEZ workflows.
- `mlp.py`: Flax MLP definition used by the surrogate model code.

## Internal Dependencies

- `PEZ/`: baseline engagement-zone models and some shared plotting/planning patterns.
- `bspline/`: spline helpers used by the path planners.
- `PLOT_COMMON/`: shared figure helpers.

## Notes

- Public names retain the original research spelling, including `nueral_network_cspez.py`, so existing imports elsewhere in the repository continue to work.
