# Probabilistic Weapon Engagement Zones

This repository contains the code used across multiple parts of my dissertation research on probabilistic, geometric, and learned weapon engagement zones. It is organized as a collection of related study folders rather than as a single installable package, so the easiest way to work with it is to run scripts from the repository root.

## Repository Layout

- `PEZ/`: baseline probabilistic engagement-zone models, plotting utilities, and path planners.
- `CSPEZ/`: chance-stochastic PEZ variants, path planners, plotting, and neural-network surrogate work.
- `GEOMETRIC_BEZ/`: geometric BEZ/PEZ formulations, rectangle approximations, and Monte Carlo studies.
- `DMC/`: DMC experiments and associated path-planning scripts.
- `OUQ_BEZ/`: uncertainty-quantification studies, grid-based planning, and supporting data files.
- `CSBEZ_LEARNING/`: learned Dubins-style BEZ experiments and planners.
- `VARIABLE_SPEED_EZ/`: variable-speed engagement-zone experiments.
- `bspline/`: shared B-spline evaluation and optimization helpers used by several planners.
- `PLOT_COMMON/`: shared plotting helpers such as airplane glyphs, Mahalanobis overlays, and annotations.
- `video/`, `video2/`: generated frame outputs and other visual artifacts.

## Environment

There is not yet a locked environment file in the repo. Based on the current imports, the codebase depends primarily on:

- Core scientific stack: `numpy`, `scipy`, `matplotlib`, `jax`, `tqdm`
- Optimization: `pyoptsparse`
- Folder-specific extras: `chaospy`, `flax`, `optax`, `scikit-learn`, `seaborn`, `h5py`, `pyDOE3`

Recommended setup:

1. Create a dedicated Python environment.
2. Install the dependencies needed for the folder you care about.
3. Run scripts from the repository root so package-style imports such as `PEZ.pez` and `bspline.spline_opt_tools` resolve correctly.

## Shared Utilities

- [`bspline/README.md`](bspline/README.md)
- [`PLOT_COMMON/README.md`](PLOT_COMMON/README.md)
