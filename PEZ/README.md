# `PEZ`

Baseline deterministic and probabilistic engagement-zone code used by several other folders in the repository.

## Files

- `bez.py`: deterministic BEZ boundary functions in NumPy and JAX form.
- `pez.py`: probabilistic engagement-zone calculations built on the deterministic BEZ model.
- `pez_plotting.py`: plotting helpers and standalone visualization scripts for BEZ and PEZ figures.
- `bez_path_planner.py`: path planning with deterministic BEZ constraints.
- `pez_path_planner.py`: path planning with probabilistic engagement-zone constraints.
- `bez_surface_plot.py`: standalone script for visualizing the local BEZ surface and tangent plane.

## Internal Dependencies

- `bspline/`: spline evaluation and optimization helpers used by the path planners.
- `PLOT_COMMON/`: shared figure helpers used by plotting and planner visualizations.

## Notes

- Public function names retain the original research spelling so existing imports and callers in the rest of the repository continue to work unchanged.
- Several files can be run directly as scripts for figure generation or experiments.
