# GEOMETRIC_BEZ

This folder contains geometric constructions of reachable regions and engagement
zones built from interception-point constraints and simple pursuer geometry.

Core modules:

- `bez_from_interceptions.py` builds potential BEZ/reachable-region boundaries
  from circle-intersection arcs and includes visualization utilities.
- `rectangle_bez.py` defines rectangular reachable-region and engagement-zone
  approximations together with the plotting helpers used in the dissertation
  figures.
- `pez_from_interceptions.py` contains the corresponding probabilistic
  engagement-zone construction.

Planner and experiment scripts:

- `*_path_planner.py` files couple the geometric region models to trajectory
  optimization or path-planning experiments.
- `monte_carlo_runner.py` and `run_sim_par.sh` support batch simulation runs.
- `beta_distribution_plot.py` and `sacraficial_planner.py` contain supporting
  analysis and figure-generation scripts.

Most files are written as research scripts, so plotting helpers and executable
examples often live in the same module as the core geometry functions.
