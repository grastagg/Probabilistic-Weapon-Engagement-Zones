# CSBEZ_LEARNING

This folder contains experiments on learning and planning with Dubins-style
chance-stochastic bounded engagement zones (CSBEZs).

Core files:

- `learning_dubins_ez.py` contains the main simulation, estimation, and
  visualization routines for learning Dubins engagement-zone parameters from
  interception data.
- `learned_dubins_ez_path_planner.py` uses the learned Dubins-style engagement
  zone model inside B-spline path planning.

Supporting scripts:

- `run_sim.sh` and `run_sim_par.sh` are batch-run helpers for launching
  simulation experiments.

This folder is more experiment-oriented than package-oriented: the main learning
script contains data generation, plotting, and optimization logic in one place,
while the planner file focuses on trajectory optimization using the learned
model.
