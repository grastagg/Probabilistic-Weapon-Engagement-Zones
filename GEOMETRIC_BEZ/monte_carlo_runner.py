from __future__ import annotations
import sys

import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import jax.numpy as jnp

from GEOMETRIC_BEZ import sacraficial_planner


def _jsonable(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return x


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)


def run_monte_carlo_simulation(
    randomSeed=0,
    numAgents=5,
    saveData=True,
    dataDir="GEOMETRIC_BEZ/data/test/",
    runName="test",
    plot=False,
    planHighPriorityPaths=True,
):
    # -----------------------------
    # Config (everything saved)
    # -----------------------------
    cfg = {
        "randomSeed": int(randomSeed),
        "numAgents": int(numAgents),
        "saveData": bool(saveData),
        "dataDir": str(dataDir),
        "runName": str(runName),
        "plot": bool(plot),
        "planHighPriorityPaths": bool(planHighPriorityPaths),
        "x_range": [-6.0, 6.0],
        "y_range": [-6.0, 6.0],
        "num_pts": 50,
        "pursuerRange": 1.0,
        "pursuerCaptureRadius": 0.2,
        "pursuerSpeed": 1.5,
        "min_box": [-2.0, -2.0],
        "max_box": [2.0, 2.0],
        "sacrificialLaunchPosition": [-5.0, -5.0],
        "sacrificialSpeed": 1.0,
        "sacrificialRange": 25.0,
        "highPriorityGoal": [5.0, 5.0],
        "highPrioritySpeed": 1.0,
        "num_cont_points": 14,
        "spline_order": 3,
        "R_min": 0.5,
        "alpha": 8.0,
        "beta": 2.0,
        "D_min_frac": 0.5,
    }
    cfg["D_min"] = cfg["D_min_frac"] * cfg["pursuerRange"]

    run_dir = Path(cfg["dataDir"]) / cfg["runName"]
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "config.json",
        {
            "meta": {
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")
            },
            **cfg,
        },
    )

    rng = np.random.default_rng(cfg["randomSeed"])

    # -----------------------------
    # Grid
    # -----------------------------
    x = jnp.linspace(cfg["x_range"][0], cfg["x_range"][1], cfg["num_pts"])
    y = jnp.linspace(cfg["y_range"][0], cfg["y_range"][1], cfg["num_pts"])
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    points = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    dx = (cfg["x_range"][1] - cfg["x_range"][0]) / (cfg["num_pts"] - 1)
    dy = (cfg["y_range"][1] - cfg["y_range"][0]) / (cfg["num_pts"] - 1)
    dArea = float(dx * dy)

    min_box = np.array(cfg["min_box"], dtype=float)
    max_box = np.array(cfg["max_box"], dtype=float)

    # -----------------------------
    # Initial states
    # -----------------------------
    sacrificialLaunchPosition = np.array(cfg["sacrificialLaunchPosition"], dtype=float)
    sacrificialSpeed = float(cfg["sacrificialSpeed"])

    highPriorityStart = sacrificialLaunchPosition.copy()
    highPriorityGoal = np.array(cfg["highPriorityGoal"], dtype=float)

    initialHighPriorityVel = np.array([1.0, 1.0]) / np.sqrt(2)
    initialSacrificialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)

    R_min = float(cfg["R_min"])
    v = sacrificialSpeed
    velocity_constraints = (0.01, v)
    curvature_constraints = (-1.0 / R_min, 1.0 / R_min)
    turn_rate_constraints = (-v / R_min, v / R_min)

    truePursuerPos = np.array(
        [rng.uniform(min_box[0], max_box[0]), rng.uniform(min_box[1], max_box[1])]
    )

    # -----------------------------
    # Storage
    # -----------------------------
    interceptionPositions = []
    interceptionRadii = []
    interceptionHistory = []

    potentialPursuerRegionAreas = [
        float((max_box[0] - min_box[0]) * (max_box[1] - min_box[1]))
    ]
    highPriorityPathTimes = []

    # -----------------------------
    # Initial HP plan
    # -----------------------------
    splineHP = None
    if cfg["planHighPriorityPaths"]:
        splineHP, tfHP = (
            sacraficial_planner.rectangle_bez_path_planner.plan_path_box_BEZ(
                min_box,
                max_box,
                cfg["pursuerRange"],
                cfg["pursuerCaptureRadius"],
                cfg["pursuerSpeed"],
                highPriorityStart,
                highPriorityGoal,
                initialHighPriorityVel,
                cfg["highPrioritySpeed"],
                cfg["num_cont_points"],
                cfg["spline_order"],
                velocity_constraints,
                turn_rate_constraints,
                curvature_constraints,
            )
        )
        highPriorityPathTimes.append(tfHP)

    # -----------------------------
    # Main loop
    # -----------------------------
    for agentIdx in range(cfg["numAgents"]):
        if not interceptionPositions:
            t0 = time.time()
            spline = sacraficial_planner.optimize_spline_path_get_intercepted(
                sacrificialLaunchPosition,
                np.array([0.0, 0.0]),
                initialSacrificialVelocity,
                cfg["num_cont_points"],
                cfg["spline_order"],
                velocity_constraints,
                turn_rate_constraints,
                curvature_constraints,
                sacrificialSpeed,
                cfg["pursuerRange"],
                cfg["pursuerCaptureRadius"],
                min_box,
                max_box,
                dArea,
                sacraficialAgentRange=cfg["sacrificialRange"],
            )
            print(f"First agent optimization time: {time.time() - t0:.3f}s")
        else:
            launchPdf = sacraficial_planner.pez_from_interceptions.uniform_pdf_from_interception_points(
                points,
                np.array(interceptionPositions),
                cfg["pursuerRange"],
                cfg["pursuerCaptureRadius"],
                dArea,
            )

            expected_launch_pos, _ = sacraficial_planner.expected_position_from_pdf(
                points, launchPdf, dArea
            )

            t0 = time.time()
            spline = sacraficial_planner.optimize_spline_path_minimize_area(
                sacrificialLaunchPosition,
                np.array(expected_launch_pos),
                initialSacrificialVelocity,
                cfg["num_cont_points"],
                cfg["spline_order"],
                velocity_constraints,
                turn_rate_constraints,
                curvature_constraints,
                sacrificialSpeed,
                cfg["pursuerRange"],
                cfg["pursuerCaptureRadius"],
                np.array(interceptionPositions),
                np.array(interceptionRadii),
                dArea,
                points,
                launchPdf,
                sacraficialAgentRange=cfg["sacrificialRange"],
            )
            print(f"time for agent {agentIdx} optimization: {time.time() - t0:.3f}s")

        isIntercepted, idx, interceptPoint, D = (
            sacraficial_planner.sample_intercept_from_spline(
                spline,
                truePursuerPos,
                cfg["pursuerRange"],
                cfg["alpha"],
                cfg["beta"],
                D_min=cfg["D_min"],
                rng=rng,
            )
        )

        interceptionHistory.append(bool(isIntercepted))

        if isIntercepted:
            print(
                f"Agent {agentIdx} intercepted at {interceptPoint} (D={float(D):.3f})"
            )
            interceptionPositions.append(np.array(interceptPoint))
            interceptionRadii.append(cfg["pursuerRange"] + cfg["pursuerCaptureRadius"])

        if cfg["planHighPriorityPaths"]:
            splineHP, arcs, tf = (
                sacraficial_planner.bez_from_interceptions_path_planner.plan_path_from_interception_points(
                    interceptionPositions,
                    cfg["pursuerRange"],
                    cfg["pursuerCaptureRadius"],
                    cfg["pursuerSpeed"],
                    highPriorityStart,
                    highPriorityGoal,
                    initialHighPriorityVel,
                    cfg["highPrioritySpeed"],
                    cfg["num_cont_points"],
                    cfg["spline_order"],
                    velocity_constraints,
                    turn_rate_constraints,
                    curvature_constraints,
                )
            )
            highPriorityPathTimes.append(tf)

        if cfg["saveData"]:
            if interceptionPositions:
                intersectionArea = sacraficial_planner.circle_intersection_area(
                    np.array(interceptionPositions),
                    np.array(interceptionRadii),
                )
            else:
                intersectionArea = potentialPursuerRegionAreas[0]
            potentialPursuerRegionAreas.append(float(intersectionArea))

    # -----------------------------
    # Save results
    # -----------------------------
    if cfg["saveData"]:
        data = {
            "truePursuerPos": truePursuerPos,
            "interceptionPositions": np.array(interceptionPositions),
            "interceptionRadii": np.array(interceptionRadii),
            "potentialPursuerRegionAreas": np.array(potentialPursuerRegionAreas),
            "highPriorityPathTimes": np.array(highPriorityPathTimes),
            "interceptionHistory": np.array(interceptionHistory),
        }

        filename = run_dir / f"{cfg['randomSeed']}.npz"
        np.savez_compressed(str(filename), **data)
        print(f"Saved simulation data to {filename}")


if __name__ == "__main__":
    # main()
    # first argument is random seed from command line
    if len(sys.argv) != 2:
        print("usage: python monte_carlo_runner.py <random_seed>")
    else:
        seed = int(sys.argv[1])
        print("running monte carlo simulation with seed", seed)
        numAgents = 5
        runName = "test"
        run_monte_carlo_simulation(
            seed,
            numAgents,
            saveData=True,
            dataDir="GEOMETRIC_BEZ/data/",
            runName=runName,
        )
