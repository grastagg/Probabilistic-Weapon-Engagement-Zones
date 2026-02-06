from __future__ import annotations
import sys
import os

import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

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


def animate_true_pursuer(pursuerPosition, pursuerRange, pursuerCaptureRadius, ax):
    ax.scatter(
        *pursuerPosition, color="red", s=50, label="True Pursuer Position", marker="o"
    )
    # plot reachable region
    circle = plt.Circle(
        pursuerPosition, pursuerRange + pursuerCaptureRadius, color="red", alpha=0.2
    )
    ax.add_artist(circle)

    ax.set_xticks([])
    ax.set_yticks([])


def _triangle_vertices(xy, heading, size):
    """
    Return 3x2 vertices for a triangle centered at xy, pointing along heading.
    Triangle shape: one tip forward, two rear corners.
    """
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])

    # local (body) coordinates: tip, rear-left, rear-right
    # (tweak these ratios if you want a "longer" or "wider" triangle)
    tri_local = (
        np.array(
            [
                [1.2, 0.0],
                [-1.0, 0.6],
                [-1.0, -0.6],
            ]
        )
        * size
    )

    return xy + tri_local @ R.T


def animate_sacraficial_trajectory_frames(
    truePursuerPos,
    spline,
    isIntercepted,
    interceptPoint,
    interceptedTime,
    frameNum,
    frameRate,  # frames per second
    cfg,
    interceptionPositions,
    interceptionRadii,
    interceptHeading=None,  # radians; if None, use spline heading at intercept time
    out_dir="video",
    triangle_size=None,  # if None, auto from plot scale
    stop_at_intercept=True,  # True: stop frames at intercept; False: continue along spline
    hold_frames=10,  # if stop_at_intercept, optionally hold N extra frames
):
    """
    Writes frames: out_dir/{frameNum:06d}.png, returns updated frameNum.

    spline: scipy.interpolate.BSpline or compatible callable with:
      - spline(t) -> (2,) for scalar t, or (N,2) for array t
      - spline.derivative(1)(t) available (recommended)
    """
    os.makedirs(out_dir, exist_ok=True)

    # BSpline "valid" time range
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]

    # frame times
    dt = 1.0 / float(frameRate)
    t_vals = np.arange(t0, tf + 1e-12, dt)

    # precompute full path for background
    path = spline(t_vals)  # (N,2)

    # precompute heading from derivative if available, else finite-diff
    try:
        dspline = spline.derivative(1)
        vel = dspline(t_vals)  # (N,2)
    except Exception:
        vel = np.gradient(path, dt, axis=0)

    headings = np.arctan2(vel[:, 1], vel[:, 0])

    # default triangle size from plot scale
    if triangle_size is None:
        xr = cfg["x_range"][1] - cfg["x_range"][0]
        yr = cfg["y_range"][1] - cfg["y_range"][0]
        triangle_size = 0.02 * min(xr, yr)

    # if interceptHeading not provided, use spline heading at intercept time
    if interceptHeading is None and isIntercepted:
        # clamp to valid range
        t_int = float(np.clip(interceptedTime, t0, tf))
        try:
            v_int = dspline(t_int)
        except Exception:
            # finite diff around t_int
            eps = 1e-3 * (tf - t0)
            p1 = spline(max(t0, t_int - eps))
            p2 = spline(min(tf, t_int + eps))
            v_int = (p2 - p1) / (2 * eps)
        interceptHeading = float(np.arctan2(v_int[1], v_int[0]))

    # set up one figure we reuse for speed
    fig, ax = plt.subplots()

    # optional: if you want consistent styling across frames
    ax.set_aspect("equal")
    ax.set_xlim(cfg["x_range"])
    ax.set_ylim(cfg["y_range"])
    # turn off axis labels

    # figure out the intercept frame index (if any)
    if isIntercepted:
        intercept_idx = int(np.searchsorted(t_vals, interceptedTime, side="left"))
        intercept_idx = int(np.clip(intercept_idx, 0, len(t_vals) - 1))
    else:
        intercept_idx = None

    last_idx = len(t_vals) - 1
    end_idx = last_idx

    if isIntercepted and stop_at_intercept:
        end_idx = intercept_idx

    for i in range(0, end_idx + 1):
        ax.cla()
        animate_true_pursuer(
            truePursuerPos, cfg["pursuerRange"], cfg["pursuerCaptureRadius"], ax
        )
        if len(interceptionPositions) > 0:
            arcs = sacraficial_planner.bez_from_interceptions.compute_potential_pursuer_region_from_interception_position(
                # np.array(interceptionPositions[0:-1]),
                np.array(interceptionPositions),
                cfg["pursuerRange"],
                cfg["pursuerCaptureRadius"],
            )

            ax.set_aspect("equal")
            sacraficial_planner.bez_from_interceptions.plot_potential_pursuer_reachable_region(
                arcs,
                cfg["pursuerRange"],
                cfg["pursuerCaptureRadius"],
                xlim=cfg["x_range"],
                ylim=cfg["y_range"],
                ax=ax,
            )
            sacraficial_planner.bez_from_interceptions.plot_circle_intersection_arcs(
                arcs, ax=ax
            )
            sacraficial_planner.bez_from_interceptions.plot_interception_points(
                np.array(interceptionPositions), np.array(interceptionRadii), ax=ax
            )
        else:
            sacraficial_planner.rectangle_bez.plot_box_pursuer_reachable_region(
                np.array(cfg["min_box"]),
                np.array(cfg["max_box"]),
                cfg["pursuerRange"],
                cfg["pursuerCaptureRadius"],
                ax=ax,
            )

        # background: full path (always)
        ax.plot(path[:, 0], path[:, 1], label="Sacrificial Agent Path")

        # history up to now (optional, looks nice)
        ax.plot(path[: i + 1, 0], path[: i + 1, 1], linewidth=2)

        # current pose
        if isIntercepted and (i >= intercept_idx):
            cur_xy = np.asarray(interceptPoint, dtype=float)
            cur_heading = float(interceptHeading)
        else:
            cur_xy = path[i]
            cur_heading = float(headings[i])

        tri = _triangle_vertices(cur_xy, cur_heading, triangle_size)
        ax.fill(tri[:, 0], tri[:, 1], alpha=0.9, label="Agent")

        # intercept marker (draw it once it happens, or always if you prefer)
        if isIntercepted and i >= intercept_idx:
            ax.scatter(
                interceptPoint[0],
                interceptPoint[1],
                s=60,
                marker="x",
                label="Intercept Point",
            )

        ax.set_aspect("equal")
        ax.set_xlim(cfg["x_range"])
        ax.set_ylim(cfg["y_range"])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Sacrificial Agent {1 + len(interceptionPositions)}")
        fig.savefig(os.path.join(out_dir, f"{frameNum}.png"))
        frameNum += 1

    # optionally hold on the intercept pose for a few extra frames
    # if isIntercepted and stop_at_intercept and hold_frames > 0:
    #     ax.cla()
    #     ax.plot(
    #         path[:intercept_idx, 0],
    #         path[:intercept_idx, 1],
    #         label="Sacrificial Agent Path",
    #     )
    #     cur_xy = np.asarray(interceptPoint, dtype=float)
    #     cur_heading = float(interceptHeading)
    #     tri = _triangle_vertices(cur_xy, cur_heading, triangle_size)
    #     ax.fill(tri[:, 0], tri[:, 1], alpha=0.9, label="Agent")
    #     ax.scatter(
    #         interceptPoint[0],
    #         interceptPoint[1],
    #         s=60,
    #         marker="x",
    #         label="Intercept Point",
    #     )
    #     ax.set_aspect("equal")
    #     ax.set_xlim(cfg["x_range"])
    #     ax.set_ylim(cfg["y_range"])
    #
    #     for _ in range(int(hold_frames)):
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         fig.savefig(os.path.join(out_dir, f"{frameNum}.png"))
    #         frameNum += 1

    plt.close(fig)
    return frameNum


def animate_hp_path(
    truePursuerPos,
    splineHP,
    interceptionPositions,
    interceptionRadii,
    frameNum,
    numFramesForHp,
    cfg,
):
    fig, ax = plt.subplots()
    animate_true_pursuer(
        truePursuerPos, cfg["pursuerRange"], cfg["pursuerCaptureRadius"], ax
    )
    ax.set_aspect("equal")
    ax.set_xlim(cfg["x_range"])
    ax.set_ylim(cfg["y_range"])
    if len(interceptionPositions) > 0:
        arcs = sacraficial_planner.bez_from_interceptions.compute_potential_pursuer_region_from_interception_position(
            # np.array(interceptionPositions[0:-1]),
            np.array(interceptionPositions),
            cfg["pursuerRange"],
            cfg["pursuerCaptureRadius"],
        )

        ax.set_aspect("equal")
        sacraficial_planner.bez_from_interceptions.plot_potential_pursuer_reachable_region(
            arcs,
            cfg["pursuerRange"],
            cfg["pursuerCaptureRadius"],
            xlim=cfg["x_range"],
            ylim=cfg["y_range"],
            ax=ax,
        )
        sacraficial_planner.bez_from_interceptions.plot_circle_intersection_arcs(
            arcs, ax=ax
        )
        sacraficial_planner.bez_from_interceptions.plot_interception_points(
            np.array(interceptionPositions), np.array(interceptionRadii), ax=ax
        )
    else:
        sacraficial_planner.rectangle_bez.plot_box_pursuer_reachable_region(
            np.array(cfg["min_box"]),
            np.array(cfg["max_box"]),
            cfg["pursuerRange"],
            cfg["pursuerCaptureRadius"],
            ax=ax,
        )
    # remove x and y label
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")

    for i in range(numFramesForHp):
        fig.savefig(f"video/{frameNum}.png")
        frameNum += 1

    sacraficial_planner.plot_spline(splineHP, ax, width=2, color="green")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(f"Safe Path")
    for i in range(numFramesForHp):
        fig.savefig(f"video/{frameNum}.png")
        frameNum += 1
    return frameNum


def run_monte_carlo_simulation(
    randomSeed=0,
    numAgents=5,
    saveData=True,
    dataDir="GEOMETRIC_BEZ/data/test/",
    runName="test",
    plot=False,
    planHighPriorityPaths=True,
    animate=False,
    measureLaunchTime=False,
    straightLineSacrificial=False,
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
        "animate": bool(animate),
        "planHighPriorityPaths": bool(planHighPriorityPaths),
        "measureLaunchTime": bool(measureLaunchTime),
        "straightLineSacrificial": bool(straightLineSacrificial),
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
        "p_min": 0.0,
    }
    cfg["D_min"] = cfg["D_min_frac"] * cfg["pursuerRange"]

    if cfg["measureLaunchTime"]:
        tmpRunName = "launchTime"
    else:
        tmpRunName = "noLaunchTime"
    if cfg["straightLineSacrificial"]:
        tmpRunName2 = "straightLine/"
    else:
        tmpRunName2 = "splinePath/"
    run_dir = Path(cfg["dataDir"]) / cfg["runName"] / tmpRunName / tmpRunName2

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
    print(f"True pursuer position: {truePursuerPos}")

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

    if cfg["animate"]:
        frameNum = 0
        numFramesForHp = 20
        frameRate = 5
        frameNum = animate_hp_path(
            truePursuerPos,
            splineHP,
            interceptionPositions,
            interceptionRadii,
            frameNum,
            numFramesForHp,
            cfg,
        )

    # -----------------------------
    # Main loop
    # -----------------------------
    for agentIdx in range(cfg["numAgents"]):
        if not interceptionPositions:
            t0 = time.time()
            if cfg["straightLineSacrificial"]:
                targetPoint = np.array([max_box[0], max_box[1]])
                if agentIdx == 1:
                    targetPoint = np.array([(min_box[0] + max_box[0]) / 2, max_box[1]])
                if agentIdx == 2:
                    targetPoint = np.array([max_box[0], (min_box[1] + max_box[1]) / 2])
                spline = sacraficial_planner.straight_line_spline(
                    cfg["sacrificialLaunchPosition"], targetPoint
                )
            else:
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
                np.array(interceptionRadii),
                dArea,
            )
            print("launch pdf sum:", jnp.sum(launchPdf) * dArea)

            expected_launch_pos, _ = sacraficial_planner.expected_position_from_pdf(
                points, launchPdf, dArea
            )
            print("expected launch pos:", expected_launch_pos)
            aspectAngle = np.arctan2(
                cfg["sacrificialLaunchPosition"][1] - expected_launch_pos[1],
                cfg["sacrificialLaunchPosition"][0] - expected_launch_pos[0],
            )
            endPointInitialGuess = (
                expected_launch_pos
                - np.array([np.cos(aspectAngle), np.sin(aspectAngle)])
                * sacrificialSpeed
                * 1.0
            )

            t0 = time.time()
            if cfg["straightLineSacrificial"]:
                spline = sacraficial_planner.straight_line_spline(
                    cfg["sacrificialLaunchPosition"], endPointInitialGuess
                )
            else:
                spline = sacraficial_planner.optimize_spline_path_minimize_area(
                    sacrificialLaunchPosition,
                    endPointInitialGuess,
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
                    pmin=cfg["p_min"],
                    pursuerSpeed=cfg["pursuerSpeed"],
                    expectedPursuerPos=expected_launch_pos,
                    measureLaunchTime=cfg["measureLaunchTime"],
                )
                print(
                    f"time for agent {agentIdx} optimization: {time.time() - t0:.3f}s"
                )

        isIntercepted, interceptedTime, interceptPoint, D, tavelTime = (
            sacraficial_planner.sample_intercept_from_spline(
                spline,
                truePursuerPos,
                cfg["pursuerRange"],
                cfg["pursuerCaptureRadius"],
                cfg["pursuerSpeed"],
                inefficacyRatio=1.05,
                alpha=cfg["alpha"],
                beta=cfg["beta"],
                D_min=cfg["D_min"],
                rng=rng,
            )
        )
        print("travel time:", tavelTime)
        if cfg["animate"]:
            frameNum = animate_sacraficial_trajectory_frames(
                truePursuerPos,
                spline,
                isIntercepted,
                interceptPoint,
                interceptedTime,
                frameNum,
                frameRate,
                cfg,
                interceptionPositions,
                interceptionRadii,
                out_dir="video",
                stop_at_intercept=True,
                hold_frames=10,
            )

        interceptionHistory.append(bool(isIntercepted))

        if isIntercepted:
            print(
                f"Agent {agentIdx} intercepted at {interceptPoint} (D={float(D):.3f})"
            )
            interceptionPositions.append(np.array(interceptPoint))

            if cfg["measureLaunchTime"]:
                radius = tavelTime * cfg["pursuerSpeed"] + cfg["pursuerCaptureRadius"]
            else:
                radius = cfg["pursuerRange"] + cfg["pursuerCaptureRadius"]
            print("radius:", radius)
            print("test radius:", cfg["pursuerRange"] + cfg["pursuerCaptureRadius"])
            interceptionRadii.append(radius)
            # interceptionRadii.append(cfg["pursuerRange"] + cfg["pursuerCaptureRadius"])

        if cfg["planHighPriorityPaths"]:
            if len(interceptionPositions) == 0:
                splineHP, tf = (
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
            else:
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
            if cfg["animate"]:
                frameNum = animate_hp_path(
                    truePursuerPos,
                    splineHP,
                    interceptionPositions,
                    interceptionRadii,
                    frameNum,
                    numFramesForHp,
                    cfg,
                )
            #     frameNum = animate_hp_path()

        if cfg["plot"]:
            fig, ax = plt.subplots()
            t0 = spline.t[spline.k]
            tf = spline.t[-1 - spline.k]
            t = np.linspace(t0, tf, 1000, endpoint=True)
            idx = -1
            # idx = np.where(t <= interceptedTime)[0][-1] + 1 if isIntercepted else -1
            pos = spline(t)[0:idx]
            ax.plot(pos[:, 0], pos[:, 1], label=f"Sacraficial Agent {agentIdx} Path")

            t0 = splineHP.t[splineHP.k]
            tf = splineHP.t[-1 - splineHP.k]
            t = np.linspace(t0, tf, 1000, endpoint=True)

            posHP = splineHP(t)
            ax.plot(posHP[:, 0], posHP[:, 1], label="High-Priority Agent Path")

            ax.set_aspect("equal")

            if len(interceptionPositions) > 0:
                arcs = sacraficial_planner.bez_from_interceptions.compute_potential_pursuer_region_from_interception_position(
                    # np.array(interceptionPositions[0:-1]),
                    np.array(interceptionPositions),
                    cfg["pursuerRange"],
                    cfg["pursuerCaptureRadius"],
                )

                ax.set_aspect("equal")
                sacraficial_planner.bez_from_interceptions.plot_potential_pursuer_reachable_region(
                    arcs,
                    cfg["pursuerRange"],
                    cfg["pursuerCaptureRadius"],
                    xlim=cfg["x_range"],
                    ylim=cfg["y_range"],
                    ax=ax,
                )
                sacraficial_planner.bez_from_interceptions.plot_circle_intersection_arcs(
                    arcs, ax=ax
                )
            else:
                sacraficial_planner.rectangle_bez.plot_box_pursuer_reachable_region(
                    min_box,
                    max_box,
                    cfg["pursuerRange"],
                    cfg["pursuerCaptureRadius"],
                    ax=ax,
                )
            ax.scatter(
                *truePursuerPos,
                color="red",
                s=50,
                label="True Pursuer Position",
                marker="o",
            )
            if isIntercepted:
                ax.scatter(
                    *interceptPoint,
                    color="blue",
                    s=50,
                    label="Intercept Point",
                    marker="x",
                )
                for i, pos in enumerate(interceptionPositions[0:-1]):
                    ax.scatter(
                        *pos,
                        color="red",
                        s=50,
                        label=f"Past Interception {i}",
                        marker="x",
                    )
            else:
                for i, pos in enumerate(interceptionPositions):
                    ax.scatter(
                        *pos,
                        color="red",
                        s=50,
                        label=f"Past Interception {i}",
                        marker="x",
                    )
            plt.show()

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
    if len(sys.argv) != 4:
        print(
            "usage: python monte_carlo_runner.py <random_seed> <measure_launch_time> <straight_line_sacrificial>"
        )
    else:
        seed = int(sys.argv[1])
        measure_launch_time = bool(int(sys.argv[2]))
        straight_line_sacrificial = bool(int(sys.argv[3]))
        print("running monte carlo simulation with seed", seed)
        numAgents = 5
        runName = "beta82"
        run_monte_carlo_simulation(
            seed,
            numAgents,
            saveData=True,
            dataDir="GEOMETRIC_BEZ/data/",
            runName=runName,
            plot=True,
            animate=False,
            measureLaunchTime=measure_launch_time,
            straightLineSacrificial=straight_line_sacrificial,
        )
