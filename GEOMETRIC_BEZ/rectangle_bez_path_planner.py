import numpy as np
import jax

from pyoptsparse import Optimization, OPT, IPOPT
from scipy.interpolate import BSpline
import time
from tqdm import tqdm
from jax import jacfwd
from jax import jit
from functools import partial
import jax.numpy as jnp
import getpass
import matplotlib.pyplot as plt
import matplotlib
import jax


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import bspline.spline_opt_tools as spline_opt_tools
import GEOMETRIC_BEZ.rectangle_bez as rectangle_bez

numSamplesPerInterval = 15


def plot_spline(spline, ax, width=1):
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, 1000, endpoint=True)
    ax.set_xlabel("X", fontsize=26)
    ax.set_ylabel("Y", fontsize=26)
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26)

    pos = spline(t)
    x = pos[:, 0]
    y = pos[:, 1]
    splineDot = spline.derivative()(t)
    xDot = splineDot[:, 0]
    yDot = splineDot[:, 1]
    agentHeadings = np.arctan2(yDot, xDot)

    pos = spline(t)
    # ez = in_dubins_engagement_zone(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerTurnRadius,
    #     pursuerCaptureRadius,
    #     pursuerRange,
    #     pursuerSpeed,
    #     pos,
    #     agentHeadings,
    #     agentSpeed,
    # )
    #
    ax.plot(x, y, linewidth=width)
    # cbar = plt.colorbar(c, shrink=0.8)
    # cbar.ax.tick_params(labelsize=26)

    ax.set_aspect(1)
    # c = plt.Circle(pursuerPosition, pursuerRange + pursuerCaptureRange, fill=False)
    # plt.scatter(pursuerPosition[0], pursuerPosition[1], c="r")
    # ax.add_artist(c)
    plt.xlabel("X")
    plt.ylabel("Y")


def create_spline(knotPoints, controlPoints, order):
    spline = BSpline(knotPoints, controlPoints, order)
    return spline


def rect_left_and_top(lower_left, upper_right, n_points_total):
    x1, y1 = lower_left
    x2, y2 = upper_right

    # Compute edge lengths
    left_len = y2 - y1
    top_len = x2 - x1
    total_len = left_len + top_len

    # Split points proportionally to edge lengths
    n_left = max(2, int(round(n_points_total * (left_len / total_len))))
    n_top = max(
        2, n_points_total - n_left + 1
    )  # +1 to avoid losing one point at the corner


def rect_left_and_top(lower_left, upper_right, n_points_total):
    x1, y1 = lower_left
    x2, y2 = upper_right

    # Compute edge lengths
    left_len = y2 - y1
    top_len = x2 - x1
    total_len = left_len + top_len

    # Split points proportionally to edge lengths
    n_left = max(2, int(round(n_points_total * (left_len / total_len))))
    n_top = max(
        2, n_points_total - n_left + 1
    )  # +1 to avoid losing one point at the corner

    # Left edge: (x = x1, y1 → y2)
    left = np.column_stack((np.full(n_left, x1), np.linspace(y1, y2, n_left)))

    # Top edge: (x1 → x2, y = y2)
    top = np.column_stack((np.linspace(x1, x2, n_top), np.full(n_top, y2)))

    # Concatenate, skipping the repeated corner (x1, y2)
    pts = np.vstack((left, top[1:]))
    return pts


def rect_bottom_and_right(lower_left, upper_right, n_points_total):
    x1, y1 = lower_left
    x2, y2 = upper_right

    # Compute edge lengths
    bottom_len = x2 - x1
    right_len = y2 - y1
    total_len = bottom_len + right_len

    # Split points proportionally to edge lengths
    n_bottom = max(2, int(round(n_points_total * (bottom_len / total_len))))
    n_right = max(
        2, n_points_total - n_bottom + 1
    )  # +1 to avoid losing one point at the corner

    # Bottom edge: (x1 → x2, y = y1)
    bottom = np.column_stack((np.linspace(x1, x2, n_bottom), np.full(n_bottom, y1)))

    # Right edge: (x = x2, y1 → y2)
    right = np.column_stack((np.full(n_right, x2), np.linspace(y1, y2, n_right)))

    # Concatenate, skipping the repeated corner (x2, y1)
    pts = np.vstack((bottom, right[1:]))
    return pts


@jax.jit
def box_BEZ_along_spline(
    controlPoints,
    tf,
    evaderSpeed,
    min_box,
    max_box,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
):
    numControlPoints = int(len(controlPoints) / 2)
    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, tf, numControlPoints, 3
    )
    evaderHeadings = spline_opt_tools.get_spline_heading(
        controlPoints, tf, 3, numSamplesPerInterval
    )
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )
    ez = rectangle_bez.box_pursuer_engagment_zone(
        pos,
        evaderHeadings,
        evaderSpeed,
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )
    return ez


@jax.jit
def compute_spline_constraints_for_box_BEZ(
    controlPoints,
    knotPoints,
    evaderSpeed,
    min_box,
    max_box,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
):
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )

    turn_rate, velocity, evaderHeadings = (
        spline_opt_tools.get_turn_rate_velocity_and_headings(
            controlPoints, knotPoints, numSamplesPerInterval
        )
    )

    curvature = turn_rate / velocity

    ez = rectangle_bez.box_pursuer_engagment_zone(
        pos,
        evaderHeadings,
        evaderSpeed,
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )

    return velocity, turn_rate, curvature, ez, pos


dBoxBEZDControlPoints = jax.jit(jacfwd(box_BEZ_along_spline, argnums=0))
dBoxBEZDtf = jax.jit(jacfwd(box_BEZ_along_spline, argnums=1))


def optimize_spline_path_box_BEZ(
    p0,
    pf,
    v0,
    num_cont_points,
    spline_order,
    velocity_constraints,
    turn_rate_constraints,
    curvature_constraints,
    num_constraint_samples,
    evaderSpeed,
    min_box,
    max_box,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    right=True,
    previous_spline=None,
):
    # Compute Jacobian of engagement zone function

    def objfunc(xDict):
        tf = xDict["tf"]
        knotPoints = spline_opt_tools.create_unclamped_knot_points(
            0, tf, num_cont_points, 3
        )
        controlPoints = xDict["control_points"]
        funcs = {}
        funcs["start"] = spline_opt_tools.get_start_constraint(controlPoints)
        funcs["end"] = spline_opt_tools.get_end_constraint(controlPoints)
        controlPoints = controlPoints.reshape((num_cont_points, 2))

        velocity, turn_rate, curvature, ez, pos = (
            compute_spline_constraints_for_box_BEZ(
                controlPoints,
                knotPoints,
                evaderSpeed,
                min_box,
                max_box,
                pursuerRange,
                pursuerCaptureRadius,
                pursuerSpeed,
            )
        )

        # funcs['start'] = self.get_start_constraint_jax(controlPoints)
        # funcs['start'] = pos[0]
        # funcs['end'] = pos[-1]
        funcs["obj"] = tf
        funcs["turn_rate"] = turn_rate
        funcs["velocity"] = velocity
        funcs["curvature"] = curvature
        # funcs['position'] = pos
        funcs["ez"] = ez
        funcs["obj"] = tf
        return funcs, False

    def sens(xDict, funcs):
        funcsSens = {}
        controlPoints = jnp.array(xDict["control_points"])
        tf = xDict["tf"]

        dStartDControlPointsVal = spline_opt_tools.get_start_constraint_jacobian(
            controlPoints
        )
        dEndDControlPointsVal = spline_opt_tools.get_end_constraint_jacobian(
            controlPoints
        )

        dEZDControlPoints = dBoxBEZDControlPoints(
            controlPoints,
            tf,
            evaderSpeed,
            min_box,
            max_box,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
        )
        dEZDtf = dBoxBEZDtf(
            controlPoints,
            tf,
            evaderSpeed,
            min_box,
            max_box,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
        )
        dVelocityDControlPointsVal = spline_opt_tools.dVelocityDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dVelocityDtfVal = spline_opt_tools.dVelocityDtf(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dTurnRateDControlPointsVal = spline_opt_tools.dTurnRateDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dTurnRateDtfVal = spline_opt_tools.dTurnRateTf(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dCurvatureDControlPointsVal = spline_opt_tools.dCurvatureDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dCurvatureDtfVal = spline_opt_tools.dCurvatureDtf(
            controlPoints, tf, 3, numSamplesPerInterval
        )

        funcsSens["obj"] = {
            "control_points": np.zeros((1, 2 * num_cont_points)),
            "tf": 1,
        }
        funcsSens["start"] = {
            "control_points": dStartDControlPointsVal,
            "tf": np.zeros((2, 1)),
        }
        funcsSens["end"] = {
            "control_points": dEndDControlPointsVal,
            "tf": np.zeros((2, 1)),
        }
        funcsSens["velocity"] = {
            "control_points": dVelocityDControlPointsVal,
            "tf": dVelocityDtfVal,
        }
        funcsSens["turn_rate"] = {
            "control_points": dTurnRateDControlPointsVal,
            "tf": dTurnRateDtfVal,
        }
        funcsSens["curvature"] = {
            "control_points": dCurvatureDControlPointsVal,
            "tf": dCurvatureDtfVal,
        }
        funcsSens["ez"] = {"control_points": dEZDControlPoints, "tf": dEZDtf}

        return funcsSens, False

    # num_constraint_samples = numSamplesPerInterval*(num_cont_points-2)-2

    optProb = Optimization("path optimization", objfunc)

    # if previous_spline is not None:
    if previous_spline is not None:
        x0 = previous_spline.c.flatten()
        tf = previous_spline.t[-1 - previous_spline.k]
    else:
        start = time.time()
        tf_initial = 1.0
        knotPoints = spline_opt_tools.create_unclamped_knot_points(
            0, tf_initial, num_cont_points, 3
        )

        # x0 = np.linspace(p0, pf, num_cont_points).flatten()
        if right:
            x0 = rect_bottom_and_right(p0, pf, num_cont_points).flatten()
        else:
            x0 = rect_left_and_top(p0, pf, num_cont_points).flatten()

        x0 = spline_opt_tools.move_first_control_point_so_spline_passes_through_start(
            x0, knotPoints, p0, v0
        )
        x0 = x0.flatten()
        x0 = spline_opt_tools.move_last_control_point_so_spline_passes_through_end(
            x0, knotPoints, pf, v0
        )
        x0 = x0.flatten()

        tf = spline_opt_tools.assure_velocity_constraint(
            x0,
            knotPoints,
            num_cont_points,
            evaderSpeed,
            velocity_constraints,
            numSamplesPerInterval,
        )

    tempVelocityContstraints = spline_opt_tools.get_spline_velocity(
        x0, 1, 3, numSamplesPerInterval
    )
    start = time.time()
    num_constraint_samples = len(tempVelocityContstraints)

    optProb.addVarGroup(
        name="control_points", nVars=2 * (num_cont_points), varType="c", value=x0
    )
    optProb.addVarGroup(name="tf", nVars=1, varType="c", value=tf, lower=0, upper=None)

    optProb.addConGroup(
        "velocity",
        num_constraint_samples,
        lower=velocity_constraints[0],
        upper=velocity_constraints[1],
        scale=1.0 / velocity_constraints[1],
    )
    optProb.addConGroup(
        "turn_rate",
        num_constraint_samples,
        lower=turn_rate_constraints[0],
        upper=turn_rate_constraints[1],
        scale=1.0 / turn_rate_constraints[1],
    )
    optProb.addConGroup(
        "curvature",
        num_constraint_samples,
        lower=curvature_constraints[0],
        upper=curvature_constraints[1],
        scale=1.0 / curvature_constraints[1],
    )
    optProb.addConGroup("ez", num_constraint_samples, lower=0.00, upper=None)
    optProb.addConGroup("start", 2, lower=p0, upper=p0)
    optProb.addConGroup("end", 2, lower=pf, upper=pf)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    opt.options["print_level"] = 0
    opt.options["max_iter"] = 1000
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    # opt.options["derivative_test"] = "first-order"
    # opt.options["warm_start_init_point"] = "yes"
    # opt.options["mu_init"] = 1e-1
    # opt.options["nlp_scaling_method"] = "gradient-based"
    opt.options["tol"] = 1e-8

    sol = opt(optProb, sens=sens)

    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, sol.xStar["tf"][0], num_cont_points, 3
    )

    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))

    print("Optimization time:", time.time() - start)
    return (
        create_spline(knotPoints, controlPoints, spline_order),
        sol.xStar["tf"][0],
    )


def plan_path_box_BEZ(
    min_box,
    max_box,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    initialEvaderPosition,
    finalEvaderPosition,
    initialEvaderVelocity,
    evaderSpeed,
    num_cont_points,
    spline_order,
    velocity_constraints,
    turn_rate_constraints,
    curvature_constraints,
    num_constraint_samples,
):
    (splineRight, tfRight) = optimize_spline_path_box_BEZ(
        initialEvaderPosition,
        finalEvaderPosition,
        initialEvaderVelocity,
        num_cont_points=num_cont_points,
        spline_order=spline_order,
        velocity_constraints=velocity_constraints,
        turn_rate_constraints=turn_rate_constraints,
        curvature_constraints=curvature_constraints,
        num_constraint_samples=num_constraint_samples,
        evaderSpeed=evaderSpeed,
        min_box=min_box,
        max_box=max_box,
        pursuerRange=pursuerRange,
        pursuerCaptureRadius=pursuerCaptureRadius,
        pursuerSpeed=pursuerSpeed,
        right=True,
        previous_spline=None,
    )
    splineLeft, tfLeft = optimize_spline_path_box_BEZ(
        initialEvaderPosition,
        finalEvaderPosition,
        initialEvaderVelocity,
        num_cont_points=num_cont_points,
        spline_order=spline_order,
        velocity_constraints=velocity_constraints,
        turn_rate_constraints=turn_rate_constraints,
        curvature_constraints=curvature_constraints,
        num_constraint_samples=num_constraint_samples,
        evaderSpeed=evaderSpeed,
        min_box=min_box,
        max_box=max_box,
        pursuerRange=pursuerRange,
        pursuerCaptureRadius=pursuerCaptureRadius,
        pursuerSpeed=pursuerSpeed,
        right=False,
        previous_spline=None,
    )
    print("Right path time", tfRight)
    print("Left path time", tfLeft)
    if tfRight < tfLeft:
        spline = splineRight
        tf = tfRight
    else:
        spline = splineLeft
        tf = tfLeft
    print("path time", tf)
    return spline


def main_box():
    initialEvaderPosition = np.array([-5.0, -5.0])
    finalEvaderPosition = np.array([5.0, 5.0])
    initialEvaderVelocity = np.array([1.0, 0.0])
    pursuerRange = 1.5
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.0
    evaderSpeed = 1.5
    min_box = np.array([-3.0, -3.0])
    max_box = np.array([3.0, 3.0])

    num_cont_points = 20
    spline_order = 3
    velocity_constraints = (0.9, 1.1)
    curvature_constraints = (-0.5, 0.5)
    turn_rate_constraints = (-1.0, 1.0)
    num_constraint_samples = 50

    spline = plan_path_box_BEZ(
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        initialEvaderPosition,
        finalEvaderPosition,
        initialEvaderVelocity,
        evaderSpeed,
        num_cont_points,
        spline_order,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_spline(spline, ax)
    rectangle_bez.plot_box_pursuer_reachable_region(
        min_box, max_box, pursuerRange, pursuerCaptureRadius, ax=ax
    )
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    # plt.legend()


if __name__ == "__main__":
    main_box()
    plt.show()
