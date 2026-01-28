import numpy as np
import jax

from pyoptsparse import Optimization, OPT, IPOPT
from scipy.interpolate import BSpline, _bsplines
import time
from tqdm import tqdm
from jax import jacfwd, jacobian
from jax import jit
from functools import partial
import jax.numpy as jnp
import getpass
import matplotlib.pyplot as plt
import matplotlib
import jax


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from GEOMETRIC_BEZ import pez_from_interceptions
from GEOMETRIC_BEZ import bez_from_interceptions_path_planner
import bspline.spline_opt_tools as spline_opt_tools

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
    ax.plot(x, y, linewidth=width)

    ax.set_aspect(1)
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
def PEZ_from_interceptions_along_spline(
    controlPoints,
    tf,
    evaderSpeed,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    launch_pdf,
    integrationPoints,
    dArea,
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
    # ez = rectangle_pez.prob_reachable_uniform_box(
    #     pos,
    #     pursuerRange,
    #     pursuerCaptureRadius,
    #     min_box,
    #     max_box,
    # )

    pez = pez_from_interceptions.pez_numerical_soft(
        pos,
        evaderHeadings,
        evaderSpeed,
        pursuerSpeed,
        integrationPoints,
        launch_pdf,
        pursuerRange,
        pursuerCaptureRadius,
        dArea,
    )

    return pez


@jax.jit
def compute_spline_constraints_for_PEZ_from_interceptions(
    controlPoints,
    knotPoints,
    evaderSpeed,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    launch_pdf,
    integrationPoints,
    dArea,
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

    pez = pez_from_interceptions.pez_numerical_soft(
        pos,
        evaderHeadings,
        evaderSpeed,
        pursuerSpeed,
        integrationPoints,
        launch_pdf,
        pursuerRange,
        pursuerCaptureRadius,
        dArea,
    )

    return velocity, turn_rate, curvature, pez, pos, evaderHeadings


dPEZDControlPoints = jax.jit(jax.jacfwd(PEZ_from_interceptions_along_spline, argnums=0))
dPEZDtf = jax.jit(jax.jacfwd(PEZ_from_interceptions_along_spline, argnums=1))


def optimize_spline_path_PEZ_from_interceptions(
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
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    launch_pdf,
    integrationPoints,
    dArea,
    right=True,
    previous_spline=None,
    pez_limit=0.1,
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

        velocity, turn_rate, curvature, ez, pos, evaderHeading = (
            compute_spline_constraints_for_PEZ_from_interceptions(
                controlPoints,
                knotPoints,
                evaderSpeed,
                pursuerRange,
                pursuerCaptureRadius,
                pursuerSpeed,
                launch_pdf,
                integrationPoints,
                dArea,
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
        funcs["pos"] = pos
        funcs["heading"] = evaderHeading
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

        dEZDControlPoints = dPEZDControlPoints(
            controlPoints,
            tf,
            evaderSpeed,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
            launch_pdf,
            integrationPoints,
            dArea,
        )
        dEZDtf = dPEZDtf(
            controlPoints,
            tf,
            evaderSpeed,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
            launch_pdf,
            integrationPoints,
            dArea,
        )
        # replace nans with zeros
        # dEZDtf = np.array(dEZDtf, copy=True)
        # dEZDtf[np.isnan(dEZDtf)] = 0.0
        # dEZDControlPoints = np.array(dEZDControlPoints, copy=True)
        # dEZDControlPoints[np.isnan(dEZDControlPoints)] = 0.0
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

        # x0 = spline_opt_tools.move_first_control_point_so_spline_passes_through_start(
        #     x0, knotPoints, p0, v0
        # )
        # x0 = x0.flatten()
        # x0 = spline_opt_tools.move_last_control_point_so_spline_passes_through_end(
        #     x0, knotPoints, pf, v0
        # )
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
    optProb.addConGroup("ez", num_constraint_samples, lower=None, upper=pez_limit)
    optProb.addConGroup("start", 2, lower=p0, upper=p0)
    optProb.addConGroup("end", 2, lower=pf, upper=pf)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 150
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    opt.options["derivative_test"] = "first-order"
    # opt.options["warm_start_init_point"] = "yes"
    # opt.options["mu_init"] = 1e-1
    # opt.options["nlp_scaling_method"] = "gradient-based"
    opt.options["tol"] = 1e-6

    # sol = opt(optProb, sens="FD")
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


def plan_path_PEZ_from_interceptions(
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    interceptionPositions,
    radii,
    xlim,
    ylim,
    numPoints,
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
    pez_limit,
):
    integrationPoints, X, Y = (
        pez_from_interceptions.bez_from_interceptions.get_meshgrid_points(
            xlim, ylim, numPoints
        )
    )

    dArea = (
        (xlim[1] - xlim[0]) / (numPoints - 1) * (ylim[1] - ylim[0]) / (numPoints - 1)
    )

    launch_pdf = pez_from_interceptions.uniform_pdf_from_interception_points(
        integrationPoints,
        interceptionPositions,
        radii,
        dArea,
    )
    splineHP, arcs, tf = (
        bez_from_interceptions_path_planner.plan_path_from_interception_points(
            interceptionPositions,
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
        )
    )
    splineLeft, tfLeft = optimize_spline_path_PEZ_from_interceptions(
        initialEvaderPosition,
        finalEvaderPosition,
        initialEvaderVelocity,
        num_cont_points,
        spline_order,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        evaderSpeed,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        launch_pdf,
        integrationPoints,
        dArea,
        right=False,
        previous_spline=splineHP,
        pez_limit=pez_limit,
    )
    return splineLeft, tfLeft
    # (splineRight, tfRight) = optimize_spline_path_PEZ_from_interceptions(
    #     initialEvaderPosition,
    #     finalEvaderPosition,
    #     initialEvaderVelocity,
    #     num_cont_points,
    #     spline_order,
    #     velocity_constraints,
    #     turn_rate_constraints,
    #     curvature_constraints,
    #     num_constraint_samples,
    #     evaderSpeed,
    #     pursuerRange,
    #     pursuerCaptureRadius,
    #     pursuerSpeed,
    #     launch_pdf,
    #     integrationPoints,
    #     dArea,
    #     right=True,
    #     previous_spline=None,
    #     pez_limit=pez_limit,
    # )
    print("Right path time", tfRight)
    print("Left path time", tfLeft)
    if tfRight < tfLeft:
        spline = splineRight
        tf = tfRight
    else:
        spline = splineLeft
        tf = tfLeft
    print("path time", tf)
    return spline, tf


def main_pez_from_interceptions():
    initialEvaderPosition = np.array([-5.0, -5.0])
    finalEvaderPosition = np.array([5.0, 5.0])
    initialEvaderVelocity = np.array([1.0, 0.0])
    pursuerRange = 1.5
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.0
    evaderSpeed = 1.5
    num_cont_points = 20
    spline_order = 3
    velocity_constraints = (0.9, 1.1)
    curvature_constraints = (-0.5, 0.5)
    turn_rate_constraints = (-1.0, 1.0)
    num_constraint_samples = 50
    pez_limit = 0.01
    numPoints = 70
    xlim = (-4, 4)
    ylim = (-4, 4)

    pursuerPosition = np.array([0.0, 0.0])
    interceptionPositions = np.array([[0.4, 0.5], [-0.8, -0.8], [-0.7, 0.9]])
    dists = np.linalg.norm(pursuerPosition - interceptionPositions, axis=1)
    launchTimes = dists / pursuerSpeed * np.random.uniform(1, 1.1, size=dists.shape)
    pursuerPathDistances = launchTimes * pursuerSpeed
    if np.any(pursuerPathDistances > pursuerRange):
        print("Warning: launch times too long")

    radii = pursuerPathDistances + pursuerCaptureRadius
    radii = (pursuerRange + pursuerCaptureRadius) * np.ones(len(interceptionPositions))

    spline, tf = plan_path_PEZ_from_interceptions(
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        interceptionPositions,
        radii,
        xlim,
        ylim,
        numPoints,
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
        pez_limit,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_spline(spline, ax)
    # rectangle_pez.plot_box_pursuer_reachable_region(
    #     min_box, max_box, pursuerRange, pursuerCaptureRadius, ax=ax
    # )
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    arcs = pez_from_interceptions.bez_from_interceptions.compute_potential_pursuer_region_from_interception_position_and_radii(
        interceptionPositions,
        radii,
    )
    pez_from_interceptions.bez_from_interceptions.plot_potential_pursuer_reachable_region(
        arcs,
        pursuerRange,
        pursuerCaptureRadius,
        xlim,
        ylim,
        numPoints=200,
        ax=ax,
    )
    pez_from_interceptions.plot_prob_reachable(
        interceptionPositions,
        radii,
        pursuerRange,
        pursuerCaptureRadius,
        numPoints,
        xlim,
        ylim,
        ax,
        levels=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    # rectangle_pez.plot_rectangle_prr(
    #     pursuerRange,
    #     pursuerCaptureRadius,
    #     min_box,
    #     max_box,
    #     xlim=(-4, 4),
    #     ylim=(-4, 4),
    #     ax=ax,
    # )

    plt.legend()


if __name__ == "__main__":
    main_pez_from_interceptions()
    plt.show()
