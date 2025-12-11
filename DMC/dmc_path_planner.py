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

import DMC.dmc as DMC
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


@jax.jit
def dmc_along_spline(
    controlPoints,
    tf,
    evaderSpeed,
    pursuerPositions,
    pursuerRanges,
    pursuerCaptureRadiuses,
    pursuerSpeeds,
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
    dmc = DMC.dmc_multiple_pursuer(
        pos,
        evaderHeadings,
        evaderSpeed,
        pursuerPositions,
        pursuerSpeeds,
        pursuerRanges,
        pursuerCaptureRadiuses,
    )

    return dmc


@jax.jit
def compute_spline_constraints_for_DMC(
    controlPoints,
    knotPoints,
    evaderSpeed,
    pursuerPositions,
    pursuerRanges,
    pursuerCaptureRadiuses,
    pursuerSpeeds,
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

    dmc = DMC.dmc_multiple_pursuer(
        pos,
        evaderHeadings,
        evaderSpeed,
        pursuerPositions,
        pursuerSpeeds,
        pursuerRanges,
        pursuerCaptureRadiuses,
    )

    return velocity, turn_rate, curvature, dmc, pos, evaderHeadings


dDMCDControlPoints = jax.jit(jax.jacfwd(dmc_along_spline, argnums=0))
dDMCDtf = jax.jit(jax.jacfwd(dmc_along_spline, argnums=1))


def get_initial_spline_control_points(p0, pf, numControlPoints):
    controlPoints = np.linspace(p0, pf, numControlPoints)
    return controlPoints


def optimize_spline_path_DMC(
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
    pursuerPositions,
    pursuerRanges,
    pursuerCaptureRadiuses,
    pursuerSpeeds,
    right=True,
    previous_spline=None,
    dmc_limit=0.1,
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

        velocity, turn_rate, curvature, dmc, pos, evaderHeading = (
            compute_spline_constraints_for_DMC(
                controlPoints,
                knotPoints,
                evaderSpeed,
                pursuerPositions,
                pursuerRanges,
                pursuerCaptureRadiuses,
                pursuerSpeeds,
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
        funcs["dmc"] = dmc
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

        dEZDControlPoints = dDMCDControlPoints(
            controlPoints,
            tf,
            evaderSpeed,
            pursuerPositions,
            pursuerRanges,
            pursuerCaptureRadiuses,
            pursuerSpeeds,
        )
        dEZDtf = dDMCDtf(
            controlPoints,
            tf,
            evaderSpeed,
            pursuerPositions,
            pursuerRanges,
            pursuerCaptureRadiuses,
            pursuerSpeeds,
        )
        # # replace nans with zeros
        # dEZDtf = np.array(dEZDtf, copy=True)
        # dEZDtf[np.isnan(dEZDtf)] = 0.0
        dEZDControlPoints = np.array(dEZDControlPoints, copy=True)
        dEZDControlPoints[np.isnan(dEZDControlPoints)] = 0.0
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
        funcsSens["dmc"] = {"control_points": dEZDControlPoints, "tf": dEZDtf}

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

        x0 = get_initial_spline_control_points(p0, pf, num_cont_points).flatten()

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

    ###end test grad

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
    optProb.addConGroup("dmc", num_constraint_samples, lower=None, upper=dmc_limit)
    optProb.addConGroup("start", 2, lower=p0, upper=p0)
    optProb.addConGroup("end", 2, lower=pf, upper=pf)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 100
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


def main_dmc():
    initialEvaderPosition = np.array([-2.5, 1.0])
    finalEvaderPosition = np.array([2.5, -1.0])
    initialEvaderVelocity = np.array([1.0, 0.0])
    pursuerPositions = np.array([[0.0, 0.0], [1.0, 1.0]])
    pursuerRanges = np.array([1.0, 1.0])
    pursuerSpeeds = np.array([2.0, 2.0])
    pursuerCaptureRadiuses = np.array([0.2, 0.2])
    evaderSpeed = 1.5
    num_cont_points = 20
    spline_order = 3
    velocity_constraints = (0.9, 1.1)
    curvature_constraints = (-0.5, 0.5)
    turn_rate_constraints = (-1.0, 1.0)
    num_constraint_samples = 50
    dmc_limit = np.deg2rad(10)

    spline, tf = optimize_spline_path_DMC(
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
        pursuerPositions=pursuerPositions,
        pursuerRanges=pursuerRanges,
        pursuerCaptureRadiuses=pursuerCaptureRadiuses,
        pursuerSpeeds=pursuerSpeeds,
        right=False,
        previous_spline=None,
        dmc_limit=dmc_limit,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_spline(spline, ax)
    ax.set_aspect("equal")
    for i, pursuerPosition in enumerate(pursuerPositions):
        pursuerRange = pursuerRanges[i]
        pursuerCaptureRadius = pursuerCaptureRadiuses[i]
        circle = plt.Circle(
            pursuerPosition, pursuerRange, color="r", fill=False, linestyle=":"
        )
        ax.add_artist(circle)
        circle = plt.Circle(
            pursuerPosition,
            pursuerRange + pursuerCaptureRadius,
            color="r",
            fill=False,
            linestyle="--",
        )
        ax.add_artist(circle)
    plt.legend()


if __name__ == "__main__":
    main_dmc()
    plt.show()
