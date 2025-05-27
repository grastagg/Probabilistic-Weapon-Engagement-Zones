import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

import fast_pursuer

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from bspline.matrix_evaluation import (
    matrix_bspline_evaluation_for_dataset,
    matrix_bspline_derivative_evaluation_for_dataset,
)

import dubinsPEZ
import nueral_network_EZ

import dubins_EZ_path_planning

import spline_opt_tools

numSamplesPerInterval = 25


# def plot_spline(spline, pursuerPosition, pursuerRange, pursuerCaptureRange,pez_limit,useProbabalistic):
def plot_spline(
    spline,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerRange,
    pusuerRangeVar,
    pursuerCaptureRadius,
    pursuerSpeed,
    pursuerSpeedVar,
    pursuerTurnRadius,
    pursuerTurnRadiusVar,
    agentSpeed,
    ax,
    plotPEZ=True,
):
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
    if plotPEZ:
        pez, _, _, _, _, _, _ = dubinsPEZ.mc_dubins_PEZ(
            # pez, _, _, _, _, _, _ = dubinsPEZ.mc_dubins_PEZ_differentiable(
            pos,
            agentHeadings,
            agentSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            pursuerTurnRadius,
            pursuerTurnRadiusVar,
            pursuerRange,
            pusuerRangeVar,
            pursuerCaptureRadius,
        )
        print("max monte carlo pez", np.max(pez))
        nnPez, _, _ = nueral_network_EZ.nueral_network_pez(
            pos,
            agentHeadings,
            agentSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            pursuerTurnRadius,
            pursuerTurnRadiusVar,
            pursuerRange,
            pusuerRangeVar,
            pursuerCaptureRadius,
        )
        print("max nn pez", np.max(nnPez))

        c = ax.scatter(x, y, c=pez, s=4, cmap="inferno")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(c, cax=cax)
        cbar.ax.tick_params(labelsize=16)
    # else:
    # ax.plot(x, y, label=f"CSPEZ Limit: {pez_limit}, Max MCCSPEZ: {maxMCpez}", linewidth=3)
    # ax.plot(x, y, linewidth=3)

    ax.set_aspect(1)
    # c = plt.Circle(pursuerPosition, pursuerRange + pursuerCaptureRange, fill=False)
    ax.scatter(pursuerPosition[0], pursuerPosition[1], c="r")
    # ax.add_artist(c)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # print max mc and linear pez in title
    # limit to 3 decimal places


def compute_spline_constraints(
    controlPoints,
    knotPoints,
):
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )

    turn_rate, velocity, agentHeadings = (
        spline_opt_tools.get_turn_rate_velocity_and_headings(
            controlPoints, knotPoints, numSamplesPerInterval
        )
    )

    curvature = turn_rate / velocity
    return velocity, turn_rate, curvature, pos


def create_spline(knotPoints, controlPoints, order):
    spline = BSpline(knotPoints, controlPoints, order)
    return spline


def smooth_max_logsumexp(x, epsilon=1e-2):
    return epsilon * jax.scipy.special.logsumexp(x / epsilon)


def max_dubins_PEZ_along_spline_nn(
    controlPoints,
    tf,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    pursuerRange,
    pusuerRangeVar,
    pursuerCaptureRadius,
    pursuerTurnRadius,
    pursuerTurnRadiusVar,
    agentSpeed,
):
    numControlPoints = int(len(controlPoints) / 2)
    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, tf, numControlPoints, 3
    )
    agentHeadings = spline_opt_tools.get_spline_heading(
        controlPoints, tf, 3, numSamplesPerInterval
    )
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )
    # pez, _, _ = dubinsPEZ.linear_dubins_pez(
    # pez, _, _ = dubinsPEZ.quadratic_dubins_pez(
    # pez, _, _, _, _, _, _ = dubinsPEZ.mc_dubins_PEZ_differentiable(
    pez, _, _ = nueral_network_EZ.nueral_network_pez(
        pos,
        agentHeadings,
        agentSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadindgVar,
        pursuerSpeed,
        pursuerSpeedVar,
        pursuerTurnRadius,
        pursuerTurnRadiusVar,
        pursuerRange,
        pusuerRangeVar,
        pursuerCaptureRadius,
    )
    # return jnp.mean(pez)
    return smooth_max_logsumexp(pez, 1e-2)
    # return jnp.max(pez)


dMaxDubinsPEZControlPoints = jax.jit(jacfwd(max_dubins_PEZ_along_spline_nn, argnums=0))
dMaxDubinsPEZtf = jax.jit(jacfwd(max_dubins_PEZ_along_spline_nn, argnums=1))


def optimize_spline_path(
    p0,
    pf,
    v0,
    num_cont_points,
    spline_order,
    velocity_constraints,
    turn_rate_constraints,
    curvature_constraints,
    num_constraint_samples,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerRange,
    pursuerRangeVar,
    pursuerCaptureRadius,
    pursuerSpeed,
    pursuerSpeedVar,
    pursuerTurnRadius,
    pursuerTurnRadiusVar,
    agentSpeed,
):
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

        velocity, turn_rate, curvature, pos = compute_spline_constraints(
            controlPoints,
            knotPoints,
        )
        maxPez = max_dubins_PEZ_along_spline_nn(
            controlPoints.flatten(),
            tf,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRadius,
            pursuerTurnRadius,
            pursuerTurnRadiusVar,
            agentSpeed,
        )

        funcs["turn_rate"] = turn_rate
        funcs["velocity"] = velocity
        funcs["curvature"] = curvature
        # funcs['position'] = pos
        funcs["obj"] = maxPez
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

        dPEZDControlPoints = dMaxDubinsPEZControlPoints(
            controlPoints,
            tf,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRadius,
            pursuerTurnRadius,
            pursuerTurnRadiusVar,
            agentSpeed,
        )
        dPEZDtf = dMaxDubinsPEZtf(
            controlPoints,
            tf,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRadius,
            pursuerTurnRadius,
            pursuerTurnRadiusVar,
            agentSpeed,
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
            "control_points": dPEZDControlPoints,
            "tf": dPEZDtf,
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

        return funcsSens, False

    # num_constraint_samples = numSamplesPerInterval*(num_cont_points-2)-2

    optProb = Optimization("path optimization", objfunc)

    tf_initial = 1.0
    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, tf_initial, num_cont_points, 3
    )
    #
    x0 = np.linspace(p0, pf, num_cont_points).flatten()
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
        agentSpeed,
        velocity_constraints,
        numSamplesPerInterval,
    )

    tempVelocityContstraints = spline_opt_tools.get_spline_velocity(
        x0, 1, 3, numSamplesPerInterval
    )
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
    optProb.addConGroup("start", 2, lower=p0, upper=p0)
    optProb.addConGroup("end", 2, lower=pf, upper=pf)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 200
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    opt.options["derivative_test"] = "first-order"

    start = time.time()
    sol = opt(optProb, sens=sens)
    print("objective", sol.fStar)
    # sol = opt(optProb, sens="FD")
    print("time to plan spline", time.time() - start)
    if sol.optInform["value"] != 0:
        print("Optimization failed")

    print("path time", sol.xStar["tf"])

    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, sol.xStar["tf"][0], num_cont_points, 3
    )
    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))
    return create_spline(knotPoints, controlPoints, spline_order)


def main():
    pursuerPosition = np.array([0.0, 0.0])
    # pursuerPositionCov = np.array([[0.05, -0.04], [-0.04, 0.4]])
    # pursuerPositionCov = np.array([[0.025, 0.04], [0.04, 0.1]])
    pursuerPositionCov = np.array([[0.025, 0.00], [0.00, 0.1]])
    # pursuerHeading = 0 * np.pi / 2
    pursuerHeading = (0.0 / 20.0) * np.pi
    # pursuerHeadingVar = 0.3
    pursuerHeadingVar = 0.1

    startingLocation = np.array([-4.0, 0.0])
    endingLocation = pursuerPosition
    initialVelocity = np.array([-1.0, 0.0]) / np.sqrt(2)
    # initialVelocity = endingLocation - startingLocation
    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity)

    numControlPoints = 12
    splineOrder = 3
    turn_rate_constraints = (-1.0, 1.0)
    curvature_constraints = (-0.2, 0.2)
    num_constraint_samples = 50

    pursuerRange = 1.0
    # pursuerRangeVar = 0.1
    pursuerRangeVar = 0.1
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    # pursuerSpeedVar = 0.1
    pursuerSpeedVar = 0.3
    pursuerTurnRadius = 0.3
    pursuerTurnRadiusVar = 0.005
    agentSpeed = 1

    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity) * agentSpeed

    velocity_constraints = (agentSpeed - 0.01, agentSpeed + 0.01)

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)

    splineDubins = optimize_spline_path(
        startingLocation,
        endingLocation,
        initialVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRadius,
        pursuerSpeed,
        pursuerSpeedVar,
        pursuerTurnRadius,
        pursuerTurnRadiusVar,
        agentSpeed,
    )
    plot_spline(
        splineDubins,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRadius,
        pursuerSpeed,
        pursuerSpeedVar,
        pursuerTurnRadius,
        pursuerTurnRadiusVar,
        agentSpeed,
        ax,
        plotPEZ=True,
    )
    nueralEZ = dubinsPEZ.plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        pursuerTurnRadius,
        pursuerTurnRadiusVar,
        0.0,
        pursuerRange,
        pursuerRangeVar,
        0.0,
        agentSpeed,
        ax,
        # useNumerical=True,
        useNueralNetwork=True,
        # useLinearPlusNueralNetwork=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
