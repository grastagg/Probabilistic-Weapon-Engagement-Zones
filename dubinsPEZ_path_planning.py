import os
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


matplotlib.use("Agg")

# get rid of type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# set font size for title, axis labels, and legend, and tick labels
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10


from bspline.matrix_evaluation import (
    matrix_bspline_evaluation_for_dataset,
    matrix_bspline_derivative_evaluation_for_dataset,
)

import dubinsPEZ
import dubinsEZ
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
    pez_limit=0.2,
    label=False,
):
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, 1000, endpoint=True)
    # set tick values
    ax.set_xticks(np.arange(-4, 4, 1))
    ax.set_yticks(np.arange(-4, 4, 1))

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
        # pez, _, _, _, _, _, _ = dubinsPEZ.mc_dubins_PEZ_differentiable(
        #     pos,
        #     agentHeadings,
        #     agentSpeed,
        #     pursuerPosition,
        #     pursuerPositionCov,
        #     pursuerHeading,
        #     pursuerHeadindgVar,
        #     pursuerSpeed,
        #     pursuerSpeedVar,
        #     pursuerTurnRadius,
        #     pursuerTurnRadiusVar,
        #     pursuerRange,
        #     pusuerRangeVar,
        #     pursuerCaptureRadius,
        # )
        # print("max monte smoothed carlo pez", np.max(pez))
        maxMCpez = np.max(pez)
        # linpez, _, _ = dubinsPEZ.linear_dubins_pez(
        #     pos,
        #     agentHeadings,
        #     agentSpeed,
        #     pursuerPosition,
        #     pursuerPositionCov,
        #     pursuerHeading,
        #     pursuerHeadindgVar,
        #     pursuerSpeed,
        #     pursuerSpeedVar,
        #     pursuerTurnRadius,
        #     pursuerTurnRadiusVar,
        #     pursuerRange,
        #     pusuerRangeVar,
        #     pursuerCaptureRadius,
        # )
        # print("max linear pez", np.max(linpez))

        # c = ax.scatter(x, y, c=pez, s=4, cmap="inferno")
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        #
        # cbar = plt.colorbar(c, cax=cax)
    # else:
    # ax.plot(x, y, label=f"CSPEZ Limit: {pez_limit}, Max MCCSPEZ: {maxMCpez}", linewidth=3)
    if label:
        # ax.plot(x, y, label=f"CSPEZ Limit: {pez_limit}", linewidth=2)
        ax.plot(x, y, label=r"$\epsilon=$" + str(pez_limit), linewidth=2)
    else:
        ax.plot(x, y, linewidth=2)

    ax.set_aspect(1)
    # c = plt.Circle(pursuerPosition, pursuerRange + pursuerCaptureRange, fill=False)
    # ax.add_artist(c)
    # print max mc and linear pez in title
    # limit to 3 decimal places


def dubins_PEZ_along_spline_nn(
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
    return pez


def dubins_PEZ_along_spline_linear(
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
    pez, _, _ = dubinsPEZ.linear_dubins_pez(
        # pez, _, _ = dubinsPEZ.quadratic_dubins_pez(
        # pez, _, _, _, _, _, _ = dubinsPEZ.mc_dubins_PEZ_differentiable(
        # pez, _, _ = nueral_network_EZ.nueral_network_pez(
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
    return pez


def dubins_PEZ_along_spline_quadratic(
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
    pez, _, _ = dubinsPEZ.quadratic_dubins_pez(
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
    return pez


def compute_spline_constraints_for_dubins_PEZ_nn(
    controlPoints,
    knotPoints,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    pursuerCaptureRadius,
    pursuerRange,
    pusuerRangeVar,
    pursuerTurnRadius,
    pursuerTurnRadiusVar,
    agentSpeed,
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
    return velocity, turn_rate, curvature, pez, pos


def compute_spline_constraints_for_dubins_PEZ_linear(
    controlPoints,
    knotPoints,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    pursuerCaptureRadius,
    pursuerRange,
    pusuerRangeVar,
    pursuerTurnRadius,
    pursuerTurnRadiusVar,
    agentSpeed,
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
    pez, _, _ = dubinsPEZ.linear_dubins_pez(
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
    return velocity, turn_rate, curvature, pez, pos


def compute_spline_constraints_for_dubins_PEZ_quadratic(
    controlPoints,
    knotPoints,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    pursuerCaptureRadius,
    pursuerRange,
    pusuerRangeVar,
    pursuerTurnRadius,
    pursuerTurnRadiusVar,
    agentSpeed,
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
    pez, _, _ = dubinsPEZ.quadratic_dubins_pez(
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
    return velocity, turn_rate, curvature, pez, pos


def create_spline(knotPoints, controlPoints, order):
    spline = BSpline(knotPoints, controlPoints, order)
    return spline


dDubinsPEZDControlPoints_nn = jax.jit(jacfwd(dubins_PEZ_along_spline_nn, argnums=0))
dDubinsPEZDtf_nn = jax.jit(jacfwd(dubins_PEZ_along_spline_nn, argnums=1))

dDubinsPEZDControlPoints_linear = jax.jit(
    jacfwd(dubins_PEZ_along_spline_linear, argnums=0)
)
dDubinsPEZDtf_linear = jax.jit(jacfwd(dubins_PEZ_along_spline_linear, argnums=1))

dDubinsPEZDControlPoints_quadratic = jax.jit(
    jacfwd(dubins_PEZ_along_spline_quadratic, argnums=0)
)
dDubinsPEZDtf_quadratic = jax.jit(jacfwd(dubins_PEZ_along_spline_quadratic, argnums=1))


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
    pez_limit,
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
    x0,
    tf,
    linearPez=True,
    quadraticPez=False,
    neuralNetworkPez=False,
):
    if linearPez:
        compute_spline_constraints_for_dubins_PEZ = (
            compute_spline_constraints_for_dubins_PEZ_linear
        )
        dDubinsPEZDControlPoints = dDubinsPEZDControlPoints_linear
        dDubinsPEZDtf = dDubinsPEZDtf_linear
    elif quadraticPez:
        compute_spline_constraints_for_dubins_PEZ = (
            compute_spline_constraints_for_dubins_PEZ_quadratic
        )
        dDubinsPEZDControlPoints = dDubinsPEZDControlPoints_quadratic
        dDubinsPEZDtf = dDubinsPEZDtf_quadratic
    elif neuralNetworkPez:
        compute_spline_constraints_for_dubins_PEZ = (
            compute_spline_constraints_for_dubins_PEZ_nn
        )
        dDubinsPEZDControlPoints = dDubinsPEZDControlPoints_nn
        dDubinsPEZDtf = dDubinsPEZDtf_nn

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

        velocity, turn_rate, curvature, pez, pos = (
            compute_spline_constraints_for_dubins_PEZ(
                controlPoints,
                knotPoints,
                pursuerPosition,
                pursuerPositionCov,
                pursuerHeading,
                pursuerHeadindgVar,
                pursuerSpeed,
                pursuerSpeedVar,
                pursuerCaptureRadius,
                pursuerRange,
                pursuerRangeVar,
                pursuerTurnRadius,
                pursuerTurnRadiusVar,
                agentSpeed,
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
        funcs["pez"] = pez
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

        dPEZDControlPoints = dDubinsPEZDControlPoints(
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
        dPEZDtf = dDubinsPEZDtf(
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
        funcsSens["pez"] = {"control_points": dPEZDControlPoints, "tf": dPEZDtf}

        return funcsSens, False

    # num_constraint_samples = numSamplesPerInterval*(num_cont_points-2)-2

    optProb = Optimization("path optimization", objfunc)

    # tf_initial = 1.0
    # knotPoints = spline_opt_tools.create_unclamped_knot_points(
    #     0, tf_initial, num_cont_points, 3
    # )
    # #
    # x0 = np.linspace(p0, pf, num_cont_points).flatten()
    # x0 = spline_opt_tools.move_first_control_point_so_spline_passes_through_start(
    #     x0, knotPoints, p0, v0
    # )
    # x0 = x0.flatten()
    # x0 = spline_opt_tools.move_last_control_point_so_spline_passes_through_end(
    #     x0, knotPoints, pf, v0
    # )
    # x0 = x0.flatten()

    # tf = spline_opt_tools.assure_velocity_constraint(
    #     x0,
    #     knotPoints,
    #     num_cont_points,
    #     agentSpeed,
    #     velocity_constraints,
    #     numSamplesPerInterval,
    # )

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
    optProb.addConGroup("pez", num_constraint_samples, lower=None, upper=pez_limit)
    optProb.addConGroup("start", 2, lower=p0, upper=p0)
    optProb.addConGroup("end", 2, lower=pf, upper=pf)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    opt.options["print_level"] = 0
    opt.options["max_iter"] = 200
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    # opt.options["derivative_test"] = "first-order"

    start = time.time()
    sol = opt(optProb, sens=sens)
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


def compare_pez_limits(
    p0,
    pf,
    v0,
    num_cont_points,
    spline_order,
    velocity_constraints,
    turn_rate_constraints,
    curvature_constraints,
    num_constraint_samples,
    pez_limits,
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
    x0,
    tf,
    ax,
    fig=None,
    linearPez=False,
    quadraticPez=False,
    neuralNetworkPez=False,
    cax=None,
    ylabel=False,
    label=False,
):
    # numFigures = len(pez_limits)
    # # two rows
    # fig, axs = plt.subplots(2, numFigures // 2, layout="tight")
    ax.set_xlabel("X")
    if ylabel:
        ax.set_ylabel("Y")

    for i, pez_limit in enumerate(pez_limits):
        print("pez limit", pez_limit)
        splineDubins = optimize_spline_path(
            p0,
            pf,
            v0,
            num_cont_points,
            spline_order,
            velocity_constraints,
            turn_rate_constraints,
            curvature_constraints,
            num_constraint_samples,
            pez_limit,
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
            x0,
            tf,
            linearPez=linearPez,
            quadraticPez=quadraticPez,
            neuralNetworkPez=neuralNetworkPez,
        )
        plot_spline(
            splineDubins,
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
            ax,
            plotPEZ=True,
            pez_limit=pez_limit,
            label=label,
        )
    if linearPez:
        ax.set_title("LCSPEZ")
        fast_pursuer.plotMahalanobisDistance(
            pursuerPosition, pursuerPositionCov, ax, fig, plotColorbar=False
        )
    elif quadraticPez:
        ax.set_title("QCSPEZ")
        fast_pursuer.plotMahalanobisDistance(
            pursuerPosition, pursuerPositionCov, ax, fig, plotColorbar=False
        )
    elif neuralNetworkPez:
        ax.set_title("NNCSPEZ")
        # ax.legend()
        fast_pursuer.plotMahalanobisDistance(
            pursuerPosition, pursuerPositionCov, ax, fig, plotColorbar=True
        )
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_xticks(np.arange(-4, 5, 2))
    ax.set_yticks(np.arange(-4, 5, 2))


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCov = np.array([[0.05, -0.04], [-0.04, 0.4]])
    # pursuerPositionCov = np.array([[0.025, 0.04], [0.04, 0.1]])
    # pursuerHeading = 0 * np.pi / 2
    pursuerHeading = (5.0 / 20.0) * np.pi
    # pursuerHeadingVar = 0.3
    pursuerHeadingVar = 0.2

    startingLocation = np.array([-4.0, -4.0])
    endingLocation = np.array([4.0, 4.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)
    initialVelocity = endingLocation - startingLocation

    numControlPoints = 14
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
    pursuerTurnRadius = 0.2
    pursuerTurnRadiusVar = 0.005
    agentSpeed = 1

    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity) * agentSpeed

    velocity_constraints = (agentSpeed - 0.001, agentSpeed + 0.001)
    detSpline = dubins_EZ_path_planning.optimize_spline_path(
        startingLocation,
        endingLocation,
        initialVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        0.0,
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        pursuerTurnRadius,
        agentSpeed,
    )
    start = time.time()
    detSpline = dubins_EZ_path_planning.optimize_spline_path(
        startingLocation,
        endingLocation,
        initialVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        0.0,
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        pursuerTurnRadius,
        agentSpeed,
    )
    print("time to plan deterministic spline", time.time() - start)

    fig, axes = plt.subplots(1, 3, figsize=(6, 3), layout="constrained")
    pez_limits = [0.01, 0.05, 0.25, 0.5]
    # pez_limits = [0.01]
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    # Create 3 subplots and 1 for the colorbar
    # pez_limits = [0.01]
    print("LIENAR")
    compare_pez_limits(
        startingLocation,
        endingLocation,
        initialVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        pez_limits,
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
        detSpline.c.flatten(),
        detSpline.t[-detSpline.k - 1],
        ax1,
        linearPez=True,
        fig=fig,
        ylabel=True,
        label=True,
    )
    print("QUADRATIC")
    compare_pez_limits(
        startingLocation,
        endingLocation,
        initialVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        pez_limits,
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
        detSpline.c.flatten(),
        detSpline.t[-detSpline.k - 1],
        ax2,
        quadraticPez=True,
        fig=fig,
    )
    print("NEURAL NETWORK")
    compare_pez_limits(
        startingLocation,
        endingLocation,
        initialVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        pez_limits,
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
        detSpline.c.flatten(),
        detSpline.t[-detSpline.k - 1],
        ax3,
        neuralNetworkPez=True,
        fig=fig,
    )
    fig.legend(
        loc="lower center",
        ncol=len(pez_limits),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    save_dir = os.path.expanduser("~/Desktop/cspez_plot")
    fig_path = os.path.join(save_dir, "path_planning_comparison.pdf")
    fig.savefig(fig_path, format="pdf")
    plt.show()


def animate_spline_path():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCov = np.array([[0.05, -0.04], [-0.04, 0.4]])
    # pursuerPositionCov = np.array([[0.025, 0.04], [0.04, 0.1]])
    # pursuerHeading = 0 * np.pi / 2
    pursuerHeading = (5.0 / 20.0) * np.pi
    # pursuerHeadingVar = 0.3
    pursuerHeadingVar = 0.2

    startingLocation = np.array([-4.0, -4.0])
    endingLocation = np.array([4.0, 4.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)
    initialVelocity = endingLocation - startingLocation

    numControlPoints = 14
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
    pursuerTurnRadius = 0.2
    pursuerTurnRadiusVar = 0.005
    agentSpeed = 1
    pez_limit = 0.25

    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity) * agentSpeed

    velocity_constraints = (agentSpeed - 0.001, agentSpeed + 0.001)
    detSpline = dubins_EZ_path_planning.optimize_spline_path(
        startingLocation,
        endingLocation,
        initialVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        0.0,
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        pursuerTurnRadius,
        agentSpeed,
    )
    spline = optimize_spline_path(
        startingLocation,
        endingLocation,
        initialVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        pez_limit,
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
        detSpline.c.flatten(),
        detSpline.t[-detSpline.k - 1],
        linearPez=False,
        quadraticPez=False,
        neuralNetworkPez=True,
    )

    currentTime = 0
    dt = 0.1
    finalTime = spline.t[-1 - spline.k]
    ind = 0
    while currentTime < finalTime:
        fig, ax = plt.subplots()
        pdot = spline.derivative(1)(currentTime)
        currentPosition = spline(currentTime)
        currentHeading = np.arctan2(pdot[1], pdot[0])

        plot_spline(
            spline,
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
            plotPEZ=False,
            pez_limit=pez_limit,
        )
        dubinsPEZ.plot_dubins_PEZ(
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
            currentHeading,
            agentSpeed,
            ax,
            useNueralNetwork=True,
            # useLinear=True,
            labelX=False,
            labelY=False,
            levels=[0.01, 0.05, 0.25, 0.5],
            # levels=[0.5],
        )
        plt.arrow(
            currentPosition[0],
            currentPosition[1],
            1e-6 * np.cos(currentHeading),  # essentially zero-length tail
            1e-6 * np.sin(currentHeading),
            head_width=0.25,
            head_length=0.3,
            width=0,  # no line
            fc="blue",
            ec="blue",
            zorder=5,
        )
        fast_pursuer.plotMahalanobisDistance(
            pursuerPosition, pursuerPositionCov, ax, fig
        )
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
        plt.show()

        fig.savefig(f"video/{ind}.png", dpi=300)
        ind += 1
        currentTime += dt
        plt.close(fig)


if __name__ == "__main__":
    animate_spline_path()
    # main()
