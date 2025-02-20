import numpy as np
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

import pez_path_planner
import dubins_EZ_path_planning

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
    pez, _, _ = dubinsPEZ.mc_dubins_PEZ(
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
    linpez, _, _ = dubinsPEZ.linear_dubins_pez(
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
    print("max linear pez", np.max(linpez))

    c = ax.scatter(x, y, c=pez, s=4, cmap="inferno")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(c, cax=cax)
    cbar.ax.tick_params(labelsize=16)

    ax.set_aspect(1)
    # c = plt.Circle(pursuerPosition, pursuerRange + pursuerCaptureRange, fill=False)
    ax.scatter(pursuerPosition[0], pursuerPosition[1], c="r")
    # ax.add_artist(c)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # print max mc and linear pez in title
    # limit to 3 decimal places
    ax.set_title(
        f"Max MC PEZ: {np.max(pez):.3f}" + f" Max Linear PEZ: {np.max(linpez):.3f}"
    )


def evaluate_spline(controlPoints, knotPoints):
    knotPoints = knotPoints.reshape((-1,))
    return matrix_bspline_evaluation_for_dataset(
        controlPoints.T, knotPoints, numSamplesPerInterval
    )


def evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, derivativeOrder):
    scaleFactor = knotPoints[-splineOrder - 1] / (len(knotPoints) - 2 * splineOrder - 1)
    return matrix_bspline_derivative_evaluation_for_dataset(
        derivativeOrder, scaleFactor, controlPoints.T, knotPoints, numSamplesPerInterval
    )


@partial(jit, static_argnums=(2, 3))
def create_unclamped_knot_points(t0, tf, numControlPoints, splineOrder):
    internalKnots = jnp.linspace(t0, tf, numControlPoints - 2, endpoint=True)
    h = internalKnots[1] - internalKnots[0]
    knots = jnp.concatenate(
        (
            jnp.linspace(t0 - splineOrder * h, t0 - h, splineOrder),
            internalKnots,
            jnp.linspace(tf + h, tf + splineOrder * h, splineOrder),
        )
    )

    return knots


@partial(jit, static_argnums=(2, 3))
def get_spline_velocity(controlPoints, tf, splineOrder, numSamplesPerInterval):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 1)
    return jnp.linalg.norm(out_d1, axis=1)


@partial(jit, static_argnums=(2, 3))
def get_spline_turn_rate(controlPoints, tf, splineOrder, numSamplesPerInterval):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 1)
    out_d2 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 2)
    v = jnp.linalg.norm(out_d1, axis=1)
    u = jnp.cross(out_d1, out_d2) / (v**2)
    return u


@partial(jit, static_argnums=(2, 3))
def get_spline_curvature(controlPoints, tf, splineOrder, numSamplesPerInterval):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 1)
    out_d2 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 2)
    v = jnp.linalg.norm(out_d1, axis=1)
    u = jnp.cross(out_d1, out_d2) / (v**2)
    return u / v


@partial(jit, static_argnums=(2, 3))
def get_spline_heading(controlPoints, tf, splineOrder, numSamplesPerInterval):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 1)
    return jnp.arctan2(out_d1[:, 1], out_d1[:, 0])


def get_start_constraint(controlPoints):
    cp1 = controlPoints[0:2]
    cp2 = controlPoints[2:4]
    cp3 = controlPoints[4:6]
    return np.array((1 / 6) * cp1 + (2 / 3) * cp2 + (1 / 6) * cp3)


def get_end_constraint(controlPoints):
    cpnMinus2 = controlPoints[-6:-4]
    cpnMinus1 = controlPoints[-4:-2]
    cpn = controlPoints[-2:]
    return (1 / 6) * cpnMinus2 + (2 / 3) * cpnMinus1 + (1 / 6) * cpn


def get_start_constraint_jacobian(controlPoints):
    numControlPoints = int(controlPoints.shape[0] / 2)
    jac = np.zeros((2, 2 * numControlPoints))
    jac[0, 0] = 1 / 6
    jac[0, 2] = 2 / 3
    jac[0, 4] = 1 / 6
    jac[1, 1] = 1 / 6
    jac[1, 3] = 2 / 3
    jac[1, 5] = 1 / 6
    return jac


def get_end_constraint_jacobian(controlPoints):
    numControlPoints = int(controlPoints.shape[0] / 2)
    jac = np.zeros((2, 2 * numControlPoints))
    jac[0, -6] = 1 / 6
    jac[0, -4] = 2 / 3
    jac[0, -2] = 1 / 6
    jac[1, -5] = 1 / 6
    jac[1, -3] = 2 / 3
    jac[1, -1] = 1 / 6
    return jac


def get_turn_rate_velocity_and_headings(controlPoints, knotPoints):
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, 3, 1)
    out_d2 = evaluate_spline_derivative(controlPoints, knotPoints, 3, 2)
    v = np.linalg.norm(out_d1, axis=1)
    u = np.cross(out_d1, out_d2) / (v**2)
    heading = np.arctan2(out_d1[:, 1], out_d1[:, 0])
    return u, v, heading


def dubins_PEZ_along_spline(
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
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, 3)
    agentHeadings = get_spline_heading(controlPoints, tf, 3, numSamplesPerInterval)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    pos = evaluate_spline(controlPoints, knotPoints)
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
    return pez


def compute_spline_constraints_for_dubins_PEZ(
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
    pos = evaluate_spline(controlPoints, knotPoints)

    turn_rate, velocity, agentHeadings = get_turn_rate_velocity_and_headings(
        controlPoints, knotPoints
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


def create_spline(knotPoints, controlPoints, order):
    spline = BSpline(knotPoints, controlPoints, order)
    return spline


dVelocityDControlPoints = jacfwd(get_spline_velocity)
sdVelocityDtf = jacfwd(get_spline_velocity, argnums=1)

dTurnRateDControlPoints = jacfwd(get_spline_turn_rate)
dTurnRateTf = jacfwd(get_spline_turn_rate, argnums=1)

dCurvatureDControlPoints = jacfwd(get_spline_curvature)
dCurvatureDtf = jacfwd(get_spline_curvature, argnums=1)

dDubinsPEZDControlPoints = jacfwd(dubins_PEZ_along_spline, argnums=0)
dDubinsPEZDtf = jacfwd(dubins_PEZ_along_spline, argnums=1)


def assure_velocity_constraint(
    controlPoints, knotPoints, num_control_points, agentSpeed, velocityBounds
):
    splineOrder = 3
    tf = np.linalg.norm(controlPoints[0] - controlPoints[-1]) / agentSpeed
    v = get_spline_velocity(controlPoints, tf, splineOrder, numSamplesPerInterval)
    while np.max(v) > velocityBounds[1]:
        tf += 0.01
        # combined_knot_points = self.create_unclamped_knot_points(0, tf, num_control_points,params.splineOrder)
        v = get_spline_velocity(controlPoints, tf, splineOrder, numSamplesPerInterval)
        # pd, u, v, pos = self.spline_constraints(radarList, controlPoints, combined_knot_points,params.numConstraintSamples)
    print("max v", np.max(v))
    return tf


def move_first_control_point_so_spline_passes_through_start(
    controlPoints, knotPoints, start, startVelocity
):
    num_control_points = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((num_control_points, 2))
    dt = knotPoints[3] - knotPoints[0]
    A = np.array(
        [
            [1 / 6, 0, 2 / 3, 0],
            [0, 1 / 6, 0, 2 / 3],
            [-3 / (2 * dt), 0, 0, 0],
            [0, -3 / (2 * dt), 0, 0],
        ]
    )
    c3x = controlPoints[2, 0]
    c3y = controlPoints[2, 1]

    b = np.array(
        [
            [start[0] - (1 / 6) * c3x],
            [start[1] - (1 / 6) * c3y],
            [startVelocity[0] - 3 / (2 * dt) * c3x],
            [startVelocity[1] - 3 / (2 * dt) * c3y],
        ]
    )

    x = np.linalg.solve(A, b)
    controlPoints[0:2, 0:2] = x.reshape((2, 2))

    return controlPoints


def move_last_control_point_so_spline_passes_through_end(
    controlPoints, knotPoints, end, endVelocity
):
    num_control_points = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((num_control_points, 2))
    dt = knotPoints[3] - knotPoints[0]
    A = np.array(
        [
            [2 / 3, 0, 1 / 6, 0],
            [0, 2 / 3, 0, 1 / 6],
            [0, 0, 3 / (2 * dt), 0],
            [0, 0, 0, 3 / (2 * dt)],
        ]
    )
    cn_minus_2_x = controlPoints[-3, 0]
    cn_minus_2_y = controlPoints[-3, 1]

    b = np.array(
        [
            [end[0] - (1 / 6) * cn_minus_2_x],
            [end[1] - (1 / 6) * cn_minus_2_y],
            [endVelocity[0] + 3 / (2 * dt) * cn_minus_2_x],
            [endVelocity[1] + 3 / (2 * dt) * cn_minus_2_y],
        ]
    )

    x = np.linalg.solve(A, b)
    controlPoints[-2:, -2:] = x.reshape((2, 2))

    return controlPoints


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
):
    # Compute Jacobian of engagement zone function

    def objfunc(xDict):
        tf = xDict["tf"]
        knotPoints = create_unclamped_knot_points(0, tf, num_cont_points, 3)
        controlPoints = xDict["control_points"]
        funcs = {}
        funcs["start"] = get_start_constraint(controlPoints)
        funcs["end"] = get_end_constraint(controlPoints)
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

        dStartDControlPointsVal = get_start_constraint_jacobian(controlPoints)
        dEndDControlPointsVal = get_end_constraint_jacobian(controlPoints)

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

        dVelocityDControlPointsVal = dVelocityDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dVelocityDtfVal = sdVelocityDtf(controlPoints, tf, 3, numSamplesPerInterval)
        dTurnRateDControlPointsVal = dTurnRateDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dTurnRateDtfVal = dTurnRateTf(controlPoints, tf, 3, numSamplesPerInterval)
        dCurvatureDControlPointsVal = dCurvatureDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dCurvatureDtfVal = dCurvatureDtf(controlPoints, tf, 3, numSamplesPerInterval)

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
    # knotPoints = create_unclamped_knot_points(0, tf_initial, num_cont_points, 3)
    #
    # x0 = np.linspace(p0, pf, num_cont_points).flatten()
    # x0 = move_first_control_point_so_spline_passes_through_start(x0, knotPoints, p0, v0)
    # x0 = x0.flatten()
    # x0 = move_last_control_point_so_spline_passes_through_end(x0, knotPoints, pf, v0)
    # x0 = x0.flatten()
    #
    # print("velocity constraints", velocity_constraints)
    # tf = assure_velocity_constraint(
    #     x0, knotPoints, num_cont_points, agentSpeed, velocity_constraints
    # )

    # x0 = np.array(
    #     [
    #         -6.99550637,
    #         -8.95872567,
    #         -4.47336006,
    #         -5.06316688,
    #         -5.11105337,
    #         -0.78860679,
    #         -3.19745253,
    #         3.19381798,
    #         0.79036742,
    #         5.04575115,
    #         5.06051916,
    #         4.51374057,
    #         8.96755593,
    #         6.89928658,
    #     ]
    # )

    tempVelocityContstraints = get_spline_velocity(x0, 1, 3, 1)
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
    optProb.addConGroup("pez", num_constraint_samples, lower=0.0, upper=pez_limit)
    optProb.addConGroup("start", 2, lower=p0, upper=p0)
    optProb.addConGroup("end", 2, lower=pf, upper=pf)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    opt.options["print_level"] = 0
    opt.options["max_iter"] = 500
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    opt.options["derivative_test"] = "first-order"

    sol = opt(optProb, sens=sens)
    # sol = opt(optProb, sens="FD")
    # print(sol)
    if sol.optInform["value"] != 0:
        print("Optimization failed")

    print("time", sol.xStar["tf"])

    knotPoints = create_unclamped_knot_points(0, sol.xStar["tf"][0], num_cont_points, 3)
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
):
    numFigures = len(pez_limits)
    # two rows
    fig, axs = plt.subplots(2, numFigures // 2, layout="tight")
    for i, pez_limit in enumerate(pez_limits):
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
        )
        ax = axs[i // 3, i % 3]
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
        )


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCov = np.eye(2) * 0.1
    pursuerHeading = np.pi / 2
    pursuerHeadingVar = 0.4

    startingLocation = np.array([-4.0, 1.0])
    endingLocation = np.array([4.0, -1.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)
    initialVelocity = endingLocation - startingLocation

    numControlPoints = 8
    splineOrder = 3
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)
    num_constraint_samples = 50

    pursuerRange = 1.0
    pursuerRangeVar = 0.2
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.2
    pursuerTurnRadius = 0.2
    pursuerTurnRadiusVar = 0.05
    agentSpeed = 1

    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity) * agentSpeed

    velocity_constraints = (agentSpeed - 0.01, agentSpeed + 0.01)
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
        pursuerRange * 2,
        pursuerCaptureRadius,
        pursuerSpeed,
        pursuerTurnRadius,
        agentSpeed,
    )
    start = time.time()
    detSplineT = dubins_EZ_path_planning.optimize_spline_path(
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

    pez_limits = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
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
    )

    # pez_limit = 0.2
    # splineDubins = optimize_spline_path(
    #     startingLocation,
    #     endingLocation,
    #     initialVelocity,
    #     numControlPoints,
    #     splineOrder,
    #     velocity_constraints,
    #     turn_rate_constraints,
    #     curvature_constraints,
    #     num_constraint_samples,
    #     pez_limit,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     pursuerCaptureRadius,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     pursuerTurnRadius,
    #     pursuerTurnRadiusVar,
    #     agentSpeed,
    #     detSpline.c.flatten(),
    #     detSpline.t[-detSpline.k - 1],
    # )
    # pez_limit = 0.1
    # start = time.time()
    # splineDubins = optimize_spline_path(
    #     startingLocation,
    #     endingLocation,
    #     initialVelocity,
    #     numControlPoints,
    #     splineOrder,
    #     velocity_constraints,
    #     turn_rate_constraints,
    #     curvature_constraints,
    #     num_constraint_samples,
    #     pez_limit,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     pursuerCaptureRadius,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     pursuerTurnRadius,
    #     pursuerTurnRadiusVar,
    #     agentSpeed,
    #     detSpline.c.flatten(),
    #     detSpline.t[-detSpline.k - 1],
    # )
    # print("time to plan probabilistic spline", time.time() - start)

    plt.show()


if __name__ == "__main__":
    main()
