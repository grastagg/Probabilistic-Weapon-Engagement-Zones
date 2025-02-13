import numpy as np
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

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from fast_pursuer import (
    plotMahalanobisDistance,
    probabalisticEngagementZoneVectorizedTemp,
    inEngagementZoneJaxVectorized,
)
from bspline.matrix_evaluation import (
    matrix_bspline_evaluation_for_dataset,
    matrix_bspline_derivative_evaluation_for_dataset,
)

from dubinsEZ import in_dubins_engagement_zone, vectorized_find_shortest_dubins_path

numSamplesPerInterval = 15


# def plot_spline(spline, pursuerPosition, pursuerRange, pursuerCaptureRange,pez_limit,useProbabalistic):
def plot_spline(
    spline,
    pursuerPosition,
    pursuerHeading,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    pursuerTurnRadius,
    agentSpeed,
    ax,
):
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, 100, endpoint=True)
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

    print("spline.c", spline.c)
    pos = spline(t)
    deriv = spline.derivative()(t)
    headings = np.arctan2(deriv[:, 1], deriv[:, 0])
    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeading,
        pursuerTurnRadius,
        pursuerCaptureRadius,
        pursuerRange,
        pursuerSpeed,
        pos,
        headings,
        agentSpeed,
    )
    len = vectorized_find_shortest_dubins_path(
        pursuerPosition, pursuerHeading, pos, pursuerTurnRadius
    )
    print("len", len)

    c = ax.scatter(x, y, c=len, s=4)
    cbar = plt.colorbar(c, shrink=0.8)
    cbar.ax.tick_params(labelsize=26)

    ax.set_aspect(1)
    # c = plt.Circle(pursuerPosition, pursuerRange + pursuerCaptureRange, fill=False)
    plt.scatter(pursuerPosition[0], pursuerPosition[1], c="r")
    ax.add_artist(c)
    plt.xlabel("X")
    plt.ylabel("Y")


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


def dubins_EZ_along_spline(
    controlPoints,
    tf,
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerTurnRadius,
    agentSpeed,
):
    numControlPoints = int(len(controlPoints) / 2)
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, 3)
    agentHeadings = get_spline_heading(controlPoints, tf, 3, numSamplesPerInterval)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    pos = evaluate_spline(controlPoints, knotPoints)
    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeading,
        pursuerTurnRadius,
        pursuerCaptureRadius,
        pursuerRange,
        pursuerSpeed,
        pos,
        agentHeadings,
        agentSpeed,
    )
    return ez


def compute_spline_constraints_for_dubins_EZ_deterministic(
    controlPoints,
    knotPoints,
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    captureRadius,
    pursuerRange,
    turnRadius,
    agentSpeed,
):
    pos = evaluate_spline(controlPoints, knotPoints)

    turn_rate, velocity, agentHeadings = get_turn_rate_velocity_and_headings(
        controlPoints, knotPoints
    )

    curvature = turn_rate / velocity

    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeading,
        turnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        pos,
        agentHeadings,
        agentSpeed,
    )
    return velocity, turn_rate, curvature, ez, pos


def create_spline(knotPoints, controlPoints, order):
    spline = BSpline(knotPoints, controlPoints, order)
    return spline


dVelocityDControlPoints = jacfwd(get_spline_velocity)
sdVelocityDtf = jacfwd(get_spline_velocity, argnums=1)

dTurnRateDControlPoints = jacfwd(get_spline_turn_rate)
dTurnRateTf = jacfwd(get_spline_turn_rate, argnums=1)

dCurvatureDControlPoints = jacfwd(get_spline_curvature)
dCurvatureDtf = jacfwd(get_spline_curvature, argnums=1)

dDubinsEZDControlPoints = jacfwd(dubins_EZ_along_spline, argnums=0)
dDubinsEZDtf = jacfwd(dubins_EZ_along_spline, argnums=1)


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
    ez_constraint_limit,
    pursuerPosition,
    pursuerHeading,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    pursuerTurnRadius,
    agentSpeed,
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

        velocity, turn_rate, curvature, ez, pos = (
            compute_spline_constraints_for_dubins_EZ_deterministic(
                controlPoints,
                knotPoints,
                pursuerPosition,
                pursuerHeading,
                pursuerSpeed,
                pursuerCaptureRadius,
                pursuerRange,
                pursuerTurnRadius,
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
        funcs["ez"] = ez
        funcs["obj"] = tf
        return funcs, False

    def sens(xDict, funcs):
        funcsSens = {}
        controlPoints = jnp.array(xDict["control_points"])
        tf = xDict["tf"]

        dStartDControlPointsVal = get_start_constraint_jacobian(controlPoints)
        dEndDControlPointsVal = get_end_constraint_jacobian(controlPoints)

        dEZDControlPoints = dDubinsEZDControlPoints(
            controlPoints,
            tf,
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerTurnRadius,
            agentSpeed,
        )
        dEZDtf = dDubinsEZDtf(
            controlPoints,
            tf,
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerTurnRadius,
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
        funcsSens["ez"] = {"control_points": dEZDControlPoints, "tf": dEZDtf}

        return funcsSens, False

    # num_constraint_samples = numSamplesPerInterval*(num_cont_points-2)-2

    optProb = Optimization("path optimization", objfunc)

    # x0_x = np.linspace(p0[0], pf[0], num_cont_points)
    # x0_y = np.linspace(p0[0], pf[0], num_cont_points)
    # x0 = np.hstack((x0_x,x0_y)).reshape((2*(num_cont_points)))
    x0 = np.array(
        [
            -6.99550637,
            -8.95872567,
            -4.47336006,
            -5.06316688,
            -5.11105337,
            -0.78860679,
            -3.19745253,
            3.19381798,
            0.79036742,
            5.04575115,
            5.06051916,
            4.51374057,
            8.96755593,
            6.89928658,
        ]
    )

    tempVelocityContstraints = get_spline_velocity(x0, 1, 3, 1)
    num_constraint_samples = len(tempVelocityContstraints)

    optProb.addVarGroup(
        name="control_points", nVars=2 * (num_cont_points), varType="c", value=x0
    )
    optProb.addVarGroup(name="tf", nVars=1, varType="c", value=10, lower=0, upper=None)

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
    optProb.addConGroup("ez", num_constraint_samples, lower=0.0, upper=None)
    optProb.addConGroup("start", 2, lower=p0, upper=p0)
    optProb.addConGroup("end", 2, lower=pf, upper=pf)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 500
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    opt.options["derivative_test"] = "first-order"

    sol = opt(optProb, sens=sens)
    print(sol)
    if sol.optInform["value"] != 0:
        print("Optimization failed")

    print("time", sol.xStar["tf"])

    knotPoints = create_unclamped_knot_points(0, sol.xStar["tf"][0], num_cont_points, 3)
    # print("knot points", knotPoints)
    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))
    return create_spline(knotPoints, controlPoints, spline_order)


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerHeading = 0

    startingLocation = np.array([-4.0, -4.0])
    endingLocation = np.array([4.0, 4.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)

    numControlPoints = 7
    splineOrder = 3
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)
    num_constraint_samples = 50

    pursuerRange = 1.0
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerTurnRadius = 0.1
    agentSpeed = 0.5

    velocity_constraints = (agentSpeed - 0.01, agentSpeed + 0.01)
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
        0.0,
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        pursuerTurnRadius,
        agentSpeed,
    )
    fig, ax = plt.subplots()
    plot_spline(
        spline,
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        pursuerTurnRadius,
        agentSpeed,
        ax,
    )

    plt.show()


if __name__ == "__main__":
    main()
