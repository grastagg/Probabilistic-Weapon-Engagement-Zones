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
import jax

# use agg
matplotlib.use("Agg")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


import CSPEZ.csbez as csbez
import CSPEZ.csbez_plotting as csbez_plotting

# import pez_path_planner
import PEZ.pez_path_planner as pez_path_planner
import PEZ.pez_plotting as pez_plotting
import PLOT_COMMON.draw_mahalanobis as draw_mahalanobis

import bspline.spline_opt_tools as spline_opt_tools


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
    # ez = csbez.in_dubins_engagement_zone(
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
    ax.plot(x, y, c="blue", linewidth=2)
    # c = ax.scatter(x, y, c=ez, s=4)
    # cbar = plt.colorbar(c, shrink=0.8)
    # cbar.ax.tick_params(labelsize=26)

    ax.set_aspect(1)
    # c = plt.Circle(pursuerPosition, pursuerRange + pursuerCaptureRange, fill=False)
    # plt.scatter(pursuerPosition[0], pursuerPosition[1], c="r")
    # ax.add_artist(c)
    plt.xlabel("X")
    plt.ylabel("Y")


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
    ez = csbez.in_dubins_engagement_zone(
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
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )

    turn_rate, velocity, agentHeadings = (
        spline_opt_tools.get_turn_rate_velocity_and_headings(
            controlPoints, knotPoints, numSamplesPerInterval
        )
    )

    curvature = turn_rate / velocity

    ez = csbez.in_dubins_engagement_zone(
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
        knotPoints = spline_opt_tools.create_unclamped_knot_points(
            0, tf, num_cont_points, 3
        )
        controlPoints = xDict["control_points"]
        funcs = {}
        funcs["start"] = spline_opt_tools.get_start_constraint(controlPoints)
        funcs["end"] = spline_opt_tools.get_end_constraint(controlPoints)
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

        dStartDControlPointsVal = spline_opt_tools.get_start_constraint_jacobian(
            controlPoints
        )
        dEndDControlPointsVal = spline_opt_tools.get_end_constraint_jacobian(
            controlPoints
        )

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

    tf_initial = 1.0
    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, tf_initial, num_cont_points, 3
    )

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
    # optProb.addConGroup(
    #     "turn_rate",
    #     num_constraint_samples,
    #     lower=turn_rate_constraints[0],
    #     upper=turn_rate_constraints[1],
    #     scale=1.0 / turn_rate_constraints[1],
    # )
    # optProb.addConGroup(
    #     "curvature",
    #     num_constraint_samples,
    #     lower=curvature_constraints[0],
    #     upper=curvature_constraints[1],
    #     scale=1.0 / curvature_constraints[1],
    # )
    optProb.addConGroup("ez", num_constraint_samples, lower=0.0, upper=None)
    optProb.addConGroup("start", 2, lower=p0, upper=p0)
    optProb.addConGroup("end", 2, lower=pf, upper=pf)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    print("TEST")
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 500
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    # opt.options["derivative_test"] = "first-order"

    sol = opt(optProb, sens=sens)
    if sol.optInform["value"] != 0:
        print("Optimization failed")

    print("time", sol.xStar["tf"])

    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, sol.xStar["tf"][0], num_cont_points, 3
    )
    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))
    return create_spline(knotPoints, controlPoints, spline_order)


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerHeading = np.pi / 2

    startingLocation = np.array([-4.0, 1.0])
    endingLocation = np.array([4.0, -1.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)
    initialVelocity = endingLocation - startingLocation

    numControlPoints = 7
    splineOrder = 3
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)
    num_constraint_samples = 50

    pursuerRange = 1.0
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerTurnRadius = 0.2
    agentSpeed = 1

    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity) * agentSpeed

    velocity_constraints = (agentSpeed - 0.01, agentSpeed + 0.01)
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
        splineDubins,
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        pursuerTurnRadius,
        agentSpeed,
        ax,
    )
    pez_plotting.plotEngagementZone(
        0.0,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        agentSpeed,
        ax,
    )
    csbez_plotting.plot_dubins_EZ(
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        pursuerTurnRadius,
        pursuerCaptureRadius,
        pursuerRange,
        0.0,
        agentSpeed,
        ax,
    )

    useProbabalistic = False
    pursuerPosition = np.array([[0.0], [0.0]])
    pez_constraint_limit = 0.5
    agentPositionCov = np.array([[0.0, 0], [0, 0.0]])
    agentHeadingVar = 0.0
    pursuerPositionCov = np.array([[0.2, 0], [0, 0.2]])
    pursuerRangeVar = 0.0
    pursuerCaptureRangeVar = 0.0
    pursuerSpeedVar = 0.0

    splineBEZ = pez_path_planner.optimize_spline_path(
        startingLocation,
        endingLocation,
        initialVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        pez_constraint_limit,
        agentPositionCov,
        agentHeadingVar,
        pursuerPosition,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRadius,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
        useProbabalistic,
    )

    pez_path_planner.plot_spline(
        splineBEZ,
        agentPositionCov,
        agentHeadingVar,
        pursuerPosition,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRadius,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
        pez_constraint_limit,
        ax,
    )

    plt.show()


def animate_spline_path():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerHeading = (-15.0 / 20.0) * np.pi

    startingLocation = np.array([-2.0, -2.0])
    endingLocation = np.array([2.0, 2.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)
    initialVelocity = endingLocation - startingLocation

    numControlPoints = 20
    splineOrder = 3
    turn_rate_constraints = (-1.0, 1.0)
    curvature_constraints = (-0.2, 0.2)
    num_constraint_samples = 50

    pursuerRange = 1.0
    # pursuerRangeVar = 0.1
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerTurnRadius = 0.2
    agentSpeed = 1

    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity) * agentSpeed

    velocity_constraints = (agentSpeed - 0.001, agentSpeed + 0.001)
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

    currentTime = 0
    dt = 0.05
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
            pursuerHeading,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
            pursuerTurnRadius,
            agentSpeed,
            ax,
        )
        csbez_plotting.plot_dubins_EZ(
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            pursuerTurnRadius,
            0.0,
            pursuerRange,
            currentHeading,
            agentSpeed,
            ax,
            alpha=1.0,
        )
        plt.arrow(
            currentPosition[0],
            currentPosition[1],
            1e-6 * np.cos(currentHeading),  # essentially zero-length tail
            1e-6 * np.sin(currentHeading),
            head_width=0.15,
            head_length=0.15,
            width=0,  # no line
            fc="blue",
            ec="blue",
            zorder=5,
        )
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        fig.savefig(f"video/{ind}.png", dpi=300)
        ind += 1
        currentTime += dt
        plt.close(fig)


if __name__ == "__main__":
    animate_spline_path()
    # main()
