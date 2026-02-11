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
plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern
plt.rcParams["mathtext.rm"] = "serif"
# get rid of type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# set font size for title, axis labels, and legend, and tick labels
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10

# set backend to agg (only if running on remote server and saving images instead of viewing them)
# matplotlib.use("agg")


import bspline.spline_opt_tools as spline_opt_tools
import PEZ.pez as pez
import PEZ.pez_plotting as pez_plotting
import PLOT_COMMON.draw_mahalanobis as draw_mahalanobis

numSamplesPerInterval = 15

# disable gpu jax


# def plot_spline(spline, pursuerPosition, pursuerRange, pursuerCaptureRange,pez_limit,useProbabalistic):
def plot_spline(
    spline,
    agentPositionCov,
    agentHeadingVar,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerRangeVar,
    pursuerCaptureRange,
    pursuerCaptureRangeVar,
    pursuerSpeed,
    pursuerSpeedVar,
    agentSpeed,
    pez_constraint_limit,
    ax,
):
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, 1000, endpoint=True)
    # if pez_constraint_limit == 0.5:
    #     ax.set_title(f"Deterministic")
    # else:
    #     ax.set_title(f"PEZ Limit: {pez_constraint_limit}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")

    pos = spline(t)
    x = pos[:, 0]
    y = pos[:, 1]
    splineDot = spline.derivative()(t)
    xDot = splineDot[:, 0]
    yDot = splineDot[:, 1]
    agentHeadings = np.arctan2(yDot, xDot)
    plotPEZAlongSpline = False
    if plotPEZAlongSpline:
        pez = pez.probabalisticEngagementZoneVectorizedTemp(
            pos,
            agentPositionCov,
            agentHeadings,
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
            agentSpeed,
        )
        pezDeterministic = bez.inEngagementZoneJaxVectorized(
            pos,
            agentHeadings,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        )
        if pez_constraint_limit == 0.5:
            pez = pezDeterministic
            # flip cmap
            cmap = "viridis"
            cbarLabel = "dist-rho"
        else:
            cmap = "viridis"
            cbarLabel = "Engagement Zone Probability"
            # draw_mahalanobis.plotMahalanobisDistance(pursuerPosition, pursuerPositionCov, ax, fig)

        c = ax.scatter(x, y, c=pez, cmap=cmap, s=4)
        cbar = plt.colorbar(c, ax=ax, shrink=0.8)

        if pez_constraint_limit == 0.5:
            cbar.set_label("")
        cbar.set_label(cbarLabel)
    else:
        if pez_constraint_limit == 0.5:
            ax.plot(x, y, label=f"BEZ", linewidth=3)
        else:
            ax.plot(x, y, label=r"$\epsilon=$" + f"{pez_constraint_limit}", linewidth=3)

    # else:
    #     cbar.set_label("dist - rho")
    # control_points = spline.c
    # ax.plot(control_points[:, 0], control_points[:, 1], marker='o',linestyle = 'dashed',c = 'tab:gray')

    ax.set_aspect(1)
    # c = plt.Circle(pursuerPosition, pursuerRange + pursuerCaptureRange, fill=False)
    #
    plt.scatter(pursuerPosition[0], pursuerPosition[1], c="r")
    # ax.add_artist(c)
    plt.xlabel("X")
    plt.ylabel("Y")


# @partial(jit, static_argnums=(2,3,4,5,6,7,8,9,10))
@jit
def get_pez_along_spline(
    controlPoints,
    tf,
    agentPositionCov,
    agentHeadingVar,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerRangeVar,
    pursuerCaptureRange,
    pursuerCaptureRangeVar,
    pursuerSpeed,
    pursuerSpeedVar,
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
    return pez.probabalisticEngagementZoneVectorizedTemp(
        spline_opt_tools.evaluate_spline(
            controlPoints, knotPoints, numSamplesPerInterval
        ),
        agentPositionCov,
        agentHeadings,
        agentHeadingVar,
        pursuerPosition,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
    )


def compute_spline_constraints(
    controlPoints,
    knotPoints,
    agentPositionCov,
    agentHeadingVar,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerRangeVar,
    pursuerCaptureRange,
    pursuerCaptureRangeVar,
    pursuerSpeed,
    pursuerSpeedVar,
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

    pez_constraint = pez.probabalisticEngagementZoneVectorizedTemp(
        pos,
        agentPositionCov,
        agentHeadings,
        agentHeadingVar,
        pursuerPosition,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
    )

    return velocity, turn_rate, curvature, pez_constraint, pos


def create_spline(knotPoints, controlPoints, order):
    spline = BSpline(knotPoints, controlPoints, order)
    return spline


dPezDControlPoints = jacfwd(get_pez_along_spline, argnums=0)
dPezDtf = jacfwd(get_pez_along_spline, argnums=1)


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
    pez_constraint_limit,
    agentPositionCov,
    agentHeadingVar,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerRangeVar,
    pursuerCaptureRange,
    pursuerCaptureRangeVar,
    pursuerSpeed,
    pursuerSpeedVar,
    agentSpeed,
    useProbabalistic,
    left=True,
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

        velocity, turn_rate, curvature, pez, pos = compute_spline_constraints(
            controlPoints,
            knotPoints,
            agentPositionCov,
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
            agentSpeed,
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
        dPezDControlPointsVal = dPezDControlPoints(
            controlPoints,
            tf,
            agentPositionCov,
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
            agentSpeed,
        )
        dPezDtfVal = dPezDtf(
            controlPoints,
            tf,
            agentPositionCov,
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
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
        funcsSens["pez"] = {"control_points": dPezDControlPointsVal, "tf": dPezDtfVal}

        return funcsSens, False

    # num_constraint_samples = numSamplesPerInterval*(num_cont_points-2)-2

    optProb = Optimization("path optimization", objfunc)

    tf_initial = 1.0
    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, tf_initial, num_cont_points, 3
    )
    if not left:
        x0 = rect_bottom_and_right(p0, pf, num_cont_points).flatten()
    else:
        x0 = rect_left_and_top(p0, pf, num_cont_points).flatten()

    print("velocity constraints", velocity_constraints)
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
    optProb.addConGroup(
        "pez", num_constraint_samples, lower=None, upper=pez_constraint_limit
    )
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
    # opt.options['derivative_test'] = 'first-order'

    sol = opt(optProb, sens=sens)
    if sol.optInform["value"] != 0:
        print("Optimization failed")

    print("time", sol.xStar["tf"])

    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, sol.xStar["tf"][0], num_cont_points, 3
    )
    # print("knot points", knotPoints)
    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))
    return create_spline(knotPoints, controlPoints, spline_order), sol.xStar["tf"][0]


def mc_spline_evaluation(
    spline,
    num_mc_runs,
    num_samples,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerRangeVar,
    pursuerCaptureRange,
    pursuerCaptureRangeVar,
    pursuerSpeed,
    pursuerSpeedVar,
    agentSpeed,
):
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, num_samples, endpoint=True)
    num_bez_violations = 0
    pos = spline(t)
    d1 = spline.derivative()(t)
    agentHeadins = np.arctan2(d1[:, 1], d1[:, 0])
    # for j in tqdm(range(num_mc_runs)):
    for j in range(num_mc_runs):
        pursuerPositionTemp = np.random.multivariate_normal(
            pursuerPosition.squeeze(), pursuerPositionCov
        ).reshape((-1, 1))
        pursuerRangeTemp = np.random.normal(pursuerRange, np.sqrt(pursuerRangeVar))
        pursuerCaptureRangeTemp = np.random.normal(
            pursuerCaptureRange, np.sqrt(pursuerCaptureRangeVar)
        )

        pursuerSpeedTemp = np.random.normal(pursuerSpeed, np.sqrt(pursuerSpeedVar))
        while pursuerSpeedTemp < agentSpeed:
            pursuerSpeedTemp = np.random.normal(pursuerSpeed, np.sqrt(pursuerSpeedVar))

        ez = bez.inEngagementZoneJaxVectorized(
            pos,
            agentHeadins,
            pursuerPositionTemp,
            pursuerRangeTemp,
            pursuerCaptureRangeTemp,
            pursuerSpeedTemp,
            agentSpeed,
        )
        if np.any(ez < 0) or np.isnan(ez).any():
            num_bez_violations += 1
            #

    return num_bez_violations / num_mc_runs


def bez_pspline_path(
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
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    pez_constraint_limit = 0.5
    agentPositionCov = np.array([[0.0, 0], [0, 0.0]])
    agentHeadingVar = 0.0
    pursuerPositionCov = np.array([[0.05, -0.06], [-0.06, 0.25]])
    pursuerRangeVar = 0.1
    pursuerCaptureRangeVar = 0.1
    pursuerSpeedVar = 0.1
    useProbabalistic = False
    leftSpline, leftTf = optimize_spline_path(
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
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
        useProbabalistic,
        left=True,
    )
    rightSpline, rightTf = optimize_spline_path(
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
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
        useProbabalistic,
        left=False,
    )
    if rightTf < leftTf:
        return rightSpline, rightTf
    else:
        return leftSpline, leftTf


def main():
    agentPositionCov = np.array([[0.0, 0], [0, 0.0]])
    agentHeadingVar = 0.0
    pursuerPosition = np.array([0.0, 0.0])

    startingLocation = np.array([-4.0, -4.0])
    endingLocation = np.array([4.0, 4.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)

    numControlPoints = 16
    splineOrder = 3
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)
    num_constraint_samples = 50
    # pez_constraint_limit_list = [.1,.2,.3,.4]
    # pez_constraint_limit_list = [.01,0.05,.1,.2,.3,.4,.5]
    pez_constraint_limit_list = [0.01, 0.05, 0.25, 0.5]
    pez_constraint_limit_list = [0.5]

    pursuerPositionCov = np.array([[0.05, -0.06], [-0.06, 0.25]])
    # pursuerPositionCov = np.array([[0.1, 0], [0, 0.1]])
    pursuerRange = 1.0
    pursuerRangeVar = 0.1
    # pursuerRangeVar = 0.0
    pursuerCaptureRange = 0.1
    pursuerCaptureRangeVar = 0.02
    # pursuerCaptureRangeVar = 0.00
    pursuerSpeed = 2.0
    # pursuerSpeedVar = 0.2
    pursuerSpeedVar = 0.0
    agentSpeed = 0.5
    velocity_constraints = (0, agentSpeed + 0.01)

    num_mc_runs = 10000

    useProbabalistic = True

    fig, ax = plt.subplots(figsize=(6, 6))
    for pez_constraint_limit in pez_constraint_limit_list:
        print("PEZ Constraint Limit: ", pez_constraint_limit)
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
            pez_constraint_limit,
            agentPositionCov,
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
            agentSpeed,
            useProbabalistic,
        )
        # bez_fail_percentage = mc_spline_evaluation(
        #     spline,
        #     num_mc_runs,
        #     1000,
        #     pursuerPosition,
        #     pursuerPositionCov,
        #     pursuerRange,
        #     pursuerRangeVar,
        #     pursuerCaptureRange,
        #     pursuerCaptureRangeVar,
        #     pursuerSpeed,
        #     pursuerSpeedVar,
        #     agentSpeed,
        # )
        # print("bez_fail_percentage", bez_fail_percentage)

        # plot_constraints(spline, velocity_constraints, turn_rate_constraints, curvature_constraints, pez_constraint_limit, useProbabalistic)
        plot_spline(
            spline,
            agentPositionCov,
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
            agentSpeed,
            pez_constraint_limit,
            ax,
        )
    draw_mahalanobis.plotMahalanobisDistance(
        pursuerPosition, pursuerPositionCov, ax, fig, plotColorbar=False
    )
    ax.legend()
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks(np.arange(-4, 5, 1))
    ax.set_title("Linear PEZ")

    # fig, axes = plt.subplots(2, 3, layout="constrained")
    #
    # for pez_constraint_limit in pez_constraint_limit_list:
    #     axis = axes.flatten()[pez_constraint_limit_list.index(pez_constraint_limit)]
    #     print("PEZ Constraint Limit: ", pez_constraint_limit)
    #     spline = optimize_spline_path(
    #         startingLocation,
    #         endingLocation,
    #         initialVelocity,
    #         numControlPoints,
    #         splineOrder,
    #         velocity_constraints,
    #         turn_rate_constraints,
    #         curvature_constraints,
    #         num_constraint_samples,
    #         pez_constraint_limit,
    #         agentPositionCov,
    #         agentHeadingVar,
    #         pursuerPosition,
    #         pursuerPositionCov,
    #         pursuerRange,
    #         pursuerRangeVar,
    #         pursuerCaptureRange,
    #         pursuerCaptureRangeVar,
    #         pursuerSpeed,
    #         pursuerSpeedVar,
    #         agentSpeed,
    #         useProbabalistic,
    #     )
    #     # plot_constraints(spline, velocity_constraints, turn_rate_constraints, curvature_constraints, pez_constraint_limit, useProbabalistic)
    #     plot_spline(
    #         spline,
    #         agentPositionCov,
    #         agentHeadingVar,
    #         pursuerPosition,
    #         pursuerPositionCov,
    #         pursuerRange,
    #         pursuerRangeVar,
    #         pursuerCaptureRange,
    #         pursuerCaptureRangeVar,
    #         pursuerSpeed,
    #         pursuerSpeedVar,
    #         agentSpeed,
    #         pez_constraint_limit,
    #         axis,
    #     )
    #     # bez_fail_percentage = mc_spline_evaluation(spline, num_mc_runs, 200, pursuerPosition, pursuerPositionCov, pursuerRange,pursuerRangeVar, pursuerCaptureRange, pursuerCaptureRangeVar, pursuerSpeed,pursuerSpeedVar, agentSpeed)
    #     # print("BEZ Fail Percentage: ", bez_fail_percentage)
    #
    # labels = ["a)", "b)", "c)", "d)", "e)", "f)"]
    # for i, ax in enumerate(axes.flat):
    #     # Positioning the label in the top-left corner of each subplot
    #     ax.text(
    #         0.05,
    #         0.9,
    #         labels[i],
    #         transform=ax.transAxes,
    #         fontsize=16,
    #         fontweight="bold",
    #         va="top",
    #         color="red",
    #     )
    # useProbabalistic = False
    # spline = optimize_spline_path(
    #     startingLocation,
    #     endingLocation,
    #     initialVelocity,
    #     numControlPoints,
    #     splineOrder,
    #     velocity_constraints,
    #     turn_rate_constraints,
    #     curvature_constraints,
    #     num_constraint_samples,
    #     pez_constraint_limit,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerRange,
    #     pursuerCaptureRange,
    #     pursuerSpeed,
    #     agentSpeed,
    #     useProbabalistic,
    # )
    # plot_constraints(
    #     spline,
    #     velocity_constraints,
    #     turn_rate_constraints,
    #     curvature_constraints,
    #     pez_constraint_limit,
    #     useProbabalistic,
    # )
    # plot_spline(
    #     spline,
    #     pursuerPosition,
    #     pursuerRange,
    #     pursuerCaptureRange,
    #     pez_constraint_limit,
    #     useProbabalistic,
    # )

    # bez_fail_percentage = mc_spline_evaluation(
    #     spline,
    #     num_mc_runs,
    #     num_constraint_samples,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerRange,
    #     pursuerCaptureRange,
    #     pursuerSpeed,
    #     agentSpeed,
    # )
    # print("BEZ Fail Percentage: ", bez_fail_percentage)

    plt.show()


def animate_ez():
    agentPositionCov = np.array([[0.0, 0], [0, 0.0]])
    agentHeadingVar = 0.0
    pursuerPosition = np.array([0.0, 0.0])

    startingLocation = np.array([-2.0, -2.0])
    endingLocation = np.array([2.0, 2.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)

    numControlPoints = 14
    splineOrder = 3
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)
    num_constraint_samples = 50
    # pez_constraint_limit_list = [.1,.2,.3,.4]
    # pez_constraint_limit_list = [.01,0.05,.1,.2,.3,.4,.5]
    pez_constraint_limit_list = [0.5]
    # pez_constraint_limit_list = [.01]

    pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
    # pursuerPositionCov = np.array([[0.0,0],[0,0.0]])
    pursuerRange = 1.5
    pursuerRangeVar = 0.1
    # pursuerRangeVar = 0.0
    pursuerCaptureRange = 0.1
    pursuerCaptureRangeVar = 0.02
    # pursuerCaptureRangeVar = 0.00
    pursuerSpeed = 2.0
    # pursuerSpeedVar = 0.2
    pursuerSpeedVar = 0.0
    agentSpeed = 0.5
    # velocity_constraints = (0,1.0)
    velocity_constraints = (agentSpeed - 0.01, agentSpeed + 0.01)

    useProbabalistic = False

    pez_constraint_limit = pez_constraint_limit_list[0]
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
        pez_constraint_limit,
        agentPositionCov,
        agentHeadingVar,
        pursuerPosition,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
        useProbabalistic,
    )

    # plot_constraints(spline, velocity_constraints, turn_rate_constraints, curvature_constraints, pez_constraint_limit, useProbabalistic)
    currentTime = 0
    dt = 0.01
    finalTime = spline.t[-1 - spline.k]
    ind = 0
    while currentTime < finalTime:
        fig, ax = plt.subplots()
        pdot = spline.derivative(1)(currentTime)
        currentPosition = spline(currentTime)
        currentHeading = np.arctan2(pdot[1], pdot[0])
        pez_plotting.plotEngagementZone(
            currentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
            ax,
        )
        # plot triangle at evader position with heading of evader
        plot_spline(
            spline,
            agentPositionCov,
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
            agentSpeed,
            pez_constraint_limit,
            ax,
        )
        plt.arrow(
            currentPosition[0],
            currentPosition[1],
            1e-6 * np.cos(currentHeading),  # essentially zero-length tail
            1e-6 * np.sin(currentHeading),
            head_width=0.2,
            head_length=0.25,
            width=0,  # no line
            fc="blue",
            ec="blue",
            zorder=5,
        )
        # plt.arrow(
        #     currentPosition[0],
        #     currentPosition[1],
        #     0.2 * np.cos(currentHeading),
        #     0.2 * np.sin(currentHeading),
        #     head_width=0.3,
        #     head_length=0.3,
        #     width=0.00001,
        #     fc="blue",
        #     ec="blue",
        #     zorder=5,
        # )
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


def animate_pez():
    agentPositionCov = np.array([[0.0, 0], [0, 0.0]])
    agentHeadingVar = 0.0
    pursuerPosition = np.array([0.0, 0.0])

    startingLocation = np.array([-4.0, -4.0])
    endingLocation = np.array([4.0, 4.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)

    numControlPoints = 14
    splineOrder = 3
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)
    num_constraint_samples = 50
    # pez_constraint_limit_list = [.1,.2,.3,.4]
    # pez_constraint_limit_list = [.01,0.05,.1,.2,.3,.4,.5]
    # pez_constraint_limit_list = [0.5]
    pez_constraint_limit_list = [0.01]

    # pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
    pursuerPositionCov = np.array([[0.1, 0], [0, 0.1]])
    pursuerPositionCov = np.array([[0.05, -0.06], [-0.06, 0.25]])
    pursuerRange = 1.0
    pursuerRangeVar = 0.1
    # pursuerRangeVar = 0.0
    pursuerCaptureRange = 0.1
    pursuerCaptureRangeVar = 0.02
    # pursuerCaptureRangeVar = 0.00
    pursuerSpeed = 2.0
    # pursuerSpeedVar = 0.2
    pursuerSpeedVar = 0.0
    agentSpeed = 0.5
    # velocity_constraints = (0,1.0)
    velocity_constraints = (agentSpeed - 0.01, agentSpeed + 0.01)

    useProbabalistic = True

    pez_constraint_limit = pez_constraint_limit_list[0]
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
        pez_constraint_limit,
        agentPositionCov,
        agentHeadingVar,
        pursuerPosition,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
        useProbabalistic,
    )

    # plot_constraints(spline, velocity_constraints, turn_rate_constraints, curvature_constraints, pez_constraint_limit, useProbabalistic)
    currentTime = 0
    dt = 0.2
    finalTime = spline.t[-1 - spline.k]
    ind = 0
    while currentTime < finalTime:
        print("Current Time: ", currentTime)
        fig, ax = plt.subplots()
        pdot = spline.derivative(1)(currentTime)
        currentPosition = spline(currentTime)
        currentHeading = np.arctan2(pdot[1], pdot[0])
        pez_plotting.plotProbablisticEngagementZone(
            agentPositionCov,
            currentHeading,
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
            agentSpeed,
            ax,
            levels=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
        )
        # plot triangle at evader position with heading of evader
        plot_spline(
            spline,
            agentPositionCov,
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
            agentSpeed,
            pez_constraint_limit,
            ax,
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
        pez_plotting.plotMahalanobisDistance(
            pursuerPosition, pursuerPositionCov, ax, fig
        )
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        fig.savefig(f"video/{ind}.png", dpi=300)
        ind += 1
        currentTime += dt
        plt.close(fig)


if __name__ == "__main__":
    main()
    # animate_pez()
