from jax._src.config import int_env
import tqdm
from functools import partial
import sys
import time
import jax
from tabulate import tabulate


import getpass
from pyoptsparse import Optimization, OPT, IPOPT
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import learned_dubins_ez_path_planner


# get rid of type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# set font size for title, axis labels, and legend, and tick labels
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10
from pyDOE3 import lhs  # or pyDOE2
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = (
    "0.30"  # try 0.30; adjust if you run 2–3 workers
)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")  # avoid X11 ("Invalid MIT-MAGIC-COOKIE-1")
# GPU + headless settings BEFORE any heavy imports
# os.environ["QT_QPA_PLATFORM"] = "offscreen"
# os.environ.pop("DISPLAY", None)
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from scipy.optimize import minimize

import dubinsEZ
# import mp_worker

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax.config.update("jax_platform_name", "gpu")")
#

positionAndHeadingOnly = True
knownSpeed = True
interceptionOnBoundary = True
randomPath = False
noisyMeasurementsFlag = False
saveResults = True
plotAllFlag = False
planHPPath = True

dataDir = "results"
if not planHPPath:
    saveDir = ""
else:
    saveDir = "plannedHP/"
if interceptionOnBoundary:
    saveDir += "boundary/"
else:
    saveDir += "interior/"

if positionAndHeadingOnly:
    saveDir += "knownSpeedAndShape"
elif knownSpeed:
    saveDir += "knownSpeed"
else:
    saveDir += "unknownSpeed"
if noisyMeasurementsFlag:
    saveDir += "WithNoise"

print("saving data to: ", saveDir)

if positionAndHeadingOnly:
    parameterMask = np.array([True, True, True, False, False, False])
elif knownSpeed:
    parameterMask = np.array([True, True, True, False, True, True])
else:
    parameterMask = np.array([True, True, True, True, True, True])


def plot_low_priority_paths(
    startPositions, interceptedList, endPoints, pathHistories, ax
):
    ax.set_aspect("equal", adjustable="box")
    # ax.set_ylabel("Y", fontsize=34)
    # ax.tick_params(labelsize=18)
    # set x and y limits
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)

    interceptedCounter = 0
    survivedCounter = 0
    for i in range(len(startPositions)):
        pathHistory = pathHistories[i]
        if interceptedList[i]:
            ax.scatter(
                endPoints[i][0],
                endPoints[i][1],
                color="blue",
                marker="x",
                s=250,
                zorder=10000,
            )
            if interceptedCounter == 0:
                ax.scatter(
                    pathHistory[:, 0],
                    pathHistory[:, 1],
                    c="blue",
                    s=10,
                )
            else:
                ax.scatter(
                    pathHistory[:, 0],
                    pathHistory[:, 1],
                    c="blue",
                    s=10,
                )
            interceptedCounter += 1
        else:
            if survivedCounter == 0:
                ax.scatter(pathHistory[:, 0], pathHistory[:, 1], c="g", s=5)
            else:
                ax.scatter(
                    pathHistory[:, 0],
                    pathHistory[:, 1],
                    c="g",
                    s=5,
                )
            survivedCounter += 1


def plot_low_priority_paths_with_ez(
    headings,
    speeds,
    startPositions,
    interceptedList,
    endPoints,
    pathHistories,
    pursuerX,
    trueParams,
    ax,
):
    ax.set_aspect("equal", adjustable="box")
    for i in range(len(startPositions)):
        pathHistory = pathHistories[i]

        heading = headings[i]
        # ez = (
        #     dubinsEZ_from_pursuerX(
        #         pursuerX,
        #         pathHistory,
        #         heading * np.ones(len(pathHistory)),
        #         speeds[i],
        #         trueParams,
        #     )
        #     < 0
        # )
        ez = dubins_reachable_set_from_pursuerX(
            pursuerX,
            pathHistory,
            trueParams,
            verbose=False,
        )
        ez = ez < 0.0  # shape (T,)

        if interceptedList[i]:
            ax.scatter(
                endPoints[i][0],
                endPoints[i][1],
                color="red",
                marker="x",
            )
            c = ax.scatter(
                pathHistory[:, 0],
                pathHistory[:, 1],
                c=ez,
                s=5,
            )
            cb = plt.colorbar(c, ax=ax)
        else:
            c = ax.scatter(
                pathHistory[:, 0],
                pathHistory[:, 1],
                c=ez,
                vmin=0.0,
                vmax=1.0,
            )
    # ax.scatter(pursuerX[0], pursuerX[1], color="blue", marker="o")


def is_inside_region(point, xbound, ybounds):
    return xbound[0] <= point[0] <= xbound[1] and ybounds[0] <= point[1] <= ybounds[1]


def first_true_index_safe(boolean_array):
    idx = jnp.argmax(boolean_array)
    found = jnp.any(boolean_array)
    return jnp.where(found, idx, -1)


def simulate_trajectory_fn(startPosition, heading, speed, tmax, numPoints):
    currentPosition = jnp.array(startPosition)
    t = jnp.linspace(0.0, tmax, numPoints)  # shape (T,)
    direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])  # shape (2,)
    displacement = t[:, None] * speed * direction  # shape (T, 2)
    pathHistory = currentPosition + displacement  # shape (T, 2)
    headings = heading * jnp.ones(numPoints)  # shape (T,)
    return pathHistory, headings


def find_interception_point_and_time(
    speed, pursuerSpeed, direction, pursuerRange, t, pathHistory, firstTrueIndex
):
    intercepted = True
    speedRatio = speed / pursuerSpeed

    interceptionPoint = (
        pathHistory[firstTrueIndex] + speedRatio * pursuerRange * direction
    )
    interceptionTime = t[firstTrueIndex] + pursuerRange / pursuerSpeed
    return (intercepted, interceptionPoint, interceptionTime)


# @partial(jax.jit, static_argnames=("tmax", "numPoints", "numSimulationPoints"))
def send_low_priority_agent_interioir(
    startPosition,
    heading,
    speed,
    pursuerPosition,
    pursuerHeading,
    minimumTurnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    tmax,
    numPoints,
    numSimulationPoints=500000,
    rng=None,
):
    t = np.linspace(0.0, tmax, numSimulationPoints)  # shape (T,)
    pathHistory, headings = simulate_trajectory_fn(
        startPosition, heading, speed, tmax, numSimulationPoints
    )
    RS = dubinsEZ.in_dubins_reachable_set(
        pursuerPosition, pursuerHeading, minimumTurnRadius, pursuerRange, pathHistory
    )
    inRS = RS < 0.0  # shape (T,)
    EZ = dubinsEZ.in_dubins_engagement_zone_agumented(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        0.0,
        pursuerRange,
        pursuerSpeed,
        pathHistory,
        headings,
        speed,
    )
    inEZ = EZ < 0.0  # shape (T,)
    intercepted = np.any(inEZ)  # shape (T,)

    # if firstTrueIndex != -1:
    if intercepted:
        # nz = np.flatnonzero(inRS)  # 1D indices of True
        # idx = rng.choice(nz)
        # interceptionPoint = pathHistory[idx]
        # interceptionTime = t[idx]
        # pathHistoryNew, headings = simulate_trajectory_fn(
        #     startPosition, heading, speed, interceptionTime, numPoints
        # )
        # direction = np.array([np.cos(heading), np.sin(heading)])  # shape (2,)
        # pursuerDistTraveled = dubinsEZ.find_shortest_dubins_path(
        #     pursuerPosition, pursuerHeading, interceptionPoint, minimumTurnRadius
        # )
        # # t_p = pursuerDistTraveled / pursuerSpeed
        # t_p = pursuerRange / pursuerSpeed
        # interceptionPointEZ = interceptionPoint - t_p * speed * direction
        #
        # interceptionTimeEZ = interceptionTime - t_p
        #
        nz = np.flatnonzero(inRS)  # 1D indices of True
        idx = rng.choice(nz)
        interceptionPoint = pathHistory[idx]
        interceptionTime = t[idx]
        pathHistoryNew, headings = simulate_trajectory_fn(
            startPosition, heading, speed, interceptionTime, numPoints
        )
        direction = np.array([np.cos(heading), np.sin(heading)])  # shape (2,)
        pursuerDistTraveled = dubinsEZ.find_shortest_dubins_path(
            pursuerPosition, pursuerHeading, interceptionPoint, minimumTurnRadius
        )
        t_p = pursuerDistTraveled / pursuerSpeed
        interceptionPointEZ = interceptionPoint - t_p * speed * direction

        interceptionTimeEZ = interceptionTime - t_p
    else:
        interceptionPoint = pathHistory[-1]
        interceptionTime = t[-1]
        interceptionPointEZ = pathHistory[-1]
        interceptionTimeEZ = t[-1]

        intercepted = False
        pathHistoryNew, headings = simulate_trajectory_fn(
            startPosition, heading, speed, tmax, numPoints
        )

    t = np.linspace(0.0, interceptionTime, numPoints)  # shape (T,)

    return (
        t,
        intercepted,
        interceptionPoint,
        interceptionTime,
        pathHistoryNew,
        interceptionPointEZ,
        interceptionTimeEZ,
    )


@partial(jax.jit, static_argnames=("numPoints", "numSimulationPoints"))
def send_low_priority_agent_expected_intercept(
    startPosition,
    heading,
    speed,
    pursuerPosition,
    pursuerHeading,
    minimumTurnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    tmax,
    numPoints,
    numSimulationPoints=500000,
):
    t = jnp.linspace(0.0, tmax, numSimulationPoints)  # shape (T,)
    simulationDt = t[1] - t[0]
    direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])  # shape (2,)
    pathHistory, headings = simulate_trajectory_fn(
        startPosition, heading, speed, tmax, numSimulationPoints
    )
    RS = dubinsEZ.in_dubins_reachable_set(
        pursuerPosition, pursuerHeading, minimumTurnRadius, pursuerRange, pathHistory
    )
    inRS = RS < 0.0  # shape (T,)
    inRSFlag = jnp.any(inRS)
    EZ = dubinsEZ.in_dubins_engagement_zone_agumented(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        0.0,
        pursuerRange,
        pursuerSpeed,
        pathHistory,
        headings,
        speed,
    )
    inEZ = EZ < 0.0  # shape (T,)
    firstTrueIndexEZ = first_true_index_safe(inEZ)

    # if firstTrueIndex != -1:
    def interception_fn():
        intercepted = True
        w = inRS.astype(pathHistory.dtype)  # (T,) float {0.,1.}

        num = (pathHistory * w[:, None]).sum(axis=0)  # (D,)
        den = w.sum()  # scalar

        # Only divide if there is at least one True; else fall back (e.g., mean of all points).
        interceptionPoint = jax.lax.cond(
            den > 0,
            lambda _: num / den,
            lambda _: jnp.mean(pathHistory, axis=0),  # choose the fallback you want
            operand=None,
        )
        interceptionTime = jnp.linalg.norm(startPosition - interceptionPoint) / speed

        # tNew = np.linspace(0.0, interceptionTime, numPoints)  # shape (T,)
        pathHistoryNew, headings = simulate_trajectory_fn(
            startPosition, heading, speed, interceptionTime, numPoints
        )

        deltaXEZ = EZ[firstTrueIndexEZ - 1] - EZ[firstTrueIndexEZ]
        zeroCrossingEZ = (EZ[firstTrueIndexEZ - 1]) / deltaXEZ
        interceptionPointEZ = pathHistory[firstTrueIndexEZ - 1] + zeroCrossingEZ * (
            speed * simulationDt * direction
        )
        interceptionTimeEZ = t[firstTrueIndexEZ - 1] + zeroCrossingEZ * simulationDt
        return (
            intercepted,
            interceptionPoint,
            interceptionTime,
            pathHistoryNew,
            interceptionPointEZ,
            interceptionTimeEZ,
        )

    # else:
    def no_interception_fn():
        interceptionPoint = pathHistory[-1]
        interceptionTime = t[-1]
        interceptionPointEZ = pathHistory[-1]
        interceptionTimeEZ = t[-1]

        intercepted = False
        pathHistoryNew, headings = simulate_trajectory_fn(
            startPosition, heading, speed, tmax, numPoints
        )
        return (
            intercepted,
            interceptionPoint,
            interceptionTime,
            pathHistoryNew,
            interceptionPointEZ,
            interceptionTimeEZ,
        )

    (
        intercepted,
        interceptionPoint,
        interceptionTime,
        pathHistory,
        interceptionPointEZ,
        interceptionTimeEZ,
    ) = jax.lax.cond(
        inRSFlag,
        interception_fn,
        no_interception_fn,
    )
    t = jnp.linspace(0.0, interceptionTime, numPoints)  # shape (T,)

    return (
        t,
        intercepted,
        interceptionPoint,
        interceptionTime,
        pathHistory,
        interceptionPointEZ,
        interceptionTimeEZ,
    )


@partial(jax.jit, static_argnames=("tmax", "numPoints", "numSimulationPoints"))
def send_low_priority_agent_boundary(
    startPosition,
    heading,
    speed,
    pursuerPosition,
    pursuerHeading,
    minimumTurnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    tmax,
    numPoints,
    numSimulationPoints=500000,
):
    t = jnp.linspace(0.0, tmax, numSimulationPoints)  # shape (T,)
    simulationDt = t[1] - t[0]
    direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])  # shape (2,)
    pathHistory, headings = simulate_trajectory_fn(
        startPosition, heading, speed, tmax, numSimulationPoints
    )
    RS = dubinsEZ.in_dubins_reachable_set(
        pursuerPosition, pursuerHeading, minimumTurnRadius, pursuerRange, pathHistory
    )
    inRS = RS < 0.0  # shape (T,)
    firstTrueIndex = first_true_index_safe(inRS)
    EZ = dubinsEZ.in_dubins_engagement_zone_agumented(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        0.0,
        pursuerRange,
        pursuerSpeed,
        pathHistory,
        headings,
        speed,
    )
    inEZ = EZ < 0.0  # shape (T,)
    firstTrueIndexEZ = first_true_index_safe(inEZ)

    # if firstTrueIndex != -1:
    def interception_fn():
        intercepted = True
        deltaX = RS[firstTrueIndex - 1] - RS[firstTrueIndex]
        zeroCrossing = (RS[firstTrueIndex - 1]) / deltaX
        interceptionPoint = pathHistory[firstTrueIndex - 1] + zeroCrossing * (
            speed * simulationDt * direction
        )
        interceptionTime = t[firstTrueIndex - 1] + zeroCrossing * simulationDt
        # tNew = np.linspace(0.0, interceptionTime, numPoints)  # shape (T,)
        pathHistoryNew, headings = simulate_trajectory_fn(
            startPosition, heading, speed, interceptionTime, numPoints
        )

        pursuerDistTraveled = dubinsEZ.find_shortest_dubins_path(
            pursuerPosition, pursuerHeading, interceptionPoint, minimumTurnRadius
        )
        t_p = pursuerDistTraveled / pursuerSpeed
        interceptionPointEZ = interceptionPoint - t_p * speed * direction
        interceptionTimeEZ = interceptionTime - t_p

        # deltaXEZ = EZ[firstTrueIndexEZ - 1] - EZ[firstTrueIndexEZ]
        # zeroCrossingEZ = (EZ[firstTrueIndexEZ - 1]) / deltaXEZ
        # interceptionPointEZ = pathHistory[firstTrueIndexEZ - 1] + zeroCrossingEZ * (
        #     speed * simulationDt * direction
        # )
        # interceptionTimeEZ = t[firstTrueIndexEZ - 1] + zeroCrossingEZ * simulationDt
        return (
            intercepted,
            interceptionPoint,
            interceptionTime,
            pathHistoryNew,
            interceptionPointEZ,
            interceptionTimeEZ,
        )

    # else:
    def no_interception_fn():
        interceptionPoint = pathHistory[-1]
        interceptionTime = t[-1]
        interceptionPointEZ = pathHistory[-1]
        interceptionTimeEZ = t[-1]

        intercepted = False
        pathHistoryNew, headings = simulate_trajectory_fn(
            startPosition, heading, speed, tmax, numPoints
        )
        return (
            intercepted,
            interceptionPoint,
            interceptionTime,
            pathHistoryNew,
            interceptionPointEZ,
            interceptionTimeEZ,
        )

    (
        intercepted,
        interceptionPoint,
        interceptionTime,
        pathHistory,
        interceptionPointEZ,
        interceptionTimeEZ,
    ) = jax.lax.cond(
        firstTrueIndex != -1,
        interception_fn,
        no_interception_fn,
    )
    t = jnp.linspace(0.0, interceptionTime, numPoints)  # shape (T,)

    return (
        t,
        intercepted,
        interceptionPoint,
        interceptionTime,
        pathHistory,
        interceptionPointEZ,
        interceptionTimeEZ,
    )


@jax.jit
def pursuerX_to_params_position_and_heading(X, trueParams):
    pursuerPosition = X[0:2]
    pursuerHeading = X[2]
    pursuerSpeed = trueParams[3]
    minimumTurnRadius = trueParams[4]
    pursuerRange = trueParams[5]
    return (
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
    )


def pursuerX_to_params_no_speed(X, trueParams):
    pursuerPosition = X[0:2]
    pursuerHeading = X[2]
    pursuerSpeed = trueParams[3]
    minimumTurnRadius = X[3]
    pursuerRange = X[4]
    return (
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
    )


def pursuerX_to_params_all(X, trueParams):
    pursuerPosition = X[0:2]
    pursuerHeading = X[2]
    pursuerSpeed = X[3]
    minimumTurnRadius = X[4]
    pursuerRange = X[5]
    return (
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
    )


if positionAndHeadingOnly:
    pursuerX_to_params = pursuerX_to_params_position_and_heading
elif knownSpeed:
    pursuerX_to_params = pursuerX_to_params_no_speed
else:
    pursuerX_to_params = pursuerX_to_params_all


def dubinsEZ_from_pursuerX(
    pursuerX,
    pathHistory,
    headings,
    speed,
    trueParams,
):
    pursuerPosition, pursuerHeading, pursuerSpeed, minimumTurnRadius, pursuerRange = (
        pursuerX_to_params(pursuerX, trueParams)
    )
    ez = dubinsEZ.in_dubins_engagement_zone_agumented(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        0.0,
        pursuerRange,
        pursuerSpeed,
        pathHistory,
        headings,
        speed,
    )
    return ez


def dubins_reachable_set_from_pursuerX(
    pursuerX, goalPosition, trueParams, verbose=False
):
    pursuerPosition, pursuerHeading, pursuerSpeed, minimumTurnRadius, pursuerRange = (
        pursuerX_to_params(pursuerX, trueParams)
    )
    rs = dubinsEZ.in_dubins_reachable_set_augmented(
        pursuerPosition, pursuerHeading, minimumTurnRadius, pursuerRange, goalPosition
    )
    return rs


def smooth_min(x, alpha=10.0):
    return -jnp.log(jnp.sum(jnp.exp(-alpha * x))) / alpha


def activation(x):
    # return jax.nn.relu(x)  # ReLU activation function
    return jax.nn.relu(x) ** 2  # ReLU activation function


def activation_smooth(z, tau=0.1):
    """Smooth approximation to ReLU/hinge: log(1 + exp(z/tau)) * tau"""
    return tau * jax.nn.softplus(z / tau)


@jax.jit
def learning_loss_on_boundary_function_single(
    pursuerX,
    heading,
    speed,
    intercepted,
    pathTime,
    pathHistory,
    interceptedPoint,
    ezPoint,
    ezTime,
    trueParams,
    interceptedPathWeight=1.0,
    flattenLearingLossAmount=0.0,
    verbose=False,
):
    rsEnd = dubins_reachable_set_from_pursuerX(
        pursuerX,
        jnp.array([interceptedPoint]),
        trueParams,
    ).squeeze()
    rsAll = dubins_reachable_set_from_pursuerX(pursuerX, pathHistory, trueParams)

    interceptedLossRSEnd = activation(rsEnd - flattenLearingLossAmount) + activation(
        -rsEnd - flattenLearingLossAmount
    )
    interceptedLossTrajectory = jnp.max(
        activation(-rsAll[:-4] - flattenLearingLossAmount)
    )
    # interceptedLossTrajectory = activation(
    #     -jnp.min(rsAll[:-4]) - flattenLearingLossAmount
    # )
    interceptedLossRS = (
        interceptedPathWeight * interceptedLossTrajectory + interceptedLossRSEnd
    )

    survivedLossRS = jnp.max(
        activation(-rsAll - flattenLearingLossAmount)
    )  # loss if survived in RS

    lossRS = jax.lax.cond(
        intercepted, lambda: interceptedLossRS, lambda: survivedLossRS
    )

    return lossRS


def time_loss_with_flatten(predicted_ezTime, ezTime, flatten_amount):
    """
    Quadratic penalty for time difference, with a flat zone around zero
    of width `flatten_amount`.
    """
    diff = jnp.abs(predicted_ezTime - ezTime) - flatten_amount
    return jax.nn.relu(diff)
    return jnp.square(jax.nn.relu(diff))


@jax.jit
def learning_loss_on_boundary_function_single_EZ(
    pursuerX,
    heading,
    speed,
    intercepted,
    pathTime,
    pathHistory,
    interceptedPoint,
    ezPoint,
    ezTime,
    trueParams,
    interceptedPathWeight=1.0,
    flattenLearingLossAmount=0.0,
    verbose=False,
):
    # headings = heading * jnp.ones(pathHistory.shape[0])
    # ezFirst = dubinsEZ_from_pursuerX(
    #     pursuerX, jnp.array([ezPoint]), jnp.array([headings[-1]]), speed, trueParams
    # ).squeeze()
    # #
    #
    # ezAll = dubinsEZ_from_pursuerX(pursuerX, pathHistory, headings, speed, trueParams)
    #
    # outEZ = pathTime < ezTime - 0.0 * 0.001
    #
    # interceptedLossEZFirst = activation(
    #     ezFirst - flattenLearingLossAmount
    # ) + activation(-ezFirst - flattenLearingLossAmount)
    #
    # interceptedLossTrajectory = jnp.where(
    #     outEZ, activation(-ezAll - flattenLearingLossAmount), 0
    # )
    # interceptedLossTrajectory = jnp.max(interceptedLossTrajectory)

    predPursuerDistanceTraveled = dubinsEZ.find_shortest_dubins_path(
        pursuerX[0:2], pursuerX[2], interceptedPoint, pursuerX[4]
    )
    pred_ezTime = pathTime[-1] - (predPursuerDistanceTraveled) / pursuerX[3]
    # pred_ezTime = (
    #     pathTime[-1] - (ezFirst + pursuerX[5]) / pursuerX[3]
    # )  # estimate time of EZ crossing

    # pred_ezTime = pathTime[-1] - jnp.linalg.norm(ezPoint - pathHistory[-1]) / speed
    # time_loss = time_loss_with_flatten(
    #     pred_ezTime, ezTime, flattenLearingLossAmount / pursuerX[3]
    # )
    # pred_ezTime = pathTime[-1] - pursuerX[5] / pursuerX[3]
    time_loss = time_loss_with_flatten(
        # pred_ezTime,
        # ezTime,
        # 0.001 * 2.3,
        pred_ezTime,
        ezTime,
        flattenLearingLossAmount / pursuerX[3] + 0.001 * 2.5,
    )

    def verbose_fn():
        jax.debug.print(
            "pred_ezTime: {}, ezTime: {}, time_loss: {}, predPursuerDistanceTraveled: {}",
            pred_ezTime,
            ezTime,
            time_loss,
            predPursuerDistanceTraveled,
        )

    jax.lax.cond(verbose, verbose_fn, lambda: None)

    lossEZ = time_loss
    # lossEZ = 0.0

    # interceptedLossEZ = (
    #     interceptedPathWeight * interceptedLossTrajectory
    #     + interceptedLossEZFirst
    #     + time_loss
    # )
    #
    # survivedLossEZ = jnp.max(
    #     activation(-ezAll - flattenLearingLossAmount)
    # )  # loss if survived in EZ
    #
    lossEZ = jax.lax.cond(intercepted, lambda: time_loss, lambda: 0.0)
    lossRS = learning_loss_on_boundary_function_single(
        pursuerX,
        heading,
        speed,
        intercepted,
        pathTime,
        pathHistory,
        interceptedPoint,
        ezPoint,
        ezTime,
        trueParams,
        interceptedPathWeight,
        flattenLearingLossAmount,
    )

    # return lossEZ
    return lossEZ + lossRS


@jax.jit
def learning_loss_function_single(
    pursuerX,
    heading,
    speed,
    intercepted,
    pathTime,
    pathHistory,
    interceptedPoint,
    ezPoint,
    ezTime,
    trueParams,
    interceptedPathWeight=1.0,
    flattenLearingLossAmount=0.0,
    verbose=False,
):
    (pursuerPosition, pursuerHeading, pursuerSpeed, minimumTurnRadius, pursuerRange) = (
        pursuerX_to_params(pursuerX, trueParams)
    )
    rsEnd = dubins_reachable_set_from_pursuerX(
        pursuerX,
        jnp.array([interceptedPoint]),
        trueParams,
    ).squeeze()

    rsAll = dubins_reachable_set_from_pursuerX(pursuerX, pathHistory, trueParams)

    interceptedLossRS = activation(rsEnd - flattenLearingLossAmount)
    # interceptedLossEZ = activation(ezEnd - flattenLearingLossAmount)

    survivedLossRS = jnp.max(
        activation(-rsAll - flattenLearingLossAmount)
    )  # loss if survived in RS
    # survivedLossEZ = jnp.max(
    #     activation(-ezAll - flattenLearingLossAmount)
    # )  # loss if survived in RS

    lossRS = jax.lax.cond(
        intercepted,
        lambda: interceptedLossRS,  # + interceptedLossEZ,
        lambda: survivedLossRS,  # + survivedLossEZ,
    )

    return lossRS


def pred_launch_time(interceptedPoint, pursuerX, trueParams, pathTime):
    pursuerPosition, pursuerHeading, pursuerSpeed, minimumTurnRadius, pursuerRange = (
        pursuerX_to_params(pursuerX, trueParams)
    )
    sigma = 0.001 * 3
    angles = jnp.linspace(0.0, 2 * jnp.pi, 64, endpoint=False)
    offsets = sigma * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    def path_len_for_offset(dxy):
        ip = interceptedPoint + dxy
        return dubinsEZ.find_shortest_dubins_path(
            pursuerPosition, pursuerHeading, ip, minimumTurnRadius
        )

    # Vectorize over offsets and average
    path_lens = jax.vmap(path_len_for_offset)(offsets)
    predPursuerDistanceTraveled = jnp.min(path_lens)
    # predPursuerDistanceTraveled = dubinsEZ.find_shortest_dubins_path(
    #     pursuerPosition, pursuerHeading, interceptedPoint, minimumTurnRadius
    # )
    pred_ezTime = pathTime[-1] - (predPursuerDistanceTraveled) / pursuerSpeed
    return (
        pred_ezTime,
        predPursuerDistanceTraveled,
        path_lens,
        interceptedPoint + offsets,
    )


@jax.jit
def learning_loss_function_single_ez(
    pursuerX,
    heading,
    speed,
    intercepted,
    pathTime,
    pathHistory,
    interceptedPoint,
    ezPoint,
    ezTime,
    trueParams,
    interceptedPathWeight=1.0,
    flattenLearingLossAmount=0.0,
    verbose=False,
):
    lossRS = learning_loss_function_single(
        pursuerX,
        heading,
        speed,
        intercepted,
        pathTime,
        pathHistory,
        interceptedPoint,
        ezPoint,
        ezTime,
        trueParams,
        interceptedPathWeight,
        flattenLearingLossAmount,
    )
    pred_ezTime, predPursuerDistanceTraveled, path_lens, points = pred_launch_time(
        interceptedPoint, pursuerX, trueParams, pathTime
    )

    time_loss = time_loss_with_flatten(
        # pred_ezTime,
        # ezTime,
        # 0.001 * 2.3,
        pred_ezTime,
        ezTime,
        flattenLearingLossAmount / pursuerX[3] + 0.001 * 2.5,
    )

    lossEZ = jax.lax.cond(intercepted, lambda: time_loss, lambda: 0.0)

    def verbose_fn():
        jax.debug.print(
            "intercepted: {},pred_ezTime: {}, ezTime: {}, time_loss: {}, predPursuerDistanceTraveled: {},path_lens: {},interceptedPoint: {}, points: {}",
            intercepted,
            pred_ezTime,
            ezTime,
            lossEZ,
            predPursuerDistanceTraveled,
            path_lens,
            interceptedPoint,
            points,
        )

    jax.lax.cond(verbose, verbose_fn, lambda: None)

    # return lossEZ
    return lossEZ + lossRS


if interceptionOnBoundary:
    if knownSpeed:
        batched_loss = jax.jit(
            jax.vmap(
                learning_loss_on_boundary_function_single,
                in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
            )
        )
    else:
        batched_loss = jax.jit(
            jax.vmap(
                learning_loss_on_boundary_function_single_EZ,
                in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
            )
        )
else:
    if knownSpeed:
        batched_loss = jax.jit(
            jax.vmap(
                learning_loss_function_single,
                in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
            )
        )
    else:
        batched_loss = jax.jit(
            jax.vmap(
                learning_loss_function_single_ez,
                in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
            )
        )


@jax.jit
def total_learning_loss(
    pursuerX,
    headings,
    speeds,
    intercepted_flags,
    pathTimes,
    pathHistories,
    interceptedPoints,
    ezPoints,
    ezTimes,
    trueParams,
    interceptedPathWeight=1.0,
    flattenLearingLossAmount=0.0,
    verbose=False,
):
    # shape (N,)
    losses = batched_loss(
        pursuerX,
        headings,
        speeds,
        intercepted_flags,
        pathTimes,
        pathHistories,
        interceptedPoints,
        ezPoints,
        ezTimes,
        trueParams,
        interceptedPathWeight,
        flattenLearingLossAmount,
        verbose,
    )

    # total loss = sum over agents
    return jnp.sum(losses) / len(losses)


dTotalLossDX = jax.jit(jax.jacfwd(total_learning_loss, argnums=0))
totalLossHessian = jax.jit(jax.jacfwd(dTotalLossDX, argnums=0))


def run_optimization_hueristic(
    headings,
    speeds,
    pathTimes,
    interceptedList,
    pathHistories,
    endPoints,
    endTimes,
    ezPoints,
    ezTimes,
    initialPursuerX,
    lowerLimit,
    upperLimit,
    trueParams,
    interceptedPathWeight=1.0,
    flattenLearingLossAmount=0.0,
    verbose=False,
):
    if positionAndHeadingOnly:
        lowerLimitSub = np.array([0.5, 0.5, 0])
        # lowerLimitSub = np.array([0.5, 0.5, 0])
    elif knownSpeed:
        lowerLimitSub = np.array([0.5, 0.5, 0, 0.2, 0.2])
        # lowerLimitSub = np.array([0.5, 0.5, 0, 0.2, 0.2])
    else:
        lowerLimitSub = np.array([0.5, 0.5, 0.0, 0.2, 0.2, 0.2])
        # lowerLimitSub = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    initialLoss = total_learning_loss(
        initialPursuerX,
        headings,
        speeds,
        interceptedList,
        pathTimes,
        pathHistories,
        endPoints,
        ezPoints,
        ezTimes,
        trueParams,
        interceptedPathWeight,
        flattenLearingLossAmount,
        verbose=False,
    )

    if verbose:
        # print("initial pursuerX:", initialPursuerX)
        # initialLoss = total_learning_loss(
        #     initialPursuerX,
        #     headings,
        #     speeds,
        #     interceptedList,
        #     pathTimes,
        #     pathHistories,
        #     endPoints,
        #     ezPoints,
        #     ezTimes,
        #     trueParams,
        #     interceptedPathWeight,
        #     flattenLearingLossAmount,
        #     verbose=verbose,
        # )
        # initialGradient = dTotalLossDX(
        #     initialPursuerX,
        #     headings,
        #     speeds,
        #     interceptedList,
        #     pathTimes,
        #     pathHistories,
        #     endPoints,
        #     ezPoints,
        #     ezTimes,
        #     trueParams,
        #     interceptedPathWeight,
        #     flattenLearingLossAmount,
        # )
        # print("initial loss:", initialLoss)
        # print("initial gradient:", initialGradient)
        print(
            "true pursuerX:",
            trueParams[parameterMask],
        )
        lossTrue = total_learning_loss(
            trueParams[parameterMask],
            headings,
            speeds,
            interceptedList,
            pathTimes,
            pathHistories,
            endPoints,
            ezPoints,
            ezTimes,
            trueParams,
            interceptedPathWeight,
            flattenLearingLossAmount,
            verbose=True,
        )
        # trueGrad = dTotalLossDX(
        #     trueParams[parameterMask],
        #     headings,
        #     speeds,
        #     interceptedList,
        #     pathTimes,
        #     pathHistories,
        #     endPoints,
        #     ezPoints,
        #     ezTimes,
        #     trueParams,
        #     interceptedPathWeight,
        #     flattenLearingLossAmount,
        # )
        print("true learning loss:", lossTrue)
        # print("true gradient:", trueGrad)

    #
    def objfunc(xDict):
        pursuerX = xDict["pursuerX"]
        loss = total_learning_loss(
            pursuerX,
            headings,
            speeds,
            interceptedList,
            pathTimes,
            pathHistories,
            endPoints,
            ezPoints,
            ezTimes,
            trueParams,
            interceptedPathWeight,
            flattenLearingLossAmount,
            verbose=False,
        )
        funcs = {}
        funcs["loss"] = loss
        return funcs, False

    def sens(xDict, funcs):
        pursuerX = xDict["pursuerX"]
        grad_x = dTotalLossDX(
            pursuerX,
            headings,
            speeds,
            interceptedList,
            pathTimes,
            pathHistories,
            endPoints,
            ezPoints,
            ezTimes,
            trueParams,
            interceptedPathWeight,
            flattenLearingLossAmount,
            verbose=False,
        )

        funcsSens = {}
        funcsSens["loss"] = {
            "pursuerX": grad_x,
        }
        return funcsSens, False

    optProb = Optimization("path optimization", objfunc)
    if positionAndHeadingOnly:
        numVars = 3
    elif knownSpeed:
        numVars = 5
    else:
        numVars = 6
    optProb.addVarGroup(
        name="pursuerX",
        nVars=numVars,
        varType="c",
        value=initialPursuerX,
        lower=lowerLimit - lowerLimitSub,
        upper=upperLimit + lowerLimitSub,
    )
    optProb.addObj("loss", scale=initialLoss)
    opt = OPT("ipopt")
    opt.options["print_level"] = 0
    opt.options["max_iter"] = 50
    opt.options["warm_start_init_point"] = "yes"
    opt.options["mu_init"] = 1e-1
    opt.options["nlp_scaling_method"] = "gradient-based"
    opt.options["tol"] = 1e-6
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"

    sol = opt(optProb, sens=sens)
    pursuerX = sol.xStar["pursuerX"]

    loss = total_learning_loss(
        pursuerX,
        headings,
        speeds,
        interceptedList,
        pathTimes,
        pathHistories,
        endPoints,
        ezPoints,
        ezTimes,
        trueParams,
        interceptedPathWeight,
        flattenLearingLossAmount,
        verbose=False,
    )
    # if loss < 1e-1:
    #     print("pursuerX:", pursuerX)
    #     print("pursuerX loss:", loss)
    #     loss = total_learning_loss(
    #         pursuerX,
    #         headings,
    #         speeds,
    #         interceptedList,
    #         pathTimes,
    #         pathHistories,
    #         endPoints,
    #         ezPoints,
    #         ezTimes,
    #         trueParams,
    #         interceptedPathWeight,
    #         flattenLearingLossAmount,
    #         verbose=True,
    #     )

    lossNoFlatten = total_learning_loss(
        pursuerX,
        headings,
        speeds,
        interceptedList,
        pathTimes,
        pathHistories,
        endPoints,
        ezPoints,
        ezTimes,
        trueParams,
        interceptedPathWeight,
        flattenLearingLossAmount=0.0,
        verbose=False,
    )
    return pursuerX, loss, lossNoFlatten


def latin_hypercube_uniform(lowerLimit, upperLimit, numSamples, rng):
    """
    Perform Latin Hypercube Sampling between lower and upper limits (uniform).

    Args:
        lowerLimit: array-like, shape (D,)
        upperLimit: array-like, shape (D,)
        numSamples: int, number of samples

    Returns:
        samples: jnp.ndarray of shape (numSamples, D)
    """
    lowerLimit = np.array(lowerLimit)
    upperLimit = np.array(upperLimit)
    dim = len(lowerLimit)

    rs = np.random.RandomState(int(rng.integers(0, 2**31 - 1, dtype=np.uint32)))
    lhs_unit = lhs(dim, samples=numSamples, random_state=rs)

    # Scale to [lower, upper]
    lhs_scaled = lowerLimit + lhs_unit * (upperLimit - lowerLimit)

    return jnp.array(lhs_scaled)


def plot_contour_of_loss(
    headings,
    speeds,
    pathTimes,
    interceptedList,
    pathHistories,
    endPoints,
    endTimes,
    ezPoints,
    ezTimes,
    trueParams,
    learnedParams,
    lowerLimit,
    upperLimit,
    pursuerXList,
    parameterIndexi=3,
    parameterIndexj=4,
    interceptedPathWeight=1.0,
    flattenLearningLossAmount=0.0,
    title="",
):
    paramterNames = ["xpos", "ypos", "heading", "turnRadius", "range"]
    numSamples = 100
    paramsi = np.linspace(
        lowerLimit[parameterIndexi], upperLimit[parameterIndexi], numSamples
    )
    paramsj = np.linspace(
        lowerLimit[parameterIndexj], upperLimit[parameterIndexj], numSamples
    )
    paramsi, paramsj = np.meshgrid(paramsi, paramsj)
    losses = np.zeros(paramsj.shape)
    # param = np.array(
    #     [trueParams[0], trueParams[1], trueParams[2], trueParams[4], trueParams[5]]
    # )
    param = learnedParams.copy()
    print("learnedParams", learnedParams)
    for i in tqdm.tqdm(range(paramsi.shape[0])):
        for j in range(paramsj.shape[1]):
            param[parameterIndexi] = paramsi[i, j]
            param[parameterIndexj] = paramsj[i, j]
            lossTrue = total_learning_loss(
                param,
                headings,
                speeds,
                interceptedList,
                pathTimes,
                pathHistories,
                endPoints,
                ezPoints,
                ezTimes,
                trueParams,
                1.0,
                flattenLearningLossAmount,
                verbose=False,
            )

            losses[i, j] = lossTrue
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(paramsi, paramsj, losses)
    cbar = fig.colorbar(c, ax=ax)
    mask = (losses < 1e-5).astype(float)
    ax.contour(
        paramsi,
        paramsj,
        mask,
        levels=[0.5],  # boundary between 0 and 1
        colors="red",
        linewidths=0.5,
    )
    if parameterIndexi > 2:
        parameterIndexiTrue = parameterIndexi + 1
    else:
        parameterIndexiTrue = parameterIndexi
    if parameterIndexj > 2:
        parameterIndexjTrue = parameterIndexj + 1
    else:
        parameterIndexjTrue = parameterIndexj
    ax.scatter(
        trueParams[parameterIndexiTrue],
        trueParams[parameterIndexjTrue],
        color="red",
        label="True Params",
    )
    ax.scatter(
        pursuerXList[:, parameterIndexi],
        pursuerXList[:, parameterIndexj],
        color="blue",
        label="All Params",
    )
    ax.scatter(
        learnedParams[parameterIndexi],
        learnedParams[parameterIndexj],
        color="green",
        label="Learned Params",
    )
    ax.set_xlabel(paramterNames[parameterIndexi])
    ax.set_ylabel(paramterNames[parameterIndexj])
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.title(title)


_G = {}  # per-worker constants


def set_constants(payload):
    (
        headings,
        speeds,
        pathTimes,
        interceptedList,
        pathHistories,
        endPoints,
        endTimes,
        ezPoints,
        ezTimes,
        trueParams,
        lowerLimit,
        upperLimit,
        interceptedPathWeight,
        flattenLearningLossAmount,
    ) = payload
    _G.update(
        headings=headings,
        speeds=speeds,
        pathTimes=pathTimes,
        interceptedList=interceptedList,
        pathHistories=pathHistories,
        endPoints=endPoints,
        endTimes=endTimes,
        ezPoints=ezPoints,
        ezTimes=ezTimes,
        trueParams=trueParams,
        lowerLimit=lowerLimit,
        upperLimit=upperLimit,
        interceptedPathWeight=float(interceptedPathWeight),
        flattenLearningLossAmount=float(flattenLearningLossAmount),
    )
    return True  # so caller can await completion


def solve_one(x0):
    return run_optimization_hueristic(
        _G["headings"],
        _G["speeds"],
        _G["pathTimes"],
        _G["interceptedList"],
        _G["pathHistories"],
        _G["endPoints"],
        _G["endTimes"],
        _G["ezPoints"],
        _G["ezTimes"],
        x0,
        _G["lowerLimit"],
        _G["upperLimit"],
        _G["trueParams"],
        interceptedPathWeight=_G["interceptedPathWeight"],
        flattenLearingLossAmount=_G["flattenLearningLossAmount"],
        verbose=False,
    )


_EXECUTOR = None
_NWORKERS = 0


def start_pool(workers=2):
    """Create a single persistent pool (once)."""
    global _EXECUTOR, _NWORKERS
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(max_workers=workers)
        _NWORKERS = workers
    return _EXECUTOR


def stop_pool():
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=True)
        _EXECUTOR = None


def _broadcast_constants(payload):
    """Ensure each worker receives the new measurement constants once."""
    ex = _EXECUTOR
    futs = [ex.submit(set_constants, payload) for _ in range(_NWORKERS)]
    for f in as_completed(futs):
        f.result()  # propagate any worker error immediately


def solve_measurement(
    headings,
    speeds,
    pathTimes,
    interceptedList,
    pathHistories,
    endPoints,
    endTimes,
    ezPoints,
    ezTimes,
    trueParams,
    lowerLimit,
    upperLimit,
    interceptedPathWeight,
    flattenLearningLossAmount,
    initialPursuerXList,
    workers=2,
):
    # 1) ensure pool exists (spawn ONCE)
    ex = start_pool(workers)

    # 2) broadcast THIS measurement's constants to every worker
    payload = (
        headings,
        speeds,
        pathTimes,
        interceptedList,
        pathHistories,
        endPoints,
        endTimes,
        ezPoints,
        ezTimes,
        trueParams,
        lowerLimit,
        upperLimit,
        interceptedPathWeight,
        flattenLearningLossAmount,
    )
    _broadcast_constants(payload)

    # 3) run the multistarts in parallel and return immediately
    results = list(
        tqdm.tqdm(
            ex.map(solve_one, initialPursuerXList),  # order preserved
            total=len(initialPursuerXList),
            desc="Multistart IPOPT",
        )
    )
    pursuerXList, lossList, lossListNoFlatten = map(
        lambda xs: np.squeeze(np.array(xs)), zip(*results)
    )
    idx = np.argsort(lossList)
    return pursuerXList[idx], lossList[idx], lossListNoFlatten[idx]


def percent_of_true_rs_covered(
    true_params,
    pursuerXList,
    numPoints,
    startPositions,
    interceptedList,
    endPoints,
    pathHistories,
    pursuerPosition,
    pursuerHeading,
    pursuerRange,
    pursuerTurnRadius,
    headings,
    speeds,
    interceptionPointEZList,
    trueParams,
    mean,
):
    testPointsx = np.linspace(-5, 5, numPoints)
    testPointsy = np.linspace(-5, 5, numPoints)
    gridCellArea = (testPointsx[1] - testPointsx[0]) * (testPointsy[1] - testPointsy[0])
    [testPointsx, testPointsy] = np.meshgrid(testPointsx, testPointsy)
    testPointsx = testPointsx.flatten()
    testPointsy = testPointsy.flatten()
    testPoints = np.stack((testPointsx, testPointsy), axis=-1)  # (N,2)
    rsPreds = jax.vmap(dubins_reachable_set_from_pursuerX, in_axes=(0, None, None))(
        pursuerXList, testPoints, true_params
    )  # (M,N)
    trueRs = dubins_reachable_set_from_pursuerX(
        true_params[parameterMask], testPoints, true_params
    )
    outsideAll = jnp.all(rsPreds > 0, axis=0)  # (N,)
    insideAny = jnp.logical_not(outsideAll)
    insideArea = jnp.sum(insideAny) * gridCellArea
    trueArea = jnp.sum(trueRs <= 0) * gridCellArea
    insideProportion = insideArea / trueArea
    outsideTrueRs = trueRs > 0
    insideTrue = jnp.logical_not(outsideTrueRs)
    # count number of points inside true RS but outside all predicted RS
    discrepancy = jnp.sum(jnp.logical_and(insideTrue, outsideAll))
    totalInsideTrue = jnp.sum(insideTrue)
    perercentOutside = discrepancy / totalInsideTrue
    print(
        "percent of true RS inside all predicted RS:", (1 - perercentOutside) * 100, "%"
    )
    # plot
    # fig, ax = plt.subplots()
    # fig, ax = plot_all(
    #     startPositions,
    #     interceptedList,
    #     endPoints,
    #     pathHistories,
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerRange,
    #     pursuerTurnRadius,
    #     headings,
    #     speeds,
    #     pursuerXList,
    #     None,
    #     None,
    #     trueParams,
    #     mean,
    # )
    # ax.contour(
    #     testPointsx.reshape((numPoints, numPoints)),
    #     testPointsy.reshape((numPoints, numPoints)),
    #     insideAny.reshape((numPoints, numPoints)),
    #     levels=[0.5],
    #     colors="magenta",
    # )
    # ax.contour(
    #     testPointsx.reshape((numPoints, numPoints)),
    #     testPointsy.reshape((numPoints, numPoints)),
    #     outsideTrueRs.reshape((numPoints, numPoints)),
    #     levels=[0.0],
    #     colors="red",
    #     linestyles="dashed",
    # )
    # ax.set_aspect("equal", adjustable="box")
    # plt.show()
    return 1.0 - float(perercentOutside), float(insideProportion)


# def RS_path(theta, path_pts, true_params):
#     # returns r_t for all points along path: shape (T,)
#     return dubins_reachable_set_from_pursuerX(theta, path_pts, true_params)


# Gradient wrt theta of vector-valued RS_path: shape (T, d)
# (If d is large and T small, consider jacrev instead.)
grad_theta_RS_path = jax.jacfwd(dubins_reachable_set_from_pursuerX, argnums=0)

grad_theta_EZ_path = jax.jacfwd(dubinsEZ_from_pursuerX, argnums=0)


def grad_theta_RS_point(theta, pt, true_params):
    # fast pointwise gradient via the same transform (single-row slice)
    return grad_theta_RS_path(theta, pt[None, :], true_params)[0]


@jax.jit
def info_for_trajectory_int(
    theta,  # (d,)
    heading,  # unused, kept for API symmetry
    speed,  # unused, kept for API symmetry
    intercepted,  # bool
    path_time,  # (T,)
    path_history,  # (T,2)
    intercepted_point,  # (2,)
    true_params,
    flatten_margin,
):
    # HIT branch: use interception point
    def interception_fn():
        r_end = dubins_reachable_set_from_pursuerX(
            theta, intercepted_point[None, :], true_params
        )[0]
        z_hit = r_end - flatten_margin
        w_hit = activation_smooth(z_hit)  # ≥ 0
        g_end = grad_theta_RS_point(theta, intercepted_point, true_params)  # (d,)
        I = w_hit * jnp.outer(g_end, g_end)
        return 0.5 * (I + I.T)

    # MISS branch: soft-select along the path (keeps gradients)
    def miss_fn():
        # RS and grads along path
        r_t = dubins_reachable_set_from_pursuerX(
            theta, path_history, true_params
        )  # (T,)

        # z_t = inside violation amount
        z_t = -r_t - flatten_margin
        w_t = activation_smooth(z_t)  # (T,)

        # pick index of max violation
        idx = jnp.argmax(w_t)

        # gradient at that index
        g = grad_theta_RS_path(theta, path_history[idx][None, :], true_params)[0]

        # scale by weight
        I = w_t[idx] * jnp.outer(g, g)
        return 0.5 * (I + I.T)

    I = jax.lax.cond(intercepted, interception_fn, miss_fn)
    # extra symmetrize for hygiene
    return 0.5 * (I + I.T)


@jax.jit
def info_for_trajectory_int_ez(
    theta,  # (d,)
    heading,  # unused, kept for API symmetry
    speed,  # unused, kept for API symmetry
    intercepted,  # bool
    path_time,  # (T,)
    path_history,  # (T,2)
    intercepted_point,  # (2,)
    true_params,
    flatten_margin,
):
    # HIT branch: use interception point
    def interception_fn():
        r_end = dubinsEZ_from_pursuerX(
            theta, intercepted_point[None, :], jnp.array([heading]), speed, true_params
        )[0]
        z_hit = r_end - flatten_margin
        w_hit = activation_smooth(z_hit)  # ≥ 0
        g_end = grad_theta_EZ_path(
            theta, intercepted_point[None, :], jnp.array([heading]), speed, true_params
        )[0]
        I = w_hit * jnp.outer(g_end, g_end)
        return 0.5 * (I + I.T)

    # MISS branch: soft-select along the path (keeps gradients)
    def miss_fn():
        # RS and grads along path
        headings = heading * jnp.ones(path_history.shape[0])
        r_t = dubinsEZ_from_pursuerX(
            theta, path_history, headings, speed, true_params
        )  # (T,)

        # z_t = inside violation amount
        z_t = -r_t - flatten_margin
        w_t = activation_smooth(z_t)  # (T,)

        # pick index of max violation
        idx = jnp.argmax(w_t)

        # gradient at that index
        g = grad_theta_EZ_path(
            theta,
            path_history[idx][None, :],
            jnp.array([headings[idx]]),
            speed,
            true_params,
        )[0]

        # scale by weight
        I = w_t[idx] * jnp.outer(g, g)
        return 0.5 * (I + I.T)

    I = jax.lax.cond(intercepted, interception_fn, miss_fn)
    # extra symmetrize for hygiene
    return 0.5 * (I + I.T)


@jax.jit
def info_for_trajectory_bound(
    theta,  # (d,)
    heading,  # unused, kept for API symmetry
    speed,  # unused, kept for API symmetry
    intercepted,  # bool
    path_time,  # (T,)
    path_history,  # (T,2)
    intercepted_point,  # (2,)
    true_params,
    flatten_margin,
):
    # HIT branch: use interception point
    def interception_fn():
        r_t = dubins_reachable_set_from_pursuerX(
            theta, path_history[:-4], true_params
        )  # (T,)

        # z_t = inside violation amount
        z_t = -r_t - flatten_margin
        w_t = activation_smooth(z_t)  # (T,)

        # pick index of max violation
        idx = jnp.argmax(w_t)

        # gradient at that index
        g = grad_theta_RS_path(theta, path_history[idx][None, :], true_params)[0]

        # scale by weight
        I_traj = w_t[idx] * jnp.outer(g, g)

        g_end = grad_theta_RS_point(theta, intercepted_point, true_params)
        r_end = RS_path(theta, intercepted_point[None, :], true_params)[0]
        z_hitn = r_end - flatten_margin
        z_hitp = -r_end - flatten_margin
        w_hit = activation_smooth(z_hitn) + activation_smooth(z_hitp)  # ≥ 0
        I_end = w_hit * jnp.outer(g_end, g_end)
        return 0.5 * (I_traj + I_traj.T) + 0.5 * (I_end + I_end.T)

    # MISS branch: soft-select along the path (keeps gradients)
    def miss_fn():
        # RS and grads along path
        r_t = dubins_reachable_set_from_pursuerX(
            theta, path_history, true_params
        )  # (T,)

        # z_t = inside violation amount
        z_t = -r_t - flatten_margin
        w_t = activation_smooth(z_t)  # (T,)

        # pick index of max violation
        idx = jnp.argmax(w_t)

        # gradient at that index
        g = grad_theta_RS_path(theta, path_history[idx][None, :], true_params)[0]

        # scale by weight
        I = w_t[idx] * jnp.outer(g, g)
        return 0.5 * (I + I.T)

    I = jax.lax.cond(intercepted, interception_fn, miss_fn)
    # extra symmetrize for hygiene
    return 0.5 * (I + I.T)


if interceptionOnBoundary:
    info_for_trajectory = info_for_trajectory_bound
else:
    if knownSpeed:
        info_for_trajectory = info_for_trajectory_int_ez
    else:
        info_for_trajectory = info_for_trajectory_int_ez


@jax.jit
def info_for_trajectories(
    pursuerX,
    headings,
    speeds,
    pathTimes,
    interceptedList,
    pathHistories,
    endPoints,
    trueParams,
    flatten_margin,
):
    I_total = jax.vmap(
        info_for_trajectory, in_axes=(None, 0, 0, 0, 0, 0, 0, None, None)
    )(
        pursuerX,
        headings,
        speeds,
        interceptedList,
        pathTimes,
        pathHistories,
        endPoints,
        trueParams,
        flatten_margin,
    )
    I_total = jnp.sum(I_total, axis=0)
    I_total = 0.5 * (I_total + I_total.T)
    return I_total


def find_average_approximate_FIM(
    headings,
    speeds,
    pathTimes,
    interceptedList,
    pathHistories,
    endPoints,
    endTimes,
    ezPoints,
    ezTimes,
    trueParams,
    interceptedPathWeight,
    flattenLearingLossAmount,
    initialPursuerXList,
):
    averageFIM = jax.jit(
        jax.vmap(
            info_for_trajectories,
            in_axes=(0, None, None, None, None, None, None, None, None),
        )
    )(
        initialPursuerXList,
        headings,
        speeds,
        pathTimes,
        interceptedList,
        pathHistories,
        endPoints,
        trueParams,
        flattenLearingLossAmount,
    )
    averageFIM = jnp.sum(averageFIM, axis=0)
    # averageFIM /= len(initialPursuerXList)
    return averageFIM


def learn_ez(
    headings,
    speeds,
    pathTimes,
    interceptedList,
    pathHistories,
    endPoints,
    endTimes,
    ezPoints,
    ezTimes,
    trueParams,
    previousPursuerXList=None,
    mean=None,
    cov=None,
    angleMean=None,
    angleKappa=None,
    # previousPursuerXList=None,
    numStartHeadings=10,
    lowerLimit=None,
    upperLimit=None,
    interceptedPathWeight=1.0,
    flattenLearningLossAmount=0.0,
    useGaussianSampling=False,
    rng=np.random.default_rng(0),
):
    # jax.config.update("jax_platform_name", "cpu")
    start = time.time()
    pursuerXList = []
    lossList = []
    lossListNoFlatten = []
    if mean is None:
        initialPursuerXList = latin_hypercube_uniform(
            lowerLimit, upperLimit, numStartHeadings, rng
        )
    else:
        initialPursuerXList = find_opt_starting_pursuerX(
            interceptedList,
            endPoints,
            lowerLimit,
            upperLimit,
            previousPursuerXList,
            mean,
            cov,
            angleMean,
            angleKappa,
            numStartHeadings,
            useGaussianSampling=useGaussianSampling,
        )
    # fim = find_average_approximate_FIM(
    #     headings,
    #     speeds,
    #     pathTimes,
    #     interceptedList,
    #     pathHistories,
    #     endPoints,
    #     endTimes,
    #     ezPoints,
    #     ezTimes,
    #     trueParams,
    #     interceptedPathWeight,
    #     flattenLearningLossAmount,
    #     initialPursuerXList,
    # )
    # eval, evect = np.linalg.eig(fim)

    # initialPursuerXList: (N, d_max)
    #
    # for i in tqdm.tqdm(range(len(initialPursuerXList))):
    #     # for i in range(len(initialPursuerXList)):
    #     # pursuerX, loss = run_optimization_hueristic_scipy(
    #     pursuerX, loss, lossNoFlatten = run_optimization_hueristic(
    #         headings,
    #         speeds,
    #         pathTimes,
    #         interceptedList,
    #         pathHistories,
    #         endPoints,
    #         endTimes,
    #         ezPoints,
    #         ezTimes,
    #         initialPursuerXList[i],
    #         lowerLimit,
    #         upperLimit,
    #         trueParams,
    #         interceptedPathWeight=interceptedPathWeight,
    #         flattenLearingLossAmount=flattenLearningLossAmount,
    #         verbose=False,
    #     )
    #     pursuerXList.append(pursuerX)
    #     lossList.append(loss)
    #     lossListNoFlatten.append(lossNoFlatten)
    #     #

    pursuerXList, lossList, lossListNoFlatten = solve_measurement(
        headings,
        speeds,
        pathTimes,
        interceptedList,
        pathHistories,
        endPoints,
        endTimes,
        ezPoints,
        ezTimes,
        trueParams,
        lowerLimit,
        upperLimit,
        interceptedPathWeight,
        flattenLearningLossAmount,
        initialPursuerXList,
    )
    # fim = find_average_approximate_FIM(
    #     headings,
    #     speeds,
    #     pathTimes,
    #     interceptedList,
    #     pathHistories,
    #     endPoints,
    #     endTimes,
    #     ezPoints,
    #     ezTimes,
    #     trueParams,
    #     interceptedPathWeight,
    #     flattenLearningLossAmount,
    #     pursuerXList[lossList < 1e-5],
    # )
    # eval, evect = np.linalg.eig(fim)

    pursuerXList = np.array(pursuerXList).squeeze()
    lossList = np.array(lossList).squeeze()
    lossListNoFlatten = np.array(lossListNoFlatten).squeeze()

    sorted_indices = np.argsort(lossList)
    lossList = lossList[sorted_indices]
    pursuerXList = pursuerXList[sorted_indices]
    lossListNoFlatten = lossListNoFlatten[sorted_indices]

    print("lossList:", lossList)

    print("time to learn ez", time.time() - start)
    # pursuerXList[:, 2] = np.unwrap(pursuerXList[:, 2])
    #

    return pursuerXList, lossList, lossListNoFlatten


def which_lp_spends_most_time_in_all_rs(
    # angle,
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
):
    def single_score(pursuerX):
        t = jnp.linspace(0.0, tmax, numPoints)
        direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])
        displacement = t[:, None] * speed * direction  # shape (T, 2)
        path = start_pos + displacement  # shape (T, 2)
        rs = dubins_reachable_set_from_pursuerX(pursuerX, path, trueParams)
        return jnp.sum(jnp.where(rs < 0.0, 1.0, 0.0))

    totalScore = jnp.sum(jax.vmap(single_score)(pursuerXList))
    return totalScore


def which_lp_path_minimizes_number_of_potential_solutions(
    # angle,
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
):
    N = pursuerXList.shape[0]

    # Simulate new path
    # start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])

    def intercpetion_point(pursuerX):
        (
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ) = pursuerX_to_params(pursuerX, trueParams)
        (
            _,
            intercepted,
            endPoint,
            endTime,
            pathHistory,
            interceptionPointEZ,
            interceptionTimeEZ,
        ) = send_low_priority_agent_boundary(
            start_pos,
            heading,
            speed,
            pursuerPosition,
            pursuerHeading,
            minimumTurnRadius,
            0.0,
            pursuerRange,
            pursuerSpeed,
            tmax,
            numPoints,
            numSimulationPoints=1000,
        )
        return endPoint, intercepted

    interceptedPoints, intercepteds = jax.vmap(intercpetion_point)(pursuerXList)

    def pairwise_disagree(i, j):
        pi = interceptedPoints[i]
        pj = interceptedPoints[j]
        intercepted_i = intercepteds[i]
        intercepted_j = intercepteds[j]

        # norm = jnp.linalg.norm(pi - pj)
        # return jnp.where(jnp.logical_and(intercepted_i, intercepted_j), norm, 0.0)
        #
        # Only compare if both intercepted
        both_intercepted = jnp.logical_and(intercepted_i, intercepted_j)
        too_far_apart = jnp.linalg.norm(pi - pj) > diff_threshold
        return jnp.where(both_intercepted, jnp.linalg.norm(pi - pj), 0.0)

    pairs = jnp.array([(i, j) for i in range(N) for j in range(i + 1, N)])

    def pair_score(pair):
        i, j = pair
        return pairwise_disagree(i, j)

    score = jnp.sum(jax.vmap(pair_score)(pairs))
    return score


def append0(arr, x):
    # append one item along axis 0 without flattening
    return jnp.concatenate([arr, jnp.asarray(x)[None, ...]], axis=0)


def point_to_line_distance(point, line_start, line_end):
    """
    Distance from a point to a line defined by two points.

    Args:
        point: (x, y) tuple for the point
        line_start: (x, y) tuple for the first point on the line
        line_end: (x, y) tuple for the second point on the line

    Returns:
        Distance from point to infinite line
    """
    p = jnp.array(point)
    a = jnp.array(line_start)
    b = jnp.array(line_end)

    # Vector from A to B and A to P
    ab = b - a
    ap = p - a

    # Cross product magnitude (in 2D this is scalar)
    cross = jnp.abs(jnp.cross(ab, ap))

    # Normalize by line length
    distance = cross / jnp.linalg.norm(ab)
    return distance


def prob_intercept_hazard_raw(rs, dt, gamma=6.0):
    lam = jax.nn.sigmoid(gamma * (-rs))  # (T,)
    # Use log-survival for stability:
    logS = -dt * jnp.sum(lam)
    return 1.0 - jnp.exp(logS)


def prob_intercept_softmin(rs_along_path, alpha=8.0, beta=20.0):
    # rs < 0 means inside; soft-min over path (lower rs = more inside)
    # smin_beta(rs) ~ min(rs); make "inside-ness" positive via negation
    smin = -(1.0 / beta) * jax.scipy.special.logsumexp(
        -beta * rs_along_path
    )  # smooth min
    score = -smin  # positive if inside-ish
    return jax.nn.sigmoid(alpha * score)


def bernoulli_entropy(p, eps=1e-8):
    p = jnp.clip(p, eps, 1 - eps)
    return -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))  # nats


def trajectory_mutual_information(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
):
    # numPoints = len(measuredPathHistories[0])

    path, _ = simulate_trajectory_fn(start_pos, heading, speed, tmax, numPoints)
    dt = tmax / numPoints

    def predicted_outcome(pursuerX):
        rs = dubins_reachable_set_from_pursuerX(pursuerX, path, trueParams)
        interceptProb = prob_intercept_softmin(rs)
        # interceptProb = prob_intercept_hazard_raw(rs, dt)
        return interceptProb

    interceptProbs = jax.vmap(predicted_outcome)(pursuerXList)
    N = interceptProbs.shape[0]
    weights = jnp.ones((N,)) / N
    p_bar = jnp.sum(weights * interceptProbs)
    H_bar = bernoulli_entropy(p_bar)
    H_ind = jnp.sum(weights * bernoulli_entropy(interceptProbs))
    MI = H_bar - H_ind
    return MI


def trajectory_average_d_score(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
    previousHessian=None,
):
    ridge = 1e-6
    eps = 1e-9
    """
    Average D-score (Bayesian D-opt incremental info) for a single candidate trajectory.
    Uses Fisher increment for a Bernoulli-style intercept probability p(θ, path).

    Returns: scalar (nats)
    """
    d = len(pursuerXList[0])
    previousHessian = previousHessian[..., None] * jnp.eye(d)

    # 1) Build the candidate trajectory once
    path, _ = simulate_trajectory_fn(start_pos, heading, speed, tmax, numPoints)

    # 2) Probability-of-intercept function p(θ) for this fixed path
    def prob_intercept_theta(pursuerX):
        rs = dubins_reachable_set_from_pursuerX(pursuerX, path, trueParams)
        p = prob_intercept_softmin(rs)  # scalar in (0,1); keep it smooth
        # clip a hair away from 0/1 for numerical stability
        return jnp.clip(p, eps, 1.0 - eps)

    # Option A (recommended): Fisher via logit gradient (numerically stable)
    def fisher_increment(pursuerX):
        p = prob_intercept_theta(pursuerX)

        # α = logit(p);  Jα = ∇θ α;  Fisher = p(1-p) * Jα Jα^T   (PSD)
        def logit_p(x):
            px = prob_intercept_theta(x)
            return jnp.log(px) - jnp.log1p(-px)  # logit(px)

        J_alpha = jax.jacfwd(logit_p)(pursuerX)  # (d,)
        return (p * (1.0 - p)) * jnp.outer(J_alpha, J_alpha)  # (d,d), PSD

    # Vectorize over models
    H_incs = jax.vmap(fisher_increment)(pursuerXList)  # (K, d, d)

    # 3) Per-model D-score: logdet(I + A^{-1} H_inc), stabilized via Cholesky
    def dscore_one(H_inc, A):
        d = A.shape[0]
        Areg = A + ridge * jnp.eye(d, dtype=A.dtype)  # ensure SPD
        L = jnp.linalg.cholesky(Areg)
        # Whiten: B = A^{-1/2} H_inc A^{-1/2}
        X = jax.scipy.linalg.solve_triangular(L, H_inc, lower=True)
        B = jax.scipy.linalg.solve_triangular(L, X.T, lower=True).T
        # D-score = sum log(1 + λ_i(B)); clip tiny negatives for safety
        lam = jnp.clip(jnp.linalg.eigvalsh(B), 0.0)
        return jnp.sum(jnp.log1p(lam))  # scalar (nats)

    gains = jax.vmap(dscore_one, in_axes=(0, None))(H_incs, previousHessian)  # (K,)

    # 4) Average over models (particles)
    weights = jnp.ones_like(gains) / gains.shape[0]
    return jnp.sum(weights * gains)


def softplus_second(z, tau=0.5):
    # φ(z) = τ log(1+exp(z/τ)); φ''(z) = (1/τ) σ(z/τ)(1-σ(z/τ))
    s = jax.nn.sigmoid(z / tau)
    return (s * (1.0 - s)) / tau


def trajectory_margin_mass(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
):
    # numPoints = len(measuredPathHistories[0])
    path, _ = simulate_trajectory_fn(start_pos, heading, speed, tmax, numPoints)
    margin = 0.01
    tau = 0.5
    ds = jnp.ones((len(path),))

    def per_model(theta):
        r = dubins_reachable_set_from_pursuerX(
            theta, path, trueParams
        ).squeeze()  # (T,)
        z = r + margin  # centered at RS=0 (sign doesn’t matter for φ'')
        w = softplus_second(z, tau)  # (T,)

        # Normalize by peak curvature φ''(0)=1/(4τ) and total length, so m∈[0,1]
        #
        peak = 1.0 / (4.0 * tau)
        m = jnp.sum(w * ds) / (peak * (jnp.sum(ds) + 1e-12))
        return m

    masses = jax.vmap(per_model)(pursuerXList)
    timeInMargin = jnp.sum(masses) / len(pursuerXList)
    return timeInMargin


def trajectory_intercept_probability(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
    meanPursuerX=None,
    pastHeadings=None,
    pastStartPos=None,
):
    # numPoints = len(measuredPathHistories[0])
    path, _ = simulate_trajectory_fn(start_pos, heading, speed, tmax, numPoints)

    def predicted_measurement(pursuerX):
        (
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ) = pursuerX_to_params(pursuerX, trueParams)
        rs = dubins_reachable_set_from_pursuerX(pursuerX, path, trueParams)
        timeInRs = jnp.sum(rs < 0.0, axis=0) * (tmax / numPoints)
        intercepted = jnp.any(rs < 0.0, axis=0)

        return intercepted, timeInRs

    intercepteds, timeInRss = jax.vmap(predicted_measurement)(pursuerXList)
    q = intercepteds.mean()
    return q


def trajectory_entropy(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
    meanPursuerX=None,
    pastHeadings=None,
    pastStartPos=None,
):
    # numPoints = len(measuredPathHistories[0])
    path, _ = simulate_trajectory_fn(start_pos, heading, speed, tmax, numPoints)

    def predicted_measurement(pursuerX):
        (
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ) = pursuerX_to_params(pursuerX, trueParams)
        rs = dubins_reachable_set_from_pursuerX(pursuerX, path, trueParams)
        timeInRs = jnp.sum(rs < 0.0, axis=0) * (tmax / numPoints)
        intercepted = jnp.any(rs < 0.0, axis=0)

        return intercepted, timeInRs

    intercepteds, timeInRss = jax.vmap(predicted_measurement)(pursuerXList)
    q = intercepteds.mean()
    # entropy = 2.0 * q * (1.0 - q)  # maximize this
    entropy = -q * jnp.log2(q + 1e-6) - (1.0 - q) * jnp.log2(1.0 - q + 1e-6)
    return entropy


def sample_true_indices_1d_jit(key, mask: jnp.ndarray, k: int = 10):
    # mask: shape (n,), dtype=bool
    n = mask.shape[0]
    perm = jax.random.permutation(key, n)  # shuffle 0..n-1
    shuffled_true = mask[perm]  # which shuffled spots are True?

    csum = jnp.cumsum(shuffled_true.astype(jnp.int32))
    keep = jnp.logical_and(shuffled_true, csum <= k)  # first k Trues in random order
    pos = jnp.nonzero(keep, size=k, fill_value=0)[0]  # length-k (padded)
    idx = perm[pos]  # the chosen indices (length-k)

    valid_count = jnp.minimum(k, jnp.sum(shuffled_true))
    valid = jnp.arange(k) < valid_count  # which of the k are real vs padded
    idx = jnp.where(valid, idx, 0)  # pad invalid with 0 (won’t be used)

    return idx, valid


def trajectory_diff(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
):
    # numPoints = len(measuredPathHistories[0])
    path, _ = simulate_trajectory_fn(start_pos, heading, speed, tmax, numPoints)

    def predicted_measurement(pursuerX):
        rs = dubins_reachable_set_from_pursuerX(pursuerX, path, trueParams)
        inRs = rs < 0.0
        sampled_indices, valid = sample_true_indices_1d_jit(
            jax.random.PRNGKey(0), inRs, k=5
        )
        potential_interception_points = path[sampled_indices]  # (k, 2)
        return potential_interception_points, valid

    def compare_models(pursuerX1, pursuerX2):
        points1, valid1 = predicted_measurement(pursuerX1)
        rs2 = dubins_reachable_set_from_pursuerX(pursuerX2, points1, trueParams)
        outRs2 = rs2 > 0.0
        return jnp.sum(jnp.where(valid1, outRs2.astype(jnp.float32), 0.0))

    def compare_to_all(pursuerX, pursuerXList):
        return jnp.sum(jax.vmap(lambda px: compare_models(pursuerX, px))(pursuerXList))

    scores = jax.vmap(lambda px: compare_to_all(px, pursuerXList))(pursuerXList)
    return jnp.sum(scores)


def smooth_score(endPoints, intercepteds, pastInterceptedPoints, alpha=10.0, eps=1e-6):
    """
    endPoints: (K,2) new candidate endpoints
    intercepteds: (K,) boolean flags
    pastInterceptedPoints: (M,2) past intercepts
    alpha: controls sharpness of soft weighting (bigger = closer to hard switch)
    """

    # --- 1. Turn hard flags into soft weights (smooth if you want it)
    # w = intercepteds.astype(jnp.float32)        # hard version (not smooth in flag)
    w = jax.nn.sigmoid(alpha * (intercepteds.astype(jnp.float32) - 0.5))  # smooth

    # --- 2. Blend between endpoint and fallback point
    fallback = pastInterceptedPoints[0]  # (2,)
    blended = w[:, None] * endPoints + (1.0 - w)[:, None] * fallback  # (K,2)

    # --- 3. Compute distances to all past points
    diffs = blended[:, None, :] - pastInterceptedPoints[None, :, :]  # (K,M,2)
    dists = jnp.linalg.norm(diffs, axis=-1)  # (K,M)

    # --- 4. Return mean (same as your code)
    return jnp.mean(dists)


def which_lp_path_maximizes_dist_to_next_intercept(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    tmax=10.0,
    numPoints=100,
    pastInterceptedPoints=None,
):
    def intercpetion_point(pursuerX):
        (
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ) = pursuerX_to_params(pursuerX, trueParams)
        _, intercepted, endPoint, endTime, pathHistory, _, _ = (
            send_low_priority_agent_boundary(
                start_pos,
                heading,
                speed,
                pursuerPosition,
                pursuerHeading,
                minimumTurnRadius,
                0.0,
                pursuerRange,
                pursuerSpeed,
                tmax,
                numPoints,
                numSimulationPoints=1000,
            )
        )
        return endPoint, intercepted

    interceptedPoints, intercepteds = jax.vmap(intercpetion_point)(pursuerXList)
    interceptedPoints = jnp.where(intercepteds[:, None], interceptedPoints, jnp.nan)
    # interceptedPoints = jnp.array(interceptedPoints)[jnp.array(intercepteds)]

    diffs = interceptedPoints[:, None, :] - pastInterceptedPoints[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    total_distances = jnp.min(dists)  # Sum distances to all other points
    return total_distances

    def intercpetion_point(pursuerX):
        (
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ) = pursuerX_to_params(pursuerX, trueParams)
        _, intercepted, endPoint, endTime, pathHistory, _, _ = (
            send_low_priority_agent_boundary(
                start_pos,
                heading,
                speed,
                pursuerPosition,
                pursuerHeading,
                minimumTurnRadius,
                0.0,
                pursuerRange,
                pursuerSpeed,
                tmax,
                numPoints,
                numSimulationPoints=1000,
            )
        )
        return endPoint, intercepted

    interceptedPoints, intercepteds = jax.vmap(intercpetion_point)(pursuerXList)
    # return smooth_score(interceptedPoints, intercepteds, pastInterceptedPoints)
    interceptedPoints = jnp.where(
        intercepteds[:, None], interceptedPoints, pastInterceptedPoints[0]
    )
    # interceptedPoints = jnp.array(interceptedPoints)[jnp.array(intercepteds)]

    diffs = interceptedPoints[:, None, :] - pastInterceptedPoints[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    total_distances = jnp.mean(dists)  # Sum distances to all other points
    return total_distances


# @jax.jit(static_argnames=("numPoints"))
def which_max_distance_obj(
    x,
    radius,
    center,
    pursuerXList,  # (K, d)
    speed,
    trueParams,
    tmax=10.0,
    numPoints=100,
    previousFims=None,
    flattenLearingLossAmount=0.0,
    meanPursuerX=None,
    covPursuerXSqrt=None,
    endPoints=None,
):
    start_pos = center + radius * jnp.array([jnp.cos(x[0]), jnp.sin(x[0])])
    heading = x[1]
    dist = which_lp_path_maximizes_dist_to_next_intercept(
        start_pos,
        heading,
        pursuerXList,
        speed,
        trueParams,
        tmax,
        numPoints,
        endPoints,
    )
    return -dist


which_max_distance_obj_derivative = jax.jit(
    jax.jacfwd(which_max_distance_obj), static_argnames=("numPoints",)
)


def which_lp_path_minimizes_number_of_potential_solutions_must_intercect_all(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    endpoints=None,
):
    N = pursuerXList.shape[0]

    # Simulate new path
    # start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])

    def intercpetion_point(pursuerX):
        (
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ) = pursuerX_to_params(pursuerX, trueParams)
        _, intercepted, endPoint, endTime, pathHistory, _, _ = (
            send_low_priority_agent_boundary(
                start_pos,
                heading,
                speed,
                pursuerPosition,
                pursuerHeading,
                minimumTurnRadius,
                0.0,
                pursuerRange,
                pursuerSpeed,
                tmax,
                numPoints,
                numSimulationPoints=1000,
            )
        )
        return endPoint, intercepted

    interceptedPoints, intercepteds = jax.vmap(intercpetion_point)(pursuerXList)

    def pairwise_disagree(i, j):
        pi = interceptedPoints[i]
        pj = interceptedPoints[j]
        intercepted_i = intercepteds[i]
        intercepted_j = intercepteds[j]

        # norm = jnp.linalg.norm(pi - pj)
        # return jnp.where(jnp.logical_and(intercepted_i, intercepted_j), norm, 0.0)
        #
        # Only compare if both intercepted
        both_intercepted = jnp.logical_and(intercepted_i, intercepted_j)
        return jnp.where(both_intercepted, jnp.linalg.norm(pi - pj), jnp.nan)

    iu, ju = jnp.triu_indices(N, 1)

    score = jnp.sum(jax.vmap(pairwise_disagree)(iu, ju))
    return score


def inside_model_disagreement_score(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    num_points=100,
    diff_threshold=0.3,
):
    N = pursuerXList.shape[0]

    # Simulate new path
    new_path, new_headings = simulate_trajectory_fn(
        start_pos, heading, speed, tmax, num_points
    )
    new_headings = jnp.reshape(new_headings, (-1,))

    # Compute min EZ value for each parameter set
    def in_rs(pX):
        rs = dubins_reachable_set_from_pursuerX(pX, new_path, trueParams)
        return rs <= 0.0

    probsRS = jax.vmap(in_rs)(pursuerXList)  # (N,)
    # probs = rs_vals < 0.0

    # Pairwise disagreement: p_i * (1 - p_j) + p_j * (1 - p_i)
    def pairwise_disagree(i, j):
        pi = probsRS[i]
        pj = probsRS[j]
        return jnp.sum(pi != pj)

    # indices = jnp.arange(N)
    pairs = jnp.array([(i, j) for i in range(N) for j in range(i + 1, N)])

    def pair_score(pair):
        i, j = pair
        return pairwise_disagree(i, j)

    score = jnp.max(jax.vmap(pair_score)(pairs))
    return score


def _offsets_with_spacing(num_headings: int, delta: float, spacing: str):
    """
    Return heading offsets in [-delta, +delta] with chosen spacing.
    - 'uniform':        uniform grid
    - 'cosine_center':  density ∝ cos( (π θ)/(2δ) )  -> many near 0, few at edges
                        inverse CDF: θ = (2δ/π) * arcsin(2q - 1),  q in (0,1)
    """
    if num_headings == 1:
        return jnp.array([0.0])

    if spacing == "uniform":
        return jnp.linspace(-delta, delta, num_headings)

    if spacing == "cosine_center":
        # Use centered quantiles to avoid endpoints (nice symmetric nodes)
        q = (jnp.arange(num_headings) + 0.5) / num_headings  # in (0,1)
        offsets = (2.0 * delta / jnp.pi) * jnp.arcsin(2.0 * q - 1.0)
        return offsets

    raise ValueError(f"Unknown spacing='{spacing}'")


def generate_headings_toward_center(
    radius,
    num_angles,
    num_headings,
    delta,
    center=(0.0, 0.0),
    theta1=0.0,
    theta2=2 * jnp.pi,
    plot=True,
    spacing: str = "cosine_center",  # <— NEW: 'uniform' or 'cosine_center'
):
    """
    Generate heading directions pointing toward the center ± delta from points on an arc.
    spacing='cosine_center' concentrates offsets near 0 (aim-at-center), which often
    produces higher-incidence RS crossings and better information.
    """
    cx, cy = center

    two_pi = 2 * jnp.pi
    theta1 = jnp.mod(theta1, two_pi)
    theta2 = jnp.mod(theta2, two_pi)
    if theta2 <= theta1:
        theta2 += two_pi

    sweep = theta2 - theta1
    full_circle = jnp.isclose(sweep, two_pi)

    angles = jnp.linspace(theta1, theta2, num_angles, endpoint=not full_circle)

    # positions on arc
    x = radius * jnp.cos(angles) + cx
    y = radius * jnp.sin(angles) + cy
    positions = jnp.stack([x, y], axis=-1)  # (A,2)

    # headings toward center
    heads_to_center = jnp.arctan2(cy - y, cx - x)  # (A,)

    # bias the offsets
    heading_offsets = _offsets_with_spacing(num_headings, delta, spacing)  # (H,)

    # broadcast to all (position, offset) combos
    pos_grid = positions[:, None, :]  # (A,1,2)
    head_grid = heads_to_center[:, None] + heading_offsets[None, :]  # (A,H)

    # wrap headings to (-π, π]
    head_grid = jnp.arctan2(jnp.sin(head_grid), jnp.cos(head_grid))

    # positions_expanded = pos_grid.reshape(-1, 2)  # (A*H,2)
    positions_expanded = jnp.repeat(positions, num_headings, axis=0)  # (N*M, 2)
    headings_expanded = head_grid.reshape(-1)  # (A*H,)

    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(cx, cy, "ro", label="Center")

        for i in range(positions_expanded.shape[0]):
            px, py = positions_expanded[i]
            heading = headings_expanded[i]
            dx = jnp.cos(heading)
            dy = jnp.sin(heading)
            plt.arrow(px, py, dx * 0.2, dy * 0.2, head_width=0.05, color="r", alpha=0.7)

        plt.gca().set_aspect("equal")
        plt.title("Headings Toward Center ± δ")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()
        plt.show()

    return positions_expanded, headings_expanded


def logdet_psd(A, ridge=1e-6):
    sign, logdet = jnp.linalg.slogdet(A)  # sign should be +1 if PD
    return logdet  # stable even when det is tiny


def sym(A):
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def lam_min(A, ridge=1e-6):
    return jnp.linalg.eigvalsh(A)[0]  # ascending


def whiten_info(I, L, ridge=1e-6):
    Linv = jnp.linalg.solve(L, jnp.eye(L.shape[0]))
    return Linv.T @ I @ Linv


def submat(A, idx):
    # A: (p,p), idx: 1D indices
    return A[jnp.ix_(idx, idx)]


def trace_inv(A, ridge=1e-6):
    Ainv = jnp.linalg.solve(A, jnp.eye(A.shape[0]))
    return jnp.trace(Ainv)


info_for_trajectory_multple_future = jax.vmap(
    info_for_trajectory, in_axes=(None, None, None, 0, 0, 0, 0, None, None)
)


def trajectory_fisher_information_cross(
    start_pos,
    heading,
    pursuerXList,  # (K, d)
    speed,
    trueParams,
    tmax=10.0,
    measuredHeadings=None,
    measuredSpeeds=None,
    measuredPathTimes=None,
    measuredInterceptedList=None,
    measuredPathHistories=None,
    measuredEndPoints=None,
    measuredEndTimes=None,
    measuredEzPoints=None,
    measuredEzTimes=None,
    previousFims=None,
    flattenLearingLossAmount=0.0,
    meanPursuerX=None,
    covPursuerXSqrt=None,
):
    K = pursuerXList.shape[0]
    numPoints = len(measuredPathHistories[0])

    model_weights = jnp.ones((K,), dtype=pursuerXList.dtype) / K

    # Hypothetical / expected measurement from candidate trajectory
    def find_future_measurements(pursuerX):
        (
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
            # ) = pursuerX_to_params(trueParams[parameterMask], trueParams)
        ) = pursuerX_to_params(pursuerX, trueParams)
        (
            futurePathTime,
            futureIntercepted,
            futureInterceptedPoint,
            futureInterceptedTime,
            futurePathHistorie,
            futureInterceptionPointEZ,
            futureInterceptionTimeEZ,
        ) = send_low_priority_agent_expected_intercept(
            start_pos,
            heading,
            speed,
            pursuerPosition,
            pursuerHeading,
            minimumTurnRadius,
            0.0,
            pursuerRange,
            pursuerSpeed,
            tmax,
            len(measuredPathHistories[0]),
            numSimulationPoints=100,
        )
        return (
            futureIntercepted,
            futureInterceptedPoint,
            futurePathTime,
            futurePathHistorie,
        )

    (
        futureIntercepteds,
        futureInterceptedPoints,
        futurePathTimes,
        futurePathHistories,
    ) = jax.vmap(find_future_measurements)(pursuerXList)

    def per_model(pursuerX):
        # Unpack params

        fims = info_for_trajectory_multple_future(
            pursuerX,
            heading,
            speed,
            futureIntercepteds,
            futurePathTimes,  #
            futurePathHistories,  # (K, T, 2)
            futureInterceptedPoints,  # (K, 2)
            trueParams,
            flattenLearingLossAmount,
        )
        return jnp.mean(fims, axis=0)

    delta_infos = jax.vmap(per_model)(pursuerXList)  # (K, p, p)

    I_past_bar = jnp.tensordot(model_weights, previousFims, axes=1)  # (p,p)
    Delta_bar = jnp.tensordot(model_weights, delta_infos, axes=1)  # (p,p)

    # Symmetrize once for numerical stability

    I_before = sym(I_past_bar)
    I_after = sym(I_past_bar + Delta_bar)

    # Compute gains once
    E_before = lam_min(I_before)
    E_after = lam_min(I_after)
    e_gain = E_after - E_before
    D_gain = logdet_psd(I_after) - logdet_psd(I_before)

    # Penalty (fix the variable name from 'pen' -> 'penalty')
    tau, beta = 0.5, 20.0
    penalty = jnp.maximum(0.0, tau - E_after)
    # return e_gain
    # return E_after

    return D_gain - beta * penalty


def trajectory_fisher_information_mean(
    start_pos,
    heading,
    pursuerXList,  # (K, d)
    speed,
    trueParams,
    tmax=10.0,
    measuredHeadings=None,
    measuredSpeeds=None,
    measuredPathTimes=None,
    measuredInterceptedList=None,
    measuredPathHistories=None,
    measuredEndPoints=None,
    measuredEndTimes=None,
    measuredEzPoints=None,
    measuredEzTimes=None,
    previousFims=None,
    flattenLearingLossAmount=0.0,
    meanPursuerX=None,
    covPursuerXSqrt=None,
):
    K = pursuerXList.shape[0]

    model_weights = jnp.ones((K,), dtype=pursuerXList.dtype) / K

    # (
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     minimumTurnRadius,
    #     pursuerRange,
    # ) = pursuerX_to_params(meanPursuerX, trueParams)
    # ) = pursuerX_to_params(trueParams[parameterMask], trueParams)
    # ) = pursuerX_to_params(pursuerX, trueParams)
    # (
    #     futurePathTime,
    #     futureIntercepted,
    #     futureInterceptedPoint,
    #     futureInterceptedTime,
    #     futurePathHistory,
    #     futureInterceptionPointEZ,
    #     futureInterceptionTimeEZ,
    # ) = send_low_priority_agent_expected_intercept(
    #     start_pos,
    #     heading,
    #     speed,
    #     pursuerPosition,
    #     pursuerHeading,
    #     minimumTurnRadius,
    #     0.0,
    #     pursuerRange,
    #     pursuerSpeed,
    #     tmax,
    #     len(measuredPathHistories[0]),
    #     numSimulationPoints=100,
    # )
    futurePathHistory, _ = simulate_trajectory_fn(
        start_pos, heading, speed, tmax, numPoints=100
    )
    futureInterceptedPoint = None
    futurePathTime = tmax
    futureIntercepted = None

    def per_model(pursuerX):
        # Unpack params

        fim = info_for_trajectory(
            pursuerX,
            heading,
            speed,
            futureIntercepted,
            futurePathTime,  # (T,)
            futurePathHistory,  #
            futureInterceptedPoint,  # (2,)
            trueParams,
            flattenLearingLossAmount,
        )
        return fim

    delta_infos = jax.vmap(per_model)(pursuerXList)  # (K, p, p)

    I_past_bar = jnp.tensordot(model_weights, previousFims, axes=1)  # (p,p)
    Delta_bar = jnp.tensordot(model_weights, delta_infos, axes=1)  # (p,p)

    # Symmetrize once for numerical stability

    I_before = sym(I_past_bar)
    I_after = sym(I_past_bar + Delta_bar)

    # Compute gains once
    E_before = lam_min(I_before)
    E_after = lam_min(I_after)
    e_gain = E_after - E_before
    D_gain = logdet_psd(I_after) - logdet_psd(I_before)

    # Penalty (fix the variable name from 'pen' -> 'penalty')
    tau, beta = 1.0, 20.0
    penalty = jnp.maximum(0.0, tau - E_after)
    # return e_gain
    # return E_after

    return D_gain - beta * penalty


if interceptionOnBoundary:
    send_lp_agent = send_low_priority_agent_boundary
else:
    send_lp_agent = send_low_priority_agent_expected_intercept


def trajectory_fisher_information(
    start_pos,
    heading,
    pursuerXList,  # (K, d)
    speed,
    trueParams,
    tmax=10.0,
    numPoints=100,
    previousFims=None,
    flattenLearingLossAmount=0.0,
    meanPursuerX=None,
    covPursuerXSqrt=None,
):
    K = pursuerXList.shape[0]

    model_weights = jnp.ones((K,), dtype=pursuerXList.dtype) / K

    # Hypothetical / expected measurement from candidate trajectory
    def find_future_measurements(pursuerX):
        (
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ) = pursuerX_to_params(pursuerX, trueParams)
        (
            futurePathTime,
            futureIntercepted,
            futureInterceptedPoint,
            futureInterceptedTime,
            futurePathHistorie,
            futureInterceptionPointEZ,
            futureInterceptionTimeEZ,
        ) = send_low_priority_agent_expected_intercept(
            # ) = send_lp_agent(
            start_pos,
            heading,
            speed,
            pursuerPosition,
            pursuerHeading,
            minimumTurnRadius,
            0.0,
            pursuerRange,
            pursuerSpeed,
            tmax,
            numPoints,
            numSimulationPoints=100,
        )
        return futureIntercepted, futureInterceptedPoint

    futureIntercepteds, futureInterceptedPoints = jax.vmap(find_future_measurements)(
        pursuerXList
    )
    interceptionProb = jnp.mean(futureIntercepteds)
    numerator = jnp.sum(futureInterceptedPoints * futureIntercepteds[:, None], axis=0)
    denominator = jnp.sum(futureIntercepteds)

    averageInterception = numerator / jnp.maximum(denominator, 1)
    futurePathTimeHit = jnp.linalg.norm(averageInterception - start_pos) / speed
    futurePathHit, _ = simulate_trajectory_fn(
        start_pos, heading, speed, futurePathTimeHit, numPoints
    )
    futurePathMiss, _ = simulate_trajectory_fn(
        start_pos, heading, speed, tmax, numPoints
    )

    def per_model(pursuerX):
        # Unpack params

        fimHit = info_for_trajectory(
            pursuerX,
            heading,
            speed,
            True,
            futurePathTimeHit,  #
            futurePathHit,  # (T, 2)
            averageInterception,  # (2,)
            trueParams,
            flattenLearingLossAmount,
        )
        fimMiss = info_for_trajectory(
            pursuerX,
            heading,
            speed,
            False,
            tmax,  #
            futurePathMiss,  # (T, 2)
            averageInterception,
            trueParams,
            flattenLearingLossAmount,
        )
        return interceptionProb * fimHit + (1 - interceptionProb) * fimMiss

    delta_infos = jax.vmap(per_model)(pursuerXList)  # (K, p, p)

    I_past_bar = jnp.tensordot(model_weights, previousFims, axes=1)  # (p,p)
    Delta_bar = jnp.tensordot(model_weights, delta_infos, axes=1)  # (p,p)

    # Symmetrize once for numerical stability

    I_before = sym(I_past_bar)
    I_after = sym(I_past_bar + Delta_bar)

    # Compute gains once
    E_before = lam_min(I_before)
    E_after = lam_min(I_after)
    e_gain = E_after - E_before
    D_gain = logdet_psd(I_after) - logdet_psd(I_before)

    # Penalty (fix the variable name from 'pen' -> 'penalty')
    tau, beta = 0.75, 20.0
    penalty = jnp.maximum(0.0, tau - E_after)
    # return e_gain
    # return E_after
    #

    return D_gain - beta * penalty


@jax.jit
def traj_info_objective(
    x,
    radius,
    center,
    pursuerXList,  # (K, d)
    speed,
    trueParams,
    tmax=10.0,
    numPoints=100,
    previousFims=None,
    flattenLearingLossAmount=0.0,
    meanPursuerX=None,
    covPursuerXSqrt=None,
):
    angle = x[0]
    heading = x[1]
    start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    return -trajectory_fisher_information(
        start_pos,
        heading,
        pursuerXList,
        speed,
        trueParams,
        tmax,
        numPoints,
        previousFims,
        flattenLearingLossAmount,
        meanPursuerX,
        covPursuerXSqrt,
    )


d_traj_info_objective = jax.jit(
    jax.jacfwd(traj_info_objective),
    static_argnames=("numPoints",),
)


def top_percent(scores, pct=10.0):
    """
    Keep the top `pct` percent by score.
    Returns (mask, idx_desc, vals_desc).
    - scores: (N,)
    - pct: percent in [0, 100]
    """
    scores = jnp.nan_to_num(scores, nan=-jnp.inf)  # treat NaNs as worst
    N = scores.size
    pct = jnp.clip(pct, 0.0, 100.0)
    k = jnp.maximum(1, jnp.ceil((pct / 100.0) * N).astype(int))

    # Top-K (desc) — O(N) via selection, no full sort
    vals_desc, idx_desc = jax.lax.top_k(scores, k)

    mask = jnp.zeros(N, dtype=bool).at[idx_desc].set(True)
    return mask, idx_desc, vals_desc


def trajectory_det(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
    meanPursuerX=None,
):
    pursuerX = trueParams[parameterMask]
    # pursuerX = meanPursuerX
    path, _ = simulate_trajectory_fn(start_pos, heading, speed, tmax, numPoints)
    trueRS = dubins_reachable_set_from_pursuerX(pursuerX, path, trueParams)
    inRS = jnp.any(trueRS < 0.00, axis=0)
    thresh = 0.01
    closeEnough = jnp.min(jnp.abs(trueRS)) < thresh
    notTooFarInside = jnp.min(trueRS) > -thresh
    return jnp.where(jnp.logical_and(closeEnough, notTooFarInside), 100, 0)


def trajectory_time_near_boundary(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
    meanPursuerX=None,
    pastHeadings=None,
    pastStartPos=None,
):
    thresh = 0.1
    path, _ = simulate_trajectory_fn(start_pos, heading, speed, tmax, numPoints)
    dt = tmax / (numPoints - 1)

    def per_model(pursuerX):
        r = dubins_reachable_set_from_pursuerX(pursuerX, path, trueParams)  # (T,)
        close = (jnp.abs(r) < thresh).astype(jnp.float32)  # (T,)
        # Disallow if the path goes deeper than -thresh at any time
        safe = (jnp.min(r) > 0).astype(jnp.float32)
        # # Sum timesteps in band → multiply by dt to get *time*
        return safe * (jnp.sum(close) * dt)

    # Sum across models; use jnp.mean(...) if you prefer average per model
    return jnp.sum(jax.vmap(per_model)(pursuerXList))


def trajectory_time_near_boundary_mean(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
    meanPursuerX=None,
    pastHeadings=None,
    pastStartPos=None,
):
    pursuerX = meanPursuerX
    path, _ = simulate_trajectory_fn(start_pos, heading, speed, tmax, numPoints)
    trueRS = dubins_reachable_set_from_pursuerX(pursuerX, path, trueParams)
    inRS = jnp.any(trueRS < 0.00, axis=0)
    thresh = 0.1
    closeEnough = jnp.min(jnp.abs(trueRS)) < thresh
    notTooFarInside = jnp.min(trueRS) > -thresh
    return jnp.where(jnp.logical_and(closeEnough, notTooFarInside), 1.0, 0)


trajectory_time_near_boundary_batch = jax.jit(
    jax.vmap(
        trajectory_time_near_boundary,
        in_axes=(
            0,  # start_pos
            0,  # heading
            None,  # pursuerXList
            None,  # speed
            None,  # trueParams
            None,  # center
            None,  # radius
            None,  # tmax
            None,  # num_points
            None,  # diff_threshold
            None,  # endPoints
            None,  # meanPursuerX
            None,  # measuredHeadings
            None,  # measuredPathHistories[:, 0] (if that's what your fn needs)
        ),
    ),
    static_argnames="numPoints",
)
trajectory_intercept_probability_batch = jax.jit(
    jax.vmap(
        trajectory_intercept_probability,
        in_axes=(
            0,  # start_pos
            0,  # heading
            None,  # pursuerXList
            None,  # speed
            None,  # trueParams
            None,  # center
            None,  # radius
            None,  # tmax
            None,  # num_points
            None,  # diff_threshold
            None,  # endPoints
            None,  # meanPursuerX
            None,  # measuredHeadings
            None,  # measuredPathHistories[:, 0] (if that's what your fn needs)
        ),
    ),
    static_argnames="numPoints",
)
trajectory_entropy_batch = jax.jit(
    jax.vmap(
        trajectory_entropy,
        in_axes=(
            0,  # start_pos
            0,  # heading
            None,  # pursuerXList
            None,  # speed
            None,  # trueParams
            None,  # center
            None,  # radius
            None,  # tmax
            None,  # num_points
            None,  # diff_threshold
            None,  # endPoints
            None,  # meanPursuerX
            None,  # measuredHeadings
            None,  # measuredPathHistories[:, 0] (if that's what your fn needs)
        ),
    ),
    static_argnames="numPoints",
)
trajectory_fisher_information_batch = jax.jit(
    jax.vmap(
        trajectory_fisher_information,
        in_axes=(
            0,  # start_pos (vary)
            0,  # heading (vary)
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ),
    static_argnames="numPoints",
)
info_for_trajectories_batch = jax.jit(
    jax.vmap(
        info_for_trajectories,
        in_axes=(0, None, None, None, None, None, None, None, None),
    )
)


def trajectory_dist_to_past(
    start_pos,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
    diff_threshold=0.3,
    pastInterceptedPoints=None,
    pastHeadings=None,
):
    dists = jax.vmap(angle_diff, in_axes=(0, None))(pastHeadings, heading)
    return jnp.min(jnp.abs(dists))


def tie_score_streaming(
    sel_positions,
    sel_headings,  # (N,2), (N,)
    pursuerXList,
    speed,
    trueParams,
    tmax,
    previousFims,
    flattenLearingLossAmount,
    meanPursuerX,
    covSqrt,
    batch_size=500,
    numPoints=100,
):
    N = sel_positions.shape[0]
    out_chunks = []
    for start in tqdm.tqdm(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        # Slice candidates only; everything else is reused
        scores_chunk = trajectory_fisher_information_batch(
            sel_positions[start:end],
            sel_headings[start:end],
            pursuerXList,
            speed,
            trueParams,
            tmax,
            numPoints,
            previousFims,
            flattenLearingLossAmount,
            meanPursuerX,
            covSqrt,
        )
        # If this is JAX, flush device work so memory is freed before next chunk:
        scores_chunk = jax.device_get(scores_chunk)
        out_chunks.append(scores_chunk)
    return jnp.concatenate([jnp.asarray(c) for c in out_chunks], axis=0)


def run_sacrificial_path_optimization(
    startingAngle,
    startingHeading,
    pursuerXList,
    trueParams,
    center,
    radius,
    speed,
    tmax,
    numPoints,
    previousFims,
    flattenLearingLossAmount=0.0,
    endPoints=None,
):
    def objfunc(xDict):
        x = xDict["x"]
        # loss = traj_info_objective(
        #     x,
        #     radius,
        #     center,
        #     pursuerXList,  # (K, d)
        #     speed,
        #     trueParams,
        #     tmax,
        #     numPoints,
        #     previousFims,
        #     flattenLearingLossAmount,
        #     meanPursuerX=None,
        #     covPursuerXSqrt=None,
        # )
        loss = which_max_distance_obj(
            x,
            radius,
            center,
            pursuerXList,  # (K, d)
            speed,
            trueParams,
            tmax,
            numPoints,
            previousFims,
            flattenLearingLossAmount,
            None,
            None,
            endPoints,
        )
        funcs = {}
        funcs["loss"] = loss
        return funcs, False

    def sens(xDict, funcs):
        x = xDict["x"]
        # grad_x = d_traj_info_objective(
        #     x,
        #     radius,
        #     center,
        #     pursuerXList,  # (K, d)
        #     speed,
        #     trueParams,
        #     tmax,
        #     numPoints,
        #     previousFims,
        #     flattenLearingLossAmount,
        #     meanPursuerX=None,
        #     covPursuerXSqrt=None,
        # )
        grad_x = which_max_distance_obj_derivative(
            x,
            radius,
            center,
            pursuerXList,  # (K, d)
            speed,
            trueParams,
            tmax,
            numPoints,
            previousFims,
            flattenLearingLossAmount,
            None,
            None,
            endPoints,
        )

        funcsSens = {}
        funcsSens["loss"] = {
            "x": grad_x,
        }
        return funcsSens, False

    optProb = Optimization("sacraficial path optimization", objfunc)
    optProb.addVarGroup(
        name="x",
        nVars=2,
        varType="c",
        value=np.array([startingAngle, startingHeading]),
        lower=[-np.pi, -np.pi],
        upper=[np.pi, np.pi],
    )
    optProb.addObj("loss", scale=1.0)
    opt = OPT("ipopt")
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 500
    # opt.options["warm_start_init_point"] = "yes"
    # opt.options["mu_init"] = 1e-1
    # opt.options["nlp_scaling_method"] = "gradient-based"
    opt.options["tol"] = 1e-8
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"

    sol = opt(optProb, sens=sens)
    # sol = opt(optProb, sens="fd")

    return sol.xStar["x"][0], sol.xStar["x"][1], sol.fStar


def optimize_next_low_priority_path(
    pursuerXList,
    headings,
    speeds,
    interceptedList,
    pathHistories,
    endPoints,
    endTimes,
    trueParams,
    center,
    radius,
    num_angles,
    num_headings,
    speed,
    tmax,
    num_points,
    rng,
    measuredHeadings=None,
    measuredSpeeds=None,
    measuredPathTimes=None,
    measuredInterceptedList=None,
    measuredPathHistories=None,
    measuredEndPoints=None,
    measuredEndTimes=None,
    measuredEzPoints=None,
    measuredEzTimes=None,
    flattenLearingLossAmount=0.0,
    previousCov=None,
    meanPursuerX=None,
):
    start = time.time()
    print("Optimizing next low-priority path...")

    # ----- random fallback (optional) -----
    if randomPath:
        upperTheta = jnp.pi * 2.0
        lowerTheta = 0.0
        best_angle = jax.random.uniform(rng, (), minval=lowerTheta, maxval=upperTheta)
        best_start_pos = center + radius * jnp.array(
            [jnp.cos(best_angle), jnp.sin(best_angle)]
        )
        heading_to_center = jnp.arctan2(
            center[1] - best_start_pos[1], center[0] - best_start_pos[0]
        )
        best_heading = heading_to_center + 0.2 * jax.random.normal(rng, ())
        return best_start_pos, best_heading

    # ----- candidate generation -----
    startHeadings = time.time()
    cand_positions, cand_headings = generate_headings_toward_center(
        radius,
        num_angles,
        num_headings,
        0.5,
        center=center,
        theta1=0.0,
        theta2=2 * jnp.pi,
        plot=False,
    )
    print("time to generate headings:", time.time() - startHeadings)

    diff_threshold = 10.0

    hueristic = interceptionOnBoundary
    if not interceptionOnBoundary and len(interceptedList) < 0:
        hueristic = True
    # if not interceptionOnBoundary:
    if not hueristic:
        # if True:
        # ----- gating by entropy (bits) -----
        # trajectory_entropy must return a scalar in [0,1] bits for each candidate
        gateStart = time.time()
        # gate_score = jnp.ones(len(cand_headings))  # placeholder: keep al

        gate_score = trajectory_entropy_batch(
            # gate_score = trajectory_intercept_probability_batch(
            # gate_score = trajectory_time_near_boundary_batch(
            cand_positions,
            cand_headings,
            pursuerXList,
            speed,
            trueParams,
            center,
            radius,
            tmax,
            num_points,
            diff_threshold,
            endPoints,
            meanPursuerX,
            measuredHeadings,
            measuredPathHistories[:, 0] if measuredPathHistories is not None else None,
        )

        print(
            "gate_score min/max:",
            float(jnp.nanmin(gate_score)),
            float(jnp.nanmax(gate_score)),
        )

        hmin_bits = -0.5  # minimum entropy to keep
        mask = jnp.isfinite(gate_score) & (gate_score >= hmin_bits)

        if not jnp.any(mask):
            # fallback: keep top-K by entropy to ensure candidates remain
            K = jnp.minimum(64, gate_score.shape[0])
            keep_idx = jnp.argsort(gate_score)[-K:]
        else:
            keep_idx = jnp.where(mask)[0]

        sel_positions = cand_positions[keep_idx]
        sel_headings = cand_headings[keep_idx]
        print("time to gate candidates:", time.time() - gateStart)
        print("candidates after gating:", int(sel_headings.shape[0]))

        # ----- prior FIMs (handle None) -----
        startFim = time.time()
        previousFims = info_for_trajectories_batch(
            pursuerXList,
            measuredHeadings,
            measuredSpeeds,
            measuredPathTimes,
            measuredInterceptedList,
            measuredPathHistories,
            measuredEndPoints,
            trueParams,
            flattenLearingLossAmount,
        )
        print("time to prepare previousFims:", time.time() - startFim)
        covSqrt = jnp.diag(jnp.sqrt(previousCov))

        # ----- score by global ΔI (your method) -----
        startTie = time.time()
        # tie_score = trajectory_time_near_boundary_batch(
        #     sel_positions,
        #     sel_headings,
        #     pursuerXList,
        #     speed,
        #     trueParams,
        #     center,
        #     radius,
        #     tmax,
        #     num_points,
        #     diff_threshold,
        #     endPoints,
        #     meanPursuerX,
        #     measuredHeadings,
        #     measuredPathHistories[:, 0],
        # )
        tie_score = tie_score_streaming(
            sel_positions,
            sel_headings,  # (N,2), (N,)
            pursuerXList,
            speed,
            trueParams,
            tmax,
            previousFims,
            flattenLearingLossAmount,
            meanPursuerX,
            covSqrt,
            batch_size=1000,
            numPoints=100,
        )
        print("time to score candidates:", time.time() - startTie)

        print(
            "tie_score min/max:",
            float(jnp.nanmin(tie_score)),
            float(jnp.nanmax(tie_score)),
        )

        local_best = jnp.nanargmax(tie_score)
        best_idx = keep_idx[local_best]
        best_start_pos = cand_positions[best_idx]

        angle = jnp.arctan2(
            best_start_pos[1] - center[1], best_start_pos[0] - center[0]
        )
        # angle, heading, score = run_sacrificial_path_optimization(
        #     angle,
        #     cand_headings[best_idx],
        #     pursuerXList,
        #     trueParams,
        #     center,
        #     radius,
        #     speed,
        #     tmax,
        #     num_points,
        #     previousFims,
        #     flattenLearingLossAmount,
        # )

    else:
        print("Using heuristic  scoring")
        # ----- boundary-mode branch (leave your scoring function here) -----
        # Ensure your which_lp_* fns are defined; typical pattern below:
        scores = jax.vmap(
            which_lp_path_maximizes_dist_to_next_intercept,
            # which_lp_path_minimizes_number_of_potential_solutions_must_intercect_all,
            in_axes=(0, 0, None, None, None, None, None, None),
        )(
            cand_positions,
            cand_headings,
            pursuerXList,
            speed,
            trueParams,
            tmax,
            num_points,
            endPoints[interceptedList] if jnp.any(interceptedList) else endPoints,
        )
        best_idx = jnp.nanargmax(scores)
        max_score = jnp.nanmax(scores)
        min_score = jnp.nanmin(scores)
        if jnp.isnan(max_score) or jnp.isnan(min_score):
            print("naN in scores, using fallback method")
            scores = jax.vmap(
                # inside_model_disagreement_score,
                which_lp_path_minimizes_number_of_potential_solutions,
                in_axes=(
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )(
                # angle_flat,
                # heading_flat,
                cand_positions,
                cand_headings,
                pursuerXList,
                speed,
                trueParams,
                center,
                radius,
                tmax,
                num_points,
                diff_threshold,
            )

        best_idx = jnp.nanargmax(scores)

    best_start_pos = cand_positions[best_idx]
    best_heading = cand_headings[best_idx]
    best_angle = jnp.arctan2(
        best_start_pos[1] - center[1], best_start_pos[0] - center[0]
    )
    # best_angle_opt, best_heading_opt, opt_score = run_sacrificial_path_optimization(
    #     0,
    #     0,
    #     pursuerXList,
    #     trueParams,
    #     center,
    #     radius,
    #     speed,
    #     tmax,
    #     num_points,
    #     None,
    #     flattenLearingLossAmount,
    #     endPoints[interceptedList],
    # )

    print("time to optimize next low-priority path:", time.time() - start)
    return best_start_pos, best_heading
    # start = time.time()
    # #     initialGradient = dTotalLossDX(
    # #         initialPursuerX,
    # #         headings,
    # #         speeds,
    # #         interceptedList,
    # #         pathTimes,
    # #         pathHistories,
    # #         endPoints,
    # #         ezPoints,
    # #         ezTimes,
    # #         trueParams,
    # #         interceptedPathWeight,
    # #         flattenLearingLossAmount,
    # #     )
    # upperTheta = jnp.pi + jnp.pi  # / 4
    # lowerTheta = jnp.pi - jnp.pi  # / 4
    # if randomPath:
    #     best_angle = rng.uniform(lowerTheta, upperTheta)
    #     best_start_pos = center + radius * jnp.array(
    #         [jnp.cos(best_angle), jnp.sin(best_angle)]
    #     )
    #     headingToCenter = np.arctan2(
    #         center[1] - best_start_pos[1],  # Δy
    #         center[0] - best_start_pos[0],  # Δx
    #     )
    #     best_heading = headingToCenter + rng.normal(0.0, 0.2)
    #     return best_start_pos, best_heading

    #     else:
    #         scores = jax.vmap(
    #             # inside_model_disagreement_score,
    #             which_lp_path_minimizes_number_of_potential_solutions_must_intercect_all,
    #             # which_lp_path_maximizes_dist_to_next_intercept,
    #             # which_lp_spends_most_time_in_all_rs,
    #             in_axes=(
    #                 0,
    #                 0,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #             ),
    #         )(
    #             # angle_flat,
    #             # heading_flat,
    #             positions,
    #             headings,
    #             pursuerXList,
    #             speed,
    #             trueParams,
    #             center,
    #             radius,
    #             tmax,
    #             num_points,
    #             diff_threshold,
    #             endPoints[interceptedList],
    #         )
    #
    #     max_score = jnp.nanmax(scores)
    #     min_score = jnp.nanmin(scores)
    #     if jnp.isnan(max_score) or jnp.isnan(min_score):
    #         print("naN in scores, using fallback method")
    #         scores = jax.vmap(
    #             # inside_model_disagreement_score,
    #             which_lp_path_minimizes_number_of_potential_solutions,
    #             in_axes=(
    #                 0,
    #                 0,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #             ),
    #         )(
    #             # angle_flat,
    #             # heading_flat,
    #             positions,
    #             headings,
    #             pursuerXList,
    #             speed,
    #             trueParams,
    #             center,
    #             radius,
    #             tmax,
    #             num_points,
    #             diff_threshold,
    #         )
    #
    #     best_idx = jnp.nanargmax(scores)
    #
    # best_start_pos = positions[best_idx]
    # best_heading = headings[best_idx]
    #
    # print("time to optimize next low-priority path:", time.time() - start)
    # return best_start_pos, best_heading
    #


def plot_true_and_learned_pursuer(
    pursuerPosition, pursuerHeading, pursuerPositionLearned, pursuerHeadingLearned, ax
):
    # plot true pursuer position and heading
    ax.scatter(
        pursuerPosition[0],
        pursuerPosition[1],
        color="blue",
        marker="o",
        label="True Pursuer Position",
    )
    ax.quiver(
        pursuerPosition[0],
        pursuerPosition[1],
        np.cos(pursuerHeading),
        np.sin(pursuerHeading),
        color="blue",
        angles="xy",
        scale_units="xy",
        scale=1,
        label="True Pursuer Heading",
    )
    # plot learned pursuer position and heading
    ax.scatter(
        pursuerPositionLearned[0],
        pursuerPositionLearned[1],
        color="orange",
        marker="o",
        label="Learned Pursuer Position",
    )
    ax.quiver(
        pursuerPositionLearned[0],
        pursuerPositionLearned[1],
        np.cos(pursuerHeadingLearned),
        np.sin(pursuerHeadingLearned),
        color="orange",
        angles="xy",
        scale_units="xy",
        scale=1,
        label="Learned Pursuer Heading",
    )


def make_axes(numPlots):
    # pick 1 or 2 rows
    nrows = min(2, numPlots)
    # compute how many columns you need
    ncols = int(np.ceil(numPlots / nrows))
    # always return a 2D array of Axes
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    # flatten and discard any extra axes
    axes = axes.flatten()[:numPlots]
    return fig, axes


def plot_all(
    startPositions,
    interceptedList,
    endPoints,
    pathHistories,
    pursuerPosition,
    pursuerHeading,
    pursuerRange,
    pursuerTurnRadius,
    headings,
    speeds,
    pursuerXList,
    lossList,
    trueParams,
    meanPursuerX,
    ezPoints,
    plotLPPaths=True,
    ax=None,
    fig=None,
):
    numPlots = len(pursuerXList)
    # make 2 rows and ceil(numPlots/2) columns
    # numPlots = 10
    numPlots = min(numPlots, 10)
    # fig1, axes = make_axes(numPlots)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    # colors = ["blue", "orange", "red", "purple", "brown", "pink", "gray"]

    alpha = 1.0 / len(pursuerXList)
    # for i in range(numPlots):
    # plot_low_priority_paths_with_ez(
    #     headings,
    #     speeds,
    #     startPositions,
    #     interceptedList,
    #     endPoints,
    #     pathHistories,
    #     trueParams[parameterMask],
    #     trueParams,
    #     ax,
    # )
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)

    for i in range(len(pursuerXList)):
        # pick ax
        # ax = axes[i]

        pursuerX = pursuerXList[i].squeeze()
        (
            pursuerPositionLearned1,
            pursuerHeadingLearned1,
            pursuerSpeedLearned1,
            minimumTurnRadiusLearned1,
            pursuerRangeLearned1,
        ) = pursuerX_to_params(pursuerX, trueParams)
        dubinsEZ.plot_dubins_reachable_set(
            pursuerPositionLearned1,
            pursuerHeadingLearned1,
            pursuerRangeLearned1,
            minimumTurnRadiusLearned1,
            ax,
            colors=["magenta"],
            # colors=[colors[i % len(colors)]],
            alpha=0.1,
        )
        # plot_true_and_learned_pursuer(
        #     pursuerPosition,
        #     pursuerHeading,
        #     pursuerPositionLearned1,
        #     pursuerHeadingLearned1,
        #     ax,
        # )
    # (
    #     pursuerPositionLearned1,
    #     pursuerHeadingLearned1,
    #     pursuerSpeedLearned1,
    #     minimumTurnRadiusLearned1,
    #     pursuerRangeLearned1,
    # ) = pursuerX_to_params(meanPursuerX, trueParams)
    # dubinsEZ.plot_dubins_reachable_set(
    #     pursuerPositionLearned1,
    #     pursuerHeadingLearned1,
    #     pursuerRangeLearned1,
    #     minimumTurnRadiusLearned1,
    #     ax,
    #     colors=["orange"],
    #     # colors=[colors[i % len(colors)]],
    #     alpha=1,
    # )
    if plotLPPaths:
        plot_low_priority_paths(
            startPositions, interceptedList, endPoints, pathHistories, ax
        )

    # proxies for legend
    linewidithSize = 5
    ax.plot(
        [],
        [],
        color="blue",
        label="Intercepted Trajectory",
        linewidth=linewidithSize,
    )
    ax.plot(
        [],
        [],
        color="green",
        label="Survived Trajectory",
        linewidth=linewidithSize,
    )
    ax.plot([], [], color="red", label="True RS", linewidth=linewidithSize)
    ax.plot(
        [],
        [],
        color="magenta",
        label="Feasible Learned RS",
        linewidth=linewidithSize,
    )
    ax.plot(
        [],
        [],
        color="orange",
        label="Mean Learned RS",
        linewidth=linewidithSize,
    )
    # ax.legend(loc="lower left")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("True and Learned Pursuer Reachable Sets")
    dubinsEZ.plot_dubins_reachable_set(
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerTurnRadius,
        ax,
        colors=["red"],
    )

    # ax.scatter(ezPoints[:, 0], ezPoints[:, 1], color="green", s=50, label="EZ Points")
    # for i in range(len(headings)):
    #     dubinsEZ.plot_dubins_EZ(
    #         trueParams[0:2],
    #         trueParams[2],
    #         trueParams[3],
    #         trueParams[4],
    #         0.0,
    #         trueParams[5],
    #         headings[i],
    #         speeds[i],
    #         ax,
    #     )
    return fig, ax


def plot_pursuer_parameters_spread(
    pursuerParameter_history,
    pursuerParameterMean_history,
    pursuerParameterVariance_history,
    lossList_history,
    trueParams,
    numOptimizerStarts,
    numLowPriorityAgents,
    keepLossThreshold,
):
    beta = 3.0
    xData = np.arange(1, len(pursuerParameter_history) + 1)  # Start at 1

    if not positionAndHeadingOnly:
        # (title, index in estimated params, index in trueParams)
        if knownSpeed:
            param_info = [
                ("Pursuer X Position", 0, 0),
                ("Pursuer Y Position", 1, 1),
                ("Pursuer Heading", 2, 2),
                # ("Pursuer Speed", 3, 3),  # omitted
                ("Pursuer Turn Radius", 3, 4),
                ("Pursuer Range", 4, 5),
            ]
            fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
            axes = axes.flatten()
        else:
            param_info = [
                ("Pursuer X Position", 0, 0),
                ("Pursuer Y Position", 1, 1),
                ("Pursuer Heading", 2, 2),
                ("Pursuer Speed", 3, 3),  # omitted
                ("Pursuer Turn Radius", 4, 4),
                ("Pursuer Range", 5, 5),
            ]
            fig, axes = plt.subplots(3, 2, figsize=(8, 6), sharex=True)
            axes = axes.flatten()

        for idx, (title, lidx, tidx) in enumerate(param_info):
            ax = axes[idx]
            ax.set_title(title)
            ax.set_xticks(xData)
            ax.axhline(trueParams[tidx], color="r", linestyle="--", label="True Value")
            mean = pursuerParameterMean_history[:, lidx]
            std = jnp.sqrt(pursuerParameterVariance_history[:, lidx])
            ax.plot(xData, mean, "b-", label="Mean")
            ax.fill_between(
                xData,
                mean - beta * std,
                mean + beta * std,
                color="b",
                alpha=0.2,
                label="±3σ",
            )

            for i in range(numOptimizerStarts):
                mask = lossList_history[:, i] <= keepLossThreshold
                c = jnp.clip(lossList_history[:, i][mask], 0.0, keepLossThreshold)
                ax.scatter(
                    xData[mask],
                    pursuerParameter_history[:, i, lidx][mask],
                    c=c,
                    s=10,
                    vmin=0.0,
                    vmax=keepLossThreshold,
                )

        axes[-2].set_xlabel("Num Sacrificial Agents")
        axes[-1].set_xlabel("Num Sacrificial Agents")
        axes[0].legend()
        fig.tight_layout()

    else:
        param_info = [
            ("Pursuer X Position", 0, 0),
            ("Pursuer Y Position", 1, 1),
            ("Pursuer Heading", 2, 2),
        ]
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        for ax, (title, lidx, tidx) in zip(axes, param_info):
            ax.set_title(title)
            ax.set_xticks(xData)
            ax.axhline(trueParams[tidx], color="r", linestyle="--", label="True Value")
            mean = pursuerParameterMean_history[:, lidx]
            std = jnp.sqrt(pursuerParameterVariance_history[:, lidx])
            ax.plot(xData, mean, "b-", label="Mean")
            ax.fill_between(
                xData,
                mean - beta * std,
                mean + beta * std,
                color="b",
                alpha=0.2,
                label="±3σ",
            )

            for i in range(numOptimizerStarts):
                mask = lossList_history[:, i] <= keepLossThreshold
                c = jnp.clip(lossList_history[:, i][mask], 0.0, keepLossThreshold)
                ax.scatter(
                    xData[mask],
                    pursuerParameter_history[:, i, lidx][mask],
                    c=c,
                    s=10,
                    vmin=0.0,
                    vmax=keepLossThreshold,
                )

        axes[-1].set_xlabel("Num Sacrificial Agents")
        axes[0].legend()
        fig.tight_layout()


# def get_unique_rows_by_proximity(arr, lossList, rtol=1e-3):
#     unique_list = []
#     unique_loss_list = []
#     for i, row in enumerate(arr):
#         # if not any(np.all(np.isclose(row, u, rtol=rtol)) for u in unique_list):
#         if not any(np.linalg.norm(row - u) < 1e-1 for u in unique_list):
#             unique_list.append(row)
#             unique_loss_list.append(lossList[i])
#     return np.array(unique_list), np.array(unique_loss_list)
def get_unique_rows_by_proximity(arr, lossList, rtol=1e-3):
    unique_list = []
    unique_loss_list = []

    for i, row in enumerate(arr):
        is_unique = True
        for u in unique_list:
            # Compute Euclidean distance, but use wrapped difference for angle (3rd element)
            diff = row - u
            diff[2] = angle_diff(row[2], u[2])
            if np.linalg.norm(diff) < rtol:
                is_unique = False
                break
        if is_unique:
            unique_list.append(row)
            unique_loss_list.append(lossList[i])

    return np.array(unique_list), np.array(unique_loss_list)


def angle_diff(theta1, theta2):
    """Compute smallest difference between two angles in radians."""
    return jnp.arctan2(jnp.sin(theta1 - theta2), jnp.cos(theta1 - theta2))


def compute_abs_error_history(mean_history, true_params, parameterMask):
    trueParams_np = true_params[parameterMask]
    errors = np.abs(mean_history - trueParams_np)

    # Correct heading error if heading is included (assume index 2 in full params)
    heading_index = (
        np.flatnonzero(parameterMask)[2] if np.sum(parameterMask) > 2 else None
    )

    if heading_index is not None:
        for i in range(mean_history.shape[0]):
            errors[i, 2] = np.abs(angle_diff(mean_history[i, 2], trueParams_np[2]))

    return errors


def compute_rmse_history(mean_history, true_params, parameterMask):
    trueParams_np = true_params[parameterMask]
    errors = mean_history - trueParams_np  # shape: (T, D)

    # Correct heading error if heading is included (assume index 2 in full params)
    heading_index = (
        np.flatnonzero(parameterMask)[2] if np.sum(parameterMask) > 2 else None
    )

    if heading_index is not None:
        for i in range(mean_history.shape[0]):
            errors[i, 2] = angle_diff(mean_history[i, 2], trueParams_np[2])

    rmse_history = np.sqrt(np.mean(errors**2, axis=1))
    return rmse_history


def find_learning_loss_flatten_amount(cov, beta):
    eigvals, eigvecs = np.linalg.eig(cov)
    max_eigval = eigvals.max()
    return beta * np.sqrt(max_eigval)  # Adjusted scaling factor


def wrap_to_pi(angle):
    """Wrap angle to [-π, π)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def fit_von_mises_with_variance(angles):
    """
    Fit a von Mises distribution to angle data and return:
      - mean direction,
      - concentration (kappa),
      - circular variance,
      - circular standard deviation (radians)

    Parameters:
        angles (np.ndarray): Angles in radians, in [-π, π) or [0, 2π)

    Returns:
        mu (float): Mean direction in radians, wrapped to [-π, π)
        kappa (float): Concentration parameter
        circular_variance (float): 1 - R, in [0, 1]
        std_dev (float): Circular standard deviation in radians
    """
    # Mean resultant vector
    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))
    mu = np.arctan2(S, C)
    R = np.sqrt(C**2 + S**2)
    circular_variance = 1 - R

    # Approximate κ using Mardia & Jupp
    def A1inv(R):
        if R < 1e-6:
            return 0.0
        elif R < 0.53:
            return 2 * R + R**3 + (5 * R**5) / 6
        elif R < 0.85:
            return -0.4 + 1.39 * R + 0.43 / (1 - R)
        else:
            return 1 / (R**3 - 4 * R**2 + 3 * R)

    kappa = A1inv(R)

    # Circular standard deviation
    if R < 1e-12:
        std_dev = np.pi / np.sqrt(3)  # Max entropy value ~1.81
    else:
        std_dev = np.sqrt(-2 * np.log(R))

    return wrap_to_pi(mu), kappa, circular_variance, std_dev


def summarize_solutions(
    X,  # shape [M, D]
    angle_idx=2,  # which column is heading
    weights=None,  # optional weights, shape [M]
):
    X = np.asarray(X)
    M, D = X.shape
    if weights is None:
        w = np.ones(M) / M
    else:
        w = np.asarray(weights).astype(float)
        w = w / (w.sum() + 1e-12)

    # 1) Circular stats for angle
    angles = X[:, angle_idx]
    # Use your function; assuming it returns mean, kappa, variance, std
    angleMean, angleKappa, angleVariance, angleStd = fit_von_mises_with_variance(angles)

    # 2) Linear means for the rest
    mean = (w[:, None] * X).sum(axis=0)
    mean[angle_idx] = angleMean  # replace with circular mean

    # 3) Build residuals with wrapped angle residuals
    R = X - mean  # linear residuals
    R[:, angle_idx] = wrap_to_pi(angles - angleMean)

    # 4) Weighted covariance (full), unbiased-ish correction
    #    (you can drop the correction if you prefer MLE)
    ess = 1.0 / (w**2).sum()  # effective sample size
    C = (w[:, None, None] * (R[:, :, None] * R[:, None, :])).sum(axis=0)
    C *= ess / max(ess - 1.0, 1.0)  # small-sample correction

    # 5) Replace the angle variance on the diagonal with circular variance
    #    (keeps covariance with other dims from the wrapped residuals)
    C[angle_idx, angle_idx] = angleVariance

    var = np.diag(C).copy()
    std = np.sqrt(var)
    std[angle_idx] = angleStd  # ensure std matches circular std

    return mean, C, var, std, angleMean, angleKappa


def find_mean_and_std(pursuerXListZeroLoss):
    # Track stats
    angles = pursuerXListZeroLoss[:, 2]
    angleMean, angleKappa, angleVariance, angleStd = fit_von_mises_with_variance(angles)
    mean = np.mean(pursuerXListZeroLoss, axis=0)
    if len(pursuerXListZeroLoss) == 1:
        cov = np.zeros_like(mean)
    else:
        cov = np.cov(pursuerXListZeroLoss, rowvar=False).diagonal().copy()
    std = np.sqrt(cov)
    std[2] = angleStd  # Replace heading std with circular std

    mean[2] = angleMean  # Replace heading with wrapped mean
    # cov[2] = angleVariance  # Replace heading variance with circular variance
    return mean, cov, std, angleMean, angleKappa


def plan_path_around_all_learned_pursuer_params(
    pursuerXList, trueParams, previousSpline
):
    startingLocation = np.array([-5.0, -5.0])
    endingLocation = np.array([5.0, 5.0])
    initialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)
    initialVelocity = endingLocation - startingLocation

    agentSpeed = 1.0
    numControlPoints = 8
    splineOrder = 3
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)
    velocity_constraints = (agentSpeed - 0.01, agentSpeed + 0.01)

    num_constraint_samples = 50

    start = time.time()
    spline, pathTime, violatedTrueEZ = (
        learned_dubins_ez_path_planner.optimize_spline_path(
            startingLocation,
            endingLocation,
            initialVelocity,
            numControlPoints,
            splineOrder,
            velocity_constraints,
            turn_rate_constraints,
            curvature_constraints,
            num_constraint_samples,
            pursuerXList,
            trueParams,
            agentSpeed,
            previousSpline,
        )
    )
    print("time to plan path around all learned pursuer params:", time.time() - start)
    print("planned path time:", pathTime)
    print("violated true EZ:", violatedTrueEZ)
    if plotAllFlag:
        fig, ax = plt.subplots()
        learned_dubins_ez_path_planner.plot_spline(
            spline,
            ax,
        )
        plot_all(
            None,
            None,
            None,
            None,
            trueParams[0:2],
            trueParams[2],
            trueParams[5],
            trueParams[4],
            None,
            None,
            pursuerXList,
            None,
            trueParams,
            None,
            None,
            plotLPPaths=False,
            ax=ax,
            fig=fig,
        )

        plt.show()
    return spline, pathTime, violatedTrueEZ


def run_simulation_with_random_pursuer(
    lower_bounds_all,
    upper_bounds_all,
    parameterMask,
    seed=0,
    numLowPriorityAgents=6,
    numOptimizerStarts=200,
    keepLossThreshold=1e-5,
    plotEvery=1,
    # dataDir="results",
    # saveDir="run",
):
    rng = np.random.default_rng(seed)
    trueParams = np.array(rng.uniform(lower_bounds_all, upper_bounds_all))

    pursuerPosition = np.array([trueParams[0], trueParams[1]])
    pursuerHeading = trueParams[2]
    pursuerSpeed = trueParams[3]
    pursuerTurnRadius = trueParams[4]
    pursuerRange = trueParams[5]
    pursuerCaptureRadius = 0.0
    agentSpeed = 1.0
    dt = 0.01
    searchCircleCenter = np.array([0, 0])
    searchCircleRadius = upper_bounds_all[5] + upper_bounds_all[0] + 1
    tmax = (2 * searchCircleRadius) / agentSpeed
    numPoints = int(tmax / dt) + 1

    if noisyMeasurementsFlag:
        lowPriorityAgentPositionCov = np.array([[0.001, 0.0], [0.0, 0.001]])
        lowPriorityAgentTimeVar = 0.001
        flattenLearingLossAmount = find_learning_loss_flatten_amount(
            lowPriorityAgentPositionCov, beta=2.3
        )
        maxStdDevThreshold = 0.05
    else:
        lowPriorityAgentPositionCov = np.array([[0.0, 0.0], [0.0, 0.0]])
        lowPriorityAgentTimeVar = 0.0
        flattenLearingLossAmount = 0.0
        maxStdDevThreshold = 0.05
    # flattenLearingLossAmount = 0.0

    # Init histories
    interceptedList, endPoints, endTimes = [], [], []
    pathTimes = []
    interceptionPointEZList, interceptionTimeEZList = [], []
    pathHistories, startPositions, headings, speeds = [], [], [], []
    pursuerParameter_history, pursuerParameterMean_history = [], []
    pursuerParameterVariance_history, lossList_history = [], []
    pursuerParameterStdDev_history = []
    downSampledPathHistories = []
    downSampledPathTimes = []
    percentOfTrueRsCovered = []
    proportionOfAreaCoverdHistory = []
    hpOptPathTimes = []
    hpViolatedTrueEZs = []

    pursuerXListZeroLoss = None
    pursuerXListZeroLossCollapsed = None
    i = 0
    singlePursuerX = False
    mean = None
    cov = None
    angleKappa = None
    angleMean = None
    splinePath = None

    while i < numLowPriorityAgents and not singlePursuerX:
        # keepLossThreshold = 2 * (i + 1) * lossThreshAmount
        print("num agents:", i + 1)
        # print("keepLossThreshold:", keepLossThreshold)
        if i == 0:
            startPosition = jnp.array([-searchCircleRadius, 0.0001])
            heading = 0.01

        else:
            searchCenter = (
                jnp.mean(pursuerXListZeroLoss[:, :2], axis=0)
                if pursuerXListZeroLoss is not None
                else searchCircleCenter
            )
            # searchCenter = searchCircleCenter
            numKeep = np.minimum(100, len(pursuerXListZeroLossCollapsed))
            tempPurseurX = pursuerXListZeroLossCollapsed[
                (
                    np.random.choice(
                        len(pursuerXListZeroLossCollapsed), numKeep, replace=False
                    ),
                )
            ]
            startPosition, heading = optimize_next_low_priority_path(
                # pursuerXListZeroLossCollapsed[:numKeep],
                tempPurseurX,
                jnp.array(headings),
                jnp.array(speeds),
                jnp.array(interceptedList),
                jnp.array(downSampledPathHistories),
                jnp.array(endPoints),
                jnp.array(endTimes),
                trueParams,
                searchCenter,
                searchCircleRadius,
                36,
                36,
                # 1,
                # 1,
                agentSpeed,
                tmax,
                numPoints // 10,
                rng,
                jnp.array(headings),
                jnp.array(speeds),
                jnp.array(downSampledPathTimes),
                jnp.array(interceptedList),
                jnp.array(downSampledPathHistories),
                jnp.array(endPoints),
                jnp.array(endTimes),
                jnp.array(interceptionPointEZList),
                jnp.array(interceptionTimeEZList),
                flattenLearingLossAmount=flattenLearingLossAmount,
                previousCov=cov,
                meanPursuerX=mean,
            )

        # Send agent
        startPositions.append(startPosition)
        headings.append(heading)
        speeds.append(agentSpeed)

        if interceptionOnBoundary:
            (
                pathTime,
                intercepted,
                endPoint,
                endTime,
                pathHistory,
                interceptionPointEZ,
                interceptionTimeEZ,
            ) = send_low_priority_agent_boundary(
                startPosition,
                heading,
                agentSpeed,
                pursuerPosition,
                pursuerHeading,
                pursuerTurnRadius,
                pursuerCaptureRadius,
                pursuerRange,
                pursuerSpeed,
                tmax,
                numPoints,
            )
        else:
            (
                pathTime,
                intercepted,
                endPoint,
                endTime,
                pathHistory,
                interceptionPointEZ,
                interceptionTimeEZ,
            ) = send_low_priority_agent_interioir(
                startPosition,
                heading,
                agentSpeed,
                pursuerPosition,
                pursuerHeading,
                pursuerTurnRadius,
                pursuerCaptureRadius,
                pursuerRange,
                pursuerSpeed,
                tmax,
                numPoints,
                rng=rng,
            )

        endPoint = endPoint + rng.multivariate_normal(
            np.zeros(2), lowPriorityAgentPositionCov
        )
        pathHistory = pathHistory + rng.multivariate_normal(
            np.zeros(2), lowPriorityAgentPositionCov, size=(numPoints,)
        )
        interceptionPointEZ = interceptionPointEZ + rng.multivariate_normal(
            np.zeros(2), lowPriorityAgentPositionCov
        )
        interceptionTimeEZ = interceptionTimeEZ + rng.normal(
            0.0, lowPriorityAgentTimeVar
        )

        pathTimes.append(pathTime)
        interceptedList.append(intercepted)
        endPoints.append(endPoint)
        endTimes.append(endTime)
        pathHistories.append(pathHistory)
        interceptionPointEZList.append(interceptionPointEZ)
        interceptionTimeEZList.append(interceptionTimeEZ)
        down = 10
        downSampledPathHistories.append(pathHistory[::down])
        downSampledPathTimes.append(pathTime[::down])

        # Learn
        pursuerXList, lossList, lossListNoFlatten = learn_ez(
            jnp.array(headings),
            jnp.array(speeds),
            jnp.array(pathTimes),
            jnp.array(interceptedList),
            jnp.array(pathHistories),
            jnp.array(endPoints),
            jnp.array(endTimes),
            jnp.array(interceptionPointEZList),
            jnp.array(interceptionTimeEZList),
            trueParams,
            pursuerXListZeroLoss,
            None,
            cov,
            angleMean,
            angleKappa,
            numOptimizerStarts,
            lowerLimit=lower_bounds_all[parameterMask],
            upperLimit=upper_bounds_all[parameterMask],
            interceptedPathWeight=1.0,
            flattenLearningLossAmount=flattenLearingLossAmount,
            useGaussianSampling=True,
            rng=rng,
        )

        # Filter good fits
        # pursuerXListZeroLoss = pursuerXList[lossList <= keepLossThreshold]
        # lossListZeroLoss = lossList[lossList <= keepLossThreshold]
        if len(pursuerXList[lossList <= keepLossThreshold]) == 0:
            print("No good models found. trying again with simpler loss")

            pursuerXList, lossList, lossListNoFlatten = learn_ez(
                jnp.array(headings),
                jnp.array(speeds),
                jnp.array(pathTimes),
                jnp.array(interceptedList),
                jnp.array(pathHistories),
                jnp.array(endPoints),
                jnp.array(endTimes),
                jnp.array(interceptionPointEZList),
                jnp.array(interceptionTimeEZList),
                trueParams,
                pursuerXListZeroLoss,
                None,
                cov,
                angleMean,
                angleKappa,
                numOptimizerStarts,
                lowerLimit=lower_bounds_all[parameterMask],
                upperLimit=upper_bounds_all[parameterMask],
                interceptedPathWeight=0.0,
                flattenLearningLossAmount=flattenLearingLossAmount,
                useGaussianSampling=True,
            )
            pursuerXListZeroLoss = pursuerXList[lossList <= keepLossThreshold]
            lossListZeroLoss = lossList[lossList <= keepLossThreshold]
            if len(lossListZeroLoss) == 0:
                print("No good models found even with simpler loss. Stopping.")
                break
        pursuerXListZeroLoss = pursuerXList[lossList <= keepLossThreshold]
        lossListZeroLoss = lossList[lossList <= keepLossThreshold]
        if planHPPath:
            splinePath, optPathTime, violatedTrueEZ = (
                plan_path_around_all_learned_pursuer_params(
                    pursuerXListZeroLoss, trueParams, splinePath
                )
            )
            hpOptPathTimes.append(optPathTime)
            hpViolatedTrueEZs.append(violatedTrueEZ)

        percent, proportionOfAreaCoverd = percent_of_true_rs_covered(
            trueParams,
            pursuerXListZeroLoss,
            1000,
            startPositions,
            interceptedList,
            endPoints,
            pathHistories,
            pursuerPosition,
            pursuerHeading,
            pursuerRange,
            pursuerTurnRadius,
            headings,
            speeds,
            # pursuerXListZeroLoss,
            lossListZeroLoss,
            trueParams,
            mean,
        )
        percentOfTrueRsCovered.append(percent)
        proportionOfAreaCoverdHistory.append(proportionOfAreaCoverd)

        # fig, ax = plt.subplots()
        # plt.hist(
        #     pursuerXListZeroLoss[:, 2], bins=10, alpha=0.5, label="Learned Headings"
        # )

        pursuerXListZeroLossCollapsed, _ = get_unique_rows_by_proximity(
            pursuerXListZeroLoss, lossListZeroLoss, rtol=0.01
        )
        # pursuerXListZeroLossCollapsed = pursuerXListZeroLoss
        print("num zero loss pursuerXList:", len(pursuerXListZeroLoss))
        print("num collapsed pursuerXListZeroLoss:", len(pursuerXListZeroLossCollapsed))
        mean, cov, std, angleMean, angleKappa = find_mean_and_std(pursuerXListZeroLoss)

        print("trueParams", trueParams)
        print("mean:", mean)
        print("std dev", std)
        pursuerParameterMean_history.append(mean)
        pursuerParameterVariance_history.append(cov)
        pursuerParameterStdDev_history.append(std)
        pursuerParameter_history.append(pursuerXList)
        lossList_history.append(lossList)
        #####plot_loss
        # if i > 20:
        #    ijPairs = [[3, 4]]
        #    for ij in ijPairs:
        #        title = f"Loss Contour for {i} Agents"
        #        plot_contour_of_loss(
        #            jnp.array(headings),
        #            jnp.array(speeds),
        #            jnp.array(pathTimes),
        #            jnp.array(interceptedList),
        #            jnp.array(pathHistories),
        #            jnp.array(endPoints),
        #            jnp.array(endTimes),
        #            jnp.array(interceptionPointEZList),
        #            jnp.array(interceptionTimeEZList),
        #            trueParams,
        #            pursuerXListZeroLoss[0],
        #            mean - 7 * std,
        #            mean + 7 * std,
        #            pursuerXList[lossList <= 1e-8],
        #            interceptedPathWeight=1.0,
        #            flattenLearningLossAmount=flattenLearingLossAmount,
        #            parameterIndexi=ij[0],
        #            parameterIndexj=ij[1],
        #            title=title,
        #        )
        #
        # Plot
        if i % plotEvery == 0 and plotAllFlag:
            fig, ax = plot_all(
                startPositions,
                interceptedList,
                endPoints,
                pathHistories,
                pursuerPosition,
                pursuerHeading,
                pursuerRange,
                pursuerTurnRadius,
                headings,
                speeds,
                # pursuerXListZeroLoss,
                pursuerXList[lossList <= keepLossThreshold],
                lossListZeroLoss,
                trueParams,
                mean,
                np.array(interceptionPointEZList),
            )
            fig.savefig(f"video/{i}.png")
            plt.close(fig)
            # close all figures to save memory
            # plt.close("all")

        i += 1
        # singlePursuerX = len(pursuerXList) == 1
        print("max std dev:", np.max(std))
        singlePursuerX = std.max() < maxStdDevThreshold
        if len(lossListZeroLoss) <= 1:
            break

    # === Final processing ===
    mean_history = np.array(pursuerParameterMean_history)
    trueParams_np = np.array(trueParams)

    # abs_error_history = np.abs(mean_history - trueParams_np[parameterMask])
    print("mean_estimates", mean_history[-1])
    print("trueParams_np", trueParams_np[parameterMask])
    # rmse_history = np.sqrt(
    #     np.mean((mean_history - trueParams_np[parameterMask]) ** 2, axis=1)
    # )
    rmse_history = compute_rmse_history(mean_history, trueParams_np, parameterMask)
    abs_error_history = compute_abs_error_history(
        mean_history, trueParams_np, parameterMask
    )
    print("final error", abs_error_history[-1])
    print("final rmse", rmse_history[-1])
    if plotAllFlag:
        plot_pursuer_parameters_spread(
            np.array(pursuerParameter_history),
            np.array(pursuerParameterMean_history),
            np.array(pursuerParameterVariance_history),
            np.array(lossList_history),
            trueParams,
            numOptimizerStarts,
            len(pursuerParameterMean_history),
            keepLossThreshold,
        )

    result = {
        "true_params": trueParams_np.tolist(),
        "interceptedList": np.array(interceptedList).tolist(),
        "mean_estimates": mean_history.tolist(),
        "cov_estimates": np.array(pursuerParameterVariance_history).tolist(),
        "std_dev_estimates": np.array(pursuerParameterStdDev_history).tolist(),
        "absolute_errors": abs_error_history.tolist(),
        "rmse_history": rmse_history.tolist(),
        "percent_of_true_rs_covered": percentOfTrueRsCovered,
        "proportion_of_area_covered": proportionOfAreaCoverdHistory,
        "hp_opt_path_times": hpOptPathTimes,
        "hp_violated_true_ezs": hpViolatedTrueEZs,
    }

    if saveResults:
        os.makedirs(saveDir, exist_ok=True)
        os.makedirs(f"{dataDir}/{saveDir}", exist_ok=True)
        with open(f"{dataDir}/{saveDir}/{seed}_results.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


def main(seed):
    # Define parameter bounds
    lowerBoundsAll = np.array([-2.0, -2.0, -np.pi, 2.0, 0.4, 2.0])
    upperBoundsAll = np.array([2.0, 2.0, np.pi, 3.0, 1.5, 3.0])

    # Determine which parameters are learnable

    lowerBounds = lowerBoundsAll[parameterMask]
    upperBounds = upperBoundsAll[parameterMask]

    print(f"Running simulation with seed {seed}")
    run_simulation_with_random_pursuer(
        lowerBoundsAll,
        upperBoundsAll,
        parameterMask,
        seed=seed,
        numLowPriorityAgents=25,
        numOptimizerStarts=100,
        keepLossThreshold=1e-4,
        plotEvery=1,
        # dataDir="results",
        # saveDir="boundary/unknownSpeed",
        # saveDir="knownSpeed",
    )


def violin_with_percentiles(groups, data_by_group, percentiles=(25, 50, 75), ax=None):
    """
    groups: list of labels
    data_by_group: list of 1D arrays matching groups
    """
    parts = ax.violinplot(data_by_group, showextrema=False)
    # overlay percentile lines
    for i, arr in enumerate(data_by_group, start=1):
        for p in percentiles:
            v = np.percentile(arr, p)
            ax.hlines(v, i - 0.25, i + 0.25, linewidth=2)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_title("Violin plots with percentile lines")
    return ax


def load_rs_history_data(results_dir, max_steps):
    p_histories = []

    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            if int(filename.split("_")[0]) <= 500:
                filepath = os.path.join(results_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                    p_history = data.get("percent_of_true_rs_covered", [])
                    if np.any(np.array(p_history) < 0.95):
                        print(
                            "low coverage found in file",
                            filename,
                            np.min(np.array(p_history)),
                        )
                    for p in p_history:
                        p_histories.append(p)
    p_histories = np.array(p_histories) * 100.0
    p_histories = p_histories.flatten()
    return p_histories
    print(
        "percenet above .90",
        len(p_histories[p_histories > 0.95 * 100]) / len(p_histories),
    )
    fig, ax = plt.subplots()
    ax.hist(p_histories, bins=100, range=(np.min(p_histories), 100))
    ax.set_xlabel("Percent of True Rs Covered (%)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Percent of True Rs Covered")
    # ax.boxplot(p_histories, vert=False, positions=[5], widths=5)
    violin_with_percentiles([0], [p_histories])


def plot_median_outer_approximation_size(
    results_dir,
    max_steps,
    color,
    fig=None,
    ax=None,
    label="Noise",
    legend=False,
    ylabel=False,
    title=None,
):
    p_histories = []

    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            if int(filename.split("_")[0]) <= 550:
                filepath = os.path.join(results_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                    p_history = data.get("proportion_of_area_covered", [])
                    p_history_padded = p_history[:max_steps]
                    if len(p_history_padded) < max_steps:
                        p_history_padded += [p_history_padded[-1]] * (
                            max_steps - len(p_history_padded)
                        )
                    p_histories.append(p_history_padded)

    median_p = np.median(np.array(p_histories), axis=0)
    first_quartile_p = np.percentile(np.array(p_histories), 25, axis=0)
    third_quartile_p = np.percentile(np.array(p_histories), 75, axis=0)
    min_p = np.min(np.array(p_histories), axis=0)
    max_p = np.max(np.array(p_histories), axis=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    ax.set_xlabel("Number of Sacrificial Agents")
    if ylabel:
        ax.set_ylabel("Size of Union")
    ax.set_xticks(np.arange(1, max_steps + 1, 2))
    ax.plot(np.arange(1, max_steps + 1), median_p, label=label, color=color)
    ax.fill_between(
        np.arange(1, max_steps + 1),
        first_quartile_p,
        third_quartile_p,
        color=color,
        alpha=0.1,
    )
    if title is not None:
        ax.set_title(title)
    # ax.plot(np.arange(1, max_steps + 1), min_p, linestyle=":", color="tab:blue")
    #
    # ax.plot(np.arange(1, max_steps + 1), max_p, linestyle=":", color="tab:blue")
    ax.plot(
        np.arange(1, max_steps + 1),
        np.ones_like(median_p),
        linestyle=":",
        color="red",
    )
    ax.grid(True)
    ax.set_ylim(0, 16)
    ax.set_yticks(np.arange(1, 16, 2))
    if legend:
        ax.legend(ncols=1)
    return fig, ax


def plot_median_rmse_and_abs_errors(
    results_dir,
    max_steps=6,
    epsilon=None,
    knownSpeed=False,
    positionAndHeadingOnly=False,
    color="tab:blue",
    fig=None,
    axes=None,
    noisy=False,
    interior=False,
):
    """
    Plot median, IQR, and min/max for RMSE and absolute errors in a 2x2 grid.
    """
    print("Plotting results from directory:", results_dir)
    rmse_histories = []
    abs_error_histories = []
    flagged_files = []

    count = 0
    minval = np.inf
    minValFile = None
    for filename in os.listdir(results_dir):
        count += 1
        if filename.endswith("_results.json"):
            if int(filename.split("_")[0]) <= 500:
                filepath = os.path.join(results_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                    rmse = data.get("rmse_history", [])
                    abs_err = data.get("absolute_errors", [])
                    # if len(abs_err[-1]) != 3:
                    #     print(filename)
                    trueParams = data.get("true_params", [])
                    if trueParams[-2] < minval:
                        minval = trueParams[-2]
                        minValFile = filename

                    if len(rmse) == 0 or len(abs_err) == 0:
                        continue

                    # Pad RMSE
                    rmse_padded = rmse[:max_steps]
                    if len(rmse_padded) < max_steps:
                        rmse_padded += [rmse_padded[-1]] * (
                            max_steps - len(rmse_padded)
                        )
                    rmse_histories.append(rmse_padded)

                    if epsilon is not None and rmse_padded[-1] > epsilon:
                        flagged_files.append((filename, rmse_padded[-1]))

                    # Pad absolute error
                    abs_err_padded = abs_err[:max_steps]
                    if len(abs_err_padded) < max_steps:
                        abs_err_padded += [abs_err_padded[-1]] * (
                            max_steps - len(abs_err_padded)
                        )
                    abs_error_histories.append(abs_err_padded)

    if not rmse_histories or not abs_error_histories:
        print("No valid data found in directory.")
        return

    # Convert to arrays
    rmse_array = np.array(rmse_histories)
    abs_error_array = np.array(abs_error_histories)  # shape: (N, max_steps, 3)
    x = np.arange(1, max_steps + 1)

    # RMSE stats
    median_rmse = np.median(rmse_array, axis=0)
    q1_rmse = np.percentile(rmse_array, 25, axis=0)
    q3_rmse = np.percentile(rmse_array, 75, axis=0)
    min_rmse = np.min(rmse_array, axis=0)
    max_rmse = np.max(rmse_array, axis=0)

    # Abs error stats
    median_abs = np.median(abs_error_array, axis=0)
    q1_abs = np.percentile(abs_error_array, 25, axis=0)
    q3_abs = np.percentile(abs_error_array, 75, axis=0)
    min_abs = np.min(abs_error_array, axis=0)
    max_abs = np.max(abs_error_array, axis=0)

    # === Plot ===
    rowXlabel = 1
    if positionAndHeadingOnly:
        rowXlabel = 0
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="tight")
        labels = ["X Position Error", "Y Position Error", "Heading Error"]
    elif knownSpeed:
        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(9, 5), layout="tight")
            plt.tight_layout(pad=0.1, h_pad=0.01)
        results_dir = "results/boundary/knownSpeedAndShapeWithNoise"
        labels = [
            "X Position Error",
            "Y Position Error",
            "Heading Error",
            "Turn Radius Error",
            "Range Error",
        ]
        axes[1, 2].axis("off")  # hides everything about the axis

    else:
        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(9, 5), layout="tight")
        labels = [
            "X Position Error",
            "Y Position Error",
            "Heading Error",
            "Speed Error",
            "Turn Radius Error",
            "Range Error",
        ]

    # stat_data = [(median_rmse, q1_rmse, q3_rmse, min_rmse, max_rmse)] + [
    #     (
    #         median_abs[:, i],
    #         q1_abs[:, i],
    #         q3_abs[:, i],
    #         min_abs[:, i],
    #         max_abs[:, i],
    #     )
    #     for i in range(abs_error_array.shape[2])
    # ]

    stat_data = [
        (
            median_abs[:, i],
            q1_abs[:, i],
            q3_abs[:, i],
            min_abs[:, i],
            max_abs[:, i],
        )
        for i in range(abs_error_array.shape[2])
    ]
    if noisy:
        labell = "Noise"
    else:
        labell = "No Noise"

    countPlot = 0
    print("rowXlabe", rowXlabel)
    for ax, (median, q1, q3, minv, maxv), label in zip(axes.flat, stat_data, labels):
        plotColIndex = countPlot // 3
        plotRowIndex = countPlot % 3
        if plotColIndex == rowXlabel:
            ax.set_xlabel("Number of Sacrificial Agents")
        if plotRowIndex == 0:
            ax.set_ylabel("Absolute Error")

        ax.plot(x, median, color=color, linewidth=2)
        ax.fill_between(x, q1, q3, color=color, alpha=0.1)
        # ax.plot(x, minv, linestyle=":", color=color, label="Min/Max")
        # ax.plot(x, maxv, linestyle=":", color=color)
        ax.set_title(label)

        ax.set_xticks(np.arange(1, max_steps + 1, 2))
        # tick font size
        ax.grid(True)
        ax.set_ylim([0, 1.5])

        if interior:
            ax.set_aspect(10)
        else:
            ax.set_aspect(6)
        # ax.set_aspect(10)
        countPlot += 1

        # ax.legend()
        print("final error", label, median[-1])
    ax.plot([], [], label=labell, color=color, linewidth=2)
    ax.legend(loc="upper right", ncols=1)

    plt.tight_layout()

    printFlaggedFiles = True
    if printFlaggedFiles:
        flagged_files_count = 0
        if flagged_files:
            print(f"\nFiles with final RMSE > {epsilon}:")
            for fname, val in flagged_files:
                flagged_files_count += 1
                print(f"  {fname}: RMSE = {val:.4f}")
        print("total runs:", count)
        print(f"Flagged files count: {flagged_files_count}")
    return fig, axes


def plot_box_rmse_and_abs_errors(results_dir, max_steps=6, epsilon=None):
    """
    Plot box and whisker plots for RMSE and absolute errors in a grid.
    Matches labels and layout from original median-based plot.
    """
    rmse_histories = []
    abs_error_histories = []
    flagged_files = []

    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                rmse = data.get("rmse_history", [])
                abs_err = data.get("absolute_errors", [])

                if len(rmse) == 0 or len(abs_err) == 0:
                    continue

                # Pad RMSE
                rmse_padded = rmse[:max_steps]
                if len(rmse_padded) < max_steps:
                    rmse_padded += [rmse_padded[-1]] * (max_steps - len(rmse_padded))
                rmse_histories.append(rmse_padded)

                if epsilon is not None and rmse_padded[-1] > epsilon:
                    flagged_files.append((filename, rmse_padded[-1]))

                # Pad absolute error
                abs_err_padded = abs_err[:max_steps]
                if len(abs_err_padded) < max_steps:
                    abs_err_padded += [abs_err_padded[-1]] * (
                        max_steps - len(abs_err_padded)
                    )
                abs_error_histories.append(abs_err_padded)

    if not rmse_histories or not abs_error_histories:
        print("No valid data found in directory.")
        return

    rmse_array = np.array(rmse_histories)
    abs_error_array = np.array(abs_error_histories)  # shape: (N, max_steps, D)
    x = np.arange(1, max_steps + 1)

    if positionAndHeadingOnly:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        labels = ["RMSE", "x error", "y error", "heading error"]
    elif knownSpeed:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        labels = [
            "RMSE",
            "x error",
            "y error",
            "heading error",
            "turn radius error",
            "range error",
        ]
    else:
        fig, axes = plt.subplots(4, 2, figsize=(12, 10))
        labels = [
            "RMSE",
            "x error",
            "y error",
            "heading error",
            "speed error",
            "turn radius error",
            "range error",
        ]

    # RMSE + each abs error dimension boxplot
    all_data = [rmse_array] + [
        abs_error_array[:, :, i] for i in range(abs_error_array.shape[2])
    ]
    print("labels", labels)

    for ax, data, label in zip(axes.flat, all_data, labels):
        ax.boxplot(data, positions=x)
        ax.set_title(label)
        ax.set_xlabel("Num Sacrificial Agents")
        ax.set_ylabel("Error")
    ax.grid(True)

    plt.title(results_dir)
    plt.tight_layout()
    plt.show()

    printFlaggedFiles = False
    if printFlaggedFiles:
        if flagged_files:
            print(f"\nFiles with final RMSE > {epsilon}:")
            for fname, val in flagged_files:
                print(f"  {fname}: RMSE = {val:.4f}")


def plot_filtered_box_rmse_and_abs_errors(
    results_dir, max_steps=6, epsilon=None, positionAndHeadingOnly=True
):
    """
    Plot box-and-whisker plots for RMSE and absolute errors from JSON files,
    filtering entries using interceptedList.
    """
    rmse_histories = []
    abs_error_histories = []
    flagged_files = []

    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            rmse_all = data.get("rmse_history", [])
            abs_err_all = data.get("absolute_errors", [])
            intercepted_list = data.get("interceptedList", [])

            # Validate structure
            if not (len(rmse_all) == len(abs_err_all) == len(intercepted_list)):
                continue

            # Filter by interceptedList
            rmse_filtered = [
                rmse_all[i] for i in range(len(intercepted_list)) if intercepted_list[i]
            ]
            abs_err_filtered = [
                abs_err_all[i]
                for i in range(len(intercepted_list))
                if intercepted_list[i]
            ]

            # Pad to max_steps
            rmse_padded = rmse_filtered[:max_steps]
            if len(rmse_padded) < max_steps:
                rmse_padded += [rmse_padded[-1]] * (max_steps - len(rmse_padded))
            rmse_histories.append(rmse_padded)

            if epsilon is not None and rmse_all[-1] > epsilon:
                flagged_files.append((filename, rmse_all[-1]))

            abs_err_padded = abs_err_filtered[:max_steps]
            if len(abs_err_padded) < max_steps:
                abs_err_padded += [abs_err_padded[-1]] * (
                    max_steps - len(abs_err_padded)
                )
            abs_error_histories.append(abs_err_padded)

    if not rmse_histories or not abs_error_histories:
        print("No valid data found.")
        return

    # Convert to arrays
    rmse_array = np.array(rmse_histories)  # shape: (N, max_steps)
    abs_error_array = np.array(abs_error_histories)  # shape: (N, max_steps, D)
    x = np.arange(1, max_steps + 1)

    if positionAndHeadingOnly:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        labels = ["RMSE", "x error", "y error", "heading error"]
    else:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        labels = [
            "RMSE",
            "x error",
            "y error",
            "heading error",
            "turn radius error",
            "range error",
        ]

    # Combine RMSE and each abs error component
    all_data = [rmse_array] + [
        abs_error_array[:, :, i] for i in range(abs_error_array.shape[2])
    ]

    for ax, data, label in zip(axes.flat, all_data, labels):
        steps = data.shape[1]
        x = np.arange(1, steps + 1)
        ax.boxplot(data, positions=x)
        ax.set_title(label)
        ax.set_xlabel("Num Sacrificial Agents")
        ax.set_ylabel("Error")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    if flagged_files:
        print(f"\nFiles with final RMSE > {epsilon}:")
        for fname, val in flagged_files:
            print(f"  {fname}: RMSE = {val:.4f}")


def summarize_percent_covered_columns(
    data_dir="boundary", thresholds=(95, 90, 85), max_steps=15
):
    """
    Build a LaTeX table where each column is a scenario (results dir),
    and rows are coverage thresholds (e.g., ≥99%, ≥95%, ≥90%).
    """
    results_dirs = [
        "knownSpeedAndShape",
        "knownSpeedAndShapeWithNoise",
        "knownSpeed",
        "knownSpeedWithNoise",
        "unknownSpeed",
        "unknownSpeedWithNoise",
    ]
    results_dirs = [f"results/{data_dir}/{d}" for d in results_dirs]
    labels = ["Case 1A", "Case 1B", "Case 2A", "Case 2B", "Case 3A", "Case 3B"]

    # Collect percentages per scenario
    scenario_percents = []  # list of [perc_at_99, perc_at_95, perc_at_90] for each scenario
    for d in results_dirs:
        p_histories = load_rs_history_data(
            d, max_steps=max_steps
        )  # shape: (num_runs, num_steps)
        num_runs = len(p_histories)
        print("num_runs for dir", d, num_runs)
        if num_runs == 0:
            scenario_percents.append([np.nan] * len(thresholds))
            continue

        percs = []
        for p in thresholds:
            thresh = p
            count = np.sum(p_histories >= thresh)
            percs.append(100.0 * count / num_runs)
        scenario_percents.append(percs)

    # Build table rows: one row per threshold
    rows = []
    for i, p in enumerate(thresholds):
        row = [f"{p}%"]
        for s in range(len(results_dirs)):
            v = scenario_percents[s][i]
            cell = "---" if np.isnan(v) else f"{v:.2f}%"
            row.append(cell)
        rows.append(row)

    headers = ["Coverage Threshold"] + labels
    latex = tabulate(
        rows, headers=headers, tablefmt="latex_booktabs", stralign="center"
    )
    print(latex)
    return latex


# def plot_percent_covered():
#     results_dir = "results/boundary/knownSpeedAndShape"
#     p_histories = load_rs_history_data(results_dir, max_steps=25)
#     print(p_histories)
#
#     # p_histories shape: (num_runs, num_steps)
#     percentiles = [99, 95, 90]
#     num_runs = p_histories.shape[0]
#
#     # Find for each run when coverage passes thresholds
#     rows = []
#     for p in percentiles:
#         thresh = p
#         count = np.sum(p_histories >= thresh)
#         percent_achieved = 100 * count / num_runs
#         rows.append([f"{p}%", f"{percent_achieved:.1f}%"])
#
#     # Create LaTeX table
#     table = tabulate(
#         rows,
#         headers=["Coverage Threshold", "Percent of Runs ≥ Threshold"],
#         tablefmt="latex_booktabs",
#         stralign="center",
#     )
#
#     print(table)
#


def plot_outer_union_approximation_size():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="tight")

    folder = "interior"
    results_dir = "results/" + folder + "/knownSpeedAndShape"
    maxSteps = 15
    if folder == "interior":
        maxSteps = 25
    plot_median_outer_approximation_size(
        results_dir,
        max_steps=maxSteps,
        color="blue",
        label="No Noise",
        fig=fig,
        ax=axes[0],
        ylabel=True,
    )
    results_dir = "results/" + folder + "/knownSpeedAndShapeWithNoise"
    plot_median_outer_approximation_size(
        results_dir,
        max_steps=maxSteps,
        color="orange",
        label="Noise",
        fig=fig,
        ax=axes[0],
        title="Known Speed and Shape",
    )
    results_dir = "results/" + folder + "/knownSpeed"
    plot_median_outer_approximation_size(
        results_dir,
        max_steps=maxSteps,
        color="blue",
        label="No Noise",
        fig=fig,
        ax=axes[1],
    )
    results_dir = "results/" + folder + "/knownSpeedWithNoise"
    plot_median_outer_approximation_size(
        results_dir,
        max_steps=maxSteps,
        color="orange",
        label="Noise",
        fig=fig,
        ax=axes[1],
        title="Known Speed",
    )
    results_dir = "results/" + folder + "/unknownSpeed"
    plot_median_outer_approximation_size(
        results_dir,
        max_steps=maxSteps,
        color="blue",
        label="No Noise",
        fig=fig,
        ax=axes[2],
    )
    results_dir = "results/" + folder + "/unknownSpeedWithNoise"
    plot_median_outer_approximation_size(
        results_dir,
        max_steps=maxSteps,
        color="orange",
        label="Noise",
        fig=fig,
        ax=axes[2],
        legend=True,
        title="All Unknown",
    )


def plot_error():
    maxSteps = 15
    knownShape = False
    knownSpeed = False
    folder = "interior"
    # folder = "boundary"
    interior = folder == "interior"
    eps = 0.1
    if interior:
        maxSteps = 25
    if knownShape:
        results_dir = "results/" + folder + "/knownSpeedAndShape"
    elif knownSpeed:
        results_dir = "results/" + folder + "/knownSpeed"
    else:
        results_dir = "results/" + folder + "/unknownSpeed"
    fig, axes = plot_median_rmse_and_abs_errors(
        results_dir,
        max_steps=maxSteps,
        epsilon=eps,
        positionAndHeadingOnly=knownShape,
        knownSpeed=knownSpeed,
        color="blue",
        noisy=False,
        interior=interior,
    )
    results_dir += "WithNoise"
    # if knownShape:
    #     results_dir = "results/" + folder + "/knownSpeedAndShapeWithNoise"
    # elif knownSpeed:
    #     results_dir = "results/" + folder + "/knownSpeedWithNoise"
    # else:
    #     results_dir = "results/" + folder + "/unknownSpeedWithNoise"

    fig, axes = plot_median_rmse_and_abs_errors(
        results_dir,
        max_steps=maxSteps,
        epsilon=eps,
        positionAndHeadingOnly=knownShape,
        knownSpeed=knownSpeed,
        fig=fig,
        axes=axes,
        color="orange",
        noisy=True,
        interior=interior,
    )
    #


if __name__ == "__main__":
    # Parse command-line seed
    #
    if len(sys.argv) != 3:
        print("usage: script.py <seed:int> <runSim:0|1>")
        sys.exit(2)

    seed = int(sys.argv[1])
    runSim = bool(int(sys.argv[2]))  # "0" -> False, "1" -> True
    print("seed", seed, "runSim", runSim)
    #
    #
    #
    if runSim:
        start = time.time()
        main(seed)
        print("Total time:", time.time() - start)
        _EXECUTOR.shutdown(wait=True)
        if plotAllFlag:
            plt.show()
        #
    else:
        plot_outer_union_approximation_size()
        plot_error()
        summarize_percent_covered_columns("interior")

plt.show()
