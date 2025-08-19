import tqdm
from functools import partial
import sys
import time
import jax


import getpass
from pyoptsparse import Optimization, OPT, IPOPT
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pyDOE2 import lhs  # or pyDOE2
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = (
    "0.30"  # try 0.30; adjust if you run 2–3 workers
)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")  # avoid X11 ("Invalid MIT-MAGIC-COOKIE-1")
# GPU + headless settings BEFORE any heavy imports
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"  # tune for worker count
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ.setdefault("MPLBACKEND", "Agg")
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
interceptionOnBoundary = False
randomPath = False
noisyMeasurementsFlag = False
saveResults = True
plotAllFlag = True
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
    ax.set_xlabel("X", fontsize=34)
    ax.set_ylabel("Y", fontsize=34)
    ax.tick_params(labelsize=18)
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
                    label="Intercepted",
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
                ax.scatter(
                    pathHistory[:, 0], pathHistory[:, 1], c="g", label="Survived", s=5
                )
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
    t = jnp.linspace(0.0, tmax, numSimulationPoints)  # shape (T,)
    pathHistory, headings = simulate_trajectory_fn(
        startPosition, heading, speed, tmax, numSimulationPoints
    )
    RS = dubinsEZ.in_dubins_reachable_set(
        pursuerPosition, pursuerHeading, minimumTurnRadius, pursuerRange, pathHistory
    )
    inRS = RS < 0.0  # shape (T,)
    intercepted = jnp.any(inRS)  # shape (T,)
    print("intercepted:", intercepted)

    # if firstTrueIndex != -1:
    def interception_fn():
        intercepted = True

        nz = np.flatnonzero(inRS)  # 1D indices of True
        idx = np.random.choice(nz)
        interceptionPoint = pathHistory[idx]
        interceptionTime = t[idx]
        pathHistoryNew, headings = simulate_trajectory_fn(
            startPosition, heading, speed, interceptionTime, numPoints
        )
        direction = jnp.array(jnp.cos(heading), jnp.sin(heading))  # shape (2,)
        pursuerDistTraveled = dubinsEZ.find_shortest_dubins_path(
            pursuerPosition, pursuerHeading, interceptionPoint, minimumTurnRadius
        )
        interceptionPointEZ = (
            interceptionPoint - pursuerDistTraveled * (speed / pursuerSpeed) * direction
        )

        interceptionTimeEZ = interceptionTime - pursuerDistTraveled / pursuerSpeed
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
        intercepted,
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
def send_low_priority_agent(
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
    ez = dubinsEZ.in_dubins_engagement_zone(
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
    return jax.nn.relu(x) ** 2  # ReLU activation function


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
    headings = heading * jnp.ones(pathHistory.shape[0])
    ezFirst = dubinsEZ_from_pursuerX(
        pursuerX, jnp.array([ezPoint]), jnp.array([headings[-1]]), speed, trueParams
    ).squeeze()
    #

    ezAll = dubinsEZ_from_pursuerX(pursuerX, pathHistory, headings, speed, trueParams)

    outEZ = pathTime < ezTime - 3 * np.sqrt(0.001)

    interceptedLossEZFirst = activation(
        ezFirst - flattenLearingLossAmount
    ) + activation(-ezFirst - flattenLearingLossAmount)

    interceptedLossTrajectory = jnp.where(
        outEZ,
        activation(-ezAll - flattenLearingLossAmount),
        0.0,
    )
    interceptedLossTrajectory = jnp.mean(interceptedLossTrajectory)

    pred_ezTime = (
        pathTime[-1] - (ezFirst + pursuerX[5]) / pursuerX[3]
    )  # estimate time of EZ crossing
    time_loss = time_loss_with_flatten(
        pred_ezTime, ezTime, flattenLearingLossAmount / pursuerX[3]
    )

    interceptedLossEZ = (
        interceptedPathWeight * interceptedLossTrajectory
        + interceptedLossEZFirst
        + time_loss
    )

    survivedLossEZ = jnp.mean(
        activation(-ezAll - flattenLearingLossAmount)
    )  # loss if survived in EZ

    lossEZ = jax.lax.cond(
        intercepted, lambda: interceptedLossEZ, lambda: survivedLossEZ
    )

    return lossEZ


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
    rsEnd = dubins_reachable_set_from_pursuerX(
        pursuerX,
        jnp.array([interceptedPoint]),
        trueParams,
    ).squeeze()
    rsAll = dubins_reachable_set_from_pursuerX(pursuerX, pathHistory, trueParams)

    interceptedLossRSEnd = activation(rsEnd - flattenLearingLossAmount)
    interceptedLossRS = interceptedLossRSEnd

    survivedLossRS = jnp.max(
        activation(-rsAll - flattenLearingLossAmount)
    )  # loss if survived in RS

    lossRS = jax.lax.cond(
        intercepted, lambda: interceptedLossRS, lambda: survivedLossRS
    )

    return lossRS


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
    batched_loss = jax.jit(
        jax.vmap(
            learning_loss_function_single,
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
        lowerLimitSub = np.array([0.5, 0.5, 0.2, 0.2, 0.2, 0.2])

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
        verbose=verbose,
    )

    # if verbose:
    #     print("initial pursuerX:", initialPursuerX)
    #     initialLoss = total_learning_loss(
    #         initialPursuerX,
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
    #         verbose=verbose,
    #     )
    #     initialGradient = dTotalLossDX(
    #         initialPursuerX,
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
    #     )
    #     print("initial loss:", initialLoss)
    #     print("initial gradient:", initialGradient)
    #     print(
    #         "true pursuerX:",
    #         trueParams[parameterMask],
    #     )
    #     lossTrue = total_learning_loss(
    #         trueParams[parameterMask],
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
    #         verbose=verbose,
    #     )
    #     trueGrad = dTotalLossDX(
    #         trueParams[parameterMask],
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
    #     )
    #     print("true learning loss:", lossTrue)
    #     print("true gradient:", trueGrad)

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
    opt.options["tol"] = 1e-8
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
        ) = send_low_priority_agent(
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


def which_lp_path_maximizes_expected_fischer_info(
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
):
    numPoints = len(measuredPathHistories[0])

    def predicted_measurement(pursuerX):
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
        ) = send_low_priority_agent(
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
        prevGrad = dTotalLossDX(
            pursuerX,
            measuredHeadings,
            measuredSpeeds,
            measuredInterceptedList,
            measuredPathTimes,
            measuredPathHistories,
            measuredEndPoints,
            measuredEzPoints,
            measuredEzTimes,
            trueParams,
            1.0,
            flattenLearingLossAmount,
            verbose=False,
        )
        headingsTemp = append0(measuredHeadings, heading)
        speedsTemp = append0(measuredSpeeds, speed)
        interceptedListTemp = append0(measuredInterceptedList, futureIntercepted)
        pathTimesTemp = append0(measuredPathTimes, futurePathTime)
        pathHistoriesTemp = append0(measuredPathHistories, futurePathHistorie)
        endPointsTemp = append0(measuredEndPoints, futureInterceptedPoint)
        ezPointsTemp = append0(measuredEzPoints, futureInterceptionPointEZ)
        ezTimesTemp = append0(measuredEzTimes, futureInterceptionTimeEZ)
        grad = dTotalLossDX(
            pursuerX,
            headingsTemp,
            speedsTemp,
            interceptedListTemp,
            pathTimesTemp,
            pathHistoriesTemp,
            endPointsTemp,
            ezPointsTemp,
            ezTimesTemp,
            trueParams,
            1.0,
            flattenLearingLossAmount,
            verbose=False,
        )
        return prevGrad, grad, futureIntercepted

    prevGrads, grads, intercepteds = jax.vmap(predicted_measurement)(pursuerXList)
    g_inc = grads - prevGrads  # shape (K, d_max)
    score_A = jnp.sum(jnp.sum(g_inc * g_inc, axis=1))  # Σ w ||g_m||^2
    return score_A


def which_lp_path_maximizes_dist_to_next_intercept(
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
    def intercpetion_point(pursuerX):
        (
            pursuerPosition,
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ) = pursuerX_to_params(pursuerX, trueParams)
        _, intercepted, endPoint, endTime, pathHistory, _, _ = send_low_priority_agent(
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
    interceptedPoints = jnp.where(intercepteds[:, None], interceptedPoints, jnp.nan)
    # interceptedPoints = jnp.array(interceptedPoints)[jnp.array(intercepteds)]

    diffs = interceptedPoints[:, None, :] - pastInterceptedPoints[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    total_distances = jnp.min(dists)  # Sum distances to all other points
    return total_distances


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
        _, intercepted, endPoint, endTime, pathHistory, _, _ = send_low_priority_agent(
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


def generate_headings_toward_center(
    radius,
    num_angles,
    num_headings,
    delta,
    center=(0.0, 0.0),
    theta1=0.0,
    theta2=2 * jnp.pi,
    plot=True,
):
    """
    Generate heading directions pointing toward the center ± delta from points on an arc.

    Args:
        radius (float): Radius of the circle.
        num_angles (int): Number of angular positions along the arc.
        num_headings (int): Number of heading deviations per point.
        delta (float): Max deviation (in radians) from heading toward center.
        center (tuple): (x, y) of the center to point toward.
        theta1 (float): Start angle of arc (in radians, counterclockwise).
        theta2 (float): End angle of arc (in radians, counterclockwise).
        plot (bool): If True, show a matplotlib plot.

    Returns:
        positions: (num_angles * num_headings, 2) array of positions on the arc
        headings:  (num_angles * num_headings,) array of headings from each position
    """
    cx, cy = center

    # Normalize angles into [0, 2π)
    theta1 = jnp.mod(theta1, 2 * jnp.pi)
    theta2 = jnp.mod(theta2, 2 * jnp.pi)

    # Ensure counterclockwise sweep
    if theta2 <= theta1:
        theta2 += 2 * jnp.pi

    # Sample angles along the arc
    angles = jnp.linspace(theta1, theta2, num_angles)

    # Compute positions on arc
    x = radius * jnp.cos(angles) + cx
    y = radius * jnp.sin(angles) + cy
    positions = jnp.stack([x, y], axis=-1)  # shape: (num_angles, 2)

    # Compute headings toward center
    headings_to_center = jnp.arctan2(cy - y, cx - x)  # shape: (num_angles,)

    # Heading deviations
    heading_offsets = jnp.linspace(
        -delta, delta, num_headings
    )  # shape: (num_headings,)

    # Broadcast to generate all combinations
    positions_expanded = jnp.repeat(positions, num_headings, axis=0)  # (N*M, 2)
    headings_expanded = jnp.repeat(headings_to_center[:, None], num_headings, axis=1)
    headings_expanded = (headings_expanded + heading_offsets[None, :]).ravel()  # (N*M,)

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
    measuredHeadings,
    measuredSpeeds,
    measuredPathTimes,
    measuredInterceptedList,
    measuredPathHistories,
    measuredEndPoints,
    measuredEndTimes,
    measuredEzPoints,
    measuredEzTimes,
    flattenLearingLossAmount=0.0,
):
    start = time.time()
    #     initialGradient = dTotalLossDX(
    #         initialPursuerX,
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
    #     )
    upperTheta = jnp.pi + jnp.pi  # / 4
    lowerTheta = jnp.pi - jnp.pi  # / 4
    if randomPath:
        best_angle = rng.uniform(lowerTheta, upperTheta)
        best_start_pos = center + radius * jnp.array(
            [jnp.cos(best_angle), jnp.sin(best_angle)]
        )
        headingToCenter = np.arctan2(
            center[1] - best_start_pos[1],  # Δy
            center[0] - best_start_pos[0],  # Δx
        )
        best_heading = headingToCenter + rng.normal(0.0, 0.2)
        return best_start_pos, best_heading
    print("Optimizing next low-priority path...")

    # Generate candidate start positions and headings
    # angles = jnp.linspace(lowerTheta, upperTheta, num_angles, endpoint=False)
    # headingsSac = jnp.linspace(-jnp.pi, jnp.pi, num_headings)
    # angle_grid, heading_grid = jnp.meshgrid(angles, headingsSac)
    # angle_flat = angle_grid.ravel()
    # heading_flat = heading_grid.ravel()
    positions, headings = generate_headings_toward_center(
        radius,
        num_angles,
        num_headings,
        0.5,
        center=center,
        theta1=lowerTheta,
        theta2=upperTheta,
        plot=False,
    )

    diff_threshold = 10.0
    if len(endPoints[interceptedList]) > 0:
        scores = jax.vmap(
            which_lp_path_maximizes_expected_fischer_info,
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
                None,
                None,
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
        )(
            positions,
            headings,
            pursuerXList,
            speed,
            trueParams,
            center,
            radius,
            tmax,
            num_points,
            diff_threshold,
            endPoints,
            measuredHeadings,
            measuredSpeeds,
            measuredPathTimes,
            measuredInterceptedList,
            measuredPathHistories,
            measuredEndPoints,
            measuredEndTimes,
            measuredEzPoints,
            measuredEzTimes,
            flattenLearingLossAmount,
        )
        # scores = jax.vmap(
        #     # inside_model_disagreement_score,
        #     # which_lp_path_minimizes_number_of_potential_solutions_must_intercect_all,
        #     which_lp_path_maximizes_dist_to_next_intercept,
        #     # which_lp_spends_most_time_in_all_rs,
        #     in_axes=(
        #         0,
        #         0,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #     ),
        # )(
        #     # angle_flat,
        #     # heading_flat,
        #     positions,
        #     headings,
        #     pursuerXList,
        #     speed,
        #     trueParams,
        #     center,
        #     radius,
        #     tmax,
        #     num_points,
        #     diff_threshold,
        #     endPoints[interceptedList],
        # )
        print("min score:", jnp.nanmin(scores))
        print("max score:", jnp.nanmax(scores))
    else:
        scores = jax.vmap(
            # inside_model_disagreement_score,
            which_lp_path_minimizes_number_of_potential_solutions_must_intercect_all,
            # which_lp_path_maximizes_dist_to_next_intercept,
            # which_lp_spends_most_time_in_all_rs,
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
                None,
            ),
        )(
            # angle_flat,
            # heading_flat,
            positions,
            headings,
            pursuerXList,
            speed,
            trueParams,
            center,
            radius,
            tmax,
            num_points,
            diff_threshold,
            endPoints[interceptedList],
        )

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
            positions,
            headings,
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

    best_start_pos = positions[best_idx]
    best_heading = headings[best_idx]

    print("time to optimize next low-priority path:", time.time() - start)
    return best_start_pos, best_heading


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
):
    numPlots = len(pursuerXList)
    # make 2 rows and ceil(numPlots/2) columns
    # numPlots = 10
    numPlots = min(numPlots, 10)
    # fig1, axes = make_axes(numPlots)

    fig, ax = plt.subplots(layout="tight")
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
    dubinsEZ.plot_dubins_reachable_set(
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerTurnRadius,
        ax,
        colors=["red"],
    )
    for i in range(len(pursuerXList)):
        # pick ax
        # ax = axes[i]

        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-5.0, 5.0)
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
    plot_low_priority_paths(
        startPositions, interceptedList, endPoints, pathHistories, ax
    )
    ax.tick_params(labelsize=30)

    return fig


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
            fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
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
            if np.linalg.norm(diff) < 1e-1:
                is_unique = False
                break
        if is_unique:
            unique_list.append(row)
            unique_loss_list.append(lossList[i])

    return np.array(unique_list), np.array(unique_loss_list)


def angle_diff(theta1, theta2):
    """Compute smallest difference between two angles in radians."""
    return np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2))


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


def run_simulation_with_random_pursuer(
    lower_bounds_all,
    upper_bounds_all,
    parameterMask,
    seed=0,
    numLowPriorityAgents=6,
    numOptimizerStarts=200,
    keepLossThreshold=1e-5,
    plotEvery=1,
    dataDir="results",
    saveDir="run",
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
            lowPriorityAgentPositionCov, beta=2.0
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

    pursuerXListZeroLoss = None
    pursuerXListZeroLossCollapsed = None
    i = 0
    singlePursuerX = False
    mean = None
    cov = None
    angleKappa = None
    angleMean = None

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
            startPosition, heading = optimize_next_low_priority_path(
                pursuerXListZeroLossCollapsed,
                jnp.array(headings),
                jnp.array(speeds),
                jnp.array(interceptedList),
                jnp.array(downSampledPathHistories),
                jnp.array(endPoints),
                jnp.array(endTimes),
                trueParams,
                searchCenter,
                searchCircleRadius,
                20,
                20,
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
            ) = send_low_priority_agent(
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
        print("intercepted", intercepted)
        print("pathTime", pathTime)
        print("endPoint", endPoint)
        print("endTime", endTime)
        print("pathHistory", pathHistory)
        print("interceptionPointEZ", interceptionPointEZ)
        print("interceptionTimeEZ", interceptionTimeEZ)

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
        downSampledPathHistories.append(pathHistory[::5])
        downSampledPathTimes.append(pathTime[::5])

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

        # fig, ax = plt.subplots()
        # plt.hist(
        #     pursuerXListZeroLoss[:, 2], bins=10, alpha=0.5, label="Learned Headings"
        # )

        pursuerXListZeroLossCollapsed, _ = get_unique_rows_by_proximity(
            pursuerXListZeroLoss, lossListZeroLoss, rtol=1
        )
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
            fig = plot_all(
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
        numLowPriorityAgents=15,
        numOptimizerStarts=100,
        keepLossThreshold=1e-5,
        plotEvery=1,
        dataDir="results",
        saveDir="interioir/knownSpeed",
        # saveDir="knownSpeed",
    )


def plot_median_rmse_and_abs_errors(
    results_dir,
    max_steps=6,
    epsilon=None,
    knownSpeed=False,
    positionAndHeadingOnly=False,
):
    """
    Plot median, IQR, and min/max for RMSE and absolute errors in a 2x2 grid.
    """
    rmse_histories = []
    abs_error_histories = []
    flagged_files = []

    count = 0
    minval = np.inf
    minValFile = None
    for filename in os.listdir(results_dir):
        count += 1
        if filename.endswith("_results.json"):
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
    if positionAndHeadingOnly:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        labels = ["RMSE", "x error", "y error", "heading error"]
        colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    elif knownSpeed:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        labels = [
            "RMSE",
            "x error",
            "y error",
            "heading error",
            "turn radius error",
            "range error",
        ]
        colors = [
            "tab:blue",
            "tab:green",
            "tab:orange",
            "tab:red",
            "tab:purple",
            "tab:brown",
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
        colors = [
            "tab:blue",
            "tab:green",
            "tab:orange",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
        ]

    stat_data = [(median_rmse, q1_rmse, q3_rmse, min_rmse, max_rmse)] + [
        (
            median_abs[:, i],
            q1_abs[:, i],
            q3_abs[:, i],
            min_abs[:, i],
            max_abs[:, i],
        )
        for i in range(abs_error_array.shape[2])
    ]

    for ax, (median, q1, q3, minv, maxv), label, color in zip(
        axes.flat, stat_data, labels, colors
    ):
        ax.plot(x, median, label="Median", color=color)
        ax.fill_between(x, q1, q3, color=color, alpha=0.2, label="IQR")
        ax.plot(x, minv, linestyle=":", color=color, label="Min/Max")
        ax.plot(x, maxv, linestyle=":", color=color)
        ax.set_title(label)

        ax.set_xlabel("Num Sacrificial Agents")
        ax.set_ylabel("Error")
        ax.grid(True)
        ax.legend()
        print("final error", label, median[-1])

    plt.tight_layout()
    plt.show()

    flagged_files_count = 0
    if flagged_files:
        print(f"\nFiles with final RMSE > {epsilon}:")
        for fname, val in flagged_files:
            flagged_files_count += 1
            print(f"  {fname}: RMSE = {val:.4f}")
    print("total runs:", count)
    print(f"Flagged files count: {flagged_files_count}")


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

    plt.tight_layout()
    plt.show()

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

        # plt.show()
    else:
        results_dir = "results/knownSpeed"
        # results_dir = "results/oldNoNoise/knownSpeed"
        # results_dir = "results/oldNoNoise/knownShapeAndSpeed"
        # results_dir = "results/knownSpeed"
        # results_dir = "results/knownShapeAndSpeedWithNoise"
        # results_dir = "results/oldNoiseExperiments/knownShapeAndSpeedWithNois"
        # results_dir = "results/knownSpeedWithNoise"
        # results_dir = "results/oldNoiseExperiments/knownSpeedWithNoise"
        # print("Plotting results from:", results_dir)
        # results_dir = "results/unknownSpeed"
        # results_dir = "results/unknownSpeedWithNoise"
        # results_dir = "results/oldNoiseExperiments/unknownSpeedWithNoise"
        plot_median_rmse_and_abs_errors(
            results_dir,
            max_steps=15,
            epsilon=0.001,
            positionAndHeadingOnly=False,
            knownSpeed=False,
        )
        # plot_box_rmse_and_abs_errors(results_dir, max_steps=15, epsilon=0.1)
        # plot_filtered_box_rmse_and_abs_errors(results_dir, max_steps=10, epsilon=0.05)
        plt.show()
