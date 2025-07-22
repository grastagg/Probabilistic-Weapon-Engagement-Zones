import tqdm
from joblib import Parallel, delayed
from functools import partial
import sys
import time
import jax
import getpass
from jax._src.sharding_impls import PositionalSharding
from pyoptsparse import Optimization, OPT, IPOPT
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs  # or pyDOE2
import os
import json
from scipy.optimize import minimize

import dubinsEZ
import dubinsPEZ

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax.config.update("jax_platform_name", "gpu")")
#

positionAndHeadingOnly = False
knownSpeed = True
interceptionOnBoundary = True
randomPath = False
noisyMeasurementsFlag = True
saveResults = True
plotAllFlag = False
if positionAndHeadingOnly:
    parameterMask = np.array([True, True, True, False, False, False])
elif knownSpeed:
    parameterMask = np.array([True, True, True, False, True, True])
else:
    parameterMask = np.array([True, True, True, True, True, True])

np.random.seed(326)  # for reproducibility


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


# @jax.jit
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
        return intercepted, interceptionPoint, interceptionTime, pathHistoryNew

    # else:
    def no_interception_fn():
        interceptionPoint = pathHistory[-1]
        interceptionTime = t[-1]
        intercepted = False
        pathHistoryNew, headings = simulate_trajectory_fn(
            startPosition, heading, speed, tmax, numPoints
        )
        return intercepted, interceptionPoint, interceptionTime, pathHistoryNew

    intercepted, interceptionPoint, interceptionTime, pathHistory = jax.lax.cond(
        firstTrueIndex != -1,
        interception_fn,
        no_interception_fn,
    )

    return intercepted, interceptionPoint, interceptionTime, pathHistory


def sample_entry_point_and_heading(xBounds, yBounds):
    """
    Sample a random entry point on the boundary of a rectangular region,
    and generate a heading that points into the region.

    Args:
        region_bounds: dict with keys 'xmin', 'xmax', 'ymin', 'ymax'

    Returns:
        start_pos: np.array of shape (2,), the boundary point
        heading: float, angle in radians pointing into the region
    """
    xmin, xmax = xBounds
    ymin, ymax = yBounds

    edge = np.random.choice(["left", "right", "top", "bottom"])

    if edge == "left":
        x = xmin
        y = np.random.uniform(ymin, ymax)
    elif edge == "right":
        x = xmax
        y = np.random.uniform(ymin, ymax)
    elif edge == "bottom":
        x = np.random.uniform(xmin, xmax)
        y = ymin
    elif edge == "top":
        x = np.random.uniform(xmin, xmax)
        y = ymax

    start_pos = np.array([x, y], dtype=np.float64)

    # Compute angle toward origin
    direction_to_center = -start_pos  # vector from point to (0,0)
    heading = np.arctan2(direction_to_center[1], direction_to_center[0])

    # Add Gaussian noise to heading
    heading_noise_std = 0.5  # Standard deviation of noise
    heading += np.random.normal(0.0, heading_noise_std)

    return start_pos, heading


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


def pursuerX_to_params_all(X, trueParams):
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


if positionAndHeadingOnly:
    pursuerX_to_params = pursuerX_to_params_position_and_heading
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

    def verbose_print():
        jax.debug.print(
            "pursuerPosition {}, puysuerHeading {}, minimumTurnRadius {}, pursuerRange {}, goalPosition {}, rs {}",
            pursuerPosition,
            pursuerHeading,
            minimumTurnRadius,
            pursuerRange,
            goalPosition,
            rs,
        )

    jax.lax.cond(verbose, verbose_print, lambda: None)

    return rs


def smooth_min(x, alpha=10.0):
    return -jnp.log(jnp.sum(jnp.exp(-alpha * x))) / alpha


# Define the components
def g(t, k):
    h = 1 / (1 + jnp.exp(-k * t))
    return t**2 * h


# Full function y(x)
def new_activation(x, c=2.96, p=0.33, k=10):
    return g(x - c, k) + g(-x - c, k) + p * x**2


def activation(x):
    return jnp.square(jax.nn.relu(x))  # ReLU activation function
    return jnp.log1p(jnp.exp(beta * x)) / beta
    return (jnp.tanh(10.0 * x) + 1.0) / 2.0 * x**2


def compute_intercept_probability(ez_min, alpha=10.0):
    # return ez_min > 0.0
    return jax.nn.sigmoid(-alpha * ez_min)


@jax.jit
def learning_loss_on_boundary_function_single(
    pursuerX,
    heading,
    speed,
    intercepted,
    pathHistory,
    interceptedPoint,
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
    # interceptedLossRSEnd = new_activation(
    #     rsEnd, flattenLearingLossAmount
    # ) + new_activation(-rsEnd, flattenLearingLossAmount)
    interceptedLossTrajectory = jnp.sum(
        activation(-rsAll[:-4] - flattenLearingLossAmount)
    )
    # interceptedLossTrajectory = jnp.max(
    #     new_activation(-rsAll[:-4], flattenLearingLossAmount)
    # )
    interceptedLossRS = (
        interceptedPathWeight * interceptedLossTrajectory + interceptedLossRSEnd
    )

    survivedLossRS = jnp.sum(
        activation(-rsAll - flattenLearingLossAmount)
    )  # loss if survived in RS

    lossRS = jax.lax.cond(
        intercepted, lambda: interceptedLossRS, lambda: survivedLossRS
    )

    def verbose_print():
        jax.debug.print(
            "interceptedPoint: {interceptedPoint}, rsEnd: {rsEnd}, lossRS: {lossRS}, minTrajRS {rsmin}, rsTrajLoss {rstlos}",
            interceptedPoint=interceptedPoint,
            rsEnd=rsEnd,
            lossRS=interceptedLossRSEnd,
            rsmin=rsAll[:-4].min(),
            rstlos=interceptedLossTrajectory,
        )

    jax.lax.cond(verbose, verbose_print, lambda: None)

    return lossRS


@jax.jit
def learning_loss_function_single(
    pursuerX,
    heading,
    speed,
    intercepted,
    pathHistory,
    interceptedPoint,
    trueParams,
    verbose=False,
):
    headings = heading * jnp.ones(pathHistory.shape[0])
    ez = dubinsEZ_from_pursuerX(
        pursuerX,
        pathHistory,
        headings,
        speed,
        trueParams,
    )  # (T,)
    # ez = jnp.where(pathMask, ez, jnp.inf)  # apply mask to ez
    rsEnd = dubins_reachable_set_from_pursuerX(
        pursuerX,
        jnp.array([interceptedPoint]),
        trueParams,
    )[0]
    rsEnd = jnp.where(jnp.isinf(rsEnd), 1000.0, rsEnd)
    rsAll = dubins_reachable_set_from_pursuerX(pursuerX, pathHistory, trueParams)
    # rsAll = jnp.where(pathMask, rsAll, jnp.inf)  # apply mask to rsAll
    # rsAll = jnp.where(jnp.isinf(rsAll), 1000.0, rsAll)

    interceptedLossEZ = activation(jnp.min(ez))  # loss if intercepted
    survivedLossEZ = activation(-jnp.min(ez))  # loss if survived
    lossEZ = jax.lax.cond(
        intercepted, lambda: interceptedLossEZ, lambda: survivedLossEZ
    )
    interceptedLossRS = activation(rsEnd) + activation(-rsEnd)
    # interceptedLossRS = jnp.square(rsEnd)  # loss if intercepted in RS
    # interceptedLossRSAll = activation(-jnp.min(rsAll[0:-5]))
    # interceptedLossRS = (
    #     interceptedLossRSAll + interceptedLossRS
    # )  # loss if intercepted in RS
    survivedLossRS = activation(-jnp.min(rsAll))  # loss if survived in RS
    lossRS = jax.lax.cond(
        intercepted, lambda: interceptedLossRS, lambda: survivedLossRS
    )
    # return rsEnd**2
    return lossRS  # + lossEZ  # total loss


if interceptionOnBoundary:
    batched_loss = jax.jit(
        jax.vmap(
            learning_loss_on_boundary_function_single,
            in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None),
        )
    )
else:
    batched_loss = jax.jit(
        jax.vmap(
            learning_loss_function_single,
            in_axes=(None, 0, 0, 0, 0, 0, None, None),
        )
    )


@jax.jit
def total_learning_loss(
    pursuerX,
    headings,
    speeds,
    intercepted_flags,
    pathHistories,
    interceptedPoints,
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
        pathHistories,
        interceptedPoints,
        trueParams,
        interceptedPathWeight,
        flattenLearingLossAmount,
        verbose,
    )

    # total loss = sum over agents
    return jnp.sum(losses) / len(losses)


dTotalLossDX = jax.jit(jax.jacfwd(total_learning_loss, argnums=0))
totalLossHessian = jax.jit(jax.jacfwd(dTotalLossDX, argnums=0))


def run_optimization_hueristic_scipy(
    headings,
    speeds,
    interceptedList,
    pathHistories,
    endPoints,
    endTimes,
    initialPursuerX,
    lowerLimit,
    upperLimit,
    trueParams,
    interceptedPathWeight=1.0,
    flattenLearingLossAmount=0.0,
    verbose=False,
):
    # Use same logic to define bounds
    global positionAndHeadingOnly
    if positionAndHeadingOnly:
        lowerLimitSub = np.array([0.1, 0.1, 0])
    else:
        lowerLimitSub = np.array([0.1, 0.1, 0, 0, 0])

    lower = lowerLimit - lowerLimitSub
    upper = upperLimit + lowerLimitSub
    bounds = list(zip(lower, upper))

    # Define the objective and gradient for scipy
    def loss_fn(x):
        return float(
            total_learning_loss(
                x,
                headings,
                speeds,
                interceptedList,
                pathHistories,
                endPoints,
                trueParams,
                interceptedPathWeight=interceptedPathWeight,
                flattenLearingLossAmount=flattenLearingLossAmount,
            )
        )

    def grad_fn(x):
        return np.array(
            dTotalLossDX(
                x,
                headings,
                speeds,
                interceptedList,
                pathHistories,
                endPoints,
                trueParams,
                interceptedPathWeight=interceptedPathWeight,
                flattenLearingLossAmount=flattenLearingLossAmount,
            )
        )

    # Run the optimization
    res = minimize(
        fun=loss_fn,
        jac=grad_fn,
        x0=initialPursuerX,
        bounds=bounds,
        method="trust-constr",
        options={"gtol": 1e-8, "maxiter": 200, "disp": False},
    )

    pursuerX = res.x
    loss = res.fun
    print("gradient at optimal", grad_fn(pursuerX))
    return pursuerX, loss


def run_optimization_hueristic(
    headings,
    speeds,
    interceptedList,
    pathHistories,
    endPoints,
    endTimes,
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
        lowerLimitSub = np.array([0.5, 0.5, 0])
    else:
        lowerLimitSub = np.array([0.5, 0.5, 0, 0.2, 0.2])
        lowerLimitSub = np.array([0.5, 0.5, 0, 0.2, 0.2])
    initialLoss = total_learning_loss(
        initialPursuerX,
        headings,
        speeds,
        interceptedList,
        pathHistories,
        endPoints,
        trueParams,
        interceptedPathWeight,
        flattenLearingLossAmount,
        verbose=False,
    )

    if verbose:
        print("initial pursuerX:", initialPursuerX)
        initialLoss = total_learning_loss(
            initialPursuerX,
            headings,
            speeds,
            interceptedList,
            pathHistories,
            endPoints,
            trueParams,
            interceptedPathWeight,
            flattenLearingLossAmount,
            verbose=verbose,
        )
        print("initial loss:", initialLoss)
        print(
            "true pursuerX:",
            trueParams[parameterMask],
        )
        lossTrue = total_learning_loss(
            trueParams[parameterMask],
            headings,
            speeds,
            interceptedList,
            pathHistories,
            endPoints,
            trueParams,
            interceptedPathWeight=interceptedPathWeight,
            flattenLearingLossAmount=flattenLearingLossAmount,
            verbose=verbose,
        )
        print("true learning loss:", lossTrue)

    def objfunc(xDict):
        pursuerX = xDict["pursuerX"]
        loss = total_learning_loss(
            pursuerX,
            headings,
            speeds,
            interceptedList,
            pathHistories,
            endPoints,
            trueParams,
            verbose=False,
            interceptedPathWeight=interceptedPathWeight,
            flattenLearingLossAmount=flattenLearingLossAmount,
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
            pathHistories,
            endPoints,
            trueParams,
            interceptedPathWeight=interceptedPathWeight,
            flattenLearingLossAmount=flattenLearingLossAmount,
        )
        # hess_x = totalLossHessian(
        #     pursuerX,
        #     headings,
        #     speeds,
        #     interceptedList,
        #     pathHistories,
        #     endPoints,
        #     trueParams,
        # )
        #
        # funcsSens = {
        #     "loss": {"pursuerX": grad_x},
        #     "sens_hess": {"loss": {"pursuerX": {"pursuerX": hess_x}}},
        # }
        # return funcsSens, False
        funcsSens = {}
        funcsSens["loss"] = {
            "pursuerX": grad_x,
        }
        return funcsSens, False

    optProb = Optimization("path optimization", objfunc)
    numVars = 5
    if positionAndHeadingOnly:
        numVars = 3
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
    opt.options["max_iter"] = 100
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    # opt.options["hessian_approximation"] = "limited-memory"
    # opt.options["hessian_approximation"] = "exact"
    # opt.setOption("hessian_approximation", "exact")  # Use your Hessian

    # opt.options["derivative_test"] = "first-order"
    # opt.options["derivative_test"] = "second-order"

    # sol = opt(optProb, sens="FD")
    sol = opt(optProb, sens=sens)
    pursuerX = sol.xStar["pursuerX"]
    # loss = sol.fStar
    # loss, _ = objfunc(sol.xStar)
    loss = total_learning_loss(
        pursuerX,
        headings,
        speeds,
        interceptedList,
        pathHistories,
        endPoints,
        trueParams,
        flattenLearingLossAmount=flattenLearingLossAmount,
        verbose=False,
    )
    # loss = loss["loss"]
    return pursuerX, loss


def latin_hypercube_uniform(lowerLimit, upperLimit, numSamples):
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

    # LHS in [0,1]^d
    lhs_unit = lhs(dim, samples=numSamples)

    # Scale to [lower, upper]
    lhs_scaled = lowerLimit + lhs_unit * (upperLimit - lowerLimit)

    return jnp.array(lhs_scaled)


def find_opt_starting_pursuerX(
    interceptedList,
    endPoints,
    lowerLimit,
    upperLimit,
    previousPursuerX,
    mean,
    cov,
    angleMean,
    angleKappa,
    numStartHeadings=10,
    useGaussianSampling=False,
):
    initialPursuerXList = []
    print("num previous pursuerX:", previousPursuerX.shape[0])

    if useGaussianSampling:
        std = np.sqrt(cov)
        initialPositionXs = mean[0] + np.random.normal(0.0, std[0], numStartHeadings)
        initialPositionYs = mean[1] + np.random.normal(0.0, std[1], numStartHeadings)
        initialHeadings = mean[2] + np.random.normal(0.0, std[2], numStartHeadings)
        # initialHeadings = np.random.vonmises(angleMean, angleKappa, numStartHeadings)
    else:
        jitter = 0.02
        initialPositionXs = np.random.choice(
            previousPursuerX[:, 0], size=numStartHeadings, replace=True
        ) + np.random.normal(0.0, jitter, numStartHeadings)
        initialPositionYs = np.random.choice(
            previousPursuerX[:, 1], size=numStartHeadings, replace=True
        ) + np.random.normal(0.0, jitter, numStartHeadings)
        initialHeadings = np.random.choice(
            previousPursuerX[:, 2], size=numStartHeadings, replace=True
        ) + np.random.normal(0.0, jitter, numStartHeadings)
    if not positionAndHeadingOnly:
        if knownSpeed:
            if useGaussianSampling:
                initialTurnRadii = mean[3] + np.random.normal(
                    0.0, std[3], numStartHeadings
                )
                initialRanges = mean[4] + np.random.normal(
                    0.0, std[4], numStartHeadings
                )
            else:
                initialTurnRadii = np.random.choice(
                    previousPursuerX[:, 3], size=numStartHeadings, replace=True
                ) + np.random.normal(0.0, jitter, numStartHeadings)
                initialRanges = (
                    np.random.choice(
                        previousPursuerX[:, 4], size=numStartHeadings, replace=True
                    )
                    + np.random.normal(0.0, jitter, numStartHeadings)
                )  # initialTurnRadii = mean[3] + np.random.normal(0.0, std[3], numStartHeadings)
            # initialRanges = mean[4] + np.random.normal(0.0, std[4], numStartHeadings)
        else:
            initialSpeeds = mean[3] + np.random.normal(0.0, std[3], numStartHeadings)
            initialTurnRadii = mean[4] + np.random.normal(0.0, std[4], numStartHeadings)
            initialRanges = mean[5] + np.random.normal(0.0, cov[5], numStartHeadings)
    for i in range(numStartHeadings):
        if positionAndHeadingOnly:
            intialPursuerX = jnp.array(
                [
                    initialPositionXs[i],
                    initialPositionYs[i],
                    initialHeadings[i],
                ]
            )
        else:
            if knownSpeed:
                intialPursuerX = jnp.array(
                    [
                        initialPositionXs[i],
                        initialPositionYs[i],
                        initialHeadings[i],
                        initialTurnRadii[i],
                        initialRanges[i],
                    ]
                )
            else:
                intialPursuerX = jnp.array(
                    [
                        initialPositionXs[i],
                        initialPositionYs[i],
                        initialHeadings[i],
                        initialSpeeds[i],
                        initialTurnRadii[i],
                        initialRanges[i],
                    ]
                )
        initialPursuerXList.append(intialPursuerX)
    return jnp.array(initialPursuerXList)


def learn_ez(
    headings,
    speeds,
    interceptedList,
    pathHistories,
    endPoints,
    endTimes,
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
):
    # jax.config.update("jax_platform_name", "cpu")
    start = time.time()
    pursuerXList = []
    lossList = []
    if mean is None:
        initialPursuerXList = latin_hypercube_uniform(
            lowerLimit, upperLimit, numStartHeadings
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

    for i in tqdm.tqdm(range(len(initialPursuerXList))):
        # for i in range(len(initialPursuerXList)):
        # pursuerX, loss = run_optimization_hueristic_scipy(
        pursuerX, loss = run_optimization_hueristic(
            headings,
            speeds,
            interceptedList,
            pathHistories,
            endPoints,
            endTimes,
            initialPursuerXList[i],
            lowerLimit,
            upperLimit,
            trueParams,
            interceptedPathWeight=interceptedPathWeight,
            flattenLearingLossAmount=flattenLearningLossAmount,
            verbose=False,
        )
        pursuerXList.append(pursuerX)
        lossList.append(loss)

    # def run_single(initialPursuerX):
    #     pursuerX, loss = run_optimization_hueristic(
    #         headings,
    #         speeds,
    #         interceptedList,
    #         pathHistories,
    #         endPoints,
    #         endTimes,
    #         initialPursuerX,
    #         lowerLimit,
    #         upperLimit,
    #         trueParams,
    #     )
    #     return pursuerX, loss
    #
    # results = Parallel(n_jobs=20)(
    #     delayed(run_single)(initialPursuerX)
    #     for initialPursuerX in tqdm.tqdm(initialPursuerXList)
    # )
    # pursuerXList, lossList = zip(*results)

    pursuerXList = np.array(pursuerXList).squeeze()
    lossList = np.array(lossList).squeeze()

    sorted_indices = np.argsort(lossList)
    lossList = lossList[sorted_indices]
    pursuerXList = pursuerXList[sorted_indices]
    print("lossList:", lossList)
    print("time to learn ez", time.time() - start)
    # pursuerXList[:, 2] = np.unwrap(pursuerXList[:, 2])
    return pursuerXList, lossList


def uniform_circular_entry_points_with_heading_noise(
    center, radius, num_agents, heading_noise_std=0.000005
):
    """
    Uniformly sample agent entry points around a circle, with headings toward the center plus noise.

    Args:
        center: tuple (x, y) — center of the circle
        radius: float — radius of the circle
        num_agents: int — number of agents to generate
        heading_noise_std: float — stddev of heading noise (in radians)

    Returns:
        List of tuples: (start_pos: np.array shape (2,), heading: float)
    """
    angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
    results = []

    for theta in angles:
        # Position on the circle
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        start_pos = np.array([x, y], dtype=np.float64)

        # Heading toward center
        direction_to_center = np.array(center) - start_pos
        heading = np.arctan2(direction_to_center[1], direction_to_center[0])

        # Add Gaussian noise
        heading += np.random.normal(0.0, heading_noise_std)

        results.append((start_pos, heading))

    return results


def softmin(x, tau=0.5):
    x = jnp.asarray(x)
    return -tau * jnp.log(jnp.sum(jnp.exp(-x / tau)))


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
        intercepted, endPoint, endTime, pathHistory = send_low_priority_agent(
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
        intercepted, endPoint, endTime, pathHistory = send_low_priority_agent(
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

    pairs = jnp.array([(i, j) for i in range(N) for j in range(i + 1, N)])

    def pair_score(pair):
        i, j = pair
        return pairwise_disagree(i, j)

    score = jnp.sum(jax.vmap(pair_score)(pairs))
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
):
    jax.config.update("jax_platform_name", "gpu")
    start = time.time()
    upperTheta = jnp.pi + jnp.pi  # / 4
    lowerTheta = jnp.pi - jnp.pi  # / 4
    if randomPath:
        best_angle = np.random.uniform(lowerTheta, upperTheta)
        best_start_pos = center + radius * jnp.array(
            [jnp.cos(best_angle), jnp.sin(best_angle)]
        )
        headingToCenter = np.arctan2(
            center[1] - best_start_pos[1],  # Δy
            center[0] - best_start_pos[0],  # Δx
        )
        best_heading = headingToCenter + np.random.normal(0.0, 0.2)
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
    scores = jax.vmap(
        # inside_model_disagreement_score,
        which_lp_path_minimizes_number_of_potential_solutions_must_intercect_all,
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
    max_score = jnp.max(scores)
    min_score = jnp.min(scores)
    if jnp.isnan(max_score) or jnp.isnan(min_score):
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
    fig, ax = plt.subplots()
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    plot_low_priority_paths(
        startPositions, interceptedList, endPoints, pathHistories, ax
    )
    dubinsEZ.plot_dubins_reachable_set(
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerTurnRadius,
        ax,
        colors=["magenta"],
    )
    # plt.legend(fontsize=18)

    numPlots = len(pursuerXList)
    # make 2 rows and ceil(numPlots/2) columns
    # numPlots = 10
    numPlots = min(numPlots, 10)
    plt.legend()
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


import numpy as np


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


def find_mean_and_std(pursuerXListZeroLoss):
    # Track stats
    angles = pursuerXListZeroLoss[:, 2]
    angleMean, angleKappa, angleVariance, angleStd = fit_von_mises_with_variance(angles)
    print(
        "angleMean:",
        angleMean,
        "angleKappa:",
        angleKappa,
        "angleVariance:",
        angleVariance,
        "angleStd:",
        angleStd,
    )

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
    dt = 0.001
    searchCircleCenter = np.array([0, 0])
    searchCircleRadius = upper_bounds_all[5] + upper_bounds_all[0] + 1
    tmax = (2 * searchCircleRadius) / agentSpeed
    numPoints = int(tmax / dt) + 1

    if noisyMeasurementsFlag:
        lowPriorityAgentPositionCov = np.array([[0.001, 0.0], [0.0, 0.001]])
        flattenLearingLossAmount = find_learning_loss_flatten_amount(
            lowPriorityAgentPositionCov, beta=3.0
        )
        # flattenLearingLossAmount = 0.0
        print("flattenLearingLossAmount:", flattenLearingLossAmount)
        # keepLossThreshold = 1e-2
        maxStdDevThreshold = 0.05
    else:
        lowPriorityAgentPositionCov = np.array([[0.0, 0.0], [0.0, 0.0]])
        flattenLearingLossAmount = 0.0
        maxStdDevThreshold = 0.1
    # flattenLearingLossAmount = 0.0

    # Init histories
    interceptedList, endPoints, endTimes = [], [], []
    pathHistories, startPositions, headings, speeds = [], [], [], []
    pursuerParameter_history, pursuerParameterMean_history = [], []
    pursuerParameterVariance_history, lossList_history = [], []
    pursuerParameterStdDev_history = []

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
            heading = 0.0001
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
                jnp.array(pathHistories),
                jnp.array(endPoints),
                jnp.array(endTimes),
                trueParams,
                searchCenter,
                # [0, 0],
                searchCircleRadius,
                num_angles=30,
                num_headings=30,
                # num_angles=4,
                # num_headings=3,
                speed=agentSpeed,
                tmax=tmax,
                num_points=numPoints // 10,
            )

        # Send agent
        startPositions.append(startPosition)
        headings.append(heading)
        speeds.append(agentSpeed)

        intercepted, endPoint, endTime, pathHistory = send_low_priority_agent(
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
        endPoint = endPoint + rng.multivariate_normal(
            np.zeros(2), lowPriorityAgentPositionCov
        )
        pathHistory = pathHistory + rng.multivariate_normal(
            np.zeros(2), lowPriorityAgentPositionCov, size=(numPoints,)
        )
        interceptedList.append(intercepted)
        endPoints.append(endPoint)
        endTimes.append(endTime)
        pathHistories.append(pathHistory)

        # Learn
        pursuerXList, lossList = learn_ez(
            jnp.array(headings),
            jnp.array(speeds),
            jnp.array(interceptedList),
            jnp.array(pathHistories),
            jnp.array(endPoints),
            jnp.array(endTimes),
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
        )

        # Filter good fits
        # pursuerXListZeroLoss = pursuerXList[lossList <= keepLossThreshold]
        # lossListZeroLoss = lossList[lossList <= keepLossThreshold]
        if len(pursuerXList[lossList <= keepLossThreshold]) == 0:
            print("No good models found. trying again with simpler loss")

            pursuerXList, lossList = learn_ez(
                jnp.array(headings),
                jnp.array(speeds),
                jnp.array(interceptedList),
                jnp.array(pathHistories),
                jnp.array(endPoints),
                jnp.array(endTimes),
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
                useGaussianSampling=False,
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
        print("num valid models:", len(pursuerXListZeroLoss))
        print("num collapsed models:", len(pursuerXListZeroLossCollapsed))
        mean, cov, std, angleMean, angleKappa = find_mean_and_std(pursuerXListZeroLoss)
        print("trueParams", trueParams)
        print("mean:", mean)
        print("std dev", std)

        pursuerParameterMean_history.append(mean)
        pursuerParameterVariance_history.append(cov)
        pursuerParameterStdDev_history.append(std)
        pursuerParameter_history.append(pursuerXList)
        lossList_history.append(lossList)

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
            plt.close("all")

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
    print("mean_estimates", mean_history)
    print("trueParams_np", trueParams_np[parameterMask])
    # rmse_history = np.sqrt(
    #     np.mean((mean_history - trueParams_np[parameterMask]) ** 2, axis=1)
    # )
    rmse_history = compute_rmse_history(mean_history, trueParams_np, parameterMask)
    abs_error_history = compute_abs_error_history(
        mean_history, trueParams_np, parameterMask
    )
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


def main():
    # Parse command-line seed
    if len(sys.argv) != 2:
        sys.exit(1)

    seed = int(sys.argv[1])
    # seed = 25

    # Define parameter bounds
    lowerBoundsAll = np.array([-2.0, -2.0, -np.pi, 1.0, 0.4, 2.0])
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
        keepLossThreshold=1e-6,
        plotEvery=1,
        dataDir="results",
        saveDir="knownSpeedWithNoise",
        # saveDir="knownSpeed",
    )


def plot_median_rmse_and_abs_errors(results_dir, max_steps=6, epsilon=None):
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
    else:
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
    # results_dir = "results/knownShapeAndSpeed"
    # results_dir = "results/knownSpeed"
    # results_dir = "results/knownShapeAndSpeedWithNois"
    results_dir = "results/knownSpeedWithNoise"
    plot_median_rmse_and_abs_errors(results_dir, max_steps=15, epsilon=0.1)
    # plot_box_rmse_and_abs_errors(results_dir, max_steps=10, epsilon=0.1)
    # plot_filtered_box_rmse_and_abs_errors(results_dir, max_steps=10, epsilon=0.05)
    # main()
    # plt.show()
