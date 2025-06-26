import re
import tqdm
import time
import jax
import getpass
from jax._src.sharding_impls import PositionalSharding
from pyoptsparse import Optimization, OPT, IPOPT
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs  # or pyDOE2

import dubinsEZ
import dubinsPEZ

jax.config.update("jax_enable_x64", True)

positionAndHeadingOnly = True
interceptionOnBoundary = True

np.random.seed(326)  # for reproducibility


def plot_low_priority_paths(
    startPositions, interceptedList, endPoints, pathHistories, ax
):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X", fontsize=24)
    ax.set_ylabel("Y", fontsize=24)
    ax.tick_params(labelsize=18)
    # set x and y limits
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)

    interceptedCounter = 0
    survivedCounter = 0
    for i in range(len(startPositions)):
        pathHistory = pathHistories[i]
        if interceptedList[i]:
            ax.scatter(
                endPoints[i][0],
                endPoints[i][1],
                color="red",
                marker="x",
            )
            if interceptedCounter == 0:
                ax.plot(
                    pathHistory[:, 0],
                    pathHistory[:, 1],
                    c="r",
                    label="Intercepted",
                )
            else:
                ax.plot(
                    pathHistory[:, 0],
                    pathHistory[:, 1],
                    c="r",
                )
            interceptedCounter += 1
        else:
            if survivedCounter == 0:
                ax.plot(
                    pathHistory[:, 0],
                    pathHistory[:, 1],
                    c="g",
                    label="Survived",
                )
            else:
                ax.plot(
                    pathHistory[:, 0],
                    pathHistory[:, 1],
                    c="g",
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
                vmin=0.0,
                vmax=1.0,
            )
        else:
            c = ax.scatter(
                pathHistory[:, 0],
                pathHistory[:, 1],
                c=ez,
                vmin=0.0,
                vmax=1.0,
            )
    ax.scatter(pursuerX[0], pursuerX[1], color="blue", marker="o")


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


def dubins_reachable_set_from_pursuerX(pursuerX, goalPosition, trueParams):
    pursuerPosition, pursuerHeading, pursuerSpeed, minimumTurnRadius, pursuerRange = (
        pursuerX_to_params(pursuerX, trueParams)
    )
    rs = dubinsEZ.in_dubins_reachable_set(
        pursuerPosition, pursuerHeading, minimumTurnRadius, pursuerRange, goalPosition
    )
    return rs


def smooth_min(x, alpha=10.0):
    return -jnp.log(jnp.sum(jnp.exp(-alpha * x))) / alpha


def activation(x, beta=10.0):
    # return jax.nn.relu(x)  # ReLU activation function
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
    verbose=False,
):
    rsEnd = dubins_reachable_set_from_pursuerX(
        pursuerX,
        jnp.array([interceptedPoint]),
        trueParams,
    ).squeeze()
    # rsEnd = jnp.where(jnp.isinf(rsEnd), 1000.0, rsEnd)
    rsAll = dubins_reachable_set_from_pursuerX(pursuerX, pathHistory, trueParams)

    # interceptedLossRSEnd = activation(rsEnd) + activation(-rsEnd)
    interceptedLossRSEnd = rsEnd**2
    # interceptedLossRSEnd = jnp.abs(rsEnd)
    interceptedLossTrajectory = jnp.sum(activation(-rsAll[:-1]))
    # interceptedLossTrajectory = activation(-jnp.min(rsAll[:-1]))

    interceptedLossRS = interceptedLossTrajectory + interceptedLossRSEnd

    survivedLossRS = jnp.sum(activation(-rsAll))  # loss if survived in RS
    # survivedLossRS = activation(-jnp.min(rsAll))  # loss if survived in RS

    lossRS = jax.lax.cond(
        intercepted, lambda: interceptedLossRS, lambda: survivedLossRS
    )

    # def verbose_func():
    #     jax.debug.print(
    #         "endRS: {}, interceptedLossRSEnd: {},interceptedLossTrajectory: {},interceptedLossRS: {}, survivedLossRS: {}, lossRS: {}",
    #         rsEnd,
    #         interceptedLossRSEnd,
    #         interceptedLossTrajectory,
    #         interceptedLossRS,
    #         survivedLossRS,
    #         lossRS,
    #     )
    #     return None
    #
    # def non_verbose_func():
    #     return None
    #
    # jax.lax.cond(verbose, verbose_func, non_verbose_func)
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
            in_axes=(None, 0, 0, 0, 0, 0, None, None),
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
        verbose,
    )

    # total loss = sum over agents
    return jnp.sum(losses) / len(losses)


# batched_loss_multiple_pursuerX = jax.jit(
#     jax.vmap(total_learning_loss, in_axes=(0, None, None, None, None, None, None, None))
# )


dTotalLossDX = jax.jit(jax.jacfwd(total_learning_loss, argnums=0))


def centroid_and_principal_axis(points):
    points = np.asarray(points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = (centered.T @ centered) / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmin(eigvals)]  # largest eigenvalue

    return centroid, principal_axis


def find_initial_position_and_heading(interceptedList, endPoints):
    interceptedPoints = endPoints[interceptedList]
    centroid, principal_axis = centroid_and_principal_axis(interceptedPoints)
    heading = np.arctan2(principal_axis[1], principal_axis[0])
    negHeading = np.arctan2(-principal_axis[1], -principal_axis[0])
    if np.isnan(heading):
        heading = 0.0
        negHeading = 0.0
        centroid = np.array([0.0, 0.0])
    return centroid, heading, negHeading


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
):
    # print("running optimization heuristic: ", initialPursuerX)
    # print(
    #     "objective function true parameters:",
    #     total_learning_loss(
    #         trueParams,
    #         headings,
    #         speeds,
    #         interceptedList,
    #         pathHistories,
    #         endPoints,
    #         trueParams,
    #     ),
    # )

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
        )
        funcs = {}
        funcs["loss"] = loss
        return funcs, False

    def sens(xDict, funcs):
        pursuerX = xDict["pursuerX"]
        dX = dTotalLossDX(
            pursuerX,
            headings,
            speeds,
            interceptedList,
            pathHistories,
            endPoints,
            trueParams,
        )
        funcsSens = {}
        funcsSens["loss"] = {
            "pursuerX": dX,
        }
        return funcsSens, False

    # initialLoss = total_learning_loss(
    #     initialPursuerX,
    #     headings,
    #     speeds,
    #     interceptedList,
    #     pathHistories,
    #     endPoints,
    #     trueParams,
    # )
    # print("initial loss:", initialLoss)
    optProb = Optimization("path optimization", objfunc)
    numVars = 6
    if positionAndHeadingOnly:
        numVars = 3
    optProb.addVarGroup(
        name="pursuerX",
        nVars=numVars,
        varType="c",
        value=initialPursuerX,
        lower=lowerLimit,
        upper=upperLimit,
    )
    optProb.addObj("loss")
    opt = OPT("ipopt")
    opt.options["print_level"] = 0
    opt.options["max_iter"] = 100
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    # opt.options["derivative_test"] = "first-order"

    # sol = opt(optProb, sens="FD")
    sol = opt(optProb, sens=sens)
    pursuerX = sol.xStar["pursuerX"]
    loss = sol.fStar
    # print(
    #     "Ran optimization hueristic...",
    #     sol.xStar["pursuerX"],
    #     " loss:",
    #     sol.fStar,
    # )
    # print("Optimization status:", sol.optInform["text"])
    loss, _ = objfunc(sol.xStar)
    loss = loss["loss"]
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
    previousPursuerXList,
    lowerLimit,
    upperLimit,
    numStartHeadings=10,
):
    startPosition, startHeading1, startHeading2 = find_initial_position_and_heading(
        interceptedList, endPoints
    )

    initialPursuerXList = []
    # initialHeadings = np.linspace(
    #     startHeading1, startHeading1 + 2 * np.pi, numStartHeadings, endpoint=False
    # )

    # create half initial headings cetered around startHeading1 and half around startHeading2 with random noise
    # initialHeadings1 = startHeading1 + np.random.normal(0.0, 0.3, numStartHeadings // 2)
    # initialHeadings2 = startHeading2 + np.random.normal(0.0, 0.3, numStartHeadings // 2)
    # initialHeadings = np.concatenate([initialHeadings1, initialHeadings2])

    mean, cov = compute_variance_of_puruser_parameters(previousPursuerXList)
    initialPositionXs = mean[0] + np.random.normal(0.0, cov[0], numStartHeadings)
    initialPositionYs = mean[1] + np.random.normal(0.0, cov[1], numStartHeadings)
    initialHeadings = mean[2] + np.random.normal(0.0, cov[2], numStartHeadings)
    if not positionAndHeadingOnly:
        initialSpeeds = mean[3] + np.random.normal(0.0, cov[3], numStartHeadings)
        initialTurnRadii = mean[4] + np.random.normal(0.0, 3 * cov[4], numStartHeadings)
        initialRanges = mean[5] + np.random.normal(0.0, cov[5], numStartHeadings)

    for i in range(numStartHeadings):
        # previousPursuerX = previousPursuerXList[i]
        if positionAndHeadingOnly:
            lowerLimit = lowerLimit[:3]
            upperLimit = upperLimit[:3]

        if positionAndHeadingOnly:
            intialPursuerX = jnp.array(
                [
                    initialPositionXs[i],
                    initialPositionYs[i],
                    initialHeadings[i],
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
            # initialPosition = startPosition
            # initialPosition = previousPursuerX[0:2]
            # initialHeading = previousPursuerX[2]
            # initialSpeed = previousPursuerX[3]
            # initialTurnRadius = previousPursuerX[4]
            # initialRange = previousPursuerX[5]
            # intialPursuerX = jnp.array(
            #     [
            #         initialPosition[0],
            #         initialPosition[1],
            #         # initialHeadings[i],
            #         initialHeading,
            #         initialSpeed,
            #         initialTurnRadius,
            #         initialRange,
            #     ]
            # )
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
    keepLossThreshold,
    previousPursuerXList=None,
    numStartHeadings=10,
):
    jax.config.update("jax_platform_name", "cpu")
    start = time.time()
    lowerLimit = jnp.array([-2.0, -2.0, -jnp.pi, 0.0, 0.0, 0.0])
    upperLimit = jnp.array([2.0, 2.0, jnp.pi, 5.0, 2.0, 5.0])
    pursuerXList = []
    lossList = []
    if positionAndHeadingOnly:
        lowerLimit = lowerLimit[:3]
        upperLimit = upperLimit[:3]
    if previousPursuerXList is None:
        initialPursuerXList = latin_hypercube_uniform(
            lowerLimit, upperLimit, numStartHeadings
        )
    else:
        # initialPursuerXList = latin_hypercube_uniform(
        #     lowerLimit, upperLimit, numStartHeadings
        # )
        initialPursuerXList = find_opt_starting_pursuerX(
            interceptedList,
            endPoints,
            previousPursuerXList,
            lowerLimit,
            upperLimit,
            numStartHeadings,
        )
    # for i in range(len(initialPursuerXList)):
    for i in tqdm.tqdm(range(len(initialPursuerXList))):
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
        )
        pursuerXList.append(pursuerX)
        lossList.append(loss)

    pursuerXList = np.array(pursuerXList).squeeze()
    lossList = np.array(lossList).squeeze()

    # sorted_indices = jnp.argsort(lossList)
    sorted_indices = np.argsort(lossList)
    lossList = lossList[sorted_indices]
    pursuerXList = pursuerXList[sorted_indices]

    print("loss", lossList)
    # only return ones with zero loss
    # pursuerXList = pursuerXList[lossList == 0.0]
    # lossList = lossList[lossList == 0.0]
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


def which_lp_path_minimizes_number_of_potential_solutions(
    angle,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    numPoints=100,
):
    N = pursuerXList.shape[0]

    # Simulate new path
    start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])

    def intercpetion_point(pursuerX):
        intercepted, endPoint, endTime, pathHistory = send_low_priority_agent(
            start_pos,
            heading,
            speed,
            pursuerX[0:2],
            pursuerX[2],
            pursuerX[4],
            0.0,
            pursuerX[5],
            pursuerX[3],
            tmax,
            numPoints,
            numSimulationPoints=100,
        )
        return endPoint

    interceptedPoints = jax.vmap(intercpetion_point)(pursuerXList)

    def pairwise_disagree(i, j):
        pi = interceptedPoints[i]
        pj = interceptedPoints[j]

        # return jnp.linalg.norm(pi - pj)
        return (~jnp.allclose(pi, pj, atol=0.1)).astype(jnp.float64)

    # indices = jnp.arange(N)
    pairs = jnp.array([(i, j) for i in range(N) for j in range(i + 1, N)])

    def pair_score(pair):
        i, j = pair
        return pairwise_disagree(i, j)

    score = jnp.sum(jax.vmap(pair_score)(pairs))
    return score


def inside_model_disagreement_score(
    angle,
    heading,
    pursuerXList,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    num_points=100,
):
    N = pursuerXList.shape[0]
    epsilon = 0.1

    # Simulate new path
    start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    new_path, new_headings = simulate_trajectory_fn(
        start_pos, heading, speed, tmax, num_points
    )
    new_headings = jnp.reshape(new_headings, (-1,))

    # Compute min EZ value for each parameter set
    def in_rs(pX):
        rs = dubins_reachable_set_from_pursuerX(pX, new_path, trueParams)
        ez = dubinsEZ_from_pursuerX(pX, new_path, new_headings, speed, trueParams)
        return rs <= 0.0, ez <= 0.0

    probsRS, probsEZ = jax.vmap(in_rs)(pursuerXList)  # (N,)
    # probs = rs_vals < 0.0

    # Pairwise disagreement: p_i * (1 - p_j) + p_j * (1 - p_i)
    def pairwise_disagree(i, j):
        pi = probsRS[i]
        pj = probsRS[j]
        pezi = probsEZ[i]
        pezj = probsEZ[j]
        return jnp.sum(pi != pj) + jnp.sum(pezi != pezj)

    # indices = jnp.arange(N)
    pairs = jnp.array([(i, j) for i in range(N) for j in range(i + 1, N)])

    def pair_score(pair):
        i, j = pair
        return pairwise_disagree(i, j)

    score = jnp.sum(jax.vmap(pair_score)(pairs))
    return score


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
    radius=3.0,
    num_angles=1,
    num_headings=1,
    speed=1.0,
    tmax=10.0,
    num_points=100,
):
    jax.config.update("jax_platform_name", "gpu")
    start = time.time()
    randomPath = False
    if randomPath:
        best_angle = np.random.uniform(-np.pi, np.pi)
        best_start_pos = center + radius * jnp.array(
            [jnp.cos(best_angle), jnp.sin(best_angle)]
        )
        headingToCenter = np.arctan2(
            center[1] - best_start_pos[1],  # Δy
            center[0] - best_start_pos[0],  # Δx
        )
        best_heading = headingToCenter + np.random.normal(0.0, 0.5)
        return best_start_pos, best_heading
    print("Optimizing next low-priority path...")

    # Generate candidate start positions and headings
    # angles = jnp.linspace(0, 2 * jnp.pi, num_angles, endpoint=False)
    angles = jnp.linspace(-jnp.pi, jnp.pi, num_angles, endpoint=False)
    headingsSac = jnp.linspace(-jnp.pi, jnp.pi, num_headings)
    angle_grid, heading_grid = jnp.meshgrid(angles, headingsSac)
    angle_flat = angle_grid.ravel()
    heading_flat = heading_grid.ravel()

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
        ),
    )(
        angle_flat,
        heading_flat,
        pursuerXList,
        speed,
        trueParams,
        center,
        radius,
        tmax,
        num_points,
    )
    print("minimum score:", jnp.min(scores))
    print("maximum score:", jnp.max(scores))

    best_idx = jnp.nanargmax(scores)

    best_angle = angle_flat[best_idx]
    best_heading = heading_flat[best_idx]
    best_start_pos = center + radius * jnp.array(
        [jnp.cos(best_angle), jnp.sin(best_angle)]
    )

    print("time to optimize next low-priority path:", time.time() - start)
    return best_start_pos, best_heading


def weighted_cov(X, weights, unbiased=False):
    """
    Compute the weighted covariance matrix of dataset X with given weights.

    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        The data matrix.
    weights : ndarray of shape (n_samples,)
        The weights for each observation. Can be unnormalized.
    unbiased : bool
        If True, use the unbiased (Bessel-corrected) estimate.

    Returns:
    --------
    cov : ndarray of shape (n_features, n_features)
        The weighted covariance matrix.
    """
    X = np.asarray(X)
    weights = np.asarray(weights)

    if X.ndim == 1:
        X = X[:, np.newaxis]

    assert (
        X.shape[0] == weights.shape[0]
    ), "Weights and data must match in number of samples"

    w_sum = np.sum(weights)
    mean = np.average(X, axis=0, weights=weights)

    X_centered = X - mean
    weighted_outer = (weights[:, np.newaxis] * X_centered).T @ X_centered

    if unbiased:
        w_squared_sum = np.sum(weights**2)
        eff_dof = w_sum - w_squared_sum / w_sum
        cov = weighted_outer / eff_dof
    else:
        cov = weighted_outer / w_sum

    return cov


def ranking_to_weights(errors, epsilon=1e-6):
    errors = np.asarray(errors)
    return 1.0 / (errors + epsilon)


def compute_variance_of_puruser_parameters(pursuerXList, ranks=None):
    if ranks is None:
        weights = np.ones(len(pursuerXList))
    else:
        weights = ranking_to_weights(ranks)
        weights = np.ones_like(weights)  # for now, use uniform weights
    pursuerXList = jnp.array(pursuerXList)
    # mean = jnp.mean(pursuerXList, axis=0)
    mean = jnp.average(pursuerXList, axis=0, weights=weights)
    # cov = jnp.cov(pursuerXList, rowvar=False)
    cov = weighted_cov(pursuerXList, weights, unbiased=False)
    print("Mean pursuer parameters:\n", mean)
    print("Covariance of pursuer parameters:\n", cov.diagonal())
    return mean, cov.diagonal()


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
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
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
    numPlots = min(numPlots, 4)
    plt.legend()
    fig1, axes = make_axes(numPlots)

    for i in range(numPlots):
        # pick ax
        ax = axes[i]

        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        pursuerX = pursuerXList[i].squeeze()
        plot_low_priority_paths_with_ez(
            headings,
            speeds,
            startPositions,
            interceptedList,
            endPoints,
            pathHistories,
            pursuerX,
            trueParams,
            ax,
        )
        dubinsEZ.plot_dubins_reachable_set(
            pursuerPosition,
            pursuerHeading,
            pursuerRange,
            pursuerTurnRadius,
            ax,
            colors=["green"],
        )
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
            colors=["red"],
        )
        plot_true_and_learned_pursuer(
            pursuerPosition,
            pursuerHeading,
            pursuerPositionLearned1,
            pursuerHeadingLearned1,
            ax,
        )
        # use loss as title
        ax.set_title(f"Loss: {lossList[i]:.4f}", fontsize=12)
    return fig1


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
    if not positionAndHeadingOnly:
        fig, axes = plt.subplots(3, 2)
        axes[2, 1].set_xlabel("Num Sacrificial Agents")
        axes[2, 0].set_xlabel("Num Sacrificial Agents")
        # set tick values of x axis
        axes[0, 0].set_xticks(range(numLowPriorityAgents))
        axes[0, 1].set_xticks(range(numLowPriorityAgents))
        axes[1, 0].set_xticks(range(numLowPriorityAgents))
        axes[1, 1].set_xticks(range(numLowPriorityAgents))
        axes[2, 0].set_xticks(range(numLowPriorityAgents))
        axes[2, 1].set_xticks(range(numLowPriorityAgents))
        axes[0, 0].set_title("Pursuer X Position")
        axes[0, 0].plot(np.ones(len(pursuerParameter_history)) * trueParams[0], "r--")
        axes[0, 1].set_title("Pursuer Y Position")
        axes[0, 1].plot(np.ones(len(pursuerParameter_history)) * trueParams[1], "r--")
        axes[1, 0].set_title("Pursuer Heading")
        axes[1, 0].plot(np.ones(len(pursuerParameter_history)) * trueParams[2], "r--")
        axes[1, 1].set_title("Pursuer Speed")
        axes[1, 1].plot(np.ones(len(pursuerParameter_history)) * trueParams[3], "r--")
        axes[2, 0].set_title("Pursuer Turn Radius")
        axes[2, 0].plot(np.ones(len(pursuerParameter_history)) * trueParams[4], "r--")
        axes[2, 1].set_title("Pursuer Range")
        axes[2, 1].plot(np.ones(len(pursuerParameter_history)) * trueParams[5], "r--")
        # plot mean and variance of pursuer parameters
        beta = 3.0
        axes[0, 0].plot(pursuerParameterMean_history[:, 0], "b-")
        axes[0, 0].fill_between(
            range(len(pursuerParameterMean_history)),
            pursuerParameterMean_history[:, 0]
            - beta * jnp.sqrt(pursuerParameterVariance_history[:, 0]),
            pursuerParameterMean_history[:, 0]
            + beta * jnp.sqrt(pursuerParameterVariance_history[:, 0]),
            color="b",
            alpha=0.2,
        )
        axes[0, 1].plot(pursuerParameterMean_history[:, 1], "b-")
        axes[0, 1].fill_between(
            range(len(pursuerParameterMean_history)),
            pursuerParameterMean_history[:, 1]
            - beta * jnp.sqrt(pursuerParameterVariance_history[:, 1]),
            pursuerParameterMean_history[:, 1]
            + beta * jnp.sqrt(pursuerParameterVariance_history[:, 1]),
            color="b",
            alpha=0.2,
        )
        axes[1, 0].plot(pursuerParameterMean_history[:, 2], "b-")
        axes[1, 0].fill_between(
            range(len(pursuerParameterMean_history)),
            pursuerParameterMean_history[:, 2]
            - beta * jnp.sqrt(pursuerParameterVariance_history[:, 2]),
            pursuerParameterMean_history[:, 2]
            + beta * jnp.sqrt(pursuerParameterVariance_history[:, 2]),
            color="b",
            alpha=0.2,
        )
        axes[1, 1].plot(pursuerParameterMean_history[:, 3], "b-")
        axes[1, 1].fill_between(
            range(len(pursuerParameterMean_history)),
            pursuerParameterMean_history[:, 3]
            - beta * jnp.sqrt(pursuerParameterVariance_history[:, 3]),
            pursuerParameterMean_history[:, 3]
            + beta * jnp.sqrt(pursuerParameterVariance_history[:, 3]),
            color="b",
            alpha=0.2,
        )
        axes[2, 0].plot(pursuerParameterMean_history[:, 4], "b-")
        axes[2, 0].fill_between(
            range(len(pursuerParameterMean_history)),
            pursuerParameterMean_history[:, 4]
            - beta * jnp.sqrt(pursuerParameterVariance_history[:, 4]),
            pursuerParameterMean_history[:, 4]
            + beta * jnp.sqrt(pursuerParameterVariance_history[:, 4]),
            color="b",
            alpha=0.2,
        )
        axes[2, 1].plot(pursuerParameterMean_history[:, 5], "b-")
        axes[2, 1].fill_between(
            range(len(pursuerParameterMean_history)),
            pursuerParameterMean_history[:, 5]
            - beta * pursuerParameterVariance_history[:, 5],
            pursuerParameterMean_history[:, 5]
            + beta * pursuerParameterVariance_history[:, 5],
            color="b",
            alpha=0.2,
        )

        for i in range(numOptimizerStarts):
            mask = lossList_history[:, i] <= keepLossThreshold
            c = lossList_history[:, i][mask]
            max = keepLossThreshold
            c = jnp.clip(c, 0.0, max)  # Ensure c is in [0, 1] for color mapping
            pointSize = 10
            axes[0, 0].scatter(
                np.arange(numLowPriorityAgents)[mask],
                pursuerParameter_history[:, i, 0][mask],
                c=c,
                vmin=0.0,
                vmax=max,
                s=pointSize,
            )
            axes[0, 1].scatter(
                np.arange(numLowPriorityAgents)[mask],
                pursuerParameter_history[:, i, 1][mask],
                c=c,
                vmin=0.0,
                vmax=max,
                s=pointSize,
            )
            axes[1, 0].scatter(
                np.arange(numLowPriorityAgents)[mask],
                pursuerParameter_history[:, i, 2][mask],
                c=c,
                vmin=0.0,
                vmax=max,
                s=pointSize,
            )
            axes[1, 1].scatter(
                np.arange(numLowPriorityAgents)[mask],
                pursuerParameter_history[:, i, 3][mask],
                c=c,
                vmin=0.0,
                vmax=max,
                s=pointSize,
            )
            axes[2, 0].scatter(
                np.arange(numLowPriorityAgents)[mask],
                pursuerParameter_history[:, i, 4][mask],
                c=c,
                vmin=0.0,
                vmax=max,
                s=pointSize,
            )
            axes[2, 1].scatter(
                np.arange(numLowPriorityAgents)[mask],
                pursuerParameter_history[:, i, 5][mask],
                c=c,
                vmin=0.0,
                vmax=max,
                s=pointSize,
            )
    else:
        fig, axes = plt.subplots(3)
        xData = np.arange(1, len(pursuerParameter_history) + 1)
        axes[0].set_title("Pursuer X Position")
        axes[0].plot(
            xData, np.ones(len(pursuerParameter_history)) * trueParams[0], "r--"
        )
        axes[1].set_title("Pursuer Y Position")
        axes[1].plot(
            xData, np.ones(len(pursuerParameter_history)) * trueParams[1], "r--"
        )
        axes[2].set_title("Pursuer Heading")
        axes[2].plot(
            xData, np.ones(len(pursuerParameter_history)) * trueParams[2], "r--"
        )
        axes[2].set_xlabel("Num Sacrificial Agents")
        axes[0].set_xticks(xData)
        axes[1].set_xticks(xData)
        axes[2].set_xticks(xData)
        axes[0].plot(xData, pursuerParameterMean_history[:, 0], "b-")
        beta = 3.0
        axes[0].fill_between(
            xData,
            pursuerParameterMean_history[:, 0]
            - beta * jnp.sqrt(pursuerParameterVariance_history[:, 0]),
            pursuerParameterMean_history[:, 0]
            + beta * jnp.sqrt(pursuerParameterVariance_history[:, 0]),
            color="b",
            alpha=0.2,
        )
        axes[1].plot(xData, pursuerParameterMean_history[:, 1], "b-")
        axes[1].fill_between(
            xData,
            pursuerParameterMean_history[:, 1]
            - beta * jnp.sqrt(pursuerParameterVariance_history[:, 1]),
            pursuerParameterMean_history[:, 1]
            + beta * jnp.sqrt(pursuerParameterVariance_history[:, 1]),
            color="b",
            alpha=0.2,
        )
        axes[2].plot(xData, pursuerParameterMean_history[:, 2], "b-")
        axes[2].fill_between(
            xData,
            pursuerParameterMean_history[:, 2] - pursuerParameterVariance_history[:, 2],
            pursuerParameterMean_history[:, 2] + pursuerParameterVariance_history[:, 2],
            color="b",
            alpha=0.2,
        )
        for i in range(numOptimizerStarts):
            mask = lossList_history[:, i] <= keepLossThreshold
            c = lossList_history[:, i][mask]
            pointSize = 10
            axes[0].scatter(
                xData[mask],
                pursuerParameter_history[:, i, 0][mask],
                c=c,
                s=pointSize,
            )
            axes[1].scatter(
                xData[mask],
                pursuerParameter_history[:, i, 1][mask],
                c=c,
                s=pointSize,
            )
            axes[2].scatter(
                xData[mask],
                pursuerParameter_history[:, i, 2][mask],
                c=c,
                s=pointSize,
            )


def get_unique_rows_by_proximity(arr, lossList, rtol=1e-3):
    unique_list = []
    unique_loss_list = []
    for i, row in enumerate(arr):
        # if not any(np.all(np.isclose(row, u, rtol=rtol)) for u in unique_list):
        if not any(np.linalg.norm(row - u) < 1e-3 for u in unique_list):
            unique_list.append(row)
            unique_loss_list.append(lossList[i])
    return np.array(unique_list), np.array(unique_loss_list)


def main():
    pursuerPosition = np.array([0.5, 0.5])
    pursuerHeading = (0.0 / 20.0) * np.pi
    pursuerRange = 2.0
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    agentSpeed = 1.0
    pursuerTurnRadius = 0.4
    dt = 0.01
    searchCircleCenter = np.array([0, 0])
    searchCircleRadius = 7.0
    tmax = (2 * searchCircleRadius) / agentSpeed
    trueParams = jnp.array(
        [
            pursuerPosition[0],
            pursuerPosition[1],
            pursuerHeading,
            pursuerSpeed,
            pursuerTurnRadius,
            pursuerRange,
        ]
    )

    numPoints = int(tmax / dt) + 1

    numOptimizerStarts = 100
    interceptedList = []
    numLowPriorityAgents = 5
    endPoints = []
    endTimes = []
    pathHistories = []
    startPositions = []
    headings = []
    speeds = []

    # agents = uniform_circular_entry_points_with_heading_noise(
    #     searchCircleCenter,
    #     searchCircleRadius,
    #     numLowPriorityAgents,
    #     heading_noise_std=0.5,
    # )
    plotEvery = 1
    pursuerParameterRMSE_history = []
    pursuerParameter_history = []
    pursuerParameterMean_history = []
    pursuerParameterVariance_history = []
    lossList_history = []
    pursuerXList = None
    singlePursuerX = False
    pursuerXListZeroLoss = None
    lossListZeroLoss = None
    i = 0

    keepLossThreshold = 1e-5

    while i < numLowPriorityAgents and not singlePursuerX:
        print("iteration:", i)
        if i == 0:
            startPosition = jnp.array([-searchCircleRadius, 0.0001])
            heading = 0.0001
        else:
            if pursuerXListZeroLoss is None:
                searchCenter = searchCircleCenter
            else:
                searchCenter = jnp.mean(pursuerXListZeroLoss[:, :2], axis=0)
            startPosition, heading = optimize_next_low_priority_path(
                # pursuerXList,
                pursuerXListZeroLoss,
                jnp.array(headings),
                jnp.array(speeds),
                jnp.array(interceptedList),
                jnp.array(pathHistories),
                jnp.array(endPoints),
                jnp.array(endTimes),
                trueParams,
                searchCenter,
                searchCircleRadius,
                num_angles=32,
                num_headings=32,
                speed=agentSpeed,
                tmax=tmax,
                num_points=numPoints,
            )

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
        interceptedList.append(intercepted)
        endPoints.append(endPoint)
        endTimes.append(endTime)
        pathHistories.append(pathHistory)
        (pursuerXList, lossList) = learn_ez(
            jnp.array(headings),
            jnp.array(speeds),
            jnp.array(interceptedList),
            jnp.array(pathHistories),
            jnp.array(endPoints),
            jnp.array(endTimes),
            trueParams,
            # pursuerXList,
            keepLossThreshold,
            pursuerXListZeroLoss,
            numOptimizerStarts,
        )
        pursuerXListZeroLoss = pursuerXList[lossList <= keepLossThreshold]
        lossListZeroLoss = lossList[lossList <= keepLossThreshold]
        print("num particles", np.sum(lossList <= keepLossThreshold))
        mean, cov = compute_variance_of_puruser_parameters(pursuerXListZeroLoss)
        pursuerXListZeroLoss, lossListZeroLoss = get_unique_rows_by_proximity(
            pursuerXListZeroLoss, lossListZeroLoss, rtol=1e-1
        )
        print("num unique particles", len(pursuerXListZeroLoss))
        if len(lossListZeroLoss) == 0:
            break
        for pursuerX in pursuerXListZeroLoss:
            loss = total_learning_loss(
                pursuerX,
                jnp.array(headings),
                jnp.array(speeds),
                jnp.array(interceptedList),
                jnp.array(pathHistories),
                jnp.array(endPoints),
                trueParams,
                verbose=False,
            )
        pursuerParameter_history.append(pursuerXList)
        lossList_history.append(lossList)
        mean, cov = compute_variance_of_puruser_parameters(
            pursuerXListZeroLoss,
        )
        pursuerParameterVariance_history.append(cov)
        pursuerParameterMean_history.append(mean)
        if i % plotEvery == 0:
            fig1 = plot_all(
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
                pursuerXListZeroLoss,
                lossListZeroLoss,
                trueParams,
            )
            fig1.savefig(f"video/{i}.png")
            plt.close(fig1)

        plt.close("all")
        i += 1
        singlePursuerX = len(pursuerXList) == 1
        if len(lossListZeroLoss) <= 1:
            break
    pursuerParameter_history = jnp.array(pursuerParameter_history)
    lossList_history = jnp.array(lossList_history)
    pursuerParameterMean_history = jnp.array(pursuerParameterMean_history)
    pursuerParameterVariance_history = jnp.array(
        pursuerParameterVariance_history
    ).squeeze()
    plot_pursuer_parameters_spread(
        pursuerParameter_history,
        pursuerParameterMean_history,
        pursuerParameterVariance_history,
        lossList_history,
        trueParams,
        numOptimizerStarts,
        len(pursuerParameterMean_history),
        keepLossThreshold,
    )


if __name__ == "__main__":
    main()
    plt.show()
