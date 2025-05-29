import re
import jax
import getpass
from pyoptsparse import Optimization, OPT, IPOPT
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import bsr_array

import dubinsEZ
import nueral_network_EZ


def plot_low_priority_paths(
    startPositions,
    interceptedList,
    endPoints,
    pathHistories,
    pathMasks,
):
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    for i in range(len(startPositions)):
        pathHistory = pathHistories[i]
        pathMask = pathMasks[i]
        if interceptedList[i]:
            ax.scatter(
                endPoints[i][0],
                endPoints[i][1],
                color="red",
                marker="x",
            )
            ax.plot(pathHistory[:, 0][pathMask], pathHistory[:, 1][pathMask], c="r")
        else:
            ax.plot(pathHistory[:, 0][pathMask], pathHistory[:, 1][pathMask], c="g")


def plot_low_priority_paths_with_ez(
    headings,
    speeds,
    startPositions,
    interceptedList,
    endPoints,
    pathHistories,
    pathMasks,
    pursuerX,
    ax,
):
    ax.set_aspect("equal", adjustable="box")
    for i in range(len(startPositions)):
        print("intercepted", interceptedList[i])
        pathHistory = pathHistories[i]

        heading = headings[i]
        ez = (
            dubinsEZ_from_pursuerX(
                pursuerX, pathHistory, heading * np.ones(len(pathHistory)), speeds[i]
            )
            < 0
        )

        print("in ez", jnp.any(ez))
        pathMask = pathMasks[i]
        if interceptedList[i]:
            ax.scatter(
                endPoints[i][0],
                endPoints[i][1],
                color="red",
                marker="x",
            )
            c = ax.scatter(
                pathHistory[:, 0][pathMask],
                pathHistory[:, 1][pathMask],
                c=ez[pathMask],
                vmin=0.0,
                vmax=1.0,
            )
        else:
            c = ax.scatter(
                pathHistory[:, 0][pathMask],
                pathHistory[:, 1][pathMask],
                c=ez[pathMask],
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
):
    currentPosition = jnp.array(startPosition)
    t = jnp.linspace(0.0, tmax, numPoints)  # shape (T,)
    direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])  # shape (2,)
    displacement = t[:, None] * speed * direction  # shape (T, 2)
    pathHistory = currentPosition + displacement  # shape (T, 2)
    headings = heading * jnp.ones(numPoints)  # shape (T,)
    EZ = dubinsEZ.in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        pathHistory,
        headings,
        speed,
    )
    inEZ = EZ < 0.0  # shape (T,)
    firstTrueIndex = first_true_index_safe(inEZ)
    if firstTrueIndex != -1:
        intercepted = True
        speedRatio = speed / pursuerSpeed

        interceptionPoint = (
            pathHistory[firstTrueIndex] + speedRatio * pursuerRange * direction
        )
        interceptionTime = t[firstTrueIndex] + pursuerRange / pursuerSpeed
        print(
            "test agent will be intercepted at",
            interceptionPoint,
            "at time",
            interceptionTime,
        )
        mask = t < interceptionTime
    else:
        interceptionPoint = pathHistory[-1]
        interceptionTime = t[-1]
        intercepted = False
        mask = jnp.ones(numPoints, dtype=bool)

    return intercepted, interceptionPoint, interceptionTime, pathHistory, mask


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


def pursuerX_to_params(X):
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


def dubinsEZ_from_pursuerX(pursuerX, pathHistory, headings, speed):
    pursuerPosition, pursuerHeading, pursuerSpeed, minimumTurnRadius, pursuerRange = (
        pursuerX_to_params(pursuerX)
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
    # temp = 10.0
    # return jax.nn.sigmoid(-ez / temp)  # scale to [0, 1] range
    return ez


def dubins_reachable_set_from_pursuerX(pursuerX, goalPosition):
    pursuerPosition, pursuerHeading, pursuerSpeed, minimumTurnRadius, pursuerRange = (
        pursuerX_to_params(pursuerX)
    )
    rs = dubinsEZ.in_dubins_reachable_set(
        pursuerPosition, pursuerHeading, minimumTurnRadius, pursuerRange, goalPosition
    )
    return rs


#### pez learning code ####
@jax.jit
def learning_loss_function_single(
    pursuerX, heading, speed, intercepted, pathHistory, pathMask, interceptedPoint
):
    headings = heading * jnp.ones(pathHistory.shape[0])
    ez = dubinsEZ_from_pursuerX(pursuerX, pathHistory, headings, speed)  # (T,)

    ez = jnp.where(pathMask, ez, jnp.inf)
    rsEnd = dubins_reachable_set_from_pursuerX(pursuerX, jnp.array([interceptedPoint]))[
        0
    ]
    # rsAll = dubins_reachable_set_from_pursuerX(pursuerX, pathHistory)

    interceptedLossEZ = jax.nn.relu(jnp.min(ez))  # loss if intercepted
    survivedLossEZ = jax.nn.relu(-jnp.min(ez))  # loss if survived
    lossEZ = jax.lax.cond(
        intercepted, lambda: interceptedLossEZ, lambda: survivedLossEZ
    )
    interceptedLossRS = jax.nn.relu(rsEnd)  # loss if intercepted in RS
    survivedLossRS = 0.0  # loss if survived in RS
    lossRS = jax.lax.cond(
        intercepted, lambda: interceptedLossRS, lambda: survivedLossRS
    )
    return lossEZ + lossRS


batched_loss = jax.jit(
    jax.vmap(learning_loss_function_single, in_axes=(None, 0, 0, 0, 0, 0, 0))
)


@jax.jit
def total_learning_loss(
    pursuerX,
    headings,
    speeds,
    intercepted_flags,
    pathHistories,
    pathMasks,
    interceptedPoints,
):
    # shape (N,)
    losses = batched_loss(
        pursuerX,
        headings,
        speeds,
        intercepted_flags,
        pathHistories,
        pathMasks,
        interceptedPoints,
    )

    # total loss = sum over agents
    return jnp.sum(losses) / len(losses)


dTotalLossDX = jax.jit(jax.grad(total_learning_loss, argnums=0))


def learn_ez(
    headings, speeds, interceptedList, pathHistories, pathMasks, endPoints, endTimes
):
    pursuerX = jnp.array(
        [
            0.0,
            0.0,
            0.0 / 20.0 * jnp.pi,
            2.0,
            0.2,
            1.0,
        ]
    )
    lowerLimit = jnp.array([-2.0, -2.0, -jnp.pi, 0.0, 0.0, 0.00])
    upperLimit = jnp.array([2.0, 2.0, jnp.pi, 5.0, 5.0, 5.00])
    # upperLimit = 10 * jnp.ones_like(lowerLimit)
    headings = jnp.array(headings)
    total_loss = total_learning_loss(
        pursuerX,
        headings,
        speeds,
        interceptedList,
        pathHistories,
        pathMasks,
        endPoints,
    )
    print("Total loss for all agents:", total_loss)
    pursuerX = (lowerLimit + upperLimit) / 2.0
    pursuerX = np.random.uniform(lowerLimit, upperLimit)
    total_loss = total_learning_loss(
        pursuerX, headings, speeds, interceptedList, pathHistories, pathMasks, endPoints
    )
    total_loss_grad = dTotalLossDX(
        pursuerX, headings, speeds, interceptedList, pathHistories, pathMasks, endPoints
    )
    print("starting loss", total_loss)
    print("Gradient of total loss:", total_loss_grad)
    num_random_starts = 1
    best_sol = None
    best_loss = np.inf
    for i in range(num_random_starts):
        print("Random start", i + 1, "of", num_random_starts)
        pursuerX = np.random.uniform(lowerLimit, upperLimit)

        def objfunc(xDict):
            pursuerX = xDict["pursuerX"]
            loss = total_learning_loss(
                pursuerX,
                headings,
                speeds,
                interceptedList,
                pathHistories,
                pathMasks,
                endPoints,
            )
            funcs = {}
            funcs["loss"] = loss
            return funcs, False

        def sens(xDict, funcs):
            dX = dTotalLossDX(
                xDict["pursuerX"],
                headings,
                speeds,
                interceptedList,
                pathHistories,
                pathMasks,
            )
            print("Gradient of loss:", dX)
            funcsSens = {}
            funcsSens["loss"] = {
                "pursuerX": dX,
            }
            return funcsSens, False

        optProb = Optimization("path optimization", objfunc)
        optProb.addVarGroup(
            name="pursuerX",
            nVars=6,
            varType="c",
            value=pursuerX,
            lower=lowerLimit,
            upper=upperLimit,
        )
        optProb.addObj("loss")
        opt = OPT("ipopt")
        opt.options["print_level"] = 0
        opt.options["max_iter"] = 1000
        username = getpass.getuser()
        opt.options["hsllib"] = (
            "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
        )
        opt.options["linear_solver"] = "ma97"
        opt.options["derivative_test"] = "first-order"

        sol = opt(optProb, sens="FD")
        # sol = opt(optProb, sens=sens)
        if sol.fStar < best_loss:
            print("New best solution found with loss:", sol.fStar)
            best_loss = sol.fStar
            best_sol = sol
    print(best_sol.xStar)
    print("Objective function value:", best_sol.fStar)
    best_loss = total_learning_loss(
        pursuerX, headings, speeds, interceptedList, pathHistories, pathMasks, endPoints
    )

    lossHessian = jax.hessian(total_learning_loss, argnums=0)(
        pursuerX, headings, speeds, interceptedList, pathHistories, pathMasks, endPoints
    )
    print("Loss Hessian:\n", lossHessian)
    cov = jnp.linalg.inv(lossHessian)
    pursuerX = best_sol.xStar["pursuerX"]
    (
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
    ) = pursuerX_to_params(pursuerX)
    return (
        pursuerX,
        cov,
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
    )


def evenly_spaced_entry_points_with_heading_noise(
    xBounds, yBounds, num_points_per_side=10, heading_noise_std=0.5
):
    """
    Evenly place entry points around the boundary of a rectangular region.
    Each point gets a heading pointing approximately toward the origin (0,0),
    with added Gaussian noise.

    Args:
        xBounds: tuple of (xmin, xmax)
        yBounds: tuple of (ymin, ymax)
        num_points_per_side: how many points per side of the rectangle
        heading_noise_std: standard deviation of heading noise

    Returns:
        List of tuples: (start_pos: np.array of shape (2,), heading: float)
    """
    xmin, xmax = xBounds
    ymin, ymax = yBounds

    # Generate evenly spaced points on each side
    left = [(xmin, y) for y in np.linspace(ymin, ymax, num_points_per_side)]
    right = [(xmax, y) for y in np.linspace(ymin, ymax, num_points_per_side)]
    bottom = [(x, ymin) for x in np.linspace(xmin, xmax, num_points_per_side)]
    top = [(x, ymax) for x in np.linspace(xmin, xmax, num_points_per_side)]

    all_points = left + right + bottom + top

    results = []
    for point in all_points:
        start_pos = np.array(point, dtype=np.float32)
        direction_to_center = -start_pos
        heading = np.arctan2(direction_to_center[1], direction_to_center[0])
        heading += np.random.normal(0.0, heading_noise_std)
        results.append((start_pos, heading))

    return results


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerHeading = (5.0 / 20.0) * np.pi
    pursuerRange = 1.0
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerTurnRadius = 0.2
    agentSpeed = 1.0
    xbounds = (-3.0, 3.0)
    ybounds = (-3.0, 3.0)
    dt = 0.1
    tmax = (
        np.sqrt((xbounds[1] - xbounds[0]) ** 2 + (ybounds[1] - ybounds[0]) ** 2)
        / agentSpeed
    )
    print("tmax", tmax)

    numPoints = int(tmax / dt) + 1

    interceptedList = []
    numLowPriorityAgents = 20

    endPoints = []
    endTimes = []
    pathHistories = []
    pathMasks = []
    startPositions = []
    headings = []
    speeds = []

    # agents = evenly_spaced_entry_points_with_heading_noise(
    #     xbounds, ybounds, num_points_per_side=5
    # )
    for _ in range(numLowPriorityAgents):
        # for startPosition, heading in agents:
        startPosition, heading = sample_entry_point_and_heading(xbounds, ybounds)
        startPositions.append(startPosition)
        headings.append(heading)
        speeds.append(agentSpeed)
        intercepted, endPoint, endTime, pathHistory, pathMask = send_low_priority_agent(
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
        pathMasks.append(pathMask)

    interceptedList = jnp.array(interceptedList)
    endPoints = jnp.array(endPoints)
    endTimes = jnp.array(endTimes)
    pathHistories = jnp.array(pathHistories)
    pathMasks = jnp.array(pathMasks)
    headings = jnp.array(headings)
    speeds = jnp.array(speeds)

    plot_low_priority_paths(
        startPositions,
        interceptedList,
        endPoints,
        pathHistories,
        pathMasks,
    )

    print("LEARNIGN")
    (
        pursuerX,
        pursuerCov,
        pursuerPositionLearned,
        pursuerHeadingLearned,
        pursuerSpeedLearned,
        minimumTurnRadiusLearned,
        pursuerRangeLearned,
    ) = learn_ez(
        headings, speeds, interceptedList, pathHistories, pathMasks, endPoints, endTimes
    )
    print("Pursuer position:", pursuerPosition)
    print("Pursuer heading:", pursuerHeading)
    print("Pursuer speed:", pursuerSpeed)
    print("Pursuer turn radius:", pursuerTurnRadius)
    print("Pursuer range:", pursuerRange)
    print("Learned pursuer position:", pursuerPositionLearned)
    print("Learned pursuer heading:", pursuerHeadingLearned)
    print("Learned pursuer speed:", pursuerSpeedLearned)
    print("Learned pursuer turn radius:", minimumTurnRadiusLearned)
    print("Learned pursuer range:", pursuerRangeLearned)
    print("Pursuer covariance matrix:\n", pursuerCov)

    fig, ax = plt.subplots()
    plot_low_priority_paths_with_ez(
        headings,
        speeds,
        startPositions,
        interceptedList,
        endPoints,
        pathHistories,
        pathMasks,
        pursuerX,
        ax,
    )
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
    dubinsEZ.plot_dubins_reachable_set(
        pursuerPosition, pursuerHeading, pursuerRange, pursuerTurnRadius, ax
    )
    dubinsEZ.plot_dubins_reachable_set(
        pursuerPositionLearned,
        pursuerHeadingLearned,
        pursuerRangeLearned,
        minimumTurnRadiusLearned,
        ax,
        colors=["green"],
    )
    plt.legend()


if __name__ == "__main__":
    main()
    plt.show()
