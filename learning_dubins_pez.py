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
    inEZ = (
        dubinsEZ.in_dubins_engagement_zone(
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
        < 0
    )
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

    start_pos = np.array([x, y], dtype=np.float32)

    # Compute angle toward origin
    direction_to_center = -start_pos  # vector from point to (0,0)
    heading = np.arctan2(direction_to_center[1], direction_to_center[0])

    # Add Gaussian noise to heading
    heading_noise_std = 0.5  # Standard deviation of noise
    heading += np.random.normal(0.0, heading_noise_std)

    return start_pos, heading


def pursuerX_to_params(X):
    pursuerPosition = X[0:2]
    pursuerXVar = X[2]
    pursuerYVar = X[3]
    pursuerXYcov = X[4]
    pursuerPositionCov = jnp.array(
        [[pursuerXVar, pursuerXYcov], [pursuerXYcov, pursuerYVar]]
    )
    pursuerHeading = X[5]
    pursuerHeadingVar = X[6]
    pursuerSpeed = X[7]
    pursuerSpeedVar = X[8]
    minimumTurnRadius = X[9]
    minimumTurnRadiusVar = X[10]
    pursuerRange = X[11]
    pursuerRangeVar = X[12]
    return (
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
    )


def nueral_network_PEZ(X, evaderPositions, evaderHeadings, evaderSpeed):
    (
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
    ) = pursuerX_to_params(X)
    ZTrue, _, _ = nueral_network_EZ.nueral_network_pez(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        0.0,
    )
    return ZTrue


#### pez learning code ####
def learning_loss_function_single(
    pursuerX,
    heading,
    speed,
    intercepted,
    pathHistory,
    pathMask,
    epsilon=1e-6,
):
    headings = heading * jnp.ones(pathHistory.shape[0])
    p_t = nueral_network_PEZ(pursuerX, pathHistory, headings, speed)  # (T,)
    p_t = jnp.clip(p_t, 1e-4, 1.0 - 1e-4)  # prevent numerical issues

    p_t = jnp.where(pathMask, p_t, 0.0)

    def intercepted_condition():
        log_escape = jnp.log1p(-p_t)
        log_escape = log_escape * pathMask
        log_prob_escape = jnp.sum(log_escape)
        prob_intercept = 1.0 - jnp.exp(log_prob_escape)
        return -jnp.log(prob_intercept + epsilon)

    def not_intercepted_condition():
        log_escape = jnp.log1p(-p_t)
        return -jnp.sum(log_escape * pathMask)

    return jax.lax.cond(intercepted, intercepted_condition, not_intercepted_condition)


batched_loss = jax.jit(
    jax.vmap(learning_loss_function_single, in_axes=(None, 0, 0, 0, 0, 0))
)


def total_learning_loss(
    pursuerX,
    headings,
    speeds,
    intercepted_flags,
    pathHistories,
    pathMasks,
):
    # shape (N,)
    losses = batched_loss(
        pursuerX,
        headings,
        speeds,
        intercepted_flags,
        pathHistories,
        pathMasks,
    )

    # total loss = sum over agents
    return jnp.sum(losses)


dTotalLossDX = jax.jit(jax.grad(total_learning_loss, argnums=0))


def learn_pez(
    headings, speeds, interceptedList, pathHistories, pathMasks, endPoints, endTimes
):
    pursuerX = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            10.0 / 20.0 * jnp.pi,
            0.0,
            2.0,
            0.0,
            0.2,
            0.00,
            1.0,
            0.0,
        ]
    )
    lowerLimit = jnp.array(
        [-2.0, -2.0, 0.0, -1.0, 0.00, -jnp.pi, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.00]
    )
    upperLimit = 10 * jnp.ones_like(lowerLimit)
    headings = jnp.array(headings)
    total_loss = total_learning_loss(
        pursuerX,
        headings,
        speeds,
        interceptedList,
        pathHistories,
        pathMasks,
    )
    print("Total loss for all agents:", total_loss)
    # pursuerX = np.random.uniform(lowerLimit, upperLimit)

    def objfunc(xDict):
        pursuerX = xDict["pursuerX"]
        loss = total_learning_loss(
            pursuerX, headings, speeds, interceptedList, pathHistories, pathMasks
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
        funcsSens = {}
        funcsSens["loss"] = {
            "pursuerX": dX,
        }
        return funcsSens, False

    optProb = Optimization("path optimization", objfunc)
    optProb.addVarGroup(
        name="pursuerX",
        nVars=13,
        varType="c",
        value=pursuerX,
        lower=lowerLimit,
        upper=upperLimit,
    )
    optProb.addObj("loss")
    opt = OPT("ipopt")
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 200
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    opt.options["derivative_test"] = "first-order"

    sol = opt(optProb, sens=sens)
    print(sol.xStar)
    print("Objective function value:", sol.fStar)

    pursuerX = sol.xStar["pursuerX"]
    (
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
    ) = pursuerX_to_params(pursuerX)
    return (
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
    )


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerHeading = (10.0 / 20.0) * np.pi
    pursuerRange = 1.0
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerTurnRadius = 0.2
    agentSpeed = 1
    xbounds = (-2.0, 2.0)
    ybounds = (-2.0, 2.0)
    dt = 0.01

    tmax = (
        np.sqrt((xbounds[1] - xbounds[0]) ** 2 + (ybounds[1] - ybounds[0]) ** 2)
        / agentSpeed
    )
    print("tmax", tmax)

    numPoints = int(tmax / dt) + 1

    numLowPriorityAgents = 20

    interceptedList = []
    endPoints = []
    endTimes = []
    pathHistories = []
    pathMasks = []
    startPositions = []
    headings = []
    speeds = []

    for _ in range(numLowPriorityAgents):
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
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
    ) = learn_pez(
        headings, speeds, interceptedList, pathHistories, pathMasks, endPoints, endTimes
    )
    print("Pursuer position:", pursuerPosition)
    print("Pursuer heading:", pursuerHeading)
    print("Pursuer speed:", pursuerSpeed)
    print("Pursuer turn radius:", minimumTurnRadius)
    print("Pursuer range:", pursuerRange)


if __name__ == "__main__":
    main()
    plt.show()
