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


def plot_low_priority_paths_with_prob(
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
        pez = nueral_network_PEZ(
            pursuerX, pathHistory, heading * np.ones(len(pathHistory)), speeds[i]
        )
        print("max pez", jnp.max(pez))
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
                c=pez[pathMask],
                vmin=0.0,
                vmax=1.0,
            )
        else:
            c = ax.scatter(
                pathHistory[:, 0][pathMask],
                pathHistory[:, 1][pathMask],
                c=pez[pathMask],
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
@jax.jit
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
    pez = nueral_network_PEZ(pursuerX, pathHistory, headings, speed)  # (T,)
    pez = jnp.clip(pez, 1e-4, 1.0 - 1e-4)  # prevent numerical issues

    pez = jnp.where(pathMask, pez, 0.0)

    def loss_if_intercepted():
        mean_p = jnp.mean(pez)
        return -jnp.log(mean_p)

    def loss_if_survived():
        log_escape = jnp.log1p(-pez)
        return -jnp.sum(log_escape)

    # def loss_if_intercepted():
    #     # Encourage at least one high PEZ prob
    #     # Using mean is more stable than product or max
    #     # return -2.0 * jnp.log(jnp.mean(p_t) + epsilon)
    #     # clipped_pez = jnp.clip(pez, 0.0, 0.95)
    #     one_minus_p = 1.0 - pez + epsilon
    #     prob_escape = jnp.prod(jnp.where(pathMask, one_minus_p, 1.0))
    #     prob_hit = 1.0 - prob_escape
    #     return -jnp.log(prob_hit + epsilon)
    #
    # def loss_if_survived():
    #     # Encourage all probs to be near zero
    #     log_escape = jnp.log1p(-pez)
    #     return -jnp.sum(log_escape * pathMask)

    return jax.lax.cond(intercepted, loss_if_intercepted, loss_if_survived)


batched_loss = jax.jit(
    jax.vmap(learning_loss_function_single, in_axes=(None, 0, 0, 0, 0, 0))
)


@jax.jit
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
    return jnp.sum(losses) / len(losses)


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
    upperLimit = jnp.array(
        [2.0, 2.0, 2.0, 1.0, 2.00, jnp.pi, 5.0, 5.0, 5.0, 5.0, 5.00, 5.0, 5.00]
    )
    # upperLimit = 10 * jnp.ones_like(lowerLimit)
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
    pursuerX = (lowerLimit + upperLimit) / 2.0
    # pursuerX = np.random.uniform(lowerLimit, upperLimit)
    num_random_starts = 5
    best_sol = None
    best_loss = np.inf
    for i in range(num_random_starts):
        print("Random start", i + 1, "of", num_random_starts)
        pursuerX = np.random.uniform(lowerLimit, upperLimit)

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
        opt.options["print_level"] = 0
        opt.options["max_iter"] = 1000
        username = getpass.getuser()
        opt.options["hsllib"] = (
            "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
        )
        opt.options["linear_solver"] = "ma97"
        opt.options["derivative_test"] = "first-order"

        sol = opt(optProb, sens=sens)
        if sol.fStar < best_loss:
            print("New best solution found with loss:", sol.fStar)
            best_loss = sol.fStar
            best_sol = sol
    print(best_sol.xStar)
    print("Objective function value:", best_sol.fStar)

    pursuerX = best_sol.xStar["pursuerX"]
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
        pursuerX,
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
    pursuerHeading = (0.0 / 20.0) * np.pi
    pursuerRange = 1.0
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerTurnRadius = 0.2
    agentSpeed = 1
    xbounds = (-2.0, 2.0)
    ybounds = (-2.0, 2.0)
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
        pursuerX,
        pursuerPositionLearned,
        pursuerPositionCovLearned,
        pursuerHeadingLearned,
        pursuerHeadingVarLearned,
        pursuerSpeedLearned,
        pursuerSpeedVarLearned,
        minimumTurnRadiusLearned,
        minimumTurnRadiusVarLearned,
        pursuerRangeLearned,
        pursuerRangeVarLearned,
    ) = learn_pez(
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

    fig, ax = plt.subplots()
    plot_low_priority_paths_with_prob(
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
    plt.legend()


if __name__ == "__main__":
    main()
    plt.show()
