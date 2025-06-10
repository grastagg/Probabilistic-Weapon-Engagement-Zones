import re
import jax
import getpass
from jax._src.sharding_impls import PositionalSharding
from pyoptsparse import Optimization, OPT, IPOPT
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import dubinsEZ
import dubinsPEZ

jax.config.update("jax_enable_x64", True)

positionAndHeadingOnly = False


np.random.seed(326)  # for reproducibility


def plot_low_priority_paths(
    startPositions, interceptedList, endPoints, pathHistories, pathMasks, ax
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
        pathMask = pathMasks[i]
        if interceptedList[i]:
            ax.scatter(
                endPoints[i][0],
                endPoints[i][1],
                color="red",
                marker="x",
            )
            if interceptedCounter == 0:
                ax.plot(
                    pathHistory[:, 0][pathMask],
                    pathHistory[:, 1][pathMask],
                    c="r",
                    label="Intercepted",
                )
            else:
                ax.plot(
                    pathHistory[:, 0][pathMask],
                    pathHistory[:, 1][pathMask],
                    c="r",
                )
            interceptedCounter += 1
        else:
            if survivedCounter == 0:
                ax.plot(
                    pathHistory[:, 0][pathMask],
                    pathHistory[:, 1][pathMask],
                    c="g",
                    label="Survived",
                )
            else:
                ax.plot(
                    pathHistory[:, 0][pathMask],
                    pathHistory[:, 1][pathMask],
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
    pathMasks,
    pursuerX,
    trueParams,
    ax,
):
    ax.set_aspect("equal", adjustable="box")
    for i in range(len(startPositions)):
        pathHistory = pathHistories[i]

        heading = headings[i]
        ez = (
            dubinsEZ_from_pursuerX(
                pursuerX,
                pathHistory,
                heading * np.ones(len(pathHistory)),
                speeds[i],
                trueParams,
            )
            < 0
        )

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
    # print(
    #     "test agent will be intercepted at",
    #     interceptionPoint,
    #     "at time",
    #     interceptionTime,
    # )
    mask = t < interceptionTime
    return (intercepted, interceptionPoint, interceptionTime, mask)


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
    # currentPosition = jnp.array(startPosition)
    t = jnp.linspace(0.0, tmax, numPoints)  # shape (T,)
    direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])  # shape (2,)
    # displacement = t[:, None] * speed * direction  # shape (T, 2)
    # pathHistory = currentPosition + displacement  # shape (T, 2)
    # headings = heading * jnp.ones(numPoints)  # shape (T,)
    pathHistory, headings = simulate_trajectory_fn(
        startPosition, heading, speed, tmax, numPoints
    )
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
        intercepted, interceptionPoint, interceptionTime, mask = (
            find_interception_point_and_time(
                speed,
                pursuerSpeed,
                direction,
                pursuerRange,
                t,
                pathHistory,
                firstTrueIndex,
            )
        )

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
    return jax.nn.relu(x)  # ReLU activation function
    return jnp.log1p(jnp.exp(beta * x)) / beta
    return (jnp.tanh(10.0 * x) + 1.0) / 2.0 * x**2


def compute_intercept_probability(ez_min, alpha=10.0):
    # return ez_min > 0.0
    return jax.nn.sigmoid(-alpha * ez_min)


@jax.jit
def learning_loss_function_single(
    pursuerX,
    heading,
    speed,
    intercepted,
    pathHistory,
    pathMask,
    interceptedPoint,
    trueParams,
):
    headings = heading * jnp.ones(pathHistory.shape[0])
    ez = dubinsEZ_from_pursuerX(
        pursuerX,
        pathHistory,
        headings,
        speed,
        trueParams,
    )  # (T,)
    rsEnd = dubins_reachable_set_from_pursuerX(
        pursuerX,
        jnp.array([interceptedPoint]),
        trueParams,
    )[0]
    rsEnd = jnp.where(jnp.isinf(rsEnd), 1000.0, rsEnd)

    interceptedLossEZ = activation(jnp.min(ez))  # loss if intercepted
    survivedLossEZ = activation(-jnp.min(ez))  # loss if survived
    lossEZ = jax.lax.cond(
        intercepted, lambda: interceptedLossEZ, lambda: survivedLossEZ
    )
    interceptedLossRS = jax.nn.relu(rsEnd)  # loss if intercepted in RS
    survivedLossRS = 0.0  # loss if survived in RS
    # survivedLossRS = jax.nn.relu(-jnp.min(rsAll))  # loss if survived in RS
    lossRS = jax.lax.cond(
        intercepted, lambda: interceptedLossRS, lambda: survivedLossRS
    )
    # return rsEnd**2
    # return lossRS
    # jax.debug.print(
    #     "intercepted: {}, ez_min: {}, rsEnd: {}, lossEZ: {}, lossRS: {}",
    #     intercepted,
    #     jnp.min(ez),
    #     rsEnd,
    #     lossEZ,
    #     lossRS,
    # )
    return lossEZ + lossRS


batched_loss = jax.jit(
    jax.vmap(
        learning_loss_function_single,
        in_axes=(None, 0, 0, 0, 0, 0, 0, None),
    )
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
    trueParams,
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
        trueParams,
    )

    # total loss = sum over agents
    return jnp.sum(losses) / len(losses)


def total_learning_log_loss(
    pursuerX,
    headings,
    speeds,
    intercepted_flags,
    pathHistories,
    pathMasks,
    interceptedPoints,
    trueParams,
):
    losses = batched_log_loss(
        pursuerX,
        headings,
        speeds,
        intercepted_flags,
        pathHistories,
        pathMasks,
        interceptedPoints,
        trueParams,
    )
    return jnp.sum(losses)


batched_loss_multiple_pursuerX = jax.jit(
    jax.vmap(total_learning_loss, in_axes=(0, None, None, None, None, None, None, None))
)


dTotalLossDX = jax.jit(jax.jacfwd(total_learning_loss, argnums=0))
dTotalLogLossDX = jax.jit(jax.jacfwd(total_learning_log_loss, argnums=0))


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


def find_covariance_of_optimal(
    pursuerXopt,  # shape (D,), the optimal parameter estimate
    headings,
    speeds,
    intercepted_flags,
    pathHistories,
    pathMasks,
    interceptedPoints,
    trueParams,
    N=1000,  # number of samples
    std=0.5,  # stddev for perturbing theta
    lambda_scale=1000.0,  # scales importance weights: exp(-lambda * loss)
    key=jax.random.PRNGKey(0),  # PRNG key for reproducibility
):
    hessian = jax.jacfwd(dTotalLogLossDX)(
        pursuerXopt,
        headings,
        speeds,
        intercepted_flags,
        pathHistories,
        pathMasks,
        interceptedPoints,
        trueParams,
    )
    cov = jnp.linalg.inv(hessian)

    return cov


def run_optimization_hueristic(
    headings,
    speeds,
    interceptedList,
    pathHistories,
    pathMasks,
    endPoints,
    endTimes,
    initialPursuerX,
    lowerLimit,
    upperLimit,
    trueParams,
):
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
            trueParams,
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
            pathMasks,
            endPoints,
            trueParams,
        )
        funcsSens = {}
        funcsSens["loss"] = {
            "pursuerX": dX,
        }
        return funcsSens, False

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
    print("Ran optimization hueristic...", sol.xStar["pursuerX"], " loss:", sol.fStar)
    return sol


def learn_ez(
    headings,
    speeds,
    interceptedList,
    pathHistories,
    pathMasks,
    endPoints,
    endTimes,
    trueParams,
    previousPursuerXList=None,
):
    lowerLimit = jnp.array([-2.0, -2.0, -jnp.pi, 0.0, 0.0, 0.0])
    upperLimit = jnp.array([2.0, 2.0, jnp.pi, 5.0, 2.0, 5.0])
    startPosition, startHeading1, startHeading2 = find_initial_position_and_heading(
        interceptedList, endPoints
    )

    numStartHeadings = 10
    initialPursuerXList = []
    initialHeadings = np.linspace(
        startHeading1, startHeading1 + 2 * np.pi, numStartHeadings
    )
    # map initial headint to [_pi, pi]
    initialHeadings = np.mod(initialHeadings + np.pi, 2 * np.pi) - np.pi
    for i in range(numStartHeadings):
        previousPursuerX = (
            previousPursuerXList[i] if previousPursuerXList is not None else None
        )
        if positionAndHeadingOnly:
            lowerLimit = lowerLimit[:3]
            upperLimit = upperLimit[:3]

        if positionAndHeadingOnly:
            intialPursuerX = jnp.array(
                [
                    startPosition[0],
                    startPosition[1],
                    initialHeadings[i],
                ]
            )
        else:
            if previousPursuerX is not None:
                initialPosition = startPosition
                initialSpeed = previousPursuerX[3]
                initialTurnRadius = previousPursuerX[4]
                initialRange = previousPursuerX[5]
            else:
                initialPosition = startPosition
                initialSpeed = (lowerLimit[3] + upperLimit[3]) / 2.0
                initialTurnRadius = (lowerLimit[4] + upperLimit[4]) / 2.0
                initialRange = (lowerLimit[5] + upperLimit[5]) / 2.0
            intialPursuerX = jnp.array(
                [
                    initialPosition[0],
                    initialPosition[1],
                    initialHeadings[i],
                    initialSpeed,
                    initialTurnRadius,
                    initialRange,
                ]
            )
        initialPursuerXList.append(intialPursuerX)
    pursuerXList = []
    lossList = []
    for i in range(len(initialPursuerXList)):
        sol = run_optimization_hueristic(
            headings,
            speeds,
            interceptedList,
            pathHistories,
            pathMasks,
            endPoints,
            endTimes,
            initialPursuerXList[i],
            lowerLimit,
            upperLimit,
            trueParams,
        )
        pursuerXList.append(sol.xStar["pursuerX"])
        lossList.append(sol.fStar)

    pursuerXList = jnp.array(pursuerXList).squeeze()
    lossList = jnp.array(lossList).squeeze()
    sorted_indices = jnp.argsort(lossList)
    lossList = lossList[sorted_indices]
    pursuerXList = pursuerXList[sorted_indices]
    print("loss", lossList)
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


def ez_probability_fn(pursuerParms, pathHistory, headings, speed, trueParams):
    ez = dubinsEZ_from_pursuerX(
        pursuerParms,
        pathHistory,
        headings,
        speed,
        trueParams,
    )
    alpha = 10.0  # Smoothing parameter for sigmoid
    return jax.nn.sigmoid(-jnp.min(ez) * alpha)  # , ez  # Probability of interception


def compute_mutual_information(
    angle,
    heading,
    theta_hat,
    theta_hat2,
    cov_theta,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    num_points=100,
    num_samples=1000,
    alpha=20.0,
    key=jax.random.PRNGKey(0),
):
    position = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    pathHistory, headings = simulate_trajectory_fn(
        position, heading, speed, tmax, num_points
    )

    # Sample theta ~ N(theta_hat, cov_theta)
    theta_samples = jax.random.multivariate_normal(
        key, mean=theta_hat, cov=cov_theta, shape=(num_samples,)
    )

    # Define p_theta(x): smooth probability of interception
    def prob_fn(theta):
        ez = dubinsEZ_from_pursuerX(theta, pathHistory, headings, speed, trueParams)
        # return jnp.min(ez) < 0.0  # True if inside engagement zone
        margin = -jnp.min(ez)  # positive if inside PEZ
        return jax.nn.sigmoid(alpha * margin)

    # Compute p_i for all theta_i
    probs = jax.vmap(prob_fn)(theta_samples)

    # Clip to avoid log(0)
    probs = jnp.clip(probs, 1e-6, 1 - 1e-6)

    # 1. Marginal interception probability
    p_bar = jnp.mean(probs)

    # 2. Entropy of marginal
    marginal_entropy = -p_bar * jnp.log(p_bar) - (1 - p_bar) * jnp.log(1 - p_bar)

    # 3. Expected conditional entropy
    conditional_entropies = -probs * jnp.log(probs) - (1 - probs) * jnp.log(1 - probs)
    expected_cond_entropy = jnp.mean(conditional_entropies)

    # 4. Mutual information
    mutual_info = marginal_entropy - expected_cond_entropy
    return mutual_info


def softmin(x, tau=0.5):
    x = jnp.asarray(x)
    return -tau * jnp.log(jnp.sum(jnp.exp(-x / tau)))


def expected_grad_score(
    angle,
    heading,
    pursuerX1,
    pursuerX2,
    cov_theta,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    num_points=100,
    headings=None,
    pathHistories=None,
    pathMasks=None,
    interceptedList=None,
    endPoints=None,
    speeds=None,
):
    start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    new_path, new_headings = simulate_trajectory_fn(
        start_pos, heading, speed, tmax, num_points
    )
    new_headings = jnp.reshape(new_headings, (-1,))

    new_mask = jnp.ones(new_path.shape[0], dtype=bool)
    new_end = new_path[-1]

    def grad_norm_if(intercepted, pursuerX):
        new_pathHistories = jnp.concatenate(
            [pathHistories, new_path[None, :, :]], axis=0
        )
        new_headings_all = jnp.concatenate(
            [headings, jnp.array([new_headings[0]])], axis=0
        )

        new_speeds = jnp.concatenate([speeds, jnp.array([speed])], axis=0)
        new_intercepted = jnp.concatenate(
            [interceptedList, jnp.array([intercepted])], axis=0
        )
        new_masks = jnp.concatenate([pathMasks, new_mask[None, :]], axis=0)
        new_endpoints = jnp.concatenate([endPoints, new_end[None, :]], axis=0)

        grad = dTotalLossDX(
            pursuerX,
            new_headings_all,
            new_speeds,
            new_intercepted,
            new_pathHistories,
            new_masks,
            new_endpoints,
            trueParams,
        )
        return grad

    # ez1 = dubinsEZ_from_pursuerX(pursuerX1, new_path, new_headings, speed, trueParams)
    # ez2 = dubinsEZ_from_pursuerX(pursuerX2, new_path, new_headings, speed, trueParams)
    # ez1_margin = softmin(ez1)
    # ez2_margin = softmin(ez2)
    # p1 = jax.nn.sigmoid(-ez1_margin / 0.5)
    # p2 = jax.nn.sigmoid(-ez2_margin / 0.5)

    g1_hit = grad_norm_if(True, pursuerX1)
    g1_miss = grad_norm_if(False, pursuerX1)
    g2_hit = grad_norm_if(True, pursuerX2)
    g2_miss = grad_norm_if(False, pursuerX2)
    p1 = 0.5
    p2 = 0.5
    grad_sq_norm_1 = p1 * jnp.sum(g1_hit**2)  # + (1 - p1) * jnp.sum(g1_miss**2)
    grad_sq_norm_2 = p2 * jnp.sum(g2_hit**2)  # + (1 - p2) * jnp.sum(g2_miss**2)
    return -grad_sq_norm_1 - grad_sq_norm_2


def maximize_loss(
    angle,
    heading,
    pursuerX1,
    pursuerX2,
    cov_theta,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    num_points=100,
    headings=None,
    pathHistories=None,
    pathMasks=None,
    interceptedList=None,
    endPoints=None,
    speeds=None,
):
    start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    new_path, new_headings = simulate_trajectory_fn(
        start_pos, heading, speed, tmax, num_points
    )
    new_headings = jnp.reshape(new_headings, (-1,))

    def loss_if(pursuerX1, pursuerX2):
        new_pathHistories = jnp.concatenate(
            [pathHistories, new_path[None, :, :]], axis=0
        )
        new_headings_all = jnp.concatenate(
            [headings, jnp.array([new_headings[0]])], axis=0
        )

        new_speeds = jnp.concatenate([speeds, jnp.array([speed])], axis=0)

        ez = dubinsEZ_from_pursuerX(
            pursuerX1, new_path, new_headings, speed, trueParams
        )
        inEZ = ez < 0.0  # shape (T,)
        intercepted = jnp.any(inEZ)

        def if_intercepted():
            firstTrueIndex = first_true_index_safe(inEZ)

            direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])  # shape (2,)
            t = jnp.linspace(0.0, tmax, num_points)  # shape (T,)
            intercepted, new_end, interceptionTime, new_mask = (
                find_interception_point_and_time(
                    speed,
                    pursuerX1[3],  # pursuerSpeed
                    direction,  # direction
                    pursuerX1[5],  # pursuerRange
                    t,
                    new_path,
                    firstTrueIndex,
                )
            )
            return new_mask, new_end

        def if_not_intercepted():
            new_mask = jnp.ones(new_path.shape[0], dtype=bool)
            new_end = new_path[-1]
            return new_mask, new_end

        new_mask, new_end = jax.lax.cond(
            intercepted, if_intercepted, if_not_intercepted
        )

        new_intercepted = jnp.concatenate(
            [interceptedList, jnp.array([intercepted])], axis=0
        )
        new_masks = jnp.concatenate([pathMasks, new_mask[None, :]], axis=0)
        new_endpoints = jnp.concatenate([endPoints, new_end[None, :]], axis=0)

        loss = total_learning_loss(
            pursuerX2,
            new_headings_all,
            new_speeds,
            new_intercepted,
            new_pathHistories,
            new_masks,
            new_endpoints,
            trueParams,
        )
        return loss

    loss1 = loss_if(pursuerX1, pursuerX2)
    loss2 = loss_if(pursuerX2, pursuerX1)
    # return jnp.maximum(loss1, loss2)

    return loss1 + loss2


def maximize_loss_multi(
    angle,
    heading,
    pursuerXList,  # shape (N, D)
    cov_theta,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    num_points=100,
    headings=None,
    pathHistories=None,
    pathMasks=None,
    interceptedList=None,
    endPoints=None,
    speeds=None,
):
    N = pursuerXList.shape[0]

    # Simulate new path
    start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    new_path, new_headings = simulate_trajectory_fn(
        start_pos, heading, speed, tmax, num_points
    )
    new_headings = jnp.reshape(new_headings, (-1,))
    direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])  # shape (2,)
    t = jnp.linspace(0.0, tmax, num_points)  # shape (T,)

    def loss_if(pX_intercept, pX_loss):
        ez = dubinsEZ_from_pursuerX(
            pX_intercept, new_path, new_headings, speed, trueParams
        )
        inEZ = ez < 0.0
        intercepted = jnp.any(inEZ)

        def if_intercepted():
            firstTrueIndex = first_true_index_safe(inEZ)
            intercepted, new_end, interceptionTime, new_mask = (
                find_interception_point_and_time(
                    speed,
                    pX_intercept[3],  # pursuerSpeed
                    direction,
                    pX_intercept[5],  # pursuerRange
                    t,
                    new_path,
                    firstTrueIndex,
                )
            )
            return new_mask, new_end

        def if_not_intercepted():
            new_mask = jnp.ones(new_path.shape[0], dtype=bool)
            new_end = new_path[-1]
            return new_mask, new_end

        new_mask, new_end = jax.lax.cond(
            intercepted, if_intercepted, if_not_intercepted
        )

        new_pathHistories = jnp.concatenate(
            [pathHistories, new_path[None, :, :]], axis=0
        )
        new_headings_all = jnp.concatenate(
            [headings, jnp.array([new_headings[0]])], axis=0
        )
        new_speeds = jnp.concatenate([speeds, jnp.array([speed])], axis=0)
        new_intercepted = jnp.concatenate(
            [interceptedList, jnp.array([intercepted])], axis=0
        )
        new_masks = jnp.concatenate([pathMasks, new_mask[None, :]], axis=0)
        new_endpoints = jnp.concatenate([endPoints, new_end[None, :]], axis=0)

        loss = total_learning_loss(
            pX_loss,
            new_headings_all,
            new_speeds,
            new_intercepted,
            new_pathHistories,
            new_masks,
            new_endpoints,
            trueParams,
        )
        return loss

    # Sum loss over all unique ordered pairs (i, j) where i ≠ j
    def loss_pair(i, j):
        return loss_if(pursuerXList[i], pursuerXList[j])

    idxs = jnp.array([(i, j) for i in range(N) for j in range(N) if i != j])
    total_loss = jnp.sum(jax.vmap(lambda ij: loss_pair(ij[0], ij[1]))(idxs))
    return total_loss


def inside_one_outside_other(
    angle,
    heading,
    pursuerXList,
    cov_theta,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    num_points=100,
    headings=None,
    pathHistories=None,
    pathMasks=None,
    interceptedList=None,
    endPoints=None,
    speeds=None,
):
    start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    new_path, new_headings = simulate_trajectory_fn(
        start_pos, heading, speed, tmax, num_points
    )
    ez1 = dubinsEZ_from_pursuerX(pursuerX1, new_path, new_headings, speed, trueParams)
    rs1 = dubins_reachable_set_from_pursuerX(pursuerX1, new_path, trueParams)
    ez2 = dubinsEZ_from_pursuerX(pursuerX2, new_path, new_headings, speed, trueParams)
    rs2 = dubins_reachable_set_from_pursuerX(pursuerX2, new_path, trueParams)

    # inside1 = rs1 < 0.0  # shape (T,)
    # inside2 = rs2 < 0.0  # shape (T,)
    # inside1_outside2 = jnp.logical_and(inside1, ~inside2)  # inside 1, outside 2
    # inside2_outside1 = jnp.logical_and(inside2, ~inside1)  # inside 2, outside 1
    # return jnp.sum(inside1_outside2) + jnp.sum(inside2_outside1)

    # min1 = jnp.min(ez1)
    # min2 = jnp.min(ez2)
    min1 = jnp.min(rs1)
    min2 = jnp.min(rs2)
    epsilon = 0.1
    # Objective: find a path that is inside *exactly one* PEZ
    inside1_outside2 = jax.nn.sigmoid(-min1 / epsilon) * (
        1 - jax.nn.sigmoid(-min2 / epsilon)
    )
    inside2_outside1 = jax.nn.sigmoid(-min2 / epsilon) * (
        1 - jax.nn.sigmoid(-min1 / epsilon)
    )

    # Score: how strongly it distinguishes between the two
    score = inside1_outside2 + inside2_outside1
    return score


def inside_model_disagreement_score(
    angle,
    heading,
    pursuerXList,
    cov_theta,
    speed,
    trueParams,
    center,
    radius,
    tmax=10.0,
    num_points=100,
    headings=None,
    pathHistories=None,
    pathMasks=None,
    interceptedList=None,
    endPoints=None,
    speeds=None,
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
    def min_rs(pX):
        rs = dubins_reachable_set_from_pursuerX(pX, new_path, trueParams)
        return jnp.min(rs)

    min_vals = jax.vmap(min_rs)(pursuerXList)  # (N,)
    probs = min_vals < 0.0
    # probs = jax.nn.sigmoid(-min_vals / epsilon)  # (N,), soft in/out decision

    # Pairwise disagreement: p_i * (1 - p_j) + p_j * (1 - p_i)
    def pairwise_disagree(i, j):
        pi = probs[i]
        pj = probs[j]
        return pi * (1 - pj) + pj * (1 - pi)

    indices = jnp.arange(N)
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
    pathMasks,
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
        maximize_loss_multi,
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
        ),
    )(
        angle_flat,
        heading_flat,
        pursuerXList,
        None,
        speed,
        trueParams,
        center,
        radius,
        tmax,
        num_points,
        headings,
        pathHistories,
        pathMasks,
        interceptedList,
        endPoints,
        speeds,
    )
    print("min scores", jnp.min(scores))
    print("max score:", jnp.max(scores))

    best_idx = jnp.nanargmax(scores)

    best_angle = angle_flat[best_idx]
    best_heading = heading_flat[best_idx]
    best_start_pos = center + radius * jnp.array(
        [jnp.cos(best_angle), jnp.sin(best_angle)]
    )
    print("best location:", best_start_pos, "best heading:", best_heading)

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


def plot_all(
    startPositions,
    interceptedList,
    endPoints,
    pathHistories,
    pathMasks,
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
        startPositions, interceptedList, endPoints, pathHistories, pathMasks, ax
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
    # numPlots = 3
    fig1, axes = plt.subplots(2, int(np.ceil(numPlots / 2)))
    for i in range(numPlots):
        # pick ax
        ax = axes[i % 2, i // 2]

        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        plt.title("1")
        pursuerX = pursuerXList[i].squeeze()
        plot_low_priority_paths_with_ez(
            headings,
            speeds,
            startPositions,
            interceptedList,
            endPoints,
            pathHistories,
            pathMasks,
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


def main():
    pursuerPosition = np.array([0.5, 0.5])
    pursuerHeading = (-5.0 / 20.0) * np.pi
    pursuerRange = 2.0
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerTurnRadius = 0.4
    agentSpeed = 1.0
    dt = 0.11
    searchCircleCenter = np.array([0, 0])
    searchCircleRadius = 3.0
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

    interceptedList = []
    numLowPriorityAgents = 25
    endPoints = []
    endTimes = []
    pathHistories = []
    pathMasks = []
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
    pursuerXList = None
    for i in range(numLowPriorityAgents):
        print("iteration:", i)
        if i == 0:
            startPosition = jnp.array([-searchCircleRadius, 0.0001])
            heading = 0.0001
        else:
            startPosition, heading = optimize_next_low_priority_path(
                pursuerXList,
                jnp.array(headings),
                jnp.array(speeds),
                jnp.array(interceptedList),
                jnp.array(pathHistories),
                jnp.array(pathMasks),
                jnp.array(endPoints),
                jnp.array(endTimes),
                trueParams,
                searchCircleCenter,
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
        (pursuerXList, lossList) = learn_ez(
            jnp.array(headings),
            jnp.array(speeds),
            jnp.array(interceptedList),
            jnp.array(pathHistories),
            jnp.array(pathMasks),
            jnp.array(endPoints),
            jnp.array(endTimes),
            trueParams,
            pursuerXList,
        )
        if i % plotEvery == 0:
            fig1 = plot_all(
                startPositions,
                interceptedList,
                endPoints,
                pathHistories,
                pathMasks,
                pursuerPosition,
                pursuerHeading,
                pursuerRange,
                pursuerTurnRadius,
                headings,
                speeds,
                pursuerXList,
                lossList,
                trueParams,
            )
            fig1.savefig(f"video/{i}.png")
            plt.close(fig1)

        plt.close("all")
        pursuerX1 = pursuerXList[0]
        rmse = np.sqrt(np.mean((trueParams - pursuerX1) ** 2))
        pursuerParameterRMSE_history.append(rmse)
        pursuerParameter_history.append(pursuerX1)
    pursuerParameter_history = jnp.array(pursuerParameter_history)
    fig, axes = plt.subplots(3, 2)
    axes[0, 0].plot(pursuerParameter_history[:, 0])
    axes[0, 0].set_title("Pursuer X Position")
    axes[0, 0].plot(np.ones(len(pursuerParameter_history)) * trueParams[0], "r--")
    axes[0, 1].plot(pursuerParameter_history[:, 1])
    axes[0, 1].set_title("Pursuer Y Position")
    axes[0, 1].plot(np.ones(len(pursuerParameter_history)) * trueParams[1], "r--")
    axes[1, 0].plot(pursuerParameter_history[:, 2])
    axes[1, 0].set_title("Pursuer Heading")
    axes[1, 0].plot(np.ones(len(pursuerParameter_history)) * trueParams[2], "r--")
    axes[1, 1].plot(pursuerParameter_history[:, 3])
    axes[1, 1].set_title("Pursuer Speed")
    axes[1, 1].plot(np.ones(len(pursuerParameter_history)) * trueParams[3], "r--")
    axes[2, 0].plot(pursuerParameter_history[:, 4])
    axes[2, 0].set_title("Pursuer Turn Radius")
    axes[2, 0].plot(np.ones(len(pursuerParameter_history)) * trueParams[4], "r--")
    axes[2, 1].plot(pursuerParameter_history[:, 5])
    axes[2, 1].set_title("Pursuer Range")
    axes[2, 1].plot(np.ones(len(pursuerParameter_history)) * trueParams[5], "r--")


if __name__ == "__main__":
    main()
    plt.show()
