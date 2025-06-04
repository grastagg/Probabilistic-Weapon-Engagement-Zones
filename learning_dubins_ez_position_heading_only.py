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


def activation(x, beta=100.0):
    # return jnp.log1p(jnp.exp(beta * x)) / beta
    return jax.nn.relu(x)  # ReLU activation function
    return (jnp.tanh(10.0 * x) + 1.0) / 2.0 * x**2


def compute_intercept_probability(ez_min, alpha=10.0):
    # return ez_min > 0.0
    return jax.nn.sigmoid(-alpha * ez_min)


@jax.jit
def learning_log_likelihood_single(
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

    ez_min = jnp.min(jnp.where(pathMask, ez, jnp.inf))  # only active along path
    p = compute_intercept_probability(ez_min)  # ∈ (0, 1)

    # Clip p to avoid log(0)
    epsilon = 1e-6
    p = jnp.clip(p, epsilon, 1.0 - epsilon)

    loglik = jnp.log(p) * intercepted + jnp.log1p(-p) * (1 - intercepted)

    # Optional: keep RS loss if you want a hybrid loss
    rsEnd = dubins_reachable_set_from_pursuerX(
        pursuerX,
        jnp.array([interceptedPoint]),
        trueParams,
    )[0]
    rsLoss = activation(rsEnd)  # Non-negative loss if in RS
    rsLoss = jnp.where(intercepted, rsLoss, 0.0)  # Only apply if intercepted
    return -loglik + rsLoss  # Negative log-likelihood + optional penalty


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

    interceptedLossEZ = activation(jnp.min(ez))  # loss if intercepted
    survivedLossEZ = activation(-jnp.min(ez))  # loss if survived
    lossEZ = jax.lax.cond(
        intercepted, lambda: interceptedLossEZ, lambda: survivedLossEZ
    )
    interceptedLossRS = jax.nn.relu(rsEnd)  # loss if intercepted in RS
    survivedLossRS = 0.0  # loss if survived in RS
    lossRS = jax.lax.cond(
        intercepted, lambda: interceptedLossRS, lambda: survivedLossRS
    )
    # return lossRS
    return lossEZ + lossRS


batched_loss = jax.jit(
    jax.vmap(
        learning_loss_function_single,
        in_axes=(None, 0, 0, 0, 0, 0, 0, None),
    )
)
# batched_loss = jax.jit(
#     jax.vmap(
#         learning_log_likelihood_single,
#         in_axes=(None, 0, 0, 0, 0, 0, 0, None),
#     )
# )


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


batched_loss_multiple_pursuerX = jax.jit(
    jax.vmap(total_learning_loss, in_axes=(0, None, None, None, None, None, None, None))
)


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
    D = pursuerXopt.shape[0]

    # 1. Sample theta vectors around the optimum
    theta_samples = pursuerXopt + std * jax.random.normal(key, shape=(N, D))

    # 2. Evaluate loss for each sample
    losses = batched_loss_multiple_pursuerX(
        theta_samples,  # shape (N, D)
        headings,
        speeds,
        intercepted_flags,
        pathHistories,
        pathMasks,
        interceptedPoints,
        trueParams,
    ).squeeze()  # shape (N,)
    print("losses:", losses.shape)

    # 3. Convert to importance weights
    scaled_losses = -lambda_scale * (
        losses - jnp.min(losses)
    )  # for numerical stability
    weights = jnp.exp(scaled_losses)
    weights = weights / jnp.sum(weights)  # shape (N,)

    # 4. Compute weighted mean
    mean_theta = jnp.sum(weights[:, None] * theta_samples, axis=0)  # shape (D,)

    # 5. Compute weighted covariance
    centered = theta_samples - mean_theta  # shape (N, D)
    cov = centered.T @ (centered * weights[:, None])  # shape (D, D)

    return cov


def run_optimization(
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
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 1000
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    # opt.options["derivative_test"] = "first-order"

    # sol = opt(optProb, sens="FD")
    sol = opt(optProb, sens=sens)

    cov = find_covariance_of_optimal(
        sol.xStar["pursuerX"],
        headings,
        speeds,
        interceptedList,
        pathHistories,
        pathMasks,
        endPoints,
        trueParams,
    )

    # lossHessian = jax.jacfwd(jax.jacfwd(total_learning_loss, argnums=0))(
    #     sol.xStar["pursuerX"],
    #     headings,
    #     speeds,
    #     interceptedList,
    #     pathHistories,
    #     pathMasks,
    #     endPoints,
    #     trueParams,
    # )  # + 1e3 * np.eye(numVars)  # Add small value to diagonal for numerical stability
    # cov = jnp.linalg.inv(lossHessian)
    print("covariance matrix:", cov)
    return sol, cov


def learn_ez(
    headings,
    speeds,
    interceptedList,
    pathHistories,
    pathMasks,
    endPoints,
    endTimes,
    trueParams,
):
    lowerLimit = jnp.array([-2.0, -2.0, -jnp.pi, 0.0, 0.0, 0.0])
    upperLimit = jnp.array([2.0, 2.0, jnp.pi, 5.0, 2.0, 5.0])
    if positionAndHeadingOnly:
        lowerLimit = lowerLimit[:3]
        upperLimit = upperLimit[:3]

    startPosition, startHeading1, startHeading2 = find_initial_position_and_heading(
        interceptedList, endPoints
    )

    if positionAndHeadingOnly:
        intialPursuerX1 = jnp.array(
            [
                startPosition[0],
                startPosition[1],
                startHeading1,
            ]
        )
        intialPursuerX2 = jnp.array(
            [
                startPosition[0],
                startPosition[1],
                startHeading2,
            ]
        )
    else:
        intialPursuerX1 = jnp.array(
            [
                startPosition[0],
                startPosition[1],
                startHeading1,
                (lowerLimit[3] + upperLimit[3]) / 2.0,
                (lowerLimit[4] + upperLimit[4]) / 2.0,
                (lowerLimit[5] + upperLimit[5]) / 2.0,
            ]
        )
        intialPursuerX2 = jnp.array(
            [
                startPosition[0],
                startPosition[1],
                startHeading2,
                (lowerLimit[3] + upperLimit[3]) / 2.0,
                (lowerLimit[4] + upperLimit[4]) / 2.0,
                (lowerLimit[5] + upperLimit[5]) / 2.0,
            ]
        )
    sol1, cov1 = run_optimization(
        headings,
        speeds,
        interceptedList,
        pathHistories,
        pathMasks,
        endPoints,
        endTimes,
        intialPursuerX1,
        lowerLimit,
        upperLimit,
        trueParams,
    )
    sol2, cov2 = run_optimization(
        headings,
        speeds,
        interceptedList,
        pathHistories,
        pathMasks,
        endPoints,
        endTimes,
        intialPursuerX2,
        lowerLimit,
        upperLimit,
        trueParams,
    )
    pursuerX1 = sol1.xStar["pursuerX"]
    pursuerX2 = sol2.xStar["pursuerX"]
    print("Pursuer X1:", pursuerX1)
    print("loss 1", sol1.fStar)
    print("Pursuer X2:", pursuerX2)
    print("loss 2", sol2.fStar)
    bestParams = pursuerX1 if sol1.fStar < sol2.fStar else pursuerX2
    bestCov = cov1 if sol1.fStar < sol2.fStar else cov2

    return (
        pursuerX1,
        pursuerX2,
        bestParams,  # Return the best parameters as well
        bestCov,  # Return the covariance of the best parameters
    )


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
    cov_theta,
    speed,
    trueParams,
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


def optimize_next_low_priority_path(
    theta_hat,
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
    num_angles=32,
    num_headings=32,
    speed=1.0,
    tmax=10.0,
    num_points=100,
):
    print("Optimizing next low-priority path...")

    # Generate candidate start positions and headings
    angles = jnp.linspace(0, 2 * jnp.pi, num_angles, endpoint=False)
    headingsSac = jnp.linspace(-jnp.pi, jnp.pi, num_headings)
    angle_grid, heading_grid = jnp.meshgrid(angles, headingsSac)
    angle_flat = angle_grid.ravel()
    heading_flat = heading_grid.ravel()

    def expected_grad_score(angle, heading):
        start_pos = center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        new_path, new_headings = simulate_trajectory_fn(
            start_pos, heading, speed, tmax, num_points
        )
        new_headings = jnp.reshape(new_headings, (-1,))

        new_mask = jnp.ones(new_path.shape[0], dtype=bool)
        new_end = new_path[-1]

        def grad_norm_if(intercepted):
            print("new_headings:", new_headings.shape)
            print("headings", headings.shape)
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
                theta_hat,
                new_headings_all,
                new_speeds,
                new_intercepted,
                new_pathHistories,
                new_masks,
                new_endpoints,
                trueParams,
            )
            print("grad", grad)
            return jnp.sum(grad**2)

        p = ez_probability_fn(theta_hat, new_path, new_headings, speed, trueParams)
        p = jnp.clip(p, 1e-6, 1 - 1e-6)

        return p * grad_norm_if(True) + (1 - p) * grad_norm_if(False)

    scores = jax.vmap(
        expected_grad_score,
        in_axes=(
            0,
            0,
        ),
    )(angle_flat, heading_flat)

    best_idx = jnp.nanargmax(scores)

    best_angle = angle_flat[best_idx]
    best_heading = heading_flat[best_idx]
    best_start_pos = center + radius * jnp.array(
        [jnp.cos(best_angle), jnp.sin(best_angle)]
    )

    print("scores", scores)
    print("best start position:", best_start_pos)
    print("best heading:", best_heading)

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
    pursuerX1,
    pursuerX2,
    trueParams,
):
    fig, ax = plt.subplots()
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
    plt.legend(fontsize=18)

    fig, ax = plt.subplots()
    plt.title("1")
    plot_low_priority_paths_with_ez(
        headings,
        speeds,
        startPositions,
        interceptedList,
        endPoints,
        pathHistories,
        pathMasks,
        pursuerX1,
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
    ) = pursuerX_to_params(pursuerX1, trueParams)
    (
        pursuerPositionLearned2,
        pursuerHeadingLearned2,
        pursuerSpeedLearned2,
        minimumTurnRadiusLearned2,
        pursuerRangeLearned2,
    ) = pursuerX_to_params(pursuerX2, trueParams)
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
    plt.legend()

    fig1, ax1 = plt.subplots()
    plt.title("2")
    plot_low_priority_paths_with_ez(
        headings,
        speeds,
        startPositions,
        interceptedList,
        endPoints,
        pathHistories,
        pathMasks,
        pursuerX2,
        trueParams,
        ax1,
    )
    dubinsEZ.plot_dubins_reachable_set(
        pursuerPosition,
        pursuerHeading,
        pursuerRange,
        pursuerTurnRadius,
        ax1,
        colors=["green"],
    )
    dubinsEZ.plot_dubins_reachable_set(
        pursuerPositionLearned2,
        pursuerHeadingLearned2,
        pursuerRangeLearned2,
        minimumTurnRadiusLearned2,
        ax1,
        colors=["red"],
    )
    plot_true_and_learned_pursuer(
        pursuerPosition,
        pursuerHeading,
        pursuerPositionLearned2,
        pursuerHeadingLearned2,
        ax1,
    )
    plt.legend()


def main():
    pursuerPosition = np.array([-1.0, 1.0])
    pursuerHeading = (5.0 / 20.0) * np.pi
    pursuerRange = 1.0
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerTurnRadius = 0.2
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
    numLowPriorityAgents = 20

    endPoints = []
    endTimes = []
    pathHistories = []
    pathMasks = []
    startPositions = []
    headings = []
    speeds = []

    num_angles = 32
    num_headings = 32

    # agents = uniform_circular_entry_points_with_heading_noise(
    #     searchCircleCenter,
    #     searchCircleRadius,
    #     numLowPriorityAgents,
    #     heading_noise_std=0.5,
    # )
    plotEvery = 5
    for i in range(numLowPriorityAgents):
        if i == 0:
            startPosition = jnp.array([-searchCircleRadius, 0.0])
            heading = 0.0
        else:
            print("test heading:", headings)
            startPosition, heading = optimize_next_low_priority_path(
                theta_hat,
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
        print("LEARNING")
        (pursuerX1, pursuerX2, theta_hat, cov_theta) = learn_ez(
            jnp.array(headings),
            jnp.array(speeds),
            jnp.array(interceptedList),
            jnp.array(pathHistories),
            jnp.array(pathMasks),
            jnp.array(endPoints),
            jnp.array(endTimes),
            trueParams,
        )
        if i % plotEvery == 0:
            plot_all(
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
                pursuerX1,
                pursuerX2,
                trueParams,
            )
            plt.show()


if __name__ == "__main__":
    main()
