import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from functools import partial
import time


import fast_pursuer

import dubinsEZ

# Vectorized function using vmap
in_dubins_engagement_zone = jax.jit(
    jax.vmap(
        dubinsEZ.in_dubins_engagement_zone_single,
        in_axes=(
            0,  # pursuerPosition
            0,  # pursuerHeading
            0,  # minimumTurnRadius
            None,  # captureRadius
            0,  # pursuerRange
            0,  # pursuerSpeed
            None,  # evaderPosition
            None,  # evaderHeading
            None,  # evaderSpeed
        ),  # Vectorizing over evaderPosition & evaderHeading
    )
)
#
# in_dubins_engagement_zone = jax.jit(
#     jax.vmap(
#         dubinsEZ.in_dubins_engagement_zone_single,
#         in_axes=(
#             None,  # pursuerPosition
#             0,  # pursuerHeading
#             None,  # minimumTurnRadius
#             None,  # captureRadius
#             None,  # pursuerRange
#             None,  # pursuerSpeed
#             None,  # evaderPosition
#             None,  # evaderHeading
#             None,  # evaderSpeed
#         ),  # Vectorizing over evaderPosition & evaderHeading
#     )
# )


def mc_dubins_pez_single(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    pursuerRange,
    captureRadius,
    numSamples,
):
    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    # var = jnp.var(ez)
    # mean = jnp.mean(ez)
    return (
        jnp.sum(ez <= 0) / numSamples,
        ez,
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
    )


mc_dubins_pez = jax.jit(
    jax.vmap(
        mc_dubins_pez_single,
        in_axes=(0, 0, None, None, None, None, None, None, None, None),
    )
)


def generate_random_key():
    seed = np.random.randint(100000)
    key = jax.random.PRNGKey(seed)
    return jax.random.split(key)


def mc_dubins_PEZ(
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
    captureRadius,
):
    numSamples = 5000

    key, subkey = generate_random_key()
    # Generate heading samples
    pursuerHeadingSamples = pursuerHeading + jnp.sqrt(
        pursuerHeadingVar
    ) * jax.random.normal(subkey, shape=(numSamples,))

    key, subkey = generate_random_key()

    # generate turn radius samples
    minimumTurnRadiusSamples = minimumTurnRadius + jnp.sqrt(
        minimumTurnRadiusVar
    ) * jax.random.normal(subkey, shape=(numSamples,))

    key, subkey = generate_random_key()
    # generate speed samples
    pursuerSpeedSamples = pursuerSpeed + jnp.sqrt(pursuerSpeedVar) * jax.random.normal(
        subkey, shape=(numSamples,)
    )

    key, subkey = generate_random_key()

    # generate position samples
    pursuerPositionSamples = jax.random.multivariate_normal(
        key, pursuerPosition, pursuerPositionCov, shape=(numSamples,)
    )

    key, subkey = generate_random_key()

    pursuerRangeSamples = pursuerRange + jnp.sqrt(pursuerRangeVar) * jax.random.normal(
        subkey, shape=(numSamples,)
    )

    return mc_dubins_pez(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        pursuerPositionSamples,
        pursuerHeadingSamples,
        pursuerSpeedSamples,
        minimumTurnRadiusSamples,
        pursuerRangeSamples,
        captureRadius,
        numSamples,
    )


# find gradients/jacobians of dubins BEZ function
dDubinsEZ_dPursuerPosition = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 0)
dDubinsEZ_dPursuerHeading = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 1)
dDubinsEZ_dMinimumTurnRadius = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 2)
dDubinsEZ_dCaptureRadius = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 3)
dDubinsEZ_dPursuerRange = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 4)
dDubinsEZ_dPursuerSpeed = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 5)
dDubinsEZ_dEvaderPosition = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 6)

# second order derivatives
d2DubinsEZ_dPursuerPosition = jax.jacfwd(dDubinsEZ_dPursuerPosition, 0)
d2DubinsEZ_dPursuerHeading = jax.jacfwd(dDubinsEZ_dPursuerHeading, 1)
d2DubinsEZ_dMinimumTurnRadius = jax.jacfwd(dDubinsEZ_dMinimumTurnRadius, 2)
d2DubinsEZ_dCaptureRadius = jax.jacfwd(dDubinsEZ_dCaptureRadius, 3)
d2DubinsEZ_dPursuerRange = jax.jacfwd(dDubinsEZ_dPursuerRange, 4)
d2DubinsEZ_dPursuerSpeed = jax.jacfwd(dDubinsEZ_dPursuerSpeed, 5)
d2DubinsEZ_dEvaderPosition = jax.jacfwd(dDubinsEZ_dEvaderPosition, 6)


def linear_dubins_PEZ_single(
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
    captureRadius,
):
    dDubinsEZ_dPursuerPositionValue = dDubinsEZ_dPursuerPosition(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    # jax.debug.print(
    #     "dDubinsEZ_dPursuerPositionValue {x}", x=dDubinsEZ_dPursuerPositionValue
    # )
    dDubinsEZ_dPursuerHeadingValue = dDubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    # jax.debug.print(
    #     "dDubinsEZ_dPursuerHeadingValue {x}", x=dDubinsEZ_dPursuerHeadingValue
    # )
    dDubinsEZ_dMinimumTurnRadiusValue = dDubinsEZ_dMinimumTurnRadius(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    # jax.debug.print(
    #     "dDubinsEZ_dMinimumTurnRadiusValue {x}", x=dDubinsEZ_dMinimumTurnRadiusValue
    # )
    dDubinsEZ_dPursuerRangeValue = dDubinsEZ_dPursuerRange(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    # jax.debug.print("dDubinsEZ_dPursuerRangeValue {x}", x=dDubinsEZ_dPursuerRangeValue)
    dDubinsEZ_dPursuerSpeedValue = dDubinsEZ_dPursuerSpeed(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    # jax.debug.print("dDubinsEZ_dPursuerSpeedValue {x}", x=dDubinsEZ_dPursuerSpeedValue)
    var = (
        dDubinsEZ_dPursuerPositionValue
        @ pursuerPositionCov
        @ dDubinsEZ_dPursuerPositionValue.T
        + dDubinsEZ_dPursuerHeadingValue**2 * pursuerHeadingVar
        + dDubinsEZ_dMinimumTurnRadiusValue**2 * minimumTurnRadiusVar
        + dDubinsEZ_dPursuerRangeValue**2 * pursuerRangeVar
        + dDubinsEZ_dPursuerSpeedValue**2 * pursuerSpeedVar
    )
    # jax.debug.print("var {x}", x=var)
    mean = dubinsEZ.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )

    return (
        jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var)),
        mean,
        var,
    )


linear_dubins_pez = jax.jit(
    jax.vmap(
        linear_dubins_PEZ_single,
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
        ),
    )
)


def quadratic_dubins_PEZ_single(
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
    captureRadius,
):
    dDubinsEZ_dPursuerPositionValue = dDubinsEZ_dPursuerPosition(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    d2DubinsEZ_dPursuerPositionValue = d2DubinsEZ_dPursuerPosition(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    dDubinsEZ_dPursuerHeadingValue = dDubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    d2DubinsEZ_dPursuerHeadingValue = d2DubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    dDubinsEZ_dMinimumTurnRadiusValue = dDubinsEZ_dMinimumTurnRadius(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    d2DubinsEZ_dMinimumTurnRadiusValue = d2DubinsEZ_dMinimumTurnRadius(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )

    # jax.debug.print(
    #     "dDubinsEZ_dMinimumTurnRadiusValue {x}", x=dDubinsEZ_dMinimumTurnRadiusValue
    # )
    dDubinsEZ_dPursuerRangeValue = dDubinsEZ_dPursuerRange(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    d2DubinsEZ_dPursuerRangeValue = d2DubinsEZ_dPursuerRange(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    # jax.debug.print("dDubinsEZ_dPursuerRangeValue {x}", x=dDubinsEZ_dPursuerRangeValue)
    dDubinsEZ_dPursuerSpeedValue = dDubinsEZ_dPursuerSpeed(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    d2DubinsEZ_dPursuerSpeedValue = d2DubinsEZ_dPursuerSpeed(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    # jax.debug.print("dDubinsEZ_dPursuerSpeedValue {x}", x=dDubinsEZ_dPursuerSpeedValue)
    var = (
        dDubinsEZ_dPursuerPositionValue
        @ pursuerPositionCov
        @ dDubinsEZ_dPursuerPositionValue.T
        + dDubinsEZ_dPursuerHeadingValue**2 * pursuerHeadingVar
        + dDubinsEZ_dMinimumTurnRadiusValue**2 * minimumTurnRadiusVar
        + dDubinsEZ_dPursuerRangeValue**2 * pursuerRangeVar
        + dDubinsEZ_dPursuerSpeedValue**2 * pursuerSpeedVar
        + 0.25 * d2DubinsEZ_dPursuerHeadingValue**2 * pursuerHeadingVar**2
        + 0.25 * d2DubinsEZ_dMinimumTurnRadiusValue**2 * minimumTurnRadiusVar**2
        + 0.25 * d2DubinsEZ_dPursuerRangeValue**2 * pursuerRangeVar**2
        + 0.25 * d2DubinsEZ_dPursuerSpeedValue**2 * pursuerSpeedVar**2
    )
    # jax.debug.print("var {x}", x=var)
    mean = dubinsEZ.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )

    return (
        jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var)),
        mean,
        var,
    )


quadratic_dubins_pez = jax.jit(
    jax.vmap(
        quadratic_dubins_PEZ_single,
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
        ),
    )
)


def symmetric_sqrt(mat):
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = jnp.clip(eigvals, a_min=0.0)
    sqrt_eigvals = jnp.sqrt(eigvals)
    return eigvecs @ (sqrt_eigvals[..., None] * eigvecs.T)


@jax.jit
def generate_sigma_points(mean, cov, alpha=1e-3, beta=2.0, kappa=0.0):
    n = mean.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n

    # Compute square root of scaled covariance
    L = symmetric_sqrt((n + lambda_) * cov)

    # Compute sigma points
    offsets = L.T
    sigma_points = jnp.vstack([mean, mean + offsets, mean - offsets])

    # Compute weights
    weight = 1.0 / (2 * (n + lambda_))
    weights = jnp.full(2 * n, weight)
    weights_mean = jnp.concatenate([jnp.array([lambda_ / (n + lambda_)]), weights])
    weights_cov = jnp.concatenate(
        [jnp.array([lambda_ / (n + lambda_) + (1 - alpha**2 + beta)]), weights]
    )

    return sigma_points, weights_mean, weights_cov


def uncented_dubins_PEZ_single(
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
    captureRadius,
):
    # Construct covariance matrix
    cov = jax.scipy.linalg.block_diag(
        pursuerPositionCov,
        jnp.array([[pursuerHeadingVar]]),
        jnp.array([[pursuerSpeedVar]]),
        jnp.array([[minimumTurnRadiusVar]]),
        jnp.array([[pursuerRangeVar]]),
    )

    # Construct mean vector
    mean = jnp.array(
        [
            pursuerPosition[0],
            pursuerPosition[1],
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ]
    )

    # Generate sigma points
    sigma_points, weights_mean, weights_cov = generate_sigma_points(mean, cov)

    # Extract individual sigma point components
    sigmaPointsPursuerPosition = sigma_points[:, :2]
    sigmaPointsPursuerHeading = sigma_points[:, 2]
    sigmaPointsPursuerSpeed = sigma_points[:, 3]
    sigmaPointsMinimumTurnRadius = sigma_points[:, 4]
    sigmaPointsPursuerRange = sigma_points[:, 5]

    # Compute transformed sigma points
    transformedSigmaPoints = in_dubins_engagement_zone(
        sigmaPointsPursuerPosition,
        sigmaPointsPursuerHeading,
        sigmaPointsMinimumTurnRadius,
        captureRadius,
        sigmaPointsPursuerRange,
        sigmaPointsPursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )

    # Compute mean and variance
    ezMean = jnp.dot(weights_mean, transformedSigmaPoints)
    ezVar = jnp.dot(weights_cov, (transformedSigmaPoints - ezMean) ** 2)

    return jax.scipy.stats.norm.cdf(0, ezMean, jnp.sqrt(ezVar)), ezMean, ezVar


uncented_dubins_pez = jax.jit(
    jax.vmap(
        uncented_dubins_PEZ_single,
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
        ),
    )
)


def plot_dubins_PEZ(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
    ax,
    useLinear=False,
    useUnscented=False,
    useQuadratic=False,
    useMC=False,
):
    if useLinear:
        numPoints = 300
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        Z = jnp.zeros_like(X)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue, _, _ = linear_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Linear Dubins PEZ", fontsize=30)
    elif useUnscented:
        numPoints = 300
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        Z = jnp.zeros_like(X)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue, _, _ = uncented_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Unscented Dubins PEZ")
    elif useMC:
        numPoints = 100
        rangeX = 2.0
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        Z = jnp.zeros_like(X)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue, _, _, _, _, _, _ = mc_dubins_PEZ(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Monte Carlo Dubins PEZ", fontsize=30)
    elif useQuadratic:
        numPoints = 300
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        Z = jnp.zeros_like(X)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue, _, _ = quadratic_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )

    ZTrue = ZTrue.reshape(numPoints, numPoints)

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    c = ax.contour(
        X,
        Y,
        ZTrue,
        levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    # c = ax.pcolormesh(X, Y, ZTrue)
    # if useUnscented:
    ax.set_xlabel("X", fontsize=26)
    ax.set_ylabel("Y", fontsize=26)
    # set tick size
    ax.tick_params(axis="both", which="major", labelsize=20)
    plt.clabel(c, inline=True, fontsize=20)

    # ax.contour(X, Y, ZGeometric, cmap="summer")
    ax.scatter(*pursuerPosition, c="r")
    ax.set_aspect("equal", "box")
    ax.set_aspect("equal", "box")
    return c


def plot_EZ_vs_pursuer_range(
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    pursuerRange,
    minimumTurnRadius,
    captureRadius,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
):
    fig, ax = plt.subplots()
    # pursuerHeading = np.linspace(-np.pi, np.pi, 100)
    # turn radius derivat
    #
    dDubinsEZ_dRange = dDubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    print("dDubinsEZ_dPursuerRange", dDubinsEZ_dRange)
    ezMean = in_dubins_engagement_zone(
        pursuerPosition,
        jnp.array([pursuerHeading]),
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )

    pursuerHeadingVec = np.linspace(-np.pi, np.pi, 1000)
    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeadingVec,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    ax.scatter(pursuerHeadingVec, ez)
    # plot tangent point
    ax.plot(
        pursuerHeadingVec,
        dDubinsEZ_dRange * (pursuerHeadingVec - pursuerHeading) + ezMean,
    )


def plot_normal(mean, var, ax, label):
    x = np.linspace(mean - 3 * np.sqrt(var), mean + 3 * np.sqrt(var), 100)
    y = jax.scipy.stats.norm.pdf(x, mean, np.sqrt(var))
    ax.plot(x, y, label=label)


def compare_distribution(
    evaderPosition,
    evaderHeading,
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
    captureRadius,
):
    (
        inEZ,
        ez,
        pursuerPositionSamples,
        pursuerHeadingSamples,
        pursuerSpeedSamples,
        turnRadiusSamples,
        rangeSamples,
    ) = mc_dubins_PEZ(
        evaderPosition,
        evaderHeading,
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
        captureRadius,
    )
    print("mc", inEZ)
    inEZ, linMean, linVar, dDubinsEZ_dPursuerHeading = linear_dubins_pez(
        evaderPosition,
        evaderHeading,
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
        captureRadius,
    )
    print("lin ", inEZ)
    inEZ, uMean, uVar = uncented_dubins_pez(
        evaderPosition,
        evaderHeading,
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
        captureRadius,
    )
    print("u ", inEZ)
    fig, ax = plt.subplots()

    sortedEZindices = jnp.argsort(ez).flatten()
    ezSorted = ez.flatten()[sortedEZindices]
    cdf = jnp.linspace(0, len(ezSorted), len(ezSorted)) / len(ezSorted)

    plt.hist(ez, bins=1000, density=True)
    print("min turn radius", np.min(turnRadiusSamples))
    plot_normal(linMean, linVar, ax, "Linear")
    plot_normal(uMean, uVar, ax, "Unscented")
    # plot vertical line at 0
    ax.axvline(0, color="k", linestyle="dashed", linewidth=1)
    ax.plot(ezSorted, cdf, label="Monte Carlo")
    # plot linear and unscented cdfs
    x = np.linspace(jnp.min(ez), np.max(ez), 100)
    y = jax.scipy.stats.norm.cdf(x, linMean, np.sqrt(linVar))
    ax.plot(x, y, label="Linear")
    y = jax.scipy.stats.norm.cdf(x, uMean, np.sqrt(uVar))
    ax.plot(x, y, label="Unscented")

    plt.legend()

    print("dDubinsEZ_dPursuerHeading", dDubinsEZ_dPursuerHeading)
    fig1, ax1 = plt.subplots()
    ax1.scatter(pursuerHeadingSamples, ez, label="Monte Carlo")
    # plot mean value
    ax1.plot(
        pursuerHeading * np.ones(100),
        np.linspace(jnp.min(ez), jnp.max(ez), 100),
        label="Mean",
    )
    linApprox = (
        dDubinsEZ_dPursuerHeading * (pursuerHeadingSamples - pursuerHeading) + linMean
    ).flatten()
    print("diff", pursuerHeadingSamples - pursuerHeading)
    print("linApprox", linApprox)
    ax1.plot(
        pursuerHeadingSamples.flatten(),
        linApprox,
        label="Tangent",
    )


def comparge_PEZ(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
):
    evaderHeading = jnp.array([(0 / 20) * np.pi])
    fig, axes = plt.subplots(2, 2)
    plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[0][0],
        useLinear=True,
        useUnscented=False,
        useQuadratic=False,
        useMC=False,
    )
    plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[0][1],
        useLinear=False,
        useUnscented=False,
        useQuadratic=False,
        useMC=True,
    )
    # fast_pursuer.plotMahalanobisDistance(
    #     pursuerPosition, pursuerPositionCov, axes[0], fig, plotColorbar=False
    # )
    # fast_pursuer.plotMahalanobisDistance(
    #     pursuerPosition, pursuerPositionCov, axes[1], fig, plotColorbar=True
    # )
    # dubinsEZ.plot_dubins_EZ(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     minimumTurnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[0],
    # )
    # dubinsEZ.plot_dubins_EZ(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     minimumTurnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[1],
    # )
    c = plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[1][0],
        useLinear=False,
        useUnscented=True,
        useQuadratic=False,
        useMC=False,
    )
    c = plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[1][1],
        useLinear=False,
        useUnscented=False,
        useQuadratic=True,
        useMC=False,
    )
    # fig = plt.gcf()
    # fig.colorbar(c)


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
    pursuerPositionCov = np.array([[0.000000000001, 0.0], [0.0, 0.00000000001]])

    pursuerHeading = (4.0 / 4.0) * np.pi
    pursuerHeadingVar = 0.3

    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.0

    pursuerRange = 1.0
    pursuerRangeVar = 0.2

    minimumTurnRadius = 0.2
    minimumTurnRadiusVar = 0.00

    captureRadius = 0.0

    evaderHeading = jnp.array([(0 / 20) * np.pi, (0 / 20) * np.pi])
    evaderHeading = jnp.array([(0.0 / 20.0) * np.pi])
    # evaderHeading = jnp.array((0.0 / 20.0) * np.pi)

    evaderSpeed = 0.5
    evaderPosition = np.array([[0.1, 0.0]])
    # evaderPosition = np.array([-0.4, 0.0])
    # evaderPosition = np.array([-0.28, -0.42])

    # plot_EZ_vs_pursuer_range(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     pursuerRange,
    #     minimumTurnRadius,
    #     captureRadius,
    #     evaderPosition,
    #     evaderHeading,
    #     evaderSpeed,
    # )
    # compare_distribution(
    #     evaderPosition,
    #     evaderHeading,
    #     evaderSpeed,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     captureRadius,
    # )
    comparge_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
    )

    plt.show()


if __name__ == "__main__":
    main()
