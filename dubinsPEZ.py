import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from functools import partial


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
#             None,  # pursuerHeading
#             None,  # minimumTurnRadius
#             None,  # captureRadius
#             0,  # pursuerRange
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
    return jnp.sum(ez <= 0) / numSamples


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
    numSamples = 2000

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

    return jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var))


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
    fig,
    useLinear=True,
):
    numPoints = 300
    rangeX = 1.5
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading

    if useLinear:
        ZTrue = linear_dubins_pez(
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
        fig.suptitle("Linear Dubins PEZ")
    else:
        ZTrue = mc_dubins_PEZ(
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
        fig.suptitle("Monte Carlo Dubins PEZ")

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
    fig.colorbar(c, ax=ax)
    # ax.contour(X, Y, ZGeometric, cmap="summer")
    ax.scatter(*pursuerPosition, c="r")
    ax.set_aspect("equal", "box")
    ax.set_aspect("equal", "box")
    return ax


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
    dDubinsEZ_dPursuerRangeVal = dDubinsEZ_dPursuerRange(
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
    print("dDubinsEZ_dPursuerRange", dDubinsEZ_dPursuerRangeVal)
    test = dDubinsEZ_dEvaderPosition(
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
    evaderHeadingVec = jnp.array([np.cos(evaderHeading), np.sin(evaderHeading)])
    test = jnp.dot(test, evaderHeadingVec)
    print(test)
    pursuerRange = np.linspace(0.5, 1.5, 100)
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
    ax.scatter(pursuerRange, ez)


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCov = np.eye(2) * 0.0000001

    pursuerHeading = (2.0 / 4.0) * np.pi
    pursuerHeadingVar = 0.0

    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.0

    pursuerRange = 1.0
    pursuerRangeVar = 0.3

    minimumTurnRadius = 0.2
    minimumTurnRadiusVar = 0.0

    captureRadius = 0.0

    evaderHeading = jnp.array([(0 / 20) * np.pi, (0 / 20) * np.pi])
    evaderHeading = jnp.array((0.0 / 20.0) * np.pi)
    evaderSpeed = 0.5
    evaderPosition = np.array([[-0.30, -0.5], [-0.20, -0.5]])
    evaderPosition = np.array([-0.20, -0.6])

    # evaderPosition = np.array([-0.5, 0.5])

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
    # inEZ = mc_dubins_PEZ(
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
    # print(inEZ)
    # inEZ = linear_dubins_pez(
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
    # print(inEZ)
    # fig, ax = plt.subplots()
    # plt.hist(ez[0].flatten())

    # evaderHeading = jnp.array([(0 / 20) * np.pi])
    fig, ax = plt.subplots()
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
        ax,
        fig,
        useLinear=True,
    )
    fig2, ax2 = plt.subplots()
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
        ax2,
        fig2,
        useLinear=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
