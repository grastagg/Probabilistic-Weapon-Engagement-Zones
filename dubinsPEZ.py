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
            None,
            0,
            0,
            None,
            None,
            0,
            None,
            None,
            None,
        ),  # Vectorizing over evaderPosition & evaderHeading
    )
)


@jax.jit
def mc_dubins_PEZ_heading_only_single(
    key,
    pursuerPosition,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    captureRadius,
):
    key, subkey = jax.random.split(key)
    numSamples = 2000

    # Generate heading samples
    pursuerHeadingSamples = pursuerHeading + jnp.sqrt(
        pursuerHeadingVar
    ) * jax.random.normal(subkey, shape=(numSamples,))
    minimumTurnRadiusSamples = minimumTurnRadius + jnp.sqrt(
        minimumTurnRadiusVar
    ) * jax.random.normal(subkey, shape=(numSamples,))
    pursuerSpeedSamples = pursuerSpeed + jnp.sqrt(pursuerSpeedVar) * jax.random.normal(
        subkey, shape=(numSamples,)
    )

    # Compute engevaderagement zone for each sampled heading
    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeadingSamples,
        minimumTurnRadiusSamples,
        captureRadius,
        pursuerRange,
        pursuerSpeedSamples,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )

    return jnp.sum(ez <= 0) / numSamples  # Return updated key


# Vectorized over multiple evaders
mc_dubins_PEZ_heading_only = jax.jit(
    jax.vmap(
        mc_dubins_PEZ_heading_only_single,
        in_axes=(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            None,
            None,
        ),
    )
)


def plot_dubins_PEZ(
    pursuerPosition,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    evaderHeading,
    evaderSpeed,
    ax,
    fig,
):
    numPoints = 200
    rangeX = 1.5
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading

    key = jax.random.PRNGKey(0)
    ZTrue = mc_dubins_PEZ_heading_only(
        key,  # Add a JAX PRNG key as input
        pursuerPosition,
        pursuerHeading,
        pursuerHeadindgVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        jnp.array([X, Y]).T,
        evaderHeadings,
        evaderSpeed,
        captureRadius,
    )

    ZTrue = ZTrue.reshape(numPoints, numPoints)

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    c = ax.contour(
        X,
        Y,
        ZTrue,
        levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
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
    turnradii = np.linspace(0.01, 2, 500)
    pursuerHeading = np.linspace(0, 2 * np.pi, 500)
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
    ax.scatter(pursuerHeading, ez)


def main():
    pursuerPosition = np.array([0, 0])

    pursuerHeading = (2 / 4) * np.pi
    pursuerHeadingVar = 0.2

    pursuerSpeed = 2
    pursuerSpeedVar = 0.0

    pursuerRange = 1

    minimumTurnRadius = 0.2
    minimumTurnRadiusVar = 0.0

    captureRadius = 0.0
    evaderHeading = (0 / 20) * np.pi
    evaderSpeed = 0.5
    evaderPosition = np.array([-3, 0])

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
    fig, ax = plt.subplots()
    plot_dubins_PEZ(
        pursuerPosition,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        evaderHeading,
        evaderSpeed,
        ax,
        fig,
    )
    plt.show()


if __name__ == "__main__":
    main()
