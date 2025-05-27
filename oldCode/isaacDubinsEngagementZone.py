import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

import findDubinsLength


def constant_turn_basic_engagement_zone(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    pursuerRange,
    minimumTurnRadius,
):
    # This function calculates the engagement zone for a pursuer with a constant turn rate, with limited turn radius, https://arxiv.org/pdf/2502.00364 equation 9-11
    speedRatio = evaderSpeed / pursuerSpeed
    pursuerPositionTranslated = pursuerPosition + jnp.array(
        [
            -speedRatio * pursuerRange * jnp.cos(evaderHeading),
            -speedRatio * pursuerRange * jnp.sin(evaderHeading),
        ]
    )
    distance = jnp.linalg.norm(evaderPosition - pursuerPositionTranslated)
    polarAngle = jnp.arctan2(
        evaderPosition[1] - pursuerPositionTranslated[1],
        evaderPosition[0] - pursuerPositionTranslated[0],
    )
    ezCheck = (pursuerRange / (polarAngle - pursuerHeading)) * jnp.sin(
        polarAngle - pursuerHeading
    ) - distance
    return ezCheck < 0


constant_turn_basic_engagement_zone_vec = jax.vmap(
    constant_turn_basic_engagement_zone,
    in_axes=(0, 0, None, None, None, None, None, None),
)


def plot_engagement_zone(
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    pursuerRange,
    minimumTurnRadius,
):
    numPoints = 1000
    x = jnp.linspace(-2, 2, numPoints)
    y = jnp.linspace(-2, 2, numPoints)
    X, Y = jnp.meshgrid(x, y)
    evaderHeadings = np.ones(numPoints**2) * evaderHeading

    Z = constant_turn_basic_engagement_zone_vec(
        np.array([X.flatten(), Y.flatten()]).T,
        evaderHeadings,
        evaderSpeed,
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        pursuerRange,
        minimumTurnRadius,
    )
    Z = Z.reshape(X.shape)
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z)
    ax.set_aspect("equal")
    findDubinsLength.plot_turn_radius_circles(
        pursuerPosition, pursuerHeading, minimumTurnRadius, ax
    )
    plt.show()


def main():
    pursuerPosition = np.array([0, 0])

    pursuerHeading = (4 / 4) * np.pi
    pursuerSpeed = 2

    pursuerRange = 1
    minimumTurnRadius = (2 / (3 * np.pi)) * pursuerRange
    captureRadius = 0.0
    evaderHeading = (0 / 20) * np.pi
    evaderSpeed = 1
    evaderPosition = np.array([-0.3, 0.3])
    plot_engagement_zone(
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        pursuerRange,
        minimumTurnRadius,
    )


if __name__ == "__main__":
    main()
