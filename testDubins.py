import numpy as np
import time
import matplotlib.pyplot as plt
from fast_pursuer import plotEngagementZone, inEngagementZone, inEngagementZoneJax
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


#
# def find_counter_clockwise_tangent_point(p1, c, r):
#     v1 = p1 - c
#     normV1 = np.linalg.norm(v1)
#     v3Perallel = (r**2) / normV1**2 * v1
#     vPerpendicularNormalized = np.array([-v1[1], v1[0]]) / normV1
#     v3Perpendicular = -np.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized
#
#     v3 = v3Perallel + v3Perpendicular
#     pt = c + v3
#     return pt


@jax.jit
def find_counter_clockwise_tangent_point(p1, c, r):
    v1 = p1 - c
    normV1 = jnp.linalg.norm(v1)
    v3Perallel = (r**2) / normV1**2 * v1
    vPerpendicularNormalized = jnp.array([-v1[1], v1[0]]) / normV1
    v3Perpendicular = -jnp.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized
    v3 = v3Perallel + v3Perpendicular
    pt = c + v3
    return pt


# def find_clockwise_tangent_point(p1, c, r):
#     v1 = p1 - c
#     normV1 = np.linalg.norm(v1)
#     v3Perallel = (r**2) / normV1**2 * v1
#     vPerpendicularNormalized = -np.array([v1[1], -v1[0]]) / normV1
#     v3Perpendicular = np.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized
#
#     v3 = v3Perallel + v3Perpendicular
#     pt = c + v3
#     return pt
#


@jax.jit
def find_clockwise_tangent_point(p1, c, r):
    v1 = p1 - c
    normV1 = jnp.linalg.norm(v1)
    v3Perallel = (r**2) / normV1**2 * v1
    vPerpendicularNormalized = -jnp.array([v1[1], -v1[0]]) / normV1
    v3Perpendicular = jnp.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized

    v3 = v3Perallel + v3Perpendicular
    pt = c + v3
    return pt


# def clockwise_angle(v1, v2):
#     # Calculate determinant and dot product
#     det = v1[0] * v2[1] - v1[1] * v2[0]
#     dot = v1[0] * v2[0] + v1[1] * v2[1]
#
#     # Compute angle and normalize to [0, 2*pi]
#     angle = np.arctan2(det, dot)
#     angle_ccw = (angle + 2 * np.pi) % (2 * np.pi)
#
#     return angle_ccw


@jax.jit
def clockwise_angle(v1, v2):
    # Calculate determinant and dot product
    det = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]

    # Compute angle and normalize to [0, 2*pi]
    angle = jnp.arctan2(det, dot)
    angle_ccw = (angle + 2 * jnp.pi) % (2 * jnp.pi)

    return angle_ccw


#
# def counterclockwise_angle(v1, v2):
#     # Calculate determinant and dot product
#     det = v1[0] * v2[1] - v1[1] * v2[0]
#     dot = v1[0] * v2[0] + v1[1] * v2[1]
#
#     # Compute clockwise angle
#     angle = np.arctan2(-det, dot)
#     angle_cw = (angle + 2 * np.pi) % (2 * np.pi)
#
#     return angle_cw


@jax.jit
def counterclockwise_angle(v1, v2):
    # Calculate determinant and dot product
    det = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]

    # Compute clockwise angle
    angle = jnp.arctan2(-det, dot)
    angle_cw = (angle + 2 * jnp.pi) % (2 * jnp.pi)

    return angle_cw


@jax.jit
def find_dubins_path_length(startPosition, startHeading, goalPosition, radius):
    # Compute left and right turn centers
    leftCenter = jnp.array(
        [
            startPosition[0] - radius * jnp.sin(startHeading),
            startPosition[1] + radius * jnp.cos(startHeading),
        ]
    )
    rightCenter = jnp.array(
        [
            startPosition[0] + radius * jnp.sin(startHeading),
            startPosition[1] - radius * jnp.cos(startHeading),
        ]
    )

    # Compute distances to goal
    dist_to_left = jnp.linalg.norm(goalPosition - leftCenter)
    dist_to_right = jnp.linalg.norm(goalPosition - rightCenter)

    # Determine which center to use
    # left_closer = dist_to_left < dist_to_right
    right_closer = dist_to_right < dist_to_left
    inside_right_circle = dist_to_right < radius
    inside_left_circle = dist_to_left < radius
    clockwise = jnp.where(
        jnp.logical_or(
            jnp.logical_and(right_closer, jnp.logical_not(inside_right_circle)),
            jnp.logical_and(jnp.logical_not(right_closer), inside_left_circle),
        ),
        True,
        False,
    )
    centerPoint = jnp.where(clockwise, rightCenter, leftCenter)
    # Compute tangent point
    tangentPoint = jax.lax.cond(
        clockwise,
        lambda _: find_clockwise_tangent_point(goalPosition, centerPoint, radius),
        lambda _: find_counter_clockwise_tangent_point(
            goalPosition, centerPoint, radius
        ),
        operand=None,
    )

    # Compute angles for arc length
    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint

    theta = jax.lax.cond(
        clockwise,
        lambda _: clockwise_angle(v3, v4),
        lambda _: counterclockwise_angle(v3, v4),
        operand=None,
    )

    # Compute final path length
    straightLineLength = jnp.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength


# def find_dubins_path_length(startPosition, startHeading, goalPosition, radius):
#     leftCenter = np.array(
#         [
#             startPosition[0] - radius * np.sin(startHeading),
#             startPosition[1] + radius * np.cos(startHeading),
#         ]
#     )
#     rightCenter = np.array(
#         [
#             startPosition[0] + radius * np.sin(startHeading),
#             startPosition[1] - radius * np.cos(startHeading),
#         ]
#     )
#
#     clockwise = False
#     if np.linalg.norm(goalPosition - leftCenter) < np.linalg.norm(
#         goalPosition - rightCenter
#     ):
#         centerPoint = leftCenter
#         clockwise = False
#         if np.linalg.norm(goalPosition - leftCenter) < radius:
#             clockwise = True
#             centerPoint = rightCenter
#
#     else:
#         centerPoint = rightCenter
#         clockwise = True
#         if np.linalg.norm(goalPosition - rightCenter) < radius:
#             clockwise = False
#             centerPoint = leftCenter
#
#     if clockwise:
#         tangentPoint = find_clockwise_tangent_point(goalPosition, centerPoint, radius)
#     else:
#         tangentPoint = find_counter_clockwise_tangent_point(
#             goalPosition, centerPoint, radius
#         )
#
#     v4 = startPosition - centerPoint
#     v3 = tangentPoint - centerPoint
#     if clockwise:
#         theta = clockwise_angle(v3, v4)
#     else:
#         theta = counterclockwise_angle(v3, v4)
#
#     straitLineLength = np.linalg.norm(goalPosition - tangentPoint)
#     arcLength = radius * np.abs(theta)
#
#     totalLength = arcLength + straitLineLength
#
#     showPlot = False
#
#     if showPlot:
#         plt.figure()
#         plt.scatter(*startPosition, c="g")
#         plt.scatter(*goalPosition, c="r")
#         plt.scatter(*leftCenter, c="g")
#         plt.scatter(*rightCenter, c="b")
#         plt.plot(*v3, "b")
#         plt.plot(*v4, "b")
#         theta = np.linspace(0, 2 * np.pi, 100)
#         xl = leftCenter[0] + radius * np.cos(theta)
#         yl = leftCenter[1] + radius * np.sin(theta)
#
#         xr = rightCenter[0] + radius * np.cos(theta)
#         yr = rightCenter[1] + radius * np.sin(theta)
#
#         plt.plot(xl, yl, "b")
#         plt.plot(xr, yr, "b")
#         plt.scatter(*tangentPoint, c="y")
#         plt.plot(
#             [goalPosition[0], tangentPoint[0]], [goalPosition[1], tangentPoint[1]], "y"
#         )
#         ax = plt.gca()
#         ax.set_aspect("equal", "box")
#     return totalLength


# def in_dubins_engagement_zone(
#     startPosition,
#     startHeading,
#     turnRadius,
#     captureRadius,
#     pursuerRange,
#     pursuerSpeed,
#     evaderPosition,
#     evaderHeading,
#     evaderSpeed,
# ):
#     speedRatio = evaderSpeed / pursuerSpeed
#     goalPosition = evaderPosition + speedRatio * pursuerRange * np.array(
#         [np.cos(evaderHeading), np.sin(evaderHeading)]
#     )
#
#     dubinsPathLength = find_dubins_path_length(
#         startPosition, startHeading, goalPosition, turnRadius
#     )
#
#     return dubinsPathLength < (captureRadius + pursuerRange)


@jax.jit
def in_dubins_engagement_zone_single(
    startPosition,
    startHeading,
    turnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
):
    speedRatio = evaderSpeed / pursuerSpeed
    goalPosition = evaderPosition + speedRatio * pursuerRange * jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )

    dubinsPathLength = find_dubins_path_length(
        startPosition, startHeading, goalPosition, turnRadius
    )

    return dubinsPathLength < (captureRadius + pursuerRange)


# Vectorized function using vmap
in_dubins_engagement_zone = jax.jit(
    jax.vmap(
        in_dubins_engagement_zone_single,
        in_axes=(
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            None,
        ),  # Vectorizing over evaderPosition & evaderHeading
    )
)


def collision_region(
    startPosition,
    startHeading,
    turnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    evaderSpeed,
    evaderPosition,
):
    dubinsPathLength = find_dubins_path_length(
        startPosition, startHeading, evaderPosition, turnRadius
    )
    return dubinsPathLength < (captureRadius + pursuerRange)


def plot_dubins_engagement_zone(
    startPosition,
    startHeading,
    turnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    evaderSpeed,
    evaderHeading,
):
    numPoints = 500
    x = np.linspace(-2, 2, numPoints)
    y = np.linspace(-2, 2, numPoints)
    [X, Y] = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    # collisionRegion = np.zeros(X.shape)
    #
    headingsVec = np.ones(X.shape) * evaderHeading

    start = time.time()
    Z = in_dubins_engagement_zone(
        startPosition,
        startHeading,
        turnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        np.array([X, Y]).T,
        headingsVec,
        evaderSpeed,
    )

    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         Z[i, j] = in_dubins_engagement_zone(
    #             startPosition,
    #             startHeading,
    #             turnRadius,
    #             captureRadius,
    #             pursuerRange,
    #             pursuerSpeed,
    #             np.array([X[i, j], Y[i, j]]),
    #             evaderHeading,
    #             evaderSpeed,
    #         )
    # collisionRegion[i, j] = collision_region(
    #     startPosition,
    #     startHeading,
    #     turnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerSpeed,
    #     evaderSpeed,
    #     np.array([X[i, j], Y[i, j]]),
    # )
    print("Time: ", time.time() - start)

    plt.figure()
    c = plt.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        Z.reshape((numPoints, numPoints)),
        levels=[0],
    )
    ax = plt.gca()
    ax.set_aspect("equal", "box")
    plt.scatter(*startPosition, c="g")
    theta = np.linspace(0, 2 * np.pi, 100)
    leftCenter = np.array(
        [
            startPosition[0] - turnRadius * np.sin(startHeading),
            startPosition[1] + turnRadius * np.cos(startHeading),
        ]
    )
    rightCenter = np.array(
        [
            startPosition[0] + turnRadius * np.sin(startHeading),
            startPosition[1] - turnRadius * np.cos(startHeading),
        ]
    )
    leftX = leftCenter[0] + turnRadius * np.cos(theta)
    leftY = leftCenter[1] + turnRadius * np.sin(theta)
    rightX = rightCenter[0] + turnRadius * np.cos(theta)
    rightY = rightCenter[1] + turnRadius * np.sin(theta)
    plt.plot(leftX, leftY, "b")
    plt.plot(rightX, rightY, "b")
    # plt.contour(X, Y, collisionRegion, levels=[0])
    return ax


def main():
    startPosition = np.array([0, 0])
    startHeading = -np.pi / 2
    turnRadius = 0.5
    captureRadius = 0.1
    pursuerRange = 1.0
    pursuerSpeed = 2
    evaderSpeed = 1
    agentHeading = np.pi / 2

    # agentPosition = np.array([0.0, -0.1])
    #
    # length = find_dubins_path_length(
    #     startPosition, startHeading, agentPosition, turnRadius
    # )
    # print("Length: ", length)

    ax = plot_dubins_engagement_zone(
        startPosition,
        startHeading,
        turnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderSpeed,
        agentHeading,
    )

    plotEngagementZone(
        agentHeading,
        startPosition,
        pursuerRange,
        captureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
    )
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
