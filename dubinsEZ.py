from jax.lax import random_gamma_grad
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import patches

# from fast_pursuer import plotEngagementZone, inEngagementZone, inEngagementZoneJax
import jax
import jax.numpy as jnp


# import dubinsReachable
# import testDubins


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

print(jax.devices())
print(jax.default_backend())


def plot_turn_radius_circles(startPosition, startHeading, turnRadius, ax):
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
    ax.plot(leftX, leftY, "b")
    ax.plot(rightX, rightY, "b")


@jax.jit
def find_counter_clockwise_tangent_point(p1, c, r):
    v1 = p1 - c
    normV1 = jnp.linalg.norm(v1)
    v3Perallel = (r**2) / normV1**2 * v1
    vPerpendicularNormalized = jnp.array([v1[1], -v1[0]]) / normV1
    v3Perpendicular = jnp.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized
    v3 = v3Perallel + v3Perpendicular
    pt = c + v3
    return pt


@jax.jit
def find_counter_clockwise_tangent_point_smooth(
    p1: jnp.ndarray, c: jnp.ndarray, r: float
) -> jnp.ndarray:
    """
    Smooth counterclockwise tangent point computation.
    Uses an internal smooth ReLU (C1) to clamp the discriminant.
    Matches the variable names of the original version.
    """
    gamma = 50.0  # smoothing sharpness

    # Original variables
    v1 = p1 - c
    normV1 = jnp.linalg.norm(v1)

    # Parallel component
    v3Perallel = (r**2) / (normV1**2) * v1

    # Perpendicular direction (90° clockwise rotation)
    vPerpendicularNormalized = jnp.array([v1[1], -v1[0]]) / normV1

    # Smooth discriminant: Delta = r^2 - r^4 / normV1^2
    Delta = r**2 - (r**4) / (normV1**2)
    # smooth ReLU: approx max(Delta,0)
    Delta_pos = jnp.logaddexp(Delta, 0.0) / gamma

    # Perpendicular component magnitude
    v3Perpendicular = jnp.sqrt(Delta_pos) * vPerpendicularNormalized

    # Combined tangent vector
    v3 = v3Perallel + v3Perpendicular

    # Tangent point
    pt = c + v3
    return pt


@jax.jit
def find_clockwise_tangent_point(p1, c, r):
    v1 = p1 - c
    normV1 = jnp.linalg.norm(v1)
    v3Perallel = (r**2) / normV1**2 * v1
    vPerpendicularNormalized = jnp.array([-v1[1], v1[0]]) / normV1
    v3Perpendicular = jnp.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized

    v3 = v3Perallel + v3Perpendicular
    pt = c + v3
    return pt


@jax.jit
def find_clockwise_tangent_point_smooth(
    p1: jnp.ndarray, c: jnp.ndarray, r: float, gamma: float = 50.0
) -> jnp.ndarray:
    """
    Smooth clockwise tangent point computation.
    Uses an internal smooth ReLU (C1) to clamp the discriminant,
    matching the variable names of the original function.
    """
    # Vector from circle center to point
    v1 = p1 - c
    normV1 = jnp.linalg.norm(v1)

    # Parallel component
    v3Perallel = (r**2) / (normV1**2) * v1

    # Perpendicular direction (90° counterclockwise rotation)
    vPerpendicularNormalized = jnp.array([-v1[1], v1[0]]) / normV1

    # Smooth discriminant: Delta = r^2 - r^4 / normV1^2
    Delta = r**2 - (r**4) / (normV1**2)
    # C1‐smooth clamp to zero
    Delta_pos = jnp.logaddexp(Delta, 0.0) / gamma

    # Perpendicular component magnitude
    v3Perpendicular = jnp.sqrt(Delta_pos) * vPerpendicularNormalized

    # Combined tangent vector
    v3 = v3Perallel + v3Perpendicular

    # Tangent point
    pt = c + v3
    return pt


@jax.jit
def counterclockwise_angle(v1, v2):
    # Calculate determinant and dot product
    det = v1[0] * v2[1] - v1[1] * v2[0]  # 2D cross product (determinant)
    dot = v1[0] * v2[0] + v1[1] * v2[1]  # Dot product

    angle = jax.lax.cond(
        jnp.abs(det) < 1e-8,  # Condition: if det is near zero (collinear)
        lambda _: jax.lax.cond(
            dot < 0,  # Further check if the vectors are in opposite directions
            lambda _: jnp.pi,  # Opposite directions: angle = pi
            lambda _: 0.0,  # Same direction: angle = 0
            operand=None,
        ),
        lambda _: jnp.arctan2(det, dot),  # Standard angle calculation
        operand=None,
    )

    # Normalize to [0, 2*pi] to ensure the result is in a consistent range
    angle_cw = (angle + 2 * jnp.pi) % (2 * jnp.pi)

    return angle_cw


@jax.jit
def clockwise_angle(v1, v2):
    # Calculate determinant and dot product
    det = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]

    # Use jax.lax.cond for the collinearity check and handling the angle
    angle = jax.lax.cond(
        jnp.abs(det) < 1e-8,  # Condition: if det is near zero (collinear)
        lambda _: jax.lax.cond(
            dot < 0,  # Further check if the vectors are in opposite directions
            lambda _: jnp.pi,  # Opposite directions: angle = pi
            lambda _: 0.0,  # Same direction: angle = 0
            operand=None,
        ),
        lambda _: jnp.arctan2(-det, dot),  # Standard angle calculation
        operand=None,
    )

    # Ensure the angle is in the range [0, 2*pi)
    angle_cw = (angle + 2 * jnp.pi) % (2 * jnp.pi)

    return angle_cw


@jax.jit
def find_dubins_path_length_right_strait(
    startPosition, startHeading, goalPosition, radius
):
    centerPoint = jnp.array(
        [
            startPosition[0] + radius * jnp.sin(startHeading),
            startPosition[1] - radius * jnp.cos(startHeading),
        ]
    )
    tangentPoint = find_clockwise_tangent_point(goalPosition, centerPoint, radius)

    # Compute angles for arc length
    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint

    theta = clockwise_angle(v4, v3)

    # Compute final path length
    straightLineLength = jnp.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength, tangentPoint


@jax.jit
def find_dubins_path_length_right_strait_smooth(
    startPosition, startHeading, goalPosition, radius
):
    centerPoint = jnp.array(
        [
            startPosition[0] + radius * jnp.sin(startHeading),
            startPosition[1] - radius * jnp.cos(startHeading),
        ]
    )
    tangentPoint = find_clockwise_tangent_point_smooth(
        goalPosition, centerPoint, radius
    )

    # Compute angles for arc length
    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint

    theta = clockwise_angle(v4, v3)

    # Compute final path length
    straightLineLength = jnp.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength, tangentPoint


find_dubins_path_length_right_strait_vec = jax.vmap(
    find_dubins_path_length_right_strait, in_axes=(None, None, 0, None)
)


@jax.jit
def find_dubins_path_length_left_strait(
    startPosition, startHeading, goalPosition, radius
):
    centerPoint = jnp.array(
        [
            startPosition[0] - radius * jnp.sin(startHeading),
            startPosition[1] + radius * jnp.cos(startHeading),
        ]
    )
    tangentPoint = find_counter_clockwise_tangent_point(
        goalPosition, centerPoint, radius
    )

    # Compute angles for arc length
    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint

    theta = counterclockwise_angle(v4, v3)

    # Compute final path length
    straightLineLength = jnp.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength, tangentPoint


@jax.jit
def find_dubins_path_length_left_strait_smooth(
    startPosition, startHeading, goalPosition, radius
):
    centerPoint = jnp.array(
        [
            startPosition[0] - radius * jnp.sin(startHeading),
            startPosition[1] + radius * jnp.cos(startHeading),
        ]
    )
    tangentPoint = find_counter_clockwise_tangent_point_smooth(
        goalPosition, centerPoint, radius
    )

    # Compute angles for arc length
    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint

    theta = counterclockwise_angle(v4, v3)

    # Compute final path length
    straightLineLength = jnp.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength, tangentPoint


find_dubins_path_length_left_strait_vec = jax.vmap(
    find_dubins_path_length_left_strait, in_axes=(None, None, 0, None)
)


@jax.jit
def soft_min(a: jnp.ndarray, b: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Smooth approximation to min(a, b) that handles NaNs:
      - If a is NaN and b is not, returns b
      - If b is NaN and a is not, returns a
      - If both are NaN, returns NaN
    Uses soft‑min:  -alpha * log(exp(-a/alpha) + exp(-b/alpha))
    """
    # Standard soft-min
    smin = -alpha * jnp.log(jnp.exp(-a / alpha) + jnp.exp(-b / alpha))
    # Identify NaNs
    nan_a = jnp.isnan(a)
    nan_b = jnp.isnan(b)
    # Where one input is NaN, take the other
    res = jnp.where(nan_a & ~nan_b, b, smin)
    res = jnp.where(nan_b & ~nan_a, a, res)
    # If both are NaN, result should be NaN
    res = jnp.where(nan_a & nan_b, jnp.nan, res)
    return res


@jax.jit
def find_shortest_dubins_path(pursuerPosition, pursuerHeading, goalPosition, radius):
    straitLeftLength, _ = find_dubins_path_length_right_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    straitRightLength, _ = find_dubins_path_length_left_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )

    lengths = jnp.array([straitLeftLength, straitRightLength])

    return jnp.nanmin(lengths)


@jax.jit
def find_shortest_dubins_path_soft_min(
    pursuerPosition, pursuerHeading, goalPosition, radius
):
    straitLeftLength, _ = find_dubins_path_length_right_strait_smooth(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    straitRightLength, _ = find_dubins_path_length_left_strait_smooth(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )

    lengths = jnp.array([straitLeftLength, straitRightLength])

    return soft_min(*lengths, 0.01)


# Vectorized version over goalPosition
vectorized_find_shortest_dubins_path = jax.vmap(
    find_shortest_dubins_path,
    in_axes=(None, None, 0, None),  # Vectorize over goalPosition (3rd argument)
)


@jax.jit
def in_dubins_engagement_zone_soft_min_single(
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

    # Compute goal positions
    direction = jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )  # Heading unit vector
    goalPositions = evaderPosition + speedRatio * pursuerRange * direction
    dubinsPathLengths = find_shortest_dubins_path_soft_min(
        startPosition, startHeading, goalPositions, turnRadius
    )

    ez = dubinsPathLengths - (captureRadius + pursuerRange)
    return ez


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

    # Compute goal positions
    direction = jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )  # Heading unit vector
    goalPositions = evaderPosition + speedRatio * pursuerRange * direction
    dubinsPathLengths = find_shortest_dubins_path(
        startPosition, startHeading, goalPositions, turnRadius
    )

    ez = dubinsPathLengths - (captureRadius + pursuerRange)
    return ez


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


@jax.jit
def in_dubins_engagement_zone_right_single(
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

    # Compute goal positions
    direction = jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )  # Heading unit vector
    goalPosition = evaderPosition + speedRatio * pursuerRange * direction
    straitRightLength, _ = find_dubins_path_length_right_strait(
        startPosition, startHeading, goalPosition, turnRadius
    )
    straitRightLength = jnp.where(
        jnp.isnan(straitRightLength), jnp.inf, straitRightLength
    )

    ez = straitRightLength - (captureRadius + pursuerRange)
    # return dubinsPathLengths
    return ez


@jax.jit
def in_dubins_engagement_zone_left_single(
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

    # Compute goal positions
    direction = jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )  # Heading unit vector
    goalPosition = evaderPosition + speedRatio * pursuerRange * direction
    straitRightLength, _ = find_dubins_path_length_left_strait(
        startPosition, startHeading, goalPosition, turnRadius
    )
    straitRightLength = jnp.where(
        jnp.isnan(straitRightLength), jnp.inf, straitRightLength
    )

    ez = straitRightLength - (captureRadius + pursuerRange)
    # return dubinsPathLengths
    return ez


def plot_dubins_EZ(
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    captureRadius,
    pursuerRange,
    evaderHeading,
    evaderSpeed,
    ax,
):
    numPoints = 1000
    rangeX = 1.2
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading

    ZTrue = (
        in_dubins_engagement_zone(
            pursuerPosition,
            pursuerHeading,
            minimumTurnRadius,
            captureRadius,
            pursuerRange,
            pursuerSpeed,
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
        )
        < 0
    )

    ZTrue = ZTrue.reshape(numPoints, numPoints)
    # ZGeometric = ZGeometric.reshape(numPoints, numPoints)

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    colors = ["green"]
    colors = ["red"]
    ax.contour(X, Y, ZTrue, levels=[0], colors=colors, zorder=10000)
    contour_proxy = plt.plot([0], [0], color=colors[0], linestyle="-", label="CSBEZ")
    # add label so it can be added to legend

    # # ax.contour(X, Y, ZGeometric, cmap="summer")
    # # ax.scatter(*pursuerPosition, c="r")
    # ax.set_aspect("equal", "box")
    return ax


def plot_dubins_path(
    startPosition, startHeading, goalPosition, radius, captureRadius, tangentPoint
):
    leftCenter = np.array(
        [
            startPosition[0] - radius * np.sin(startHeading),
            startPosition[1] + radius * np.cos(startHeading),
        ]
    )
    rightCenter = np.array(
        [
            startPosition[0] + radius * np.sin(startHeading),
            startPosition[1] - radius * np.cos(startHeading),
        ]
    )
    fig, ax = plt.subplots()
    ax.scatter(*startPosition, c="g")
    ax.scatter(*goalPosition, c="r")
    ax.scatter(*leftCenter, c="g")
    ax.scatter(*rightCenter, c="b")
    theta = np.linspace(0, 2 * np.pi, 100)
    xl = leftCenter[0] + radius * np.cos(theta)
    yl = leftCenter[1] + radius * np.sin(theta)

    xr = rightCenter[0] + radius * np.cos(theta)
    yr = rightCenter[1] + radius * np.sin(theta)

    xcr = goalPosition[0] + captureRadius * np.cos(theta)
    ycr = goalPosition[1] + captureRadius * np.sin(theta)
    ax.plot(xcr, ycr, "r")

    ax.plot(xl, yl, "b")
    ax.plot(xr, yr, "b")
    ax.scatter(*tangentPoint, c="y")
    ax.plot([goalPosition[0], tangentPoint[0]], [goalPosition[1], tangentPoint[1]], "y")
    ax = plt.gca()
    ax.set_aspect("equal", "box")


def plot_dubins_reachable_set(
    pursuerPosition, pursuerHeading, pursuerRange, radius, ax
):
    numPoints = 1000
    rangeX = 1.1
    x = np.linspace(-rangeX, rangeX, numPoints)
    y = np.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    X = X.flatten()
    Y = Y.flatten()
    Z = vectorized_find_shortest_dubins_path(
        pursuerPosition, pursuerHeading, np.array([X, Y]).T, radius
    )
    Z = Z.reshape(numPoints, numPoints) < pursuerRange
    # Z = np.isclose(Z, pursuerRange, atol=1e-1)
    Z = Z.reshape(numPoints, numPoints)
    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)

    colors = ["brown"]
    ax.contour(X, Y, Z, colors=colors, levels=[0], zorder=10000)
    contour_proxy = plt.plot(
        [0], [0], color=colors[0], linestyle="-", label="Reachable Set"
    )
    ax.set_aspect("equal", "box")
    return ax


def main():
    pursuerVelocity = 2
    minimumTurnRadius = 1
    pursuerRange = 1.0 * np.pi
    pursuerPosition = np.array([0, 0])
    pursuerHeading = np.pi / 2

    # point = np.array([1.0, -2.5])
    point = np.array([1.0, 1.0])

    # length = find_shortest_dubins_path(
    #     pursuerPosition, pursuerHeading, point, minimumTurnRadius
    # )
    leftCenter = np.array(
        [
            pursuerPosition[0] - minimumTurnRadius * np.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * np.cos(pursuerHeading),
        ]
    )
    rightCenter = np.array(
        [
            pursuerPosition[0] + minimumTurnRadius * np.sin(pursuerHeading),
            pursuerPosition[1] - minimumTurnRadius * np.cos(pursuerHeading),
        ]
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    xl = leftCenter[0] + minimumTurnRadius * np.cos(theta)
    yl = leftCenter[1] + minimumTurnRadius * np.sin(theta)
    xr = rightCenter[0] + minimumTurnRadius * np.cos(theta)
    yr = rightCenter[1] + minimumTurnRadius * np.sin(theta)

    offset = 0.5
    length_temp = pursuerRange - offset
    arcAngle = length_temp / minimumTurnRadius
    thetaTemp = -np.linspace(0, arcAngle, 100) - np.pi
    xr_temp = rightCenter[0] + minimumTurnRadius * np.cos(thetaTemp)
    yr_temp = (rightCenter[1] + offset) + minimumTurnRadius * np.sin(thetaTemp)

    # xcr = centerPoint2[0] + minimumTurnRadius * np.cos(theta)
    # ycr = centerPoint2[1] + minimumTurnRadius * np.sin(theta)

    ax = plot_dubins_reachable_set(
        pursuerPosition, pursuerHeading, pursuerRange, minimumTurnRadius
    )
    ax.plot(xl, yl)
    ax.plot(xr_temp, yr_temp)
    # ax.plot(xcr, ycr, c="g")

    ax.scatter(*pursuerPosition, c="r")
    ax.scatter(*point, c="k", marker="x")
    # ax.scatter(centerPoint2[0], centerPoint2[1], c="g")
    ax.plot(xr, yr)

    ax.set_aspect("equal", "box")
    # plot_dubins_path(
    #     pursuerPosition, pursuerHeading, point, minimumTurnRadius, 0.0, tangentPoint
    # )
    # pursuerTime = pursuerRange / pursuerVelocity
    # ax2 = dubinsReachable.plot_dubins_reachable_set(
    #     pursuerHeading - np.pi / 2, pursuerVelocity, minimumTurnRadius, pursuerTime
    # )

    plt.show()


def plot_test_grad(
    pursuerPosition,
    pursuerHeading,
    minimumTurnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
):
    numPoints = 500
    x = jnp.linspace(-2, 2, numPoints)
    y = jnp.linspace(-2, 2, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()

    grad = d_find_dubins_path_length_left_right_d_goal_position_vec(
        pursuerPosition, pursuerHeading, np.array([X, Y]).T, minimumTurnRadius
    )
    speedRatio = evaderSpeed / pursuerSpeed

    dLdxDirectional = (
        jnp.dot(grad, jnp.array([np.cos(evaderHeading), np.sin(evaderHeading)]))
        * speedRatio
        * pursuerRange
        - pursuerRange
    )
    leftCenter = np.array(
        [
            pursuerPosition[0] - minimumTurnRadius * np.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * np.cos(pursuerHeading),
        ]
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    xl = leftCenter[0] + 3 * minimumTurnRadius * np.cos(theta)
    yl = leftCenter[1] + 3 * minimumTurnRadius * np.sin(theta)

    dLdxDirectional = dLdxDirectional.reshape(numPoints, numPoints)
    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    fig, ax = plt.subplots()
    c = ax.contour(X, Y, dLdxDirectional, levels=[0, 1])
    testEvaderXVales = np.linspace(-1.0, 0.5, 10)
    testEvaderYValues = -np.ones_like(testEvaderXVales) * 0.2
    # testEvaderStart = np.array([-0.4, 0.1])
    ax.axis("equal")
    for i, x in enumerate(testEvaderXVales):
        testEvaderStart = np.array([x, testEvaderYValues[i]])

        inEz, goalPosition = in_dubins_engagement_zone_left_right_single(
            pursuerPosition,
            pursuerHeading,
            minimumTurnRadius,
            captureRadius,
            pursuerRange,
            pursuerSpeed,
            testEvaderStart,
            evaderHeading,
            evaderSpeed,
            ax,
        )
    plot_turn_radius_circles(pursuerPosition, pursuerHeading, minimumTurnRadius, ax)
    fig.colorbar(c, ax=ax)
    pursuerTime = pursuerRange / pursuerSpeed
    # ax.scatter(*goalPosition, c="r")
    ax.plot(xl, yl, "b")


def add_arrow(ax, start, end, color, label, annotationFontSize):
    # Draw arrow on plot
    # arrow = FancyArrowPatch(start, end, arrowstyle="->", color=color, lw=2)
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="->",
        color=color,
        lw=2,
        mutation_scale=15,  # Controls arrowhead size
        transform=ax.transData,  # Use data coordinates
    )
    ax.add_patch(arrow)

    # Midpoint text label
    # mid = np.mean([start, end], axis=0)
    # ax.text(mid[0], mid[1], label, fontsize=annotationFontSize)

    # Create proxy for legend
    proxy = plt.plot([0], [0], color=color, lw=2, label=label)
    return proxy


def plot_theta_and_vectors_left_turn(
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    pursuerRange,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    ax,
):
    speedRatio = evaderSpeed / pursuerSpeed
    T = evaderPosition
    P = pursuerPosition
    F = evaderPosition + speedRatio * pursuerRange * np.array(
        [np.cos(evaderHeading), np.sin(evaderHeading)]
    )
    C = jnp.array(
        [
            pursuerPosition[0] - minimumTurnRadius * jnp.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * jnp.cos(pursuerHeading),
        ]
    )
    G = find_counter_clockwise_tangent_point(F, C, minimumTurnRadius)

    circleTheta = np.linspace(0, 2 * np.pi, 100)
    leftCircleX = C[0] + minimumTurnRadius * np.cos(circleTheta)
    leftCircleY = C[1] + minimumTurnRadius * np.sin(circleTheta)

    annotationFontSize = 17

    ax.scatter(*T, c="k")
    ax.scatter(*F, c="k")
    ax.scatter(*C, c="k")
    ax.scatter(*G, c="k")
    ax.scatter(*P, c="k")
    # put labels on points
    ax.text(
        T[0] - 0.001,  # shift left
        T[1] + 0.001,  # shift up
        "E",
        fontsize=annotationFontSize,
        ha="right",
        va="bottom",
    )
    ax.text(F[0], F[1] + 0.02, "F", fontsize=annotationFontSize)
    ax.text(C[0], C[1], r"$C_\ell$", fontsize=annotationFontSize, ha="right", va="top")
    ax.text(G[0] + 0.01, G[1] + 0.01, r"$G_\ell$", fontsize=annotationFontSize)
    ax.text(
        P[0] - 0.01, P[1] - 0.01, "P", fontsize=annotationFontSize, ha="right", va="top"
    )

    # plot circles
    # ax.plot(leftCircleX, leftCircleY, c="b")
    # ax.set_aspect("equal", "box")

    legend_proxies = []

    legend_proxies.append(
        add_arrow(
            ax,
            C,
            F,
            color="cyan",
            label=r"$v_1$",
            annotationFontSize=annotationFontSize,
        )
    )
    legend_proxies.append(
        add_arrow(
            ax, G, F, color="m", label="$v_2$", annotationFontSize=annotationFontSize
        )
    )
    legend_proxies.append(
        add_arrow(
            ax, C, G, color="r", label="$v_3$", annotationFontSize=annotationFontSize
        )
    )
    legend_proxies.append(
        add_arrow(
            ax, C, P, color="b", label="$v_4$", annotationFontSize=annotationFontSize
        )
    )
    legend_proxies.append(
        add_arrow(
            ax,
            T,
            F,
            color="orange",
            label="Evader Path",
            annotationFontSize=annotationFontSize,
        )
    )

    v3 = G - C
    v4 = P - C
    theta = counterclockwise_angle(v4, v3)
    print(theta)
    startAngle = np.arctan2(v4[1], v4[0])
    endAngle = np.arctan2(v3[1], v3[0])
    arcDrawDistance = 1.0 / 3.0
    # label theta
    vmean = np.mean([v3, v4], axis=0)
    ax.text(
        C[0] + 0.04,
        C[1] - 0.01,
        r"$\theta_\ell$",
        fontsize=annotationFontSize,
        va="center",
    )


def main_EZ():
    pursuerPosition = np.array([0, 0])

    pursuerHeading = (1 / 4) * np.pi
    pursuerSpeed = 1

    pursuerRange = 1
    minimumTurnRadius = 0.2
    captureRadius = 0.0
    evaderHeading = (0 / 20) * np.pi
    evaderSpeed = 0.5
    evaderPosition = np.array([-0.8858, 0.8512])
    startTime = time.time()

    # length, tangentPoint = find_dubins_path_length_right_strait(
    #     pursuerPosition, pursuerHeading, evaderPosition, minimumTurnRadius
    # )
    # print("length", length)
    # plot_dubins_path(
    #     pursuerPosition,
    #     pursuerHeading,
    #     evaderPosition,
    #     minimumTurnRadius,
    #     0.0,
    #     tangentPoint,
    # )
    #
    fig, ax = plt.subplots(figsize=(8, 8))
    pursuerRange = 1.0
    # plotEngagementZone(
    #     evaderHeading,
    #     pursuerPosition,
    #     pursuerRange,
    #     captureRadius,
    #     pursuerSpeed,
    #     evaderSpeed,
    #     ax,
    # )
    plot_theta_and_vectors_left_turn(
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        ax,
    )
    plot_dubins_reachable_set(
        pursuerPosition, pursuerHeading, pursuerRange, minimumTurnRadius, ax
    )
    plot_dubins_EZ(
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        evaderHeading,
        evaderSpeed,
        ax,
    )
    plt.xlabel("X", fontsize=20)
    plt.ylabel("Y", fontsize=20)
    # set tick fonhtsize
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0.0, fontsize=18
    )
    plt.show()


if __name__ == "__main__":
    main_EZ()
    # fig, ax = plt.subplots()
    # ax.scatter(0, 0, c="r")
    # plt.show()
