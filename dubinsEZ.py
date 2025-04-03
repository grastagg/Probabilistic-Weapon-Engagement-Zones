from jax.lax import random_gamma_grad
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import patches
from fast_pursuer import plotEngagementZone, inEngagementZone, inEngagementZoneJax
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


find_dubins_path_length_left_strait_vec = jax.vmap(
    find_dubins_path_length_left_strait, in_axes=(None, None, 0, None)
)


@jax.jit
def circle_intersection(c1, c2, r1, r2):
    x1, y1 = c1
    x2, y2 = c2

    # Distance between centers
    d = jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check for no intersection or identical circles
    no_intersection = (d > r1 + r2) | (d < jnp.abs(r1 - r2)) | (d == 0)

    def no_intersect_case():
        return jnp.full((2, 2), jnp.inf)  # Return NaN-filled array of shape (2,2)

    def intersect_case():
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = jnp.sqrt(r1**2 - a**2)

        xm = x1 + a * (x2 - x1) / d
        ym = y1 + a * (y2 - y1) / d

        x3_1 = xm + h * (y2 - y1) / d
        y3_1 = ym - h * (x2 - x1) / d
        x3_2 = xm - h * (y2 - y1) / d
        y3_2 = ym + h * (x2 - x1) / d

        return jnp.array([[x3_1, y3_1], [x3_2, y3_2]])

    return jax.lax.cond(no_intersection, no_intersect_case, intersect_case)


@jax.jit
def find_dubins_path_length_left_right(
    startPosition, startHeading, goalPosition, radius
):
    centerPoint = jnp.array(
        [
            startPosition[0] - radius * jnp.sin(startHeading),
            startPosition[1] + radius * jnp.cos(startHeading),
        ]
    )

    secondTurnCenterPoints = circle_intersection(
        centerPoint, goalPosition, 2 * radius, radius
    )

    # Check if no intersection points exist
    no_intersection = secondTurnCenterPoints.shape[0] == 0

    def nan_case():
        return jnp.inf, jnp.array([jnp.inf, jnp.inf])

    def valid_case():
        secondTurnCenterPoint1 = secondTurnCenterPoints[0]
        secondTurnCenterPoint2 = secondTurnCenterPoints[1]

        tangent1 = (secondTurnCenterPoint1 + centerPoint) / 2
        tangent2 = (secondTurnCenterPoint2 + centerPoint) / 2

        v4 = startPosition - centerPoint

        theta1 = counterclockwise_angle(v4, tangent1 - centerPoint)
        theta2 = counterclockwise_angle(v4, tangent2 - centerPoint)

        use_second = theta1 > theta2
        secondCenterPoint = jax.lax.select(
            use_second, secondTurnCenterPoint2, secondTurnCenterPoint1
        )
        arcDistanceAroundTurn1 = jax.lax.select(use_second, theta2, theta1)
        tangentPoint = jax.lax.select(use_second, tangent2, tangent1)

        arcDistanceAroundTurn2 = clockwise_angle(
            tangentPoint - secondCenterPoint, goalPosition - secondCenterPoint
        )

        length = radius * (arcDistanceAroundTurn1 + arcDistanceAroundTurn2)

        return length, secondCenterPoint

    return jax.lax.cond(no_intersection, nan_case, valid_case)


find_dubins_path_length_left_right_vec = jax.vmap(
    find_dubins_path_length_left_right, in_axes=(None, None, 0, None)
)


@jax.jit
def find_dubins_path_length_right_left(
    startPosition, startHeading, goalPosition, radius
):
    centerPoint = jnp.array(
        [
            startPosition[0] + radius * jnp.sin(startHeading),
            startPosition[1] - radius * jnp.cos(startHeading),
        ]
    )

    secondTurnCenterPoints = circle_intersection(
        centerPoint, goalPosition, 2 * radius, radius
    )

    # Check if no intersection points exist
    no_intersection = secondTurnCenterPoints.shape[0] == 0

    def nan_case():
        return jnp.inf, jnp.array([jnp.inf, jnp.inf])

    def valid_case():
        secondTurnCenterPoint1 = secondTurnCenterPoints[0]
        secondTurnCenterPoint2 = secondTurnCenterPoints[1]

        tangent1 = (secondTurnCenterPoint1 + centerPoint) / 2
        tangent2 = (secondTurnCenterPoint2 + centerPoint) / 2

        v4 = startPosition - centerPoint

        theta1 = clockwise_angle(v4, tangent1 - centerPoint)
        theta2 = clockwise_angle(v4, tangent2 - centerPoint)

        use_second = theta1 > theta2
        secondCenterPoint = jax.lax.select(
            use_second, secondTurnCenterPoint2, secondTurnCenterPoint1
        )
        arcDistanceAroundTurn1 = jax.lax.select(use_second, theta2, theta1)
        tangentPoint = jax.lax.select(use_second, tangent2, tangent1)

        arcDistanceAroundTurn2 = counterclockwise_angle(
            tangentPoint - secondCenterPoint, goalPosition - secondCenterPoint
        )

        length = radius * (arcDistanceAroundTurn1 + arcDistanceAroundTurn2)

        return length, secondCenterPoint

    return jax.lax.cond(no_intersection, nan_case, valid_case)


find_dubins_path_length_right_left_vec = jax.vmap(
    find_dubins_path_length_right_left, in_axes=(None, None, 0, None)
)


# def differentiable_min(a, b):
#     epsilon = 10
#     return -(1.0 / epsilon) * jnp.log(jnp.exp(-(epsilon * a)) + jnp.exp(-(epsilon * b)))


def differentiable_min(a, b):
    epsilon = 10

    # Replace NaNs with a large value to avoid discontinuities
    safe_a = jnp.where(jnp.isnan(a), jnp.inf, a)
    safe_b = jnp.where(jnp.isnan(b), jnp.inf, b)

    # Numerically stable log-sum-exp trick
    smooth_min = -(1.0 / epsilon) * jnp.log(
        jnp.exp(-(epsilon * safe_a)) + jnp.exp(-(epsilon * safe_b))
    )

    # Preserve NaN if both inputs were NaN
    both_nan = jnp.isnan(a) & jnp.isnan(b)
    return jnp.where(both_nan, jnp.nan, smooth_min)


@jax.jit
def find_shortest_dubins_path(pursuerPosition, pursuerHeading, goalPosition, radius):
    # leftRightLength, _ = find_dubins_path_length_left_right(
    #     pursuerPosition, pursuerHeading, goalPosition, radius
    # )
    # rightLeftLength, _ = find_dubins_path_length_right_left(
    #     pursuerPosition, pursuerHeading, goalPosition, radius
    # )
    straitLeftLength, _ = find_dubins_path_length_right_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    straitRightLength, _ = find_dubins_path_length_left_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )

    # lengths = jnp.array(
    #     [leftRightLength, rightLeftLength, straitLeftLength, straitRightLength]
    # )
    lengths = jnp.array([straitLeftLength, straitRightLength])
    # lengths = jnp.array([straitLeftLength])  # , straitRightLength])

    return jnp.nanmin(lengths)
    # return differentiable_min(*lengths)


# Vectorized version over goalPosition
vectorized_find_shortest_dubins_path = jax.vmap(
    find_shortest_dubins_path,
    in_axes=(None, None, 0, None),  # Vectorize over goalPosition (3rd argument)
)


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
    # return dubinsPathLengths
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
    numPoints = 1500
    rangeX = 2.0
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading
    # startTime = time.time()
    # ZGeometric = in_dubins_engagement_zone_geometric(
    #     pursuerPosition,
    #     pursuerHeading,
    #     minimumTurnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerSpeed,
    #     jnp.array([X, Y]).T,
    #     evaderHeadings,
    #     evaderSpeed,
    # )
    # startTime = time.time()
    # ZGeometric = in_dubins_engagement_zone_geometric(
    #     pursuerPosition,
    #     pursuerHeading,
    #     minimumTurnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerSpeed,
    #     jnp.array([X, Y]).T,
    #     evaderHeadings,
    #     evaderSpeed,
    # )
    # print("time to run geometric", time.time() - startTime)

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
    # startTime = time.time()
    # ZTrue = in_dubins_engagement_zone(
    #     pursuerPosition,
    #     pursuerHeading,
    #     minimumTurnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerSpeed,
    #     jnp.array([X, Y]).T,
    #     evaderHeadings,
    #     evaderSpeed,
    # )
    # ZTrue = ZTrue.reshape(numPoints, numPoints)
    # print("time to run true", time.time() - startTime)

    ZTrue = ZTrue.reshape(numPoints, numPoints)
    # ZGeometric = ZGeometric.reshape(numPoints, numPoints)

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    colors = ["red"]
    ax.contour(X, Y, ZTrue, levels=[0], colors=colors, zorder=10000)
    # ax.contour(X, Y, ZGeometric, cmap="summer")
    ax.scatter(*pursuerPosition, c="r")
    ax.set_aspect("equal", "box")
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
    rangeX = 1.2
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

    ax.contour(X, Y, Z)
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

    ax.scatter(*T, c="k", label="T")
    ax.scatter(*F, c="k", label="F")
    ax.scatter(*C, c="k", label="C")
    ax.scatter(*G, c="k", label="G")
    ax.scatter(*P, c="k", label="P")
    # put labels on points
    ax.text(T[0], T[1], "T", fontsize=12)
    ax.text(F[0], F[1], "F", fontsize=12)
    ax.text(C[0], C[1], "C", fontsize=12)
    ax.text(G[0], G[1], "G", fontsize=12)
    ax.text(P[0], P[1], "P", fontsize=12, verticalalignment="bottom")

    # plot circles
    ax.plot(leftCircleX, leftCircleY, c="b")
    ax.set_aspect("equal", "box")

    # plot arrow vector from C to G and label v3
    ax.annotate(
        "",
        xy=G,
        xytext=C,
        arrowprops=dict(arrowstyle="->", lw=2, color="r"),
    )
    ax.text(np.mean([C[0], G[0]]), np.mean([C[1], G[1]]), "v3", fontsize=12)

    # plot arrow vector from G to F and label v2
    ax.annotate("", xy=F, xytext=G, arrowprops=dict(arrowstyle="->", lw=2, color="r"))
    ax.text(
        np.mean([G[0], F[0]]),
        np.mean(
            [G[1], F[1]],
        ),
        "v2",
        fontsize=12,
    )

    # plot arrow vector from C to P and label v4
    ax.annotate("", xy=P, xytext=C, arrowprops=dict(arrowstyle="->", lw=2, color="r"))
    ax.text(
        np.mean([C[0], P[0]]),
        np.mean(
            [C[1], P[1]],
        ),
        "v4",
        fontsize=12,
    )
    # plot arrow vector from C to F and label v1
    ax.annotate("", xy=F, xytext=C, arrowprops=dict(arrowstyle="->", lw=2, color="r"))
    ax.text(
        np.mean([C[0], F[0]]),
        np.mean(
            [C[1], F[1]],
        ),
        "v1",
        fontsize=12,
    )
    # plot arrow vector from T to F and label mu*v*t
    ax.annotate("", xy=F, xytext=T, arrowprops=dict(arrowstyle="->", lw=2, color="r"))
    ax.text(np.mean([T[0], F[0]]), np.mean([T[1], F[1]]), r"$\mu v t$", fontsize=12)

    v3 = G - C
    v4 = P - C
    theta = counterclockwise_angle(v4, v3)
    startAngle = np.arctan2(v4[1], v4[0])
    endAngle = np.arctan2(v3[1], v3[0])
    arcDrawDistance = 1.0 / 3.0
    arc = patches.Arc(
        tuple(C),
        minimumTurnRadius * 2 * arcDrawDistance,
        minimumTurnRadius * 2 * arcDrawDistance,
        angle=0.0,
        theta1=np.degrees(startAngle),
        theta2=np.degrees(endAngle),
        color="g",
        linewidth=2,
    )
    ax.add_patch(arc)
    # label theta
    vmean = np.mean([v3, v4], axis=0)
    ax.text(
        C[0] + arcDrawDistance * 1.2 * vmean[0],
        C[1] + arcDrawDistance * 1.2 * vmean[1],
        r"$\theta$",
        fontsize=12,
        multialignment="center",
    )


def main_EZ():
    pursuerPosition = np.array([0, 0])

    pursuerHeading = (2 / 4) * np.pi
    pursuerSpeed = 2

    pursuerRange = 1
    minimumTurnRadius = 0.5
    captureRadius = 0.0
    evaderHeading = (0 / 20) * np.pi
    evaderSpeed = 0.5
    evaderPosition = np.array([-0.4, 0.0])
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
    fig, ax = plt.subplots()
    pursuerRange = 1.0
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
    # plotEngagementZone(
    #     evaderHeading,
    #     pursuerPosition,
    #     pursuerRange,
    #     captureRadius,
    #     pursuerSpeed,
    #     evaderSpeed,
    #     ax,
    # )
    # plot_theta_and_vectors_left_turn(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     minimumTurnRadius,
    #     pursuerRange,
    #     evaderPosition,
    #     evaderHeading,
    #     evaderSpeed,
    #     ax,
    # )
    # plot_dubins_reachable_set(
    #     pursuerPosition, pursuerHeading, pursuerRange, minimumTurnRadius, ax
    # )
    plt.show()


if __name__ == "__main__":
    main_EZ()
