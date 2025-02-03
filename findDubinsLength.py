import numpy as np
import time
import matplotlib.pyplot as plt
from fast_pursuer import plotEngagementZone, inEngagementZone, inEngagementZoneJax
import jax
import jax.numpy as jnp
from matplotlib.patches import Circle
import matplotlib.cm as cm


import dubinsReachable
import testDubins


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


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
def circle_intersection(c1, c2, r1, r2):
    x1, y1 = c1
    x2, y2 = c2

    # Distance between centers
    d = jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check for no intersection or identical circles
    no_intersection = (d > r1 + r2) | (d < jnp.abs(r1 - r2)) | (d == 0)

    def no_intersect_case():
        return jnp.full((2, 2), jnp.nan)  # Return NaN-filled array of shape (2,2)

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
        return jnp.nan, jnp.array([jnp.nan, jnp.nan])

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
        return jnp.nan, jnp.array([jnp.nan, jnp.nan])

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


@jax.jit
def find_shortest_dubins_path(pursuerPosition, pursuerHeading, goalPosition, radius):
    leftRightLength, _ = find_dubins_path_length_left_right(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    rightLeftLength, _ = find_dubins_path_length_right_left(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    straitLeftLength, _ = find_dubins_path_length_right_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    straitRightLength, _ = find_dubins_path_length_left_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )

    lengths = jnp.array(
        [leftRightLength, rightLeftLength, straitLeftLength, straitRightLength]
    )

    return jnp.nanmin(lengths)


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
    numPoints = 100
    lam = jnp.linspace(0, 1, numPoints)[:, None]

    # Compute goal positions
    direction = jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )  # Heading unit vector
    goalPositions = evaderPosition + lam * speedRatio * pursuerRange * direction
    dubinsPathLengths = vectorized_find_shortest_dubins_path(
        startPosition, startHeading, goalPositions, turnRadius
    )

    ez = dubinsPathLengths - lam.flatten() * (captureRadius + pursuerRange)

    # ezMin = jnp.nanmin(ez)
    inEz = ez < 0
    inEz = jnp.any(inEz)
    return inEz


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


def plot_dubins_EZ(
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    captureRadius,
    pursuerRange,
    evaderHeading,
    evaderSpeed,
):
    numPoints = 500
    x = jnp.linspace(-2, 2, numPoints)
    y = jnp.linspace(-2, 2, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading
    Z = in_dubins_engagement_zone(
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
    Z = Z.reshape(numPoints, numPoints)
    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z)
    ax.scatter(*pursuerPosition, c="r")
    ax.set_aspect("equal", "box")
    ax.set_aspect("equal", "box")
    theta = np.linspace(0, 2 * np.pi, 100)
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
    leftX = leftCenter[0] + minimumTurnRadius * np.cos(theta)
    leftY = leftCenter[1] + minimumTurnRadius * np.sin(theta)
    rightX = rightCenter[0] + minimumTurnRadius * np.cos(theta)
    rightY = rightCenter[1] + minimumTurnRadius * np.sin(theta)
    plt.plot(leftX, leftY, "b")
    plt.plot(rightX, rightY, "b")
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


def plot_dubins_reachable_set(pursuerPosition, pursuerHeading, pursuerRange, radius):
    numPoints = 1000
    x = np.linspace(-5, 5, numPoints)
    y = np.linspace(-5, 5, numPoints)
    [X, Y] = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    X = X.flatten()
    Y = Y.flatten()
    Z = vectorized_find_shortest_dubins_path(
        pursuerPosition, pursuerHeading, np.array([X, Y]).T, radius
    )
    Z = Z.reshape(numPoints, numPoints) < pursuerRange
    # Z = np.isclose(Z, pursuerRange, atol=1e-1)
    print(Z)
    Z = Z.reshape(numPoints, numPoints)
    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)

    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, Z)
    ax.scatter(*pursuerPosition, c="r")
    ax.set_aspect("equal", "box")
    ax.set_aspect("equal", "box")
    theta = np.linspace(0, 2 * np.pi, 100)
    leftCenter = np.array(
        [
            pursuerPosition[0] - radius * np.sin(pursuerHeading),
            pursuerPosition[1] + radius * np.cos(pursuerHeading),
        ]
    )
    rightCenter = np.array(
        [
            pursuerPosition[0] + radius * np.sin(pursuerHeading),
            pursuerPosition[1] - radius * np.cos(pursuerHeading),
        ]
    )
    leftX = leftCenter[0] + radius * np.cos(theta)
    leftY = leftCenter[1] + radius * np.sin(theta)
    rightX = rightCenter[0] + radius * np.cos(theta)
    rightY = rightCenter[1] + radius * np.sin(theta)
    plt.plot(leftX, leftY, "b")
    plt.plot(rightX, rightY, "b")
    return ax


def main():
    pursuerVelocity = 1
    minimumTurnRadius = 1
    pursuerRange = 1.5 * np.pi
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


def main_EZ():
    pursuerPosition = np.array([0, 0])
    pursuerHeading = np.pi / 2
    pursuerSpeed = 2
    pursuerRange = 1
    minimumTurnRadius = 0.2
    captureRadius = 0.0
    evaderHeading = np.pi / 2
    evaderSpeed = 1
    evaderPosition = np.array([0.0, 0.9])
    inEz = in_dubins_engagement_zone_single(
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
    print(inEz)
    ax = plot_dubins_EZ(
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        evaderHeading,
        evaderSpeed,
    )
    plotEngagementZone(
        evaderHeading,
        pursuerPosition,
        pursuerRange,
        captureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
    )
    plt.show()


if __name__ == "__main__":
    main_EZ()
