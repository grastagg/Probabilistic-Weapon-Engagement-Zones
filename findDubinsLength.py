import numpy as np
import time
import matplotlib.pyplot as plt
from fast_pursuer import plotEngagementZone, inEngagementZone, inEngagementZoneJax
import jax
import jax.numpy as jnp
from matplotlib.patches import Circle
import matplotlib.cm as cm


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


# @jax.jit
def find_counter_clockwise_tangent_point(p1, c, r):
    v1 = p1 - c
    normV1 = jnp.linalg.norm(v1)
    v3Perallel = (r**2) / normV1**2 * v1
    vPerpendicularNormalized = jnp.array([-v1[1], v1[0]]) / normV1
    v3Perpendicular = -jnp.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized
    v3 = v3Perallel + v3Perpendicular
    pt = c + v3
    return pt


# @jax.jit
def find_clockwise_tangent_point(p1, c, r):
    v1 = p1 - c
    normV1 = jnp.linalg.norm(v1)
    v3Perallel = (r**2) / normV1**2 * v1
    vPerpendicularNormalized = -jnp.array([v1[1], -v1[0]]) / normV1
    v3Perpendicular = jnp.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized

    v3 = v3Perallel + v3Perpendicular
    pt = c + v3
    return pt


# @jax.jit
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


# @jax.jit
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
    print(theta)

    # Compute final path length
    straightLineLength = jnp.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength, tangentPoint


def find_dubins_path_length_left_strait(
    startPosition, startHeading, goalPosition, radius
):
    centerPoint = jnp.array(
        [
            startPosition[0] - radius * np.sin(startHeading),
            startPosition[1] + radius * np.cos(startHeading),
        ]
    )
    tangentPoint = find_counter_clockwise_tangent_point(
        goalPosition, centerPoint, radius
    )

    # Compute angles for arc length
    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint

    theta = counterclockwise_angle(v4, v3)
    print(theta)

    # Compute final path length
    straightLineLength = jnp.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength, tangentPoint


def circle_intersection(c1, c2, r1, r2):
    """
    Computes the intersection point of two circles in 2D.
    If the circles intersect tangentially, returns a single point.

    Parameters:
    c1 : tuple (x1, y1) - Center of the first circle
    c2 : tuple (x2, y2) - Center of the second circle
    r1 : float - Radius of the first circle
    r2 : float - Radius of the second circle

    Returns:
    A list of intersection points [(x, y)] or an empty list if no intersection.
    """
    x1, y1 = c1
    x2, y2 = c2

    # Distance between centers
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check if there is no intersection or if the circles are identical
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []  # No intersection or coincident circles

    # If the circles intersect tangentially
    if d == r1 + r2 or d == abs(r1 - r2):
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = 0  # No height since it’s a single point

        # Midpoint between the centers
        xm = x1 + a * (x2 - x1) / d
        ym = y1 + a * (y2 - y1) / d

        # Single intersection point
        return [(xm, ym)]

    # If the circles intersect at two points (not tangential)
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(r1**2 - a**2)

    # Midpoint between the centers
    xm = x1 + a * (x2 - x1) / d
    ym = y1 + a * (y2 - y1) / d

    # Two intersection points
    x3_1 = xm + h * (y2 - y1) / d
    y3_1 = ym - h * (x2 - x1) / d

    x3_2 = xm - h * (y2 - y1) / d
    y3_2 = ym + h * (x2 - x1) / d

    return np.array([[x3_1, y3_1], [x3_2, y3_2]])


def find_dunbins_path_length_left_right(
    startPosition, startHeading, goalPosition, radius
):
    centerPoint = jnp.array(
        [
            startPosition[0] - radius * np.sin(startHeading),
            startPosition[1] + radius * np.cos(startHeading),
        ]
    )
    secondTurnCenterPoints = circle_intersection(
        centerPoint, goalPosition, 2 * radius, radius
    )
    secondTurnCenterPoint1 = secondTurnCenterPoints[0]
    secondTurnCenterPoint2 = secondTurnCenterPoints[1]

    tangent1 = (secondTurnCenterPoint1 + centerPoint) / 2
    tangent2 = (secondTurnCenterPoint2 + centerPoint) / 2
    print("tangent1", tangent1)
    print("tangent2", tangent2)

    v4 = startPosition - centerPoint

    theta1 = counterclockwise_angle(v4, tangent1 - centerPoint)
    theta2 = counterclockwise_angle(v4, tangent2 - centerPoint)
    print("theta1", theta1)
    print("theta2", theta2)

    secondCenterPoint = secondTurnCenterPoint1
    arcDistanceAroundTurn1 = theta1
    tangentPoint = tangent1
    if theta1 > theta2:
        secondCenterPoint = secondTurnCenterPoint2
        arcDistanceAroundTurn1 = theta2
        tangentPoint = tangent2

    arcDistanceAroundTurn2 = clockwise_angle(
        tangentPoint - secondCenterPoint, goalPosition - secondCenterPoint
    )
    print("arcDistanceAroundTurn1", arcDistanceAroundTurn1)
    print("arcDistanceAroundTurn2", arcDistanceAroundTurn2)

    length = radius * (arcDistanceAroundTurn1 + arcDistanceAroundTurn2)

    return length, secondCenterPoint


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


def main():
    velocity = 1
    minimumTurnRadius = 0.5
    pursuerRange = 1
    pursuerPosition = np.array([0, 0])
    pursuerHeading = np.pi / 2

    # point = np.array([1.0, -2.5])
    point = np.array([-1.9999, 0])

    # length, tangentPoint = find_dubins_path_length_left_strait(
    #     pursuerPosition, pursuerHeading, point, minimumTurnRadius
    # )

    length, centerPoint2 = find_dunbins_path_length_left_right(
        pursuerPosition, pursuerHeading, point, minimumTurnRadius
    )
    print("length", length)
    leftCenter = np.array(
        [
            pursuerPosition[0] - minimumTurnRadius * np.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * np.cos(pursuerHeading),
        ]
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    xl = leftCenter[0] + minimumTurnRadius * np.cos(theta)
    yl = leftCenter[1] + minimumTurnRadius * np.sin(theta)

    xcr = centerPoint2[0] + minimumTurnRadius * np.cos(theta)
    ycr = centerPoint2[1] + minimumTurnRadius * np.sin(theta)

    fig, ax = plt.subplots()
    ax.plot(xl, yl)
    ax.plot(xcr, ycr, c="g")
    ax.scatter(*pursuerPosition, c="r")
    ax.scatter(*point, c="k", marker="x")
    ax.scatter(centerPoint2[0], centerPoint2[1], c="g")

    ax.set_aspect("equal", "box")
    # plot_dubins_path(
    #     pursuerPosition, pursuerHeading, point, minimumTurnRadius, 0.0, tangentPoint
    # )
    plt.show()


if __name__ == "__main__":
    main()
