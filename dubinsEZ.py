from jax.lax import random_gamma_grad
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
# jax.config.update("jax_platform_name", "gpu")

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
    totalLength = jnp.where(jnp.isnan(totalLength), jnp.inf, totalLength)

    return totalLength, theta


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
    totalLength = jnp.where(jnp.isnan(totalLength), jnp.inf, totalLength)

    return totalLength, theta


find_dubins_path_length_left_strait_vec = jax.vmap(
    find_dubins_path_length_left_strait, in_axes=(None, None, 0, None)
)


# def smooth_min(x, alpha=200.0, epsilon=1e-6):
#     return -jnp.log(jnp.sum(jnp.exp(-alpha * x) + epsilon)) / alpha
def smooth_min(x, alpha=10.0, epsilon=1e-6):
    # Clip inputs to prevent exp overflow
    x = jnp.clip(x, -1e3, 1e3)

    # Replace NaNs (if not already done) — just in case
    x = jnp.where(jnp.isnan(x), 1e6, x)

    # Compute softmin
    softmin = -jnp.log(jnp.sum(jnp.exp(-alpha * x) + epsilon)) / alpha
    return softmin


@jax.jit
def find_shortest_dubins_path(pursuerPosition, pursuerHeading, goalPosition, radius):
    straitLeftLength, _ = find_dubins_path_length_right_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    straitRightLength, _ = find_dubins_path_length_left_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )

    lengths = jnp.array([straitLeftLength, straitRightLength])

    # return smooth_min(lengths)
    # return sofmin(*lengths, 0.01)
    return jnp.min(lengths)
    # return jnp.nanmin(lengths)


# Vectorized version over goalPosition
vectorized_find_shortest_dubins_path = jax.vmap(
    find_shortest_dubins_path,
    in_axes=(None, None, 0, None),  # Vectorize over goalPosition (3rd argument)
)


@jax.jit
def in_dubins_reachable_set_single(
    startPosition,
    startHeading,
    turnRadius,
    pursuerRange,
    goalPosition,
):
    dubinsPathLengths = find_shortest_dubins_path(
        startPosition, startHeading, goalPosition, turnRadius
    )

    rs = dubinsPathLengths - pursuerRange
    return rs


def distance_to_circle(point, center, radius):
    return radius - jnp.linalg.norm(jnp.array(point) - jnp.array(center))


@jax.jit
def angle_in_arc(theta_p, theta1, theta2):
    theta_p = jnp.mod(theta_p, 2 * jnp.pi)
    theta1 = jnp.mod(theta1, 2 * jnp.pi)
    theta2 = jnp.mod(theta2, 2 * jnp.pi)

    in_order = theta1 <= theta2
    return jnp.where(
        in_order,
        jnp.logical_and(theta1 <= theta_p, theta_p <= theta2),
        jnp.logical_or(theta_p >= theta1, theta_p <= theta2),
    )


@jax.jit
def distance_to_arc(point, center, radius, theta1, theta2):
    vec = point - center
    r_p = jnp.linalg.norm(vec)
    theta_p = jnp.arctan2(vec[1], vec[0])

    in_arc = angle_in_arc(theta_p, theta1, theta2)

    # Distance to arc if projection falls on the arc
    dist_to_arc = jnp.abs(radius - r_p)

    # Distance to endpoints if projection is outside the arc
    arc1 = center + radius * jnp.array([jnp.cos(theta1), jnp.sin(theta1)])
    arc2 = center + radius * jnp.array([jnp.cos(theta2), jnp.sin(theta2)])
    dist_to_endpoints = jnp.minimum(
        jnp.linalg.norm(point - arc1), jnp.linalg.norm(point - arc2)
    )

    return jnp.where(in_arc, dist_to_arc, dist_to_endpoints)


@jax.jit
def normalize_angle(theta):
    return (theta + 2 * jnp.pi) % (2 * jnp.pi)


#
# @jax.jit
# def angle_in_arc(theta_p, theta1, theta2):
#     theta_p = normalize_angle(theta_p)
#     theta1 = normalize_angle(theta1)
#     theta2 = normalize_angle(theta2)
#     if_angle_wraps = theta2 < theta1
#     return jnp.where(
#         if_angle_wraps,
#         jnp.logical_or(theta_p >= theta1, theta_p <= theta2),
#         jnp.logical_and(theta_p >= theta1, theta_p <= theta2),
#     )
#
#
@jax.jit
def signed_distance_to_arc(point, center, radius, theta1, theta2):
    vec = point - center
    r_p = jnp.linalg.norm(vec)
    theta_p = jnp.arctan2(vec[1], vec[0])

    in_arc = angle_in_arc(theta_p, theta1, theta2)

    # Signed distance: positive inside (r_p < radius), negative outside (r_p > radius)
    signed_radial = radius - r_p

    # Distance to nearest endpoint (only used when point is outside angular bounds)
    arc1 = center + radius * jnp.array([jnp.cos(theta1), jnp.sin(theta1)])
    arc2 = center + radius * jnp.array([jnp.cos(theta2), jnp.sin(theta2)])
    dist_to_endpoints = jnp.minimum(
        jnp.linalg.norm(point - arc1), jnp.linalg.norm(point - arc2)
    )

    return jnp.where(in_arc, signed_radial, -dist_to_endpoints)


def compute_theta1_theta2_left(center, point_on_circle, arc_length, radius):
    # Compute angle from center to the point
    vec = point_on_circle - center
    theta1 = jnp.arctan2(vec[1], vec[0])

    # Counter-clockwise arc length maps to angle delta = arc_length / radius
    delta_theta = arc_length / radius

    # Wrap to [0, 2π)
    theta2 = jnp.mod(theta1 + delta_theta, 2 * jnp.pi)
    theta1 = jnp.mod(theta1, 2 * jnp.pi)

    return theta1, theta2


def compute_theta1_theta2_right(center, point_on_circle, arc_length, radius):
    # Compute angle from center to the point
    vec = point_on_circle - center
    theta1 = jnp.arctan2(vec[1], vec[0])

    # Counter-clockwise arc length maps to angle delta = arc_length / radius
    delta_theta = arc_length / radius

    # Wrap to [0, 2π)
    theta2 = jnp.mod(theta1 - delta_theta, 2 * jnp.pi)
    theta1 = jnp.mod(theta1, 2 * jnp.pi)

    return theta2, theta1


def signed_distance_to_circle(point, center, radius):
    """
    Computes the signed distance from a point to the boundary of a circle.

    Positive inside, zero on the boundary, negative outside.

    Args:
        point (array-like): Coordinates of the point (x, y).
        center (array-like): Coordinates of the circle center (cx, cy).
        radius (float): Radius of the circle.

    Returns:
        float: Signed distance to the circle.
    """
    point = jnp.asarray(point)
    center = jnp.asarray(center)
    distance_to_center = jnp.linalg.norm(point - center)
    signed_distance = radius - distance_to_center
    return signed_distance


@jax.jit
def side_of_vector(A, B, P):
    v = B - A
    w = P - A
    return v[0] * w[1] - v[1] * w[0]


@jax.jit
def find_dubins_path_length_augmented(
    startPosition, startHeading, turnRadius, pursuerRange, goalPosition
):
    rightCenter = jnp.array(
        [
            startPosition[0] + turnRadius * jnp.sin(startHeading),
            startPosition[1] - turnRadius * jnp.cos(startHeading),
        ]
    )
    leftCenter = jnp.array(
        [
            startPosition[0] - turnRadius * jnp.sin(startHeading),
            startPosition[1] + turnRadius * jnp.cos(startHeading),
        ]
    )
    theta1Left, theta2Left = compute_theta1_theta2_left(
        leftCenter, startPosition, pursuerRange, turnRadius
    )
    theta1Right, theta2Right = compute_theta1_theta2_right(
        rightCenter, startPosition, pursuerRange, turnRadius
    )
    distanceToRightArc = distance_to_arc(
        goalPosition, rightCenter, turnRadius, theta1Right, theta2Right
    )
    distanceToRightArcSigned = signed_distance_to_arc(
        goalPosition, rightCenter, turnRadius, theta1Right, theta2Right
    )
    distanceToLeftArcSigned = signed_distance_to_arc(
        goalPosition, leftCenter, turnRadius, theta1Left, theta2Left
    )
    distanceToLeftArc = distance_to_arc(
        goalPosition, leftCenter, turnRadius, theta1Left, theta2Left
    )

    straitLeftLength, arcLeft = find_dubins_path_length_left_strait(
        startPosition, startHeading, goalPosition, turnRadius
    )
    straitRightLength, arcRight = find_dubins_path_length_right_strait(
        startPosition, startHeading, goalPosition, turnRadius
    )
    straitLeftLength -= pursuerRange
    straitRightLength -= pursuerRange

    bothNegLeft = jnp.logical_and(straitLeftLength < 0, distanceToLeftArcSigned < 0)

    def both_left_neg():
        return jnp.where(
            jnp.abs(straitLeftLength) < jnp.abs(distanceToLeftArcSigned),
            straitLeftLength,
            distanceToLeftArcSigned,
        )

    def not_both_left_neg():
        return jnp.min(
            jnp.array([distanceToLeftArc, straitLeftLength, straitRightLength])
        )

    leftLength = jax.lax.cond(bothNegLeft, both_left_neg, not_both_left_neg)

    bothNegRight = jnp.logical_and(straitRightLength < 0, distanceToRightArcSigned < 0)

    def both_right_neg():
        return jnp.where(
            jnp.abs(straitRightLength) < jnp.abs(distanceToRightArcSigned),
            straitRightLength,
            distanceToRightArcSigned,
        )

    def not_both_right_neg():
        return jnp.min(
            jnp.array([distanceToRightArc, straitLeftLength, straitRightLength])
        )

    rightLength = jax.lax.cond(bothNegRight, both_right_neg, not_both_right_neg)

    side = side_of_vector(
        startPosition,
        startPosition + jnp.array([jnp.cos(startHeading), jnp.sin(startHeading)]),
        goalPosition,
    )
    right = side < 0.0
    return jnp.where(right, rightLength, leftLength)

    # lengths = jnp.array([rightLength, leftLength])
    #
    # dubinsPathLength = jnp.nanmin(lengths)
    # return dubinsPathLength


#
# @jax.jit
# def find_dubins_path_length_augmented(
#     startPosition, startHeading, turnRadius, pursuerRange, goalPosition
# ):
#     rightCenter = jnp.array(
#         [
#             startPosition[0] + turnRadius * jnp.sin(startHeading),
#             startPosition[1] - turnRadius * jnp.cos(startHeading),
#         ]
#     )
#     leftCenter = jnp.array(
#         [
#             startPosition[0] - turnRadius * jnp.sin(startHeading),
#             startPosition[1] + turnRadius * jnp.cos(startHeading),
#         ]
#     )
#     theta1Left, theta2Left = compute_theta1_theta2_left(
#         leftCenter, startPosition, pursuerRange, turnRadius
#     )
#     theta1Right, theta2Right = compute_theta1_theta2_right(
#         rightCenter, startPosition, pursuerRange, turnRadius
#     )
#     straitRightLength, arcRight = find_dubins_path_length_right_strait(
#         startPosition, startHeading, goalPosition, turnRadius
#     )
#     # straighPath = straitRightLength - arcRight * turnRadius
#     # turnPath = arcRight * turnRadius
#     # straitRightLength = straighPath + turnPath / straitRightLength * pursuerRange
#     distanceToRightArc = (
#         distance_to_arc(goalPosition, rightCenter, turnRadius, theta1Right, theta2Right)
#         + pursuerRange
#     )
#     distanceToLeftArc = (
#         distance_to_arc(goalPosition, leftCenter, turnRadius, theta1Left, theta2Left)
#         + pursuerRange
#     )
#     straitLeftLength, arcLeft = find_dubins_path_length_left_strait(
#         startPosition, startHeading, goalPosition, turnRadius
#     )
#     # straighPath = straitLeftLength - arcLeft * turnRadius
#     # turnPath = arcLeft * turnRadius
#     # straitLeftLength = straighPath + turnPath / straitLeftLength * pursuerRange
#     lengths = jnp.array(
#         [straitRightLength, straitLeftLength, distanceToRightArc, distanceToLeftArc]
#     )
#     dubinsPathLength = jnp.nanmin(lengths)
#     return dubinsPathLength


@jax.jit
def in_dubins_reachable_set_augmented_single(
    startPosition,
    startHeading,
    turnRadius,
    pursuerRange,
    goalPosition,
):
    dubinsPathLength = find_dubins_path_length_augmented(
        startPosition, startHeading, turnRadius, pursuerRange, goalPosition
    )
    rs = dubinsPathLength  # - pursuerRange
    return rs


in_dubins_reachable_set = jax.jit(
    jax.vmap(in_dubins_reachable_set_single, in_axes=(None, None, None, None, 0))
)
in_dubins_reachable_set_augmented = jax.jit(
    jax.vmap(
        in_dubins_reachable_set_augmented_single, in_axes=(None, None, None, None, 0)
    )
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
    # dubinsPathLengths = find_shortest_dubins_path(
    #     startPosition, startHeading, goalPositions, turnRadius
    # )
    dubinsPathLengths = find_dubins_path_length_augmented(
        startPosition, startHeading, turnRadius, pursuerRange, goalPositions
    )
    return dubinsPathLengths

    # ez = dubinsPathLengths - pursuerRange
    # return ez


@jax.jit
def in_dubins_engagement_zone_augmented_single(
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
    # dubinsPathLengths = find_shortest_dubins_path(
    #     startPosition, startHeading, goalPositions, turnRadius
    # )
    dubinsPathLengths = find_dubins_path_length_augmented(
        startPosition, startHeading, turnRadius, pursuerRange, goalPositions
    )

    ez = dubinsPathLengths - pursuerRange
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

# Vectorized function using vmap
in_dubins_engagement_zone_agumented = jax.jit(
    jax.vmap(
        in_dubins_engagement_zone_augmented_single,
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
    rangeX = 3.2
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading

    ZTrue = in_dubins_engagement_zone_agumented(
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

    ZTrue = ZTrue.reshape(numPoints, numPoints)
    # ZGeometric = ZGeometric.reshape(numPoints, numPoints)

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    colors = ["green"]
    colors = ["red"]
    ax.contour(X, Y, ZTrue, levels=[0], colors=colors, zorder=10000)
    # ax.contourf(
    #     X, Y, ZTrue, levels=np.linspace(np.min(ZTrue), np.max(ZTrue), 100), alpha=0.5
    # )
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
    pursuerPosition,
    pursuerHeading,
    pursuerRange,
    radius,
    ax,
    colors=["brown"],
    alpha=1.0,
):
    numPoints = 1000
    rangeX = 5.0
    x = np.linspace(-rangeX, rangeX, numPoints)
    y = np.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    X = X.flatten()
    Y = Y.flatten()
    # Z = vectorized_find_shortest_dubins_path(
    #     pursuerPosition, pursuerHeading, np.array([X, Y]).T, radius
    # )
    # Z = Z.reshape(numPoints, numPoints) - pursuerRange
    Z = in_dubins_reachable_set_augmented(
        pursuerPosition, pursuerHeading, radius, pursuerRange, np.array([X, Y]).T
    )
    # Z = np.isclose(Z, pursuerRange, atol=1e-1)
    Z = Z.reshape(numPoints, numPoints)
    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)

    ax.contour(X, Y, Z <= 0.0, colors=colors, levels=[0], zorder=10000, alpha=alpha)
    # c = ax.contourf(X, Y, Z, levels=np.linspace(-1, 1, 101), alpha=0.5)
    # c = ax.pcolormesh(X, Y, Z, alpha=0.5, cmap="coolwarm", vmin=-1, vmax=1)
    # cbar = plt.colorbar(c, ax=ax)
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

    pursuerHeading = 0
    pursuerSpeed = 2

    pursuerRange = 2
    minimumTurnRadius = 0.5
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
    plot_dubins_reachable_set(
        pursuerPosition, pursuerHeading, pursuerRange, minimumTurnRadius, ax
    )
    # plot_dubins_EZ(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     minimumTurnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     evaderHeading,
    #     evaderSpeed,
    #     ax,
    # )
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
