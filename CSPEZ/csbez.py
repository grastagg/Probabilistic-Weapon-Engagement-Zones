import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


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


@jax.jit
def find_shortest_dubins_path(pursuerPosition, pursuerHeading, goalPosition, radius):
    straitLeftLength, _ = find_dubins_path_length_right_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    straitRightLength, _ = find_dubins_path_length_left_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )

    lengths = jnp.array([straitLeftLength, straitRightLength])

    return jnp.min(lengths)
    #


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
    dubinsPathLengths = find_shortest_dubins_path(
        startPosition, startHeading, goalPositions, turnRadius
    )
    ez = dubinsPathLengths - pursuerRange
    return ez


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

    ez = dubinsPathLengths  # - pursuerRange
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
