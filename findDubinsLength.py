import numpy as np
import time
import matplotlib.pyplot as plt
from fast_pursuer import plotEngagementZone, inEngagementZone, inEngagementZoneJax
import jax
import jax.numpy as jnp

np.set_printoptions(precision=64, floatmode="maxprec")

# import dubinsReachable
# import testDubins


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(jax.devices())
print(jax.default_backend())


def plot_turn_radius_circles(
    startPosition, startHeading, turnRadius, captureRadius, ax
):
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

    leftXcpInner = leftCenter[0] + (turnRadius - captureRadius) * np.cos(theta)
    leftYcpInner = leftCenter[1] + (turnRadius - captureRadius) * np.sin(theta)
    rightXcpInner = rightCenter[0] + (turnRadius - captureRadius) * np.cos(theta)
    rightYcpInner = rightCenter[1] + (turnRadius - captureRadius) * np.sin(theta)

    ax.plot(leftX, leftY, "b")
    ax.plot(rightX, rightY, "b")
    ax.plot(leftXcpInner, leftYcpInner, "b")
    ax.plot(rightXcpInner, rightYcpInner, "b")


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
def circle_intersection(c1, c2, r1, r2):
    x1, y1 = c1
    x2, y2 = c2

    # Distance between centers
    d = jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # No intersection cases: too far apart, one circle inside another, or identical circles
    no_intersection = (d > r1 + r2) | (d < jnp.abs(r1 - r2))

    def no_intersect_case():
        return jnp.full((2, 2), jnp.nan)  # Use NaN instead of inf for clarity

    def intersect_case():
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h_sq = jnp.maximum(
            0, r1**2 - a**2
        )  # Prevent negative sqrt due to precision issues
        h = jnp.sqrt(h_sq)

        xm = x1 + a * (x2 - x1) / d
        ym = y1 + a * (y2 - y1) / d

        x3_1 = xm + h * (y2 - y1) / d
        y3_1 = ym - h * (x2 - x1) / d
        x3_2 = xm - h * (y2 - y1) / d
        y3_2 = ym + h * (x2 - x1) / d

        return jnp.array([[x3_1, y3_1], [x3_2, y3_2]])

    return jax.lax.cond(no_intersection, no_intersect_case, intersect_case)


def move_goal_point_turn_strait_by_capture_radius_right(
    pursuerPosition, centerPoint, goalPosition, turnRadius, captureRadius
):
    """To find the shortest path to a circle, move the goal position,
    if it is more than sqrt(cap^2 + turn^2) away from the centerPoint don't move it
    if it is inside that distance, move it to the edge of the circle"""
    distance = jnp.linalg.norm(goalPosition - centerPoint)

    def move_point_circle():
        interseciontPoint1, intersectionPoint2 = circle_intersection(
            centerPoint, goalPosition, turnRadius, captureRadius
        )
        v1 = pursuerPosition - centerPoint
        v2 = interseciontPoint1 - centerPoint
        v3 = intersectionPoint2 - centerPoint
        theta1 = clockwise_angle(v1, v2)
        theta2 = clockwise_angle(v1, v3)

        return jnp.where(
            theta1 < theta2, interseciontPoint1, intersectionPoint2
        ), jnp.where(theta1 < theta2, interseciontPoint1, intersectionPoint2)

    def move_point():
        tangentPoint = find_clockwise_tangent_point(
            goalPosition, centerPoint, turnRadius
        )
        direction = goalPosition - tangentPoint
        direction = direction / jnp.linalg.norm(direction)
        return goalPosition - captureRadius * direction, tangentPoint

    return jax.lax.cond(
        distance < jnp.sqrt(captureRadius**2 + turnRadius**2),
        move_point_circle,
        move_point,
    )


def move_goal_point_turn_strait_by_capture_radius_left(
    pursuerPosition, centerPoint, goalPosition, turnRadius, captureRadius
):
    """To find the shortest path to a circle, move the goal position,
    if it is more than sqrt(cap^2 + turn^2) away from the centerPoint don't move it
    if it is inside that distance, move it to the edge of the circle"""
    distance = jnp.linalg.norm(goalPosition - centerPoint)

    def move_point_circle():
        interseciontPoint1, intersectionPoint2 = circle_intersection(
            centerPoint, goalPosition, turnRadius, captureRadius
        )
        v1 = pursuerPosition - centerPoint
        v2 = interseciontPoint1 - centerPoint
        v3 = intersectionPoint2 - centerPoint
        theta1 = counterclockwise_angle(v1, v2)
        theta2 = counterclockwise_angle(v1, v3)

        return jnp.where(
            theta1 < theta2, interseciontPoint1, intersectionPoint2
        ), jnp.where(theta1 < theta2, interseciontPoint1, intersectionPoint2)

    def move_point():
        tangentPoint = find_counter_clockwise_tangent_point(
            goalPosition, centerPoint, turnRadius
        )
        direction = goalPosition - tangentPoint
        direction = direction / jnp.linalg.norm(direction)
        return goalPosition - captureRadius * direction, tangentPoint

    return jax.lax.cond(
        distance < jnp.sqrt(captureRadius**2 + turnRadius**2),
        move_point_circle,
        move_point,
    )


@jax.jit
def find_dubins_path_length_right_strait(
    startPosition, startHeading, goalPosition, radius, captureRadius
):
    centerPoint = jnp.array(
        [
            startPosition[0] + radius * jnp.sin(startHeading),
            startPosition[1] - radius * jnp.cos(startHeading),
        ]
    )
    newGoalPosition, tangentPoint = move_goal_point_turn_strait_by_capture_radius_right(
        startPosition, centerPoint, goalPosition, radius, captureRadius
    )
    # jax.debug.print("newGoalPosition: {x}", x=newGoalPosition)
    # tangentPoint = find_clockwise_tangent_point(goalPosition, centerPoint, radius)

    # Compute angles for arc length
    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint

    theta = clockwise_angle(v4, v3)

    # Compute final path length
    straightLineLength = jnp.linalg.norm(newGoalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength, tangentPoint


find_dubins_path_length_right_strait_vec = jax.vmap(
    find_dubins_path_length_right_strait, in_axes=(None, None, 0, None, None)
)


@jax.jit
def find_dubins_path_length_left_strait(
    startPosition, startHeading, goalPosition, radius, captureRadius
):
    centerPoint = jnp.array(
        [
            startPosition[0] - radius * jnp.sin(startHeading),
            startPosition[1] + radius * jnp.cos(startHeading),
        ]
    )
    newGoalPosition, tangentPoint = move_goal_point_turn_strait_by_capture_radius_left(
        startPosition, centerPoint, goalPosition, radius, captureRadius
    )
    # tangentPoint = find_counter_clockwise_tangent_point(
    #     goalPosition, centerPoint, radius
    # )

    # Compute angles for arc length
    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint

    theta = counterclockwise_angle(v4, v3)

    # Compute final path length
    straightLineLength = jnp.linalg.norm(newGoalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength, tangentPoint


find_dubins_path_length_left_strait_vec = jax.vmap(
    find_dubins_path_length_left_strait, in_axes=(None, None, 0, None, None)
)


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


def temp_find_left_right(pursuerPosition, PursuerHeading, goalPosition, radius):
    length, secondCenterPoint = find_dubins_path_length_left_right(
        pursuerPosition, PursuerHeading, goalPosition, radius
    )
    return length


d_find_dubins_path_length_left_right_d_goal_position = jax.grad(
    temp_find_left_right, argnums=2
)
d_find_dubins_path_length_left_right_d_goal_position_vec = jax.vmap(
    d_find_dubins_path_length_left_right_d_goal_position, in_axes=(None, None, 0, None)
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


@jax.jit
def find_shortest_dubins_path(
    pursuerPosition, pursuerHeading, goalPosition, radius, captureRadius
):
    leftRightLength, _ = find_dubins_path_length_left_right(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    rightLeftLength, _ = find_dubins_path_length_right_left(
        pursuerPosition, pursuerHeading, goalPosition, radius
    )
    straitRightLength, _ = find_dubins_path_length_right_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius, captureRadius
    )
    straitLeftLength, _ = find_dubins_path_length_left_strait(
        pursuerPosition, pursuerHeading, goalPosition, radius, captureRadius
    )

    lengths = jnp.array(
        [leftRightLength, rightLeftLength, straitLeftLength, straitRightLength]
    )

    return jnp.nanmin(lengths)


# Vectorized version over goalPosition
vectorized_find_shortest_dubins_path = jax.vmap(
    find_shortest_dubins_path,
    in_axes=(None, None, 0, None, None),  # Vectorize over goalPosition (3rd argument)
)


@jax.jit
def in_dubins_engagement_zone_left_right_geometric_single(
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
    direction = jnp.array([jnp.cos(evaderHeading), jnp.sin(evaderHeading)])
    lam = 1.0

    goalPosition = evaderPosition + lam * speedRatio * pursuerRange * direction
    dubinsPathLength, _ = find_dubins_path_length_left_right(
        startPosition, startHeading, goalPosition, turnRadius
    )
    ez = dubinsPathLength - lam * (pursuerRange)
    inEz = ez < 0
    return inEz


@jax.jit
def in_dubins_engagement_zone_right_left_geometric_single(
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
    direction = jnp.array([jnp.cos(evaderHeading), jnp.sin(evaderHeading)])
    lam = 1.0

    goalPosition = evaderPosition + lam * speedRatio * pursuerRange * direction
    dubinsPathLength, _ = find_dubins_path_length_right_left(
        startPosition, startHeading, goalPosition, turnRadius
    )
    ez = dubinsPathLength - lam * (pursuerRange)
    inEz = ez < 0
    return inEz


@jax.jit
def circle_segment_intersection(circle_center, radius, A, B, startPoint):
    """
    Computes the single intersection point of a line segment AB with a circle using JAX.

    Args:
        circle_center (tuple): (h, k) center of the circle.
        radius (float): Radius of the circle.
        A (tuple): (x1, y1) first point of the segment.
        B (tuple): (x2, y2) second point of the segment.

    Returns:
        jnp.ndarray: Intersection point (x, y) or empty array if no intersection.
    """
    h, k = jnp.array(circle_center, dtype=jnp.float32)
    r = radius
    A = jnp.array(A, dtype=jnp.float64)
    B = jnp.array(B, dtype=jnp.float64)

    # Direction vector of the line segment
    d = B - A

    # Quadratic coefficients
    A_coeff = jnp.dot(d, d)
    B_coeff = 2 * jnp.dot(d, A - jnp.array([h, k]))
    C_coeff = jnp.dot(A - jnp.array([h, k]), A - jnp.array([h, k])) - r**2

    # Compute discriminant
    discriminant = B_coeff**2 - 4 * A_coeff * C_coeff

    sqrt_disc = jnp.sqrt(discriminant)
    t1 = (-B_coeff + sqrt_disc) / (
        2 * A_coeff
    )  # Only considering one root (single intersection)
    t2 = (-B_coeff - sqrt_disc) / (2 * A_coeff)

    t1Valid = jnp.logical_and(0 <= t1, t1 <= 1)
    t2Valid = jnp.logical_and(0 <= t2, t2 <= 1)

    # Compute the intersection point
    intersection_point1 = A + t1 * d
    intersection_point2 = A + t2 * d

    dist1ToStart = jnp.linalg.norm(intersection_point1 - startPoint)
    dist2ToStart = jnp.linalg.norm(intersection_point2 - startPoint)

    noIntersections = jnp.array([jnp.inf, jnp.inf])

    chooseT1 = jnp.logical_or(
        jnp.logical_and(
            dist1ToStart <= dist2ToStart, jnp.logical_and(t1Valid, t2Valid)
        ),
        t1Valid,
    )
    chooseT2 = jnp.logical_or(
        jnp.logical_and(
            dist2ToStart <= dist1ToStart, jnp.logical_and(t2Valid, t1Valid)
        ),
        t2Valid,
    )

    chooseNeither = jnp.logical_not(jnp.logical_or(chooseT1, chooseT2))

    def find_intersection_point():
        return jax.lax.cond(
            chooseT1, lambda: intersection_point1, lambda: intersection_point2
        )

    def compute_intersection_point():
        return jax.lax.cond(
            chooseNeither,
            lambda: noIntersections,
            find_intersection_point,
        )

    intersection_point = jax.lax.cond(
        discriminant >= 0, compute_intersection_point, lambda: noIntersections
    )

    # Check if t is within the segment range [0, 1]

    # intersection_point = intersection_point

    return intersection_point


@jax.jit
def line_segment_intersection(A1, A2, B1, B2):
    # Vector from A1 to A2
    d1 = A2 - A1
    # Vector from B1 to B2
    d2 = B2 - B1
    # Vector from A1 to B1
    r = B1 - A1

    # Cross product of d1 and d2
    denom = jnp.cross(d1, d2)

    # # If denom is zero, the lines are parallel or collinear
    # if jnp.isclose(denom, 0):
    #     return None  # No intersection or infinite intersections (collinear)

    # Calculate t and u, the parameters for intersection point
    t = jnp.cross(r, d2) / denom
    u = jnp.cross(r, d1) / denom
    tValid = jnp.logical_and(0 <= t, t <= 1)
    uValid = jnp.logical_and(0 <= u, u <= 1)

    # Check if t and u are within [0, 1], which means the intersection point is within both segments
    noIntersections = jnp.array([jnp.inf, jnp.inf])
    intersection = jax.lax.cond(
        tValid & uValid, lambda: A1 + t * d1, lambda: noIntersections
    )
    # intersection = A1 + t * d1  # or B1 + u * d2, both should be the same
    return intersection


# @jax.jit
# def nanmin_jax(ez1, ez2, ez3):
#     values = jnp.array([ez1, ez2, ez3])
#     valid_values = jnp.where(
#         jnp.isnan(values), jnp.inf, values
#     )  # Replace NaNs with +inf
#     return jnp.min(valid_values)  # Compute min ignoring NaNs


@jax.jit
def find_goal_point_and_lam_circle_intersection(
    pursuerPosition,
    pursuerHeading,
    turnRadius,
    evaderPosition,
    evaderFinalPosition,
    pursuerRange,
    speedRatio,
    circleCenter,
    captureRadius,
):
    intersectionPoints = circle_segment_intersection(
        circleCenter,
        turnRadius - captureRadius,
        evaderPosition,
        evaderFinalPosition,
        evaderPosition,
    )
    goalPosition = intersectionPoints
    # ensure goal position is slightly outside circle
    directionVector = goalPosition - circleCenter
    directionVector = directionVector / jnp.linalg.norm(directionVector)
    goalPosition = (
        circleCenter + ((turnRadius - captureRadius) * 1.001) * directionVector
    )

    lam = jnp.linalg.norm(goalPosition - evaderPosition) / (speedRatio * pursuerRange)
    lam = jnp.where(lam > 1, jnp.nan, lam)
    lam = jnp.where(lam < 0, jnp.nan, lam)
    return lam, goalPosition


@jax.jit
def find_goal_point_and_lam_line_intersection(
    pursuerPosition,
    pursuerHeading,
    evaderPosition,
    evaderFinalPosition,
    pursuerRange,
    speedRatio,
    captureRadius,
):
    pursuerDirection = jnp.array([jnp.cos(pursuerHeading), jnp.sin(pursuerHeading)])
    goalPosition = line_segment_intersection(
        pursuerPosition - pursuerDirection * captureRadius,
        pursuerPosition + pursuerDirection * pursuerRange,
        evaderPosition,
        evaderFinalPosition,
    )
    lam = jnp.linalg.norm(goalPosition - evaderPosition) / (speedRatio * pursuerRange)
    lam = jnp.where(lam > 1, jnp.nan, lam)
    lam = jnp.where(lam < 0, jnp.nan, lam)
    return lam, goalPosition


# @jax.jit
def in_dubins_engagement_zone_right_strait_geometric_single(
    pursuerPosition,
    pursuerHeading,
    turnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
):
    speedRatio = evaderSpeed / pursuerSpeed

    finalPoint = evaderPosition + speedRatio * pursuerRange * jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )
    rightCenter = jnp.array(
        [
            pursuerPosition[0] + turnRadius * jnp.sin(pursuerHeading),
            pursuerPosition[1] - turnRadius * jnp.cos(pursuerHeading),
        ]
    )

    lamCircleIntersection, goalPositionCicleIntersection = (
        find_goal_point_and_lam_circle_intersection(
            pursuerPosition,
            pursuerHeading,
            turnRadius,
            evaderPosition,
            finalPoint,
            pursuerRange,
            speedRatio,
            rightCenter,
            captureRadius,
        )
    )

    lamLineIntersection, goalPositionLineIntersection = (
        find_goal_point_and_lam_line_intersection(
            pursuerPosition,
            pursuerHeading,
            evaderPosition,
            finalPoint,
            pursuerRange,
            speedRatio,
            captureRadius,
        )
    )

    captureRadiusCircleIntersection = circle_segment_intersection(
        pursuerPosition, captureRadius, evaderPosition, finalPoint, pursuerPosition
    )
    inCaptureRadius = jnp.logical_not(jnp.isinf(captureRadiusCircleIntersection).all())

    lamFinal = 1.0
    goalPositionFinal = finalPoint
    lams = jnp.array([lamCircleIntersection, lamLineIntersection, lamFinal])

    goalPositions = jnp.array(
        [
            goalPositionCicleIntersection,
            goalPositionLineIntersection,
            goalPositionFinal,
        ]
    )
    dubinsPathLengths, _ = find_dubins_path_length_right_strait_vec(
        pursuerPosition, pursuerHeading, goalPositions, turnRadius, captureRadius
    )

    ez = dubinsPathLengths - lams * (pursuerRange)
    # jax.debug.print("lams: {x}", x=lams)
    # jax.debug.print("dupinsPathLengths: {x}", x=dubinsPathLengths)
    # jax.debug.print("ezs: {x}", x=ez)

    ez = jnp.nanmin(ez)
    inEz = jnp.logical_or(ez < 0, inCaptureRadius)

    return inEz


@jax.jit
def in_dubins_engagement_zone_left_strait_geometric_single(
    pursuerPosition,
    pursuerHeading,
    turnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
):
    speedRatio = evaderSpeed / pursuerSpeed

    finalPoint = evaderPosition + speedRatio * pursuerRange * jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )

    leftCenter = jnp.array(
        [
            pursuerPosition[0] - turnRadius * jnp.sin(pursuerHeading),
            pursuerPosition[1] + turnRadius * jnp.cos(pursuerHeading),
        ]
    )

    lamCircleIntersection, goalPositionCicleIntersection = (
        find_goal_point_and_lam_circle_intersection(
            pursuerPosition,
            pursuerHeading,
            turnRadius,
            evaderPosition,
            finalPoint,
            pursuerRange,
            speedRatio,
            leftCenter,
            captureRadius,
        )
    )

    lamLineIntersection, goalPositionLineIntersection = (
        find_goal_point_and_lam_line_intersection(
            pursuerPosition,
            pursuerHeading,
            evaderPosition,
            finalPoint,
            pursuerRange,
            speedRatio,
            captureRadius,
        )
    )

    lamFinal = 1.0
    goalPositionFinal = finalPoint
    captureRadiusCircleIntersection = circle_segment_intersection(
        pursuerPosition, captureRadius, evaderPosition, finalPoint, pursuerPosition
    )
    lamStart = jnp.linalg.norm(captureRadiusCircleIntersection - evaderPosition) / (
        speedRatio * pursuerRange
    )
    goalPositionStart = captureRadiusCircleIntersection

    lams = jnp.array([lamCircleIntersection, lamLineIntersection, lamStart, lamFinal])

    goalPositions = jnp.array(
        [
            goalPositionCicleIntersection,
            goalPositionLineIntersection,
            goalPositionStart,
            goalPositionFinal,
        ]
    )
    dubinsPathLengths, _ = find_dubins_path_length_left_strait_vec(
        pursuerPosition, pursuerHeading, goalPositions, turnRadius, captureRadius
    )

    ez = dubinsPathLengths - lams * (pursuerRange)
    # jax.debug.print("lams: {x}", x=lams)
    # jax.debug.print("dupinsPathLengths: {x}", x=dubinsPathLengths)
    # jax.debug.print("ezs: {x}", x=ez)

    ez = jnp.nanmin(ez)
    inCaptureRadius = dubinsPathLengths < captureRadius
    inEz = jnp.logical_or(ez < 0, jnp.any(inCaptureRadius))

    return inEz


in_dubins_engagement_zone_right_strait_geometric = jax.jit(
    jax.vmap(
        in_dubins_engagement_zone_right_strait_geometric_single,
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

in_dubins_engagement_zone_left_strait_geometric = jax.jit(
    jax.vmap(
        in_dubins_engagement_zone_left_strait_geometric_single,
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

in_dubins_engagement_zone_left_right_geometric = jax.jit(
    jax.vmap(
        in_dubins_engagement_zone_left_right_geometric_single,
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

in_dubins_engagement_zone_right_left_geometric = jax.jit(
    jax.vmap(
        in_dubins_engagement_zone_right_left_geometric_single,
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


# @jax.jit
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
        startPosition, startHeading, goalPositions, turnRadius, captureRadius
    )

    ez = dubinsPathLengths - lam.flatten() * (pursuerRange)

    # ezMin = jnp.nanmin(ez)
    inEz = ez < 0
    inEz = jnp.any(inEz)
    #
    # rightCenter = np.array(
    #     [
    #         startPosition[0] + turnRadius * np.sin(startHeading),
    #         startPosition[1] - turnRadius * np.cos(startHeading),
    #     ]
    # )
    # lamCircleIntersection, goalPositionCicleIntersection = (
    #     find_goal_point_and_lam_circle_intersection(
    #         startPosition,
    #         startHeading,
    #         turnRadius,
    #         evaderPosition,
    #         evaderPosition + speedRatio * pursuerRange * direction,
    #         pursuerRange,
    #         speedRatio,
    #         rightCenter,
    #         captureRadius,
    #     )
    # )
    #
    # lamCircleIntersection1, goalPositionCicleIntersection1 = (
    #     find_goal_point_and_lam_circle_intersection(
    #         startPosition,
    #         startHeading,
    #         turnRadius,
    #         evaderPosition,
    #         evaderPosition + speedRatio * pursuerRange * direction,
    #         pursuerRange,
    #         speedRatio,
    #         rightCenter,
    #         0.0,
    #     )
    # )

    # fig, ax = plt.subplots()
    # ax.scatter(lam, ez, c=lam)
    # ax.scatter(lam, dubinsPathLengths)
    # print("lam min", lam[jnp.nanargmin(ez)])
    # print("point min", goalPositions[jnp.nanargmin(ez)])
    # fig2, ax2 = plt.subplots()
    # plot_turn_radius_circles(
    #     startPosition, startHeading, turnRadius, captureRadius, ax2
    # )
    #
    # ax2.scatter(goalPositions[:, 0], goalPositions[:, 1], c=lam)
    # ax2.scatter(
    #     goalPositions[jnp.nanargmin(ez), 0], goalPositions[jnp.nanargmin(ez), 1]
    # )
    # ax2.set_aspect("equal")
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


def in_dubins_engagement_zone_geometric(
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
    ezRightStrait = in_dubins_engagement_zone_right_strait_geometric(
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
    ezLeftStrait = in_dubins_engagement_zone_left_strait_geometric(
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
    ezLeftRight = in_dubins_engagement_zone_left_right_geometric(
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
    ezRightLeft = in_dubins_engagement_zone_right_left_geometric(
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
    # return jnp.logical_or(ezRightStrait, ezLeftStrait)
    return jnp.logical_or(
        jnp.logical_or(ezRightStrait, ezLeftStrait),
        jnp.logical_or(ezRightLeft, ezLeftRight),
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
    numPoints = 400
    x = jnp.linspace(-2, 2, numPoints)
    y = jnp.linspace(-2, 2, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading
    startTime = time.time()
    ZGeometric = in_dubins_engagement_zone_geometric(
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

    ZTrue = in_dubins_engagement_zone(
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
    print("x where not same", X[ZTrue != ZGeometric])
    print("y where not same", Y[ZTrue != ZGeometric])
    ZTrue = ZTrue.reshape(numPoints, numPoints)
    # print("time to run true", time.time() - startTime)

    ZTrue = ZTrue.reshape(numPoints, numPoints)
    ZGeometric = ZGeometric.reshape(numPoints, numPoints)

    fig, ax = plt.subplots()
    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    ax.contour(X, Y, ZTrue)
    ax.contour(X, Y, ZGeometric, cmap="summer")
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

    xCapt = goalPosition[0] + captureRadius * np.cos(theta)
    yCapt = goalPosition[1] + captureRadius * np.sin(theta)

    ax.plot(xCapt, yCapt, "r")

    ax.plot(xcr, ycr, "r")

    ax.plot(xl, yl, "b")
    ax.plot(xr, yr, "b")
    ax.scatter(*tangentPoint, c="y")
    ax.plot([goalPosition[0], tangentPoint[0]], [goalPosition[1], tangentPoint[1]], "y")
    ax = plt.gca()
    ax.set_aspect("equal", "box")
    return fig, ax


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


def main_EZ():
    pursuerPosition = np.array([0, 0])

    pursuerHeading = np.pi / 2
    pursuerSpeed = 2

    pursuerRange = 1
    minimumTurnRadius = 1.0
    captureRadius = 0.1
    evaderSpeed = 1
    evaderPosition = np.array([0.09523809523809501, -0.02506265664160423])
    evaderHeading = (5 / 20) * np.pi
    #
    inEZ = in_dubins_engagement_zone_single(
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
    print("inEZ", inEZ)
    inEZ = in_dubins_engagement_zone_right_strait_geometric_single(
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
    print("inEZ", inEZ)
    # length = find_dubins_path_length_right_strait(
    #     pursuerPosition,
    #     pursuerHeading,
    #     evaderPosition,
    #     minimumTurnRadius,
    #     captureRadius,
    # )
    # print("length", length)
    # ax.scatter(*newGoalPosition, c="g")
    # fig.colorbar(c, ax=ax)
    #
    # ax = plot_dubins_EZ(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     minimumTurnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     evaderHeading,
    #     evaderSpeed,
    # )
    # plotEngagementZone(
    #     evaderHeading,
    #     pursuerPosition,
    #     pursuerRange,
    #     captureRadius,
    #     pursuerSpeed,
    #     evaderSpeed,
    #     ax,
    # )
    # theta = np.linspace(0, 2 * np.pi, 100)
    # captureRadiusX = pursuerPosition[0] + captureRadius * np.cos(theta)
    # captureRadiusY = pursuerPosition[1] + captureRadius * np.sin(theta)
    # ax.plot(captureRadiusX, captureRadiusY, "r")
    plt.show()


if __name__ == "__main__":
    main_EZ()
