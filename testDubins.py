import numpy as np
import time
import matplotlib.pyplot as plt
from fast_pursuer import plotEngagementZone, inEngagementZone, inEngagementZoneJax
import jax
import jax.numpy as jnp
from matplotlib.patches import Circle


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


# @jax.jit
# def clockwise_angle(v1, v2):
#     # Calculate determinant and dot product
#     det = v1[0] * v2[1] - v1[1] * v2[0]
#     dot = v1[0] * v2[0] + v1[1] * v2[1]
#
#     # Compute angle and normalize to [0, 2*pi]
#     angle = jnp.arctan2(det, dot)
#     angle_ccw = (angle + 2 * jnp.pi) % (2 * jnp.pi)
#
#     return angle_ccw


@jax.jit
def clockwise_angle(v1, v2):
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
# def counterclockwise_angle(v1, v2):
#     # Calculate determinant and dot product
#     det = v1[0] * v2[1] - v1[1] * v2[0]
#     dot = v1[0] * v2[0] + v1[1] * v2[1]
#
#     # Compute clockwise angle
#     angle = jnp.arctan2(-det, dot)
#     angle_cw = (angle + 2 * jnp.pi) % (2 * jnp.pi)
#
#     return angle_cw


@jax.jit
def counterclockwise_angle(v1, v2):
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
def find_dubins_path_length(
    startPosition, startHeading, goalPosition, radius, captureRadius
):
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
    not_inside_right_circle = dist_to_right > (radius)
    inside_left_circle = dist_to_left < (radius)
    clockwise = jnp.where(
        jnp.logical_or(
            jnp.logical_and(right_closer, not_inside_right_circle),
            jnp.logical_and(jnp.logical_not(right_closer), inside_left_circle),
        ),
        True,
        False,
    )
    # inside_right_circle = dist_to_right < radius
    # inside_left_circle = dist_to_left < radius
    # inside_right_circle_close_enough_to_edge = np.logical_and(
    #     inside_right_circle, dist_to_right > radius + captureRadius
    # )
    # inside_left_circle_close_enough_to_edge = np.logical_and(
    #     inside_left_circle, dist_to_left > radius + captureRadius
    # )
    # clockwise = jnp.where(
    #     jnp.logical_or(
    #         jnp.logical_and(right_closer, jnp.logical_not(inside_right_circle)),
    #         jnp.logical_and(jnp.logical_not(right_closer), inside_left_circle),
    #     ),
    #     True,
    #     False,
    # )
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
    # jax.debug.print(
    #     "goalPosition: {goalPosition}, Theta:{theta}, Clockwise: {cw}",
    #     goalPosition=goalPosition,
    #     theta=theta,
    #     cw=clockwise,
    # )

    # Compute final path length
    straightLineLength = jnp.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * theta
    totalLength = arcLength + straightLineLength

    return totalLength


@jax.jit
def counterclockwise_circle_intersection(x1, y1, r1, x2, y2, r2):
    """
    Finds the counterclockwise-most intersection point of two circles using JAX.

    Parameters:
        x1, y1, r1 : float - Center (x1, y1) and radius r1 of the first circle.
        x2, y2, r2 : float - Center (x2, y2) and radius r2 of the second circle.

    Returns:
        Tuple (x, y): The counterclockwise-most intersection point.
    """
    # Distance between centers
    d = jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Compute intersection points
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = jnp.sqrt(r1**2 - a**2)

    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d

    x_offset = h * (y2 - y1) / d
    y_offset = h * (x2 - x1) / d

    intersection1 = (x3 + x_offset, y3 - y_offset)
    intersection2 = (x3 - x_offset, y3 + y_offset)

    # Compute angles relative to the second circle's center
    angle1 = jnp.arctan2(intersection1[1] - y1, intersection1[0] - x1)
    angle2 = jnp.arctan2(intersection2[1] - y1, intersection2[0] - x1)

    # Return the counterclockwise-most intersection first
    ccw_intersection, other_intersection = jax.lax.cond(
        angle1 > angle2,
        lambda _: (intersection1, intersection2),
        lambda _: (intersection2, intersection1),
        None,
    )

    return ccw_intersection, other_intersection


def plot_circles_and_intersection(
    x1, y1, r1, x2, y2, r2, intersection, otherIntersection
):
    """Plots the two circles and highlights the intersection points."""
    fig, ax = plt.subplots(figsize=(6, 6))
    circle1 = plt.Circle(
        (x1, y1), r1, color="b", fill=False, linestyle="dashed", label="Circle 1"
    )
    circle2 = plt.Circle(
        (x2, y2), r2, color="r", fill=False, linestyle="dashed", label="Circle 2"
    )

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.set_xlim(min(x1, x2) - max(r1, r2), max(x1, x2) + max(r1, r2))
    ax.set_ylim(min(y1, y2) - max(r1, r2), max(y1, y2) + max(r1, r2))
    ax.set_aspect("equal")

    # Plot centers and intersection points
    ax.plot([x1, x2], [y1, y2], "ko", markersize=5, label="Centers")
    ax.plot(
        *intersection,
        "ro",
        markersize=10,
        markeredgecolor="k",
        label="Counterclockwise-most",
    )
    ax.plot(
        *otherIntersection,
        markersize=10,
        markeredgecolor="k",
        label="Other Intersection",
    )

    plt.legend()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Circle Intersections")
    plt.grid()
    plt.show()


# Example Usage
x1, y1, r1 = 0.0, 0.0, 1.0
x2, y2, r2 = 0.0, 0.91, 0.1

ccw_point, otherIntersection = counterclockwise_circle_intersection(
    x1, y1, r1, x2, y2, r2
)
ccw_point = tuple(ccw_point)  # Convert from JAX array to tuple for plotting
otherIntersection = tuple(otherIntersection)
print("Clockwise-most intersection:", otherIntersection)
print("Counterclockwise-most intersection:", ccw_point)

plot_circles_and_intersection(x1, y1, r1, x2, y2, r2, ccw_point, otherIntersection)


def find_dubins_path_length_no_jax(
    startPosition, startHeading, goalPosition, radius, ax
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

    clockwise = False
    if np.linalg.norm(goalPosition - leftCenter) < np.linalg.norm(
        goalPosition - rightCenter
    ):
        centerPoint = leftCenter
        clockwise = False
        if np.linalg.norm(goalPosition - leftCenter) < radius:
            clockwise = True
            centerPoint = rightCenter

    else:
        centerPoint = rightCenter
        clockwise = True
        if np.linalg.norm(goalPosition - rightCenter) < radius:
            clockwise = False
            centerPoint = leftCenter

    if clockwise:
        tangentPoint = find_clockwise_tangent_point(goalPosition, centerPoint, radius)
    else:
        tangentPoint = find_counter_clockwise_tangent_point(
            goalPosition, centerPoint, radius
        )

    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint
    if clockwise:
        theta = clockwise_angle(v3, v4)
    else:
        theta = counterclockwise_angle(v3, v4)

    straitLineLength = np.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * np.abs(theta)

    totalLength = arcLength + straitLineLength

    showPlot = True

    if showPlot:
        ax.scatter(*startPosition, c="g")
        ax.scatter(*goalPosition, c="r")
        ax.scatter(*leftCenter, c="g")
        ax.scatter(*rightCenter, c="b")
        ax.plot(*v3, "b")
        ax.plot(*v4, "b")
        theta = np.linspace(0, 2 * np.pi, 100)
        xl = leftCenter[0] + radius * np.cos(theta)
        yl = leftCenter[1] + radius * np.sin(theta)

        xr = rightCenter[0] + radius * np.cos(theta)
        yr = rightCenter[1] + radius * np.sin(theta)

        ax.plot(xl, yl, "b")
        ax.plot(xr, yr, "b")
        ax.scatter(*tangentPoint, c="y")
        ax.plot(
            [goalPosition[0], tangentPoint[0]], [goalPosition[1], tangentPoint[1]], "y"
        )
        ax = plt.gca()
        ax.set_aspect("equal", "box")
    return totalLength


# Vectorized version over goalPosition
find_dubins_path_length_vectorized = jax.jit(
    jax.vmap(find_dubins_path_length, in_axes=(None, None, 0, None, None))
)


# @jax.jit
def in_dubins_ez_objective_function(
    lam,
    startPosition,
    startHeading,
    evaderStart,
    evaderHeading,
    pursuerRange,
    turnRadius,
    captureRadius,
):
    # Compute goal positions
    direction = jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )  # Heading unit vector
    goalPosition = evaderStart + lam * pursuerRange * direction
    dubinsLengths = find_dubins_path_length(
        startPosition, startHeading, goalPosition, turnRadius
    )
    ez = dubinsLengths - lam * pursuerRange - captureRadius
    return ez


in_dubins_ez_objective_function_dLambda = jax.jit(
    jax.grad(in_dubins_ez_objective_function, argnums=0)
)
in_dubins_ez_objective_function_dLambda = jax.grad(
    in_dubins_ez_objective_function, argnums=0
)


# @jax.jit
def newtons_method(f, df, x0, args=(), tol=1e-6, max_iter=50):
    """Newton's method for root-finding in JAX with extra arguments.

    Args:
        f: Function whose root we want to find. Should accept x and *args.
        df: Derivative of f. Should accept x and *args.
        x0: Initial guess.
        args: Tuple of extra arguments to pass to f and df.
        tol: Convergence tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Approximate root of the function.
    """

    def body_fn(state):
        i, x, error = state
        step = f(x, *args) / df(x, *args)  # Newton step
        alpha = 0.1

        x_new = x - step  # Newton update
        # x_new = x - f(x, *args) / df(x, *args)  # Newton update
        x_new = jnp.clip(x_new, 0, 1)  # Ensure the root stays within [0, 1]

        return i + 1, x_new, jnp.abs(x_new - x)  # Update error

    def cond_fn(state):
        i, _, error = state
        return jnp.logical_and(error > tol, i < max_iter)  # Use logical_and

    # Run the while loop for Newton's method
    i, x_final, _ = jax.lax.while_loop(cond_fn, body_fn, (0, x0, jnp.inf))

    # jax.debug.print(
    #     "Final x: {x}, Final f(x): {f}, iterations: {iter}",
    #     x=x_final,
    #     f=f(x_final, *args),
    #     iter=i,
    # )

    return jnp.isclose(
        f(x_final, *args), 0
    )  # Return the root, not the function value at the root


def find_closest_collision_point(
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
    lam = jnp.linspace(0, 1, 500)[:, None]  # Shape (100, 1) for broadcasting

    # Compute goal positions
    direction = jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )  # Heading unit vector
    goalPositions = evaderPosition + lam * speedRatio * pursuerRange * direction
    dubinsLengths = find_dubins_path_length_vectorized(
        startPosition, startHeading, goalPositions, turnRadius, captureRadius
    )

    ez = dubinsLengths - lam.flatten() * pursuerRange - captureRadius

    lamClosestToZero = lam[jnp.argmin(jnp.abs(ez))]
    xClosestToZero = (
        evaderPosition + lamClosestToZero * speedRatio * pursuerRange * direction
    )
    return xClosestToZero


@jax.jit
def new_in_dubins_engagement_zone_single(
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
    # root = newtons_method(
    #     f=in_dubins_ez_objective_function,
    #     df=in_dubins_ez_objective_function_dLambda,
    #     x0=0.0,
    #     args=(
    #         startPosition,
    #         startHeading,
    #         evaderPosition,
    #         evaderHeading,
    #         pursuerRange,
    #         turnRadius,
    #         captureRadius,
    #     ),
    # )
    lam = jnp.linspace(0, 1, 500)[:, None]  # Shape (100, 1) for broadcasting

    # Compute goal positions
    direction = jnp.array(
        [jnp.cos(evaderHeading), jnp.sin(evaderHeading)]
    )  # Heading unit vector
    goalPositions = evaderPosition + lam * speedRatio * pursuerRange * direction
    dubinsLengths = find_dubins_path_length_vectorized(
        startPosition, startHeading, goalPositions, turnRadius, captureRadius
    )

    ez = dubinsLengths - lam.flatten() * pursuerRange - captureRadius
    inEz = jnp.min(ez) < 0

    #
    # showPlot = True
    # if showPlot:
    #     fig, ax = plt.subplots()
    #
    #     ax.plot(lam, dubinsLengths)
    #     fig2, ax2 = plt.subplots()
    #     ax2.plot(goalPositions[:, 0], goalPositions[:, 1])
    #     theta = np.linspace(0, 2 * np.pi, 100)
    #     leftCenter = np.array(
    #         [
    #             startPosition[0] - turnRadius * np.sin(startHeading),
    #             startPosition[1] + turnRadius * np.cos(startHeading),
    #         ]
    #     )
    #     rightCenter = np.array(
    #         [
    #             startPosition[0] + turnRadius * np.sin(startHeading),
    #             startPosition[1] - turnRadius * np.cos(startHeading),
    #         ]
    #     )
    #     leftX = leftCenter[0] + turnRadius * np.cos(theta)
    #     leftY = leftCenter[1] + turnRadius * np.sin(theta)
    #     rightX = rightCenter[0] + turnRadius * np.cos(theta)
    #     rightY = rightCenter[1] + turnRadius * np.sin(theta)
    #     ax2.plot(leftX, leftY)
    #     ax2.plot(rightX, rightY)
    return inEz


# Vectorized function using vmap
new_in_dubins_engagement_zone = jax.jit(
    jax.vmap(
        new_in_dubins_engagement_zone_single,
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
    Z = new_in_dubins_engagement_zone(
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
    print(Z)

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
    startHeading = np.pi / 2
    turnRadius = 0.5
    captureRadius = 0.0
    pursuerRange = 1.0
    pursuerSpeed = 2
    evaderSpeed = 1
    agentHeading = -np.pi

    # agentPosition = np.array([0.0, -0.1])
    #
    # length = find_dubins_path_length(
    #     startPosition, startHeading, agentPosition, turnRadius
    # )
    # print("Length: ", length)
    evaderPosition = np.array([0.25, 0.4])

    inEZ = new_in_dubins_engagement_zone_single(
        startPosition,
        startHeading,
        turnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        agentHeading,
        evaderSpeed,
    )
    print("In EZ: ", inEZ)

    # targetPoint = np.array([-0.14408818, -0.25])
    # length = find_dubins_path_length_no_jax(
    #     startPosition, startHeading, targetPoint, turnRadius
    # )
    # print("Length: ", length)

    # ax = plot_dubins_engagement_zone(
    #     startPosition,
    #     startHeading,
    #     turnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerSpeed,
    #     evaderSpeed,
    #     agentHeading,
    # )
    # #
    # plotEngagementZone(
    #     agentHeading,
    #     startPosition,
    #     pursuerRange,
    #     captureRadius,
    #     pursuerSpeed,
    #     evaderSpeed,
    #     ax,
    # )
    # plt.scatter(*evaderPosition)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
