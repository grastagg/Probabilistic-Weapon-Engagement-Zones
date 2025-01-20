import numpy as np
import time
import matplotlib.pyplot as plt
from fast_pursuer import plotEngagementZone, inEngagementZone, inEngagementZoneJax
import jax
import jax.numpy as jnp


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


# Vectorized version over goalPosition
find_dubins_path_length_vectorized = jax.jit(
    jax.vmap(find_dubins_path_length, in_axes=(None, None, 0, None))
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
        startPosition, startHeading, goalPositions, turnRadius
    )

    ez = dubinsLengths - lam.flatten() * pursuerRange - captureRadius

    root = jnp.min(ez) < 0

    #
    # showPlot = True
    # if showPlot:
    #     fig, ax = plt.subplots()
    #     ax.plot(lam, ez)
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
    return root


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
    # evaderPosition = np.array([-1.0, -1.0])
    # inEZ = new_in_dubins_engagement_zone_single(
    #     startPosition,
    #     startHeading,
    #     turnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerSpeed,
    #     evaderPosition,
    #     agentHeading,
    #     evaderSpeed,
    # )
    # print("In EZ: ", inEZ)

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
