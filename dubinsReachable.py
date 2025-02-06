from jaxlib.xla_client import make_convolution_dimension_numbers
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root, root_scalar
import scipy


def dubins_reachable_set_foward_boundary_right(
    heading, velocity, minimumTurnRadius, time, theta
):
    """
    This computes the boundary (distance from origin) of the reachable set of a dubins vehicle using equations 10 and 11 from
    Cockayne, E. J., and G. W. C. Hall. "Plane motion of a particle subject to curvature constraints." SIAM Journal on Control 13.1 (1975): 197-220.
    """
    rho = 1 / minimumTurnRadius
    vt = velocity * time

    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    x = rho * (1 - cosTheta) + (vt - rho * theta) * sinTheta
    y = rho * sinTheta + (vt - rho * theta) * cosTheta

    return np.vstack([x, y]).T


def dubins_reachable_set_foward_boundary_left(
    heading, velocity, minimumTurnRadius, time, theta
):
    """
    This computes the boundary (distance from origin) of the reachable set of a dubins vehicle using equations 10 and 11 from
    Cockayne, E. J., and G. W. C. Hall. "Plane motion of a particle subject to curvature constraints." SIAM Journal on Control 13.1 (1975): 197-220.
    """
    rho = 1 / minimumTurnRadius
    vt = velocity * time

    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    x = -(rho * (1 - cosTheta) + (vt - rho * theta) * sinTheta)
    y = rho * sinTheta + (vt - rho * theta) * cosTheta

    return np.vstack([x, y]).T


def dubins_reachable_set_backward_boundary_right(
    heading, velocity, minimumTurnRadius, time, theta
):
    rho = 1 / minimumTurnRadius
    vt = velocity * time

    sinTheta = np.sin(theta)

    x = rho * (2 * np.cos(theta) - 1 - np.cos(2 * theta - vt / rho))
    y = rho * (2 * sinTheta - np.sin(2 * theta - vt / rho))
    return np.vstack([x, y]).T


def dubins_reachable_set_backward_boundary_left(
    heading, velocity, minimumTurnRadius, time, theta
):
    rho = 1 / minimumTurnRadius
    vt = velocity * time

    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    # x = -rho * (2 * cosTheta - 1 - np.cos(2 * theta - vt / rho))
    # y = rho * (2 * sinTheta - np.sin(2 * theta - vt / rho))
    x = -rho * (2 * np.cos(theta) - 1 - np.cos(2 * theta - vt / rho))
    y = rho * (2 * sinTheta - np.sin(2 * theta - vt / rho))
    return np.vstack([x, y]).T


def find_theta_backward_boundary(heading, velocity, minimumTurnRadius, time):
    def equation(theta):
        return (
            1
            / minimumTurnRadius
            * (
                2 * np.cos(theta)
                - 1
                - np.cos(2 * theta - velocity * time * minimumTurnRadius)
            )
        )

    theta_solutions = fsolve(equation, [0.0, 0.5, 0.9])
    theta_solutions = theta_solutions[
        (theta_solutions >= 0) & (theta_solutions <= 2 * np.pi)
    ]
    return theta_solutions


def check_forward_boundaries_going_right(
    point, heading, velocity, minimumTurnRadius, time
):
    rho = 1 / minimumTurnRadius

    def func(theta):
        return (
            rho * np.sin(theta)
            + (velocity * time - rho * theta) * np.cos(theta)
            - point[1]
        ) ** 2

    def func_prime(theta):
        return (
            2
            * (
                rho * np.sin(theta)
                + (velocity * time - rho * theta) * np.cos(theta)
                - point[1]
            )
            * (-(velocity * time - rho * theta) * np.sin(theta))
        )
        # return -(velocity * time - rho * theta) * np.sin(theta)

    thetaLimit = velocity * time * minimumTurnRadius
    initialGuesses = [0.0, thetaLimit / 2, thetaLimit]
    thetaSolutions = []
    for initialGuess in initialGuesses:
        thetaSol = root(
            func,
            x0=initialGuess,
            jac=func_prime,
        ).x[0]
        if np.isclose(func(thetaSol), 0, atol=1e-6):
            thetaSolutions.append(thetaSol)
    thetaSolutions = np.array(thetaSolutions)
    thetaSolutions = np.unique(thetaSolutions)
    print("thetaSolutions", thetaSolutions)
    print("func(thetaSolutions)", func(thetaSolutions))

    if len(thetaSolutions) == 0:
        return np.array([[-np.inf, -np.inf]])

    print("thetaLimit", thetaLimit)
    thetaSolutions = np.array(thetaSolutions)
    # thetaSolutions = thetaSolutions[thetaSolutions > 0]
    # thetaSolutions = thetaSolutions[thetaSolutions < thetaLimit]
    print("thetaSolutions", thetaSolutions)
    intersectionPoint = dubins_reachable_set_foward_boundary_right(
        heading, velocity, minimumTurnRadius, time, thetaSolutions
    )
    return intersectionPoint


def check_backward_boundaries_going_right(
    point, heading, velocity, minimumTurnRadius, time, thetaLimit
):
    rho = 1 / minimumTurnRadius

    def func(theta):
        return (
            rho * (2 * np.sin(theta) - np.sin(2 * theta - velocity * time / rho))
            - point[1]
        )

    def func_prime(theta):
        return 2 * rho * (np.cos(theta) - np.cos(2 * theta - velocity * time / rho))

    thetaSolutions = root(func, x0=thetaLimit / 2, jac=func_prime).x

    thetaSolutions = np.array(thetaSolutions)
    thetaSolutions = thetaSolutions[thetaSolutions > 0]
    thetaSolutions = thetaSolutions[thetaSolutions < thetaLimit]
    if thetaSolutions.size == 0:
        return np.array([[-np.inf, -np.inf]])
    intersectionPoint = dubins_reachable_set_backward_boundary_right(
        heading, velocity, minimumTurnRadius, time, thetaSolutions[0]
    )
    return intersectionPoint


def point_in_dubins_reachable_set(
    point, heading, velocity, minimumTurnRadius, time, backThetaLimit
):
    leftHalfPlane = False
    if point[0] < 0:
        # if point is in left half plan, reflect accross y-axis and check ray going right
        point[0] = -point[0]
        leftHalfPlane = True
    # check ray going right
    intersectionPointForwardBoundary = check_forward_boundaries_going_right(
        point, heading, velocity, minimumTurnRadius, time
    )
    print("intersectionPointForwardBoundary", intersectionPointForwardBoundary)
    intersectionPointBackwardBoundary = check_backward_boundaries_going_right(
        point, heading, velocity, minimumTurnRadius, time, backThetaLimit
    )
    print("intersectionPointBackwardBoundary", intersectionPointBackwardBoundary)
    intersectionPointXValues = np.hstack(
        [
            intersectionPointForwardBoundary[:, 0],
            intersectionPointBackwardBoundary[:, 0],
        ]
    )
    numIntersection = np.count_nonzero(intersectionPointXValues > point[0])
    inReachable = numIntersection % 2 == 1
    print("inReachable", inReachable)
    if leftHalfPlane:
        intersectionPointForwardBoundary[:, 0] = -intersectionPointForwardBoundary[:, 0]
        intersectionPointBackwardBoundary[:, 0] = -intersectionPointBackwardBoundary[
            :, 0
        ]
        point[0] = -point[0]
    return (
        inReachable,
        intersectionPointForwardBoundary,
        intersectionPointBackwardBoundary,
    )


def plot_dubins_reachable_set(heading, velocity, minimumTurnRadius, time, ax):
    numSamples = 100
    maxTheta = find_theta_backward_boundary(heading, velocity, minimumTurnRadius, time)
    thetaBackwardRight = np.linspace(0, maxTheta[0], numSamples)
    thetaBackwardLeft = np.linspace(0, maxTheta[0], numSamples)
    thetaForwardRight = np.linspace(0, velocity * time * minimumTurnRadius, numSamples)
    thetaForwardLeft = np.linspace(0, velocity * time * minimumTurnRadius, numSamples)

    pointsForwardLeft = dubins_reachable_set_foward_boundary_left(
        heading, velocity, minimumTurnRadius, time, thetaForwardLeft
    )

    pointsForwardRight = dubins_reachable_set_foward_boundary_right(
        heading, velocity, minimumTurnRadius, time, thetaForwardRight
    )

    pointsBackwardLeft = dubins_reachable_set_backward_boundary_left(
        heading, velocity, minimumTurnRadius, time, thetaBackwardLeft
    )
    pointsBackwardRight = dubins_reachable_set_backward_boundary_right(
        heading, velocity, minimumTurnRadius, time, thetaBackwardRight
    )

    startPosition = np.array([0, 0])
    leftCenter = np.array(
        [
            startPosition[0] - minimumTurnRadius * np.cos(heading),
            startPosition[1] + minimumTurnRadius * np.sin(heading),
        ]
    )
    rightCenter = np.array(
        [
            startPosition[0] + minimumTurnRadius * np.cos(heading),
            startPosition[1] - minimumTurnRadius * np.sin(heading),
        ]
    )
    theta = np.linspace(0, 2 * np.pi, numSamples)
    xRight = rightCenter[0] + minimumTurnRadius * np.cos(theta)
    yRight = rightCenter[1] + minimumTurnRadius * np.sin(theta)
    xLeft = leftCenter[0] + minimumTurnRadius * np.cos(theta)
    yLeft = leftCenter[1] + minimumTurnRadius * np.sin(theta)

    ax.plot(xRight, yRight, c="k")
    ax.plot(xLeft, yLeft, c="k")
    ax.plot(pointsForwardLeft[:, 0], pointsForwardLeft[:, 1], c="r")
    ax.plot(pointsForwardRight[:, 0], pointsForwardRight[:, 1], c="r")
    ax.plot(pointsBackwardLeft[:, 0], pointsBackwardLeft[:, 1], c="r")
    ax.plot(pointsBackwardRight[:, 0], pointsBackwardRight[:, 1], c="r")

    ax = plt.gca()
    ax.set_aspect("equal")
    plt.grid()
    return ax


def main():
    heading = 0
    velocity = 1
    minimumTurnRadius = 1
    pursuerRange = 1.5 * np.pi
    time = pursuerRange / velocity
    point = np.array([-0.5, -1.1])
    thetaLimit = find_theta_backward_boundary(
        heading, velocity, minimumTurnRadius, time
    )
    inReachable, intersectionPointF, intersectionPointB = point_in_dubins_reachable_set(
        point, heading, velocity, minimumTurnRadius, time, thetaLimit
    )

    # time = (6.0 / 3.0) * np.pi
    # time = 1
    ax = plot_dubins_reachable_set(heading, velocity, minimumTurnRadius, time)

    c = "g"
    if inReachable:
        c = "r"

    ax.scatter(*point, c=c)
    ax.scatter(intersectionPointF[:, 0], intersectionPointF[:, 1], c="g", marker="x")
    ax.scatter(intersectionPointB[:, 0], intersectionPointB[:, 1], c="g", marker="x")
    plt.show()


if __name__ == "__main__":
    main()
