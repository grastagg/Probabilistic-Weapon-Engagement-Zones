from jaxlib.xla_client import make_convolution_dimension_numbers
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


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

    return np.array([x, y])


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

    return np.array([x, y])


def dubins_reachable_set_backward_boundary_right(
    heading, velocity, minimumTurnRadius, time, theta
):
    rho = 1 / minimumTurnRadius
    vt = velocity * time

    sinTheta = np.sin(theta)

    x = rho * (2 * np.cos(theta) - 1 - np.cos(2 * theta - vt / rho))
    y = rho * (2 * sinTheta - np.sin(2 * theta - vt / rho))
    return np.array([x, y])


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
    return np.array([x, y])


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
        )

    thetaLimit = velocity * time * minimumTurnRadius

    theta_solutions = fsolve(func, [thetaLimit / 2])
    thetaSolutions = np.array(theta_solutions)
    thetaSolutions = theta_solutions[thetaSolutions > 0]
    thetaSolutions = theta_solutions[thetaSolutions < thetaLimit]
    intersectionPoint = dubins_reachable_set_foward_boundary_right(
        heading, velocity, minimumTurnRadius, time, theta_solutions[0]
    )
    return intersectionPoint


def check_backward_boundaries_going_right(
    point, heading, velocity, minimumTurnRadius, time
):
    rho = 1 / minimumTurnRadius

    def func(theta):
        return (
            rho * (2 * np.sin(theta) - np.sin(2 * theta - velocity * time / rho))
            - point[1]
        )

    thetaLimit = velocity * time * minimumTurnRadius

    theta_solutions = fsolve(func, [0.0])
    thetaSolutions = np.array(theta_solutions)
    thetaSolutions = theta_solutions[thetaSolutions > 0]
    thetaSolutions = theta_solutions[thetaSolutions < thetaLimit]
    intersectionPoint = dubins_reachable_set_backward_boundary_right(
        heading, velocity, minimumTurnRadius, time, theta_solutions[0]
    )
    return intersectionPoint


def point_in_dubins_reachable_set(point, heading, velocity, minimumTurnRadius, time):
    leftHalfPlane = False
    if point[0] < 0:
        # if point is in left half plan, reflect accross y-axis and check ray going right
        point[0] = -point[0]
        leftHalfPlane = True
    # check ray going right
    intersectionPointForwardBoundary = check_forward_boundaries_going_right(
        point, heading, velocity, minimumTurnRadius, time
    )
    intersectionPointBackwardBoundary = check_backward_boundaries_going_right(
        point, heading, velocity, minimumTurnRadius, time
    )

    intersectionPointXValues = np.array(
        [intersectionPointForwardBoundary[0], intersectionPointBackwardBoundary[0]]
    )
    numIntersection = np.count_nonzero(intersectionPointXValues > point[0])
    inReachable = numIntersection % 2 == 1
    print("inReachable", inReachable)
    if leftHalfPlane:
        intersectionPointForwardBoundary[0] = -intersectionPointForwardBoundary[0]
        intersectionPointBackwardBoundary[0] = -intersectionPointBackwardBoundary[0]
    return intersectionPointForwardBoundary, intersectionPointBackwardBoundary


def plot_dubins_reachable_set(heading, velocity, minimumTurnRadius, time):
    numSamples = 100
    theta = np.linspace(
        -velocity * time * minimumTurnRadius,
        velocity * time * minimumTurnRadius,
        numSamples,
    )
    maxTheta = find_theta_backward_boundary(heading, velocity, minimumTurnRadius, time)
    thetaBackwardRight = np.linspace(0, maxTheta[0], numSamples)
    thetaBackwardLeft = np.linspace(0, maxTheta[0], numSamples)
    thetaForwardRight = np.linspace(0, velocity * time * minimumTurnRadius, numSamples)
    thetaForwardLeft = np.linspace(0, velocity * time * minimumTurnRadius, numSamples)

    pointsForwardLeft = []
    pointsForwardRight = []
    pointsBackwardLeft = []
    pointsBackwardRight = []

    for i in range(numSamples):
        pointsForwardLeft.append(
            dubins_reachable_set_foward_boundary_left(
                heading, velocity, minimumTurnRadius, time, thetaForwardLeft[i]
            )
        )
        pointsForwardRight.append(
            dubins_reachable_set_foward_boundary_right(
                heading, velocity, minimumTurnRadius, time, thetaForwardRight[i]
            )
        )
        pointsBackwardLeft.append(
            dubins_reachable_set_backward_boundary_left(
                heading, velocity, minimumTurnRadius, time, thetaBackwardLeft[i]
            )
        )
        pointsBackwardRight.append(
            dubins_reachable_set_backward_boundary_right(
                heading, velocity, minimumTurnRadius, time, thetaBackwardRight[i]
            )
        )

    pointsForwardLeft = np.array(pointsForwardLeft)
    pointsForwardRight = np.array(pointsForwardRight)
    pointsBackwardLeft = np.array(pointsBackwardLeft)
    pointsBackwardRight = np.array(pointsBackwardRight)
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

    plt.plot(xRight, yRight)
    plt.plot(xLeft, yLeft)
    plt.plot(pointsForwardLeft[:, 0], pointsForwardLeft[:, 1])
    plt.plot(pointsForwardRight[:, 0], pointsForwardRight[:, 1])
    plt.plot(pointsBackwardLeft[:, 0], pointsBackwardLeft[:, 1])
    plt.plot(pointsBackwardRight[:, 0], pointsBackwardRight[:, 1])

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
    point = np.array([2, 2])
    intersectionPointF, intersectionPointB = point_in_dubins_reachable_set(
        point, heading, velocity, minimumTurnRadius, time
    )

    # time = (6.0 / 3.0) * np.pi
    # time = 1
    ax = plot_dubins_reachable_set(heading, velocity, minimumTurnRadius, time)
    ax.scatter(*point, c="r")
    ax.scatter(*intersectionPointF, c="g")
    ax.scatter(*intersectionPointB, c="g")
    plt.show()


if __name__ == "__main__":
    main()
