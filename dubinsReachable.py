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
    print("equation", equation(theta_solutions))
    theta_solutions = theta_solutions[
        (theta_solutions >= 0) & (theta_solutions <= 2 * np.pi)
    ]
    return theta_solutions


def plot_dubins_reachable_set(heading, velocity, minimumTurnRadius, time):
    numSamples = 100
    print(velocity * time * minimumTurnRadius)
    theta = np.linspace(
        -velocity * time * minimumTurnRadius,
        velocity * time * minimumTurnRadius,
        numSamples,
    )
    maxTheta = find_theta_backward_boundary(heading, velocity, minimumTurnRadius, time)
    print("maxTheta", maxTheta)
    print(
        "backward",
        dubins_reachable_set_backward_boundary_right(
            heading, velocity, minimumTurnRadius, time, maxTheta[0]
        ),
    )
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


if __name__ == "__main__":
    heading = 0
    velocity = 1
    minimumTurnRadius = 1
    pursuerRange = np.pi
    time = pursuerRange / velocity
    print("time", time)
    point = np.array([2, 2])
    # time = (6.0 / 3.0) * np.pi
    # time = 1
    ax = plot_dubins_reachable_set(heading, velocity, minimumTurnRadius, time)
    plt.show()
