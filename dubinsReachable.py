import numpy as np
import matplotlib.pyplot as plt


def dubins_reachable_set(heading, velocity, minimumTurnRadius, time, theta):
    """
    This computes the boundary (distance from origin) of the reachable set of a dubins vehicle using equations 10 and 11 from
    Cockayne, E. J., and G. W. C. Hall. "Plane motion of a particle subject to curvature constraints." SIAM Journal on Control 13.1 (1975): 197-220.
    """
    rho = 1 / minimumTurnRadius
    vt = velocity * time
    print(vt / rho)

    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    if 0 <= theta <= vt / rho:
        print(theta)
        x = rho * (1 - cosTheta) + (vt - rho * theta) * sinTheta
        y = rho * sinTheta + (vt - rho * theta) * cosTheta
    else:
        # x = rho * (2 * cosTheta - 1 - np.cos(2 * theta - vt / rho))
        # y = rho * (2 * sinTheta - np.sin(2 * theta - vt / rho))
        x = 0
        y = 0

    return np.array([x, y])


def plot_dubins_reachable_set(heading, velocity, minimumTurnRadius, time):
    numSamples = 100
    theta = np.linspace(0, 2 * np.pi, numSamples)
    print(theta)

    points = []

    for i in range(numSamples):
        points.append(
            dubins_reachable_set(heading, velocity, minimumTurnRadius, time, theta[i])
        )

    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1])
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    heading = 0
    velocity = 1
    minimumTurnRadius = np.pi
    time = 1
    plot_dubins_reachable_set(heading, velocity, minimumTurnRadius, time)
