import numpy as np
import matplotlib.pyplot as plt

from fast_pursuer import plotEngagementZone, inEngagementZone, inEngagementZoneJax


# def find_counter_clockwise_tangent_point(p1, c, r):
#     v2 = p1 - c
#
#     cosTheta = -r / np.linalg.norm(v2)
#
#     v2_normalized = v2 / np.linalg.norm(v2)
#
#     nx = v2_normalized[0] * cosTheta - v2_normalized[1] * np.sqrt(1 - cosTheta**2)
#     ny = v2_normalized[0] * np.sqrt(1 - cosTheta**2) + v2_normalized[1] * cosTheta
#
#     n = np.array([nx, ny])
#
#     pt = c - r * n
#     return pt


# def fing_clockwise_tangent_point(p1, c, r):
#     v2 = p1 - c
#     cosTheta = -r / np.linalg.norm(v2)
#
#     v2_normalized = v2 / np.linalg.norm(v2)
#     nx = v2_normalized[0] * cosTheta + v2_normalized[1] * np.sqrt(1 - cosTheta**2)
#     ny = v2_normalized[0] * np.sqrt(1 - cosTheta**2) - v2_normalized[1] * cosTheta
#     n = np.array([nx, ny])
#     pt = c - r * n
#     return pt


def find_counter_clockwise_tangent_point(p1, c, r):
    v1 = p1 - c
    normV1 = np.linalg.norm(v1)
    v3Perallel = -(r**2) / normV1**2 * v1
    vPerpendicularNormalized = np.array([-v1[1], v1[0]]) / normV1
    v3Perpendicular = np.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized

    v3 = v3Perallel + v3Perpendicular
    pt = c - v3
    return pt


def find_clockwise_tangent_point(p1, c, r):
    v1 = p1 - c
    normV1 = np.linalg.norm(v1)
    v3Perallel = -(r**2) / normV1**2 * v1
    vPerpendicularNormalized = np.array([v1[1], -v1[0]]) / normV1
    v3Perpendicular = np.sqrt(r**2 - r**4 / normV1**2) * vPerpendicularNormalized

    v3 = v3Perallel + v3Perpendicular
    pt = c - v3
    return pt


def temp_BEZ(
    startPosition,
    pursuerRange,
    captureRadius,
    pursuerSpeed,
    evaderSpeed,
    evaPosition,
    evaHeading,
):
    speedRatio = evaderSpeed / pursuerSpeed
    goalPosition = evaPosition + speedRatio * pursuerRange * np.array(
        [np.cos(evaHeading), np.sin(evaHeading)]
    )
    distanceToGoal = np.linalg.norm(goalPosition - startPosition)
    # testJax = inEngagementZoneJax(
    #     evaPosition,
    #     evaHeading,
    #     startPosition,
    #     pursuerRange,
    #     captureRadius,
    #     pursuerSpeed,
    #     evaderSpeed,
    # )
    # testNotJax = inEngagementZone(
    #     evaPosition,
    #     evaHeading,
    #     startPosition,
    #     pursuerRange,
    #     captureRadius,
    #     pursuerSpeed,
    #     evaderSpeed,
    # )
    # return testJax
    return distanceToGoal < (captureRadius + pursuerRange)


def find_dubins_path_length(startPosition, startHeading, goalPosition, radius):
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

    if np.linalg.norm(goalPosition - leftCenter) > np.linalg.norm(
        goalPosition - rightCenter
    ):
        tangentPoint = find_counter_clockwise_tangent_point(
            goalPosition, leftCenter, radius
        )
        centerPoint = leftCenter
    else:
        tangentPoint = find_clockwise_tangent_point(goalPosition, rightCenter, radius)
        centerPoint = rightCenter

    v4 = startPosition - centerPoint
    v3 = tangentPoint - centerPoint
    cosTheta = np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4))
    theta = np.arccos(cosTheta)

    straitLineLength = np.linalg.norm(goalPosition - tangentPoint)
    arcLength = radius * theta

    totalLength = arcLength + straitLineLength

    showPlot = False

    if showPlot:
        plt.figure()
        plt.scatter(*startPosition, c="g")
        plt.scatter(*goalPosition, c="r")
        plt.scatter(*leftCenter, c="b")
        plt.scatter(*rightCenter, c="b")
        theta = np.linspace(0, 2 * np.pi, 100)
        xl = leftCenter[0] + radius * np.cos(theta)
        yl = leftCenter[1] + radius * np.sin(theta)

        xr = rightCenter[0] + radius * np.cos(theta)
        yr = rightCenter[1] + radius * np.sin(theta)

        plt.plot(xl, yl, "b")
        plt.plot(xr, yr, "b")
        plt.scatter(*tangentPoint, c="y")
        plt.plot(
            [goalPosition[0], tangentPoint[0]], [goalPosition[1], tangentPoint[1]], "y"
        )
        ax = plt.gca()
        ax.set_aspect("equal", "box")
        plt.show()
    return totalLength


def in_dubins_engagement_zone(
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
    goalPosition = evaderPosition + speedRatio * pursuerRange * np.array(
        [np.cos(evaderHeading), np.sin(evaderHeading)]
    )

    dubinsPathLength = find_dubins_path_length(
        startPosition, startHeading, goalPosition, turnRadius
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
    Z = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = in_dubins_engagement_zone(
                startPosition,
                startHeading,
                turnRadius,
                captureRadius,
                pursuerRange,
                pursuerSpeed,
                np.array([X[i, j], Y[i, j]]),
                evaderHeading,
                evaderSpeed,
            )
            # Z[i, j] = temp_BEZ(
            #     startPosition,
            #     pursuerRange,
            #     captureRadius,
            #     pursuerSpeed,
            #     evaderSpeed,
            #     np.array([X[i, j], Y[i, j]]),
            #     0,
            # )

    plt.figure()
    c = plt.contour(X, Y, Z, levels=[0])
    ax = plt.gca()
    ax.set_aspect("equal", "box")
    plt.scatter(*startPosition, c="g")
    return ax


def main():
    startPosition = np.array([0, 0])
    startHeading = np.pi / 2
    turnRadius = 1.0
    captureRadius = 0.1
    pursuerRange = 1
    pursuerSpeed = 2
    evaderSpeed = 1
    agentHeading = 0

    # find_dubins_path_length(startPosition, startHeading, goalPosition, radius)
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
    plt.show()

    # p1 = np.array([3, 4])
    # c = np.array([1, -2])
    #
    # r = 0.5
    #
    # pt = find_counter_clockwise_tangent_point(p1, c, r)
    # pcw = find_clockwise_tangent_point(p1, c, r)
    #
    # #     # Draw the circle
    # theta = np.linspace(0, 2 * np.pi, 100)
    # x = c[0] + r * np.cos(theta)
    # y = c[1] + r * np.sin(theta)
    # plt.plot(x, y, "b")
    # plt.scatter(*c, c="r")
    # plt.scatter(*p1, c="g")
    # plt.scatter(*pt, c="y")
    # plt.scatter(*pcw, c="y")
    # plt.plot([pt[0], p1[0]], [pt[1], p1[1]], "r")
    # plt.plot([pcw[0], p1[0]], [pcw[1], p1[1]], "r")
    #
    # ax = plt.gca()
    # ax.set_aspect("equal", "box")
    # plt.show()


if __name__ == "__main__":
    main()
