import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class CircleArc:
    def __init__(self, startTheta, endTheta, radius, center, clockwise):
        print("startTheta: ", startTheta)
        print("endTheta: ", endTheta)
        self.startTheta = startTheta
        self.endTheta = endTheta
        self.radius = radius
        self.center = np.array(center)  # Ensure center is a NumPy array
        self.clockwise = clockwise

    def evaluate(self, numSamples):
        if self.clockwise:
            theta = np.linspace(self.startTheta, self.endTheta, numSamples)[
                ::-1
            ]  # Reverse order
        else:
            theta = np.linspace(self.startTheta, self.endTheta, numSamples)

        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)

        return np.vstack((x, y))  # Stack x and y as rows


def reachable_set(initialPose, minimumTurnRadius, maxRange):
    turnTheta = maxRange / minimumTurnRadius
    heading = initialPose[2]
    leftCenter = np.array(
        [
            initialPose[0] - minimumTurnRadius * np.sin(heading),
            initialPose[0] + minimumTurnRadius * np.cos(heading),
        ]
    )
    rightCenter = np.array(
        [
            initialPose[0] + minimumTurnRadius * np.sin(heading),
            initialPose[1] - minimumTurnRadius * np.cos(heading),
        ]
    )
    leftArc = CircleArc(
        heading - np.pi / 2,
        heading - np.pi / 2 + turnTheta,
        minimumTurnRadius,
        leftCenter,
        False,
    )
    print("right")
    rightArc = CircleArc(
        heading + np.pi / 2,
        heading + np.pi / 2 - turnTheta,
        minimumTurnRadius,
        rightCenter,
        True,
    )

    fig, ax = plt.subplots()
    # plot right circle
    ax.add_patch(Circle(rightCenter, minimumTurnRadius, fill=False))
    # plot left circle
    ax.add_patch(Circle(leftCenter, minimumTurnRadius, fill=False))
    rightCircle = rightArc.evaluate(100)
    leftCircle = leftArc.evaluate(100)
    ax.plot(rightCircle[0], rightCircle[1], "r")
    ax.plot(leftCircle[0], leftCircle[1], "r")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    initialPose = np.array([0, 0, np.pi / 2])
    minimumTurnRadius = 1
    maxRange = 10
    reachable_set(initialPose, minimumTurnRadius, maxRange)
