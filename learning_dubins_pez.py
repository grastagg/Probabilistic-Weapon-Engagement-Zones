import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import dubinsEZ


def plot_low_priority_paths(
    startPositions, headings, speeds, dt, interceptedList, endPoints, endTimes
):
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    for i in range(len(startPositions)):
        numPoints = int(endTimes[i] / dt)
        pathHistory = np.linspace(startPositions[i], endPoints[i], numPoints)
        color = "green" if not interceptedList[i] else "red"
        ax.plot(pathHistory[:, 0], pathHistory[:, 1], c=color)
        print(interceptedList[i])
        if interceptedList[i]:
            ax.scatter(
                endPoints[i][0],
                endPoints[i][1],
                color="red",
                marker="x",
            )


def is_inside_region(point, xbound, ybounds):
    return xbound[0] <= point[0] <= xbound[1] and ybounds[0] <= point[1] <= ybounds[1]


def send_low_priority_agent(
    startPosition,
    heading,
    speed,
    dt,
    ybounds,
    xbounds,
    pursuerPosition,
    pursuerHeading,
    minimumTurnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
):
    currentPosition = jnp.array(startPosition)
    currentTime = 0.0

    intercepted = False

    interceptionPoint = None
    interceptionTime = None

    while not intercepted and is_inside_region(currentPosition, ybounds, xbounds):
        # check if agent is inside engagement zone
        inEZ = (
            dubinsEZ.in_dubins_engagement_zone_single(
                pursuerPosition,
                pursuerHeading,
                minimumTurnRadius,
                captureRadius,
                pursuerRange,
                pursuerSpeed,
                currentPosition,
                heading,
                speed,
            )
            < 0
        )
        if inEZ:
            intercepted = True
            speedRatio = speed / pursuerSpeed

            # Compute goal positions
            direction = jnp.array(
                [jnp.cos(heading), jnp.sin(heading)]
            )  # Heading unit vector
            interceptionPoint = currentPosition + speedRatio * pursuerRange * direction
            interceptionTime = currentTime + pursuerRange / speed
            print(
                "agent will be intercepted at",
                interceptionPoint,
                "at time",
                interceptionTime,
            )

        # Update position
        currentPosition = (
            currentPosition
            + jnp.array([jnp.cos(heading), jnp.sin(heading)]) * speed * dt
        )
        currentTime += dt
    endTime = currentTime
    if not intercepted:
        # If not intercepted, set interception point to the last position
        interceptionPoint = currentPosition

    return intercepted, interceptionPoint, endTime


def sample_entry_point_and_heading(xBounds, yBounds):
    """
    Sample a random entry point on the boundary of a rectangular region,
    and generate a heading that points into the region.

    Args:
        region_bounds: dict with keys 'xmin', 'xmax', 'ymin', 'ymax'

    Returns:
        start_pos: np.array of shape (2,), the boundary point
        heading: float, angle in radians pointing into the region
    """
    xmin, xmax = xBounds
    ymin, ymax = yBounds

    edge = np.random.choice(["left", "right", "top", "bottom"])

    if edge == "left":
        x = xmin
        y = np.random.uniform(ymin, ymax)
    elif edge == "right":
        x = xmax
        y = np.random.uniform(ymin, ymax)
    elif edge == "bottom":
        x = np.random.uniform(xmin, xmax)
        y = ymin
    elif edge == "top":
        x = np.random.uniform(xmin, xmax)
        y = ymax

    start_pos = np.array([x, y], dtype=np.float32)

    # Compute angle toward origin
    direction_to_center = -start_pos  # vector from point to (0,0)
    heading = np.arctan2(direction_to_center[1], direction_to_center[0])

    # Add Gaussian noise to heading
    heading_noise_std = 0.5  # Standard deviation of noise
    heading += np.random.normal(0.0, heading_noise_std)

    return start_pos, heading


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerHeading = (10.0 / 20.0) * np.pi
    pursuerRange = 1.0
    pursuerCaptureRadius = 0.0
    pursuerSpeed = 2.0
    pursuerTurnRadius = 0.2
    agentSpeed = 1
    xbounds = (-2.0, 2.0)
    ybounds = (-2.0, 2.0)

    tmax = np.sqrt(xbounds[1] ** 2 + ybounds[1] ** 2) / pursuerSpeed

    numLowPriorityAgents = 10

    interceptedList = []
    endPoints = []
    endTimes = []
    startPositions = []
    headings = []
    speeds = []

    for _ in range(numLowPriorityAgents):
        startPosition, heading = sample_entry_point_and_heading(xbounds, ybounds)
        startPositions.append(startPosition)
        headings.append(heading)
        speeds.append(agentSpeed)
        intercepted, endPoint, endTime = send_low_priority_agent(
            startPosition,
            heading,
            agentSpeed,
            0.1,
            ybounds,
            xbounds,
            pursuerPosition,
            pursuerHeading,
            pursuerTurnRadius,
            pursuerCaptureRadius,
            pursuerRange,
            pursuerSpeed,
        )
        interceptedList.append(intercepted)
        endPoints.append(endPoint)
        endTimes.append(endTime)

    plot_low_priority_paths(
        startPositions,
        headings,
        speeds,
        0.1,
        interceptedList,
        endPoints,
        endTimes,
    )


if __name__ == "__main__":
    main()
    plt.show()
