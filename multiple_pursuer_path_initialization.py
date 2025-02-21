import jax
import numpy as np
import matplotlib.pyplot as plt


import dubinsEZ
import fast_pursuer


shortestest_dubins_path_multiple_pursuers = jax.vmap(
    dubinsEZ.vectorized_find_shortest_dubins_path, in_axes=(0, 0, None, 0)
)


def discratized_dubins_voronoi_diagram(
    pursuerPositionList, pursuerHeadingList, pursuerTurnRadiusList, pursuerSpeedList
):
    numPoints = 1000
    x = np.linspace(-10, 10, numPoints)
    y = np.linspace(-10, 10, numPoints)
    [X, Y] = np.meshgrid(x, y)
    positions = np.array([X.flatten(), Y.flatten()]).T
    distances = shortestest_dubins_path_multiple_pursuers(
        pursuerPositionList, pursuerHeadingList, positions, pursuerTurnRadiusList
    )
    times = distances / pursuerSpeedList[:, np.newaxis]
    print(distances.shape)
    cellAssignment = np.argmin(times, axis=0)
    print(cellAssignment.shape)
    fig, ax = plt.subplots()
    ax.pcolormesh(
        X.reshape(numPoints, numPoints),
        Y.reshape(numPoints, numPoints),
        cellAssignment.reshape(numPoints, numPoints),
    )
    for i in range(len(pursuerPositionList)):
        dubinsEZ.plot_turn_radius_circles(
            pursuerPositionList[i], pursuerHeadingList[i], pursuerTurnRadiusList[i], ax
        )
        # plot heading vector
        ax.quiver(
            pursuerPositionList[i][0],
            pursuerPositionList[i][1],
            0.2 * np.cos(pursuerHeadingList[i]),
            0.2 * np.sin(pursuerHeadingList[i]),
            scale=5,
            color="black",
        )
    ax.set_aspect("equal")
    plt.show()


def main():
    numPursuers = 15
    # pursuerPositions = np.array([[0, 0], [-4, 1], [6, 2], [-6, -7]])
    pursuerPositions = np.random.uniform(-10, 10, (numPursuers, 2))
    # pursuerHeadings = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    pursuerHeadings = np.random.uniform(-np.pi, np.pi, numPursuers)
    # pursuerTurnRadius = np.array([0.2, 1.5, 0.75, 1])
    pursuerTurnRadius = np.random.uniform(0.5, 1, numPursuers)
    # pursuerTurnRadius = np.ones(numPursuers) * 0.5
    # pursuerSpeeds = np.random.uniform(1.5, 2, numPursuers)
    pursuerSpeeds = np.ones(numPursuers) * 1.5
    discratized_dubins_voronoi_diagram(
        pursuerPositions, pursuerHeadings, pursuerTurnRadius, pursuerSpeeds
    )


if __name__ == "__main__":
    main()
