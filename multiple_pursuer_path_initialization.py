import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from wevo_py import weighted_voronoi_diagram


import dubinsEZ
import fast_pursuer

import weighted_voronoi


shortestest_dubins_path_multiple_pursuers = jax.vmap(
    dubinsEZ.vectorized_find_shortest_dubins_path, in_axes=(0, 0, None, 0)
)
np.random.seed(1000)


def max_ez_single(
    goalPoisition,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerTurnRadius,
    pursuerSpeed,
    pursuerRange,
):
    # evaderHeading points straight at pursuer
    evaderHeading = jnp.arctan2(
        pursuerPosition[1] - goalPoisition[1], pursuerPosition[0] - goalPoisition[0]
    )
    evaderHeading = 0.0
    return dubinsEZ.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        pursuerTurnRadius,
        0.0,
        pursuerRange,
        pursuerSpeed,
        goalPoisition,
        evaderHeading,
        evaderSpeed,
    )


max_ez_multiple = jax.vmap(
    max_ez_single, in_axes=(0, None, None, None, None, None, None)
)

max_ez_multiple_pursuers = jax.vmap(
    max_ez_multiple, in_axes=(None, None, 0, 0, 0, 0, 0)
)


def discratized_dubins_voronoi_diagram(
    pursuerPositionList,
    pursuerHeadingList,
    pursuerTurnRadiusList,
    pursuerSpeedList,
    pursuerRangeList,
):
    numPoints = 1000
    x = np.linspace(0, 10, numPoints)
    y = np.linspace(0, 10, numPoints)
    [X, Y] = np.meshgrid(x, y)
    positions = np.array([X.flatten(), Y.flatten()]).T
    # distances = shortestest_dubins_path_multiple_pursuers(
    #     pursuerPositionList, pursuerHeadingList, positions, pursuerTurnRadiusList
    # )
    distances = np.linalg.norm(
        positions[:, np.newaxis, :] - pursuerPositionList[np.newaxis, :, :], axis=-1
    )

    print(pursuerSpeedList)
    times = distances / pursuerSpeedList[np.newaxis, :]

    # maxEZ = max_ez_multiple_pursuers(
    #     positions,
    #     0.5,
    #     pursuerPositionList,
    #     pursuerHeadingList,
    #     pursuerTurnRadiusList,
    #     pursuerSpeedList,
    #     pursuerRangeList,
    # )
    # print(maxEZ)
    # cellAssignment = np.argmin(maxEZ, axis=0)
    cellAssignment = np.argmin(times, axis=1)
    # print(cellAssignment.shape)
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
    # plot maxEZ
    # for i in range(len(pursuerPositionList)):
    #     dubinsEZ.plot_dubins_EZ(
    #         pursuerPositionList[i],
    #         pursuerHeadingList[i],
    #         pursuerSpeedList[i],
    #         pursuerTurnRadiusList[i],
    #         0.0,
    #         pursuerRangeList[i],
    #         0.0,
    #         0.5,
    #         ax,
    #     )
    arcs, boundarySegments = pursuer_weighted_voronoi(
        pursuerPositionList, pursuerSpeedList
    )

    weighted_voronoi.plot_weighted_voronoi_arcs(arcs, boundarySegments, ax)

    plt.show()


def pursuer_weighted_voronoi(purusuerPositions, pursuerSpeeds):
    print(pursuerSpeeds)
    weights = pursuerSpeeds
    weighted_voronoi.save_points_and_weights_to_file(
        purusuerPositions, weights, "my_input.pnts"
    )

    weighted_voronoi.weighted_voronoi_diagram()

    filename = "output.txt"

    arcs = weighted_voronoi.load_weighted_voronoi_segments_from_file(filename)

    arcs = weighted_voronoi.combine_attached_arcs(arcs)

    arcs, boundarySegments = weighted_voronoi.intersect_arcs_with_boundary(
        arcs, [10, 10]
    )
    boundarySegments = np.array(boundarySegments)
    return arcs, boundarySegments


def main():
    numPursuers = 5
    # pursuerPositions = np.array([[0, 0], [-4, 1], [6, 2], [-6, -7]])
    bounds = [10, 10]
    pursuerPositions = np.random.uniform(0, 10, (numPursuers, 2))
    # pursuerHeadings = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    pursuerHeadings = np.random.uniform(-np.pi, np.pi, numPursuers)
    # pursuerTurnRadius = np.array([0.2, 1.5, 0.75, 1])
    maxTurnRadius = 0.5
    pursuerTurnRadius = np.random.uniform(0.1, maxTurnRadius, numPursuers)
    # pursuerTurnRadius = np.ones(numPursuers) * 0.5
    pursuerSpeeds = np.random.uniform(1.0, 2, numPursuers)

    minRange = (3 / 2) * np.pi * maxTurnRadius
    pursuerRanges = np.random.uniform(minRange, 1.2 * minRange, numPursuers)
    # pursuerRanges = pursuerTurnRadius * (3 / 2) * np.pi
    discratized_dubins_voronoi_diagram(
        pursuerPositions,
        pursuerHeadings,
        pursuerTurnRadius,
        pursuerSpeeds,
        pursuerRanges,
    )


if __name__ == "__main__":
    main()
