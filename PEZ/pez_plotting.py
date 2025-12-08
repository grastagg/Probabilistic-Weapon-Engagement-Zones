import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd, value_and_grad

from scipy.stats import norm
import time
from jax import jit
import jax
from jax import vmap

jax.config.update("jax_enable_x64", True)

# jax.config.update("jax_platform_name", "cpu")
# jax.default_device(jax.devices("cpu")[0])
# jax.config.update("jax_platform_name", "cpu")


from math import erf, sqrt
from math import erfc
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern
plt.rcParams["mathtext.rm"] = "serif"
# get rid of type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# set font size for title, axis labels, and legend, and tick labels
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10

import PEZ.pez as pez
import PEZ.bez as bez
import PLOT_COMMON.draw_mahalanobis as draw_mahalanobis


def plotEngagementZone(
    agentHeading,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
    ax,
    alpha=1.0,
    width=1,
    color="green",
):
    numPoints = 500
    y = np.linspace(-5, 5, numPoints)
    x = np.linspace(-5, 5, numPoints)
    [X, Y] = np.meshgrid(x, y)
    agentPositions = jnp.vstack([X.flatten(), Y.flatten()]).T
    agentHeadings = jnp.ones(agentPositions.shape[0]) * agentHeading
    engagementZonePlot = bez.inEngagementZoneJaxVectorized(
        agentPositions,
        agentHeadings,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
    )

    # engagementZonePlot = np.zeros(X.shape)

    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         engagementZonePlot[i, j] = inEngagementZone(np.array([[X[i, j]], [Y[i, j]]]), agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
    # c = plt.pcolormesh(X, Y, engagementZonePlot)
    # plt.colorbar(c, ax=ax)
    # c = plt.Circle([0,0], pursuerRange+pursuerCaptureRange, fill=False)
    # ax.add_artist(c)
    # plt.contour(X, Y, engagementZonePlot, levels=[0, 1], colors=['red'])
    ax.contour(
        X,
        Y,
        engagementZonePlot.reshape((numPoints, numPoints)),
        levels=[0],
        colors=[color],
        linewidths=width,
        alpha=alpha,
    )
    ax.plot([], color=color, label="BEZ", linewidth=width, alpha=alpha)
    # ax.scatter(pursuerPosition[0], pursuerPosition[1], color="red")
    # c = plt.Circle(pursuerPosition, pursuerRange, fill=False, color="black")
    # ax.add_artist(c)

    return


def plotProbablisticEngagementZone(
    agentPositionCov,
    agentHeading,
    agentHeadingVar,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerRangeVar,
    pursuerCaptureRange,
    pursuerCaptureRangeVar,
    pursuerSpeed,
    pursuerSpeedVar,
    agentSpeed,
    ax,
    levels=None,
):
    ax.set_aspect("equal")

    # Define the grid
    numPoints = 100
    x = jnp.linspace(-5, 5, numPoints)
    y = jnp.linspace(-5, 5, numPoints)
    X, Y = jnp.meshgrid(x, y)
    agentPositions = jnp.vstack([X.ravel(), Y.ravel()]).T
    agentHeadings = jnp.ones(numPoints * numPoints) * agentHeading

    # Compute Jacobian of engagement zone function
    # dPezDPursuerPosition = jacfwd(inEngagementZoneJax, argnums=2)
    # dPezDAgentPosition = jacfwd(inEngagementZoneJax, argnums=0)
    # dPezDPursuerRange = jacfwd(inEngagementZoneJax, argnums=3)
    # dPezDPursuerCaptureRange = jacfwd(inEngagementZoneJax, argnums=4)
    # dPezDAgentHeading = jacfwd(inEngagementZoneJax, argnums=1)

    start = time.time()
    # Compute engagement zone probabilities
    engagementZonePlot = pez.probabalisticEngagementZoneVectorizedTemp(
        agentPositions,
        agentPositionCov,
        agentHeadings,
        agentHeadingVar,
        pursuerPosition,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
    )
    print(engagementZonePlot.shape)
    print("total time: ", time.time() - start)

    # Convert result to NumPy array for plotting
    engagementZonePlot_np = np.array(engagementZonePlot)

    # Reshape for contour plotting
    engagementZonePlot_reshaped = engagementZonePlot_np.reshape(X.shape)

    # Plotting
    if levels is None:
        levels = np.linspace(0.1, 1, 10)
    c = ax.contour(X, Y, engagementZonePlot_reshaped, levels=levels, linewidths=2)
    inLine = False
    if inLine:
        ax.clabel(c, inline=True)
    else:
        handles, labels = c.legend_elements()
        labels = [f"$\\epsilon={lvl:.2f}$" for lvl in c.levels]

        ax.legend(
            handles,
            labels,
            title="PEZ Level",
            loc="lower right",
            framealpha=0.8,
        )

    # Add circle representing pursuer's range
    # for i in range(3):
    #     c = plt.Circle(pursuerPosition, pursuerRange + i*np.sqrt(pursuerRangeVar) + pursuerCaptureRange, fill=False, color='r', linestyle='--')
    #     ax.add_artist(c)
    # c = plt.Circle(pursuerPosition, pursuerRange, fill=False, color="black")
    # ax.add_artist(c)

    return engagementZonePlot_reshaped


def plotMCProbablisticEngagementZone(
    agentPositionCov,
    agentHeading,
    agentHeadingVar,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerRangeCov,
    pursuerCaptureRange,
    pursuerCaptureRangeVar,
    pursuerSpeed,
    pursuerSpeedVar,
    agentSpeed,
    ax,
):
    ax.set_aspect("equal")
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    [X, Y] = np.meshgrid(x, y)

    malhalanobisDistance = np.zeros(X.shape)
    engagementZonePlot = np.zeros(X.shape)

    numMcTrials = 2000
    # numMcTrials = 20

    pursuerPositionSamples = np.random.multivariate_normal(
        pursuerPosition.squeeze(), pursuerPositionCov, numMcTrials
    )
    pursuerRangeSamples = np.random.normal(
        pursuerRange, np.sqrt(pursuerRangeCov), numMcTrials
    )
    pursuerCaptureRangeSamples = np.random.normal(
        pursuerCaptureRange, np.sqrt(pursuerCaptureRangeVar), numMcTrials
    )
    agentHeadingSamples = np.random.normal(
        agentHeading, np.sqrt(agentHeadingVar), numMcTrials
    )
    pursuerSpeedSamples = np.random.normal(
        pursuerSpeed, np.sqrt(pursuerSpeedVar), numMcTrials
    )

    for i in range(X.shape[0]):
        print(i)
        for j in range(X.shape[1]):
            # engagementZonePlot[i, j] = probabalisticEngagementZone(np.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
            agentPositionSamples = np.random.multivariate_normal(
                np.array([[X[i, j]], [Y[i, j]]]).squeeze(),
                agentPositionCov,
                numMcTrials,
            )
            engagementZonePlot[i, j] = pez.monte_carlo_probalistic_engagment_zone(
                np.array([[X[i, j]], [Y[i, j]]]),
                agentHeading,
                pursuerPosition,
                pursuerPositionCov,
                pursuerRange,
                pursuerCaptureRange,
                pursuerSpeed,
                agentSpeed,
                numMcTrials,
                pursuerPositionSamples,
                pursuerRangeSamples,
                pursuerCaptureRangeSamples,
                agentHeadingSamples,
                pursuerSpeedSamples,
                agentPositionSamples,
            )
    # c = ax.pcolormesh(X, Y, engagementZonePlot)
    # c = plt.Circle(pursuerPosition, pursuerRange+pursuerCaptureRange, fill=False)
    # ax.add_artist(c)
    c = ax.contour(
        X, Y, engagementZonePlot, levels=np.linspace(0.1, 1, 10), linewidths=2
    )
    ax.clabel(c, inline=True)
    ax.add_artist(c)

    return engagementZonePlot


def plot_abs_diff(linPez, mcPez, ax, fig):
    absdiff = np.abs(linPez - mcPez)

    x = jnp.linspace(-2, 2, 50)
    y = jnp.linspace(-2, 2, 50)
    X, Y = jnp.meshgrid(x, y)
    c = ax.pcolormesh(X, Y, absdiff.reshape(X.shape), cmap="viridis")
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Absolute Difference")

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Linear vs MC PEZ")


# probabalisticEngagementZone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, np.array([[0.1, 0], [0, 0.1]]), pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
# inEngagementZone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)


def plot_potential_BEZS():
    pursuerRange = 1.0
    pursuerRangeVar = 0.1
    pursuerCaptureRange = 0.1
    pursuerCaptureRangeVar = 0.02
    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.0
    agentSpeed = 0.5

    agentPositionCov = np.array([[0.0, 0.0], [0.0, 0.0]])
    pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
    pursuerInitialPosition = np.array([0.0, 0.0])

    agentInitialHeading = 0.0
    agentHeadingVar = 0.0

    numPotentialBEZS = 500
    fig, ax = plt.subplots(1, 1)
    plotEngagementZone(
        agentInitialHeading,
        pursuerInitialPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
        ax,
        alpha=1.0,
        width=2.5,
    )
    draw_mahalanobis.plotMahalanobisDistance(
        pursuerInitialPosition, pursuerPositionCov, ax, fig, plotColorbar=False
    )
    for i in range(numPotentialBEZS):
        potentialPursuerPosition = np.random.multivariate_normal(
            pursuerInitialPosition, pursuerPositionCov
        )
        plotEngagementZone(
            agentInitialHeading,
            potentialPursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
            ax,
            alpha=0.2,
        )
        ax.scatter(
            potentialPursuerPosition[0], potentialPursuerPosition[1], color="red"
        )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    ax.scatter(*pursuerInitialPosition, color="darkred", s=80)

    ax.set_aspect("equal")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    plt.show()


def main():
    pursuerRange = 1.0
    pursuerRangeVar = 0.1
    pursuerCaptureRange = 0.1
    pursuerCaptureRangeVar = 0.02
    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.0
    agentSpeed = 0.5

    agentPositionCov = np.array([[0.0, 0.0], [0.0, 0.0]])
    pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
    pursuerInitialPosition = np.array([0.0, 0.0])

    agentInitialHeading = 0.0
    agentHeadingVar = 0.0

    fig, ax = plt.subplots(1, 2)
    mcAx = ax[1]
    if np.any(pursuerPositionCov):
        draw_mahalanobis.plotMahalanobisDistance(
            pursuerInitialPosition, pursuerPositionCov, mcAx, fig
        )
    mcEz = plotMCProbablisticEngagementZone(
        agentPositionCov,
        agentInitialHeading,
        agentHeadingVar,
        pursuerInitialPosition,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
        mcAx,
    )
    # plotEngagementZone(
    #     agentInitialHeading,
    #     pursuerInitialPosition.squeeze(),
    #     pursuerRange,
    #     pursuerCaptureRange,
    #     pursuerSpeed,
    #     agentSpeed,
    #     mcAx,
    # )
    mcAx.set_xlabel("X")
    mcAx.set_ylabel("Y")
    mcAx.set_aspect("equal")
    mcAx.set_title("Monte Carlo PEZ")
    # linFig, linAx = plt.subplots(1, 1)
    linAx = ax[0]
    if np.any(pursuerPositionCov):
        draw_mahalanobis.plotMahalanobisDistance(
            pursuerInitialPosition, pursuerPositionCov, linAx, fig, plotColorbar=False
        )

    linAx.set_aspect("equal")
    linPez = plotProbablisticEngagementZone(
        agentPositionCov,
        agentInitialHeading,
        agentHeadingVar,
        pursuerInitialPosition,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
        linAx,
    )
    # plotEngagementZone(
    #     agentInitialHeading,
    #     pursuerInitialPosition,
    #     pursuerRange,
    #     pursuerCaptureRange,
    #     pursuerSpeed,
    #     agentSpeed,
    #     linAx,
    # )
    linAx.set_xlabel("X")
    linAx.set_ylabel("Y")
    linAx.set_title("Linearized PEZ")
    fig.set_size_inches(20, 20)
    # plt.savefig("/home/ggs24/Desktop/PEZ.png", dpi=500, bbox_inches="tight")
    fig, ax = plt.subplots(1, 1)
    # plot_abs_diff(linPez, mcEz, ax, fig)
    plt.show()
    #
    # fig,axes = plt.subplots(2,4,layout='constrained')
    #
    #
    # for case in range(4):
    #     if case == 0:
    #         pursuerPositionCov = np.array([[0.2, 0.0], [0.0, 0.2]])
    #
    #     elif case == 1:
    #         pursuerPositionCov = np.array([[0.0, 0.0], [0.0, 0.0]])
    #         pursuerRangeVar = 0.2
    #     elif case == 2:
    #         pursuerRangeVar = 0.0
    #         pursuerCaptureRangeVar = 0.02
    #     elif case == 3:
    #         pursuerPositionCov = np.array([[0.2, 0.0], [0.0, 0.2]])
    #         pursuerRangeVar = 0.2
    #         pursuerCaptureRangeVar = 0.02
    #
    #
    #     mcAx = axes[1][case]
    #     if np.any(pursuerPositionCov):
    #       drawMahlonobis.plotMahalanobisDistance(pursuerInitialPosition, pursuerPositionCov, mcAx)
    # # mcFig,mcAx = plt.subplots(1,1)
    #     mcEz = plotMCProbablisticEngagementZone(agentPositionCov,agentInitialHeading,agentHeadingVar, pursuerInitialPosition, pursuerPositionCov, pursuerRange,pursuerRangeVar, pursuerCaptureRange,pursuerCaptureRangeVar, pursuerSpeed,pursuerSpeedVar, agentSpeed, mcAx)
    #     plotEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,mcAx)
    #     mcAx.set_xlabel("X",fontsize = 16)
    #     mcAx.set_ylabel("Y",fontsize = 16)
    #     mcAx.set_aspect('equal')
    #
    #
    #
    #
    #     # linFig,linAx = plt.subplots(1,1)
    #     linAx = axes[0][case]
    #     if np.any(pursuerPositionCov):
    #        drawMahlonobis.plotMahalanobisDistance(pursuerInitialPosition, pursuerPositionCov, linAx)
    #
    #     linAx.set_aspect('equal')
    #     linPez = plotProbablisticEngagementZone(agentPositionCov,agentInitialHeading,agentHeadingVar, pursuerInitialPosition, pursuerPositionCov, pursuerRange, pursuerRangeVar, pursuerCaptureRange,pursuerCaptureRangeVar, pursuerSpeed,pursuerSpeedVar, agentSpeed,linAx)
    #     plotEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,linAx)
    #     linAx.set_xlabel("X",fontsize = 16)
    #     linAx.set_ylabel("Y",fontsize = 16)
    #
    #
    #
    #     # mse = np.sqrt(np.mean((mcEz - linPez)**2))
    #     # print(mse)
    #     # absPercentError = np.mean(np.abs(mcEz - linPez))
    #     # print("Absolute Percent Error: ", absPercentError)
    #
    #
    # cols = ['Case 1', 'Case 2', 'Case 3', 'Case 4']
    # rows = ['Linearized', 'Monte Carlo']
    # for ax, col in zip(axes[0], cols):
    #     ax.set_title(col,fontsize = 20)
    # for ax, row in zip(axes[:,0], rows):
    #     ax.set_ylabel(row, rotation=90,fontsize = 20)
    #
    # # plt.tight_layout()
    # plt.show()
    #


if __name__ == "__main__":
    main()
    # plot_potential_BEZS()
