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


# pursuerRange = .8
# pursuerCaptureRange = 0.1
# pursuerSpeed = 1
# agentSpeed = .9


def inEngagementZone(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    rotationMinusHeading = np.array(
        [
            [np.cos(agentHeading), np.sin(agentHeading)],
            [-np.sin(agentHeading), np.cos(agentHeading)],
        ]
    )
    pursuerPositionHat = rotationMinusHeading @ (pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading @ (agentPosition - pursuerPosition)

    speedRatio = agentSpeed / pursuerSpeed
    # print("agentPosition not jax: ", agentPosition)
    # print("pursuerPosition not jax: ", pursuerPosition)
    distance = np.linalg.norm(agentPosition - pursuerPosition)
    # epsilon = agentHeading + np.arctan2(agentPosition[1] - pursuerPosition[1], pursuerPosition[0] - agentPosition[0])
    # epsilon = np.arctan2(agentPositionHat[1] - pursuerPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    epsilon = np.arctan2(
        pursuerPositionHat[1] - agentPositionHat[1],
        pursuerPositionHat[0] - agentPositionHat[0],
    )
    # print(epsilon)
    rho = (
        speedRatio
        * pursuerRange
        * (
            np.cos(epsilon)
            + np.sqrt(
                np.cos(epsilon) ** 2
                - 1
                + (pursuerRange + pursuerCaptureRange) ** 2
                / (speedRatio**2 * pursuerRange**2)
            )
        )
    )

    # return distance < rho
    # print("rho not jax: ", rho)
    # print("distance not jax: ", distance)
    return distance - rho
    # return rho


def plotMahalanobisDistance(
    pursuerPosition, pursuerPositionCov, ax, fig, plotColorbar=False, cax=None
):
    # Define the grid
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Stack X, Y coordinates into a single array
    points = np.stack([X.ravel(), Y.ravel()]).T

    # Compute the inverse of the covariance matrix
    inv_cov = np.linalg.inv(pursuerPositionCov)

    # Compute Mahalanobis distance for each point (vectorized)
    delta = points - pursuerPosition.T
    malhalanobisDistance = np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta))
    malhalanobisDistance = malhalanobisDistance.reshape(X.shape)

    # Specify darker shades of red
    colors = ["#CC0000", "#FF6666", "#FFCCCC"]

    # Plot filled contours for Mahalanobis distance with darker red shades
    c = ax.contourf(
        X,
        Y,
        malhalanobisDistance,
        levels=[0, 1, 2, 3],
        colors=colors,
        alpha=0.75,
    )
    # c = ax.pcolormesh(X, Y, malhalanobisDistance)

    # Mark the pursuer position with a dark red dot
    # ax.scatter(
    #     pursuerPosition[0],
    #     pursuerPosition[1],
    #     color="darkred",
    # )

    # Add a color bar and increase font size
    if plotColorbar:
        divider = make_axes_locatable(ax)  # ax_nn is the last subplot axis
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(c, cax=cax)
        # l, b, w, h = ax.get_position().bounds
        # cax = fig.add_axes([l + w + 0.02, b, 0.02, h])  # new colorbar axis
        cbar = plt.colorbar(c, cax=cax, ticks=[0, 1, 2, 3], shrink=0.5)
        cbar.set_label("Pursuer Std Dev")
        cbar.ax.tick_params()
    # if plotColorbar:
    #     l, b, w, h = ax.get_position().bounds
    #     cax = fig.add_axes([l + w + 0.02, b, 0.02, h])
    #     # cbar = fig.colorbar(c, ax=cax, cax=cax, ticks=[0, 1, 2, 3], shrink=0.5)
    #     cbar = plt.colorbar(c, ax=cax, ticks=[0, 1, 2, 3])
    #     cbar.set_label("Pursuer Std Dev", fontsize=26)


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
    engagementZonePlot = inEngagementZoneJaxVectorized(
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


def analytic_epsilonJac(agentPositionHat, pursuerPositionHat):
    y_p = pursuerPositionHat[1]
    x_p = pursuerPositionHat[0]
    y_a = agentPositionHat[1]
    x_a = agentPositionHat[0]
    epsilonJac = np.zeros((1, 2))
    epsilonJac[0][0] = (
        -(y_p - y_a) / ((x_p - x_a) ** 2 * ((y_p - y_a) ** 2 / (x_p - x_a) ** 2 + 1))
    )[0]
    epsilonJac[0][1] = (1 / ((x_p - x_a) * ((y_p - y_a) ** 2 / (x_p - x_a) ** 2 + 1)))[
        0
    ]

    return epsilonJac.squeeze()


def analytic_rhoJac(epsilon, speedRatio, pursuerRange, pursuerCaptureRange):
    return (
        pursuerRange
        * speedRatio
        * (
            -(np.cos(epsilon) * np.sin(epsilon))
            / np.sqrt(
                np.cos(epsilon) ** 2
                + (pursuerCaptureRange + pursuerRange) ** 2
                / (pursuerRange**2 * speedRatio**2)
                - 1
            )
            - np.sin(epsilon)
        )
    ).squeeze()


def analytic_distJac(agentPosition, pursuerPosition):
    return -(agentPosition - pursuerPosition) / np.linalg.norm(
        agentPosition - pursuerPosition
    )


def probabalisticEngagementZone(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    start = time.time()
    rotationMinusHeading = np.array(
        [
            [np.cos(agentHeading), np.sin(agentHeading)],
            [-np.sin(agentHeading), np.cos(agentHeading)],
        ]
    )

    pursuerPositionHat = rotationMinusHeading @ (pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading @ (agentPosition - pursuerPosition)
    epsilon = np.arctan2(
        pursuerPositionHat[1] - agentPositionHat[1],
        pursuerPositionHat[0] - agentPositionHat[0],
    )

    epsilonJac = analytic_epsilonJac(agentPositionHat, pursuerPositionHat)
    speedRatio = agentSpeed / pursuerSpeed

    rho = (
        speedRatio
        * pursuerRange
        * (
            np.cos(epsilon)
            + np.sqrt(
                np.cos(epsilon) ** 2
                - 1
                + (pursuerRange + pursuerCaptureRange) ** 2
                / (speedRatio**2 * pursuerRange**2)
            )
        )
    )

    rhoJac = analytic_rhoJac(epsilon, speedRatio, pursuerRange, pursuerCaptureRange)

    dist = np.linalg.norm(agentPosition - pursuerPosition)
    distJac = analytic_distJac(agentPosition, pursuerPosition)

    overallJac = (distJac).reshape((-1, 1)).squeeze() - (
        rotationMinusHeading.T @ epsilonJac * rhoJac
    ).squeeze()

    mean = dist - rho
    cov = overallJac @ pursuerPositionCov @ overallJac.T

    diffDistribution = norm(mean, np.sqrt(cov))

    return diffDistribution.cdf(0)


@jit
def inEngagementZoneJax(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    rotationMinusHeading = jnp.array(
        [
            [jnp.cos(agentHeading), jnp.sin(agentHeading)],
            [-jnp.sin(agentHeading), jnp.cos(agentHeading)],
        ]
    )
    pursuerPositionHat = rotationMinusHeading @ (pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading @ (agentPosition - pursuerPosition)

    speedRatio = agentSpeed / pursuerSpeed
    # jax.debug.print("agentPosition jax: {x}", x=agentPosition)
    # jax.debug.print("pursuerPosition jax: {x}", x=pursuerPosition)
    distance = jnp.linalg.norm(agentPosition - pursuerPosition)
    # epsilon = agentHeading + np.arctan2(agentPosition[1] - pursuerPosition[1], pursuerPosition[0] - agentPosition[0])
    # epsilon = np.arctan2(agentPositionHat[1] - pursuerPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    epsilon = jnp.arctan2(
        pursuerPositionHat[1] - agentPositionHat[1],
        pursuerPositionHat[0] - agentPositionHat[0],
    )
    # print(epsilon)
    rho = (
        speedRatio
        * pursuerRange
        * (
            jnp.cos(epsilon)
            + jnp.sqrt(
                jnp.cos(epsilon) ** 2
                - 1
                + (pursuerRange + pursuerCaptureRange) ** 2
                / (speedRatio**2 * pursuerRange**2)
            )
        )
    )

    # return distnce < rho
    # jax.debug.print("rho jax: {x}", x=rho)
    # jax.debug.print("distance jax: {x}", x=distance)
    # return distance - rho[0]
    return distance - rho


def inEngagementZoneJaxVectorized(
    agentPositions,
    agentHeadings,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    single_agent_prob = lambda agentPosition, agentHeading: inEngagementZoneJax(
        agentPosition,
        agentHeading,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
    )
    # agentPosition = agentPosition.reshape(-1, 1)
    return vmap(single_agent_prob)(agentPositions, agentHeadings)


# @jit
def probabalisticEngagementZoneTemp(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
    dPezDPursuerPosition,
):
    mean = inEngagementZoneJax(
        agentPosition,
        agentHeading,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
    ).squeeze()
    dPezDPursuerPositionJac = dPezDPursuerPosition(
        agentPosition,
        agentHeading,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
    ).squeeze()
    cov = dPezDPursuerPositionJac @ pursuerPositionCov @ dPezDPursuerPositionJac.T
    diffDistribution = norm(mean, np.sqrt(cov))
    return diffDistribution.cdf(0)


dPezDPursuerPosition = jacfwd(inEngagementZoneJax, argnums=2)
dPezDAgentPosition = jacfwd(inEngagementZoneJax, argnums=0)
dPezDPursuerRange = jacfwd(inEngagementZoneJax, argnums=3)
dPezDPursuerCaptureRange = jacfwd(inEngagementZoneJax, argnums=4)
dPezDAgentHeading = jacfwd(inEngagementZoneJax, argnums=1)
dPezDPusuerSpeed = jacfwd(inEngagementZoneJax, argnums=5)


def probabalisticEngagementZoneVectorizedTemp(
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
    pusuerSpeedVar,
    agentSpeed,
):
    # Define vectorized operations with vmap
    def single_agent_prob(agentPosition, agentHeading):
        agentPosition = agentPosition.reshape(
            -1,
        )

        # Calculate the mean for the engagement zone
        mean = inEngagementZoneJax(
            agentPosition,
            agentHeading,
            pursuerPosition.squeeze(),
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()

        # Calculate the Jacobian for the engagement zone
        dPezDPursuerPositionJac = dPezDPursuerPosition(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDAgentPositionJac = dPezDAgentPosition(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDPursuerRangeJac = dPezDPursuerRange(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDPursuerCaptureRangeJac = dPezDPursuerCaptureRange(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDAgentHeadingJac = dPezDAgentHeading(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDPursuerSpeedJac = dPezDPusuerSpeed(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()

        # Compute the covariance matrix
        cov = (
            dPezDPursuerPositionJac @ pursuerPositionCov @ dPezDPursuerPositionJac.T
            + dPezDAgentPositionJac @ agentPositionCov @ dPezDAgentPositionJac.T
            + dPezDPursuerRangeJac**2 * pursuerRangeVar
            + dPezDPursuerCaptureRangeJac**2 * pursuerCaptureRangeVar
            + dPezDAgentHeadingJac**2 * agentHeadingVar
            + dPezDPursuerSpeedJac**2 * pusuerSpeedVar
        )

        # Return the CDF at 0
        return jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(cov))

    # Apply vectorization over agentPositions and agentHeadings
    return vmap(single_agent_prob)(agentPositions, agentHeadings)


# def  probabalisticEngagementZoneVectorizedTemp(agentPositions,agentPositionCov, agentHeading,agentHeadingVar, pursuerPosition, pursuerPositionCov, pursuerRange,pursuerRangeVar, pursuerCaptureRange,pursuerCaptureRangeVar, pursuerSpeed, agentSpeed, dPezDPursuerPosition, dPezDAgentPosition,dPezDPursuerRange,dPezDPursuerCaptureRange,dPezDAgentHeading):
#     # Define vectorized operations with vmap
#     def single_agent_prob(agentPosition):
#         agentPosition = agentPosition.reshape(-1,1)
#         # Calculate the mean for the engagement zone
#         mean = inEngagementZoneJax(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()

#         # Calculate the Jacobian for the engagement zone
#         dPezDPursuerPositionJac = dPezDPursuerPosition(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
#         dPezDAgentPositionJac = dPezDAgentPosition(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
#         dPezDPursuerRangeJac = dPezDPursuerRange(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
#         dPezDPursuerCaptureRangeJac = dPezDPursuerCaptureRange(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
#         dPezDAgentHeadingJac = dPezDAgentHeading(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()

#         # Compute the covariance matrix
#         cov = dPezDPursuerPositionJac @ pursuerPositionCov @ dPezDPursuerPositionJac.T + dPezDAgentPositionJac @ agentPositionCov @ dPezDAgentPositionJac.T+ dPezDPursuerRangeJac**2 * pursuerRangeVar + dPezDPursuerCaptureRangeJac**2 * pursuerCaptureRangeVar+ dPezDAgentHeadingJac**2 * agentHeadingVar

#         # Define the normal distribution based on mean and covariance
#         # diffDistribution = jax.scipy.stats.norm(mean, jnp.sqrt(cov))
#         # jax.scipy.stats.norm.

#         # # Return the CDF at 0
#         # return diffDistribution.cdf(0)
#         return jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(cov))


#     # Apply vectorization over agentPositions
#     return vmap(single_agent_prob)(agentPositions)

# def plotProbablisticEngagemenpedalck([X.ravel(), Y.ravel()]).T


#     dPezDPursuerPosition = jacfwd(inEngagementZoneJax, argnums=2)


#     engagementZonePlot = np.zeros(X.shape)


#     totalTime = 0

#     engagementZonePlot = probabalisticEngagementZoneVectorizedTemp(agentPositions, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,dPezDPursuerPosition)
#     # for i in range(X.shape[0]):
#     #     print(i)
#     #     for j in range(X.shape[1]):
#     #         start = time.time()
#     #         # engagementZonePlot[i, j] = probabalisticEngagementZone(np.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
#     #         engagementZonePlot[i, j] = probabalisticEngagementZoneTemp(jnp.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,dPezDPursuerPosition)
#             # totalTime += time.time() - start
#     # print(totalTime)
#     # c = ax.pcolormesh(X, Y, engagementZonePlot)
#     # c = plt.Circle(pursuerPosition, pursuerRange+pursuerCaptureRange, fill=False)
#     # ax.add_artist(c)
#     c = ax.contour(X, Y, engagementZonePlot.reshape(50,50), levels=np.linspace(0,1,11))


#     return
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
    engagementZonePlot = probabalisticEngagementZoneVectorizedTemp(
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
            engagementZonePlot[i, j] = monte_carlo_probalistic_engagment_zone(
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


def monte_carlo_probalistic_engagment_zone(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
    numMonteCarloTrials,
    pursurPositionSamples=None,
    pursuerRangeSamples=None,
    pursuerCaptureRangeSamples=None,
    agentHeadingSamples=None,
    pursuerSpeedSamples=None,
    agentPositionSamples=None,
):
    if pursurPositionSamples is None:
        # randomly sample from the pursuer position distribution
        pursurPositionSamples = np.random.multivariate_normal(
            pursuerPosition.squeeze(), pursuerPositionCov, numMonteCarloTrials
        )
    # randomly sample from the pursuer position distribution

    numInEngagementZone = 0
    for i, pursuerPositionSample in enumerate(pursurPositionSamples):
        pursuerPositionSample = pursuerPositionSample.reshape(-1, 1)

        ez = inEngagementZone(
            agentPositionSamples[i].reshape(-1, 1),
            agentHeadingSamples[i],
            pursuerPositionSample,
            pursuerRangeSamples[i],
            pursuerCaptureRangeSamples[i],
            pursuerSpeedSamples[i],
            agentSpeed,
        )
        if ez < 0:
            numInEngagementZone += 1

    return numInEngagementZone / numMonteCarloTrials


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
    plotMahalanobisDistance(
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
        plotMahalanobisDistance(pursuerInitialPosition, pursuerPositionCov, mcAx, fig)
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
        plotMahalanobisDistance(
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
    plot_abs_diff(linPez, mcEz, ax, fig)
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
    #       plotMahalanobisDistance(pursuerInitialPosition, pursuerPositionCov, mcAx)
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
    #        plotMahalanobisDistance(pursuerInitialPosition, pursuerPositionCov, linAx)
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
    # main()
    plot_potential_BEZS()
