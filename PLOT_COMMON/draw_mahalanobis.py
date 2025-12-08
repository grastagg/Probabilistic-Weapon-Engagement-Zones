import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
