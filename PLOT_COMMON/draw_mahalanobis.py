"""Helpers for plotting Mahalanobis-distance contours."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10

_GRID_LIMITS = (-2.0, 2.0)
_GRID_SAMPLES = 100
_CONTOUR_LEVELS = [0, 1, 2, 3]
_CONTOUR_COLORS = ["#CC0000", "#FF6666", "#FFCCCC"]


def plotMahalanobisDistance(
    pursuerPosition, pursuerPositionCov, ax, fig, plotColorbar=False, cax=None
):
    """Plot Mahalanobis-distance contours for a pursuer position covariance."""
    x = np.linspace(*_GRID_LIMITS, _GRID_SAMPLES)
    y = np.linspace(*_GRID_LIMITS, _GRID_SAMPLES)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    inverse_covariance = np.linalg.inv(pursuerPositionCov)
    delta = points - np.asarray(pursuerPosition).T
    mahalanobis_distance = np.sqrt(
        np.einsum("ij,jk,ik->i", delta, inverse_covariance, delta)
    ).reshape(x_grid.shape)

    contour = ax.contourf(
        x_grid,
        y_grid,
        mahalanobis_distance,
        levels=_CONTOUR_LEVELS,
        colors=_CONTOUR_COLORS,
        alpha=0.75,
    )

    if plotColorbar:
        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = fig.colorbar(contour, cax=cax, ticks=_CONTOUR_LEVELS, shrink=0.5)
        colorbar.set_label("Pursuer Std Dev")
        colorbar.ax.tick_params()

    return contour
