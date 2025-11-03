from jax._src.config import int_env
from jax._src.prng import random_wrap
import sys
import time
import jax
from tabulate import tabulate


import getpass
from pyoptsparse import Optimization, OPT, IPOPT
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# get rid of type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# set font size for title, axis labels, and legend, and tick labels
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10


import dubinsEZ

# set maplotlib to faster backend


def interception_probability_exp(pursuerParams, evaderPosition):
    pursuerPosition = pursuerParams[:2]
    pursuerHeading = pursuerParams[2]
    pursuerSpeed = pursuerParams[3]
    pursuerTurnRadius = pursuerParams[4]
    pursuerRange = pursuerParams[5]
    pathLength, turnAngle = dubinsEZ.find_shortest_dubins_path_and_turn_amount(
        pursuerPosition, pursuerHeading, evaderPosition, pursuerTurnRadius
    )
    inRS = pathLength < pursuerRange
    turnWeight = 0.8
    lengthWeight = 1 - turnWeight
    pathEffort = (
        turnWeight * jnp.abs(turnAngle) / (2 * jnp.pi)
        + lengthWeight * pathLength / pursuerRange
    )
    nominalProb = 1.0
    probDecayRate = 0.5
    interceptionProb = (
        inRS * nominalProb * jnp.exp(-(pathEffort**2) / (2 * probDecayRate**2))
    )
    return interceptionProb


interception_probability_exp_mult = jax.jit(
    jax.vmap(interception_probability_exp, in_axes=(None, 0))
)


def plot_interception_probability(pursuerParams, evaderPosition):
    numPoints = 1000
    x = np.linspace(-2, 2, numPoints)
    y = np.linspace(-2, 2, numPoints)
    [X, Y] = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    evaderPositions = jnp.vstack((X, Y)).T
    probabilities = interception_probability_exp_mult(pursuerParams, evaderPositions)
    print("here")

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.pcolormesh(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        probabilities.reshape((numPoints, numPoints)),
        shading="auto",
        cmap="viridis",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(c, ax=ax, label="Interception Probability")
    plt.show()


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerHeading = 0.0
    pursuerSpeed = 1.0
    pursuerTurnRadius = 0.2
    pursuerRange = 1.0
    pursuerParams = np.array(
        [
            pursuerPosition[0],
            pursuerPosition[1],
            pursuerHeading,
            pursuerSpeed,
            pursuerTurnRadius,
            pursuerRange,
        ]
    )
    plot_interception_probability(
        pursuerParams,
        evaderPosition=np.array([1.0, 0.0]),
    )


if __name__ == "__main__":
    main()
