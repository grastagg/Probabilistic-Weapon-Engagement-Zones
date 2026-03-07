from jax._src.source_info_util import current
import numpy as np
import json
from pyoptsparse import Optimization, OPT, IPOPT
from scipy.interpolate import BSpline
import time
from tqdm import tqdm
from jax import jacfwd
from jax import jit
import jax
from functools import partial
import jax.numpy as jnp
import getpass
import matplotlib.pyplot as plt
import matplotlib


import PEZ.pez_plotting as pez_plotting
import PEZ.bez_path_planner as bez_path_planner

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


def max_prob_single(
    evaderPosition,
    evaderHeading,
    speedRatio,
    captureRadius,
    minRange,
    maxRange,
    meanRange,
):
    x, y = evaderPosition

    a = speedRatio**2 - 1.0
    alpha = x * jnp.cos(evaderHeading) + y * jnp.sin(evaderHeading)
    b = 2.0 * (speedRatio * alpha - captureRadius)
    c = x**2 + y**2 - captureRadius**2

    discriminant = b**2 - 4.0 * a * c

    # Safe sqrt so we never take sqrt of a negative number
    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))

    # Safe denominator in case a == 0
    denom = 2.0 * a
    R1 = jnp.where(jnp.abs(denom) > 1e-12, (-b - sqrt_disc) / denom, jnp.inf)

    prob_when_real_root = jnp.where(
        R1 < meanRange,
        1.0,
        jnp.where(
            R1 <= maxRange,
            (meanRange - minRange) / (R1 - minRange),
            0.0,
        ),
    )

    # If discriminant < 0, return 1.0
    return jnp.where(discriminant < 0.0, 1.0, prob_when_real_root)


max_prob = jax.jit(
    jax.vmap(
        max_prob_single,
        in_axes=(0, 0, None, None, None, None, None),
    )
)


def ouq_range_only_path(
    pursuerPosition,
    minRange,
    maxRange,
    meanRange,
    captureRadius,
    pursuerSpeed,
    evaderSpeed,
    initialEvaderPosition,
    finalEvaderPosition,
    initialEvaderVelocity,
    num_cont_points,
    spline_order,
    velocity_constraints,
    turn_rate_constraints,
    curvature_constraints,
    num_constraint_samples,
    pez_limit,
):
    safetyProbThresholdRange = np.clip(
        (meanRange - minRange) / pez_limit + minRange, meanRange, maxRange
    )
    spline, tf = bez_path_planner.plan_path_BEZ(
        pursuerPosition,
        safetyProbThresholdRange,
        captureRadius,
        pursuerSpeed,
        initialEvaderPosition,
        finalEvaderPosition,
        initialEvaderVelocity,
        evaderSpeed,
        num_cont_points,
        spline_order,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
    )
    return spline, tf


def plot_prob_contours(
    psi,
    speedRatio,
    captureRadius,
    minRange,
    maxRange,
    meanRange,
    pursuerSpeed,
    evaderSpeed,
):
    psiS = psi
    psi = np.ones(500 * 500) * psi
    x = np.linspace(-4, 4, 500)
    y = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    prob = max_prob(
        points,
        psi,
        speedRatio,
        captureRadius,
        minRange,
        maxRange,
        meanRange,
    )
    safetyProbThreshold = 0.2
    safetyProbThresholdRange = np.clip(
        (meanRange - minRange) / safetyProbThreshold + minRange, minRange, maxRange
    )
    print("safetyProbThresholdRange", safetyProbThresholdRange)

    probAtMaxRange = (meanRange - minRange) / (maxRange - minRange)

    print(f"prob at max range: {probAtMaxRange}")

    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, prob.reshape(X.shape), cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    ax.contour(
        X,
        Y,
        prob.reshape(X.shape),
        levels=[safetyProbThreshold],
        linewidths=3,
        colors="green",
    )
    plt.colorbar(c, label="Max Probability of Capture")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    pez_plotting.plotEngagementZone(
        psiS,
        np.array([0.0, 0.0]),
        meanRange,
        captureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="red",
        label=True,
    )
    pez_plotting.plotEngagementZone(
        psiS,
        np.array([0.0, 0.0]),
        maxRange,
        captureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="red",
        label=True,
    )
    pez_plotting.plotEngagementZone(
        psiS,
        np.array([0.0, 0.0]),
        safetyProbThresholdRange,
        captureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="red",
        label=True,
    )
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)


def main():
    x = np.linspace(-4, 4, 500)
    y = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    psi = 0.0 * np.ones(points.shape[0])
    pursuerSpeed = 1.0
    evaderSpeed = 0.9

    speedRatio = evaderSpeed / pursuerSpeed
    captureRadius = 0.1
    minRange = 0.5
    maxRange = 2.0
    meanRange = 0.75
    prob = max_prob(
        points,
        psi,
        speedRatio,
        captureRadius,
        minRange,
        maxRange,
        meanRange,
    )
    safetyProbThreshold = 0.2
    safetyProbThresholdRange = np.clip(
        (meanRange - minRange) / safetyProbThreshold + minRange, minRange, maxRange
    )
    print("safetyProbThresholdRange", safetyProbThresholdRange)

    probAtMaxRange = (meanRange - minRange) / (maxRange - minRange)

    print(f"prob at max range: {probAtMaxRange}")

    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, prob.reshape(X.shape), cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    ax.contour(
        X,
        Y,
        prob.reshape(X.shape),
        levels=[safetyProbThreshold],
        linewidths=3,
        colors="green",
    )
    plt.colorbar(c, label="Max Probability of Capture")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    pez_plotting.plotEngagementZone(
        0.0,
        np.array([0.0, 0.0]),
        meanRange,
        captureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="red",
        label=True,
    )
    pez_plotting.plotEngagementZone(
        0.0,
        np.array([0.0, 0.0]),
        maxRange,
        captureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="red",
        label=True,
    )
    pez_plotting.plotEngagementZone(
        0.0,
        np.array([0.0, 0.0]),
        safetyProbThresholdRange,
        captureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="red",
        label=True,
    )
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plt.show()


def animate_spline_path():
    initialEvaderPosition = np.array([-5.0, -5.0])
    finalEvaderPosition = np.array([5.0, 5.0])
    initialEvaderVelocity = np.array([1.0, 0.0])
    minRange = 0.5
    maxRange = 2.0
    meanRange = 0.75
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.0
    evaderSpeed = 1.0
    pursuerPosition = np.array([0.0, 0.0])

    num_cont_points = 20
    spline_order = 3
    velocity_constraints = (0.0, 1.0)
    curvature_constraints = (-0.5, 0.5)
    turn_rate_constraints = (-1.0, 1.0)
    num_constraint_samples = 50

    pez_limit = 0.2

    spline, tf = ouq_range_only_path(
        pursuerPosition,
        minRange,
        maxRange,
        meanRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        evaderSpeed,
        initialEvaderPosition,
        finalEvaderPosition,
        initialEvaderVelocity,
        num_cont_points,
        spline_order,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        pez_limit,
    )

    currentTime = 0
    dt = 0.05
    finalTime = spline.t[-1 - spline.k]
    t = np.linspace(0, finalTime, 500)
    pos = spline(t)
    ind = 0
    while currentTime < finalTime:
        fig, ax = plt.subplots()
        pdot = spline.derivative(1)(currentTime)
        currentPosition = spline(currentTime)
        currentHeading = np.arctan2(pdot[1], pdot[0])
        ax.plot(pos[0], pos[1], color="blue")
        plot_prob_contours(
            currentHeading,
            evaderSpeed / pursuerSpeed,
            pursuerCaptureRadius,
            minRange,
            maxRange,
            meanRange,
            pursuerSpeed,
            evaderSpeed,
        )

        plt.arrow(
            currentPosition[0],
            currentPosition[1],
            1e-6 * np.cos(currentHeading),  # essentially zero-length tail
            1e-6 * np.sin(currentHeading),
            head_width=0.25,
            head_length=0.3,
            width=0,  # no line
            fc="blue",
            ec="blue",
            zorder=5,
        )
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        fig.savefig(f"video/{ind}.png", dpi=300)
        ind += 1
        currentTime += dt
        plt.close(fig)


if __name__ == "__main__":
    animate_spline_path()
    # main()
