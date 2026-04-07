"""OUQ approximations for engagement zones with range-only pursuer uncertainty."""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib


import PEZ.pez_plotting as pez_plotting
import PEZ.bez_path_planner as bez_path_planner
import PLOT_COMMON.draw_airplanes as draw_airplanes

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
    """Return the worst-case capture probability under range-only OUQ bounds."""
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


def prob_uniform_single(
    evaderPosition,
    evaderHeading,
    speedRatio,
    captureRadius,
    minRange,
    maxRange,
):
    """Return the capture probability under a uniform range prior."""
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
        R1 < minRange,
        1.0,
        jnp.where(
            R1 <= maxRange,
            (maxRange - R1) / (maxRange - minRange),
            0.0,
        ),
    )

    # If discriminant < 0, return 1.0
    return jnp.where(discriminant < 0.0, 1.0, prob_when_real_root)


uniform_prob = jax.jit(
    jax.vmap(
        prob_uniform_single,
        in_axes=(0, 0, None, None, None, None),
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
    """Plan a conservative BEZ path induced by the OUQ range threshold."""
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


def uniform_range_only_path(
    pursuerPosition,
    minRange,
    maxRange,
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
    """Plan a BEZ path induced by a uniform range-uncertainty threshold."""
    safetyProbThresholdRange = np.clip(
        maxRange - pez_limit * (maxRange - minRange), minRange, maxRange
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
    safetyProbThreshold,
    ax,
    plotMinMax=True,
):
    """Plot OUQ probability contours for the range-only uncertainty model."""
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
    safetyProbThresholdRange = np.clip(
        (meanRange - minRange) / safetyProbThreshold + minRange, minRange, maxRange
    )

    probAtMaxRange = (meanRange - minRange) / (maxRange - minRange)

    # c = ax.pcolormesh(X, Y, prob.reshape(X.shape), cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    c = ax.contour(
        X,
        Y,
        prob.reshape(X.shape),
        levels=[0.05, 0.1, 0.25, 0.5],
        linewidths=3,
    )
    handles, labels = c.legend_elements()
    labels = [f"$\\epsilon={lvl:.2f}$" for lvl in c.levels]

    ax.legend(
        handles,
        labels,
        title="PEZ Level",
        loc="lower right",
        framealpha=0.8,
    )
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    if plotMinMax:
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


def main_max():
    """Run the max-probability range-only OUQ visualization demo."""
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

    safetyProbThreshold = 0.3
    safetyProbThresholdRange = np.clip(
        (meanRange - minRange) / safetyProbThreshold + minRange, minRange, maxRange
    )

    probAtMaxRange = (meanRange - minRange) / (maxRange - minRange)

    fig, ax = plt.subplots()
    # c = ax.pcolormesh(X, Y, prob.reshape(X.shape), cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    c = ax.contour(
        X,
        Y,
        prob.reshape(X.shape),
        levels=np.arange(0.0, 1.01, 0.1),
        linewidths=3,
    )
    # create legend for co
    clabels = ax.clabel(c, inline=True, fontsize=8, fmt="%.1f")
    # plt.colorbar(c, label="Max Probability of Capture")
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


def main_uniform():
    """Run the uniform-range OUQ visualization demo."""
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

    prob = uniform_prob(
        points,
        psi,
        speedRatio,
        captureRadius,
        minRange,
        maxRange,
    )

    safetyProbThreshold = 0.2
    safetyProbThresholdRange = np.clip(
        maxRange - safetyProbThreshold * (maxRange - minRange), minRange, maxRange
    )

    probAtMaxRange = (meanRange - minRange) / (maxRange - minRange)

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
        minRange,
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
    """Render animation frames for the range-only OUQ path-planning demo."""
    initialEvaderPosition = np.array([-4.0, -4.0])
    finalEvaderPosition = np.array([4.0, 4.0])
    initialEvaderVelocity = np.array([1.0, 0.0])
    minRange = 0.5
    maxRange = 2.0
    meanRange = 0.75
    # meanRange = (minRange + maxRange) / 2.0
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.0
    evaderSpeed = 1.0
    pursuerPosition = np.array([0.0, 0.0])

    num_cont_points = 25
    spline_order = 3
    velocity_constraints = (0.0, 1.0)
    curvature_constraints = (-0.5, 0.5)
    turn_rate_constraints = (-1.0, 1.0)
    num_constraint_samples = 50

    pez_limit = 0.25

    x = np.linspace(-4, 4, 500)
    y = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)

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

    splineUniform, tfUniform = uniform_range_only_path(
        pursuerPosition,
        minRange,
        maxRange,
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
    print("OUQ path duration:", tf)
    print("Uniform path duration:", tfUniform)

    currentTime = 0
    dt = 0.08
    finalTime = spline.t[-1 - spline.k]
    t = np.linspace(0, finalTime, 500)
    uniformSplinePoints = splineUniform(t)
    uniformSplineVelocities = splineUniform.derivative(1)(t)
    uniformSplineHeadings = np.arctan2(
        uniformSplineVelocities[:, 1], uniformSplineVelocities[:, 0]
    )
    ouqProbAtUniformPoints = max_prob(
        uniformSplinePoints,
        uniformSplineHeadings,
        evaderSpeed / pursuerSpeed,
        pursuerCaptureRadius,
        minRange,
        maxRange,
        meanRange,
    )
    print("OUQ probabilities at uniform points:", np.max(ouqProbAtUniformPoints))
    ouqSplinePos = spline(t)
    ouqSplineVel = spline.derivative(1)(t)
    ouqSplineHeadings = np.arctan2(ouqSplineVel[:, 1], ouqSplineVel[:, 0])
    ouqProbAtOUQPoints = max_prob(
        ouqSplinePos,
        ouqSplineHeadings,
        evaderSpeed / pursuerSpeed,
        pursuerCaptureRadius,
        minRange,
        maxRange,
        meanRange,
    )
    print("OUQ probabilities at OUQ points:", np.max(ouqProbAtOUQPoints))
    uniformProbAtOUQPoints = uniform_prob(
        ouqSplinePos,
        ouqSplineHeadings,
        evaderSpeed / pursuerSpeed,
        pursuerCaptureRadius,
        minRange,
        maxRange,
    )
    print("Uniform probabilities at OUQ points:", np.max(uniformProbAtOUQPoints))
    pos = ouqSplinePos
    # vel = spline.derivative(1)(t)

    ind = 0

    while currentTime < finalTime:
        fig, ax = plt.subplots()
        pdot = spline.derivative(1)(currentTime)
        currentPosition = spline(currentTime)
        currentHeading = np.arctan2(pdot[1], pdot[0])
        plot_prob_contours(
            currentHeading,
            evaderSpeed / pursuerSpeed,
            pursuerCaptureRadius,
            minRange,
            maxRange,
            meanRange,
            pursuerSpeed,
            evaderSpeed,
            pez_limit,
            ax,
            plotMinMax=False,
        )

        draw_airplanes.draw_airplane(
            ax,
            currentPosition,
            angle=currentHeading - np.pi / 2,
            color="blue",
            size=0.4,
        )
        ax.plot(pos[:, 0], pos[:, 1], color="blue")
        # ax.plot(uniformSplinePoints[:, 0], uniformSplinePoints[:, 1], color="magenta")
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
    #


if __name__ == "__main__":
    animate_spline_path()
    # main_uniform()
    # main_max()
