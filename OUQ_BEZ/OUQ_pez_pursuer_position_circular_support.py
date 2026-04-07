"""OUQ approximations for uncertain pursuer position inside a rectangular support."""

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


import PLOT_COMMON.draw_airplanes as draw_airplanes
import GEOMETRIC_BEZ.rectangle_bez as rectangle_bez
import GEOMETRIC_BEZ.rectangle_bez_path_planner as rectangle_bez_path_planner


def ouq_inner_circle_for_alpha(
    alpha,
    x_pmean,
    y_pmean,
    x_center,
    y_center,
    radius,
):
    pass


def max_rectangle_in_bounds(x0, y0, q, Xmin, Xmax, Ymin, Ymax):
    """Return the largest rectangle around `(x0, y0)` satisfying OUQ mass fraction `q`."""
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")

    W = min((x0 - Xmin) / (1.0 - q), (Xmax - x0) / q)
    H = min((y0 - Ymin) / (1.0 - q), (Ymax - y0) / q)

    xmin = x0 - (1.0 - q) * W
    xmax = x0 + q * W
    ymin = y0 - (1.0 - q) * H
    ymax = y0 + q * H

    return xmin, xmax, ymin, ymax


def plan_ouq_path(
    pez_limit,
    meanX,
    meanY,
    minX,
    maxX,
    minY,
    maxY,
    pursuerRange,
    pursuerCaptureRadius,
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
):
    """Plan a box-BEZ path using the OUQ inner rectangle at level `pez_limit`."""
    minXlim, maxXlim, minYlim, maxYlim = ouq_inner_rectangle_for_alpha(
        pez_limit, meanX, meanY, minX, maxX, minY, maxY
    )
    print("minXlim, maxXlim, minYlim, maxYlim:", minXlim, maxXlim, minYlim, maxYlim)
    spline, tf = rectangle_bez_path_planner.plan_path_box_BEZ(
        np.array([minXlim, minYlim]),
        np.array([maxXlim, maxYlim]),
        pursuerRange,
        pursuerCaptureRadius,
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
        num_constraint_samples=None,
    )
    return spline, tf


def main():
    """Run the OUQ pursuer-position contour and inner-rectangle demo."""
    x = np.linspace(-4, 4, 500)
    y = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    psi = np.deg2rad(0.0) * np.ones(points.shape[0])
    pursuerSpeed = 1.0
    evaderSpeed = 0.5

    speedRatio = evaderSpeed / pursuerSpeed
    captureRadius = 0.1
    pursuerRange = 1.0
    minX = -2.0
    maxX = 2.0
    minY = -1.0
    maxY = 1.0
    meanX = -1.2
    meanY = 0.7
    test = max_ouq_prob_pursuer_position_uncertainty_single(
        2.0,
        1.4,
        0.0,
        pursuerRange,
        captureRadius,
        speedRatio,
        meanX,
        meanY,
        minX,
        maxX,
        minY,
        maxY,
    )
    print(test)
    prob = max_ouq_prob_pursuer_position_uncertainty(
        points[:, 0],
        points[:, 1],
        psi,
        pursuerRange,
        captureRadius,
        speedRatio,
        meanX,
        meanY,
        minX,
        maxX,
        minY,
        maxY,
    )

    fig, ax = plt.subplots()
    ax.grid(True)
    c = ax.contour(
        X,
        Y,
        prob.reshape(X.shape),
        cmap="viridis",
        shading="auto",
        levels=np.arange(0, 1.2, 0.1),
    )
    # inline labels for contours
    clabels = ax.clabel(c, inline=True, fontsize=8, fmt="%.1f")

    # c = ax.pcolormesh(X, Y, prob.reshape(X.shape), cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    plt.colorbar(c, label="Max Probability of Capture")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    # plot minx, maxx, miny, maxy box
    plt.plot([minX, maxX, maxX, minX, minX], [minY, minY, maxY, maxY, minY], "r--")
    plt.scatter(meanX, meanY, color="red", label="Mean Position")

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    pez_limit = 0.3
    minXlim, maxXlim, minYlim, maxYlim = ouq_inner_rectangle_for_alpha(
        pez_limit, meanX, meanY, minX, maxX, minY, maxY
    )
    # plot rectangle to be expanded for alpha
    plt.plot(
        [minXlim, maxXlim, maxXlim, minXlim, minXlim],
        [minYlim, minYlim, maxYlim, maxYlim, minYlim],
        "g--",
        label=f"OUQ Inner Rectangle for alpha={pez_limit}",
    )
    rectangle_bez.plot_box_pursuer_engagement_zone(
        np.array([minXlim, minYlim]),
        np.array([maxXlim, maxYlim]),
        pursuerRange,
        captureRadius,
        pursuerSpeed,
        psi[0],
        evaderSpeed,
        (-4, 4),
        (-4, 4),
        ax,
        color="green",
    )

    plt.show()


def animate_spline_path():
    """Render animation frames for the OUQ pursuer-position path-planning demo."""
    initialEvaderPosition = np.array([-4.0, -4.0])
    finalEvaderPosition = np.array([4.0, 4.0])
    initialEvaderVelocity = np.array([1.0, 0.0])
    pursuerRange = 1.0
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.1
    evaderSpeed = 1.0
    pursuerPositionMean = np.array([-1.2, 0.7])
    pursuerMinX = -2.0
    pursuerMaxX = 2.0
    pursuerMinY = -1.0
    pursuerMaxY = 1.0
    x = np.linspace(-4, 4, 500)
    y = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)

    num_cont_points = 25
    spline_order = 3
    velocity_constraints = (0.0, 1.0)
    curvature_constraints = (-0.5, 0.5)
    turn_rate_constraints = (-1.0, 1.0)

    pez_limit = 0.2

    spline, tf = plan_ouq_path(
        pez_limit,
        pursuerPositionMean[0],
        pursuerPositionMean[1],
        pursuerMinX,
        pursuerMaxX,
        pursuerMinY,
        pursuerMaxY,
        pursuerRange,
        pursuerCaptureRadius,
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
    )

    print("OUQ path duration:", tf)

    currentTime = 0
    dt = 0.08
    finalTime = spline.t[-1 - spline.k]
    t = np.linspace(0, finalTime, 500)
    ouqSplinePos = spline(t)
    pos = ouqSplinePos
    # vel = spline.derivative(1)(t)

    ind = 0

    while currentTime < finalTime:
        fig, ax = plt.subplots()
        pdot = spline.derivative(1)(currentTime)
        currentPosition = spline(currentTime)
        currentHeading = np.arctan2(pdot[1], pdot[0])

        draw_airplanes.draw_airplane(
            ax,
            currentPosition,
            angle=currentHeading - np.pi / 2,
            color="blue",
            size=0.4,
        )
        ax.plot(pos[:, 0], pos[:, 1])
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        prob = max_ouq_prob_pursuer_position_uncertainty(
            points[:, 0],
            points[:, 1],
            currentHeading * np.ones(points.shape[0]),
            pursuerRange,
            pursuerCaptureRadius,
            evaderSpeed / pursuerSpeed,
            pursuerPositionMean[0],
            pursuerPositionMean[1],
            pursuerMinX,
            pursuerMaxX,
            pursuerMinY,
            pursuerMaxY,
        )
        c = ax.contour(
            X,
            Y,
            prob.reshape(X.shape),
            cmap="viridis",
            shading="auto",
            levels=np.arange(0, 1.2, 0.1),
        )
        # inline labels for contours
        clabels = ax.clabel(c, inline=True, fontsize=8, fmt="%.1f")
        fig.savefig(f"video/{ind}.png", dpi=300)
        ind += 1
        currentTime += dt
        plt.close(fig)


if __name__ == "__main__":
    animate_spline_path()
    # main()
