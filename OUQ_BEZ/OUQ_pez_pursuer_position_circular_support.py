"""OUQ approximations for uncertain pursuer position inside a rectangular support."""

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


import PLOT_COMMON.draw_airplanes as draw_airplanes
import GEOMETRIC_BEZ.bez_from_interceptions as bez_from_interceptions
import GEOMETRIC_BEZ.bez_from_interceptions_path_planner as bez_from_interceptions_path_planner


def ouq_inner_circle_for_alpha(
    alpha,
    x_pmean,
    y_pmean,
    x_center,
    y_center,
    radius,
):
    s = (1 - alpha) / alpha
    x_center_new = x_pmean - s * (x_center - x_pmean)
    y_center_new = y_pmean - s * (y_center - y_pmean)
    radius_new = s * radius
    return x_center_new, y_center_new, radius_new


def ouq_inner_shape_for_alpha(
    alpha,
    x_pmean,
    y_pmean,
    x_center,
    y_center,
    radius,
):
    x_center_new, y_center_new, radius_new = ouq_inner_circle_for_alpha(
        alpha, x_pmean, y_pmean, x_center, y_center, radius
    )
    centers = np.array([[x_center, y_center], [x_center_new, y_center_new]])
    radii = np.array([radius, radius_new])
    arcs = bez_from_interceptions.intersection_arcs(
        centers=centers,
        radii=radii,
    )
    return arcs


def plot_ouq_contour(
    alpha,
    x_pmean,
    y_pmean,
    x_center,
    y_center,
    radius,
    pursuerRange,
    captureRadius,
    pursuerSpeed,
    evaderSpeed,
    evaderHeading,
    ax,
):
    arcs = ouq_inner_shape_for_alpha(
        alpha, x_pmean, y_pmean, x_center, y_center, radius
    )
    bez_from_interceptions.plot_potential_pursuer_engagement_zone(
        arcs,
        pursuerRange,
        captureRadius,
        pursuerSpeed,
        evaderHeading,
        evaderSpeed,
        xlim=(-6, 6),
        ylim=(-6, 6),
        ax=ax,
    )


def plan_ouq_path(
    pez_limit,
    meanX,
    meanY,
    supportCenterX,
    supportCenterY,
    supportRadius,
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
    supportCenterX = 0.0
    supportCenterY = 0.0
    supportRadius = 2.0
    meanX = -1.2
    meanY = 0.7

    pez_limits = [0.1, 0.2, 0.3, 0.4, 0.5]
    fig, ax = plt.subplots()
    for pez_limit in pez_limits:
        plot_ouq_contour(
            pez_limit,
            meanX,
            meanY,
            supportCenterX,
            supportCenterY,
            supportRadius,
            pursuerRange,
            captureRadius,
            pursuerSpeed,
            evaderSpeed,
            psi,
            ax,
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
