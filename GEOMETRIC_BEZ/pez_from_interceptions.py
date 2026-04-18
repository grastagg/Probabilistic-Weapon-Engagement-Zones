"""Probabilistic engagement-zone models built from interception constraints."""

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib
import matplotlib.pyplot as plt

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

import GEOMETRIC_BEZ.bez_from_interceptions as bez_from_interceptions


def uniform_pdf_from_interception_points(
    points,  # (N,2)
    interceptionPositions,  # (M,2)
    radii,  # (M,)  unique radius per interception disc
    dA,  # scalar integration area element
    eps=1e-12,  # avoid divide-by-zero if intersection is empty
):
    """
    Uniform PDF over the intersection of M discs centered at interceptionPositions
    with per-disc radii given by 'radii'.

    points: (N,2) evaluation/integration grid
    interceptionPositions: (M,2) centers of discs
    radii: (M,) radius for each disc (same units as points)
    dA: area element for numerical integration

    Returns:
        pdf: (N,) array, uniform over the intersection region
    """
    points = jnp.asarray(points)
    interceptionPositions = jnp.asarray(interceptionPositions)
    radii = jnp.asarray(radii)

    # distances: (N, M)
    diff = points[:, None, :] - interceptionPositions[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)

    # inside intersection if inside ALL discs (broadcast radii to (1,M))
    inside_all = jnp.all(dists <= radii[None, :], axis=1)  # (N,)

    # numeric area of intersection region
    area = jnp.sum(inside_all) * dA

    # uniform pdf: 1/area inside, 0 outside
    inv_area = 1.0 / jnp.maximum(area, eps)
    pdf = jnp.where(inside_all, inv_area, 0.0)

    return pdf


@jax.jit
def prob_reach_numerical(eval_points, integration_points, pdf_vals, R_eff, dA):
    """
    eval_points: (K,2)
    integration_points: (M,2)
    pdf_vals: (M,)
    """
    # diff: (K, M, 2)
    diff = integration_points[None, :, :] - eval_points[:, None, :]
    dists = jnp.linalg.norm(diff, axis=-1)  # (K, M)
    inRange = dists <= R_eff  # (K, M)
    # tau = 0.1
    # inRange = jax.nn.sigmoid((R_eff - dists) / tau)  # (K,M)

    # broadcast pdf_vals: (M,) -> (K, M)
    weighted = inRange * pdf_vals[None, :]
    return jnp.sum(weighted, axis=1) * dA  # (K,)


@jax.jit
def prob_reach_numerical_soft(
    eval_points, integration_points, pdf_vals, R_eff, dA, tau
):
    """
    Smooth approximation to P(reach) = ∫ 1(||x - e|| <= R_eff) p(x) dx
    using a sigmoid boundary.

    tau: softness length scale (same units as positions). Smaller -> sharper.
    """
    diff = integration_points[None, :, :] - eval_points[:, None, :]
    dists = jnp.linalg.norm(diff, axis=-1)  # (K, M)

    # soft indicator in (0,1)
    in_range_soft = jax.nn.sigmoid((R_eff - dists) / tau)  # (K, M)

    weighted = in_range_soft * pdf_vals[None, :]
    return jnp.sum(weighted, axis=1) * dA  # (K,)


@jax.jit
def pez_numerical_soft(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerSpeed,
    integration_points,
    pdf_vals,
    pursuerRange,
    pursuerCaptureRadius,
    dA,
):
    """Evaluate a smoothed PEZ contour using numerical integration over launch points."""
    futureEvaderPositions = (
        evaderPosition
        + (evaderSpeed / pursuerSpeed)
        * pursuerRange
        * jnp.vstack([jnp.cos(evaderHeading), jnp.sin(evaderHeading)]).T
    )
    return prob_reach_numerical_soft(
        futureEvaderPositions,
        integration_points,
        pdf_vals,
        pursuerRange + pursuerCaptureRadius,
        dA,
        tau=0.01,
    )


@jax.jit
def pez_numerical(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerSpeed,
    integration_points,
    pdf_vals,
    pursuerRange,
    pursuerCaptureRadius,
    dA,
):
    """Evaluate the PEZ exactly on a discretized launch-position grid."""
    futureEvaderPositions = (
        evaderPosition
        + (evaderSpeed / pursuerSpeed)
        * pursuerRange
        * jnp.vstack([jnp.cos(evaderHeading), jnp.sin(evaderHeading)]).T
    )
    return prob_reach_numerical(
        futureEvaderPositions,
        integration_points,
        pdf_vals,
        pursuerRange + pursuerCaptureRadius,
        dA,
    )


def plot_prob_reachable(
    interceptionPositions,
    radii,
    pursuerRange,
    pursuerCaptureRadius,
    numPoints,
    xlim,
    ylim,
    ax,
    levels,
):
    """Plot contours of reachable probability under a uniform launch-region PDF."""
    points, X, Y = bez_from_interceptions.get_meshgrid_points(xlim, ylim, numPoints)

    dArea = (
        (xlim[1] - xlim[0]) / (numPoints - 1) * (ylim[1] - ylim[0]) / (numPoints - 1)
    )

    launch_pdf = uniform_pdf_from_interception_points(
        points, interceptionPositions, radii, dArea
    )
    # numerically integrate to check pdf
    integral = jnp.sum(launch_pdf) * dArea
    print("Launch PDF integral check:", integral)
    probReachable = prob_reach_numerical(
        points, points, launch_pdf, pursuerRange + pursuerCaptureRadius, dArea
    )
    c = ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        probReachable.reshape((numPoints, numPoints)),
        levels=levels,
    )


def plot_pez_from_interception():
    """Generate the interception-driven probabilistic reachable-region figure."""
    pursuerSpeed = 2.0
    pursuerRange = 1.5
    pursuerCaptureRadius = 0.1
    evaderHeading = np.pi / 4
    evaderSpeed = 1.0
    pursuerPosition = np.array([0.0, 0.0])
    interceptionPositions = np.array([[0.4, 0.5], [-0.8, -0.8], [-0.7, 0.9]])

    dists = np.linalg.norm(pursuerPosition - interceptionPositions, axis=1)
    launchTimes = dists / pursuerSpeed * np.random.uniform(1, 1.1, size=dists.shape)
    pursuerPathDistances = launchTimes * pursuerSpeed
    if np.any(pursuerPathDistances > pursuerRange):
        print("Warning: launch times too long")

    radii = pursuerPathDistances + pursuerCaptureRadius
    radii = (pursuerRange + pursuerCaptureRadius) * np.ones(len(interceptionPositions))

    numPoints = 160
    xlim = (-4, 4)
    ylim = (-4, 4)
    points, X, Y = bez_from_interceptions.get_meshgrid_points(xlim, ylim, numPoints)

    dArea = (
        (xlim[1] - xlim[0]) / (numPoints - 1) * (ylim[1] - ylim[0]) / (numPoints - 1)
    )

    launch_pdf = uniform_pdf_from_interception_points(
        points, interceptionPositions, radii, dArea
    )
    # numerically integrate to check pdf
    integral = jnp.sum(launch_pdf) * dArea
    print("Launch PDF integral check:", integral)
    probReachable = prob_reach_numerical(
        points, points, launch_pdf, pursuerRange + pursuerCaptureRadius, dArea
    )
    probEngagmentZone = pez_numerical(
        points,
        np.ones(len(points)) * evaderHeading,
        evaderSpeed,
        pursuerSpeed,
        points,
        launch_pdf,
        pursuerRange,
        pursuerCaptureRadius,
        dArea,
    )
    arcs = bez_from_interceptions.compute_potential_pursuer_region_from_interception_position_and_radii(
        interceptionPositions,
        radii,
    )
    fig = plt.figure(figsize=(6.5, 6.8))

    # Manual layout:
    # top-left  : [left, bottom, width, height]
    # top-right : shifted right by width + gap
    # bottom    : same size, centered
    ax_pRR = fig.add_axes([0.08, 0.55, 0.38, 0.38])  # top-left
    ax_pEZ = fig.add_axes([0.54, 0.55, 0.38, 0.38])  # top-right
    ax_pdf = fig.add_axes([0.31, 0.08, 0.38, 0.38])  # bottom center

    # ---------- Top-left: p_RR ----------
    ax_pRR.set_title(r"$p_{RR}$")
    ax_pRR.set_aspect("equal")
    ax_pRR.set_xlabel("X")
    ax_pRR.set_ylabel("Y")
    ax_pRR.set_xlim([-4, 4])
    ax_pRR.set_ylim([-4, 4])
    ax_pRR.set_xticks(np.arange(-4, 5, 1))
    ax_pRR.set_yticks(np.arange(-4, 5, 1))
    levels = [0.01, 0.10, 0.30, 0.50, 0.70, 0.90]

    c = ax_pRR.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        probReachable.reshape((numPoints, numPoints)),
        levels=levels,
    )
    ax_pRR.clabel(c, inline=True)
    # bez_from_interceptions.plot_interception_points(
    #     interceptionPositions,
    #     np.ones(len(interceptionPositions)) * (pursuerRangeMean + pursuerCaptureRadius),
    #     ax_pRR,
    # )
    # bez_from_interceptions.plot_circle_intersection_arcs(arcs, ax=ax_pRR)
    bez_from_interceptions.plot_potential_pursuer_reachable_region(
        arcs,
        pursuerRange,
        pursuerCaptureRadius,
        xlim,
        ylim,
        numPoints=200,
        ax=ax_pRR,
    )

    # ---------- Top-right: p_EZ ----------
    ax_pEZ.set_title(r"$p_{EZ}$")
    ax_pEZ.set_aspect("equal")
    ax_pEZ.set_xlabel("X")
    ax_pEZ.set_xlim([-4, 4])
    ax_pEZ.set_ylim([-4, 4])
    ax_pEZ.set_xticks(np.arange(-4, 5, 1))
    ax_pEZ.set_yticks(np.arange(-4, 5, 1))

    c = ax_pEZ.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        probEngagmentZone.reshape((numPoints, numPoints)),
        levels=levels,
    )
    ax_pEZ.clabel(c, inline=True)
    bez_from_interceptions.plot_potential_pursuer_engagement_zone(
        arcs,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        evaderHeading,
        evaderSpeed,
        xlim=(-4, 4),
        ylim=(-4, 4),
        ax=ax_pEZ,
    )

    # ---------- Bottom center: launch PDF ----------
    ax_pdf.set_aspect("equal")
    ax_pdf.set_xlabel("X")
    ax_pdf.set_ylabel("Y")
    ax_pdf.set_xlim([-4, 4])
    ax_pdf.set_ylim([-4, 4])
    ax_pdf.set_xticks(np.arange(-4, 5, 1))
    ax_pdf.set_yticks(np.arange(-4, 5, 1))

    c = ax_pdf.pcolormesh(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        launch_pdf.reshape((numPoints, numPoints)),
    )
    c.set_edgecolor("face")
    fig.colorbar(c, ax=ax_pdf, fraction=0.046, pad=0.04)

    bez_from_interceptions.plot_interception_points(
        interceptionPositions,
        radii,
        ax_pdf,
    )
    ax_pdf.set_title(r"$f_{\boldsymbol{x}_P}(\boldsymbol{x})$")

    # ---------- Legend ----------
    fig.legend(
        loc="lower left",
        ncol=1,
        bbox_to_anchor=(0.01, 0.1),
    )

    plt.show()


if __name__ == "__main__":
    plot_pez_from_interception()
