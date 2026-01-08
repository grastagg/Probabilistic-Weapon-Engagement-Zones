import numpy as np
import time
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


def prob_launch_feasible_from_intercept(
    evaluation_point, interception_position, range_mean, range_std
):
    """P(candidate launch point is feasible given ONE intercept and range uncertainty)."""
    dist = jnp.linalg.norm(interception_position - evaluation_point)
    return 1.0 - jax.scipy.stats.norm.cdf(dist, loc=range_mean, scale=range_std)


prob_launch_feasible_from_intercept_vmap = jax.jit(
    jax.vmap(prob_launch_feasible_from_intercept, in_axes=(0, None, None, None))
)


def prob_launch_feasible_from_intercepts(
    evaluation_point, interception_positions, range_mean, range_std
):
    """P(candidate launch point is feasible given MULTIPLE intercepts and range uncertainty)."""
    dists = jnp.linalg.norm(interception_positions - evaluation_point, axis=1)
    max_dist = jnp.max(dists)
    return 1.0 - jax.scipy.stats.norm.cdf(max_dist, loc=range_mean, scale=range_std)


prob_launch_feasible_from_intercepts_vmap = jax.jit(
    jax.vmap(
        prob_launch_feasible_from_intercepts,
        in_axes=(0, None, None, None),
    )
)


def launch_region_pdf_from_intercepts_find_normalization_constant(
    interception_positions, range_mean, range_std, xlim, ylim, numPoints=200
):
    points, X, Y = bez_from_interceptions.get_meshgrid_points(xlim, ylim, numPoints)
    probs = prob_launch_feasible_from_intercepts_vmap(
        points, interception_positions, range_mean, range_std
    )
    dx = (xlim[1] - xlim[0]) / (numPoints - 1)
    dy = (ylim[1] - ylim[0]) / (numPoints - 1)
    integral = jnp.sum(probs) * dx * dy
    return integral


def launch_region_pdf_from_intercepts(
    points, interception_positions, range_mean, range_std, normalization_constant
):
    probs = prob_launch_feasible_from_intercepts_vmap(
        points, interception_positions, range_mean, range_std
    )
    return probs / normalization_constant


def build_launch_region_pdf_range_uncertainty(
    interception_positions, range_mean, range_std, xlim, ylim, numPoints=200
):
    # grid of launch candidates
    integrationPoints, X, Y = bez_from_interceptions.get_meshgrid_points(
        xlim, ylim, numPoints
    )

    # unnormalized weights w(c)
    w = prob_launch_feasible_from_intercepts_vmap(
        integrationPoints, interception_positions, range_mean, range_std
    )

    dx = (xlim[1] - xlim[0]) / (numPoints - 1)
    dy = (ylim[1] - ylim[0]) / (numPoints - 1)
    dArea = dx * dy

    Z = jnp.sum(w) * dArea
    launch_region_pdf = w / Z  # proper pdf: sum(pdf)*dArea ≈ 1

    return integrationPoints, launch_region_pdf, dArea, X, Y


def prob_reachable_given_pdf(
    point,
    integrationPoints,
    launch_region_pdf,
    range_mean,
    range_std,
    dArea,
):
    dists = jnp.linalg.norm(integrationPoints - point, axis=1)
    # survival function = P(R >= d)
    # probs = jax.scipy.stats.norm.sf(dists, loc=range_mean, scale=range_std)
    probs = 1.0 - jax.scipy.stats.norm.cdf(dists, loc=range_mean, scale=range_std)

    return jnp.sum(probs * launch_region_pdf) * dArea


prob_reachable_given_pdf_grad = jax.jacfwd(prob_reachable_given_pdf, argnums=0)

prob_reachable_given_pdf_grad_vmap = jax.jit(
    jax.vmap(prob_reachable_given_pdf_grad, in_axes=(0, None, None, None, None, None))
)


prob_reachable_given_pdf_vmap = jax.jit(
    jax.vmap(
        prob_reachable_given_pdf,
        in_axes=(0, None, None, None, None, None),
    )
)


def pez_from_launch_region_pdf(
    points,
    evaderHeading,
    evaderSpeed,
    pursuerSpeed,
    integrationPoints,
    launch_region_pdf,
    range_mean,
    range_std,
    dArea,
):
    futureEvaderPositions = (
        points
        + (evaderSpeed / pursuerSpeed)
        * range_mean
        * jnp.vstack([jnp.cos(evaderHeading), jnp.sin(evaderHeading)]).T
    )
    return prob_reachable_given_pdf_vmap(
        futureEvaderPositions,
        integrationPoints,
        launch_region_pdf,
        range_mean,
        range_std,
        dArea,
    )


def uniform_pdf_from_interception_points(
    points,  # (N,2)
    interceptionPositions,  # (M,2)
    pursuerRange,
    pursuerCaptureRadius,
    dA,  # integration area element
):
    """
    Compute a uniform PDF over the intersection of M discs centered at
    the interception positions, with radius (pursuerRange + pursuerCaptureRadius).

    points: (N,2) evaluation/integration grid
    interceptionPositions: (M,2) centers of discs
    pursuerRange, pursuerCaptureRadius: scalars
    dA: area element for numerical integration

    Returns:
        pdf: (N,) array, uniform over the intersection region
    """

    R_eff = pursuerRange + pursuerCaptureRadius  # effective radius

    # distances: (N, M)
    diff = points[:, None, :] - interceptionPositions[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)

    # inside intersection if inside ALL discs
    inside_all = jnp.all(dists <= R_eff, axis=1)  # (N,)

    # numeric area of intersection region
    area = jnp.sum(inside_all) * dA

    # uniform pdf: 1/area inside, 0 outside
    pdf = jnp.where(inside_all, 1.0 / area, 0.0)

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


def plot_potential_pursuer_launch_with_range_uncertainty():
    pursuerRangeMean = 1.5
    pursuerRangeStd = 0.3
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.0
    evaderHeading = np.pi / 4
    evaderSpeed = 1.0
    # interceptionPositions = np.array([[0.4, 0.5]])
    interceptionPositions = np.array([[0.4, 0.5]])
    interceptionPositions = np.array([[0.4, 0.5], [-1.2, -1.2]])
    interceptionPositions = np.array([[0.4, 0.5], [-1.2, -1.2], [-0.7, 0.9]])

    numPoints = 150
    xlim = (-4, 4)
    ylim = (-4, 4)
    points, X, Y = bez_from_interceptions.get_meshgrid_points(xlim, ylim, numPoints)
    dArea = (
        (xlim[1] - xlim[0]) / (numPoints - 1) * (ylim[1] - ylim[0]) / (numPoints - 1)
    )
    normalization_constant = (
        launch_region_pdf_from_intercepts_find_normalization_constant(
            interceptionPositions,
            pursuerRangeMean,
            pursuerRangeStd,
            xlim,
            ylim,
            numPoints,
        )
    )
    prob = launch_region_pdf_from_intercepts(
        points,
        interceptionPositions,
        pursuerRangeMean,
        pursuerRangeStd,
        normalization_constant,
    )
    print("Normalization constant:", normalization_constant)
    integral = (
        jnp.sum(prob)
        * (xlim[1] - xlim[0])
        / (numPoints - 1)
        * (ylim[1] - ylim[0])
        / (numPoints - 1)
    )
    print("Integral of PDF over grid:", integral)

    integrationPoints, launch_pdf, dArea, Xint, Yint = (
        build_launch_region_pdf_range_uncertainty(
            interceptionPositions,
            pursuerRangeMean,
            pursuerRangeStd,
            xlim,
            ylim,
            numPoints,
        )
    )
    print("launch pdf integral check:", jnp.sum(launch_pdf) * dArea)
    probReachable = prob_reachable_given_pdf_vmap(
        points,
        integrationPoints,
        launch_pdf,
        pursuerRangeMean,
        pursuerRangeStd,
        dArea,
    )
    probEngagmentZone = pez_from_launch_region_pdf(
        points,
        evaderHeading,
        evaderSpeed,
        pursuerSpeed,
        integrationPoints,
        launch_pdf,
        pursuerRangeMean,
        pursuerRangeStd,
        dArea,
    )
    print("Prob reachable min:", jnp.min(probReachable))
    print("Prob reachable max:", jnp.max(probReachable))
    testPoints = np.random.uniform(-2, 2, (5, 2))
    testProb = prob_reachable_given_pdf_vmap(
        testPoints,
        integrationPoints,
        launch_pdf,
        pursuerRangeMean,
        pursuerRangeStd,
        dArea,
    )
    testProbGrads = prob_reachable_given_pdf_grad_vmap(
        testPoints,
        integrationPoints,
        launch_pdf,
        pursuerRangeMean,
        pursuerRangeStd,
        dArea,
    )
    print("Test point prob reachable:", testProb)
    print("Test point prob reachable grad:", testProbGrads)
    arcs = bez_from_interceptions.compute_potential_pursuer_region_from_interception_position(
        interceptionPositions,
        pursuerRangeMean,
        pursuerCaptureRadius,
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
        pursuerRangeMean,
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
        pursuerRangeMean,
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
        np.ones(len(interceptionPositions)) * pursuerRangeMean,
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


def plot_pez_from_interception():
    pursuerSpeed = 2.0
    pursuerRange = 1.5
    pursuerCaptureRadius = 0.0
    evaderHeading = np.pi / 4
    evaderSpeed = 1.0
    interceptionPositions = np.array([[0.4, 0.5], [-0.8, -0.8], [-0.7, 0.9]])

    numPoints = 160
    xlim = (-4, 4)
    ylim = (-4, 4)
    points, X, Y = bez_from_interceptions.get_meshgrid_points(xlim, ylim, numPoints)

    dArea = (
        (xlim[1] - xlim[0]) / (numPoints - 1) * (ylim[1] - ylim[0]) / (numPoints - 1)
    )

    launch_pdf = uniform_pdf_from_interception_points(
        points, interceptionPositions, pursuerRange, pursuerCaptureRadius, dArea
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
    arcs = bez_from_interceptions.compute_potential_pursuer_region_from_interception_position(
        interceptionPositions,
        pursuerRange,
        pursuerCaptureRadius,
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
        np.ones(len(interceptionPositions)) * pursuerRange,
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
    # fig, axes = plt.subplots(1, 2, figsize=(6.5, 4), layout="constrained")
    # ax = axes[0]
    # ax.set_title(r"$p_{RR}$")
    # ax.set_aspect("equal")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_xlim([-4, 4])
    # ax.set_ylim([-4, 4])
    # ax.set_xticks(np.arange(-4, 5, 1))
    # ax.set_yticks(np.arange(-4, 5, 1))
    # c = ax.contour(
    #     X.reshape((numPoints, numPoints)),
    #     Y.reshape((numPoints, numPoints)),
    #     # prob.reshape((numPoints, numPoints)),
    #     probReachable.reshape((numPoints, numPoints)),
    #     levels=[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
    # )
    # ax.clabel(c, inline=True)
    # bez_from_interceptions.plot_interception_points(
    #     interceptionPositions,
    #     np.ones(len(interceptionPositions)) * (pursuerRange + pursuerCaptureRadius),
    #     ax,
    # )
    # bez_from_interceptions.plot_circle_intersection_arcs(arcs, ax=ax)
    # bez_from_interceptions.plot_potential_pursuer_reachable_region(
    #     arcs, pursuerRange, pursuerCaptureRadius, xlim, ylim, numPoints=200, ax=ax
    # )
    #
    # ax = axes[1]
    # ax.set_title(r"$p_{EZ}$")
    # ax.set_aspect("equal")
    # ax.set_xlabel("X")
    # ax.set_xlim([-4, 4])
    # ax.set_ylim([-4, 4])
    # ax.set_xticks(np.arange(-4, 5, 1))
    # ax.set_yticks(np.arange(-4, 5, 1))
    # c = ax.contour(
    #     X.reshape((numPoints, numPoints)),
    #     Y.reshape((numPoints, numPoints)),
    #     probEngagmentZone.reshape((numPoints, numPoints)),
    #     levels=[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
    # )
    # ax.clabel(c, inline=True)
    # bez_from_interceptions.plot_potential_pursuer_engagement_zone(
    #     arcs,
    #     pursuerRange,
    #     pursuerCaptureRadius,
    #     pursuerSpeed,
    #     evaderHeading,
    #     evaderSpeed,
    #     xlim=(-4, 4),
    #     ylim=(-4, 4),
    #     ax=ax,
    # )
    # fig.legend(
    #     loc="outside lower center",
    #     ncol=5,
    # )
    #
    # fig2, ax2 = plt.subplots(figsize=(6, 6))
    # c = ax2.pcolormesh(
    #     X.reshape((numPoints, numPoints)),
    #     Y.reshape((numPoints, numPoints)),
    #     launch_pdf.reshape((numPoints, numPoints)),
    # )
    # plt.colorbar(c, ax=ax2)
    # ax2.set_aspect("equal")
    # bez_from_interceptions.plot_interception_points(
    #     interceptionPositions,
    #     np.ones(len(interceptionPositions)) * pursuerRange,
    #     ax2,
    # )
    # # plot gradient direction vectors
    # ax2.set_title("Pursuer Launch Region PDF")
    # ax2.set_aspect("equal")
    #
    # plt.show()


if __name__ == "__main__":
    # plot_potential_pursuer_launch_with_range_uncertainty()
    plot_pez_from_interception()
