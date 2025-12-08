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

import GEOMETRIC_BEZ.potential_bez_from_interceptions as potential_bez_from_interceptions


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
    points, X, Y = potential_bez_from_interceptions.get_meshgrid_points(
        xlim, ylim, numPoints
    )
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


def build_launch_region_pdf(
    interception_positions, range_mean, range_std, xlim, ylim, numPoints=200
):
    # grid of launch candidates
    integrationPoints, X, Y = potential_bez_from_interceptions.get_meshgrid_points(
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


def plot_potential_pursuer_launch_with_range_uncertainty():
    pursuerRangeMean = 1.5
    pursuerRangeStd = 0.1
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.0
    evaderHeading = np.pi / 4
    evaderSpeed = 1.0
    # interceptionPositions = np.array([[0.4, 0.5]])
    interceptionPositions = np.array([[0.4, 0.5]])
    interceptionPositions = np.array([[0.4, 0.5], [-1.2, -1.2]])
    interceptionPositions = np.array([[0.4, 0.5], [-1.2, -1.2], [-0.7, 0.9]])

    numPoints = 120
    xlim = (-4, 4)
    ylim = (-4, 4)
    points, X, Y = potential_bez_from_interceptions.get_meshgrid_points(
        xlim, ylim, numPoints
    )
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

    integrationPoints, launch_pdf, dArea, Xint, Yint = build_launch_region_pdf(
        interceptionPositions, pursuerRangeMean, pursuerRangeStd, xlim, ylim, 120
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

    fig, ax = plt.subplots(figsize=(6, 6))
    c = ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        # prob.reshape((numPoints, numPoints)),
        probReachable.reshape((numPoints, numPoints)),
        levels=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    ax.clabel(c, inline=True)
    # c = ax.pcolormesh(
    #     X.reshape((numPoints, numPoints)),
    #     Y.reshape((numPoints, numPoints)),
    #     probReachable.reshape((numPoints, numPoints)),
    # )
    # plt.colorbar(c, ax=ax)
    ax.set_aspect("equal")
    # plot_interception_points(
    #     interceptionPositions,
    #     np.ones(len(interceptionPositions)) * pursuerRangeMean,
    #     ax,
    # )
    arcs = potential_bez_from_interceptions.compute_potential_pursuer_region_from_interception_position(
        interceptionPositions,
        pursuerRangeMean,
        pursuerCaptureRadius,
    )

    potential_bez_from_interceptions.plot_potential_pursuer_reachable_region(
        arcs, pursuerRangeMean, pursuerCaptureRadius, xlim=(-4, 4), ylim=(-4, 4), ax=ax
    )
    for testPoint, testProbGrad in zip(testPoints, testProbGrads):
        ax.arrow(
            testPoint[0],
            testPoint[1],
            testProbGrad[0],
            testProbGrad[1],
            head_width=0.1,
            color="black",
        )
    ax.set_title("Probability Pursuer Can Reach Evader")

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    c = ax2.pcolormesh(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        launch_pdf.reshape((numPoints, numPoints)),
    )
    plt.colorbar(c, ax=ax2)
    ax2.set_aspect("equal")
    potential_bez_from_interceptions.plot_interception_points(
        interceptionPositions,
        np.ones(len(interceptionPositions)) * pursuerRangeMean,
        ax2,
    )
    # plot gradient direction vectors
    ax2.set_title("Pursuer Launch Region PDF")
    ax2.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    plot_potential_pursuer_launch_with_range_uncertainty()
