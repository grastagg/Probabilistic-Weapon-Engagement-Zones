from jax._src.source_info_util import raw_frame_to_frame
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
import GEOMETRIC_BEZ.rectangle_bez as rectangle_bez


def circle_rectangle_intersection_area(center, R, min_box, max_box, num_quad=512):
    """
    Compute area of intersection between a circle and an axis-aligned rectangle.

    center: (2,) array-like, circle center [x_c, y_c]
    R: scalar, circle radius
    min_box: (2,) [xmin, ymin]
    max_box: (2,) [xmax, ymax]
    num_quad: number of x-samples for 1D quadrature
    """
    x_c, y_c = center
    xmin, ymin = min_box
    xmax, ymax = max_box

    # x samples along the rectangle width
    xs = jnp.linspace(xmin, xmax, num_quad)

    dx = xs - x_c
    # Only x within radius contribute
    inside_x = jnp.abs(dx) <= R

    # For those x, circle intersects as y = y_c ± sqrt(R^2 - dx^2)
    dx = xs - x_c
    # inside_x = jnp.abs(dx) <= R
    #
    # sqrt_arg = R**2 - dx**2
    # sqrt_arg = jnp.maximum(sqrt_arg, 0.0)  # <-- key line for AD / jacfwd
    #
    # y_half = jnp.where(inside_x, jnp.sqrt(sqrt_arg), 0.0)
    y_half = jnp.where(inside_x, jnp.sqrt(R**2 - dx**2), 0.0)
    y_low_circ = y_c - y_half
    y_high_circ = y_c + y_half

    # Intersection of [y_low_circ, y_high_circ] with [ymin, ymax]
    y_low_int = jnp.maximum(y_low_circ, ymin)
    y_high_int = jnp.minimum(y_high_circ, ymax)

    # Vertical intersection length; zero if no overlap
    length_y = jnp.maximum(0.0, y_high_int - y_low_int)
    # eps = 1e-2  # smoothing width
    # length_y = jax.nn.softplus((y_high_int - y_low_int) / eps) * eps

    # 1D integral over x using trapezoidal rule
    area = jnp.trapezoid(length_y, xs)

    #     "\n"
    #     "================ CIRCLE DEBUG ================\n"
    #     "center             = {center}\n"
    #     "R                  = {R}\n"
    #     "min_box            = {min_box}\n"
    #     "max_box            = {max_box}\n"
    #     "box width          = {w}\n"
    #     "box height         = {h}\n"
    #     "\n"
    #     "dx min/max         = {dxmin} / {dxmax}\n"
    #     "\n"
    #     "y_half min/max     = {yhmin} / {yhmax}\n"
    #     "y_half NaN?        = {yhnan}\n"
    #     "\n"
    #     "y_low_circ NaN?    = {ylcnan}\n"
    #     "y_high_circ NaN?   = {yhcnan}\n"
    #     "\n"
    #     "length_y min/max   = {lymin} / {lymax}\n"
    #     "length_y NaN?      = {lynan}\n"
    #     "\n"
    #     "area               = {area}\n"
    #     "area NaN?          = {areanan}\n"
    #     "==============================================\n",
    #     center=center,
    #     R=R,
    #     min_box=min_box,
    #     max_box=max_box,
    #     w=xmax - xmin,
    #     h=ymax - ymin,
    #     dxmin=jnp.min(dx),
    #     dxmax=jnp.max(dx),
    #     yhmin=jnp.min(y_half),
    #     yhmax=jnp.max(y_half),
    #     yhnan=jnp.isnan(y_half).any(),
    #     ylcnan=jnp.isnan(y_low_circ).any(),
    #     yhcnan=jnp.isnan(y_high_circ).any(),
    #     lymin=jnp.min(length_y),
    #     lymax=jnp.max(length_y),
    #     lynan=jnp.isnan(length_y).any(),
    #     area=area,
    #     areanan=jnp.isnan(area),
    # )

    return area


@jax.jit
def prob_reachable_uniform_box(
    points,
    pursuerRange,
    pursuerCaptureRadius,
    min_box,
    max_box,
):
    """
    points: (N, 2) evader positions (or evaluation points)
    min_box, max_box: (2,) launch box corners
    R: scalar, pursuerRange + captureRadius
    num_quad: number of quadrature points in x

    Returns: (N,) probabilities
    """
    num_quad = 1024
    R = pursuerRange + pursuerCaptureRadius
    min_box = jnp.asarray(min_box)
    max_box = jnp.asarray(max_box)

    area_box = (max_box[0] - min_box[0]) * (max_box[1] - min_box[1])

    def single_prob(point):
        area = circle_rectangle_intersection_area(point, R, min_box, max_box, num_quad)
        return area / area_box

    return jax.vmap(single_prob)(points)


def prob_engagment_zone_uniform_box(
    points,
    headings,
    evaderSpeed,
    pursuerSpeed,
    pursuerRange,
    pursuerCaptureRadius,
    min_box,
    max_box,
):
    futureEvaderPositions = (
        points
        + (evaderSpeed / pursuerSpeed)
        * pursuerRange
        * jnp.vstack([jnp.cos(headings), jnp.sin(headings)]).T
    )
    return prob_reachable_uniform_box(
        futureEvaderPositions,
        pursuerRange,
        pursuerCaptureRadius,
        min_box,
        max_box,
    )


dPEZdPos = jax.jacfwd(prob_engagment_zone_uniform_box)
dPEZdHeading = jax.jacfwd(prob_engagment_zone_uniform_box, argnums=1)


def plot_rectangle_prr(
    pursuerRange,
    pursuerCaptureRadius,
    min_box,
    max_box,
    xlim,
    ylim,
    ax,
    numPoints=120,
):
    points, X, Y = bez_from_interceptions.get_meshgrid_points(xlim, ylim, numPoints)
    probReachable = prob_reachable_uniform_box(
        points, pursuerRange, pursuerCaptureRadius, min_box, max_box
    )
    c = ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        probReachable.reshape((numPoints, numPoints)),
        levels=[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
    )
    ax.clabel(c, inline=True)


def rectangle_pez_plot():
    pursuerRange = 1.5
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.1
    evaderHeading = np.pi / 4
    evaderSpeed = 1.0
    min_box = np.array([-1.0, -1.0])
    max_box = np.array([2.0, 1.0])

    numPoints = 120
    xlim = (-4, 4)
    ylim = (-4, 4)
    points, X, Y = bez_from_interceptions.get_meshgrid_points(xlim, ylim, numPoints)
    dArea = (
        (xlim[1] - xlim[0]) / (numPoints - 1) * (ylim[1] - ylim[0]) / (numPoints - 1)
    )

    probReachable = prob_reachable_uniform_box(
        points, pursuerRange, pursuerCaptureRadius, min_box, max_box
    )
    headings = evaderHeading * jnp.ones(points.shape[0])
    probEngagmentZone = prob_engagment_zone_uniform_box(
        points,
        headings,
        evaderSpeed,
        pursuerSpeed,
        pursuerRange,
        pursuerCaptureRadius,
        min_box,
        max_box,
    )

    # test gradients
    gradPoints = np.random.uniform(-4, 4, (10, 2))
    grads = dPEZdPos(
        gradPoints,
        headings[:10],
        evaderSpeed,
        pursuerSpeed,
        pursuerRange,
        pursuerCaptureRadius,
        min_box,
        max_box,
    )
    headings_grad = dPEZdHeading(
        gradPoints,
        headings[:10],
        evaderSpeed,
        pursuerSpeed,
        pursuerRange,
        pursuerCaptureRadius,
        min_box,
        max_box,
    )
    print("Gradient samples:", grads)
    print("headings", headings_grad)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4), layout="constrained")
    ax = axes[0]
    ax.set_title(r"$p_{RR}$")
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks(np.arange(-4, 5, 1))
    c = ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        # prob.reshape((numPoints, numPoints)),
        probReachable.reshape((numPoints, numPoints)),
        levels=[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
    )
    ax.clabel(c, inline=True)

    rectangle_bez.plot_box_pursuer_reachable_region(
        min_box, max_box, pursuerRange, pursuerCaptureRadius, ax=ax
    )

    ax = axes[1]
    ax.set_title(r"$p_{EZ}$")
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks(np.arange(-4, 5, 1))
    c = ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        probEngagmentZone.reshape((numPoints, numPoints)),
        levels=[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
    )
    ax.clabel(c, inline=True)
    rectangle_bez.plot_box_pursuer_engagement_zone(
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        evaderHeading,
        evaderSpeed,
        xlim,
        ylim,
        ax,
    )
    fig.legend(
        loc="outside lower center",
        ncol=3,
    )
    for i, gradient in enumerate(grads):
        print("Gradient:", gradient[i])
        # draw gradient arrows at random points
        ax.arrow(
            gradPoints[i, 0],
            gradPoints[i, 1],
            gradient[i][0],
            gradient[i][1],
            head_width=0.1,
            head_length=0.1,
            fc="k",
            ec="k",
        )

    plt.show()


if __name__ == "__main__":
    rectangle_pez_plot()
