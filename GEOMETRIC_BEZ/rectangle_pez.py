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
    y_half = jnp.where(inside_x, jnp.sqrt(R**2 - dx**2), 0.0)
    y_low_circ = y_c - y_half
    y_high_circ = y_c + y_half

    # Intersection of [y_low_circ, y_high_circ] with [ymin, ymax]
    y_low_int = jnp.maximum(y_low_circ, ymin)
    y_high_int = jnp.minimum(y_high_circ, ymax)

    # Vertical intersection length; zero if no overlap
    length_y = jnp.maximum(0.0, y_high_int - y_low_int)

    # 1D integral over x using trapezoidal rule
    area = jnp.trapezoid(length_y, xs)
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

    plt.show()


if __name__ == "__main__":
    rectangle_pez_plot()
