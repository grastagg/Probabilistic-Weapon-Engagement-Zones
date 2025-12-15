import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib

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

from PLOT_COMMON.curlyBrace import curlyBrace


def signed_distance_to_box(point, box_min, box_max):
    """
    Signed distance to an axis-aligned box defined by min/max corners.
    Positive outside, negative inside.

    Args:
        point: (2,) array [x, y]
        box_min: (2,) array [xmin, ymin]
        box_max: (2,) array [xmax, ymax]
    """
    # Distance to box boundaries
    d_out = jnp.maximum(jnp.maximum(box_min - point, point - box_max), 0.0)
    outside = jnp.linalg.norm(d_out)
    inside = jnp.minimum(
        jnp.maximum(jnp.max(box_min - point), jnp.max(point - box_max)), 0.0
    )
    return outside + inside


signed_distance_to_box_vmap = jax.jit(
    jax.vmap(signed_distance_to_box, in_axes=(0, None, None))
)


def box_reachable_region(points, box_min, box_max, pursuerRange, pursuerCaptureRadius):
    dists = signed_distance_to_box_vmap(points, box_min, box_max)
    return dists - (pursuerRange + pursuerCaptureRadius)


def box_pursuer_engagment_zone(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    box_min,
    box_max,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
):
    speedRatio = evaderSpeed / pursuerSpeed
    futureEvaderPositions = (
        evaderPositions
        + speedRatio
        * pursuerRange
        * jnp.vstack([jnp.cos(evaderHeadings), jnp.sin(evaderHeadings)]).T
    )
    ez = box_reachable_region(
        futureEvaderPositions,
        box_min,
        box_max,
        pursuerRange,
        pursuerCaptureRadius,
    )
    return ez


def get_meshgrid_points(xlim, ylim, numPoints):
    x = jnp.linspace(xlim[0], xlim[1], numPoints)
    y = jnp.linspace(ylim[0], ylim[1], numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    return jnp.vstack((X.flatten(), Y.flatten())).T, X, Y


def plot_box_pursuer_reachable_region(
    min_box,
    max_box,
    pursuerRange,
    pursuerCaptureRadius,
    ax,
    color="magenta",
    linestyle="solid",
    fill=False,
    alpha=1.0,
    numPoints=200,
):
    box = plt.Rectangle(
        min_box,
        max_box[0] - min_box[0],
        max_box[1] - min_box[1],
        color="red",
        fill=False,
        linewidth=1.5,
    )
    ax.plot(
        [],
        color="red",
        label=r"$\partial \mathcal{P}_{\text{rect}}$",
    )
    ax.add_artist(box)
    xlim = (
        min_box[0] - (pursuerRange + pursuerCaptureRadius) - 0.3,
        max_box[0] + (pursuerRange + pursuerCaptureRadius) + 0.3,
    )
    ylim = (
        min_box[1] - (pursuerRange + pursuerCaptureRadius) - 0.3,
        max_box[1] + (pursuerRange + pursuerCaptureRadius) + 0.3,
    )
    points, X, Y = get_meshgrid_points(xlim, ylim, numPoints)

    RR = box_reachable_region(
        points, min_box, max_box, pursuerRange, pursuerCaptureRadius
    )
    if fill:
        ax.contourf(
            X.reshape((numPoints, numPoints)),
            Y.reshape((numPoints, numPoints)),
            RR.reshape((numPoints, numPoints)),
            levels=[-1000, 0],
            colors=color,
            linestyles=linestyle,
            alpha=alpha,
        )
    else:
        ax.contour(
            X.reshape((numPoints, numPoints)),
            Y.reshape((numPoints, numPoints)),
            RR.reshape((numPoints, numPoints)),
            levels=[0],
            colors=color,
            linestyles=linestyle,
            alpha=alpha,
        )
    ax.plot([], color=color, label=r"$\partial \mathcal{R}_{\text{rect}}$")
    return ax


def plot_box_pursuer_engagement_zone(
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
    color="green",
):
    numPoints = 200
    points, X, Y = get_meshgrid_points(xlim, ylim, numPoints)
    evaderHeadings = evaderHeading * jnp.ones((points.shape[0],))

    EZ = box_pursuer_engagment_zone(
        points,
        evaderHeadings,
        evaderSpeed,
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )
    ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        EZ.reshape((numPoints, numPoints)),
        levels=[0],
        colors=color,
    )
    ax.plot([], label=r"$\partial \mathcal{Z}_{\text{rect}}$", color="green")
    return ax


def annotate_box_EZ_plot(
    evader_start,
    evader_heading,
    evader_speed,
    pursuerSpeed,
    pursuerRange,
    ax,
    fig,
    p1=[0, 2.6],
    p2=[0, 1],
):
    ax.plot(evader_start[0], evader_start[1], "k")
    speedRatio = evader_speed / pursuerSpeed
    dist = speedRatio * pursuerRange
    ax.arrow(
        evader_start[0],
        evader_start[1],
        dist * jnp.cos(evader_heading),
        dist * jnp.sin(evader_heading),
        color="black",
        head_width=0.1,
        length_includes_head=True,
    )
    ax.text(
        (evader_start[0] + evader_start[0] + dist * np.cos(evader_heading)) / 2,
        (evader_start[1] + evader_start[1] + dist * np.sin(evader_heading)) / 2 - 0.25,
        r"$\mu R \hat{\mathbf{v}}_E$",
        color="black",
    )
    curlyBrace(
        fig,
        ax,
        p1,
        p2,
        k_r=0.1,
        bool_auto=True,
        str_text=r"$R+r$",
        int_line_num=2,
        fontdict={},
        color="black",
    )


def bez_learning_rect_ez_plot():
    pursuerRange = 1.5
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.1
    evaderHeading = np.pi / 4
    evaderSpeed = 1.0
    min_box = np.array([-1.0, -1.0])
    max_box = np.array([2.0, 1.0])

    fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plot_box_pursuer_reachable_region(
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        ax=ax,
    )
    plot_box_pursuer_engagement_zone(
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        evaderHeading,
        evaderSpeed,
        xlim=(-4, 4),
        ylim=(-4, 4),
        ax=ax,
    )
    annotate_box_EZ_plot(
        [-2.67, -2.67], evaderHeading, evaderSpeed, pursuerSpeed, pursuerRange, ax, fig
    )
    plt.legend(ncols=3, loc="upper center")


if __name__ == "__main__":
    bez_learning_rect_ez_plot()
    plt.show()
