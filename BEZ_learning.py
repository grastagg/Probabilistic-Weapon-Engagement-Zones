import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from curlyBrace import curlyBrace
from matplotlib.patches import Arc, Circle
import fast_pursuer

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


def in_circle(center, raduis, points):
    dists = jnp.linalg.norm(points - center, axis=1)
    return dists <= raduis


in_circle_vmap = jax.vmap(in_circle, in_axes=(0, None, None))


def in_circle_intersection(centers, radius, points):
    in_circles = in_circle_vmap(centers, radius, points)
    return jnp.all(in_circles, axis=0)


def _circle_intersections(c1, r1, c2, r2, tol=1e-9):
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    d = np.linalg.norm(c2 - c1)
    if d < tol or d > r1 + r2 + tol or d < abs(r1 - r2) - tol:
        return np.empty((0, 2))
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h2 = r1**2 - a**2
    if h2 < -tol:
        return np.empty((0, 2))
    h = np.sqrt(max(h2, 0.0))
    p0 = c1 + a * (c2 - c1) / d
    if h <= tol:
        return p0[None, :]
    perp = np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / d
    return np.vstack([p0 + h * perp, p0 - h * perp])


def _wrap_angle(theta):
    return np.mod(theta, 2 * np.pi)


def _ccw_delta(t1, t2):
    return (t2 - t1) % (2 * np.pi)


def _inside_all(p, centers, radii, tol=1e-9):
    for c, r in zip(centers, radii):
        if np.linalg.norm(p - c) > r + tol:
            return False
    return True


def _unique(pts, tol=1e-8):
    out = []
    for p in pts:
        if all(np.linalg.norm(p - q) > tol for q in out):
            out.append(p)
    return out


def intersection_arcs(centers, radii, tol=1e-9):
    """
    Compute arcs defining the boundary of the intersection of multiple circles.

    Returns
    -------
    arcs : list of dicts
        Each arc = {
            'center': np.ndarray(2,),
            'radius': float,
            'theta_start': float,  # radians, CCW
            'theta_end': float,    # radians (unwrapped)
        }
    """
    n = len(centers)
    centers = [np.asarray(c, float) for c in centers]
    radii = np.asarray(radii, float)

    # Gather candidate intersection vertices
    verts = []
    for i in range(n):
        for j in range(i + 1, n):
            pts = _circle_intersections(centers[i], radii[i], centers[j], radii[j], tol)
            for p in pts:
                if _inside_all(p, centers, radii, tol):
                    verts.append(p)
    verts = _unique(verts, tol)

    # Handle full containment or empty intersection
    if len(verts) == 0:
        for i in range(n):
            if all(
                np.linalg.norm(centers[i] - centers[j]) + radii[i] <= radii[j] + tol
                for j in range(n)
                if j != i
            ):
                return [
                    {
                        "center": centers[i],
                        "radius": radii[i],
                        "theta_start": 0.0,
                        "theta_end": 2 * np.pi,
                    }
                ]
        return []

    arcs = []
    for i in range(n):
        c, r = centers[i], radii[i]
        # vertices lying on this circle
        pts_i = [p for p in verts if abs(np.linalg.norm(p - c) - r) <= tol]
        if len(pts_i) < 2:
            continue

        thetas = np.array([np.arctan2(p[1] - c[1], p[0] - c[0]) for p in pts_i])
        order = np.argsort(thetas)
        pts_i = [pts_i[k] for k in order]
        thetas = [float(thetas[k]) for k in order]

        m = len(pts_i)
        for k in range(m):
            t1 = thetas[k]
            t2 = thetas[(k + 1) % m]
            dtheta = _ccw_delta(t1, t2)
            mid = c + r * np.array([np.cos(t1 + dtheta / 2), np.sin(t1 + dtheta / 2)])
            if _inside_all(mid, centers, radii, tol):
                arcs.append(
                    {
                        "center": c,
                        "radius": r,
                        "theta_start": _wrap_angle(t1),
                        "theta_end": t1 + dtheta,  # unwrapped
                    }
                )
    return arcs


def is_between_angles_radians(start_angle, stop_angle, angle):
    """
    Checks if a third angle lies between a start and stop angle counter-clockwise (radians).

    Args:
        start_angle: The starting angle in radians (0 to 2*pi).
        stop_angle: The stopping angle in radians (0 to 2*pi).
        angle: The angle to check if it lies between start and stop (0 to 2*pi).

    Returns:
        True if the angle lies between start and stop counter-clockwise, False otherwise.
    """

    # Normalize angles to 0-2*pi range
    start_angle = start_angle % (2 * jnp.pi)
    stop_angle = stop_angle % (2 * jnp.pi)
    angle = angle % (2 * jnp.pi)

    flag = stop_angle < start_angle
    same = jnp.isclose(start_angle, stop_angle, rtol=1e-9)

    def same_case():
        return jnp.ones_like(angle, dtype=bool)

    def not_same_case():
        def true_case():
            return jnp.logical_or(angle > start_angle, angle < stop_angle)

        def false_case():
            return jnp.logical_and(start_angle < angle, angle < stop_angle)

        return jax.lax.cond(flag, true_case, false_case)

    return jax.lax.cond(same, same_case, not_same_case)


def dist_point_to_arc(point, center, radius, theta1, theta2):
    endPointA = center + radius * jnp.array([jnp.cos(theta1), jnp.sin(theta1)])
    endPointB = center + radius * jnp.array([jnp.cos(theta2), jnp.sin(theta2)])

    theta = jnp.arctan2(point[1] - center[1], point[0] - center[0])
    inRange = is_between_angles_radians(theta1, theta2, theta)

    dist_A = jnp.linalg.norm(point - endPointA)
    dist_B = jnp.linalg.norm(point - endPointB)

    def in_range_case():
        intersectionPoint = (
            center + radius * jnp.array([jnp.cos(theta), jnp.sin(theta)]).T
        )
        return jnp.linalg.norm(intersectionPoint - point)

    def out_of_range_case():
        return jnp.minimum(dist_A, dist_B)

    return jax.lax.cond(inRange, in_range_case, out_of_range_case)


dists_point_to_arcs = jax.vmap(dist_point_to_arc, in_axes=(None, 0, 0, 0, 0))


def dist_point_to_arcs(point, centers, radii, theta_start, theta_end):
    dists = dists_point_to_arcs(point, centers, radii, theta_start, theta_end)
    min_dist = jnp.min(dists)
    min_index = jnp.argmin(dists)
    return min_dist, min_index


signed_distance_to_arcs_vmap = jax.vmap(
    dist_point_to_arcs, in_axes=(0, None, None, None, None)
)


def potential_reachable_region(
    points, centers, radii, theta_start, theta_end, pursuerRange, pursuerCaptureRadius
):
    dists, _ = signed_distance_to_arcs_vmap(
        points, centers, radii, theta_start, theta_end
    )
    return dists - (pursuerRange + pursuerCaptureRadius)


def potential_pursuer_engagment_zone(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    centers,
    radii,
    theta_start,
    theta_end,
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
    ez = potential_reachable_region(
        futureEvaderPositions,
        centers,
        radii,
        theta_start,
        theta_end,
        pursuerRange,
        pursuerCaptureRadius,
    )
    return ez


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


def plot_in_circle_intersection(centers, radaii, fig, ax):
    numPoints = 500
    x = jnp.linspace(-5, 5, numPoints)
    y = jnp.linspace(-5, 5, numPoints)
    [X, T] = jnp.meshgrid(x, y)
    points = jnp.vstack((X.flatten(), T.flatten())).T
    in_intersection = in_circle_intersection(centers, radaii, points)
    ax.set_aspect("equal")
    ax.pcolormesh(
        X.reshape((numPoints, numPoints)),
        T.reshape((numPoints, numPoints)),
        in_intersection.reshape((numPoints, numPoints)),
    )
    ax.scatter(centers[:, 0], centers[:, 1], color="red")


def plot_pursuer_reachable_region(
    pursuerPosition, pursuerRange, pursuerCaptureRadius, fig, ax
):
    circle = plt.Circle(
        (pursuerPosition[0], pursuerPosition[1]),
        pursuerRange + pursuerCaptureRadius,
        color="blue",
        fill=False,
        linestyle="--",
    )
    ax.add_artist(circle)
    ax.scatter(
        pursuerPosition[0],
        pursuerPosition[1],
        color="blue",
        marker="o",
        label="Pursuer Position",
    )
    ax.plot([], color="blue", label="True Pursuer Reachable Region")


def plot_circle_intersection_arcs(
    arcs, centers=None, radii=None, show_circles=True, ax=None
):
    """
    Plot arcs returned by circle_intersection_arcs().

    Parameters
    ----------
    arcs : list of dict
        Output from circle_intersection_arcs().
    centers : list of (2,), optional
        Circle centers, only needed if show_circles=True.
    radii : list of float, optional
        Circle radii, only needed if show_circles=True.
    show_circles : bool, optional
        If True, o  verlay the full circle outlines and centers.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; one is created if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """

    # Plot the intersection arcs
    for a in arcs:
        r = a["radius"]
        c = a["center"]

        # Compute CCW extent (Matplotlib uses degrees)
        t1, t2 = np.degrees(a["theta_start"]), np.degrees(a["theta_end"])
        extent = (t2 - t1) % 360  # ensure positive CCW extent

        arc_patch = Arc(
            c, 2 * r, 2 * r, angle=0, theta1=t1, theta2=t1 + extent, color="red", lw=2
        )
        ax.add_patch(arc_patch)

    # proxy for legend
    ax.plot([], [], color="red", lw=2, label="Potential Pursuer Position")

    # Optionally show endpoints

    return ax


def arcs_to_arrays(arcs):
    """
    Convert list of arc dictionaries into NumPy arrays suitable for
    signed_distance_to_arcs_jax().

    Parameters
    ----------
    arcs : list of dict
        Each arc has keys:
          {
            'center': array_like, shape (2,),
            'radius': float,
            'theta_start': float,
            'theta_end': float
          }

    Returns
    -------
    centers : ndarray, shape (N, 2)
    radii : ndarray, shape (N,)
    theta_start : ndarray, shape (N,)
    theta_end : ndarray, shape (N,)
    """
    N = len(arcs)
    if N == 0:
        raise ValueError("No arcs provided — intersection is empty.")

    centers = np.stack([np.asarray(a["center"], float) for a in arcs])
    radii = np.array([a["radius"] for a in arcs], float)
    theta_start = np.array([a["theta_start"] for a in arcs], float)
    theta_end = np.array([a["theta_end"] for a in arcs], float)

    return centers, radii, theta_start, theta_end


def plot_potential_pursuer_engagement_zone(
    arcs,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    evaderHeading,
    evaderSpeed,
    xlim,
    ylim,
    numPoints=200,
    ax=None,
):
    centers, radii, theta_start, theta_end = arcs_to_arrays(arcs)
    x = jnp.linspace(xlim[0], xlim[1], numPoints)
    y = jnp.linspace(ylim[0], ylim[1], numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    points = jnp.vstack((X.flatten(), Y.flatten())).T
    headings = evaderHeading * jnp.ones((points.shape[0],))
    ez = potential_pursuer_engagment_zone(
        points,
        headings,
        evaderSpeed,
        centers,
        radii,
        theta_start,
        theta_end,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    # c = ax.pcolormesh(
    #     X.reshape((numPoints, numPoints)),
    #     Y.reshape((numPoints, numPoints)),
    #     dists.reshape((numPoints, numPoints)),
    # )
    ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        ez.reshape((numPoints, numPoints)),
        levels=[0],
        colors="green",
        linestyles="--",
    )
    ax.plot(
        [], color="green", label="Potential Pursuer Engagement Zone", linestyle="--"
    )
    # plt.colorbar(c, ax=ax, label="Signed Distance")


def plot_potential_pursuer_reachable_region(
    arcs, pursuerRange, pursuerCaptureRadius, xlim, ylim, numPoints=200, ax=None
):
    centers, radii, theta_start, theta_end = arcs_to_arrays(arcs)
    x = jnp.linspace(xlim[0], xlim[1], numPoints)
    y = jnp.linspace(ylim[0], ylim[1], numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    points = jnp.vstack((X.flatten(), Y.flatten())).T
    rr = potential_reachable_region(
        points,
        centers,
        radii,
        theta_start,
        theta_end,
        pursuerRange,
        pursuerCaptureRadius,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    # c = ax.pcolormesh(
    #     X.reshape((numPoints, numPoints)),
    #     Y.reshape((numPoints, numPoints)),
    #     dists.reshape((numPoints, numPoints)),
    # )
    ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        rr.reshape((numPoints, numPoints)),
        levels=[0],
        colors="magenta",
    )
    ax.plot([], color="magenta", label="Potential Pursuer Reachable Region")
    # plt.colorbar(c, ax=ax, label="Signed Distance")


def plot_interception_points(interceptionPositions, pursuerRange, ax):
    ax.scatter(
        interceptionPositions[:, 0],
        interceptionPositions[:, 1],
        color="red",
        label="Interception Locations",
        marker="x",
    )
    for pos in interceptionPositions:
        Circle = plt.Circle(pos, pursuerRange, color="red", fill=False, linestyle=":")
        ax.add_artist(Circle)


def plot_box_pursuer_reachable_region(
    min_box,
    max_box,
    pursuerRange,
    pursuerCaptureRadius,
    ax,
    color="magenta",
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
    numPoints = 200
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
    ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        RR.reshape((numPoints, numPoints)),
        levels=[0],
        colors=color,
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
        colors="green",
    )
    ax.plot([], label=r"$\partial \mathcal{Z}_{\text{rect}}$", color="green")
    return ax


def annotate_box_EZ_plot(
    evader_start, evader_heading, evader_speed, pursuerSpeed, pursuerRange, ax, fig
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
        [0, 2.6],
        [0, 1],
        k_r=0.1,
        bool_auto=True,
        str_text=r"$R+r$",
        int_line_num=2,
        fontdict={},
        color="black",
    )


def main():
    pursuerRange = 1.5
    pursuerPosition = np.array([0.0, 0.0])
    interceptionPositions = np.array([[1.0, 1.0]])
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.1
    evaderHeading = 0.0
    evaderSpeed = 1.5

    interceptionPositions = np.array([[1.0, 1.0], [-1.0, -1.0]])
    # interceptionPositions = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    # interceptionPositions = np.array(
    #     [[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]
    # )
    # interceptionPositions = np.random.uniform(-1, 1, (2, 2))
    arcs = intersection_arcs(
        interceptionPositions,
        [pursuerRange + pursuerCaptureRadius] * np.ones(len(interceptionPositions)),
    )
    print("arcs:", arcs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    # plot_in_circle_intersection(interceptionPositions, pursuerRange, fig, ax)
    plot_potential_pursuer_reachable_region(
        arcs, pursuerRange, pursuerCaptureRadius, xlim=(-4, 4), ylim=(-4, 4), ax=ax
    )
    plot_potential_pursuer_engagement_zone(
        arcs,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        evaderHeading,
        evaderSpeed,
        xlim=(-4, 4),
        ylim=(-4, 4),
        ax=ax,
    )
    plot_pursuer_reachable_region(
        pursuerPosition, pursuerRange, pursuerCaptureRadius, fig, ax
    )
    fast_pursuer.plotEngagementZone(
        evaderHeading,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
    )
    plot_interception_points(
        interceptionPositions, pursuerRange + pursuerCaptureRadius, ax
    )
    plot_circle_intersection_arcs(arcs, ax=ax)
    plt.legend()
    plt.show()


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
    plt.show()


if __name__ == "__main__":
    bez_learning_rect_ez_plot()
