import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from matplotlib.patches import Arc, Circle

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


def plot_pursuer_reachable_region(pursuerPosition, pursuerRange, fig, ax):
    circle = plt.Circle(
        (pursuerPosition[0], pursuerPosition[1]),
        pursuerRange,
        color="blue",
        fill=False,
        linestyle="--",
    )
    ax.add_artist(circle)


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
        If True, overlay the full circle outlines and centers.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; one is created if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")

    # Plot full circles (optional)
    if show_circles and centers is not None and radii is not None:
        for i, (c, r) in enumerate(zip(centers, radii)):
            circ = Circle(c, r, fill=False, ls="--", color="gray", alpha=0.5)
            ax.add_patch(circ)
            ax.plot(*c, "ko", ms=4)
            ax.text(
                c[0], c[1], f"C{i}", ha="center", va="center", fontsize=8, color="k"
            )

    # Plot the intersection arcs
    for a in arcs:
        r = a["radius"]
        c = a["center"]

        # Compute CCW extent (Matplotlib uses degrees)
        t1, t2 = np.degrees(a["theta_start"]), np.degrees(a["theta_end"])
        extent = (t2 - t1) % 360  # ensure positive CCW extent

        arc_patch = Arc(
            c, 2 * r, 2 * r, angle=0, theta1=t1, theta2=t1 + extent, color="C0", lw=2
        )
        ax.add_patch(arc_patch)

        # Optionally show endpoints

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.autoscale()
    return ax


def main():
    pursuerRange = 2.0
    pursuerPosition = np.array([0.0, 0.0])
    interceptionPositions = np.array([[-1.3, 1.3], [1.3, -1.3]])

    arcs = intersection_arcs(interceptionPositions, [pursuerRange] * 2)
    print("found arcs")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_in_circle_intersection(interceptionPositions, pursuerRange, fig, ax)
    plot_pursuer_reachable_region(pursuerPosition, pursuerRange, fig, ax)
    plot_circle_intersection_arcs(arcs, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
