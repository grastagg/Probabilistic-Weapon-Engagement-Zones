import time
import jax.numpy as jnp
from matplotlib import colors
import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib
import scipy

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


import GEOMETRIC_BEZ.rectangle_bez as rectangle_bez
import DMC.dmc as dmc


def _angle_diff(a, b):
    # Smallest difference between two angles a, b ∈ ℝ
    return jnp.arctan2(jnp.sin(a - b), jnp.cos(a - b))


def _wrap_angle(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def angle_between_ccw(theta, start, stop):
    two_pi = 2 * np.pi
    return ((theta - start) % two_pi) <= ((stop - start) % two_pi)


def rect_dmc(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    box_min,
    box_max,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    dmcVal,
):
    rectBezNominal = rectangle_bez.box_pursuer_engagment_zone(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        box_min,
        box_max,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )

    rectBez1 = rectangle_bez.box_pursuer_engagment_zone(
        evaderPositions,
        evaderHeadings - dmcVal,
        evaderSpeed,
        box_min,
        box_max,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )
    rectBez2 = rectangle_bez.box_pursuer_engagment_zone(
        evaderPositions,
        evaderHeadings + dmcVal,
        evaderSpeed,
        box_min,
        box_max,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )
    # return jnp.maximum(rectBez1, rectBez2)
    return jnp.maximum(rectBez1, jnp.maximum(rectBez2, rectBezNominal))


def rect_dmc_val_solve(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    box_min,
    box_max,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
):
    def func(xiStar):
        return rectangle_bez.box_pursuer_engagment_zone(
            evaderPosition,
            xiStar,
            evaderSpeed,
            box_min,
            box_max,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
        )

    xiStar = _wrap_angle(scipy.optimize.newton(func, 0)[0])
    xiStar2 = np.abs(_wrap_angle(scipy.optimize.newton(func, np.pi)[0]))
    # test if evaderHeading lies inbetween -xiStar and xistar func_counterclockwise

    bezClockwise = rectangle_bez.box_pursuer_engagment_zone(
        evaderPosition,
        xiStar,
        evaderSpeed,
        box_min,
        box_max,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )
    bezCounterClockwise = rectangle_bez.box_pursuer_engagment_zone(
        evaderPosition,
        -xiStar,
        evaderSpeed,
        box_min,
        box_max,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )
    # if not angle_between_ccw(evaderHeading, xiStar - np.pi, xiStar):
    #     return 0.0, 0.0
    if np.abs(bezClockwise) < np.abs(bezCounterClockwise):
        return _angle_diff(xiStar, evaderHeading), _angle_diff(xiStar2, evaderHeading)
    else:
        return -_angle_diff(xiStar, evaderHeading), -_angle_diff(xiStar2, evaderHeading)


def _shortest_root_in_interval(func, lo, hi, evaderHeading, iters=20, num_samples=64):
    """
    Find ALL zero crossings in [lo, hi] and return the shortest-magnitude root.
    JAX-safe, jit/vmap compatible.
    """
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)

    # ---- sample ----
    xis = jnp.linspace(lo, hi, num_samples)

    def f(x):
        return jnp.squeeze(func(x))

    fvals = jax.vmap(f)(xis)

    # ---- detect sign changes ----
    f0 = fvals[:-1]
    f1 = fvals[1:]
    has_root = f0 * f1 <= 0.0

    lo_b = xis[:-1]
    hi_b = xis[1:]

    # ---- bisection on one bracket ----
    def bisect(lo, hi):
        def body(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            same = jnp.sign(f(lo)) == jnp.sign(f(mid))
            lo = jnp.where(same, mid, lo)
            hi = jnp.where(same, hi, mid)
            return lo, hi

        lo, hi = jax.lax.fori_loop(0, iters, body, (lo, hi))
        return 0.5 * (lo + hi)

    # ---- refine all brackets (masked) ----
    roots = jax.vmap(lambda lo, hi, valid: jnp.where(valid, bisect(lo, hi), jnp.inf))(
        lo_b, hi_b, has_root
    )

    # ---- shortest root ----
    # return jnp.nanmin(_angle_diff(roots, evaderHeading))
    return roots[jnp.nanargmin(jnp.abs(_angle_diff(roots, evaderHeading)))]


def rect_dmc_val_solve_jax(
    evaderPosition,  # (2,)
    evaderHeading,  # scalar
    evaderSpeed,
    box_min,
    box_max,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
):
    def bez_at_xi(xi):
        return rectangle_bez.box_pursuer_engagment_zone(
            evaderPosition,
            xi,
            evaderSpeed,
            box_min,
            box_max,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
        )

    # ---- FIND SHORTEST ROOT IN EACH HALF-INTERVAL ----
    xiStar1 = _shortest_root_in_interval(bez_at_xi, -jnp.pi, jnp.pi, evaderHeading)

    bezNominal = rectangle_bez.box_pursuer_engagment_zone(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        box_min,
        box_max,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )
    rrNoEscape = rectangle_bez.box_reachable_region(
        evaderPosition[None, :],
        box_min,
        box_max,
        (1 - (evaderSpeed / pursuerSpeed)) * pursuerRange + pursuerCaptureRadius,
        0.0,
    )
    inNoEscapeRegion = rrNoEscape <= 0.0

    inBez = bezNominal <= 0.0

    angle_diff_xiStar = _angle_diff(xiStar1, evaderHeading)

    return jnp.where(inNoEscapeRegion, 0.0, jnp.where(inBez, angle_diff_xiStar, 0.0))


rect_dmc_val_solve_vmap = jax.jit(
    jax.vmap(
        rect_dmc_val_solve_jax,
        in_axes=(0, 0, None, None, None, None, None, None),
    )
)


def ray_rect_intersection_or_closest(p0, theta, minbox, maxbox, eps=1e-12):
    """
    Returns:
        point: (2,) closest intersection point if exists,
               otherwise closest point on rectangle to p0
        hit:   bool (True if ray intersects rectangle)
    """
    d = jnp.array([jnp.cos(theta), jnp.sin(theta)])

    # Avoid division by zero
    inv_d = jnp.where(jnp.abs(d) > eps, 1.0 / d, jnp.inf)

    t1 = (minbox - p0) * inv_d
    t2 = (maxbox - p0) * inv_d

    tmin = jnp.minimum(t1, t2)
    tmax = jnp.maximum(t1, t2)

    t_enter = jnp.max(tmin)
    t_exit = jnp.min(tmax)

    hit = (t_exit >= 0.0) & (t_enter <= t_exit)

    # Intersection point (if hit)
    t_star = jnp.maximum(t_enter, 0.0)
    p_hit = p0 + t_star * d

    # Closest point fallback
    p_closest = jnp.minimum(jnp.maximum(p0, minbox), maxbox)

    point = jnp.where(hit, p_hit, p_closest)
    return point, hit


def line_rect_intersection_or_closest(p0, theta, minbox, maxbox, eps=1e-12):
    """
    p0      : (2,) point on line
    theta   : scalar heading (rad)
    minbox  : (2,)
    maxbox  : (2,)

    returns:
        point : (2,)
        hit   : bool
    """

    # --- Line definition ---
    d = jnp.array([jnp.cos(theta), jnp.sin(theta)])

    # --- Slab intersection (infinite line) ---
    inv_d = jnp.where(jnp.abs(d) > eps, 1.0 / d, jnp.inf)

    t1 = (minbox - p0) * inv_d
    t2 = (maxbox - p0) * inv_d

    tmin = jnp.minimum(t1, t2)
    tmax = jnp.maximum(t1, t2)

    t_enter = jnp.max(tmin)
    t_exit = jnp.min(tmax)

    hit = t_enter <= t_exit

    # Closest intersection point
    t_star = jnp.where(
        jnp.abs(t_enter) <= jnp.abs(t_exit),
        t_enter,
        t_exit,
    )
    p_intersect = p0 + t_star * d

    # --- Rectangle edges (vectorized) ---
    corners = jnp.array(
        [
            [minbox[0], minbox[1]],
            [maxbox[0], minbox[1]],
            [maxbox[0], maxbox[1]],
            [minbox[0], maxbox[1]],
        ]
    )  # (4,2)

    a = corners
    b = jnp.roll(corners, shift=-1, axis=0)
    e = b - a  # edge vectors (4,2)

    # Solve p0 + t d = a + s e  (least-squares, vectorized)
    d_rep = jnp.broadcast_to(d[None, :], e.shape)  # (4,2)
    A = jnp.stack([d_rep, -e], axis=2)  # (4,2,2)

    # A = jnp.stack([d, -e], axis=2)  # (4,2,2)
    rhs = a - p0  # (4,2)

    ts = jnp.linalg.solve(A, rhs)  # (4,2)
    s = jnp.clip(ts[:, 1], 0.0, 1.0)

    candidates = a + s[:, None] * e  # (4,2)

    # Distance from point to infinite line
    v = candidates - p0
    proj = jnp.sum(v * d, axis=1, keepdims=True) * d
    perp = v - proj
    dists = jnp.linalg.norm(perp, axis=1)

    p_closest = candidates[jnp.argmin(dists)]

    # --- Final selection ---
    point = jnp.where(hit, p_intersect, p_closest)
    return point, hit


def rect_dmc_new(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    box_min,
    box_max,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    dmcVal,
):
    worstPursuer, hit = line_rect_intersection_or_closest(
        evaderPosition.flatten(), evaderHeading.flatten(), box_min, box_max
    )

    dmcForWorstPursuer = dmc.dmc(
        evaderPosition,  # shape (2,)
        evaderHeading,  # scalar (rad)
        evaderSpeed,  # scalar
        worstPursuer,  # shape (2,)
        pursuerSpeed,  # scalar
        pursuerRange,  # R
        pursuerCaptureRadius,  # r
    )
    return dmcForWorstPursuer


rect_dmc_new_vmap = jax.vmap(
    rect_dmc_new, in_axes=(0, 0, None, None, None, None, None, None, None)
)


def plot_rect_dmc(
    min_box,
    max_box,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    evaderHeading,
    evaderSpeed,
    dmcVal,
    xlim,
    ylim,
    ax,
    color="purple",
    numPoints=500,
):
    points, X, Y = rectangle_bez.get_meshgrid_points(xlim, ylim, numPoints)
    evaderHeadings = evaderHeading * jnp.ones((points.shape[0],))

    dmc = rect_dmc(
        points,
        evaderHeadings,
        evaderSpeed,
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        dmcVal,
    )
    ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        dmc.reshape((numPoints, numPoints)),
        levels=[0],
        colors=color,
    )
    ax.plot([], label=r"$\partial \mathcal{Z}_{\text{rect}}$", color="green")
    return ax


def plot_dmc_rect_solve(
    min_box,
    max_box,
    pursuerRange,
    pursuerCaptureRadius,
    pursuerSpeed,
    evaderHeading,
    evaderSpeed,
    dmcVal,
    xlim=(-4, 4),
    ylim=(-4, 4),
    ax=None,
    color="purple",
    numPoints=500,
):
    points, X, Y = rectangle_bez.get_meshgrid_points(xlim, ylim, numPoints)
    evaderHeadings = evaderHeading * jnp.ones((points.shape[0],))
    dmcVals = np.abs(
        rect_dmc_val_solve_vmap(
            points,
            evaderHeadings,
            evaderSpeed,
            min_box,
            max_box,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
        )
    )
    start = time.time()
    dmcVals = np.abs(
        rect_dmc_val_solve_vmap(
            points,
            evaderHeadings,
            evaderSpeed,
            min_box,
            max_box,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
        )
    )
    print("JAX DMC computation time:", time.time() - start)
    # c = ax.pcolormesh(
    #     X.reshape((numPoints, numPoints)),
    #     Y.reshape((numPoints, numPoints)),
    #     dmcVals.reshape((numPoints, numPoints)),
    #     # levels=[dmcVal],
    #     # colors=color,
    # )
    # plt.colorbar(c, ax=ax, label="DMC (rad)")
    # c = ax.pcolormesh(
    c = ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        dmcVals.reshape((numPoints, numPoints)),
        levels=[dmcVal],
        colors=color,
    )
    # plt.colorbar(c, ax=ax, label="DMC (rad)")


def main_solve():
    pursuerRange = 1.5
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.2
    evaderHeading = np.deg2rad(45.0)
    evaderSpeed = 1.0
    min_box = np.array([-1.0, -1.0])
    max_box = np.array([2.0, 1.0])
    dmcVal = np.deg2rad(100)

    fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    ax.set_aspect("equal")
    rectangle_bez.plot_box_pursuer_reachable_region(
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        ax,
        color="red",
        linestyle="--",
    )
    rectangle_bez.plot_box_pursuer_reachable_region(
        min_box,
        max_box,
        pursuerRange,
        0.0,
        ax,
        color="red",
        linestyle=":",
    )
    rectangle_bez.plot_box_pursuer_reachable_region(
        min_box,
        max_box,
        (1 - (evaderSpeed / pursuerSpeed)) * pursuerRange + pursuerCaptureRadius,
        0.0,
        ax,
        color="red",
        fill=True,
        alpha=0.2,
    )
    rectangle_bez.plot_box_pursuer_engagement_zone(
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
    rectangle_bez.plot_box_pursuer_engagement_zone(
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        evaderHeading + dmcVal,
        evaderSpeed,
        xlim=(-4, 4),
        ylim=(-4, 4),
        ax=ax,
        color="lime",
    )
    rectangle_bez.plot_box_pursuer_engagement_zone(
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        evaderHeading - dmcVal,
        evaderSpeed,
        xlim=(-4, 4),
        ylim=(-4, 4),
        ax=ax,
        color="magenta",
    )
    plot_dmc_rect_solve(
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        evaderHeading,
        evaderSpeed,
        dmcVal,
        xlim=(-4, 4),
        ylim=(-4, 4),
        ax=ax,
        color="orange",
    )


def main():
    pursuerRange = 1.5
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.2
    evaderHeading = np.deg2rad(45.0)
    evaderSpeed = 1.0
    min_box = np.array([-1.0, -1.0])
    max_box = np.array([2.0, 1.0])
    point = np.array([-2.0, 2.0])
    point = np.array([-3, 0.6])
    # dmcValTest, dmcValTest2 = rect_dmc_val_solve(
    #     point,
    #     evaderHeading,
    #     evaderSpeed,
    #     min_box,
    #     max_box,
    #     pursuerRange,
    #     pursuerCaptureRadius,
    #     pursuerSpeed,
    # )
    # print(
    #     "DMC Value at point ",
    #     point,
    #     " is ",
    #     dmcValTest,
    #     "in degrees:",
    #     np.rad2deg(dmcValTest),
    # )
    # print("DMC Value 2:", dmcValTest2, "in degrees:", np.rad2deg(dmcValTest2))
    # test jax solve
    jaxDmc = rect_dmc_val_solve_jax(
        point,
        evaderHeading,
        evaderSpeed,
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
    )
    print(
        "JAX DMC Value at point ",
        point,
        " is ",
        jaxDmc,
        "in degrees:",
        np.rad2deg(jaxDmc),
    )

    dmcVals = np.array([dmcValTest, dmcValTest2])

    fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    numPoints = 500
    ax.scatter(point[0], point[1], color="black", marker="x", label="Test Point")
    # Colormap and normalization
    min = dmcVals.min() - 0.1
    max = dmcVals.max() + 0.1
    colors = plt.cm.viridis(0.2 + 0.8 * (dmcVals - min) / (max - min))
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=min - 0.1, vmax=max + 0.1)

    # ScalarMappable required for colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for older matplotlib versions
    cbar = plt.colorbar(sm, ax=ax, label="DMC (rad)")

    # Optional: force ticks at your discrete DMC values
    cbar.set_ticks(dmcVals)
    cbar.set_ticklabels([f"{v:.2f}" for v in dmcVals])
    # rectangle_bez.plot_box_pursuer_reachable_region(
    #     min_box,
    #     max_box,
    #     pursuerRange,
    #     pursuerCaptureRadius,
    #     ax,
    #     color="red",
    #     linestyle="--",
    # )
    # rectangle_bez.plot_box_pursuer_reachable_region(
    #     min_box,
    #     max_box,
    #     pursuerRange,
    #     0.0,
    #     ax,
    #     color="red",
    #     linestyle=":",
    # )
    # rectangle_bez.plot_box_pursuer_reachable_region(
    #     min_box,
    #     max_box,
    #     (1 - (evaderSpeed / pursuerSpeed)) * pursuerRange + pursuerCaptureRadius,
    #     0.0,
    #     ax,
    #     color="red",
    #     fill=True,
    #     alpha=0.2,
    # )
    rectangle_bez.plot_box_pursuer_engagement_zone(
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
    for i, dmcVal in enumerate(dmcVals):
        dmcVal = np.abs(dmcVal)
        rectangle_bez.plot_box_pursuer_engagement_zone(
            min_box,
            max_box,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
            evaderHeading + dmcVal,
            evaderSpeed,
            xlim=(-4, 4),
            ylim=(-4, 4),
            ax=ax,
            color="lime",
        )
        rectangle_bez.plot_box_pursuer_engagement_zone(
            min_box,
            max_box,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
            evaderHeading - dmcVal,
            evaderSpeed,
            xlim=(-4, 4),
            ylim=(-4, 4),
            ax=ax,
            color="magenta",
        )
        plot_rect_dmc(
            min_box,
            max_box,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
            evaderHeading,
            evaderSpeed,
            dmcVal,
            (-4, 4),
            (-4, 4),
            ax,
            color=colors[i],
            numPoints=numPoints,
        )
        # plot dmc for four corners of the box
        pursuerPositions = np.array(
            [
                [min_box[0], min_box[1]],
                [min_box[0], max_box[1]],
                [max_box[0], min_box[1]],
                [max_box[0], max_box[1]],
            ]
        )
        for pursuerPosition in pursuerPositions:
            dmc.plot_dmc(
                evaderHeading,
                evaderSpeed,
                pursuerPosition,
                pursuerSpeed,
                pursuerRange,
                pursuerCaptureRadius,
                xlim=(-4, 4),
                ylim=(-4, 4),
                numPoints=numPoints,
                levels=[dmcVal],
                ax=ax,
                color=colors[i],
            )

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def test():
    pursuerRange = 1.5
    pursuerSpeed = 2.0
    pursuerCaptureRadius = 0.2
    evaderHeading = np.deg2rad(45.0)
    evaderSpeed = 1.0
    min_box = np.array([-1.0, -1.0])
    max_box = np.array([2.0, 1.0])
    dmcVal = np.deg2rad(10)

    numPoints = 500

    points, X, Y = rectangle_bez.get_meshgrid_points((-4, 4), (-4, 4), numPoints)
    fig, ax = plt.subplots()

    dmc = np.abs(
        rect_dmc_new_vmap(
            points,
            evaderHeading * jnp.ones((points.shape[0],)),
            evaderSpeed,
            min_box,
            max_box,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
            dmcVal,
        )
    )
    ax.contour(
        X.reshape((numPoints, numPoints)),
        Y.reshape((numPoints, numPoints)),
        dmc.reshape((numPoints, numPoints)),
        levels=[dmcVal],
        colors="blue",
    )


if __name__ == "__main__":
    main_solve()
    # main()
    # test()
    plt.show()
