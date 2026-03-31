from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


import PEZ.pez_plotting as pez_plotting
import GEOMETRIC_BEZ.rectangle_bez as rectangle_bez
import GEOMETRIC_BEZ.rectangle_bez_path_planner as rectangle_bez_path_planner


def first_forward_intersection(p0, p1, center, radius, tol=1e-9):
    """
    Returns the first forward intersection of a ray starting at p0 and going toward p1.

    Parameters
    ----------
    p0 : (2,) start point
    p1 : (2,) defines direction
    center : (2,) circle center
    radius : float

    Returns
    -------
    t : scalar
        Smallest t >= 0 where intersection occurs (NaN if none)
    pt : (2,)
        Intersection point (NaN if none)
    hit : bool
        Whether a forward intersection exists
    """

    d = p1 - p0
    f = p0 - center

    a = jnp.dot(d, d)
    b = 2.0 * jnp.dot(f, d)
    c = jnp.dot(f, f) - radius * radius

    disc = b * b - 4.0 * a * c

    # No real intersection
    has_real = disc >= -tol

    disc = jnp.maximum(disc, 0.0)
    sqrt_disc = jnp.sqrt(disc)

    two_a = 2.0 * a

    t1 = (-b - sqrt_disc) / two_a
    t2 = (-b + sqrt_disc) / two_a

    # Keep only forward intersections
    t1 = jnp.where(t1 >= 0.0, t1, jnp.inf)
    t2 = jnp.where(t2 >= 0.0, t2, jnp.inf)

    t = jnp.minimum(t1, t2)

    # Check if valid
    hit = has_real & jnp.isfinite(t)

    t = jnp.where(hit, t, jnp.nan)
    pt = p0 + t * d
    pt = jnp.where(hit, pt, jnp.array([jnp.nan, jnp.nan]))
    p0_inside = jnp.dot(f, f) < (radius**2 - tol)
    pt = jnp.where(p0_inside, jnp.array([jnp.nan, jnp.nan]), pt)

    return pt


def circle_rectangle_intersections(cx, cy, r, xmin, xmax, ymin, ymax, tol=1e-9):
    """
    JAX/JIT-friendly intersection of a circle with an axis-aligned box boundary.

    Circle:
        (x - cx)^2 + (y - cy)^2 = r^2

    Box boundary:
        x = xmin, x = xmax, y = ymin, y = ymax
        with x in [xmin, xmax], y in [ymin, ymax]

    Returns
    -------
    pts : array, shape (2, 2)
        The first two unique valid intersection points.
        If fewer than 2 unique intersections exist, unused rows are NaN.
    """

    r2 = r * r
    nan = jnp.array(jnp.nan)

    # Intersections with vertical side x = x0
    def vertical_side(x0):
        inside = r2 - (x0 - cx) ** 2
        valid_base = inside >= -tol
        inside = jnp.maximum(inside, 0.0)
        dy = jnp.sqrt(inside)

        y_a = cy - dy
        y_b = cy + dy

        p_a = jnp.array([x0, y_a])
        p_b = jnp.array([x0, y_b])

        valid_a = valid_base & (y_a >= ymin - tol) & (y_a <= ymax + tol)
        valid_b = valid_base & (y_b >= ymin - tol) & (y_b <= ymax + tol)

        return p_a, valid_a, p_b, valid_b

    # Intersections with horizontal side y = y0
    def horizontal_side(y0):
        inside = r2 - (y0 - cy) ** 2
        valid_base = inside >= -tol
        inside = jnp.maximum(inside, 0.0)
        dx = jnp.sqrt(inside)

        x_a = cx - dx
        x_b = cx + dx

        p_a = jnp.array([x_a, y0])
        p_b = jnp.array([x_b, y0])

        valid_a = valid_base & (x_a >= xmin - tol) & (x_a <= xmax + tol)
        valid_b = valid_base & (x_b >= xmin - tol) & (x_b <= xmax + tol)

        return p_a, valid_a, p_b, valid_b

    p1, v1, p2, v2 = vertical_side(xmin)
    p3, v3, p4, v4 = vertical_side(xmax)
    p5, v5, p6, v6 = horizontal_side(ymin)
    p7, v7, p8, v8 = horizontal_side(ymax)

    candidates = jnp.stack([p1, p2, p3, p4, p5, p6, p7, p8], axis=0)
    valids = jnp.array([v1, v2, v3, v4, v5, v6, v7, v8])

    # Replace invalid candidates with NaN
    candidates = jnp.where(valids[:, None], candidates, jnp.nan)

    # Deduplicate:
    # A point is "first occurrence" if it is valid and not equal to any earlier valid point.
    def point_equal(p, q):
        return jnp.all(jnp.abs(p - q) <= 1e-7)

    eq_matrix = jax.vmap(lambda p: jax.vmap(lambda q: point_equal(p, q))(candidates))(
        candidates
    )

    valid_matrix = valids[:, None] & valids[None, :]
    same_valid = eq_matrix & valid_matrix

    # lower triangular mask excluding diagonal: earlier duplicates
    earlier_mask = jnp.tril(jnp.ones((8, 8), dtype=bool), k=-1)
    has_earlier_duplicate = jnp.any(same_valid & earlier_mask, axis=1)

    is_unique = valids & (~has_earlier_duplicate)

    # Give unique points a sortable key, others inf
    idx = jnp.arange(8)
    sort_key = jnp.where(is_unique, idx, 999)

    order = jnp.argsort(sort_key)
    unique_sorted = candidates[order]

    return unique_sorted[:2]


def compute_prob(a, p, xmin, xmax, ymin, ymax, tol=1e-9):
    """
    Compute p_x and p_y given point a = (x_a, y_a)

    Parameters
    ----------
    a : (2,) array -> (x_a, y_a)
    p : (2,) array -> (x_p, y_p)
    xmin, xmax, ymin, ymax : box bounds

    Returns
    -------
    px, py : scalars
    """

    x_a, y_a = a
    x_p, y_p = p

    # ----- p_x -----
    px_left = (xmax - x_p) / (xmax - x_a)
    px_right = (x_p - xmin) / (x_a - xmin)

    px = jnp.where(
        jnp.abs(x_a - x_p) <= tol, 1.0, jnp.where(x_a < x_p, px_left, px_right)
    )

    # ----- p_y -----
    py_down = (ymax - y_p) / (ymax - y_a)
    py_up = (y_p - ymin) / (y_a - ymin)

    py = jnp.where(jnp.abs(y_a - y_p) <= tol, 1.0, jnp.where(y_a < y_p, py_down, py_up))

    invalid = jnp.any(jnp.isnan(a))
    return jnp.where(invalid, jnp.nan, jnp.nanmin(jnp.array([px, py])))


def point_in_box(a, xmin, xmax, ymin, ymax):
    x_a, y_a = a
    return (x_a >= xmin) & (x_a <= xmax) & (y_a >= ymin) & (y_a <= ymax)


def max_ouq_prob_pursuer_position_uncertainty_single(
    x_e,
    y_e,
    psi_e,
    pursuerRange,
    captureRadius,
    speedRatio,
    x_pmean,
    y_pmean,
    x_pmin,
    x_pmax,
    y_pmin,
    y_pmax,
):
    # circle where if pursuer is inside the evader is in the EZ
    c_x = x_e + speedRatio * pursuerRange * jnp.cos(psi_e)
    c_y = y_e + speedRatio * pursuerRange * jnp.sin(psi_e)
    center = jnp.array([c_x, c_y])
    pursuerMean = jnp.array([x_pmean, y_pmean])

    # circle extreme points
    rightCirc = center + jnp.array([pursuerRange + captureRadius, 0.0])
    leftCirc = center + jnp.array([-pursuerRange - captureRadius, 0.0])
    topCirc = center + jnp.array([0.0, pursuerRange + captureRadius])
    bottomCirc = center + jnp.array([0.0, -pursuerRange - captureRadius])
    # make sure extreme points are within the box, if not set to NaN
    rightCirc = jnp.where(
        point_in_box(rightCirc, x_pmin, x_pmax, y_pmin, y_pmax),
        rightCirc,
        jnp.array([jnp.nan, jnp.nan]),
    )
    leftCirc = jnp.where(
        point_in_box(leftCirc, x_pmin, x_pmax, y_pmin, y_pmax),
        leftCirc,
        jnp.array([jnp.nan, jnp.nan]),
    )
    topCirc = jnp.where(
        point_in_box(topCirc, x_pmin, x_pmax, y_pmin, y_pmax),
        topCirc,
        jnp.array([jnp.nan, jnp.nan]),
    )
    bottomCirc = jnp.where(
        point_in_box(bottomCirc, x_pmin, x_pmax, y_pmin, y_pmax),
        bottomCirc,
        jnp.array([jnp.nan, jnp.nan]),
    )

    # box extreme points
    upperRight = jnp.array([x_pmax, y_pmax])
    lowerRight = jnp.array([x_pmax, y_pmin])
    lowerLeft = jnp.array([x_pmin, y_pmin])
    upperLeft = jnp.array([x_pmin, y_pmax])

    # project box extreme points onto circle
    upperRightProj = first_forward_intersection(
        upperRight, pursuerMean, center, pursuerRange + captureRadius
    )
    lowerRightProj = first_forward_intersection(
        lowerRight, pursuerMean, center, pursuerRange + captureRadius
    )
    lowerLeftProj = first_forward_intersection(
        lowerLeft, pursuerMean, center, pursuerRange + captureRadius
    )
    upperLeftProj = first_forward_intersection(
        upperLeft, pursuerMean, center, pursuerRange + captureRadius
    )
    # make sure projected points are within the box, if not set to NaN
    upperRightProj = jnp.where(
        point_in_box(upperRightProj, x_pmin, x_pmax, y_pmin, y_pmax),
        upperRightProj,
        jnp.array([jnp.nan, jnp.nan]),
    )
    upperLeftProj = jnp.where(
        point_in_box(upperLeftProj, x_pmin, x_pmax, y_pmin, y_pmax),
        upperLeftProj,
        jnp.array([jnp.nan, jnp.nan]),
    )
    lowerRightProj = jnp.where(
        point_in_box(lowerRightProj, x_pmin, x_pmax, y_pmin, y_pmax),
        lowerRightProj,
        jnp.array([jnp.nan, jnp.nan]),
    )
    lowerLeftProj = jnp.where(
        point_in_box(lowerLeftProj, x_pmin, x_pmax, y_pmin, y_pmax),
        lowerLeftProj,
        jnp.array([jnp.nan, jnp.nan]),
    )

    rightCircProb = compute_prob(rightCirc, pursuerMean, x_pmin, x_pmax, y_pmin, y_pmax)
    leftCircProb = compute_prob(leftCirc, pursuerMean, x_pmin, x_pmax, y_pmin, y_pmax)
    topCircProb = compute_prob(topCirc, pursuerMean, x_pmin, x_pmax, y_pmin, y_pmax)
    bottomCircProb = compute_prob(
        bottomCirc, pursuerMean, x_pmin, x_pmax, y_pmin, y_pmax
    )

    lowerRightProjProb = compute_prob(
        lowerRightProj, pursuerMean, x_pmin, x_pmax, y_pmin, y_pmax
    )
    lowerLeftProjProb = compute_prob(
        lowerLeftProj, pursuerMean, x_pmin, x_pmax, y_pmin, y_pmax
    )
    upperLeftProjProb = compute_prob(
        upperLeftProj, pursuerMean, x_pmin, x_pmax, y_pmin, y_pmax
    )
    upperRightProjProb = compute_prob(
        upperRightProj, pursuerMean, x_pmin, x_pmax, y_pmin, y_pmax
    )

    # also cehck intersection points
    rectIntersections = circle_rectangle_intersections(
        center[0],
        center[1],
        pursuerRange + captureRadius,
        x_pmin,
        x_pmax,
        y_pmin,
        y_pmax,
    )

    rectIntersectionProbs = jax.vmap(
        lambda a: compute_prob(a, pursuerMean, x_pmin, x_pmax, y_pmin, y_pmax)
    )(rectIntersections)

    meanInsideCircle = (
        jnp.linalg.norm(pursuerMean - center) <= pursuerRange + captureRadius
    )

    cands = jnp.concatenate(
        [
            jnp.array(
                [
                    rightCircProb,
                    leftCircProb,
                    topCircProb,
                    bottomCircProb,
                    lowerRightProjProb,
                    lowerLeftProjProb,
                    upperLeftProjProb,
                    upperRightProjProb,
                ]
            ),
            rectIntersectionProbs,
        ]
    )
    # print(cands)

    any_valid = jnp.any(jnp.isfinite(cands))
    best = jnp.where(any_valid, jnp.nanmax(cands), 0.0)

    return jnp.where(meanInsideCircle, 1.0, best)


max_ouq_prob_pursuer_position_uncertainty = jax.jit(
    jax.vmap(
        max_ouq_prob_pursuer_position_uncertainty_single,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None),
    )
)


def ouq_inner_rectangle_for_alpha(
    alpha,
    x_pmean,
    y_pmean,
    x_pmin,
    x_pmax,
    y_pmin,
    y_pmax,
):
    """
    Rectangle Z_alpha = K ∩ A_alpha such that:
        p(z) >= alpha  iff  z in Z_alpha
    for the 2-atom OUQ probability p(z)=b/(a+b).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    s = (1.0 - alpha) / alpha

    axmin = x_pmean - s * (x_pmax - x_pmean)
    axmax = x_pmean + s * (x_pmean - x_pmin)

    aymin = y_pmean - s * (y_pmax - y_pmean)
    aymax = y_pmean + s * (y_pmean - y_pmin)

    # must also lie inside the support box K
    zxmin = jnp.maximum(x_pmin, axmin)
    zxmax = jnp.minimum(x_pmax, axmax)
    zymin = jnp.maximum(y_pmin, aymin)
    zymax = jnp.minimum(y_pmax, aymax)

    return zxmin, zxmax, zymin, zymax


def max_rectangle_in_bounds(x0, y0, q, Xmin, Xmax, Ymin, Ymax):
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")

    W = min((x0 - Xmin) / (1.0 - q), (Xmax - x0) / q)
    H = min((y0 - Ymin) / (1.0 - q), (Ymax - y0) / q)

    xmin = x0 - (1.0 - q) * W
    xmax = x0 + q * W
    ymin = y0 - (1.0 - q) * H
    ymax = y0 + q * H

    return xmin, xmax, ymin, ymax


def main():
    x = np.linspace(-4, 4, 500)
    y = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    psi = np.deg2rad(45.0) * np.ones(points.shape[0])
    pursuerSpeed = 1.0
    evaderSpeed = 0.5

    speedRatio = evaderSpeed / pursuerSpeed
    captureRadius = 0.1
    pursuerRange = 1.0
    minX = -2.0
    maxX = 2.0
    minY = -1.0
    maxY = 1.0
    meanX = -1.5
    meanY = 0.5
    test = max_ouq_prob_pursuer_position_uncertainty_single(
        2.0,
        1.4,
        0.0,
        pursuerRange,
        captureRadius,
        speedRatio,
        meanX,
        meanY,
        minX,
        maxX,
        minY,
        maxY,
    )
    print(test)
    prob = max_ouq_prob_pursuer_position_uncertainty(
        points[:, 0],
        points[:, 1],
        psi,
        pursuerRange,
        captureRadius,
        speedRatio,
        meanX,
        meanY,
        minX,
        maxX,
        minY,
        maxY,
    )

    fig, ax = plt.subplots()
    ax.grid(True)
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

    # c = ax.pcolormesh(X, Y, prob.reshape(X.shape), cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    # plt.colorbar(c, label="Max Probability of Capture")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    # plot minx, maxx, miny, maxy box
    plt.plot([minX, maxX, maxX, minX, minX], [minY, minY, maxY, maxY, minY], "r--")
    plt.scatter(meanX, meanY, color="red", label="Mean Position")

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    pez_limit = 0.3
    minXlim, maxXlim, minYlim, maxYlim = ouq_inner_rectangle_for_alpha(
        pez_limit, meanX, meanY, minX, maxX, minY, maxY
    )
    # plot rectangle to be expanded for alpha
    plt.plot(
        [minXlim, maxXlim, maxXlim, minXlim, minXlim],
        [minYlim, minYlim, maxYlim, maxYlim, minYlim],
        "g--",
        label=f"OUQ Inner Rectangle for alpha={pez_limit}",
    )
    rectangle_bez.plot_box_pursuer_engagement_zone(
        np.array([minXlim, minYlim]),
        np.array([maxXlim, maxYlim]),
        pursuerRange,
        captureRadius,
        pursuerSpeed,
        psi[0],
        evaderSpeed,
        (-3, 3),
        (-3, 3),
        ax,
        color="green",
    )

    plt.show()


if __name__ == "__main__":
    main()
