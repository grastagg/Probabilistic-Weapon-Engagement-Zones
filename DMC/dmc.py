import numpy as np

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from PEZ import pez
from PEZ import bez
import PEZ.pez_plotting as pez_plotting


def _safe_acos(x):
    # acos argument must be in [-1, 1]; clip for numerical safety
    return jnp.arccos(jnp.clip(x, -1.0, 1.0))


def _angle_diff(a, b):
    # Smallest difference between two angles a, b ∈ ℝ
    return jnp.arctan2(jnp.sin(a - b), jnp.cos(a - b))


# @jax.jit
def dmc_from_xi_d(xi, d, mu, R, r):
    """
    Pure DMC(ξ, d; μ, R, r) as in eq. (7).
    This assumes:
      - d ≥ 0 is the distance from threat to agent
      - xi is the current aspect angle of the agent
    """

    # ----- eq. (5): ξ* -----
    num = d**2 + (mu * R) ** 2 - (R + r) ** 2
    den = 2.0 * mu * R * d

    # eq. (6): if this fails, ξ* is undefined ⇒ ξ_unsafe is empty
    feasible = jnp.abs(num) <= jnp.abs(den)

    # avoid divide-by-zero in degenerate case
    arg = num / den
    print("arcos arg", arg)
    xi_star = _safe_acos(arg)  # in [0, π]
    print("xi star", xi_star)

    # unsafe set ξ ∈ [−ξ*, ξ*]
    inside = jnp.logical_and(
        feasible,
        jnp.logical_and(xi >= -xi_star, xi <= xi_star),
    )

    # eq. (7): sign(ξ) * min_{ξ_c∈{-ξ*,ξ*}} |ξ − ξ_c|
    # distance to left and right boundaries
    dist_left = jnp.abs(_angle_diff(xi, -xi_star))
    dist_right = jnp.abs(_angle_diff(xi, xi_star))
    print("xi", xi)
    print("dist left", dist_left)
    print("dist right", dist_right)

    dmc_val = jnp.sign(xi) * jnp.minimum(dist_left, dist_right)

    # outside ξ_unsafe ⇒ DMC = 0
    return jnp.where(inside, dmc_val, 0.0)
    return jnp.abs(jnp.where(inside, dmc_val, 0.0))


def _wrap_angle(a):
    # Wrap to (-pi, pi]
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


# @jax.jit
def dmc(
    agentPosition,  # shape (2,)
    agentHeading,  # scalar (rad)
    agentSpeed,  # scalar
    pursuerPosition,  # shape (2,)
    pursuerSpeed,  # scalar
    pursuerRange,  # R
    pursuerCaptureRadius,  # r
):
    # speed ratio μ = v_A / v_T (μ < 1 in the paper)
    mu = agentSpeed / pursuerSpeed
    R = pursuerRange
    r = pursuerCaptureRadius

    # distance d between threat and agent
    rel = pursuerPosition - agentPosition  # from threat to agent
    d = jnp.linalg.norm(rel)

    # LOS angle from threat to agent
    los_angle = jnp.arctan2(rel[1], rel[0])

    # aspect angle ξ: vehicle heading relative to LOS
    xi = _wrap_angle(agentHeading - los_angle)

    # plug into analytic formula
    return dmc_from_xi_d(xi, d, mu, R, r)


dmc_vmap = jax.jit(jax.vmap(dmc, in_axes=(0, 0, None, None, None, None, None)))
dmc_batched_vmap = jax.jit(jax.vmap(dmc_vmap, in_axes=(None, None, None, 0, 0, 0, 0)))


def dmc_multiple_pursuer(
    agentPositions,
    agentHeadings,
    agentSpeed,
    pursuerPositions,
    pursuerSpeeds,
    pursuerRanges,
    pursuerCaptureRadiuses,
):
    dmcs = dmc_batched_vmap(
        agentPositions,
        agentHeadings,
        agentSpeed,
        pursuerPositions,
        pursuerSpeeds,
        pursuerRanges,
        pursuerCaptureRadiuses,
    )
    return jnp.max(dmcs, axis=0)


# @jax.jit
# def in_dmc(
#     agentPosition,  # shape (2,)
#     agentHeading,  # scalar (rad)
#     agentSpeed,  # scalar
#     pursuerPosition,  # shape (2,)
#     pursuerSpeed,  # scalar
#     pursuerRange,  # R
#     pursuerCaptureRadius,  # r
#     dmcVal,
# ):
#     speedRatio = agentSpeed / pursuerSpeed
#     c1 = pursuerPosition - speedRatio * (pursuerRange) * jnp.array(
#         [jnp.cos(agentHeading - dmcVal), jnp.sin(agentHeading - dmcVal)]
#     )
#     c2 = pursuerPosition - speedRatio * (pursuerRange) * jnp.array(
#         [jnp.cos(agentHeading + dmcVal), jnp.sin(agentHeading + dmcVal)]
#     )
#     d1 = jnp.linalg.norm(agentPosition - c1)
#     d2 = jnp.linalg.norm(agentPosition - c2)
#     d = jnp.maximum(d1, d2)
#     return d - (pursuerRange + pursuerCaptureRadius)
#
#
# in_dmc_vmap = jax.jit(
#     jax.vmap(in_dmc, in_axes=(0, 0, None, None, None, None, None, None))
# )
#
def in_dmc(
    agentPosition,  # shape (2,)
    agentHeading,  # scalar (rad)
    agentSpeed,  # scalar
    pursuerPosition,  # shape (2,)
    pursuerSpeed,  # scalar
    pursuerRange,  # R
    pursuerCaptureRadius,  # r
    dmcVal,
):
    bezNominal = bez.inEngagementZoneJax(
        agentPosition,
        agentHeading,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        agentSpeed,
    )
    bez1 = bez.inEngagementZoneJax(
        agentPosition,
        agentHeading + dmcVal,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        agentSpeed,
    )
    bez2 = bez.inEngagementZoneJax(
        agentPosition,
        agentHeading - dmcVal,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        agentSpeed,
    )
    d = jnp.linalg.norm(agentPosition - pursuerPosition)
    inNoEscape = (
        d <= (1 - (agentSpeed / pursuerSpeed)) * pursuerRange + pursuerCaptureRadius
    )
    return jnp.where(
        inNoEscape, 100.0, jnp.maximum(bez1, jnp.maximum(bez2, bezNominal))
    )


in_dmc_vmap = jax.jit(
    jax.vmap(in_dmc, in_axes=(0, 0, None, None, None, None, None, None))
)


def plot_dmc(
    agentHeading,
    agentSpeed,
    pursuerPosition,
    pursuerSpeed,
    pursuerRange,
    pursuerCaptureRadius,
    xlim,
    ylim,
    numPoints=100,
    levels=[0],
    ax=None,
    color="purple",
    contour=True,
):
    x = jnp.linspace(xlim[0], xlim[1], numPoints)
    y = jnp.linspace(ylim[0], ylim[1], numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    points = jnp.vstack((X.flatten(), Y.flatten())).T
    headings = jnp.ones(points.shape[0]) * agentHeading

    dmc_values = jnp.abs(
        dmc_vmap(
            points,
            headings,
            agentSpeed,
            pursuerPosition,
            pursuerSpeed,
            pursuerRange,
            pursuerCaptureRadius,
        )
    )

    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect("equal")
    if contour:
        c = ax.contour(
            X.reshape(numPoints, numPoints),
            Y.reshape(numPoints, numPoints),
            dmc_values.reshape(numPoints, numPoints),
            levels=levels,
            colors=color,
        )
    else:
        c = ax.pcolormesh(
            X.reshape(numPoints, numPoints),
            Y.reshape(numPoints, numPoints),
            dmc_values.reshape(numPoints, numPoints),
        )
        cbar = plt.colorbar(c, ax=ax)
    plt.scatter(*pursuerPosition, color="r")
    circle = plt.Circle(
        pursuerPosition, pursuerRange, color="r", fill=False, linestyle=":"
    )
    ax.add_artist(circle)
    circle = plt.Circle(
        pursuerPosition,
        pursuerRange + pursuerCaptureRadius,
        color="r",
        fill=False,
        linestyle="--",
    )
    ax.add_artist(circle)
    return ax


def plot_in_dmc(
    dmcVal,
    agentHeading,
    agentSpeed,
    pursuerPosition,
    pursuerSpeed,
    pursuerRange,
    pursuerCaptureRadius,
    xlim,
    ylim,
    numPoints=100,
    ax=None,
):
    x = jnp.linspace(xlim[0], xlim[1], numPoints)
    y = jnp.linspace(ylim[0], ylim[1], numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    points = jnp.vstack((X.flatten(), Y.flatten())).T
    headings = jnp.ones(points.shape[0]) * agentHeading

    in_dmc_values = in_dmc_vmap(
        points,
        headings,
        agentSpeed,
        pursuerPosition,
        pursuerSpeed,
        pursuerRange,
        pursuerCaptureRadius,
        dmcVal,
    )
    # plot zero contour
    if ax is None:
        fig, ax = plt.subplots()

    ax.contour(
        X.reshape(numPoints, numPoints),
        Y.reshape(numPoints, numPoints),
        in_dmc_values.reshape(numPoints, numPoints),
        levels=[0],
        colors="blue",
    )


def main():
    agentHeading = 0
    agentSpeed = 1.5
    pursuerPosition = jnp.array([0.0, 0.0])
    pursuerSpeed = 2.0
    pursuerRange = 1.5
    pursuerCaptureRadius = 0.1
    xlim = (-4, 4)
    ylim = (-4, 4)
    numPoints = 500
    # dmcVal = np.deg2rad(60)
    point = np.array([[0.355, 0.435]])
    dmcVal = dmc(
        point[0],
        agentHeading,
        agentSpeed,
        pursuerPosition,
        pursuerSpeed,
        pursuerRange,
        pursuerCaptureRadius,
    )
    print("dmcVal", dmcVal)
    # dmcVal = 1.57
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.scatter(point[0, 0], point[0, 1], color="black", zorder=5)
    # plot_dmc(
    #     agentHeading,
    #     agentSpeed,
    #     pursuerPosition,
    #     pursuerSpeed,
    #     pursuerRange,
    #     pursuerCaptureRadius,
    #     xlim,
    #     ylim,
    #     numPoints,
    #     levels=[dmcVal],
    #     ax=ax,
    #     contour=False,
    # )
    #
    print("DMC value (rad): ", dmcVal)
    pez_plotting.plotEngagementZone(
        agentHeading + dmcVal,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        agentSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="lime",
    )
    pez_plotting.plotEngagementZone(
        agentHeading - dmcVal,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        agentSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="magenta",
    )

    pez_plotting.plotEngagementZone(
        agentHeading,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        agentSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="green",
    )

    speedRatio = agentSpeed / pursuerSpeed
    c1 = pursuerPosition - speedRatio * (pursuerRange) * np.array(
        [np.cos(agentHeading - dmcVal), np.sin(agentHeading - dmcVal)]
    )
    ax.scatter(*c1, color="magenta")
    c2 = pursuerPosition - speedRatio * (pursuerRange) * np.array(
        [np.cos(agentHeading + dmcVal), np.sin(agentHeading + dmcVal)]
    )
    ax.scatter(*c2, color="lime")

    plot_in_dmc(
        dmcVal,
        agentHeading,
        agentSpeed,
        pursuerPosition,
        pursuerSpeed,
        pursuerRange,
        pursuerCaptureRadius,
        xlim,
        ylim,
        numPoints,
        ax,
    )
    circ = plt.Circle(
        pursuerPosition,
        (1 - (agentSpeed / pursuerSpeed)) * pursuerRange + pursuerCaptureRadius,
        color="red",
        fill=True,
        alpha=0.2,
    )
    ax.add_artist(circ)
    dmcVal = np.deg2rad(90)
    pez_plotting.plotEngagementZone(
        agentHeading + dmcVal,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        agentSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="lime",
    )
    pez_plotting.plotEngagementZone(
        agentHeading - dmcVal,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRadius,
        pursuerSpeed,
        agentSpeed,
        ax,
        alpha=1.0,
        width=1,
        color="magenta",
    )
    plot_in_dmc(
        dmcVal,
        agentHeading,
        agentSpeed,
        pursuerPosition,
        pursuerSpeed,
        pursuerRange,
        pursuerCaptureRadius,
        xlim,
        ylim,
        numPoints,
        ax,
    )

    plt.show()


if __name__ == "__main__":
    main()
