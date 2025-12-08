import os

from jax.lax import random_gamma_grad
from matplotlib.markers import MarkerStyle
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import patches

import jax
import jax.numpy as jnp


# import dubinsReachable
# import testDubins
# get rid of type 3 fonts
import matplotlib

import CSPEZ.csbez as csbez
import PEZ.bez as bez
import PLOT_COMMON.draw_airplanes as draw_airplanes
import PEZ.pez_plotting as pez_plotting


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# set font size for title, axis labels, and legend, and tick labels
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10


def _arc_between_points(center, radius, p_start, p_end, ccw=True, num=200):
    """
    Compute points on an arc from p_start to p_end around `center`
    with radius `radius`, going CCW or CW.
    """
    ang1 = np.arctan2(p_start[1] - center[1], p_start[0] - center[0])
    ang2 = np.arctan2(p_end[1] - center[1], p_end[0] - center[0])

    if ccw:
        # ensure ang2 is ahead of ang1 in CCW direction
        if ang2 <= ang1:
            ang2 += 2 * np.pi
    else:
        # ensure ang2 is behind ang1 in CW direction
        if ang2 >= ang1:
            ang2 -= 2 * np.pi

    theta = np.linspace(ang1, ang2, num)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y


def plot_turn_radius_circles(startPosition, startHeading, turnRadius, ax):
    theta = np.linspace(0, 2 * np.pi, 100)
    leftCenter = np.array(
        [
            startPosition[0] - turnRadius * np.sin(startHeading),
            startPosition[1] + turnRadius * np.cos(startHeading),
        ]
    )
    rightCenter = np.array(
        [
            startPosition[0] + turnRadius * np.sin(startHeading),
            startPosition[1] - turnRadius * np.cos(startHeading),
        ]
    )
    leftX = leftCenter[0] + turnRadius * np.cos(theta)
    leftY = leftCenter[1] + turnRadius * np.sin(theta)
    rightX = rightCenter[0] + turnRadius * np.cos(theta)
    rightY = rightCenter[1] + turnRadius * np.sin(theta)
    ax.plot(leftX, leftY, "b")
    ax.plot(rightX, rightY, "b")


def plot_dubins_EZ(
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    captureRadius,
    pursuerRange,
    evaderHeading,
    evaderSpeed,
    ax,
    alpha=1.0,
):
    numPoints = 1000
    rangeX = 8.2
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading
    print(evaderHeadings.shape)

    ZTrue = csbez.in_dubins_engagement_zone_agumented(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        jnp.array([X, Y]).T,
        evaderHeadings,
        evaderSpeed,
    )

    ZTrue = ZTrue.reshape(numPoints, numPoints)
    # ZGeometric = ZGeometric.reshape(numPoints, numPoints)

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    colors = ["green"]
    # colors = ["red"]
    ax.contour(X, Y, ZTrue, levels=[0], colors=colors, linewidths=2, alpha=alpha)
    # ax.contourf(
    #     X, Y, ZTrue, levels=np.linspace(np.min(ZTrue), np.max(ZTrue), 100), alpha=0.5
    # )
    contour_proxy = plt.plot(
        [0], [0], color=colors[0], linewidth=2, label="CSBEZ", alpha=alpha
    )
    # add label so it can be added to legend

    # # ax.contour(X, Y, ZGeometric, cmap="summer")
    # # ax.scatter(*pursuerPosition, c="r")
    # ax.set_aspect("equal", "box")
    return ax


def plot_dubins_path(
    startPosition,
    startHeading,
    goalPosition,
    radius,
    captureRadius,
    tangentPoint,
    ax,
    path_type="LS",  # "LS" = left-then-straight, "RS" = right-then-straight
    width=3,
):
    # Compute circle centers from start pose
    leftCenter = np.array(
        [
            startPosition[0] - radius * np.sin(startHeading),
            startPosition[1] + radius * np.cos(startHeading),
        ]
    )
    rightCenter = np.array(
        [
            startPosition[0] + radius * np.sin(startHeading),
            startPosition[1] - radius * np.cos(startHeading),
        ]
    )

    # --- Capture circle arc (unchanged from your basic idea) ---
    theta_cap = np.linspace(0, -np.pi / 4, 100)
    xcr = goalPosition[0] + captureRadius * np.cos(theta_cap)
    ycr = goalPosition[1] + captureRadius * np.sin(theta_cap)
    ax.plot(xcr, ycr, "r", linewidth=width)

    # --- Turning arc from startPosition to tangentPoint ---
    if path_type in ("LS", "L"):
        # Left turn: CCW on left circle
        x_arc, y_arc = _arc_between_points(
            leftCenter, radius, startPosition, tangentPoint, ccw=True
        )
        ax.plot(x_arc, y_arc, "r", linewidth=width)

    if path_type in ("RS", "R"):
        # Right turn: CW on right circle
        x_arc, y_arc = _arc_between_points(
            rightCenter, radius, startPosition, tangentPoint, ccw=False
        )
        ax.plot(x_arc, y_arc, "r", linewidth=3)

    # --- Straight segment from tangent point to goal (or goal center) ---
    ax.plot(
        [tangentPoint[0], goalPosition[0]],
        [tangentPoint[1], goalPosition[1]],
        "r",
        linewidth=3,
    )

    ax.set_aspect("equal", "box")


def plot_dubins_reachable_set(
    pursuerPosition,
    pursuerHeading,
    pursuerRange,
    radius,
    ax,
    colors=["magenta"],
    alpha=1.0,
):
    numPoints = 1000
    rangeX = 8.0
    x = np.linspace(-rangeX, rangeX, numPoints)
    y = np.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    X = X.flatten()
    Y = Y.flatten()
    # Z = vectorized_find_shortest_dubins_path(
    #     pursuerPosition, pursuerHeading, np.array([X, Y]).T, radius
    # )
    # Z = Z.reshape(numPoints, numPoints) - pursuerRange
    Z = csbez.in_dubins_reachable_set_augmented(
        pursuerPosition, pursuerHeading, radius, pursuerRange, np.array([X, Y]).T
    )
    # Z = np.isclose(Z, pursuerRange, atol=1e-1)
    Z = Z.reshape(numPoints, numPoints)
    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)

    ax.contour(X, Y, Z <= 0.0, colors=colors, levels=[0], alpha=alpha, linewidths=2)
    # c = ax.contourf(X, Y, Z, levels=np.linspace(-1, 1, 101), alpha=0.5)
    # c = ax.pcolormesh(X, Y, Z, alpha=0.5, cmap="coolwarm", vmin=-1, vmax=1)
    # cbar = plt.colorbar(c, ax=ax)
    # contour_proxy = plt.plot(
    #     [0], [0], color=colors[0], linestyle="-", label="Reachable Set", linewidth=20
    # )
    contour_proxy = plt.plot([0], [0], color=colors[0], linewidth=2, label="RR")
    ax.set_aspect("equal", "box")
    return ax


def main():
    pursuerVelocity = 2
    minimumTurnRadius = 0.2
    pursuerRange = 3.0
    pursuerPosition = np.array([0, 0])
    pursuerHeading = np.pi / 2

    # point = np.array([1.0, -2.5])
    point = np.array([1.0, 1.0])

    # length = find_shortest_dubins_path(
    #     pursuerPosition, pursuerHeading, point, minimumTurnRadius
    # )
    leftCenter = np.array(
        [
            pursuerPosition[0] - minimumTurnRadius * np.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * np.cos(pursuerHeading),
        ]
    )
    rightCenter = np.array(
        [
            pursuerPosition[0] + minimumTurnRadius * np.sin(pursuerHeading),
            pursuerPosition[1] - minimumTurnRadius * np.cos(pursuerHeading),
        ]
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    xl = leftCenter[0] + minimumTurnRadius * np.cos(theta)
    yl = leftCenter[1] + minimumTurnRadius * np.sin(theta)
    xr = rightCenter[0] + minimumTurnRadius * np.cos(theta)
    yr = rightCenter[1] + minimumTurnRadius * np.sin(theta)

    offset = 0.5
    length_temp = pursuerRange - offset
    arcAngle = length_temp / minimumTurnRadius
    thetaTemp = -np.linspace(0, arcAngle, 100) - np.pi
    xr_temp = rightCenter[0] + minimumTurnRadius * np.cos(thetaTemp)
    yr_temp = (rightCenter[1] + offset) + minimumTurnRadius * np.sin(thetaTemp)

    # xcr = centerPoint2[0] + minimumTurnRadius * np.cos(theta)
    # ycr = centerPoint2[1] + minimumTurnRadius * np.sin(theta)

    ax = plot_dubins_reachable_set(
        pursuerPosition, pursuerHeading, pursuerRange, minimumTurnRadius
    )
    ax.plot(xl, yl)
    ax.plot(xr_temp, yr_temp)
    # ax.plot(xcr, ycr, c="g")

    ax.scatter(*pursuerPosition, c="r")
    ax.scatter(*point, c="k", marker="x")
    # ax.scatter(centerPoint2[0], centerPoint2[1], c="g")
    ax.plot(xr, yr)

    ax.set_aspect("equal", "box")
    # plot_dubins_path(
    #     pursuerPosition, pursuerHeading, point, minimumTurnRadius, 0.0, tangentPoint
    # )
    # pursuerTime = pursuerRange / pursuerVelocity
    # ax2 = dubinsReachable.plot_dubins_reachable_set(
    #     pursuerHeading - np.pi / 2, pursuerVelocity, minimumTurnRadius, pursuerTime
    # )

    plt.show()


def plot_test_grad(
    pursuerPosition,
    pursuerHeading,
    minimumTurnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
):
    numPoints = 500
    x = jnp.linspace(-2, 2, numPoints)
    y = jnp.linspace(-2, 2, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()

    grad = d_find_dubins_path_length_left_right_d_goal_position_vec(
        pursuerPosition, pursuerHeading, np.array([X, Y]).T, minimumTurnRadius
    )
    speedRatio = evaderSpeed / pursuerSpeed

    dLdxDirectional = (
        jnp.dot(grad, jnp.array([np.cos(evaderHeading), np.sin(evaderHeading)]))
        * speedRatio
        * pursuerRange
        - pursuerRange
    )
    leftCenter = np.array(
        [
            pursuerPosition[0] - minimumTurnRadius * np.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * np.cos(pursuerHeading),
        ]
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    xl = leftCenter[0] + 3 * minimumTurnRadius * np.cos(theta)
    yl = leftCenter[1] + 3 * minimumTurnRadius * np.sin(theta)

    dLdxDirectional = dLdxDirectional.reshape(numPoints, numPoints)
    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    fig, ax = plt.subplots()
    c = ax.contour(X, Y, dLdxDirectional, levels=[0, 1])
    testEvaderXVales = np.linspace(-1.0, 0.5, 10)
    testEvaderYValues = -np.ones_like(testEvaderXVales) * 0.2
    # testEvaderStart = np.array([-0.4, 0.1])
    ax.axis("equal")
    for i, x in enumerate(testEvaderXVales):
        testEvaderStart = np.array([x, testEvaderYValues[i]])

        inEz, goalPosition = in_dubins_engagement_zone_left_right_single(
            pursuerPosition,
            pursuerHeading,
            minimumTurnRadius,
            captureRadius,
            pursuerRange,
            pursuerSpeed,
            testEvaderStart,
            evaderHeading,
            evaderSpeed,
            ax,
        )
    plot_turn_radius_circles(pursuerPosition, pursuerHeading, minimumTurnRadius, ax)
    fig.colorbar(c, ax=ax)
    pursuerTime = pursuerRange / pursuerSpeed
    # ax.scatter(*goalPosition, c="r")
    ax.plot(xl, yl, "b")


def add_arrow(ax, start, end, color, label, annotationFontSize):
    # Draw arrow on plot
    # arrow = FancyArrowPatch(start, end, arrowstyle="->", color=color, lw=2)
    arrow = patches.FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="->",
        color=color,
        lw=2,
        mutation_scale=15,  # Controls arrowhead size
        transform=ax.transData,  # Use data coordinates
    )
    ax.add_patch(arrow)

    # Midpoint text label
    # mid = np.mean([start, end], axis=0)
    # ax.text(mid[0], mid[1], label, fontsize=annotationFontSize)

    # Create proxy for legend
    proxy = plt.plot([0], [0], color=color, lw=2, label=label)
    return proxy


def plot_theta_and_vectors_left_turn(
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    pursuerRange,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    ax,
):
    speedRatio = evaderSpeed / pursuerSpeed
    T = evaderPosition
    P = pursuerPosition
    F = evaderPosition + speedRatio * pursuerRange * np.array(
        [np.cos(evaderHeading), np.sin(evaderHeading)]
    )
    C = jnp.array(
        [
            pursuerPosition[0] - minimumTurnRadius * jnp.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * jnp.cos(pursuerHeading),
        ]
    )
    G = csbez.find_counter_clockwise_tangent_point(F, C, minimumTurnRadius)

    circleTheta = np.linspace(0, 2 * np.pi, 100)
    leftCircleX = C[0] + minimumTurnRadius * np.cos(circleTheta)
    leftCircleY = C[1] + minimumTurnRadius * np.sin(circleTheta)

    annotationFontSize = 12

    ax.scatter(*T, c="k")
    ax.scatter(*F, c="k")
    ax.scatter(*C, c="k")
    ax.scatter(*G, c="k")
    ax.scatter(*P, c="k")
    # put labels on points
    ax.text(
        T[0] - 0.001,  # shift left
        T[1] + 0.001,  # shift up
        "E",
        fontsize=annotationFontSize,
        ha="right",
        va="bottom",
    )
    ax.text(F[0], F[1] + 0.02, "F", fontsize=annotationFontSize)
    ax.text(C[0], C[1], r"$C_\ell$", fontsize=annotationFontSize, ha="right", va="top")
    ax.text(G[0] + 0.01, G[1] + 0.01, r"$G_\ell$", fontsize=annotationFontSize)
    ax.text(
        P[0] - 0.01, P[1] - 0.01, "P", fontsize=annotationFontSize, ha="right", va="top"
    )

    # plot circles
    # ax.plot(leftCircleX, leftCircleY, c="b")
    # ax.set_aspect("equal", "box")

    legend_proxies = []

    legend_proxies.append(
        add_arrow(
            ax,
            C,
            F,
            color="cyan",
            label=r"$v_1$",
            annotationFontSize=annotationFontSize,
        )
    )
    legend_proxies.append(
        add_arrow(
            ax, G, F, color="m", label="$v_2$", annotationFontSize=annotationFontSize
        )
    )
    legend_proxies.append(
        add_arrow(
            ax, C, G, color="r", label="$v_3$", annotationFontSize=annotationFontSize
        )
    )
    legend_proxies.append(
        add_arrow(
            ax, C, P, color="b", label="$v_4$", annotationFontSize=annotationFontSize
        )
    )
    legend_proxies.append(
        add_arrow(
            ax,
            T,
            F,
            color="orange",
            label="Evader Path",
            annotationFontSize=annotationFontSize,
        )
    )

    v3 = G - C
    v4 = P - C
    theta = csbez.counterclockwise_angle(v4, v3)
    print(theta)
    startAngle = np.arctan2(v4[1], v4[0])
    endAngle = np.arctan2(v3[1], v3[0])
    arcDrawDistance = 1.0 / 3.0
    # label theta
    vmean = np.mean([v3, v4], axis=0)
    ax.text(
        C[0] + 0.04,
        C[1] - 0.01,
        r"$\theta_\ell$",
        fontsize=annotationFontSize,
        va="center",
    )


def main_EZ():
    pursuerPosition = np.array([0, 0])

    pursuerHeading = np.pi / 4

    pursuerSpeed = 2

    pursuerRange = 2.5
    minimumTurnRadius = 0.47
    captureRadius = 0.0
    evaderHeading = (0 / 20) * np.pi
    evaderSpeed = 1
    evaderPosition = np.array([-1.14, 2.46])
    evaderPosition = np.array([-1.14, 2.66])
    startTime = time.time()

    # length, tangentPoint = find_dubins_path_length_right_strait(
    #     pursuerPosition, pursuerHeading, evaderPosition, minimumTurnRadius
    # )
    # print("length", length)
    # plot_dubins_path(
    #     pursuerPosition,
    #     pursuerHeading,
    #     evaderPosition,
    #     minimumTurnRadius,
    #     0.0,
    #     tangentPoint,
    # )
    #
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")

    plot_dubins_reachable_set(
        pursuerPosition, pursuerHeading, pursuerRange, minimumTurnRadius, ax
    )
    plot_dubins_EZ(
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        evaderHeading,
        evaderSpeed,
        ax,
    )

    speedRatio = evaderSpeed / pursuerSpeed
    F = evaderPosition + speedRatio * pursuerRange * np.array(
        [np.cos(evaderHeading), np.sin(evaderHeading)]
    )
    ax.plot([evaderPosition[0], F[0]], [evaderPosition[1], F[1]], c="b", linewidth=3)
    C = jnp.array(
        [
            pursuerPosition[0] - minimumTurnRadius * jnp.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * jnp.cos(pursuerHeading),
        ]
    )
    G = csbez.find_counter_clockwise_tangent_point(F, C, minimumTurnRadius)
    plot_dubins_path(pursuerPosition, pursuerHeading, F, minimumTurnRadius, 0.0, G, ax)

    # plot_theta_and_vectors_left_turn(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     minimumTurnRadius,
    #     pursuerRange,
    #     evaderPosition,
    #     evaderHeading,
    #     evaderSpeed,
    #     ax,
    # )
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_xlim([-3.1, 3.1])
    ax.set_ylim([-3.1, 3.1])
    draw_airplanes.draw_airplane(
        ax, evaderPosition, color="blue", size=0.35, angle=evaderHeading - np.pi / 2
    )  # Blue airplane (evader) pointing toward x-axis
    draw_airplanes.draw_airplane(
        ax, pursuerPosition, color="red", size=0.35, angle=pursuerHeading - np.pi / 2
    )
    # set tick fonhtsize
    ax.tick_params(axis="both", which="major")
    # put legend outside the plot
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.scatter(*F, c="r", s=100, zorder=2000)
    plt.show()


def bez_csbez_comparison():
    pursuerPosition = np.array([0, 0])

    pursuerHeading = np.pi / 4

    pursuerSpeed = 2

    pursuerRange = 2.5
    minimumTurnRadius = 0.47
    captureRadius = 0.0
    evaderHeading = (0 / 20) * np.pi
    evaderSpeed = 1
    evaderPosition = np.array([-1.14, 2.46])
    startTime = time.time()

    # length, tangentPoint = find_dubins_path_length_right_strait(
    #     pursuerPosition, pursuerHeading, evaderPosition, minimumTurnRadius
    # )
    # print("length", length)
    # plot_dubins_path(
    #     pursuerPosition,
    #     pursuerHeading,
    #     evaderPosition,
    #     minimumTurnRadius,
    #     0.0,
    #     tangentPoint,
    # )
    #
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    pez_plotting.plotEngagementZone(
        evaderHeading,
        pursuerPosition,
        pursuerRange,
        captureRadius,
        pursuerSpeed,
        evaderSpeed,
        ax,
        width=2,
        color="orange",
    )
    plot_dubins_EZ(
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        evaderHeading,
        evaderSpeed,
        ax,
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_xlim([-4.1, 3.1])
    ax.set_ylim([-4.1, 3.1])
    ax.tick_params(axis="both", which="major")
    ax.set_aspect("equal", "box")
    # put legend outside the plot
    ax.legend(loc="lower right")
    plt.show()


def pursuer_heading_vs_ez_boundary():
    pursuerPosition = np.array([0, 0])

    pursuerSpeed = 2

    pursuerRange = 2.5
    minimumTurnRadius = 0.47
    captureRadius = 0.0
    evaderHeading = (0 / 20) * np.pi
    evaderSpeed = 1
    evaderPosition = np.array([-1.54, 0.76])
    startTime = time.time()

    numHeadings = 200
    headings = np.linspace(0, 2 * np.pi, numHeadings)
    ez_boundary_points = []
    headings_plot = []
    for i, heading in enumerate(headings):
        ez = bez.in_dubins_engagement_zone_single(
            pursuerPosition,
            heading,
            minimumTurnRadius,
            captureRadius,
            pursuerRange,
            pursuerSpeed,
            evaderPosition,
            evaderHeading,
            evaderSpeed,
        )
        print(ez)

        ez_boundary_points.append(ez)
        headings_plot.append(heading)

        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([-2.2, 1.5])
        ax.scatter(
            headings_plot,
            ez_boundary_points,
            c="b",
            linewidth=2,
        )
        ax.set_xlabel("Pursuer Heading (rad)")
        ax.set_ylabel("Engagement Zone Function")
        plt.savefig(f"video/{i}.png")
        fig2, ax2 = plt.subplots(figsize=(6, 5), layout="constrained")

        # plot evader final position and turn radius and dubins path
        ax2.set_xlim([-3.1, 3.1])
        ax2.set_ylim([-3.1, 3.1])
        plot_dubins_reachable_set(
            pursuerPosition, heading, pursuerRange, minimumTurnRadius, ax2
        )
        # plot_dubins_EZ(
        #     pursuerPosition,
        #     heading,
        #     pursuerSpeed,
        #     minimumTurnRadius,
        #     captureRadius,
        #     pursuerRange,
        #     evaderHeading,
        #     evaderSpeed,
        #     ax2,
        # )

        speedRatio = evaderSpeed / pursuerSpeed
        F = evaderPosition + speedRatio * pursuerRange * np.array(
            [np.cos(evaderHeading), np.sin(evaderHeading)]
        )
        ax2.plot(
            [evaderPosition[0], F[0]], [evaderPosition[1], F[1]], c="b", linewidth=3
        )
        Cl = jnp.array(
            [
                pursuerPosition[0] - minimumTurnRadius * jnp.sin(heading),
                pursuerPosition[1] + minimumTurnRadius * jnp.cos(heading),
            ]
        )
        Cr = jnp.array(
            [
                pursuerPosition[0] + minimumTurnRadius * jnp.sin(heading),
                pursuerPosition[1] - minimumTurnRadius * jnp.cos(heading),
            ]
        )
        Gl = csbez.find_counter_clockwise_tangent_point(F, Cl, minimumTurnRadius)
        Gr = csbez.find_clockwise_tangent_point(F, Cr, minimumTurnRadius)
        rightLength, _ = csbez.find_dubins_path_length_right_strait(
            pursuerPosition, heading, F, minimumTurnRadius
        )
        leftLength, _ = csbez.find_dubins_path_length_left_strait(
            pursuerPosition, heading, F, minimumTurnRadius
        )
        if rightLength < leftLength:
            G = Gr
            C = Cr
            type = "RS"
        else:
            G = Gl
            C = Cl
            type = "LS"
        plot_dubins_path(
            pursuerPosition, heading, F, minimumTurnRadius, 0.0, G, ax2, path_type=type
        )

        # plot_theta_and_vectors_left_turn(
        #     pursuerPosition,
        #     pursuerHeading,
        #     pursuerSpeed,
        #     minimumTurnRadius,
        #     pursuerRange,
        #     evaderPosition,
        #     evaderHeading,
        #     evaderSpeed,
        #     ax,
        # )
        plt.xlabel("X")
        plt.ylabel("Y")
        ax2.set_xlim([-3.1, 3.1])
        ax2.set_ylim([-3.1, 3.1])
        draw_airplanes.draw_airplane(
            ax2,
            evaderPosition,
            color="blue",
            size=0.35,
            angle=evaderHeading - np.pi / 2,
        )  # Blue airplane (evader) pointing toward x-axis
        draw_airplanes.draw_airplane(
            ax2,
            pursuerPosition,
            color="red",
            size=0.35,
            angle=heading - np.pi / 2,
        )
        plt.savefig(f"video2/{i}.png")

        # close all figs
        plt.close("all")


def uncertain_dubins_ez_plot():
    pursuerPositionRange = 4

    pursuerHeadingRange = np.pi

    pursuerRangeRange = [1.6, 2]
    minimumTurnRadius = 0.47
    pursuerMinimumTurnRadiusRange = [0.30, 0.5]
    fig, ax = plt.subplots()
    numPursuerSamples = 20
    for i in range(numPursuerSamples):
        pursuerPosition = np.array(
            [
                np.random.uniform(-pursuerPositionRange, pursuerPositionRange),
                np.random.uniform(-pursuerPositionRange, pursuerPositionRange),
            ]
        )
        pursuerHeading = np.random.uniform(-pursuerHeadingRange, pursuerHeadingRange)
        pursuerRange = np.random.uniform(pursuerRangeRange[0], pursuerRangeRange[1])
        minimumTurnRadius = np.random.uniform(
            pursuerMinimumTurnRadiusRange[0], pursuerMinimumTurnRadiusRange[1]
        )
        plot_dubins_reachable_set(
            pursuerPosition,
            pursuerHeading,
            pursuerRange,
            minimumTurnRadius,
            ax,
            alpha=0.3,
        )
    # plot box of potential pursuer startPosition
    ax.plot(
        [
            -pursuerPositionRange,
            pursuerPositionRange,
            pursuerPositionRange,
            -pursuerPositionRange,
            -pursuerPositionRange,
        ],
        [
            -pursuerPositionRange,
            -pursuerPositionRange,
            pursuerPositionRange,
            pursuerPositionRange,
            -pursuerPositionRange,
        ],
        c="red",
        linewidth=2,
    )
    ax.set_aspect("equal", "box")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    plt.show()


if __name__ == "__main__":
    # pursuer_heading_vs_ez_boundary()
    # main_EZ()
    # uncertain_dubins_ez_plot()
    bez_csbez_comparison()
    # fig, ax = plt.subplots()
    # ax.scatter(0, 0, c="r")
# plt.show()
