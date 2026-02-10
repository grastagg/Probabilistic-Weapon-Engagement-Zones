import os


from operator import inv
from jax.profiler import start_trace, stop_trace

import os
from jax.lax import le
import numpy as np
import chaospy as cp

import jax.numpy as jnp
import jax
import time

from scipy.stats import zmap


import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg")

# get rid of type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# set font size for title, axis labels, and legend, and tick labels
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10

import CSPEZ.csbez as csbez
import CSPEZ.cspez as cspez

import CSPEZ.nueral_network_cspez as nueral_network_cspez


numPoints = 150


def plot_dubins_PEZ(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
    ax,
    useLinear=False,
    useUnscented=False,
    useQuadratic=False,
    useMC=False,
    useNumerical=False,
    useNueralNetwork=False,
    useLinearPlusNueralNetwork=False,
    labelX=True,
    labelY=True,
    levels=None,
):
    rangeX = 5.5
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading
    if useLinear:
        start = time.time()
        ZTrue, _, _ = cspez.linear_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        time.sleep(1)
        print("linear_dubins_pez time", time.time() - start)
        ax.set_title("LCSPEZ")
    elif useUnscented:
        ZTrue, _, _ = cspez.uncented_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Unscented Dubins PEZ")
    elif useMC:
        start = time.time()
        ZTrue, _, _, _, _, _, _ = cspez.mc_dubins_PEZ(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        print("mc_dubins_PEZ time", time.time() - start)
        ax.set_title("MCCSPEZ")
    elif useQuadratic:
        start = time.time()
        ZTrue, _, _ = cspez.quadratic_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        print("quadratic_dubins_pez time", time.time() - start)
        ax.set_title("QCSPEZ")
    elif useNumerical:
        start = time.time()
        ZTrue, _, _ = cspez.dubins_pez_numerical_integration_sparse(
            # ZTrue, _, _, _ = numerical_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
            cspez.nodes,
            cspez.weights,
        )
        print("numerical integration time", time.time() - start)
        ax.set_title("Numerical Integration Dubins PEZ")
    elif useNueralNetwork:
        start = time.time()
        ZTrue, _, _ = nueral_network_cspez.nueral_network_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("NNCSPEZ")
        print("nueral_network_EZ time", time.time() - start)
    elif useLinearPlusNueralNetwork:
        start = time.time()
        ZLinear, _, _ = cspez.linear_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ZNueralNetwork = nueral_network_cspez.nueral_network_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ZTrue = ZLinear - ZNueralNetwork

    ZTrue = np.array(ZTrue.block_until_ready())
    ZTrue = ZTrue.reshape(numPoints, numPoints)

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    if levels is None:
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    c = ax.contour(
        X,
        Y,
        ZTrue,
        levels=levels,
    )
    inLine = False
    if inLine:
        ax.clabel(c, inline=True)
    else:
        handles, labels = c.legend_elements()
        labels = [f"$\\epsilon={lvl:.2f}$" for lvl in c.levels]

        ax.legend(
            handles,
            labels,
            title="PEZ Level",
            loc="lower right",
            framealpha=0.8,
        )
    # c = ax.pcolormesh(X, Y, ZTrue)
    if labelX:
        ax.set_xlabel("X")
    if labelY:
        ax.set_ylabel("Y")
    # set tick size
    ax.tick_params(axis="both", which="major")
    ax.set_aspect("equal", "box")
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlim([-1.5, 1.5])
    ax.set_xticks(np.arange(-1.0, 2.0, 1.0))
    ax.set_yticks(np.arange(-1.0, 2.0, 1.0))

    # ax.contour(X, Y, ZGeometric, cmap="summer")
    # ax.scatter(*pursuerPosition, c="r")
    ax.set_aspect("equal", "box")
    ax.set_aspect("equal", "box")

    return ZTrue.flatten()


def plot_dubins_PEZ_diff(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
    ax,
    useLinear=False,
    useUnscented=False,
    useQuadratic=False,
    useNumerical=False,
    useNueralNetwork=False,
    useLinearPlusNueralNetwork=False,
    plotColorBar=False,
    ylabel=False,
):
    rangeX = 1.5
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    points = jnp.array([X, Y]).T
    evaderHeadings = np.ones_like(X) * evaderHeading
    ZMC, _, _, _, _, _, _ = cspez.mc_dubins_PEZ(
        points,
        evaderHeadings,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadindgVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        captureRadius,
    )
    ZMC = np.array(ZMC.block_until_ready())
    print("max ZMC", jnp.max(ZMC))
    print("min ZMC", jnp.min(ZMC))
    if useLinear:
        print("Linear")
        ZTrue, _, _ = cspez.linear_dubins_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("LCSPEZ")
    elif useUnscented:
        print("Unscented")
        ZTrue, _, _ = cspez.uncented_dubins_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Unscented Dubins PEZ")
    elif useQuadratic:
        print("Quadratic")
        ZTrue, _, _ = cspez.quadratic_dubins_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("QCSPEZ")
    elif useNumerical:
        print("Piecewise Linear")
        ZTrue, _, _ = cspez.dubins_pez_numerical_integration_sparse(
            # ZTrue, _, _, _ = numerical_dubins_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
            cspez.nodes,
            cspez.weights,
        )
        ax.set_title("Numerical Dubins PEZ")
    elif useNueralNetwork:
        print("Neural Network")
        ZTrue, _, _ = nueral_network_cspez.nueral_network_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("NNCSPEZ")
    elif useLinearPlusNueralNetwork:
        print("Linear + Neural Network")
        ZLinear, _, _ = cspez.linear_dubins_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ZNueralNetwork, _, _ = nueral_network_cspez.nueral_network_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ZTrue = ZLinear - ZNueralNetwork

    ZTrue = np.array(ZTrue.block_until_ready())
    rmse = jnp.sqrt(jnp.mean((ZTrue - ZMC) ** 2))
    print("RMSE", rmse)
    average_abs_diff = jnp.mean(jnp.abs(ZTrue - ZMC))
    print("Average Abs Diff", average_abs_diff)
    max_abs_diff = jnp.max(jnp.abs(ZTrue - ZMC))
    print("Max Abs Diff", max_abs_diff)
    ZMC = ZMC.reshape(numPoints, numPoints)
    ZTrue = ZTrue.reshape(numPoints, numPoints)
    # write rmse on image
    font = 8
    ax.text(
        0.0,
        1.35,
        f"RMSE: {rmse:.4f}",
        fontsize=font,
        ha="center",
        va="center",
        color="white",
    )
    ax.text(
        0.0,
        1.15,
        f"Avg Abs Diff: {average_abs_diff:.4f}",
        fontsize=font,
        ha="center",
        va="center",
        color="white",
    )
    ax.text(
        0.0,
        0.95,
        f"Max Abs Diff: {max_abs_diff:.4f}",
        fontsize=font,
        ha="center",
        va="center",
        color="white",
    )

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    c = ax.pcolormesh(
        X,
        Y,
        jnp.abs(ZTrue - ZMC),
        vmin=0,
        vmax=0.5,
        shading="nearest",
        edgecolors="none",
        rasterized=True,
    )
    # make colorbar smaller
    #
    if plotColorBar:
        cb = plt.colorbar(c, ax=ax, shrink=0.5)

    # if useUnscented:
    ax.set_xlabel("X")
    if ylabel:
        ax.set_ylabel("Y")
    ax.set_xticks(np.arange(-1.0, 2.0, 1.0))
    ax.set_yticks(np.arange(-1.0, 2.0, 1.0))

    # ax.contour(X, Y, ZGeometric, cmap="summer")
    # ax.scatter(*pursuerPosition, c="r")
    ax.set_aspect("equal", "box")
    ax.set_aspect("equal", "box")
    # csbez.plot_dubins_EZ(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     minimumTurnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     evaderHeading,
    #     evaderSpeed,
    #     ax,
    # )

    return ZTrue.flatten()


def plot_EZ_vs_pursuer_range(
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    pursuerRange,
    minimumTurnRadius,
    captureRadius,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
):
    fig, ax = plt.subplots()
    # pursuerHeading = np.linspace(-np.pi, np.pi, 100)
    # turn radius derivat
    #
    dDubinsEZ_dRange = cspez.dDubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    print("dDubinsEZ_dPursuerRange", dDubinsEZ_dRange)
    ezMean = cspez.in_dubins_engagement_zone(
        pursuerPosition,
        jnp.array([pursuerHeading]),
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )

    pursuerHeadingVec = np.linspace(-np.pi, np.pi, 1000)
    ez = cspez.in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeadingVec,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    ax.scatter(pursuerHeadingVec, ez)
    # plot tangent point
    ax.plot(
        pursuerHeadingVec,
        dDubinsEZ_dRange * (pursuerHeadingVec - pursuerHeading) + ezMean,
    )


def plot_normal(mean, var, ax, label):
    x = np.linspace(mean - 3 * np.sqrt(var), mean + 3 * np.sqrt(var), 100)
    y = jax.scipy.stats.norm.pdf(x, mean, np.sqrt(var))
    ax.plot(x, y, label=label)


def compare_distribution(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    (
        inEZ,
        ez,
        pursuerPositionSamples,
        pursuerHeadingSamples,
        pursuerSpeedSamples,
        turnRadiusSamples,
        rangeSamples,
    ) = cspez.mc_dubins_PEZ(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        captureRadius,
    )
    print("mc", inEZ)
    print("mc mean", np.mean(ez))
    print("mc var", np.var(ez))
    inEZ, linMean, linVar = cspez.linear_dubins_pez(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        captureRadius,
    )
    print("lin ", inEZ)
    print("lin mean", linMean)
    print("lin var", linVar)
    inEZ, qMean, qVar = cspez.quadratic_dubins_pez(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        captureRadius,
    )
    print("quadratic", inEZ)
    print("quadratic mean", qMean)
    print("quadratic var", qVar)
    fig, ax = plt.subplots()
    inEZ, mean, var = cspez.cubic_dubins_PEZ(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        captureRadius,
    )
    print("cubic", inEZ)
    print("cubic mean", mean)
    print("cubic var", var)
    pursuerParams = jnp.concatenate(
        [
            pursuerPosition,  # (2,)
            jnp.array([pursuerHeading]),  # (1,)
            jnp.array([pursuerSpeed]),  # (1,)
            jnp.array([minimumTurnRadius]),  # (1,)
            jnp.array([pursuerRange]),  # (1,)
        ]
    )
    evaderParams = jnp.concatenate(
        [
            evaderPosition.flatten(),
            jnp.array([evaderHeading[0]]),
            jnp.array([evaderSpeed]),
        ]
    )
    inEZ, meanrightlin, varrightlin, dDubinsEZRight_dPursuerParamsValue = (
        cspez.linear_dubins_pez_right(
            evaderPosition,
            evaderHeading,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadingVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
    )
    print("linear right", inEZ)
    print("linear right mean", meanrightlin)
    print("linear right var", varrightlin)
    inEZ, meanleftlin, varleftlin, dDubinsEZLeft_dPursuerParamsValue = (
        cspez.linear_dubins_pez_left(
            evaderPosition,
            evaderHeading,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadingVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
    )
    print("linear left", inEZ)
    print("linear left mean", meanleftlin)
    print("linear left var", varleftlin)
    inEZ = cspez.combined_left_right_dubins_PEZ(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        captureRadius,
    )
    print("combined linear", inEZ)
    inEZ, slopes, intercepts, bounds = cspez.piecewise_linear_dubins_pez_heading_only(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        captureRadius,
    )
    slopes = slopes.flatten()
    intercepts = intercepts.flatten()
    bounds = bounds.flatten()
    print("piecewise linear", inEZ)
    print("slopes", slopes)
    print("intercepts", intercepts)
    print("bounds", bounds)

    sortedEZindices = jnp.argsort(ez).flatten()
    ezSorted = ez.flatten()[sortedEZindices]

    plt.hist(ez, bins=1000, density=True)
    print("min turn radius", np.min(turnRadiusSamples))
    plot_normal(linMean, linVar, ax, "Linear")
    plot_normal(qMean, qVar, ax, "Quadratic")
    # plot_normal(meanrightlin, varrightlin, ax, "Right Linear")
    # plot_normal(meanleftlin, varleftlin, ax, "Left Linear")
    # plot vertical line at 0
    ax.axvline(0, color="k", linestyle="dashed", linewidth=1)
    # plot linear and unscented cdfs

    plt.legend()

    dDubinsEZ_dPursuerHeadingValue = cspez.dDubinsEZ_dPursuerHeading(
        pursuerPosition.flatten(),
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition.flatten(),
        evaderHeading[0],
        evaderSpeed,
    )
    d2DubinsEZ_dPursuerHeadingValue = cspez.d2DubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition.flatten(),
        evaderHeading[0],
        evaderSpeed,
    )
    d3DubinsEZ_dPursuerHeadingValue = cspez.d3DubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition.flatten(),
        evaderHeading[0],
        evaderSpeed,
    )
    print("dDubinsEZ_dPursuerHeadingValue", dDubinsEZ_dPursuerHeadingValue)
    print("d2DubinsEZ_dPursuerHeadingValue", d2DubinsEZ_dPursuerHeadingValue)
    print("d3DubinsEZ_dPursuerHeadingValue", d3DubinsEZ_dPursuerHeadingValue)

    # right and left derivatives
    print("dDubinsEZright_dPursuerParamsValue", dDubinsEZRight_dPursuerParamsValue)
    dDubinsEZright_dPursuerHeadingValue = dDubinsEZRight_dPursuerParamsValue[0][2]
    dDubinsEZleft_dPursuerHeadingValue = dDubinsEZLeft_dPursuerParamsValue[0][2]

    print("dDubinsEZright_dPursuerHeadingValue", dDubinsEZright_dPursuerHeadingValue)
    print("dDubinsEZleft_dPursuerHeadingValue", dDubinsEZleft_dPursuerHeadingValue)

    sortIndex = np.argsort(pursuerHeadingSamples.flatten())
    ez = ez.flatten()[sortIndex]
    pursuerHeadingSamples = pursuerHeadingSamples.flatten()[sortIndex]

    fig1, ax1 = plt.subplots()
    ax1.scatter(pursuerHeadingSamples, ez, label="Monte Carlo")
    linApprox = (
        dDubinsEZ_dPursuerHeadingValue * (pursuerHeadingSamples - pursuerHeading)
        + linMean
    ).flatten()
    leftLinApprox = (
        dDubinsEZleft_dPursuerHeadingValue * (pursuerHeadingSamples - pursuerHeading)
        + meanleftlin
    ).flatten()
    rightLinApprox = (
        dDubinsEZright_dPursuerHeadingValue * (pursuerHeadingSamples - pursuerHeading)
        + meanrightlin
    ).flatten()
    quadApprox = (
        0.5
        * d2DubinsEZ_dPursuerHeadingValue
        * (pursuerHeadingSamples - pursuerHeading) ** 2
        + linApprox
    )
    cubicApprox = (
        (1 / 6)
        * d3DubinsEZ_dPursuerHeadingValue
        * (pursuerHeadingSamples - pursuerHeading) ** 3
        + quadApprox
    ).flatten()
    ax1.plot(
        pursuerHeadingSamples.flatten(),
        linApprox,
        label="Tangent",
    )
    ax1.plot(pursuerHeadingSamples.flatten(), quadApprox.flatten(), label="Quadratic")
    numSamples = 100

    for i in range(len(slopes)):
        if i % 2 == 0:
            color = "r"
        else:
            color = "lime"
        a = slopes[i]
        b = intercepts[i]
        bound = (bounds[i], bounds[i + 1])
        y = cspez.evaluate_linear_model(a, b, bound, numSamples)
        ax1.plot(
            jnp.linspace(bound[0], bound[1], numSamples),
            y,
            label="Piecewise Linear Model",
            color=color,
        )
    # ax1.scatter(pursuerHeadingSamples.flatten(), rightEZ, label="Right")
    # ax1.scatter(pursuerHeadingSamples.flatten(), leftEZ, label="Left")
    # ax1.plot(pursuerHeadingSamples.flatten(), leftLinApprox, label="Left Linear")
    # ax1.plot(pursuerHeadingSamples.flatten(), rightLinApprox, label="Right Linear")


def plot_heading_vector(pursuerPosition, pursuerHeading, pursuerPositionCov, ax):
    ax.quiver(
        pursuerPosition[0],
        pursuerPosition[1],
        0.5 * np.cos(pursuerHeading),
        0.5 * np.sin(pursuerHeading),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="r",
    )
    # plot eigenvectors of covariance matrix
    [eigenvalues, eigenvectors] = np.linalg.eig(pursuerPositionCov)
    for i in range(len(eigenvalues)):
        length = eigenvalues[i]
        ax.quiver(
            pursuerPosition[0],
            pursuerPosition[1],
            length * eigenvectors[0][i],
            length * eigenvectors[1][i],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="b",
        )


def compare_PEZ(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
):
    fig, axes = plt.subplots(2, 2, figsize=(4, 4), layout="constrained")
    linEZ = plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[0][1],
        useLinear=True,
        labelX=False,
        labelY=False,
    )
    mcEZ = plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[0][0],
        useMC=True,
        labelX=False,
        labelY=True,
    )
    # plot_heading_vector(pursuerPosition, pursuerHeading, pursuerPositionCov, axes)
    quadEZ = plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[1][0],
        useQuadratic=True,
        labelX=True,
        labelY=True,
    )
    # numerical = plot_dubins_PEZ(
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[2],
    #     useNumerical=True,
    # )
    nueralEZ = plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[1][1],
        # useNumerical=True,
        useNueralNetwork=True,
        # useLinearPlusNueralNetwork=True,
        labelX=True,
        labelY=False,
    )

    #
    # print("unscented rmse", np.sqrt(np.mean((usEZ - mcEZ) ** 2)))
    # print("quadratic rmse", np.sqrt(np.mean((quadEZ - mcEZ) ** 2)))
    # fig = plt.gcf()
    # fig.colorbar(c)
    # tight layout
    # fig.tight_layout()
    if True:
        save_dir = os.path.expanduser("~/Desktop/cspez_plot")
        fig_path = os.path.join(save_dir, "pez_example.pdf")
        fig.savefig(fig_path, format="pdf", bbox_inches="tight")


2


def plot_all_error(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
):
    evaderHeading = jnp.array([(0 / 20) * np.pi])
    fig, axes = plt.subplots(1, 3, constrained_layout=True, figsize=(6, 2.5))
    linEZ = plot_dubins_PEZ_diff(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[0],
        useLinear=True,
        ylabel=True,
    )
    # usEZ = plot_dubins_PEZ_diff(
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[1],
    #     useNumerical=True,
    # )
    usEZ = plot_dubins_PEZ_diff(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[2],
        # useNumerical=True,
        useNueralNetwork=True,
        # useLinearPlusNueralNetwork=True,
        plotColorBar=True,
    )
    quadEZ = plot_dubins_PEZ_diff(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[1],
        useQuadratic=True,
    )
    if True:
        save_dir = os.path.expanduser("~/Desktop/cspez_plot")
        fig_path = os.path.join(save_dir, "pez_example_error.pdf")
        fig.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCov = np.array([[0.025, 0.04], [0.04, 0.1]])
    # pursuerPositionCov = np.array([[0.5, 0.0], [0.0, 0.25]])
    # pursuerPositionCov = np.array([[0.000000000001, 0.0], [0.0, 0.00000000001]])

    pursuerHeading = (10.0 / 20.0) * np.pi
    evaderHeading = jnp.array([(0.0 / 20.0) * np.pi])

    # create rotation matrix that rotates pursuerheading to zero
    rotatePursuerHeading = False
    if rotatePursuerHeading:
        rotation = np.array(
            [
                [np.cos(pursuerHeading), -np.sin(pursuerHeading)],
                [np.sin(pursuerHeading), np.cos(pursuerHeading)],
            ]
        )
        pursuerPositionCov = rotation.T @ pursuerPositionCov @ rotation

        pursuerHeading -= pursuerHeading
        evaderHeading -= pursuerHeading

    pursuerHeadingVar = 0.2

    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.3
    # pursuerSpeedVar = 0.0

    pursuerRange = 1.0
    pursuerRangeVar = 0.1
    # pursuerRangeVar = 0.0

    minimumTurnRadius = 0.2
    minimumTurnRadiusVar = 0.005
    # minimumTurnRadiusVar = 0.0

    captureRadius = 0.0

    # evaderHeading = jnp.array((0.0 / 20.0) * np.pi)

    evaderSpeed = 0.5
    evaderPosition = np.array([0.2, -0.6])
    # compute mahalonobis distance or evader position
    z1 = csbez.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading[0],
        evaderSpeed,
    )
    evaderHeadingVector = np.array([np.cos(evaderHeading)[0], np.sin(evaderHeading)[0]])

    speedRatio = evaderSpeed / pursuerSpeed
    evaderFinalPosition = evaderPosition + evaderHeadingVector * speedRatio
    evaderMalahalobisDistance = np.sqrt(
        (evaderFinalPosition - pursuerPosition).T
        @ np.linalg.inv(pursuerPositionCov)
        @ (evaderFinalPosition - pursuerPosition)
    )
    print("Evader Mahalonobis Distance", evaderMalahalobisDistance)
    print("z1", z1)
    # compute heading mahalonobis distance
    evaderHeadingMahalonobisDistance = np.sqrt(
        (evaderHeading - pursuerHeading) ** 2 / pursuerHeadingVar
    )
    print("Evader Heading Mahalonobis Distance", evaderHeadingMahalonobisDistance)
    evaderPostion2 = np.array([0.0, 0.71])
    z2 = csbez.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPostion2,
        evaderHeading[0],
        evaderSpeed,
    )
    evaderFinalPosition2 = evaderPostion2 + evaderHeadingVector * speedRatio
    evaderMalahalobisDistance2 = np.sqrt(
        (evaderFinalPosition2 - pursuerPosition).T
        @ np.linalg.inv(pursuerPositionCov)
        @ (evaderFinalPosition2 - pursuerPosition)
    )
    print("Evader Mahalonobis Distance 2", evaderMalahalobisDistance2)
    print("z2", z2)

    # evaderPosition = np.array([[0.7487437185929648, 0.04522613065326636]])
    # evaderPosition = np.array([[0.205, -0.89]])

    # compare_distribution(
    #     evaderPosition,
    #     evaderHeading,
    #     evaderSpeed,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     captureRadius,
    # )
    plot_all_error(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
    )
    compare_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
    )
    #
    plt.show()


if __name__ == "__main__":
    main()
