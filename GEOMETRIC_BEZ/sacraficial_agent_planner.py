import numpy as np
from pyoptsparse import Optimization, OPT, IPOPT
from scipy.interpolate import BSpline
import time
from tqdm import tqdm
from jax import jacfwd
from jax import jit
from functools import partial
import jax.numpy as jnp
import getpass
import matplotlib.pyplot as plt
import matplotlib
import jax


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import GEOMETRIC_BEZ.pez_from_interceptions as pez_from_interceptions


import bspline.spline_opt_tools as spline_opt_tools


numSamplesPerInterval = 15


def plot_spline(spline, ax, width=1):
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, 1000, endpoint=True)
    ax.set_xlabel("X", fontsize=26)
    ax.set_ylabel("Y", fontsize=26)
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26)

    pos = spline(t)
    x = pos[:, 0]
    y = pos[:, 1]
    splineDot = spline.derivative()(t)
    xDot = splineDot[:, 0]
    yDot = splineDot[:, 1]
    agentHeadings = np.arctan2(yDot, xDot)

    pos = spline(t)
    ax.plot(x, y, linewidth=width)

    ax.set_aspect(1)
    plt.xlabel("X")
    plt.ylabel("Y")


# @jax.jit
# def area_of_circle_intersections(centers, radii, points, dArea):
#     dists_squared = jnp.square(points[:, None, :] - centers[None, :, :]).sum(axis=-1)
#     in_all_circles = jnp.all(dists_squared <= radii[None, :] ** 2, axis=-1)
#     area = jnp.sum(in_all_circles) * dArea
#     return area
#
#
# def area_diff_single(
#     newInterseptionLocation,
#     radius,
#     pastInterseptionLocations,
#     pastRadaii,
#     dArea,
#     integrationPoints,
#     oldArea,
# ):
#     combinedCenters = jnp.vstack([pastInterseptionLocations, newInterseptionLocation])
#     combinedRadii = jnp.hstack([pastRadaii, radius])
#     newArea = area_of_circle_intersections(
#         combinedCenters, combinedRadii, integrationPoints, dArea
#     )
#     # probReachable = pez_from_interceptions.prob_reach_numerical(
#     #     jnp.array([newInterseptionLocation]),
#     #     points,
#     #     launchPdf,
#     #     pursuerRange + pursuerCaptureRadius,
#     #     dArea,
#     # )[0]
#
#     return oldArea - newArea  # * probReachable
#
#
# # vectorized version
# area_diff = jax.jit(
#     jax.vmap(
#         area_diff_single,
#         in_axes=(0, None, None, None, None, None, None),
#     )
# )
#
#
# def hazard_from_reach(p_reach, ds, alpha=1.0):
#     # small-hazard approximation (stable and simple)
#     h = alpha * p_reach * ds
#     return jnp.clip(h, 0.0, 1.0)
#
#
# def area_objective_function_trajectory(
#     controlPoints,
#     knotPoints,
#     pursuerRange,
#     pursuerCaptureRadius,
#     pastInterseptionLocations,
#     pastRadaii,
#     dArea,
#     integrationPoints,
#     oldArea,
#     launchPdf,
# ):
#     controlPoints = controlPoints.reshape((-1, 2))
#     pos = spline_opt_tools.evaluate_spline(
#         controlPoints, knotPoints, numSamplesPerInterval
#     )
#     areaDiffs = area_diff(
#         pos,
#         pursuerRange + pursuerCaptureRadius,
#         pastInterseptionLocations,
#         pastRadaii,
#         dArea,
#         integrationPoints,
#         oldArea,
#     )
#
#     # Reach probability at each trajectory sample: (K,)
#     p_reach = pez_from_interceptions.prob_reach_numerical(
#         pos, integrationPoints, launchPdf, pursuerRange + pursuerCaptureRadius, dArea
#     )
#
#     # Step length along path (K-1,), pad to (K,)
#     deltas = pos[1:] - pos[:-1]
#     ds = jnp.linalg.norm(deltas, axis=1)
#     ds = jnp.concatenate([ds, ds[-1:]])  # (K,)
#
#     # Hazard per step (K,)
#     h = hazard_from_reach(p_reach, ds, alpha=1.0)
#     # Survival-weighted expected gain
#     # S_{k-1} for each k can be built via cumulative product.
#     # S_prev[0]=1, S_prev[k]=prod_{i<k}(1-h_i)
#     S_prev = jnp.concatenate([jnp.ones((1,)), jnp.cumprod(1.0 - h[:-1])])  # (K,)
#
#     expected_gain = jnp.sum(S_prev * h * areaDiffs)  # scalar
#
#     return -expected_gain
#


@jax.jit
def old_feasible_mask(points, past_centers, past_radii):
    """
    points: (M,2)          integration grid over launch-space
    past_centers: (N,2)    past kill locations (centers of constraint disks in launch-space)
    past_radii: (N,)       corresponding radii
    returns:
        mask: (M,) bool, True where point is inside ALL past disks
    """
    # (M,N,2) -> (M,N)
    dists_squared = jnp.sum(
        jnp.square(points[:, None, :] - past_centers[None, :, :]), axis=-1
    )
    return jnp.all(dists_squared <= past_radii[None, :] ** 2, axis=1)


@jax.jit
def area_from_mask(mask, dArea):
    """mask: (M,) bool"""
    return jnp.sum(mask) * dArea


@jax.jit
def area_diff_single_from_oldmask(
    newInterseptionLocation,
    radius,
    integrationPoints,
    old_mask,
    oldArea,
    dArea,
):
    """
    Computes oldArea - newArea where:
      oldArea = area of points inside all past circles (given by old_mask)
      newArea = area of points inside old_mask AND inside the new circle centered at newInterseptionLocation
    """
    dsq_new = jnp.sum(
        jnp.square(integrationPoints - newInterseptionLocation[None, :]), axis=-1
    )  # (M,)
    in_new = dsq_new <= radius**2  # (M,)
    new_mask = old_mask & in_new
    newArea = jnp.sum(new_mask) * dArea
    return oldArea - newArea


# Vectorized over a trajectory of candidate interception/kill locations: pos is (K,2)
area_diff_from_oldmask = jax.jit(
    jax.vmap(
        area_diff_single_from_oldmask,
        in_axes=(0, None, None, None, None, None),
    )
)


@jax.jit
def hazard_from_reach(p_reach, ds, alpha=1.0):
    """
    Convert reachability 'likelihood' into a per-step hazard.
    Uses an exponential survival model so it's stable for larger ds.
    p_reach: (K,)
    ds: (K,)
    """
    return 1.0 - jnp.exp(-alpha * p_reach * ds)


@jax.jit
def survival_prefix(h):
    """
    Given hazard h_k = P(kill at step k | alive at step k),
    returns S_prev where:
      S_prev[0] = 1
      S_prev[k] = Π_{i<k} (1 - h_i)
    """
    return jnp.concatenate([jnp.ones((1,)), jnp.cumprod(1.0 - h[:-1])])


@jax.jit
def area_objective_function_trajectory(
    controlPoints,
    knotPoints,
    pursuerRange,
    pursuerCaptureRadius,
    pastInterseptionLocations,
    pastRadaii,
    dArea,
    integrationPoints,
    launchPdf,
    alpha=1.0,
):
    """
    Fast, non-differentiable (hard-mask) expected-gain objective.
    - Precomputes old feasible mask once.
    - Computes areaDiffs using only the new circle test for each trajectory sample.

    NOTE: This recomputes oldArea from past circles for consistency (so you don't
    have to pass oldArea in). If you already have oldArea elsewhere and want to
    trust it, you can pass it and remove the recompute.
    """
    controlPoints = controlPoints.reshape((-1, 2))

    # (K,2) sampled trajectory positions (candidate kill points)
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )

    R_eff = pursuerRange + pursuerCaptureRadius

    # Precompute old feasible launch-region mask and area
    old_mask = old_feasible_mask(
        integrationPoints, pastInterseptionLocations, pastRadaii
    )
    oldArea = area_from_mask(old_mask, dArea)

    # ΔM_k = oldArea - newArea(pos_k)
    areaDiffs = area_diff_from_oldmask(
        pos, R_eff, integrationPoints, old_mask, oldArea, dArea
    )  # (K,)

    # Reach probability at each trajectory sample: (K,)
    p_reach = pez_from_interceptions.prob_reach_numerical(
        pos, integrationPoints, launchPdf, R_eff, dArea
    )

    # Step length along path (K-1,), pad to (K,)
    deltas = pos[1:] - pos[:-1]
    ds = jnp.linalg.norm(deltas, axis=1)
    ds = jnp.concatenate([ds, ds[-1:]])  # (K,)

    # Hazard per step and survival weighting
    h = hazard_from_reach(p_reach, ds, alpha=alpha)  # (K,)
    S_prev = survival_prefix(h)  # (K,)

    expected_gain = jnp.sum(S_prev * h * areaDiffs)
    return -expected_gain


@jax.jit
def compute_spline_constraints(
    controlPoints,
    knotPoints,
):
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )

    turn_rate, velocity, evaderHeadings = (
        spline_opt_tools.get_turn_rate_velocity_and_headings(
            controlPoints, knotPoints, numSamplesPerInterval
        )
    )

    curvature = turn_rate / velocity

    return velocity, turn_rate, curvature, pos, evaderHeadings


def create_spline(knotPoints, controlPoints, order):
    spline = BSpline(knotPoints, controlPoints, order)
    return spline


def optimize_spline_path(
    p0,
    pf,
    v0,
    num_cont_points,
    spline_order,
    velocity_constraints,
    turn_rate_constraints,
    curvature_constraints,
    num_constraint_samples,
    sacraficialAgentSpeed,
    pursuerRange,
    pursuerCaptureRadius,
    pastInterseptionLocations,
    pastRadaii,
    dArea,
    integrationPoints,
    launchPdf,
    sacraficialAgentRange=5.5,
):
    # Compute Jacobian of engagement zone function

    tf = sacraficialAgentRange / sacraficialAgentSpeed

    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, tf, num_cont_points, 3
    )

    def objfunc(xDict):
        controlPoints = xDict["control_points"]
        funcs = {}
        funcs["start"] = spline_opt_tools.get_start_constraint(controlPoints)
        controlPoints = controlPoints.reshape((num_cont_points, 2))

        velocity, turn_rate, curvature, pos, evaderHeading = compute_spline_constraints(
            controlPoints,
            knotPoints,
        )

        expected_gain = area_objective_function_trajectory(
            controlPoints,
            knotPoints,
            pursuerRange,
            pursuerCaptureRadius,
            pastInterseptionLocations,
            pastRadaii,
            dArea,
            integrationPoints,
            launchPdf,
            alpha=1.0,
        )

        funcs["obj"] = expected_gain
        funcs["turn_rate"] = turn_rate
        funcs["velocity"] = velocity
        funcs["curvature"] = curvature
        funcs["pos"] = pos
        funcs["heading"] = evaderHeading
        return funcs, False

    def sens(xDict, funcs):
        funcsSens = {}
        controlPoints = jnp.array(xDict["control_points"])

        dStartDControlPointsVal = spline_opt_tools.get_start_constraint_jacobian(
            controlPoints
        )

        dVelocityDControlPointsVal = spline_opt_tools.dVelocityDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dTurnRateDControlPointsVal = spline_opt_tools.dTurnRateDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dCurvatureDControlPointsVal = spline_opt_tools.dCurvatureDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )

        funcsSens["obj"] = {
            "control_points": np.zeros((1, 2 * num_cont_points)),
        }
        funcsSens["start"] = {
            "control_points": dStartDControlPointsVal,
        }
        funcsSens["velocity"] = {
            "control_points": dVelocityDControlPointsVal,
        }
        funcsSens["turn_rate"] = {
            "control_points": dTurnRateDControlPointsVal,
        }
        funcsSens["curvature"] = {
            "control_points": dCurvatureDControlPointsVal,
        }

        return funcsSens, False

    # num_constraint_samples = numSamplesPerInterval*(num_cont_points-2)-2

    optProb = Optimization("path optimization", objfunc)

    x0 = np.linspace(p0, pf, num_cont_points).flatten()
    x0 = spline_opt_tools.move_first_control_point_so_spline_passes_through_start(
        x0, knotPoints, p0, v0
    ).flatten()

    tempVelocityContstraints = spline_opt_tools.get_spline_velocity(
        x0, 1, 3, numSamplesPerInterval
    )
    start = time.time()
    num_constraint_samples = len(tempVelocityContstraints)

    optProb.addVarGroup(
        name="control_points", nVars=2 * (num_cont_points), varType="c", value=x0
    )

    optProb.addConGroup(
        "velocity",
        num_constraint_samples,
        lower=velocity_constraints[0],
        upper=velocity_constraints[1],
        scale=1.0 / velocity_constraints[1],
    )
    optProb.addConGroup(
        "turn_rate",
        num_constraint_samples,
        lower=turn_rate_constraints[0],
        upper=turn_rate_constraints[1],
        scale=1.0 / turn_rate_constraints[1],
    )
    optProb.addConGroup(
        "curvature",
        num_constraint_samples,
        lower=curvature_constraints[0],
        upper=curvature_constraints[1],
        scale=1.0 / curvature_constraints[1],
    )
    optProb.addConGroup("start", 2, lower=p0, upper=p0)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 1000
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    # opt.options["derivative_test"] = "first-order"
    # opt.options["warm_start_init_point"] = "yes"
    # opt.options["mu_init"] = 1e-1
    # opt.options["nlp_scaling_method"] = "gradient-based"
    opt.options["tol"] = 1e-8

    sol = opt(optProb, sens="FD")
    # sol = opt(optProb, sens=sens)

    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))

    print("Optimization time:", time.time() - start)
    return create_spline(knotPoints, controlPoints, spline_order)


def plot_area_objective_function(interceptionPositions, oldRadii, fig, ax):
    x_range = [-5.0, 5.0]
    y_range = [-5.0, 5.0]
    num_pts = 50
    pursuerRange = 1.0
    pursuerCaptureRadius = 0.2

    x = jnp.linspace(x_range[0], x_range[1], num_pts)
    y = jnp.linspace(y_range[0], y_range[1], num_pts)
    X, Y = jnp.meshgrid(x, y)

    points = jnp.stack([X.flatten(), Y.flatten()], axis=-1)  # (M,2)

    dx = (x_range[1] - x_range[0]) / (num_pts - 1)
    dy = (y_range[1] - y_range[0]) / (num_pts - 1)
    dArea = dx * dy

    R_eff = pursuerRange + pursuerCaptureRadius

    # Old feasible mask / area in the same grid where you'll evaluate the objective
    old_mask = old_feasible_mask(points, interceptionPositions, oldRadii)
    oldArea = area_from_mask(old_mask, dArea)

    # Evaluate ΔM(x) = oldArea - newArea(x) at every grid point x
    # Here the "new circle" radius is R_eff (consistent with your trajectory objective)
    areaDiff = area_diff_from_oldmask(
        points,  # treat each grid point as a candidate new interception location
        R_eff,
        points,  # integration points (same grid)
        old_mask,
        oldArea,
        dArea,
    )  # (M,)

    # If you want to plot the objective you minimize (negative expected gain proxy):
    obj = -areaDiff  # (M,)

    c = ax.pcolormesh(X, Y, obj.reshape((num_pts, num_pts)))
    ax.scatter(interceptionPositions[:, 0], interceptionPositions[:, 1], color="red")

    plt.colorbar(c, ax=ax, label="-ΔArea (lower is better)")


def main():
    centers = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    radii = jnp.array([1.0, 1.0])
    x_range = [-5.0, 5.0]
    y_range = [-5.0, 5.0]
    num_pts = 5000
    x = jnp.linspace(x_range[0], x_range[1], num_pts)
    y = jnp.linspace(y_range[0], y_range[1], num_pts)
    [X, Y] = jnp.meshgrid(x, y)
    points = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    dArea = (x_range[1] - x_range[0]) / num_pts * (y_range[1] - y_range[0]) / num_pts
    area = area_of_circle_intersections(centers, radii, points, dArea)
    print("Area of intersection:", area)


def main_planner():
    x_range = [-5.0, 5.0]
    y_range = [-5.0, 5.0]
    num_pts = 50
    pursuerRange = 1.0
    pursuerCaptureRadius = 0.2
    x = jnp.linspace(x_range[0], x_range[1], num_pts)
    y = jnp.linspace(y_range[0], y_range[1], num_pts)
    [X, Y] = jnp.meshgrid(x, y)
    points = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    dArea = (x_range[1] - x_range[0]) / num_pts * (y_range[1] - y_range[0]) / num_pts

    interceptionPositions = jnp.array([[0.0, 0.0], [0.5, 0.5]])
    interceptionPositions = jnp.array([[0.0, 0.0]])
    oldRadii = jnp.array(
        [pursuerRange + pursuerCaptureRadius, pursuerRange + pursuerCaptureRadius]
    )
    oldRadii = jnp.array([pursuerRange + pursuerCaptureRadius])

    launchPdf = pez_from_interceptions.uniform_pdf_from_interception_points(
        points, interceptionPositions, pursuerRange, pursuerCaptureRadius, dArea
    )

    sacraficialLaunchPosition = np.array([-2.0, -2.0])
    initialGoal = np.array([0.0, 0.0])
    initialSacraficialVelocity = np.array([1.0, 0.0])
    pursuerCaptureRadius = 0.0
    sacraficialSpeed = 1.0
    num_cont_points = 8
    spline_order = 3
    velocity_constraints = (0.0, sacraficialSpeed)
    curvature_constraints = (-0.5, 0.5)
    turn_rate_constraints = (-1.0, 1.0)
    num_constraint_samples = 50

    spline = optimize_spline_path(
        sacraficialLaunchPosition,
        initialGoal,
        initialSacraficialVelocity,
        num_cont_points,
        spline_order,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        sacraficialSpeed,
        pursuerRange,
        pursuerCaptureRadius,
        interceptionPositions,
        oldRadii,
        dArea,
        points,
        launchPdf,
        sacraficialAgentRange=10,
    )
    fig, ax = plt.subplots()
    plot_area_objective_function(interceptionPositions, oldRadii, fig, ax)
    plot_spline(spline, ax)
    plt.show()


if __name__ == "__main__":
    main_planner()
    # plot_area_objective_function()
    # main()
