"""
Sacrificial agent B-spline planner (pyOptSparse + IPOPT + JAX).
"""

from __future__ import annotations

import getpass
import os
import time

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyoptsparse import OPT, Optimization
from scipy.interpolate import BSpline


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from GEOMETRIC_BEZ import bez_from_interceptions
import GEOMETRIC_BEZ.pez_from_interceptions as pez_from_interceptions


import bspline.spline_opt_tools as spline_opt_tools


NUM_SAMPLES_PER_INTERVAL = 15
numSamplesPerInterval = NUM_SAMPLES_PER_INTERVAL


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
    ax.plot(x, y, linewidth=width)

    ax.set_aspect(1)


def _survival_prefix(h):
    """
    h: (K,) with values in [0,1)
    returns S_prev: (K,) where S_prev[k] = Π_{j<k} (1 - h[j])
    """
    one_minus = 1.0 - h
    cp = jnp.cumprod(one_minus + 1e-12)
    return jnp.concatenate([jnp.array([1.0], dtype=cp.dtype), cp[:-1]])


@jax.jit
def dist_objective_function_trajectory(
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
    *,
    # smoothing / conditioning knobs
    tau_reach_scale=0.75,  # reach softness in units of sqrt(dArea)
    tau_dist=0.25,  # distance softmin length scale (same units as pos)
    ds_floor=1e-6,
    normalize_ds=True,
    # "shotdown process" knobs
    hazard_gain=8.0,  # sharpness of converting p_reach->hazard (bigger = more "eventy")
    p_power=2.0,  # emphasize high p_reach in hazard (>=1)
    # "informativeness" knobs
    d_star=1.0,  # distance threshold: below this is "low info"
    d_tau=0.25,  # softness of the threshold
    # optional: reward shaping
    use_log_reward=False,  # use log(1+d) instead of d to reduce incentive for extreme d
):
    """
    Event-based objective:
      - first exposure should be maximally informative
      - subsequent exposures should also be informative
      - low-info exposures are discouraged because they terminate survival early with little utility

    Expected utility:
      E = Σ_k S_prev[k] * h[k] * U(d_k)

    where:
      h[k] = 1 - exp(-hazard_gain * (p_reach[k]^p_power) * ds_k)
      U(d) ≈ 0 when d < d_star, increases when d > d_star (smooth gate)
    """
    controlPoints = controlPoints.reshape((-1, 2))
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )  # (K,2)

    R_eff = pursuerRange + pursuerCaptureRadius

    # ---- smooth reach ----
    tau_reach = tau_reach_scale * jnp.sqrt(dArea)
    p_reach = pez_from_interceptions.prob_reach_numerical_soft(
        pos, integrationPoints, launchPdf, R_eff, dArea, tau_reach
    )  # (K,)

    # ---- smooth distance-to-past interceptions ----
    active = pastRadaii > 0.0
    any_active = jnp.any(active)

    diff = pos[:, None, :] - pastInterseptionLocations[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)  # (K,N)
    dists = jnp.where(active[None, :], dists, jnp.inf)

    d_softmin = -tau_dist * jax.scipy.special.logsumexp(
        -dists / tau_dist, axis=1
    )  # (K,)
    d_softmin = jnp.where(any_active, d_softmin, 0.0)

    # ---- arc-length weights ----
    deltas = pos[1:] - pos[:-1]
    ds = jnp.linalg.norm(deltas, axis=1)
    ds = jnp.maximum(ds, ds_floor)
    ds = jnp.concatenate([ds, ds[-1:]])  # (K,)

    if normalize_ds:
        ds = ds / (jnp.sum(ds) + 1e-12)

    # ---- hazard: probability of being shot down "at this step given survival so far" ----
    # Make hazard concentrate where p_reach is truly high.
    p_eff = jnp.clip(p_reach, 0.0, 1.0) ** p_power
    h = 1.0 - jnp.exp(-hazard_gain * p_eff * ds)  # (K,) in [0,1)

    # ---- informativeness utility U(d): ~0 below d_star, grows above ----
    gate = jax.nn.sigmoid((d_softmin - d_star) / d_tau)  # ~0 low-info, ~1 high-info
    if use_log_reward:
        U = gate * jnp.log1p(d_softmin)
    else:
        U = gate * d_softmin

    # ---- expected utility of the first (and only) exposure event ----
    S_prev = _survival_prefix(h)  # (K,)
    w = S_prev * h  # (K,) first-hit distribution
    expected_utility = jnp.sum(w * U)

    return -expected_utility


dDistaObjectiveDControlPoints = jax.jit(jax.jacfwd(dist_objective_function_trajectory))


# ---------------------------
# Smooth disk membership
# ---------------------------
@jax.jit
def _soft_inside_disk(dist, radius, tau, eps=1e-12):
    """
    Smooth approximation to 1{dist <= radius} using a sigmoid on signed distance.

    dist  : (...,) >= 0
    radius: scalar or (...,)
    tau   : softness length scale (same units as dist). Larger = softer boundary.
    """
    # signed distance: positive inside, negative outside
    s = (radius - dist) / (tau + eps)
    return jax.nn.sigmoid(s)


# ---------------------------
# Smooth feasible mask for intersection of past disks
# ---------------------------
@jax.jit
def smooth_feasible_mask(points, past_centers, past_radii, tau_area, eps=1e-12):
    """
    points      : (M,2) integration grid over launch-space
    past_centers: (N,2) past kill locations (centers of constraint disks in launch-space)
    past_radii  : (N,)  corresponding radii (<=0 means inactive)
    tau_area    : scalar softness for disk boundary (launch-space units)

    returns:
        mask_soft: (M,) in (0,1], approx 1 inside ALL disks, ~0 outside
    """
    # (M,N,2) -> (M,N)
    diff = points[:, None, :] - past_centers[None, :, :]
    dsq = jnp.sum(diff * diff, axis=-1)  # (M,N)
    dist = jnp.sqrt(dsq + 1e-12)

    active = past_radii > 0.0  # (N,)
    # Soft membership per disk: (M,N)
    phi = _soft_inside_disk(dist, past_radii[None, :], tau_area, eps=eps)
    # Inactive disks should contribute multiplicative identity = 1
    phi = jnp.where(active[None, :], phi, 1.0)

    # Intersection via product, computed stably in log-space:
    log_phi = jnp.log(phi + eps)  # (M,N)
    log_mask = jnp.sum(log_phi, axis=1)  # (M,)
    mask_soft = jnp.exp(log_mask)  # (M,)

    return mask_soft


@jax.jit
def area_from_soft_mask(mask_soft, dArea):
    """
    mask_soft: (M,) in [0,1]
    """
    return jnp.sum(mask_soft) * dArea


# ---------------------------
# Smooth area difference for adding a new disk
# ---------------------------
@jax.jit
def area_diff_single_from_oldmask_smooth(
    newInterseptionLocation,
    radius,
    integrationPoints,
    old_mask_soft,
    oldArea,
    dArea,
    tau_area,
    eps=1e-12,
):
    """
    Smooth version of:
      areaDiff = oldArea - newArea
    where:
      oldArea = sum(old_mask_soft) * dArea
      newArea = sum(old_mask_soft * soft_inside_new_disk) * dArea

    newInterseptionLocation: (2,)
    radius                : scalar
    integrationPoints     : (M,2)
    old_mask_soft         : (M,) in [0,1]
    oldArea               : scalar
    """
    diff = integrationPoints - newInterseptionLocation[None, :]  # (M,2)
    dsq_new = jnp.sum(diff * diff, axis=-1)  # (M,)
    dist_new = jnp.sqrt(dsq_new + 1e-12)

    phi_new = _soft_inside_disk(dist_new, radius, tau_area, eps=eps)  # (M,)
    new_mask_soft = old_mask_soft * phi_new
    newArea = jnp.sum(new_mask_soft) * dArea
    return oldArea - newArea


# Vectorized over a trajectory of candidate kill locations: pos is (K,2)
area_diff_from_oldmask_smooth = jax.jit(
    jax.vmap(
        area_diff_single_from_oldmask_smooth,
        in_axes=(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
        ),  # pos[k], radius, pts, old_mask, oldArea, dArea, tau_area
    )
)


# ---------------------------
# Hazard + survival (already smooth, kept as-is)
# ---------------------------
@jax.jit
def hazard_from_reach(p_reach, ds, alpha=0.35):
    """
    Smooth per-step hazard via exponential survival model.
    p_reach: (K,) in [0,1]
    ds     : (K,) >= 0
    """
    return 1.0 - jnp.exp(-alpha * jnp.clip(p_reach, 0.0, 1.0) * ds)


@jax.jit
def survival_prefix(h, eps=1e-12):
    """
    h: (K,) in [0,1)
    returns S_prev: (K,) where S_prev[k] = Π_{j<k} (1 - h[j])
    Uses cumprod; if K is large, a log-space version is also possible.
    """
    one_minus = 1.0 - h
    cp = jnp.cumprod(one_minus + eps)
    return jnp.concatenate([jnp.ones((1,), dtype=cp.dtype), cp[:-1]])


# ---------------------------
# Smooth area objective along a trajectory
# ---------------------------
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
    *,
    tau_area=None,  # softness in launch-space units
    tau_area_scale=1.0,  # used only if tau_area is None: tau_area = tau_area_scale * sqrt(dArea)
    tau_reach_scale=0.75,  # reach softness in units of sqrt(dArea) (as you had)
    ds_floor=1e-6,
    normalize_ds=False,
):
    """
    Smooth expected area-gain objective:

      expected_gain = Σ_k S_prev[k] * h[k] * ΔA_soft(pos_k)

    where ΔA_soft is computed with soft intersections in launch-space.
    """
    controlPoints = controlPoints.reshape((-1, 2))

    # (K,2) sampled trajectory positions (candidate kill points)
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )

    R_eff = pursuerRange + pursuerCaptureRadius

    # Choose tau_area: tie to grid resolution if you do not want "tuning"
    # If integrationPoints is a uniform grid, sqrt(dArea) is the cell side length.
    if tau_area is None:
        tau_area = tau_area_scale * jnp.sqrt(dArea)

    # Precompute old feasible launch-region soft mask and soft area
    old_mask_soft = smooth_feasible_mask(
        integrationPoints, pastInterseptionLocations, pastRadaii, tau_area
    )
    oldArea = area_from_soft_mask(old_mask_soft, dArea)

    # ΔA_k = oldArea - newArea_soft(pos_k)
    areaDiffs = area_diff_from_oldmask_smooth(
        pos, R_eff, integrationPoints, old_mask_soft, oldArea, dArea, tau_area
    )  # (K,)

    # Smooth reach probability at each trajectory sample: (K,)
    tau_reach = tau_reach_scale * jnp.sqrt(dArea)
    p_reach = pez_from_interceptions.prob_reach_numerical_soft(
        pos, integrationPoints, launchPdf, R_eff, dArea, tau_reach
    )

    # Step length along path (K-1,), pad to (K,)
    deltas = pos[1:] - pos[:-1]
    ds = jnp.linalg.norm(deltas, axis=1)
    ds = jnp.maximum(ds, ds_floor)
    ds = jnp.concatenate([ds, ds[-1:]])  # (K,)

    if normalize_ds:
        ds = ds / (jnp.sum(ds) + 1e-12)

    # Hazard per step and survival weighting
    h = hazard_from_reach(p_reach, ds, alpha=alpha)  # (K,)
    S_prev = survival_prefix(h)  # (K,)

    expected_gain = jnp.sum(S_prev * h * areaDiffs)
    return -expected_gain


dAreaObjectiveDControlPoints = jax.jit(jax.jacfwd(area_objective_function_trajectory))

# objfunction_trajectory = dist_objective_function_trajectory
# objfunction_trajectory_jacobian = dDistaObjectiveDControlPoints
objfunction_trajectory = area_objective_function_trajectory
objfunction_trajectory_jacobian = dAreaObjectiveDControlPoints


@jax.jit
def compute_spline_constraints(
    controlPoints,
    knotPoints,
):
    """
    Evaluate constraint sample arrays along the spline.

    Returns:
        velocity:  (K,)
        turn_rate: (K,)
        curvature: (K,)  computed as turn_rate / max(velocity, 1e-8)
        pos:       (K,2)
        heading:   (K,)
    """
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )

    turn_rate, velocity, evaderHeadings = (
        spline_opt_tools.get_turn_rate_velocity_and_headings(
            controlPoints, knotPoints, numSamplesPerInterval
        )
    )

    curvature = turn_rate / jnp.maximum(velocity, 1e-8)

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
    sacraficialAgentSpeed,
    pursuerRange,
    pursuerCaptureRadius,
    pastInterseptionLocations,
    pastRadaii,
    dArea,
    integrationPoints,
    launchPdf,
    sacraficialAgentRange=5.5,
    *,
    use_smooth_objective=False,
    smooth_eps=None,
    alpha=1.0,
    gain_weight=0.1,
    smoothness_weight=1e-2,
    debug=None,
):
    """
    Optimize B-spline control points subject to kinematic constraints.

    Returns a SciPy `BSpline` using the optimized control points.
    """
    tf = sacraficialAgentRange / sacraficialAgentSpeed

    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, tf, num_cont_points, 3
    )

    if velocity_constraints[0] <= 0.0:
        v_floor = max(1e-3, 0.05 * float(velocity_constraints[1]))
        velocity_constraints = (v_floor, float(velocity_constraints[1]))

    eval_counter = {"n": 0}
    debug_every = int(os.environ.get("SACRIFICIAL_DEBUG_EVERY", "20"))

    def objfunc(xDict):
        control_points_flat = xDict["control_points"]
        funcs = {}
        funcs["start"] = spline_opt_tools.get_start_constraint(control_points_flat)
        control_points = control_points_flat.reshape((num_cont_points, 2))

        velocity, turn_rate, curvature, _, _ = compute_spline_constraints(
            control_points,
            knotPoints,
        )

        obj_val = objfunction_trajectory(
            control_points_flat,
            knotPoints,
            pursuerRange,
            pursuerCaptureRadius,
            pastInterseptionLocations,
            pastRadaii,
            dArea,
            integrationPoints,
            launchPdf,
        )

        funcs["obj"] = float(np.asarray(obj_val))
        funcs["turn_rate"] = np.asarray(turn_rate)
        funcs["velocity"] = np.asarray(velocity)
        funcs["curvature"] = np.asarray(curvature)

        if debug:
            eval_counter["n"] += 1
            if eval_counter["n"] % max(1, debug_every) == 0:
                v = funcs["velocity"]
                tr = funcs["turn_rate"]
                k = funcs["curvature"]
                print(
                    f"[eval {eval_counter['n']}] obj={funcs['obj']:.6e} "
                    f"v[min,max]=({v.min():.3e},{v.max():.3e}) "
                    f"tr[min,max]=({tr.min():.3e},{tr.max():.3e}) "
                    f"k[min,max]=({k.min():.3e},{k.max():.3e})"
                )

        fail = (
            (not np.isfinite(funcs["obj"]))
            or (not np.all(np.isfinite(funcs["turn_rate"])))
            or (not np.all(np.isfinite(funcs["velocity"])))
            or (not np.all(np.isfinite(funcs["curvature"])))
        )
        return funcs, fail

    def sens(xDict, funcs):
        funcsSens = {}
        control_points_flat = jnp.array(xDict["control_points"])

        dStartDControlPointsVal = spline_opt_tools.get_start_constraint_jacobian(
            control_points_flat
        )

        dVelocityDControlPointsVal = spline_opt_tools.dVelocityDControlPoints(
            control_points_flat, tf, 3, numSamplesPerInterval
        )
        dTurnRateDControlPointsVal = spline_opt_tools.dTurnRateDControlPoints(
            control_points_flat, tf, 3, numSamplesPerInterval
        )
        dCurvatureDControlPointsVal = spline_opt_tools.dCurvatureDControlPoints(
            control_points_flat, tf, 3, numSamplesPerInterval
        )

        dObjDControlPointsVal = objfunction_trajectory_jacobian(
            control_points_flat,
            knotPoints,
            pursuerRange,
            pursuerCaptureRadius,
            pastInterseptionLocations,
            pastRadaii,
            dArea,
            integrationPoints,
            launchPdf,
        )

        dObj = np.asarray(dObjDControlPointsVal)
        dVel = np.asarray(dVelocityDControlPointsVal)
        dTr = np.asarray(dTurnRateDControlPointsVal)
        dCurv = np.asarray(dCurvatureDControlPointsVal)

        funcsSens["obj"] = {"control_points": dObj}
        funcsSens["start"] = {
            "control_points": dStartDControlPointsVal,
        }
        funcsSens["velocity"] = {"control_points": dVel}
        funcsSens["turn_rate"] = {"control_points": dTr}
        funcsSens["curvature"] = {"control_points": dCurv}

        fail = (
            (not np.all(np.isfinite(dObj)))
            or (not np.all(np.isfinite(dVel)))
            or (not np.all(np.isfinite(dTr)))
            or (not np.all(np.isfinite(dCurv)))
        )
        return funcsSens, fail

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
    #
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
    opt.options["derivative_test"] = "first-order"
    # opt.options["hessian_approximation"] = "limited-memory"
    # opt.options["nlp_scaling_method"] = "gradient-based"
    # opt.options["mu_strategy"] = "adaptive"
    username = getpass.getuser()
    hsllib = "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    opt.options["hsllib"] = hsllib
    opt.options["linear_solver"] = "ma97"
    opt.options["tol"] = 1e-8

    # sol = opt(optProb, sens="FD")
    sol = opt(optProb, sens=sens)
    print(sol)

    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))

    if debug:
        print("Optimization time:", time.time() - start)
    return create_spline(knotPoints, controlPoints, spline_order)


@jax.jit
def expected_position_from_pdf(integrationPoints, launchPdf, dArea, eps=1e-12):
    """
    integrationPoints: (M,2)
    launchPdf: (M,)    values of a PDF at integrationPoints
    dArea: scalar      area per grid cell
    Returns:
        mu: (2,) expected position
        Z:  scalar normalization check (≈1 if already normalized)
    """
    Z = jnp.sum(launchPdf) * dArea
    w = launchPdf / (
        Z + eps
    )  # normalized weights per point (still need dArea in expectation)
    mu = jnp.sum(integrationPoints * w[:, None], axis=0) * dArea
    return mu, Z


def plot_area_reduction_field(
    interceptionPositions,
    pursuerRange,
    pursuerCaptureRadius,
    oldRadii,
    fig,
    ax,
    *,
    x_range=(-5.0, 5.0),
    y_range=(-5.0, 5.0),
    num_pts=150,
    tau_area=None,  # if None: tau_area = tau_area_scale * sqrt(dArea)
    tau_area_scale=1.0,
    cmap=None,  # e.g. "viridis" if you want, otherwise matplotlib default
    show_old_mask_contour=True,
    contour_level=0.5,
):
    """
    Plots ΔA(c) over candidate centers c in the same 2D coordinate system as interceptionPositions.

    This visualizes ONLY the geometric area-reduction term:
      ΔA(c) = area(old feasible set) - area(old feasible set ∩ disk(center=c, radius=R_eff))

    Notes:
    - This uses a smooth disk membership with softness tau_area.
    - oldRadii <= 0 are treated as inactive (ignored).
    """
    R_eff = pursuerRange + pursuerCaptureRadius

    x = jnp.linspace(x_range[0], x_range[1], num_pts)
    y = jnp.linspace(y_range[0], y_range[1], num_pts)
    X, Y = jnp.meshgrid(x, y)
    points = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)  # (M,2)

    dx = (x_range[1] - x_range[0]) / (num_pts - 1)
    dy = (y_range[1] - y_range[0]) / (num_pts - 1)
    dArea = dx * dy

    if tau_area is None:
        tau_area = tau_area_scale * jnp.sqrt(dArea)

    # Soft old feasible set (intersection of prior disks)
    old_mask_soft = smooth_feasible_mask(
        points, interceptionPositions, oldRadii, tau_area
    )
    oldArea = jnp.sum(old_mask_soft) * dArea

    # For each candidate center c (= each grid point), compute ΔA(c)
    areaDiffs = area_diff_from_oldmask_smooth(
        points, R_eff, points, old_mask_soft, oldArea, dArea, tau_area
    )  # (M,)

    field = areaDiffs.reshape((num_pts, num_pts))

    ax.set_aspect("equal")

    # Plot: by default pcolormesh likes numpy arrays
    c = ax.pcolormesh(
        jnp.asarray(X),
        jnp.asarray(Y),
        jnp.asarray(field),
        shading="auto",
        cmap=cmap,
    )

    # Past interceptions
    ax.scatter(
        jnp.asarray(interceptionPositions[:, 0]),
        jnp.asarray(interceptionPositions[:, 1]),
        color="red",
        s=25,
        label="Past interceptions",
    )

    # Optional: show the current feasible set as a contour of the soft mask
    if show_old_mask_contour:
        old_field = old_mask_soft.reshape((num_pts, num_pts))
        ax.contour(
            jnp.asarray(X),
            jnp.asarray(Y),
            jnp.asarray(old_field),
            levels=[contour_level],
            linewidths=1.5,
        )

    cb = fig.colorbar(c, ax=ax)
    cb.set_label(
        r"$\Delta A(c)$ (expected area reduction if next disk centered at $c$)"
    )

    ax.set_title("Smooth area-reduction field (geometry only)")
    ax.legend(loc="upper right")


def plot_dist_objective_function(
    interceptionPositions, pursuerRange, pursuerCaptureRadius, oldRadii, fig, ax
):
    x_range = [-5.0, 5.0]
    y_range = [-5.0, 5.0]
    num_pts = 100

    R_eff = pursuerRange + pursuerCaptureRadius

    x = jnp.linspace(x_range[0], x_range[1], num_pts)
    y = jnp.linspace(y_range[0], y_range[1], num_pts)
    X, Y = jnp.meshgrid(x, y)

    # Use the same grid as both eval_points and integration_points
    points = jnp.stack([X.flatten(), Y.flatten()], axis=-1)  # (M,2)

    dx = (x_range[1] - x_range[0]) / (num_pts - 1)
    dy = (y_range[1] - y_range[0]) / (num_pts - 1)
    dArea = dx * dy

    launchPdf = pez_from_interceptions.uniform_pdf_from_interception_points(
        points, interceptionPositions, pursuerRange, pursuerCaptureRadius, dArea
    )

    R_eff = pursuerRange + pursuerCaptureRadius  # kept for p_reach calc

    # Reach probability at each trajectory sample: (K,)
    p_reach = pez_from_interceptions.prob_reach_numerical(
        points, points, launchPdf, R_eff, dArea
    )

    # --- Distance-to-past-interception-centers term ---
    # Treat entries with pastRadaii <= 0 as inactive (so you can keep fixed-size arrays).

    # Pairwise distances: (K,N)
    diff = points[:, None, :] - interceptionPositions[None, :, :]
    dists = jnp.linalg.norm(diff, axis=-1)

    d_min = jnp.min(dists, axis=1)  # (K,)

    # Expected "distance gain" weighted by p_reach
    obj = -p_reach * d_min

    ax.set_aspect("equal")

    c = ax.pcolormesh(X, Y, obj.reshape((num_pts, num_pts)))
    ax.scatter(interceptionPositions[:, 0], interceptionPositions[:, 1], color="red")

    plt.colorbar(c, ax=ax)


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
    dx = (x_range[1] - x_range[0]) / (num_pts - 1)
    dy = (y_range[1] - y_range[0]) / (num_pts - 1)
    dArea = dx * dy

    interceptionPositions = jnp.array([[0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    # interceptionPositions = jnp.array([[-0.5, 0.5], [0.5, -0.5]])
    # interceptionPositions = jnp.array([[0.0, 0.0]])
    oldRadii = jnp.array(
        [
            pursuerRange + pursuerCaptureRadius,
            pursuerRange + pursuerCaptureRadius,
            pursuerRange + pursuerCaptureRadius,
            # pursuerRange + pursuerCaptureRadius,
        ]
    )
    # oldRadii = jnp.array([pursuerRange + pursuerCaptureRadius])

    launchPdf = pez_from_interceptions.uniform_pdf_from_interception_points(
        points, interceptionPositions, pursuerRange, pursuerCaptureRadius, dArea
    )

    expected_launch_pos, Z = expected_position_from_pdf(points, launchPdf, dArea)
    # expected_launch_pos = np.array([2.5, 2.5])
    print("Expected launch position:", expected_launch_pos, "Z=", Z)

    sacraficialRange = 10.0

    sacraficialLaunchPosition = np.array([-4.0, -4.0])
    initialGoal = expected_launch_pos
    direction = initialGoal - sacraficialLaunchPosition
    sacraficialSpeed = 1.0
    initialSacraficialVelocity = (
        direction / np.linalg.norm(direction)
    ) * sacraficialSpeed

    # initialGoal = (
    # direction / np.linalg.norm(direction) * sacraficialRange * 0.5
    #     + sacraficialLaunchPosition
    # )

    pursuerCaptureRadius = 0.0
    num_cont_points = 14
    spline_order = 3
    # velocity_constraints = (0.0, sacraficialSpeed)
    # curvature_constraints = (-0.5, 0.5)
    # turn_rate_constraints = (-1.0, 1.0)
    v = sacraficialSpeed
    R_min = 0.2

    velocity_constraints = (0.01, v)  # don’t allow near-zero speed
    curvature_constraints = (-1.0 / R_min, 1.0 / R_min)  # ±0.167
    turn_rate_constraints = (-v / R_min, v / R_min)  # consistent with curvature

    spline = optimize_spline_path(
        sacraficialLaunchPosition,
        initialGoal,
        initialSacraficialVelocity,
        num_cont_points,
        spline_order,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        sacraficialSpeed,
        pursuerRange,
        pursuerCaptureRadius,
        interceptionPositions,
        oldRadii,
        dArea,
        points,
        launchPdf,
        sacraficialAgentRange=sacraficialRange,
    )
    fig, ax = plt.subplots()
    # plot_area_objective_function(
    #     interceptionPositions, pursuerRange, pursuerCaptureRadius, oldRadii, fig, ax
    # )
    plot_area_reduction_field(
        interceptionPositions, pursuerRange, pursuerCaptureRadius, oldRadii, fig, ax
    )
    plot_spline(spline, ax)
    arcs = bez_from_interceptions.compute_potential_pursuer_region_from_interception_position(
        interceptionPositions,
        pursuerRange,
        pursuerCaptureRadius,
    )

    ax.set_aspect("equal")
    bez_from_interceptions.plot_potential_pursuer_reachable_region(
        arcs, pursuerRange, pursuerCaptureRadius, xlim=(-4, 4), ylim=(-4, 4), ax=ax
    )
    plt.show()


if __name__ == "__main__":
    main_planner()
    # plot_area_objective_function()
