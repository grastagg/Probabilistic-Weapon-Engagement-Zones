"""
idx=-1
Sacrificial agent B-spline planner (pyOptSparse + IPOPT + JAX).
"""

from __future__ import annotations

import sys
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
from concurrent.futures import ProcessPoolExecutor, as_completed

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = (
    "0.30"  # try 0.30; adjust if you run 2–3 workers
)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")  # avoid X11 ("Invalid MIT-MAGIC-COOKIE-1")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from GEOMETRIC_BEZ import bez_from_interceptions
import GEOMETRIC_BEZ.pez_from_interceptions as pez_from_interceptions
import GEOMETRIC_BEZ.rectangle_pez as rectangle_pez
import GEOMETRIC_BEZ.rectangle_bez as rectangle_bez
from GEOMETRIC_BEZ import rectangle_bez_path_planner
from GEOMETRIC_BEZ import bez_from_interceptions_path_planner


import bspline.spline_opt_tools as spline_opt_tools


NUM_SAMPLES_PER_INTERVAL = 5
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


def circle_intersection_area(centers, radii, num_integration_points=5000, pad=0.0):
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must be (N,2)")
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("radii must be (N,) matching centers")
    if np.any(radii < 0):
        raise ValueError("radii must be nonnegative")

    # tight bounding box around all circles
    min_box = np.min(centers - radii[:, None], axis=0) - pad
    max_box = np.max(centers + radii[:, None], axis=0) + pad

    K = int(num_integration_points)
    x = np.linspace(min_box[0], max_box[0], K)
    y = np.linspace(min_box[1], max_box[1], K)
    dA = (x[1] - x[0]) * (y[1] - y[0])

    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)  # (M,2)

    dx = pts[:, None, 0] - centers[None, :, 0]
    dy = pts[:, None, 1] - centers[None, :, 1]
    inside = (dx * dx + dy * dy) <= (radii[None, :] * radii[None, :])

    return np.count_nonzero(np.all(inside, axis=1)) * dA


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
def area_objective_function_trajectory_and_intercepted(
    controlPoints,
    knotPoints,
    pursuerRange,
    pursuerCaptureRadius,
    pastInterseptionLocations,
    pastRadaii,
    dArea,
    integrationPoints,
    launchPdf,
):
    """
    Smooth expected area-gain objective:

      expected_gain = Σ_k S_prev[k] * h[k] * ΔA_soft(pos_k)

    where ΔA_soft is computed with soft intersections in launch-space.
    """
    alpha = 1.0
    tau_area = None  # softness in launch-space units
    tau_area_scale = (
        1.0  # used only if tau_area is None: tau_area = tau_area_scale * sqrt(dArea)
    )
    tau_reach_scale = 0.75  # reach softness in units of sqrt(dArea) (as you had)
    ds_floor = 1e-6
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

    # Hazard per step and survival weighting
    h = hazard_from_reach(p_reach, ds, alpha=alpha)  # (K,)
    S_prev = survival_prefix(h)  # (K,)
    intercepted = 1.0 - jnp.prod(1.0 - h + 1e-12)

    expected_gain = jnp.sum(S_prev * h * areaDiffs)
    intercepted = 1.0 - jnp.prod(1.0 - h + 1e-12)

    return -expected_gain, intercepted


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
):
    l, intercepted = area_objective_function_trajectory_and_intercepted(
        controlPoints,
        knotPoints,
        pursuerRange,
        pursuerCaptureRadius,
        pastInterseptionLocations,
        pastRadaii,
        dArea,
        integrationPoints,
        launchPdf,
    )
    return l


def intercepted(
    controlPoints,
    knotPoints,
    pursuerRange,
    pursuerCaptureRadius,
    pastInterseptionLocations,
    pastRadaii,
    dArea,
    integrationPoints,
    launchPdf,
):
    l, intercepted = area_objective_function_trajectory_and_intercepted(
        controlPoints,
        knotPoints,
        pursuerRange,
        pursuerCaptureRadius,
        pastInterseptionLocations,
        pastRadaii,
        dArea,
        integrationPoints,
        launchPdf,
    )
    return intercepted


dAreaObjectiveDControlPoints = jax.jit(jax.jacfwd(area_objective_function_trajectory))
dAreaInterceptedDControlPoints = jax.jit(jax.jacfwd(intercepted))

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


def optimize_spline_path_minimize_area(
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

        obj_val, intercepted = area_objective_function_trajectory_and_intercepted(
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
        funcs["intercepted"] = float(np.asarray(intercepted))

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
        dInterceptedDControlPointsVal = dAreaInterceptedDControlPoints(
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
        dIntercepted = np.asarray(dInterceptedDControlPointsVal)

        funcsSens["obj"] = {"control_points": dObj}
        funcsSens["start"] = {
            "control_points": dStartDControlPointsVal,
        }
        funcsSens["velocity"] = {"control_points": dVel}
        funcsSens["turn_rate"] = {"control_points": dTr}
        funcsSens["curvature"] = {"control_points": dCurv}
        funcsSens["intercepted"] = {"control_points": dIntercepted}

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
    optProb.addConGroup("intercepted", 1, lower=0.90, upper=None)

    optProb.addObj("obj")

    opt = OPT("ipopt")
    opt.options["print_level"] = 0

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

    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))

    return create_spline(knotPoints, controlPoints, spline_order)


def hazard_with_coverage(p_reach, ds, alpha=0.35):
    K = p_reach.shape[0]

    def step(carry, inputs):
        S_prev = carry
        p, ds_k = inputs

        # hazard modulated by remaining uncovered mass
        h = 1.0 - jnp.exp(-alpha * p * ds_k * S_prev)

        # update remaining uncovered
        S_new = S_prev * (1.0 - p)

        return S_new, h

    _, h = jax.lax.scan(step, 1.0, (p_reach, ds))
    return h


def _sigmoid(x):
    # stable-ish sigmoid using tanh
    return 0.5 * (jnp.tanh(0.5 * x) + 1.0)


def _make_launch_grid(min_box, max_box, nx=32, ny=32):
    """
    Deterministic uniform grid of launch points inside the rectangle.
    min_box, max_box: (2,) arrays [xmin, ymin], [xmax, ymax]
    returns: (M,2), dA
    """
    xs = jnp.linspace(min_box[0], max_box[0], nx)
    ys = jnp.linspace(min_box[1], max_box[1], ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    pts = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=1)  # (M,2)

    # cell area (approx). If nx,ny include endpoints, use (nx-1),(ny-1) spacing.
    dx = (max_box[0] - min_box[0]) / jnp.maximum(nx - 1, 1)
    dy = (max_box[1] - min_box[1]) / jnp.maximum(ny - 1, 1)
    dA = dx * dy
    return pts, dA


@jax.jit
def get_hit_objective_function(
    controlPoints,
    knotPoints,
    pursuerRange,
    pursuerCaptureRadius,
    min_box,
    max_box,
):
    """
    Drop-in replacement that *spreads* by maximizing launch-rectangle coverage:

      coverage = Area( (box ∩ ⋃_k Disk(pos_k, R_eff)) ) / Area(box)

    Approximated by a deterministic grid of launch points + smooth softmin.

    Returns: negative coverage (so minimizing objective => maximize coverage).
    """
    controlPoints = controlPoints.reshape((-1, 2))

    # (K,2) sampled trajectory positions
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )

    R_eff = pursuerRange + pursuerCaptureRadius

    # Deterministic launch grid (M,2) and cell area dA
    launch_pts, dA = _make_launch_grid(min_box, max_box, nx=32, ny=32)  # M=1024

    # Characteristic length tied to grid spacing
    cell = jnp.sqrt(dA + 1e-12)

    # Softness parameters (reasonable defaults)
    tau_d = 0.75 * cell  # softmin temperature (distance units)
    tau = 1.00 * cell  # softness around R_eff

    # dist(i,k): (M,K)
    diff = launch_pts[:, None, :] - pos[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)

    # softmin over k of dist(i,k): (M,)
    d_soft = -tau_d * jax.nn.logsumexp(-dist / (tau_d + 1e-12), axis=1)

    # soft "covered" indicator per launch point
    covered_i = _sigmoid((R_eff - d_soft) / (tau + 1e-12))  # (M,)

    # coverage fraction in [0,1] (grid-weighted)
    box_area = (max_box[0] - min_box[0]) * (max_box[1] - min_box[1])
    coverage = jnp.sum(covered_i) * dA / (box_area + 1e-12)

    return -coverage


dHitObjectiveDControlPoints = jax.jit(jax.jacfwd(get_hit_objective_function))


def optimize_spline_path_get_intercepted(
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
    min_box,
    max_box,
    dArea,
    sacraficialAgentRange=5.5,
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

        obj_val = get_hit_objective_function(
            control_points_flat,
            knotPoints,
            pursuerRange,
            pursuerCaptureRadius,
            min_box,
            max_box,
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

        dObjDControlPointsVal = dHitObjectiveDControlPoints(
            control_points_flat,
            knotPoints,
            pursuerRange,
            pursuerCaptureRadius,
            min_box,
            max_box,
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
    opt.options["print_level"] = 0

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

    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))

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


def sample_intercept(traj_xy, x_p, R, alpha, beta, D_min=0.0, rng=None):
    traj = np.asarray(traj_xy, float)
    x_p = np.asarray(x_p, float)

    if rng is None:
        rng = np.random.default_rng()

    # Sample D in [D_min, R]
    U = rng.beta(alpha, beta)
    D = D_min + (R - D_min) * U
    print("commitmet distance D =", D)

    dists = np.linalg.norm(traj - x_p, axis=1)
    print("minimum distance to pursuer along traj:", dists.min())

    inside = dists <= D
    crossings = np.where((~inside[:-1]) & inside[1:])[0]

    if len(crossings) == 0:
        return False, None, None, D

    idx = crossings[0] + 1
    return True, idx, traj[idx], D


def sample_intercept_from_spline(
    spline, truePursuerPos, pursuerRange, alpha=2.0, beta=2.0, D_min=0.0, rng=None
):
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, 1000, endpoint=True)
    pos = spline(t)
    isIntercepted, idx, interceptPoint, D = sample_intercept(
        pos, truePursuerPos, pursuerRange, alpha, beta, D_min, rng=rng
    )
    return isIntercepted, idx, interceptPoint, D


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
    interceptionPositions = jnp.array([[-0.5, 0.5], [0.5, -0.5]])
    # interceptionPositions = jnp.array([[0.0, 0.0]])
    oldRadii = jnp.array(
        [
            pursuerRange + pursuerCaptureRadius,
            pursuerRange + pursuerCaptureRadius,
            # pursuerRange + pursuerCaptureRadius,
            # pursuerRange + pursuerCaptureRadius,
        ]
    )
    # oldRadii = jnp.array([pursuerRange + pursuerCaptureRadius])

    launchPdf = pez_from_interceptions.uniform_pdf_from_interception_points(
        points, interceptionPositions, pursuerRange, pursuerCaptureRadius, dArea
    )

    expected_launch_pos, Z = expected_position_from_pdf(points, launchPdf, dArea)
    # expected_launch_pos = np.array([2.5, 2.5])

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

    spline = optimize_spline_path_minimize_area(
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
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, 1000, endpoint=True)
    pos = spline(t)
    alpha = 8
    beta = 2
    isIntercepted, idx, interceptPoint, D = sample_intercept(
        pos, expected_launch_pos, pursuerRange, alpha, beta, D_min=0.5 * pursuerRange
    )
    print("Is intercepted:", isIntercepted)
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
    ax.scatter(*interceptPoint, color="red", s=50, label="Intercept Point", marker="x")
    plt.show()


def main_planner_box():
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
    min_box = jnp.array([-2.0, -2.0])
    max_box = jnp.array([2.0, 0.0])
    # sample true pursuer position uniformly in box
    truePursuerPos = np.array(
        [
            np.random.uniform(min_box[0], max_box[0]),
            np.random.uniform(min_box[1], max_box[1]),
        ]
    )

    expected_launch_pos = np.array([0.0, 0.0])
    # expected_launch_pos = np.array([2.5, 2.5])
    # print("Expected launch position:", expected_launch_pos, "Z=", Z)

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
    v = sacraficialSpeed
    R_min = 0.2

    velocity_constraints = (0.01, v)  # don’t allow near-zero speed
    curvature_constraints = (-1.0 / R_min, 1.0 / R_min)  # ±0.167
    turn_rate_constraints = (-v / R_min, v / R_min)  # consistent with curvature

    spline = optimize_spline_path_get_intercepted(
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
        min_box,
        max_box,
        dArea,
        sacraficialAgentRange=sacraficialRange,
    )

    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, 1000, endpoint=True)
    pos = spline(t)
    isIntercepted, idx, interceptPoint, D = sample_intercept(
        pos, truePursuerPos, pursuerRange, 2.0, 2.0, D_min=0.5 * pursuerRange
    )
    print("Is intercepted:", isIntercepted)
    fig, ax = plt.subplots()
    plot_spline(spline, ax)

    ax.set_aspect("equal")

    # bez_from_interceptions.plot_potential_pursuer_reachable_region(
    #     arcs, pursuerRange, pursuerCaptureRadius, xlim=(-4, 4), ylim=(-4, 4), ax=ax
    # )
    rectangle_bez.plot_box_pursuer_reachable_region(
        min_box,
        max_box,
        pursuerRange,
        pursuerCaptureRadius,
        ax=ax,
    )
    ax.scatter(*interceptPoint, color="red", s=50, label="Intercept Point", marker="x")
    ax.scatter(
        *truePursuerPos, color="red", s=50, label="True Pursuer Position", marker="o"
    )
    plt.show()


# simulation for monte carlo runs, sample single pursuer position, send agents sequentially
def run_monte_carlo_simulation(
    randomSeed=0,
    numAgents=5,
    saveData=True,
    dataDir="GEOMETRIC_BEZ/data/test/",
    runName="test",
):
    # make data directory and directory for runName is they don't exist
    os.makedirs(dataDir, exist_ok=True)
    os.makedirs(os.path.join(dataDir, runName), exist_ok=True)
    dataDir = os.path.join(dataDir, runName) + "/"
    rng = np.random.default_rng(randomSeed)

    x_range = [-6.0, 6.0]
    y_range = [-6.0, 6.0]
    num_pts = 50
    pursuerRange = 1.0
    pursuerCaptureRadius = 0.2
    pursuerSpeed = 1.5
    x = jnp.linspace(x_range[0], x_range[1], num_pts)
    y = jnp.linspace(y_range[0], y_range[1], num_pts)
    [X, Y] = jnp.meshgrid(x, y)
    points = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    dx = (x_range[1] - x_range[0]) / (num_pts - 1)
    dy = (y_range[1] - y_range[0]) / (num_pts - 1)
    dArea = dx * dy
    min_box = jnp.array([-2.0, -2.0])
    max_box = jnp.array([2.0, 2.0])

    sacraficialLaunchPosition = np.array([-5.0, -5.0])
    sacraficialSpeed = 1.0

    highPriorityStart = sacraficialLaunchPosition.copy()
    highPriorityGoal = np.array([5.0, 5.0])
    initialHighPriorityVel = np.array([1.0, 1.0]) / np.sqrt(2)
    highPrioritySpeed = 1.0

    initialSacraficialVelocity = np.array([1.0, 1.0]) / np.sqrt(2)
    num_cont_points = 14
    spline_order = 3
    sacraficialRange = 25.0
    v = sacraficialSpeed
    R_min = 0.5

    velocity_constraints = (0.01, v)  # don’t allow near-zero speed
    curvature_constraints = (-1.0 / R_min, 1.0 / R_min)  # ±0.167
    turn_rate_constraints = (-v / R_min, v / R_min)  # consistent with curvature

    alpha = 8.0
    beta = 2.0

    truePursuerPos = np.array(
        [rng.uniform(min_box[0], max_box[0]), rng.uniform(min_box[1], max_box[1])]
    )

    interceptionPositions = []
    interceptionRadii = []

    potentialPursuerRegionAreas = []
    highPriorityPathTimes = []
    interceptionHistory = []

    potentialPursuerRegionAreas.append(
        (max_box[0] - min_box[0]) * (max_box[1] - min_box[1])
    )

    planHighPriorityPaths = True
    plot = False

    if planHighPriorityPaths:
        splineHP, tfHP = rectangle_bez_path_planner.plan_path_box_BEZ(
            min_box,
            max_box,
            pursuerRange,
            pursuerCaptureRadius,
            pursuerSpeed,
            highPriorityStart,
            highPriorityGoal,
            initialHighPriorityVel,
            highPrioritySpeed,
            num_cont_points,
            spline_order,
            velocity_constraints,
            turn_rate_constraints,
            curvature_constraints,
        )
        highPriorityPathTimes.append(tfHP)

    for agentIdx in range(numAgents):
        if len(interceptionPositions) == 0:
            launchPdf = None
            start = time.time()
            spline = optimize_spline_path_get_intercepted(
                sacraficialLaunchPosition,
                np.array([0.0, 0.0]),
                initialSacraficialVelocity,
                num_cont_points,
                spline_order,
                velocity_constraints,
                turn_rate_constraints,
                curvature_constraints,
                sacraficialSpeed,
                pursuerRange,
                pursuerCaptureRadius,
                min_box,
                max_box,
                dArea,
                sacraficialAgentRange=sacraficialRange,
            )
            print("First agent optimization time:", time.time() - start)
        else:
            launchPdf = pez_from_interceptions.uniform_pdf_from_interception_points(
                points,
                np.array(interceptionPositions),
                pursuerRange,
                pursuerCaptureRadius,
                dArea,
            )

            expected_launch_pos, Z = expected_position_from_pdf(
                points, launchPdf, dArea
            )
            start = time.time()
            spline = optimize_spline_path_minimize_area(
                sacraficialLaunchPosition,
                expected_launch_pos,
                initialSacraficialVelocity,
                num_cont_points,
                spline_order,
                velocity_constraints,
                turn_rate_constraints,
                curvature_constraints,
                sacraficialSpeed,
                pursuerRange,
                pursuerCaptureRadius,
                np.array(interceptionPositions),
                np.array(interceptionRadii),
                dArea,
                points,
                launchPdf,
                sacraficialAgentRange=sacraficialRange,
            )
            print("time for agent", agentIdx, "optimization:", time.time() - start)
        isIntercepted, idx, interceptPoint, D = sample_intercept_from_spline(
            spline,
            truePursuerPos,
            pursuerRange,
            alpha,
            beta,
            D_min=0.5 * pursuerRange,
            rng=rng,
        )
        interceptionHistory.append(isIntercepted)
        if isIntercepted:
            print(f"Agent {agentIdx} intercepted at point {interceptPoint} (D={D})")
            interceptionPositions.append(np.array(interceptPoint))
            interceptionRadii.append(pursuerRange + pursuerCaptureRadius)
        if planHighPriorityPaths:
            splineHP, arcs, tf = (
                bez_from_interceptions_path_planner.plan_path_from_interception_points(
                    interceptionPositions,
                    pursuerRange,
                    pursuerCaptureRadius,
                    pursuerSpeed,
                    highPriorityStart,
                    highPriorityGoal,
                    initialHighPriorityVel,
                    highPrioritySpeed,
                    num_cont_points,
                    spline_order,
                    velocity_constraints,
                    turn_rate_constraints,
                    curvature_constraints,
                )
            )
            highPriorityPathTimes.append(tf)

        if saveData:
            intersectionArea = circle_intersection_area(
                np.array(interceptionPositions),
                np.array(interceptionRadii),
            )
            potentialPursuerRegionAreas.append(intersectionArea)
        if plot:
            fig, ax = plt.subplots()
            t0 = spline.t[spline.k]
            tf = spline.t[-1 - spline.k]
            t = np.linspace(t0, tf, 1000, endpoint=True)
            idx = -1
            pos = spline(t)[0:idx]
            ax.plot(pos[:, 0], pos[:, 1], label=f"Sacraficial Agent {agentIdx} Path")

            t0 = splineHP.t[splineHP.k]
            tf = splineHP.t[-1 - splineHP.k]
            t = np.linspace(t0, tf, 1000, endpoint=True)

            posHP = splineHP(t)
            ax.plot(posHP[:, 0], posHP[:, 1], label="High-Priority Agent Path")

            ax.set_aspect("equal")

            if len(interceptionPositions) > 0:
                arcs = bez_from_interceptions.compute_potential_pursuer_region_from_interception_position(
                    # np.array(interceptionPositions[0:-1]),
                    np.array(interceptionPositions),
                    pursuerRange,
                    pursuerCaptureRadius,
                )

                ax.set_aspect("equal")
                bez_from_interceptions.plot_potential_pursuer_reachable_region(
                    arcs,
                    pursuerRange,
                    pursuerCaptureRadius,
                    xlim=x_range,
                    ylim=y_range,
                    ax=ax,
                )
                bez_from_interceptions.plot_circle_intersection_arcs(arcs, ax=ax)
            else:
                rectangle_bez.plot_box_pursuer_reachable_region(
                    min_box,
                    max_box,
                    pursuerRange,
                    pursuerCaptureRadius,
                    ax=ax,
                )
            ax.scatter(
                *truePursuerPos,
                color="red",
                s=50,
                label="True Pursuer Position",
                marker="o",
            )
            if isIntercepted:
                ax.scatter(
                    *interceptPoint,
                    color="blue",
                    s=50,
                    label="Intercept Point",
                    marker="x",
                )
                for i, pos in enumerate(interceptionPositions[0:-1]):
                    ax.scatter(
                        *pos,
                        color="red",
                        s=50,
                        label=f"Past Interception {i}",
                        marker="x",
                    )
            else:
                for i, pos in enumerate(interceptionPositions):
                    ax.scatter(
                        *pos,
                        color="red",
                        s=50,
                        label=f"Past Interception {i}",
                        marker="x",
                    )
            plt.show()
    if saveData:
        data = {
            "truePursuerPos": truePursuerPos,
            "interceptionPositions": np.array(interceptionPositions),
            "interceptionRadii": np.array(interceptionRadii),
            "potentialPursuerRegionAreas": np.array(potentialPursuerRegionAreas),
            "highPriorityPathTimes": np.array(highPriorityPathTimes),
            "interceptionHistory": np.array(interceptionHistory),
        }
        filename = dataDir + f"{randomSeed}.npz"
        np.savez_compressed(filename, **data)
        print(f"Saved simulation data to {filename}")


def median_iqr_plot(data, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()
    # average intersection areas over all runs
    numAgents = len(data[0]) - 1
    data = np.array(data)

    ax.plot(np.arange(numAgents + 1), np.percentile(data, 50, axis=0), marker="o")
    # fill in IQR
    first_quartile = np.percentile(data, 25, axis=0)
    third_quartile = np.percentile(data, 75, axis=0)
    ax.fill_between(np.arange(numAgents + 1), first_quartile, third_quartile, alpha=0.2)
    ax.set_xticks(np.arange(numAgents + 1))


def parse_data(dataDir):
    allIntersectionAreas = []
    allPathTimes = []
    allInterceptionHistories = []
    for filename in os.listdir(dataDir):
        if filename.endswith(".npz"):
            data = np.load(os.path.join(dataDir, filename))
            # process data as needed
            print(f"Loaded data from {filename}:")
            print("True Pursuer Position:", data["truePursuerPos"])
            print("Interception Positions:", data["interceptionPositions"])
            print(
                "Potential Pursuer Region Areas:", data["potentialPursuerRegionAreas"]
            )
            print("highPriorityPathTimes:", data["highPriorityPathTimes"])
            print("interceptionHistory:", data["interceptionHistory"])
            intersectionAreas = data["potentialPursuerRegionAreas"]
            pathTimes = data["highPriorityPathTimes"]
            allIntersectionAreas.append(intersectionAreas)
            allPathTimes.append(pathTimes)
            allInterceptionHistories.append(data["interceptionHistory"])
    allIntersectionAreas = np.array(allIntersectionAreas)
    # count true per agent interceptions
    print(
        "Interception statistics per agent:",
        np.sum(np.array(allInterceptionHistories), axis=0),
    )
    # median_iqr_plot(allIntersectionAreas)
    # median_iqr_plot(allPathTimes)

    plt.show()


if __name__ == "__main__":
    # main()
    # first argument is random seed from command line
    if len(sys.argv) != 2:
        parse_data(dataDir="GEOMETRIC_BEZ/data/test/")
    else:
        seed = int(sys.argv[1])
        print("running monte carlo simulation with seed", seed)
        numAgents = 3
        runName = "test"
        run_monte_carlo_simulation(
            seed, 3, saveData=True, dataDir="GEOMETRIC_BEZ/data/", runName=runName
        )
    # main_planner()
    # main_planner_box()
    # plot_area_objective_function()
