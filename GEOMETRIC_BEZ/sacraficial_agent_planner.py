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


def _sigmoid(z):
    return 0.5 * (jnp.tanh(0.5 * z) + 1.0)


def _soft_inside_disc_weights(points, centers, radii, eps):
    """
    Soft intersection of discs:
      w(p) = Π_i sigmoid( (r_i^2 - ||p-c_i||^2) / (eps * r_i^2) )
    """
    # points: (M,2), centers: (N,2), radii: (N,)
    # (M,N,2) -> (M,N)
    d2 = jnp.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    r2 = radii[None, :] ** 2
    # normalize eps by r^2 so eps is scale-invariant-ish
    s = (r2 - d2) / (eps * (r2 + 1e-12))
    w = jnp.prod(_sigmoid(s), axis=1)
    return w  # (M,)


def _make_soft_objective(
    knotPoints,
    num_cont_points,
    pursuerRange,
    pursuerCaptureRadius,
    pastInterseptionLocations,
    pastRadaii,
    dArea,
    integrationPoints,
    launchPdf,
    *,
    alpha=1.0,
    eps=0.05,
    ds_floor=1e-6,
    gain_weight=1.0,
    smoothness_weight=1e-3,
    use_hazard_model=True,  # <-- choose your interpretation
    debug=False,
):
    knotPoints = jnp.asarray(knotPoints)
    integrationPoints = jnp.asarray(integrationPoints)
    launchPdf = jnp.asarray(launchPdf)
    pastInterseptionLocations = jnp.asarray(pastInterseptionLocations)
    pastRadaii = jnp.asarray(pastRadaii)
    dArea = jnp.asarray(dArea)

    R_eff = jnp.asarray(pursuerRange + pursuerCaptureRadius)
    R2 = R_eff**2

    # Precompute old feasible weights + old area once
    old_w = _soft_inside_disc_weights(
        integrationPoints, pastInterseptionLocations, pastRadaii, eps
    )  # (M,)
    oldArea = jnp.sum(old_w) * dArea

    # Optional: ensure launchPdf is normalized (harmless if already normalized)
    Z = jnp.sum(launchPdf) * dArea
    launchPdfN = jnp.where(Z > 0, launchPdf / Z, launchPdf)

    def obj(controlPointsFlat):
        controlPoints = controlPointsFlat.reshape((num_cont_points, 2))

        pos = spline_opt_tools.evaluate_spline(
            controlPoints, knotPoints, numSamplesPerInterval
        )  # (K,2)

        # --- ΔA at each pos[k] ---
        def new_area_at_position(p):
            d2 = jnp.sum((integrationPoints - p[None, :]) ** 2, axis=1)  # (M,)
            w_new = _sigmoid((R2 - d2) / (eps * (R2 + 1e-12)))
            return jnp.sum(old_w * w_new) * dArea

        newAreas = jax.vmap(new_area_at_position)(pos)  # (K,)
        areaDiffs = oldArea - newAreas  # (K,)

        # --- p_reach(pos[k]) ---
        def p_reach_at_position(p):
            d2 = jnp.sum((integrationPoints - p[None, :]) ** 2, axis=1)
            w = _sigmoid((R2 - d2) / (eps * (R2 + 1e-12)))
            return jnp.sum(w * launchPdfN) * dArea

        p_reach = jax.vmap(p_reach_at_position)(pos)  # (K,)

        # --- path steps ds ---
        K = pos.shape[0]
        deltas = pos[1:] - pos[:-1]  # (K-1,2)
        ds = jnp.linalg.norm(deltas, axis=1)  # (K-1,)
        ds = jnp.where(K > 1, ds, jnp.array([ds_floor]))
        ds = jnp.concatenate([ds, ds[-1:]])  # (K,)
        ds = jnp.maximum(ds, ds_floor)

        h = hazard_from_reach(p_reach, ds, alpha=alpha)  # (K,)
        S_prev = survival_prefix(h)  # (K,)
        expected_gain = jnp.sum(S_prev * h * areaDiffs)

        # --- smoothness regularizer (2nd difference of control points) ---
        dd = controlPoints[2:] - 2.0 * controlPoints[1:-1] + controlPoints[:-2]
        smoothness = jnp.sum(dd * dd)

        val = gain_weight * (-expected_gain) + smoothness_weight * smoothness

        # NaN/Inf guard
        finite = jnp.isfinite(val)
        val = jax.lax.cond(
            finite, lambda _: val, lambda _: 1e9 + smoothness, operand=None
        )

        # Debug printing without Python branching during tracing
        def _dbg(_):
            jax.debug.print(
                "J={J} gain={G} smooth={S} oldArea={A} "
                "ΔA[min,max]=({dmin},{dmax}) p[min,max]=({pmin},{pmax}) ds[min,max]=({dsmin},{dsmax})",
                J=val,
                G=expected_gain,
                S=smoothness,
                A=oldArea,
                dmin=jnp.min(areaDiffs),
                dmax=jnp.max(areaDiffs),
                pmin=jnp.min(p_reach),
                pmax=jnp.max(p_reach),
                dsmin=jnp.min(ds),
                dsmax=jnp.max(ds),
            )
            return 0

        _ = jax.lax.cond(jnp.array(debug), _dbg, lambda _: 0, operand=None)

        return val

    return jax.jit(obj)


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
def hazard_from_reach(p_reach, ds, alpha=0.35):
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
    # deltas = pos[1:] - pos[:-1]
    # ds = jnp.linalg.norm(deltas, axis=1)
    # ds = jnp.concatenate([ds, ds[-1:]])  # (K,)
    # # Hazard per step and survival weighting
    # h = hazard_from_reach(p_reach, ds, alpha=alpha)  # (K,)
    # S_prev = survival_prefix(h)  # (K,)
    #
    # expected_gain = jnp.sum(S_prev * h * areaDiffs)
    expected_gain = jnp.sum(p_reach * areaDiffs)

    #     jax.debug.print(
    #         """
    # --- AREA OBJECTIVE DEBUG ---
    # oldArea        = {oldArea}
    # ΔM min / max   = {dmin} / {dmax}
    # p_reach min/max= {prmin} / {prmax}
    # hazard min/max = {hmin} / {hmax}
    # S_prev(end)    = {Send}
    # expected_gain  = {J}
    # """,
    #         oldArea=oldArea,
    #         dmin=jnp.min(areaDiffs),
    #         dmax=jnp.max(areaDiffs),
    #         prmin=jnp.min(p_reach),
    #         prmax=jnp.max(p_reach),
    #         hmin=jnp.min(h),
    #         hmax=jnp.max(h),
    #         Send=S_prev[-1],
    #         J=expected_gain,
    #     )
    return -expected_gain


dAreaObjectiveDControlPoints = jax.jit(jax.jacfwd(area_objective_function_trajectory))


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

    if debug is None:
        debug = os.environ.get("SACRIFICIAL_DEBUG", "0") not in (
            "0",
            "",
            "false",
            "False",
        )

    tf = sacraficialAgentRange / sacraficialAgentSpeed

    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, tf, num_cont_points, 3
    )

    if velocity_constraints[0] <= 0.0:
        v_floor = max(1e-3, 0.05 * float(velocity_constraints[1]))
        velocity_constraints = (v_floor, float(velocity_constraints[1]))

    if smooth_eps is None:
        # Use a transition width comparable to grid spacing.
        smooth_eps = 1.5 * float(np.sqrt(dArea))

    if use_smooth_objective:
        obj_jax = _make_soft_objective(
            knotPoints,
            num_cont_points,
            pursuerRange,
            pursuerCaptureRadius,
            pastInterseptionLocations,
            pastRadaii,
            dArea,
            integrationPoints,
            launchPdf,
            alpha=alpha,
            eps=float(smooth_eps),
            gain_weight=float(gain_weight),
            smoothness_weight=float(smoothness_weight),
            debug=bool(debug),
        )
        dobj_jax = jax.jit(jax.jacfwd(obj_jax))
    else:
        obj_jax = lambda cp: area_objective_function_trajectory(
            cp.reshape((num_cont_points, 2)),
            knotPoints,
            pursuerRange,
            pursuerCaptureRadius,
            pastInterseptionLocations,
            pastRadaii,
            dArea,
            integrationPoints,
            launchPdf,
            alpha=float(alpha),
        )
        dobj_jax = jax.jit(jax.jacfwd(obj_jax))

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

        obj_val = obj_jax(jnp.asarray(control_points_flat))

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

        dObjDControlPointsVal = dobj_jax(control_points_flat)

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

    opt.options["max_iter"] = int(os.environ.get("IPOPT_MAX_ITER", "400"))
    opt.options["hessian_approximation"] = "limited-memory"
    opt.options["nlp_scaling_method"] = "gradient-based"
    opt.options["acceptable_tol"] = float(
        os.environ.get("IPOPT_ACCEPTABLE_TOL", "1e-4")
    )
    opt.options["acceptable_constr_viol_tol"] = float(
        os.environ.get("IPOPT_ACCEPTABLE_CONSTR_VIOL_TOL", "1e-6")
    )
    opt.options["acceptable_dual_inf_tol"] = float(
        os.environ.get("IPOPT_ACCEPTABLE_DUAL_INF_TOL", "1e-3")
    )
    opt.options["acceptable_iter"] = int(os.environ.get("IPOPT_ACCEPTABLE_ITER", "10"))
    opt.options["mu_strategy"] = "adaptive"
    username = getpass.getuser()
    hsllib = "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    opt.options["hsllib"] = hsllib
    opt.options["linear_solver"] = "ma97"
    # opt.options["warm_start_init_point"] = "yes"
    # opt.options["mu_init"] = 1e-1
    opt.options["tol"] = 1e-4

    # sol = opt(optProb, sens="FD")
    sol = opt(optProb, sens=sens)
    if debug:
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


def plot_area_objective_function(
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

    # ---- Old feasible set (for launch locations) ----
    old_mask = old_feasible_mask(points, interceptionPositions, oldRadii)  # (M,) bool
    oldArea = area_from_mask(old_mask, dArea)

    launchPdf = pez_from_interceptions.uniform_pdf_from_interception_points(
        points, interceptionPositions, pursuerRange, pursuerCaptureRadius, dArea
    )

    # ---- ΔM(x): area removed if a kill occurs at x ----
    areaDiff = area_diff_from_oldmask(
        points,  # candidate kill locations (M,2)
        R_eff,
        points,  # integration points (M,2)
        old_mask,
        oldArea,
        dArea,
    )  # (M,)

    # ---- p_reach(x): probability reachable from launchPdf ----
    # p_reach(x) = ∫ 1(||L-x|| <= R_eff) * launchPdf(L) dL
    p_reach = pez_from_interceptions.prob_reach_numerical(
        points,  # eval_points (M,2)
        points,  # integration_points (M,2)
        launchPdf,  # pdf_vals (M,)
        R_eff,
        dArea,
    )  # (M,)

    # ---- "Trajectory-like" per-point objective proxy ----
    # In the trajectory objective you use S_prev * h * ΔM. For a static plot, a good proxy is:
    # score(x) = ΔM(x) * p_reach(x)
    score = areaDiff * p_reach

    # Plot the objective you'd MINIMIZE (negative score)
    obj = -score

    ax.set_aspect("equal")

    c = ax.pcolormesh(X, Y, p_reach.reshape((num_pts, num_pts)))
    ax.scatter(interceptionPositions[:, 0], interceptionPositions[:, 1], color="red")

    plt.colorbar(c, ax=ax, label="-(ΔArea · p_reach)  (lower is better)")


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

    interceptionPositions = jnp.array([[-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    # interceptionPositions = jnp.array([[-1.0, 0.0]])
    oldRadii = jnp.array(
        [
            pursuerRange + pursuerCaptureRadius,
            pursuerRange + pursuerCaptureRadius,
            pursuerRange + pursuerCaptureRadius,
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
    num_cont_points = 8
    spline_order = 3
    # velocity_constraints = (0.0, sacraficialSpeed)
    # curvature_constraints = (-0.5, 0.5)
    # turn_rate_constraints = (-1.0, 1.0)
    v = sacraficialSpeed
    R_min = 0.3

    velocity_constraints = (0.6 * v, v)  # don’t allow near-zero speed
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
    plot_area_objective_function(
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
