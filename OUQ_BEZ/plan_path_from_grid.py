import numpy as np
import json
from pyoptsparse import Optimization, OPT, IPOPT
from scipy.interpolate import BSpline
import time
from tqdm import tqdm
from jax import jacfwd
from jax import jit
import jax
from functools import partial
import jax.numpy as jnp
import getpass
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
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

# set backend to agg (only if running on remote server and saving images instead of viewing them)
# matplotlib.use("agg")


import bspline.spline_opt_tools as spline_opt_tools
import PEZ.pez as pez
import PEZ.pez_plotting as pez_plotting
import PLOT_COMMON.draw_mahalanobis as draw_mahalanobis

numSamplesPerInterval = 15


def load_grid_data(grid_file_path, param_file_path):
    grid_data = np.load(grid_file_path)
    paramDict = json.load(open(param_file_path, "r"))
    return (
        grid_data,
        paramDict["minx"],
        paramDict["dx"],
        paramDict["nx"],
        paramDict["miny"],
        paramDict["dy"],
        paramDict["ny"],
        paramDict["minpsi"],
        paramDict["dpsi"],
        paramDict["npsi"],
    )


def plot_grid_data(
    X,
    Y,
    grid_data,
    levels=None,
    colors="viridis",
    ax=None,
    inLine=False,
):
    if levels is None:
        levels = np.linspace(0.1, 1, 10)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.contour(X, Y, grid_data, levels=levels, linewidths=2, cmap=colors)
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
    ax.set_aspect("equal")


@jax.jit
def trilerp_uniform_periodic_psi_single(
    x,
    y,
    psi,
    x_min,
    dx,
    nx,
    y_min,
    dy,
    ny,
    psi_min,
    dpsi,
    npsi,
    P_xypsi,  # shape: (nx, ny, npsi)
):
    """
    Trilinear interpolation on a uniform (x,y,psi) grid with periodic psi.

    Grid definition:
      X[i]   = x_min   + i*dx,   i=0..nx-1
      Y[j]   = y_min   + j*dy,   j=0..ny-1
      Psi[k] = psi_min + k*dpsi, k=0..npsi-1   (period = npsi*dpsi)

    P_xypsi shape: (nx, ny, npsi) where P_xypsi[i,j,k] = P(X[i], Y[j], Psi[k])

    Notes:
      - x,y are clamped to the grid domain (no extrapolation).
      - psi is wrapped periodically.
    """
    # --- out-of-bounds (x,y only; psi is periodic) ---
    x_max = x_min + dx * (nx - 1)
    y_max = y_min + dy * (ny - 1)
    # oob computed but not used; keep if you later want conservative behavior
    # oob = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)

    # --- clamp x,y for stable indexing ---
    xq = jnp.clip(x, x_min, x_max)
    yq = jnp.clip(y, y_min, y_max)

    # --- periodic wrap for psi into [psi_min, psi_min + 2*pi) (assumes full 2π coverage) ---
    # If your psi grid uses a different period, replace 2*pi with (dpsi*npsi).
    psiq = psi_min + jnp.mod(psi - psi_min, 2.0 * jnp.pi)

    # --- compute fractional indices ---
    fx = (xq - x_min) / dx
    fy = (yq - y_min) / dy
    fp = (psiq - psi_min) / dpsi

    i = jnp.floor(fx).astype(jnp.int32)
    j = jnp.floor(fy).astype(jnp.int32)
    k = jnp.floor(fp).astype(jnp.int32)

    # clamp to valid *cell* indices (need i+1, j+1)
    i = jnp.clip(i, 0, nx - 2)
    j = jnp.clip(j, 0, ny - 2)

    # psi is periodic so wrap indices
    k = jnp.mod(k, npsi)
    k1 = jnp.mod(k + 1, npsi)

    # fractional coords within the cell
    tx = fx - i
    ty = fy - j
    tp = fp - jnp.floor(fp)  # in [0,1)

    # --- gather 8 corners (x,y,psi ordering) ---
    p000 = P_xypsi[i, j, k]
    p100 = P_xypsi[i + 1, j, k]
    p010 = P_xypsi[i, j + 1, k]
    p110 = P_xypsi[i + 1, j + 1, k]

    p001 = P_xypsi[i, j, k1]
    p101 = P_xypsi[i + 1, j, k1]
    p011 = P_xypsi[i, j + 1, k1]
    p111 = P_xypsi[i + 1, j + 1, k1]

    # --- trilinear blend ---
    p00 = p000 * (1.0 - tx) + p100 * tx
    p10 = p010 * (1.0 - tx) + p110 * tx
    p01 = p001 * (1.0 - tx) + p101 * tx
    p11 = p011 * (1.0 - tx) + p111 * tx

    p0 = p00 * (1.0 - ty) + p10 * ty
    p1 = p01 * (1.0 - ty) + p11 * ty

    p = p0 * (1.0 - tp) + p1 * tp
    return p


trilerp_uniform_periodic_psi = jax.jit(
    jax.vmap(
        trilerp_uniform_periodic_psi_single,
        in_axes=(
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
)


def create_spline(knotPoints, controlPoints, order):
    spline = BSpline(knotPoints, controlPoints, order)
    return spline


def rect_left_and_top(lower_left, upper_right, n_points_total):
    x1, y1 = lower_left
    x2, y2 = upper_right

    # Compute edge lengths
    left_len = y2 - y1
    top_len = x2 - x1
    total_len = left_len + top_len

    # Split points proportionally to edge lengths
    n_left = max(2, int(round(n_points_total * (left_len / total_len))))
    n_top = max(
        2, n_points_total - n_left + 1
    )  # +1 to avoid losing one point at the corner

    # Left edge: (x = x1, y1 → y2)
    left = np.column_stack((np.full(n_left, x1), np.linspace(y1, y2, n_left)))

    # Top edge: (x1 → x2, y = y2)
    top = np.column_stack((np.linspace(x1, x2, n_top), np.full(n_top, y2)))

    # Concatenate, skipping the repeated corner (x1, y2)
    pts = np.vstack((left, top[1:]))
    return pts


def rect_bottom_and_right(lower_left, upper_right, n_points_total):
    x1, y1 = lower_left
    x2, y2 = upper_right

    # Compute edge lengths
    bottom_len = x2 - x1
    right_len = y2 - y1
    total_len = bottom_len + right_len

    # Split points proportionally to edge lengths
    n_bottom = max(2, int(round(n_points_total * (bottom_len / total_len))))
    n_right = max(
        2, n_points_total - n_bottom + 1
    )  # +1 to avoid losing one point at the corner

    # Bottom edge: (x1 → x2, y = y1)
    bottom = np.column_stack((np.linspace(x1, x2, n_bottom), np.full(n_bottom, y1)))

    # Right edge: (x = x2, y1 → y2)
    right = np.column_stack((np.full(n_right, x2), np.linspace(y1, y2, n_right)))

    # Concatenate, skipping the repeated corner (x2, y1)
    pts = np.vstack((bottom, right[1:]))
    return pts


@jax.jit
def interp_PEZ_along_spline(
    controlPoints,
    tf,
    x_min,
    dx,
    nx,
    y_min,
    dy,
    ny,
    psi_min,
    dpsi,
    npsi,
    P_yxpsi,
):
    numControlPoints = int(len(controlPoints) / 2)
    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, tf, numControlPoints, 3
    )
    evaderHeadings = spline_opt_tools.get_spline_heading(
        controlPoints, tf, 3, numSamplesPerInterval
    )
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )
    # ez = rectangle_pez.prob_reachable_uniform_box(
    #     pos,
    #     pursuerRange,
    #     pursuerCaptureRadius,
    #     min_box,
    #     max_box,
    # )

    interpPez = trilerp_uniform_periodic_psi(
        pos[:, 0],
        pos[:, 1],
        evaderHeadings,
        x_min,
        dx,
        nx,
        y_min,
        dy,
        ny,
        psi_min,
        dpsi,
        npsi,
        P_yxpsi,
    )
    return interpPez


# @jax.jit
def compute_spline_constraints_for_interp_PEZ(
    controlPoints,
    knotPoints,
    x_min,
    dx,
    nx,
    y_min,
    dy,
    ny,
    psi_min,
    dpsi,
    npsi,
    P_yxpsi,
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

    interpPez = trilerp_uniform_periodic_psi(
        pos[:, 0],
        pos[:, 1],
        evaderHeadings,
        x_min,
        dx,
        nx,
        y_min,
        dy,
        ny,
        psi_min,
        dpsi,
        npsi,
        P_yxpsi,
    )

    return velocity, turn_rate, curvature, interpPez, pos, evaderHeadings


dInterpPEZDControlPoints = jax.jit(jax.jacfwd(interp_PEZ_along_spline, argnums=0))
dInterpPEZDtf = jax.jit(jax.jacfwd(interp_PEZ_along_spline, argnums=1))


def optimize_spline_path_interp_Pez(
    p0,
    pf,
    v0,
    num_cont_points,
    spline_order,
    velocity_constraints,
    turn_rate_constraints,
    curvature_constraints,
    num_constraint_samples,
    x_min,
    dx,
    nx,
    y_min,
    dy,
    ny,
    psi_min,
    dpsi,
    npsi,
    P_yxpsi,
    right=True,
    previous_spline=None,
    pez_limit=0.1,
):
    # Compute Jacobian of engagement zone function

    def objfunc(xDict):
        tf = xDict["tf"]
        knotPoints = spline_opt_tools.create_unclamped_knot_points(
            0, tf, num_cont_points, 3
        )
        controlPoints = xDict["control_points"]
        funcs = {}
        funcs["start"] = spline_opt_tools.get_start_constraint(controlPoints)
        funcs["end"] = spline_opt_tools.get_end_constraint(controlPoints)
        controlPoints = controlPoints.reshape((num_cont_points, 2))

        velocity, turn_rate, curvature, ez, pos, evaderHeading = (
            compute_spline_constraints_for_interp_PEZ(
                controlPoints,
                knotPoints,
                x_min,
                dx,
                nx,
                y_min,
                dy,
                ny,
                psi_min,
                dpsi,
                npsi,
                P_yxpsi,
            )
        )

        # funcs['start'] = self.get_start_constraint_jax(controlPoints)
        # funcs['start'] = pos[0]
        # funcs['end'] = pos[-1]
        funcs["obj"] = tf
        funcs["turn_rate"] = turn_rate
        funcs["velocity"] = velocity
        funcs["curvature"] = curvature
        # funcs['position'] = pos
        funcs["ez"] = ez
        funcs["obj"] = tf
        funcs["pos"] = pos
        funcs["heading"] = evaderHeading
        return funcs, False

    def sens(xDict, funcs):
        funcsSens = {}
        controlPoints = jnp.array(xDict["control_points"])
        tf = xDict["tf"]

        dStartDControlPointsVal = spline_opt_tools.get_start_constraint_jacobian(
            controlPoints
        )
        dEndDControlPointsVal = spline_opt_tools.get_end_constraint_jacobian(
            controlPoints
        )

        dEZDControlPoints = dInterpPEZDControlPoints(
            controlPoints,
            tf,
            x_min,
            dx,
            nx,
            y_min,
            dy,
            ny,
            psi_min,
            dpsi,
            npsi,
            P_yxpsi,
        )
        dEZDtf = dInterpPEZDtf(
            controlPoints,
            tf,
            x_min,
            dx,
            nx,
            y_min,
            dy,
            ny,
            psi_min,
            dpsi,
            npsi,
            P_yxpsi,
        )
        # replace nans with zeros
        # dEZDtf = np.array(dEZDtf, copy=True)
        # dEZDtf[np.isnan(dEZDtf)] = 0.0
        # dEZDControlPoints = np.array(dEZDControlPoints, copy=True)
        # dEZDControlPoints[np.isnan(dEZDControlPoints)] = 0.0
        dVelocityDControlPointsVal = spline_opt_tools.dVelocityDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dVelocityDtfVal = spline_opt_tools.dVelocityDtf(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dTurnRateDControlPointsVal = spline_opt_tools.dTurnRateDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dTurnRateDtfVal = spline_opt_tools.dTurnRateTf(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dCurvatureDControlPointsVal = spline_opt_tools.dCurvatureDControlPoints(
            controlPoints, tf, 3, numSamplesPerInterval
        )
        dCurvatureDtfVal = spline_opt_tools.dCurvatureDtf(
            controlPoints, tf, 3, numSamplesPerInterval
        )

        funcsSens["obj"] = {
            "control_points": np.zeros((1, 2 * num_cont_points)),
            "tf": 1,
        }
        funcsSens["start"] = {
            "control_points": dStartDControlPointsVal,
            "tf": np.zeros((2, 1)),
        }
        funcsSens["end"] = {
            "control_points": dEndDControlPointsVal,
            "tf": np.zeros((2, 1)),
        }
        funcsSens["velocity"] = {
            "control_points": dVelocityDControlPointsVal,
            "tf": dVelocityDtfVal,
        }
        funcsSens["turn_rate"] = {
            "control_points": dTurnRateDControlPointsVal,
            "tf": dTurnRateDtfVal,
        }
        funcsSens["curvature"] = {
            "control_points": dCurvatureDControlPointsVal,
            "tf": dCurvatureDtfVal,
        }
        funcsSens["ez"] = {"control_points": dEZDControlPoints, "tf": dEZDtf}

        return funcsSens, False

    # num_constraint_samples = numSamplesPerInterval*(num_cont_points-2)-2

    optProb = Optimization("path optimization", objfunc)

    # if previous_spline is not None:
    if previous_spline is not None:
        x0 = previous_spline.c.flatten()
        tf = previous_spline.t[-1 - previous_spline.k]
    else:
        start = time.time()
        tf_initial = 1.0
        knotPoints = spline_opt_tools.create_unclamped_knot_points(
            0, tf_initial, num_cont_points, 3
        )

        # x0 = np.linspace(p0, pf, num_cont_points).flatten()
        if right:
            x0 = rect_bottom_and_right(p0, pf, num_cont_points).flatten()
        else:
            x0 = rect_left_and_top(p0, pf, num_cont_points).flatten()

        # x0 = spline_opt_tools.move_first_control_point_so_spline_passes_through_start(
        #     x0, knotPoints, p0, v0
        # )
        # x0 = x0.flatten()
        # x0 = spline_opt_tools.move_last_control_point_so_spline_passes_through_end(
        #     x0, knotPoints, pf, v0
        # )
        x0 = x0.flatten()

        tf = spline_opt_tools.assure_velocity_constraint(
            x0,
            knotPoints,
            num_cont_points,
            velocity_constraints[1],
            velocity_constraints,
            numSamplesPerInterval,
        )

    tempVelocityContstraints = spline_opt_tools.get_spline_velocity(
        x0, 1, 3, numSamplesPerInterval
    )
    start = time.time()
    num_constraint_samples = len(tempVelocityContstraints)

    optProb.addVarGroup(
        name="control_points", nVars=2 * (num_cont_points), varType="c", value=x0
    )
    optProb.addVarGroup(name="tf", nVars=1, varType="c", value=tf, lower=0, upper=None)

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
    optProb.addConGroup("ez", num_constraint_samples, lower=None, upper=pez_limit)
    optProb.addConGroup("start", 2, lower=p0, upper=p0)
    optProb.addConGroup("end", 2, lower=pf, upper=pf)

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

    # sol = opt(optProb, sens="FD")
    sol = opt(optProb, sens=sens)
    print(sol)

    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        0, sol.xStar["tf"][0], num_cont_points, 3
    )

    controlPoints = sol.xStar["control_points"].reshape((num_cont_points, 2))

    print("Optimization time:", time.time() - start)
    return (
        create_spline(knotPoints, controlPoints, spline_order),
        sol.xStar["tf"][0],
    )


def plan_path_interp_PEZ(
    x_min,
    dx,
    nx,
    y_min,
    dy,
    ny,
    psi_min,
    dpsi,
    npsi,
    P_yxpsi,
    initialEvaderPosition,
    finalEvaderPosition,
    initialEvaderVelocity,
    num_cont_points,
    spline_order,
    velocity_constraints,
    turn_rate_constraints,
    curvature_constraints,
    num_constraint_samples,
    pez_limit,
):
    splineLeft, tfLeft = optimize_spline_path_interp_Pez(
        initialEvaderPosition,
        finalEvaderPosition,
        initialEvaderVelocity,
        num_cont_points=num_cont_points,
        spline_order=spline_order,
        velocity_constraints=velocity_constraints,
        turn_rate_constraints=turn_rate_constraints,
        curvature_constraints=curvature_constraints,
        num_constraint_samples=num_constraint_samples,
        x_min=x_min,
        dx=dx,
        nx=nx,
        y_min=y_min,
        dy=dy,
        ny=ny,
        psi_min=psi_min,
        dpsi=dpsi,
        npsi=npsi,
        P_yxpsi=P_yxpsi,
        right=False,
        previous_spline=None,
        pez_limit=pez_limit,
    )
    splineRight, tfRight = optimize_spline_path_interp_Pez(
        initialEvaderPosition,
        finalEvaderPosition,
        initialEvaderVelocity,
        num_cont_points=num_cont_points,
        spline_order=spline_order,
        velocity_constraints=velocity_constraints,
        turn_rate_constraints=turn_rate_constraints,
        curvature_constraints=curvature_constraints,
        num_constraint_samples=num_constraint_samples,
        x_min=x_min,
        dx=dx,
        nx=nx,
        y_min=y_min,
        dy=dy,
        ny=ny,
        psi_min=psi_min,
        dpsi=dpsi,
        npsi=npsi,
        P_yxpsi=P_yxpsi,
        right=True,
        previous_spline=None,
        pez_limit=pez_limit,
    )
    print("Right path time", tfRight)
    print("Left path time", tfLeft)
    if tfRight < tfLeft:
        spline = splineRight
        tf = tfRight
    else:
        spline = splineLeft
        tf = tfLeft
    print("path time", tf)
    return spline, tf


def main():
    grid_file_path = "OUQ_BEZ/linPez.npy"
    param_file_path = "OUQ_BEZ/linPezParams.json"
    print(f"Loading grid data from {grid_file_path}...")
    pez, minx, dx, nx, miny, dy, ny, minpsi, dpsi, npsi = load_grid_data(
        grid_file_path, param_file_path
    )
    print("grid data shape", pez.shape)
    print("grid parameters", minx, dx, nx, miny, dy, ny, minpsi, dpsi, npsi)

    new_x = np.linspace(-2, 2, 500)
    new_y = np.linspace(-2, 2, 500)
    new_X, new_Y = np.meshgrid(new_x, new_y)
    points = np.stack([new_X.ravel(), new_Y.ravel()], axis=-1)
    heading = np.deg2rad(0.0)
    new_psi = heading * np.ones(points.shape[0])
    print("shape of points", points.shape)
    print("shape of new psi", new_psi.shape)
    interpPez = trilerp_uniform_periodic_psi(
        points[:, 0],
        points[:, 1],
        new_psi,
        minx,
        dx,
        nx,
        miny,
        dy,
        ny,
        minpsi,
        dpsi,
        npsi,
        pez,
    ).reshape(new_X.shape)
    # interp_pez = pez_lookup_rotated_bilinear_vmap(
    #     X, Y, pez, points[:, 0], points[:, 1], new_psi, 0.0, 0.0
    # )
    #
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(
        new_X,
        new_Y,
        interpPez.reshape(new_X.shape),
        levels=10,
        alpha=0.5,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    pezParams = json.load(open("OUQ_BEZ/linPezParamsPEZ.json", "r"))
    pursuerRange = pezParams["pursuerRange"]
    pursuerRangeVar = pezParams["pursuerRangeVar"]
    pursuerCaptureRange = pezParams["pursuerCaptureRange"]
    pursuerCaptureRangeVar = pezParams["pursuerCaptureRangeVar"]
    pursuerPositionMean = jnp.asarray(pezParams["pursuerPosition"])
    pursuerPositionCov = jnp.asarray(pezParams["pursuerPositionCov"])
    pursuerSpeed = pezParams["pursuerSpeed"]
    pursuerSpeedVar = pezParams["pursuerSpeedVar"]
    agentSpeed = pezParams["agentSpeed"]
    pez_plotting.plotProbablisticEngagementZone(
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        heading,
        0.0,
        pursuerPositionMean,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
        ax,
        levels=None,
        colors="viridis",
    )
    ax.set_aspect("equal")

    # Define the grid
    numPoints = 100

    plt.show()


def main_path():
    initialEvaderPosition = np.array([-4.0, -4.0])
    finalEvaderPosition = np.array([4.0, 4.0])
    initialEvaderVelocity = np.array([1.0, 1.0]) / np.sqrt(2)

    numControlPoints = 18
    splineOrder = 3
    turn_rate_constraints = (-5.0, 5.0)
    curvature_constraints = (-1.0, 1.0)
    num_constraint_samples = 50
    # velocity_constraints = (0,1.0)
    velocity_constraints = (0.0, 1.0)

    grid_file_path = "OUQ_BEZ/linPez.npy"
    param_file_path = "OUQ_BEZ/linPezParams.json"
    print(f"Loading grid data from {grid_file_path}...")
    P_yxpsi, minx, dx, nx, miny, dy, ny, minpsi, dpsi, npsi = load_grid_data(
        grid_file_path, param_file_path
    )

    pez_limit = 0.01
    spline, tf = plan_path_interp_PEZ(
        minx,
        dx,
        nx,
        miny,
        dy,
        ny,
        minpsi,
        dpsi,
        npsi,
        P_yxpsi,
        initialEvaderPosition,
        finalEvaderPosition,
        initialEvaderVelocity,
        numControlPoints,
        splineOrder,
        velocity_constraints,
        turn_rate_constraints,
        curvature_constraints,
        num_constraint_samples,
        pez_limit,
    )

    pos = spline(np.linspace(0, tf, num=500))
    vel = spline.derivative()(np.linspace(0, tf, num=500))
    heading = np.arctan2(vel[:, 1], vel[:, 0])

    pezParams = json.load(open("OUQ_BEZ/linPezParamsPEZ.json", "r"))
    pursuerRange = pezParams["pursuerRange"]
    pursuerRangeVar = pezParams["pursuerRangeVar"]
    pursuerCaptureRange = pezParams["pursuerCaptureRange"]
    pursuerCaptureRangeVar = pezParams["pursuerCaptureRangeVar"]
    pursuerPositionMean = jnp.asarray(pezParams["pursuerPosition"])
    pursuerPositionCov = jnp.asarray(pezParams["pursuerPositionCov"])
    pursuerSpeed = pezParams["pursuerSpeed"]
    pursuerSpeedVar = pezParams["pursuerSpeedVar"]
    agentSpeed = pezParams["agentSpeed"]

    pezData = pez.probabalisticEngagementZoneVectorizedTemp(
        pos,
        jnp.array([[0.0, 0.0], [0, 0]]),
        heading,
        0.0,
        pursuerPositionMean,
        pursuerPositionCov,
        pursuerRange,
        pursuerRangeVar,
        pursuerCaptureRange,
        pursuerCaptureRangeVar,
        pursuerSpeed,
        pursuerSpeedVar,
        agentSpeed,
    )
    print("max pez along path", np.max(pezData))
    interPezData = trilerp_uniform_periodic_psi(
        pos[:, 0],
        pos[:, 1],
        heading,
        minx,
        dx,
        nx,
        miny,
        dy,
        ny,
        minpsi,
        dpsi,
        npsi,
        P_yxpsi,
    )
    print("max interp pez along path", np.max(interPezData))
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = spline(np.linspace(0, tf, num=100))
    ax.plot(pos[:, 0], pos[:, 1], label="Optimized Path")
    ax.set_aspect("equal")
    plt.show()


def create_lin_pez_grid():
    numPoints = 100
    numHeadings = 50

    pursuerRange = 1.0
    pursuerRangeVar = 0.1
    pursuerCaptureRange = 0.1
    pursuerCaptureRangeVar = 0.02
    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.0
    agentSpeed = 1.0

    agentPositionCov = np.array([[0.0, 0.0], [0.0, 0.0]])
    pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
    pursuerPosition = np.array([0.0, 0.0])

    agentInitialHeading = 0.0
    agentHeadingVar = 0.0
    headings = np.linspace(-np.pi, np.pi, numHeadings)
    x = jnp.linspace(-5, 5, numPoints)
    y = jnp.linspace(-5, 5, numPoints)
    X, Y = jnp.meshgrid(x, y)
    agentPositions = jnp.vstack([X.ravel(), Y.ravel()]).T
    allData = []
    for heading in headings:
        print(heading)
        pezData = pez.probabalisticEngagementZoneVectorizedTemp(
            agentPositions,
            agentPositionCov,
            heading * np.ones(numPoints * numPoints),
            agentHeadingVar,
            pursuerPosition,
            pursuerPositionCov,
            pursuerRange,
            pursuerRangeVar,
            pursuerCaptureRange,
            pursuerCaptureRangeVar,
            pursuerSpeed,
            pursuerSpeedVar,
            agentSpeed,
        )
        allData.append(pezData.reshape(X.shape))

    allData = np.array(allData)
    allData = jnp.transpose(allData, (2, 1, 0))
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    nx = X.shape[1]
    ny = Y.shape[0]
    npsi = len(headings)
    dHeading = float(headings[1] - headings[0])
    minX = float(X.min())
    minY = float(Y.min())
    minHeading = float(headings.min())
    params = {
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "npsi": npsi,
        "dpsi": dHeading,
        "minx": minX,
        "miny": minY,
        "minpsi": minHeading,
    }
    pez_params = {
        "agentPositionCov": agentPositionCov.tolist(),
        "agentHeadingVar": agentHeadingVar,
        "pursuerPosition": pursuerPosition.tolist(),
        "pursuerPositionCov": pursuerPositionCov.tolist(),
        "pursuerRange": pursuerRange,
        "pursuerRangeVar": pursuerRangeVar,
        "pursuerCaptureRange": pursuerCaptureRange,
        "pursuerCaptureRangeVar": pursuerCaptureRangeVar,
        "pursuerSpeed": pursuerSpeed,
        "pursuerSpeedVar": pursuerSpeedVar,
        "agentSpeed": agentSpeed,
    }
    np.save("./OUQ_BEZ/linPez.npy", allData)
    # save params
    json.dump(params, open("./OUQ_BEZ/linPezParams.json", "w"))
    json.dump(pez_params, open("./OUQ_BEZ/linPezParamsPEZ.json", "w"))


if __name__ == "__main__":
    main_path()
    # main()
