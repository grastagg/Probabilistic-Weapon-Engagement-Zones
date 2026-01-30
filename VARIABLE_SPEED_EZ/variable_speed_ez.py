import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from PEZ import pez_plotting


@jax.jit
def inside_EZ_pwconstspeed_batch(
    p0_batch,  # (B, 2) initial evader positions in pursuer-start frame
    psi_batch,  # (B,)   headings (rad)
    v_e,  # scalar
    t_breaks,  # (S+1,) [t0=0, ..., tS=T]
    v_p_segments,  # (S,)   pursuer speeds on [t_i, t_{i+1})
    tol=1e-12,
    eps_a=1e-12,
):
    """
    Returns: captured (B,) bool
      True  => evader ever enters EZ (||p(t)|| <= R(t)) for some t in [0,T]
      False => stays outside for whole horizon

    Model:
      p(t) = p0 + v_e * [cos psi, sin psi] * t
      R(t) = ∫ v_p(τ) dτ where v_p is piecewise-constant => R(t) piecewise-linear
    """

    # --- Evader direction and projection ---
    u = jnp.stack([jnp.cos(psi_batch), jnp.sin(psi_batch)], axis=-1)  # (B,2)
    r0_sq = jnp.sum(p0_batch * p0_batch, axis=-1)  # (B,)
    Sdot = jnp.sum(p0_batch * u, axis=-1)  # (B,) = p0^T u

    # --- Build piecewise-linear R(t) from piecewise-constant speeds ---
    t0 = t_breaks[:-1]  # (S,)
    t1 = t_breaks[1:]  # (S,)
    dt = t1 - t0  # (S,)
    m = v_p_segments  # (S,) slope on each segment

    # R_at[k] = R(t_breaks[k])
    R_at = jnp.concatenate(
        [jnp.array([0.0], dtype=t_breaks.dtype), jnp.cumsum(m * dt)]
    )  # (S+1,)

    # R(t) = m_i t + c_i on segment i
    c = R_at[:-1] - m * t0  # (S,)

    # --- Quadratic gap g(t)=||p(t)||^2 - R(t)^2 on each segment ---
    # g_i(t) = a_i t^2 + b_i t + d_i
    # with heading dependence only through Sdot
    a = (v_e**2 - m**2)[None, :]  # (1,S) -> (B,S)
    b = 2.0 * (v_e * Sdot)[:, None] - 2.0 * (m * c)[None, :]  # (B,S)
    d = r0_sq[:, None] - (c**2)[None, :]  # (B,S)

    # Evaluate at endpoints
    t0b = t0[None, :]
    t1b = t1[None, :]

    g0 = a * t0b * t0b + b * t0b + d  # (B,S)
    g1 = a * t1b * t1b + b * t1b + d  # (B,S)

    hit_endpoints = (g0 <= tol) | (g1 <= tol)

    # Vertex check for true quadratics (a != 0)
    a_ok = jnp.abs(a) > eps_a
    t_star = -b / (2.0 * a)  # (B,S)
    in_seg = (t_star > t0b) & (t_star < t1b)

    g_star = a * t_star * t_star + b * t_star + d
    hit_vertex = a_ok & in_seg & (g_star <= tol)

    hit_any_seg = hit_endpoints | hit_vertex  # (B,S)
    return jnp.any(hit_any_seg, axis=1)  # (B,)


# ----------------------------
# Angle + interval utilities
# ----------------------------
def wrap_to_pi(a):
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def intervals_union(intervals, eps=0.0):
    """Union of 1D closed intervals [(lo,hi),...], assumes lo<=hi. Returns sorted, merged."""
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    out = [ints[0]]
    for lo, hi in ints[1:]:
        plo, phi = out[-1]
        if lo <= phi + eps:
            out[-1] = (plo, max(phi, hi))
        else:
            out.append((lo, hi))
    return out


def intervals_intersection(A, B, eps=0.0):
    """Intersection of two lists of closed intervals."""
    out = []
    i = j = 0
    A = sorted(A)
    B = sorted(B)
    while i < len(A) and j < len(B):
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if lo <= hi + eps:
            out.append((lo, hi))
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out


def interval_subtract(base, cut, eps=0.0):
    """
    base: list of intervals, cut: single interval (c_lo, c_hi)
    returns base \ cut
    """
    c_lo, c_hi = cut
    out = []
    for lo, hi in base:
        # no overlap
        if hi < c_lo - eps or lo > c_hi + eps:
            out.append((lo, hi))
            continue
        # overlap: possibly left remainder
        if lo < c_lo - eps:
            out.append((lo, min(hi, c_lo)))
        # possibly right remainder
        if hi > c_hi + eps:
            out.append((max(lo, c_hi), hi))
    return out


def normalize_angle_intervals(intervals):
    """
    Normalize arbitrary real-angle intervals to a union of intervals in [-pi, pi).
    """
    out = []
    for lo, hi in intervals:
        lo = wrap_to_pi(lo)
        hi = wrap_to_pi(hi)
        # If interval doesn't wrap:
        if lo <= hi:
            out.append((lo, hi))
        else:
            # wraps across pi -> split
            out.append((lo, np.pi))
            out.append((-np.pi, hi))
    return intervals_union(out, eps=0.0)


# -----------------------------------------
# Core: safe heading cones for BEZ model
# -----------------------------------------


def safe_heading_cones_piecewise_constant_pursuer_speed(
    p0,  # array-like (2,) initial evader position in pursuer-start frame
    v_e,  # evader speed (scalar > 0)
    t_breaks,  # (S+1,) with t_breaks[0]=0, increasing, final is T
    v_p_segments,  # (S,) pursuer speeds on [t_i, t_{i+1})
    strict=False,  # if True uses ">" safety; else ">= with tiny tol"
    tol=1e-12,
):
    """
    Returns list of safe heading intervals [(psi_lo, psi_hi), ...] in radians,
    normalized to [-pi, pi), where choosing any psi in the union guarantees:
        ||p0 + v_e*[cos psi, sin psi]*t|| > R(t)  for all t in [0,T]
    under the BEZ disk-reach model with piecewise-constant pursuer speed.

    Notes:
      - In general the result is a union of disjoint intervals.
      - If there are no safe headings, returns [].
    """

    p0 = np.asarray(p0, dtype=float)
    t_breaks = np.asarray(t_breaks, dtype=float)
    v_p_segments = np.asarray(v_p_segments, dtype=float)

    assert p0.shape == (2,)
    assert t_breaks.ndim == 1 and v_p_segments.ndim == 1
    assert len(t_breaks) == len(v_p_segments) + 1
    assert np.all(np.diff(t_breaks) > 0)
    assert abs(t_breaks[0]) < 1e-15
    assert v_e > 0.0

    # Safety inequality uses g(t) = ||p(t)||^2 - R(t)^2 >= 0 (or > 0).
    # We'll implement with a small tolerance for numerical stability.
    eps = tol if not strict else 0.0

    x0, y0 = p0
    r0 = np.hypot(x0, y0)
    if r0 == 0.0:
        # At origin initially: if R(0)=0 you're "on the boundary"; otherwise inside immediately.
        # With strict safety, no heading can help at t=0.
        return []

    phi = np.arctan2(y0, x0)

    # Build R(t_breaks[k]) from piecewise-constant speeds
    dt = np.diff(t_breaks)  # (S,)
    R_at = np.zeros_like(t_breaks)
    R_at[1:] = np.cumsum(v_p_segments * dt)

    # We'll carry safe sets in S-space:
    # S = p0^T u(psi) = r0 cos(psi - phi) in [-r0, r0]
    S_safe = [(-r0, r0)]  # start with all possible S

    r0_sq = r0 * r0

    for i in range(len(v_p_segments)):
        t0 = t_breaks[i]
        t1 = t_breaks[i + 1]
        m = v_p_segments[i]  # slope of R(t) on this segment
        c = R_at[i] - m * t0  # intercept, so R(t)=m t + c on [t0,t1]

        # Segment gap: g(t) = ||p0 + v_e u t||^2 - (m t + c)^2
        # = (v_e^2 - m^2) t^2 + 2(v_e S - m c) t + (r0^2 - c^2)
        a = v_e**2 - m**2
        d = r0_sq - c**2

        # --- Base constraints from endpoints (min for concave/linear always at endpoints;
        #     for convex also needed) ---
        # g(t) = alpha(t) + 2 v_e t * S, where alpha(t)=v_e^2 t^2 + r0^2 - (m t + c)^2
        def alpha(t):
            return (v_e**2) * t * t + r0_sq - (m * t + c) ** 2

        base = [(-np.inf, np.inf)]
        # t=0 endpoint: constraint independent of S
        if t0 == 0.0:
            if alpha(t0) < eps:  # g(0)=r0^2 - R(0)^2 = r0^2 - c^2
                return []
        else:
            # g(t0) >= 0 => S >= -alpha(t0)/(2 v_e t0)
            L0 = (-alpha(t0) + eps) / (2.0 * v_e * t0)
            base = intervals_intersection(base, [(L0, np.inf)], eps=0.0)

        # t1 endpoint:
        if t1 == 0.0:
            if alpha(t1) < eps:
                return []
        else:
            L1 = (-alpha(t1) + eps) / (2.0 * v_e * t1)
            base = intervals_intersection(base, [(L1, np.inf)], eps=0.0)

        # Intersect base with feasible S range [-r0, r0]
        base = intervals_intersection(base, [(-r0, r0)], eps=0.0)
        base = intervals_union(base)

        if not base:
            return []

        # If a <= 0, min over [t0,t1] is at endpoints, so base is sufficient.
        if a <= 0.0 or d <= 0.0:
            seg_safe = base
        else:
            # a > 0 (convex). If the unconstrained vertex lies inside the segment,
            # we also need g(t*) >= 0. Vertex time depends on S:
            # t*(S) = (m c - v_e S)/a
            # Vertex inside iff t0 <= t*(S) <= t1,
            # which gives an interval in S:
            # (m c - a t1)/v_e <= S <= (m c - a t0)/v_e
            Sv_lo = (m * c - a * t1) / v_e
            Sv_hi = (m * c - a * t0) / v_e
            if Sv_lo > Sv_hi:
                Sv_lo, Sv_hi = Sv_hi, Sv_lo

            # When vertex is inside, enforce g(t*) = d - (v_e S - m c)^2 / a >= 0
            # => |v_e S - m c| <= sqrt(a d)
            rad = np.sqrt(a * d)
            Sg_lo = (m * c - rad) / v_e
            Sg_hi = (m * c + rad) / v_e

            # Compute parts:
            # - Outside vertex-inside region: base is enough
            # - Inside vertex-inside region: also require S in [Sg_lo,Sg_hi]
            vertex_region = (Sv_lo, Sv_hi)
            inside = intervals_intersection(base, [vertex_region], eps=0.0)
            inside = intervals_intersection(inside, [(Sg_lo, Sg_hi)], eps=0.0)
            inside = intervals_union(inside)

            outside = interval_subtract(base, vertex_region, eps=0.0)
            outside = intervals_union(outside)

            seg_safe = intervals_union(outside + inside)

        # Intersect across segments
        S_safe = intervals_intersection(S_safe, seg_safe, eps=0.0)
        S_safe = intervals_union(S_safe)
        if not S_safe:
            return []

    # Map S-intervals to heading intervals using S = r0 cos(theta), theta = psi - phi
    psi_intervals = []
    for Slo, Shi in S_safe:
        # Clamp to [-r0,r0] already, but numerical guard:
        A = np.clip(Slo / r0, -1.0, 1.0)
        B = np.clip(Shi / r0, -1.0, 1.0)
        if A > B:
            A, B = B, A

        # cos(theta) in [A,B] -> |theta| in [arccos(B), arccos(A)]
        th1 = np.arccos(B)  # smaller
        th2 = np.arccos(A)  # larger

        # Two symmetric theta intervals: [th1, th2] and [-th2, -th1]
        # Convert to psi = phi + theta
        psi_intervals.append((phi + th1, phi + th2))
        psi_intervals.append((phi - th2, phi - th1))

    # Normalize to [-pi, pi) and union
    psi_intervals = normalize_angle_intervals(psi_intervals)
    return psi_intervals


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


def plot_heading_cones(
    cones,  # list of (psi_lo, psi_hi) in radians, assumed in [-pi, pi)
    ax=None,
    r=1.0,
    center=(0.0, 0.0),
    show_unit_circle=True,
    title="Safe heading cones",
):
    """
    Plots heading cones (angular intervals) as wedges on a unit circle.

    Parameters
    ----------
    cones : list[(float,float)]
        List of angular intervals (psi_lo, psi_hi) in radians. Intervals may be
        disjoint. Assumed normalized to [-pi, pi) and non-wrapping (lo <= hi).
        If you have wrapping intervals, split them before calling.
    ax : matplotlib axis
        If None, creates a new figure+axis.
    r : float
        Radius of the drawn circle/wedges.
    center : tuple
        (x,y) center for the circle.
    show_unit_circle : bool
        Draw the circle outline.
    title : str
        Plot title.

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    cx, cy = center

    # Optional outline circle
    if show_unit_circle:
        th = np.linspace(-np.pi, np.pi, 512)
        ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), linewidth=1)

    # Draw wedges for each interval
    # Matplotlib Wedge uses degrees and angles measured CCW from +x, same as our psi.
    for lo, hi in cones:
        lo_deg = np.degrees(lo)
        hi_deg = np.degrees(hi)
        # Wedge draws the filled sector from lo to hi (CCW).
        wedge = Wedge(center=(cx, cy), r=r, theta1=lo_deg, theta2=hi_deg)
        ax.add_patch(wedge)

    ax.set_aspect("equal", adjustable="box")
    # ax.set_xlabel("vx direction")
    # ax.set_ylabel("vy direction")
    # ax.set_title(title)
    # ax.set_xlim(cx - 1.1 * r, cx + 1.1 * r)
    # ax.set_ylim(cy - 1.1 * r, cy + 1.1 * r)

    # Light axes
    # ax.axhline(cy, linewidth=1)
    # ax.axvline(cx, linewidth=1)

    return ax


def plot_safe_cones_for_case(
    p0,
    v_e,
    t_breaks,
    v_p_segments,
    cones_fn,
    r=1.0,
    title_prefix="Safe heading cones",
):
    """
    Convenience wrapper: compute cones, then plot.

    cones_fn should be your safe cone function, e.g.
      safe_heading_cones_piecewise_constant_pursuer_speed
    """
    cones = cones_fn(p0, v_e, t_breaks, v_p_segments)

    ax = plot_heading_cones(cones, r=r, title=f"{title_prefix}\n p0={p0}, v_e={v_e}")

    # draw the direction from origin to p0 for reference
    phi = np.arctan2(p0[1], p0[0])
    ax.plot([0, r * np.cos(phi)], [0, r * np.sin(phi)], linewidth=2)

    return cones, ax


def main():
    psi_e = np.deg2rad(0.0)  # single evader heading (rad)
    # psi_e = 2.2255

    v_e = 1.0  # evader speed

    t_breaks = jnp.array([0.0, 0.5, 1.0])  # (S+1,)
    v_p = jnp.array([3.5, 1.5])  # (S,)
    # find average pursuer speed
    dt = jnp.diff(t_breaks)
    v_avg = jnp.sum(v_p * dt) / jnp.sum(dt)
    print(f"Average pursuer speed: v_avg={v_avg}")
    # find the distance traveled by pursuer
    R = v_avg * (t_breaks[-1] - t_breaks[0])

    # domain / grid resolution
    xmin, xmax, Nx = -4.0, 4.0, 1001
    ymin, ymax, Ny = -4.0, 4.0, 1001

    x = jnp.linspace(xmin, xmax, Nx)
    y = jnp.linspace(ymin, ymax, Ny)
    X, Y = jnp.meshgrid(x, y, indexing="xy")  # (Ny,Nx)

    p0_grid = jnp.stack([X, Y], axis=-1)  # (Ny,Nx,2)

    # --- evaluate EZ membership ---
    captured_mask = inside_EZ_pwconstspeed_batch(
        p0_grid.reshape(-1, 2), psi_e * jnp.ones((Nx * Ny,)), v_e, t_breaks, v_p
    ).reshape(Ny, Nx)
    Z = np.array(captured_mask, dtype=np.float32)  # convert for matplotlib

    # --- plot boundary ---
    fig, ax = plt.subplots()
    plt.contour(
        np.array(X), np.array(Y), Z, levels=[0.5]
    )  # boundary between safe/unsafe
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"EZ boundary (psi={np.rad2deg(psi_e):.1f} deg, v_e={v_e})")
    plt.xlabel("x")
    plt.ylabel("y")
    # plot BEZ of average speed (this only works if average speed > evader speed)
    pez_plotting.plotEngagementZone(
        psi_e,
        jnp.array([0.0, 0.0]),
        R,
        0,
        v_avg,
        v_e,
        ax,
    )
    # plot speed profile
    fig2, ax2 = plt.subplots()
    ax2.step(x=t_breaks, y=jnp.concatenate([v_p, v_p[-1:]]), where="post")
    # plot average speed line
    ax2.hlines(
        v_avg,
        xmin=t_breaks[0],
        xmax=t_breaks[-1],
        colors="red",
        linestyles="dashed",
        label=f"Avg speed: {v_avg:.2f}",
    )
    # plot evader speed line
    ax2.hlines(
        v_e,
        xmin=t_breaks[0],
        xmax=t_breaks[-1],
        colors="green",
        linestyles="dashed",
        label=f"Evader speed: {v_e:.2f}",
    )
    plt.legend()
    plt.title("Pursuer Speed Profile")

    evaderPosition = np.array([-1.4, 0.0])
    cones = safe_heading_cones_piecewise_constant_pursuer_speed(
        evaderPosition, v_e, t_breaks, v_p
    )
    print("cones:", cones)
    plot_heading_cones(
        cones, ax, center=evaderPosition, title="Safe heading cones (wedges)"
    )
    ax.scatter(*evaderPosition, color="blue", label="Evader Position")
    ax.scatter(0, 0)
    plt.show()


if __name__ == "__main__":
    main()
