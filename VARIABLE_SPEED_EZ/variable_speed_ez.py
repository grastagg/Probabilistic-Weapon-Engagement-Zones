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


def main():
    psi_e = np.deg2rad(0.0)  # single evader heading (rad)

    v_e = 1.0  # evader speed

    t_breaks = jnp.array([0.0, 0.5, 1.0])  # (S+1,)
    v_p = jnp.array([3.5, 0.5])  # (S,)
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

    plt.show()


if __name__ == "__main__":
    main()
