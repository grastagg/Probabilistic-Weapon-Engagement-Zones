import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# BEZ function
# -----------------------------
def bez_z(x_p, y_p, x_e, y_e, psi_e, nu, R, r):
    return (
        (x_e - x_p + nu * R * np.cos(psi_e)) ** 2
        + (y_e - y_p + nu * R * np.sin(psi_e)) ** 2
        - (R + r) ** 2
    )


def grad(x_p0, y_p0, x_e, y_e, psi_e, nu, R):
    a = x_e - x_p0 + nu * R * np.cos(psi_e)
    b = y_e - y_p0 + nu * R * np.sin(psi_e)
    return -2 * a, -2 * b


# -----------------------------
# Parameters
# -----------------------------
x_e, y_e = 0.0, 0.0
psi_e = np.deg2rad(30)
nu = 0.7
R = 6.0
r = 1.0

# tangency point
x0, y0 = 0.0, 0.0
z0 = bez_z(x0, y0, x_e, y_e, psi_e, nu, R, r)
dzdx, dzdy = grad(x0, y0, x_e, y_e, psi_e, nu, R)

# -----------------------------
# Local curved patch (smaller)
# -----------------------------
surf_half = 3.0
xs = np.linspace(x0 - surf_half, x0 + surf_half, 160)
ys = np.linspace(y0 - surf_half, y0 + surf_half, 160)
XS, YS = np.meshgrid(xs, ys)
ZS = bez_z(XS, YS, x_e, y_e, psi_e, nu, R, r)

# -----------------------------
# Tangent plane patch (larger)
# -----------------------------
plane_half = 1.0
xp = np.linspace(x0 - plane_half, x0 + plane_half, 2)
yp = np.linspace(y0 - plane_half, y0 + plane_half, 2)
XP, YP = np.meshgrid(xp, yp)
ZP = z0 + dzdx * (XP - x0) + dzdy * (YP - y0)

# border of tangent plane
bx = np.array([xp[0], xp[-1], xp[-1], xp[0], xp[0]])
by = np.array([yp[0], yp[0], yp[-1], yp[-1], yp[0]])
bz = z0 + dzdx * (bx - x0) + dzdy * (by - y0)

# -----------------------------
# Tangent directions and normal
# -----------------------------
tx = np.array([1.0, 0.0, dzdx])
ty = np.array([0.0, 1.0, dzdy])
n = np.array([-dzdx, -dzdy, 1.0])


def normalize(v):
    return v / np.linalg.norm(v)


tx = normalize(tx)
ty = normalize(ty)
n = normalize(n)

arrow_len = 1.4
tx *= arrow_len
ty *= arrow_len
n *= 1.8

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

# cleaner projection for slides
ax.set_proj_type("ortho")

# tangent plane first
ax.plot_surface(
    XP,
    YP,
    ZP,
    color="#A8BE7D",
    alpha=0.60,
    edgecolor="none",
    linewidth=0,
    antialiased=False,
    shade=False,
)

# curved BEZ patch second
ax.plot_surface(
    XS,
    YS,
    ZS,
    color="#9FD8E5",
    alpha=0.82,
    edgecolor="none",
    linewidth=0,
    antialiased=False,
    shade=False,
)

# plane border
ax.plot(bx, by, bz, color="#7B9155", linewidth=1.5)

# tangency point
ax.scatter(x0, y0, z0, color="#222222", s=180, depthshade=False, zorder=20)

# tangent arrows
# ax.quiver(
#     x0,
#     y0,
#     z0,
#     tx[0],
#     tx[1],
#     tx[2],
#     color="#222222",
#     linewidth=1.8,
#     arrow_length_ratio=0.15,
# )
# ax.quiver(
#     x0,
#     y0,
#     z0,
#     ty[0],
#     ty[1],
#     ty[2],
#     color="#222222",
#     linewidth=1.8,
#     arrow_length_ratio=0.15,
# )
# ax.quiver(
#     x0,
#     y0,
#     z0,
#     -tx[0],
#     -tx[1],
#     -tx[2],
#     color="#222222",
#     linewidth=1.2,
#     arrow_length_ratio=0.15,
# )
# ax.quiver(
#     x0,
#     y0,
#     z0,
#     -ty[0],
#     -ty[1],
#     -ty[2],
#     color="#222222",
#     linewidth=1.2,
#     arrow_length_ratio=0.15,
# )
#
# # normal arrow
# ax.quiver(
#     x0,
#     y0,
#     z0,
#     n[0],
#     n[1],
#     n[2],
#     color="#222222",
#     linewidth=2.0,
#     arrow_length_ratio=0.12,
# )
#
# clean look
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor((1, 1, 1, 0))
    axis.line.set_color((1, 1, 1, 0))

# local framing only
ax.set_xlim(x0 - plane_half, x0 + plane_half)
ax.set_ylim(y0 - plane_half, y0 + plane_half)

zmin = min(ZS.min(), ZP.min())
zmax = max(ZS.max(), ZP.max())
ax.set_zlim(zmin - 0.8, zmax + 1.2)

# important: view from above so plane is underneath, not "on top"
ax.view_init(elev=24, azim=-128)
ax.set_box_aspect([1, 1, 0.55])

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

plt.tight_layout()
plt.show()

# plt.savefig("bez_tangent_plane_clear.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
