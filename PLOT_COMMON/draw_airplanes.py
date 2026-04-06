"""Airplane glyphs and a small engagement-zone geometry demo."""

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Polygon

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def draw_airplane(ax, position, color="red", size=1.8, angle=0, zorder=100):
    """Draw a stylized airplane centered at ``position``."""
    position = np.asarray(position)

    fuselage = (
        np.array(
            [
                [0.00, 1.00],
                [-0.10, 0.55],
                [-0.12, -0.55],
                [-0.06, -1.00],
                [0.06, -1.00],
                [0.12, -0.55],
                [0.10, 0.55],
            ]
        )
        * size
    )
    left_wing = (
        np.array(
            [
                [-0.10, -0.10],
                [-0.95, -0.45],
                [-0.20, -0.42],
                [-0.02, -0.22],
            ]
        )
        * size
    )
    right_wing = (
        np.array(
            [
                [0.10, -0.10],
                [0.02, -0.22],
                [0.20, -0.42],
                [0.95, -0.45],
            ]
        )
        * size
    )
    left_tail = (
        np.array(
            [
                [-0.08, -0.72],
                [-0.42, -0.95],
                [-0.02, -0.88],
            ]
        )
        * size
    )
    right_tail = (
        np.array(
            [
                [0.08, -0.72],
                [0.02, -0.88],
                [0.42, -0.95],
            ]
        )
        * size
    )
    vertical_tail = (
        np.array(
            [
                [0.00, -0.72],
                [-0.04, -1.02],
                [0.04, -1.02],
            ]
        )
        * size
    )
    canopy = (
        np.array(
            [
                [0.00, 0.72],
                [-0.055, 0.45],
                [0.055, 0.45],
            ]
        )
        * size
    )

    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])

    def transform(points):
        return points @ rotation.T + position

    outline = [pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()]
    patch_kwargs = dict(
        closed=True,
        edgecolor="black",
        linewidth=0.8,
        joinstyle="round",
        zorder=zorder,
        clip_on=False,
    )

    for points in (
        left_wing,
        right_wing,
        left_tail,
        right_tail,
        vertical_tail,
        fuselage,
    ):
        patch = Polygon(transform(points), facecolor=color, **patch_kwargs)
        patch.set_path_effects(outline)
        ax.add_patch(patch)

    canopy_patch = Polygon(
        transform(canopy),
        closed=True,
        facecolor="#87CEEB",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.95,
        zorder=zorder + 1,
        clip_on=False,
    )
    canopy_patch.set_path_effects(outline)
    ax.add_patch(canopy_patch)

    nose_center = transform(np.array([[0.0, 0.92 * size]]))[0]
    ax.add_patch(
        Circle(
            nose_center,
            radius=0.035 * size,
            facecolor="white",
            edgecolor="none",
            alpha=0.35,
            zorder=zorder + 2,
            clip_on=False,
        )
    )


def angle_to_target(p1, heading, p2):
    """Return the legacy angle used by the original demo figure."""
    del heading  # retained for compatibility with older callers
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_to_point = np.arctan2(dy, dx)
    return angle_to_point % (2 * np.pi)


def _plot_engagement_case(ax, blue_plane, title):
    """Draw one of the two engagement-zone geometry panels used in the demo."""
    from PEZ import pez_plotting

    red_plane = np.array([0.0, 0.0])
    pursuer_range = 1.0
    pursuer_capture_range = 0.2
    pursuer_speed = 2.0
    agent_speed = 1.5
    evader_heading = 0.0
    pursuer_initial_position = np.array([0.0, 0.0])
    blue_heading = -np.pi / 2

    speed_ratio = agent_speed / pursuer_speed
    evader_terminal = blue_plane + speed_ratio * pursuer_range * np.array(
        [np.cos(evader_heading), np.sin(evader_heading)]
    )
    theta = np.arctan2(
        evader_terminal[1] - red_plane[1], evader_terminal[0] - red_plane[0]
    )
    target_point = red_plane + pursuer_range * np.array([np.cos(theta), np.sin(theta)])

    ax.add_artist(
        plt.Circle(red_plane, pursuer_range, color="black", fill=False, linestyle="--")
    )
    ax.plot(
        [red_plane[0], target_point[0]],
        [red_plane[1], target_point[1]],
        "black",
        linestyle="-.",
        label="Pursuer Path",
    )
    ax.add_artist(
        plt.Circle(
            target_point,
            pursuer_capture_range,
            color="black",
            fill=False,
            linestyle=":",
        )
    )

    theta2 = theta
    radius_end_point = target_point + pursuer_capture_range * np.array(
        [np.cos(theta2), np.sin(theta2)]
    )
    ax.plot(
        [target_point[0], radius_end_point[0]],
        [target_point[1], radius_end_point[1]],
        "black",
        linestyle="-",
    )

    ax.text(red_plane[0] - 0.2, red_plane[1], r"$x_P$", fontsize=20, color="red")
    draw_airplane(ax, red_plane, color="red", size=0.2, angle=theta - np.pi / 2)

    ax.text(blue_plane[0] - 0.35, blue_plane[1], r"$x_E$", fontsize=20, color="blue")
    draw_airplane(ax, blue_plane, color="blue", size=0.2, angle=blue_heading)

    blue_line_length = speed_ratio * pursuer_range
    ax.plot(
        [blue_plane[0], blue_plane[0] + blue_line_length],
        [blue_plane[1], blue_plane[1]],
        color="black",
        linestyle=":",
    )
    ax.text(
        blue_plane[0] + 0.5 * blue_line_length,
        blue_plane[1] + 0.05,
        r"$\nu R$",
        fontsize=20,
        color="red",
    )

    arc_angle_rad = angle_to_target(blue_plane, blue_heading, red_plane)
    arc_radius = 0.25
    arc_angle = np.degrees(arc_angle_rad)
    ax.add_artist(
        Arc(
            blue_plane,
            arc_radius * 2,
            arc_radius * 2,
            theta1=arc_angle,
            theta2=0,
            color="blue",
            linestyle="--",
        )
    )

    ax.plot(
        [red_plane[0], blue_plane[0]],
        [red_plane[1], blue_plane[1]],
        "black",
        linestyle="--",
    )
    midpoint = (red_plane + blue_plane) / 2
    ax.text(midpoint[0] - 0.1, midpoint[1], r"$\rho$", fontsize=23, color="black", ha="center")

    arc_label_position = blue_plane + 0.6 * arc_radius * np.array(
        [np.cos(-np.pi / 4), np.sin(-np.pi / 4)]
    )
    ax.text(
        arc_label_position[0],
        arc_label_position[1] - 0.03,
        r"$\xi$",
        fontsize=18,
        color="blue",
        ha="center",
    )

    range_label_position = (
        red_plane
        + (pursuer_range / 2) * np.array([np.cos(theta), np.sin(theta)])
        + np.array([0.05, 0.05])
    )
    ax.text(
        range_label_position[0] + 0.1,
        range_label_position[1] - 0.1,
        r"$R$",
        fontsize=23,
        color="red",
        ha="right",
    )

    capture_label_position = target_point + (pursuer_capture_range / 2) * np.array(
        [np.cos(theta2), np.sin(theta2)]
    )
    ax.text(
        capture_label_position[0] + 0.03,
        capture_label_position[1] - 0.04,
        r"$r$",
        fontsize=23,
        color="red",
        ha="left",
    )

    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_aspect("equal")
    ax.set_xlim(-pursuer_range - 1.0, pursuer_range + 1.0)
    ax.set_ylim(-pursuer_range - 1.0, pursuer_range + 1.0)

    pez_plotting.plotEngagementZone(
        0.0,
        pursuer_initial_position,
        pursuer_range,
        pursuer_capture_range,
        pursuer_speed,
        agent_speed,
        ax,
    )
    ax.set_xlabel("X", fontsize=30)
    ax.set_ylabel("Y", fontsize=30)
    ax.set_title(title, fontsize=30)


def main():
    """Render the two-panel demo figure used while developing the airplane glyph."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    _plot_engagement_case(axes[0], np.array([-0.1, 1.0]), "Inside BEZ")
    _plot_engagement_case(axes[1], np.array([-0.1, 1.3]), "Outside BEZ")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
