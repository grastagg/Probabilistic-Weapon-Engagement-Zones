import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Arc
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import fast_pursuer


# Function to draw a more realistic and larger airplane shape
def draw_airplane(ax, position, color="red", size=1.5, angle=0):
    # Fuselage (main body)
    fuselage = np.array([[0, 0.6], [-0.08, -0.8], [0.08, -0.8]]) * size

    # Left wing
    left_wing = np.array([[-0.08, -0.2], [-0.6, -0.5], [-0.08, -0.5]]) * size

    # Right wing
    right_wing = np.array([[0.08, -0.2], [0.08, -0.5], [0.6, -0.5]]) * size

    # Tail fin
    tail = np.array([[-0.03, -0.8], [0.03, -0.8], [0, -1]]) * size

    # Cockpit (optional detail)
    cockpit = np.array([[0, 0.4], [-0.05, 0], [0.05, 0]]) * size

    # Rotate all parts by the specified angle (in radians)
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    fuselage_rotated = fuselage @ rotation_matrix.T
    left_wing_rotated = left_wing @ rotation_matrix.T
    right_wing_rotated = right_wing @ rotation_matrix.T
    tail_rotated = tail @ rotation_matrix.T
    cockpit_rotated = cockpit @ rotation_matrix.T

    # Translate the airplane to its position
    fuselage_translated = fuselage_rotated + position
    left_wing_translated = left_wing_rotated + position
    right_wing_translated = right_wing_rotated + position
    tail_translated = tail_rotated + position
    cockpit_translated = cockpit_rotated + position

    # Create polygons for each part and add them to the plot
    ax.add_patch(Polygon(fuselage_translated, closed=True, color=color))
    ax.add_patch(Polygon(left_wing_translated, closed=True, color=color))
    ax.add_patch(Polygon(right_wing_translated, closed=True, color=color))
    ax.add_patch(Polygon(tail_translated, closed=True, color=color))
    ax.add_patch(
        Polygon(cockpit_translated, closed=True, color="gray")
    )  # Cockpit in a different color


def angle_to_target(p1, heading, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_to_point = np.arctan2(dy, dx)
    heading = 0
    relative_angle = (angle_to_point - heading) % (2 * np.pi)
    return relative_angle


def angle_to_target(p1, heading, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_to_point = np.arctan2(dy, dx)
    heading = 0
    relative_angle = (angle_to_point - heading) % (2 * np.pi)
    return relative_angle


def main():
    # Setup figure and axis
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    # Coordinates for the red plane (pursuer) and blue plane (evader)
    red_plane = np.array([0, 0])  # Red plane at origin
    blue_plane = np.array([-0.255, 1.17])  # Blue plane at the requested position
    blue_plane = np.array([-0.1, 1.0])  # Blue plane at the requested position
    # blue_plane = np.array([-0.1, 1.3])  # Blue plane at the requested position
    # blue_plane = np.array([-0.255, 1.3])  # Blue plane at the requested position

    # Pursuer range and capture radius (scaled down)
    pursuerRange = 1.0
    pursuerCaptureRange = 0.2
    pursuerSpeed = 2.0
    agentSpeed = 1.5
    evaderHeading = 0  # Blue plane heading towards the x-axis

    pursuerInitialPosition = np.array([0.0, 0.0])

    # Angle for the line (calculated to point toward the blue plane)
    #
    speedRatio = agentSpeed / pursuerSpeed
    evaderTerminal = blue_plane + speedRatio * pursuerRange * np.array(
        [np.cos(evaderHeading), np.sin(evaderHeading)]
    )
    theta = np.arctan2(blue_plane[1] - red_plane[1], blue_plane[0] - red_plane[0])

    theta = np.arctan2(
        evaderTerminal[1] - red_plane[1], evaderTerminal[0] - red_plane[0]
    )
    # Calculate the point on the circle (pursuer range boundary)
    target_point = red_plane + pursuerRange * np.array([np.cos(theta), np.sin(theta)])

    # Plot pursuer range as a black circle
    pursuer_range_circle = plt.Circle(
        red_plane, pursuerRange, color="black", fill=False, linestyle="--"
    )
    ax.add_artist(pursuer_range_circle)

    # Draw line from red plane to the point on the circle
    ax.plot(
        [red_plane[0], target_point[0]],
        [red_plane[1], target_point[1]],
        "black",
        linestyle="-.",
        label="Pursuer Path",
    )

    # Add smaller capture radius circle at the target point
    capture_circle = plt.Circle(
        target_point, pursuerCaptureRange, color="black", fill=False, linestyle=":"
    )
    ax.add_artist(capture_circle)

    # Draw a radius inside the capture radius circle
    theta2 = theta  # Angle for the radius line
    radius_end_point = target_point + pursuerCaptureRange * np.array(
        [np.cos(theta2), np.sin(theta2)]
    )
    ax.plot(
        [target_point[0], radius_end_point[0]],
        [target_point[1], radius_end_point[1]],
        "black",
        linestyle="-",
    )

    # Draw the airplanes (larger and more realistic)
    ax.text(
        red_plane[0] - 0.2,
        red_plane[1],
        r"$x_P$",
        fontsize=20,
        color="red",
    )
    draw_airplane(
        ax, red_plane, color="red", size=0.2, angle=theta - np.pi / 2
    )  # Red airplane (pursuer)
    ax.text(
        blue_plane[0] - 0.35,
        blue_plane[1],
        r"$x_E$",
        fontsize=20,
        color="blue",
    )
    blueHeading = -np.pi / 2
    draw_airplane(
        ax, blue_plane, color="blue", size=0.2, angle=blueHeading
    )  # Blue airplane (evader) pointing toward x-axis

    # Add a dotted blue line extending straight out from the blue plane in the direction of its heading (x-axis)
    blue_line_length = (
        speedRatio * pursuerRange
    )  # Adjust this value for the length of the dotted line
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

    # Draw the arc marking the angle between the blue plane's heading and the red plane

    arcAngleRad = 0 - theta
    arcAngleRad = angle_to_target(blue_plane, blueHeading, red_plane)
    arc_radius = 0.25  # Adjust this value for the size of the arc
    arc_angle = np.degrees(arcAngleRad)  # Convert angle to degrees
    arc = Arc(
        blue_plane,
        arc_radius * 2,
        arc_radius * 2,
        theta1=arc_angle,
        theta2=0,
        color="blue",
        linestyle="--",
    )
    ax.add_artist(arc)
    ax.plot(
        [red_plane[0], blue_plane[0]],
        [red_plane[1], blue_plane[1]],
        "black",
        linestyle="--",
    )
    ave = (red_plane + blue_plane) / 2
    ax.text(ave[0] - 0.1, ave[1], r"$\rho$", fontsize=23, color="black", ha="center")

    # Label the arc angle with the Greek letter xi
    arc_label_position = blue_plane + 0.6 * arc_radius * np.array(
        [np.cos(-np.pi / 4), np.sin(-np.pi / 4)]
    )  # Place label halfway along arc
    ax.text(
        arc_label_position[0],
        arc_label_position[1] - 0.03,
        r"$\xi$",
        fontsize=18,
        color="blue",
        ha="center",
    )

    # Label the pursuer range as 'R' near the radius line
    R_label_position = (
        red_plane
        + (pursuerRange / 2) * np.array([np.cos(theta), np.sin(theta)])
        + np.array([0.05, 0.05])
    )
    ax.text(
        R_label_position[0] + 0.1,
        R_label_position[1] - 0.1,
        r"$R$",
        fontsize=23,
        color="red",
        ha="right",
    )  # Label for R

    # Label the capture radius as 'r' near the radius line of the capture circle
    r_label_position = target_point + (pursuerCaptureRange / 2) * np.array(
        [np.cos(theta2), np.sin(theta2)]
    )
    ax.text(
        r_label_position[0] + 0.03,
        r_label_position[1] - 0.04,
        r"$r$",
        fontsize=23,
        color="red",
        ha="left",
    )  # Label for r

    # Set x and y tick sizes
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)

    # Label and display the plot
    ax.set_aspect("equal")
    ax.set_xlim(-pursuerRange - 1.0, pursuerRange + 1.0)
    ax.set_ylim(-pursuerRange - 1.0, pursuerRange + 1.0)

    agentInitialHeading = 0.0
    fast_pursuer.plotEngagementZone(
        agentInitialHeading,
        pursuerInitialPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
        ax,
    )
    ax.set_xlabel("X", fontsize=30)
    ax.set_ylabel("Y", fontsize=30)
    ax.set_title("Inside BEZ", fontsize=30)

    ########delete everthing below this line to get rid of second plot
    ax = axes[1]
    # Coordinates for the red plane (pursuer) and blue plane (evader)
    red_plane = np.array([0, 0])  # Red plane at origin
    blue_plane = np.array([-0.255, 1.17])  # Blue plane at the requested position
    # blue_plane = np.array([-0.1, 1.0])  # Blue plane at the requested position
    blue_plane = np.array([-0.1, 1.3])  # Blue plane at the requested position
    # blue_plane = np.array([-0.255, 1.3])  # Blue plane at the requested position

    # Pursuer range and capture radius (scaled down)
    pursuerRange = 1.0
    pursuerCaptureRange = 0.2
    pursuerSpeed = 2.0
    agentSpeed = 1.5
    evaderHeading = 0  # Blue plane heading towards the x-axis

    pursuerInitialPosition = np.array([0.0, 0.0])

    # Angle for the line (calculated to point toward the blue plane)
    #
    speedRatio = agentSpeed / pursuerSpeed
    evaderTerminal = blue_plane + speedRatio * pursuerRange * np.array(
        [np.cos(evaderHeading), np.sin(evaderHeading)]
    )
    theta = np.arctan2(blue_plane[1] - red_plane[1], blue_plane[0] - red_plane[0])

    theta = np.arctan2(
        evaderTerminal[1] - red_plane[1], evaderTerminal[0] - red_plane[0]
    )
    # Calculate the point on the circle (pursuer range boundary)
    target_point = red_plane + pursuerRange * np.array([np.cos(theta), np.sin(theta)])

    # Plot pursuer range as a black circle
    pursuer_range_circle = plt.Circle(
        red_plane, pursuerRange, color="black", fill=False, linestyle="--"
    )
    ax.add_artist(pursuer_range_circle)

    # Draw line from red plane to the point on the circle
    ax.plot(
        [red_plane[0], target_point[0]],
        [red_plane[1], target_point[1]],
        "black",
        linestyle="-.",
        label="Pursuer Path",
    )

    # Add smaller capture radius circle at the target point
    capture_circle = plt.Circle(
        target_point, pursuerCaptureRange, color="black", fill=False, linestyle=":"
    )
    ax.add_artist(capture_circle)

    # Draw a radius inside the capture radius circle
    theta2 = theta  # Angle for the radius line
    radius_end_point = target_point + pursuerCaptureRange * np.array(
        [np.cos(theta2), np.sin(theta2)]
    )
    ax.plot(
        [target_point[0], radius_end_point[0]],
        [target_point[1], radius_end_point[1]],
        "black",
        linestyle="-",
    )

    # Draw the airplanes (larger and more realistic)
    ax.text(
        red_plane[0] - 0.2,
        red_plane[1],
        r"$x_P$",
        fontsize=20,
        color="red",
    )
    draw_airplane(
        ax, red_plane, color="red", size=0.2, angle=theta - np.pi / 2
    )  # Red airplane (pursuer)
    ax.text(
        blue_plane[0] - 0.35,
        blue_plane[1],
        r"$x_E$",
        fontsize=20,
        color="blue",
    )
    blueHeading = -np.pi / 2
    draw_airplane(
        ax, blue_plane, color="blue", size=0.2, angle=blueHeading
    )  # Blue airplane (evader) pointing toward x-axis

    # Add a dotted blue line extending straight out from the blue plane in the direction of its heading (x-axis)
    blue_line_length = (
        speedRatio * pursuerRange
    )  # Adjust this value for the length of the dotted line
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

    # Draw the arc marking the angle between the blue plane's heading and the red plane

    arcAngleRad = 0 - theta
    arcAngleRad = angle_to_target(blue_plane, blueHeading, red_plane)
    arc_radius = 0.25  # Adjust this value for the size of the arc
    arc_angle = np.degrees(arcAngleRad)  # Convert angle to degrees
    arc = Arc(
        blue_plane,
        arc_radius * 2,
        arc_radius * 2,
        theta1=arc_angle,
        theta2=0,
        color="blue",
        linestyle="--",
    )
    ax.add_artist(arc)
    ax.plot(
        [red_plane[0], blue_plane[0]],
        [red_plane[1], blue_plane[1]],
        "black",
        linestyle="--",
    )
    ave = (red_plane + blue_plane) / 2
    ax.text(ave[0] - 0.1, ave[1], r"$\rho$", fontsize=23, color="black", ha="center")

    # Label the arc angle with the Greek letter xi
    arc_label_position = blue_plane + 0.6 * arc_radius * np.array(
        [np.cos(-np.pi / 4), np.sin(-np.pi / 4)]
    )  # Place label halfway along arc
    ax.text(
        arc_label_position[0],
        arc_label_position[1] - 0.03,
        r"$\xi$",
        fontsize=18,
        color="blue",
        ha="center",
    )

    # Label the pursuer range as 'R' near the radius line
    R_label_position = (
        red_plane
        + (pursuerRange / 2) * np.array([np.cos(theta), np.sin(theta)])
        + np.array([0.05, 0.05])
    )
    ax.text(
        R_label_position[0] + 0.1,
        R_label_position[1] - 0.1,
        r"$R$",
        fontsize=23,
        color="red",
        ha="right",
    )  # Label for R

    # Label the capture radius as 'r' near the radius line of the capture circle
    r_label_position = target_point + (pursuerCaptureRange / 2) * np.array(
        [np.cos(theta2), np.sin(theta2)]
    )
    ax.text(
        r_label_position[0] + 0.03,
        r_label_position[1] - 0.04,
        r"$r$",
        fontsize=23,
        color="red",
        ha="left",
    )  # Label for r

    # Set x and y tick sizes
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)

    # Label and display the plot
    ax.set_aspect("equal")
    ax.set_xlim(-pursuerRange - 1.0, pursuerRange + 1.0)
    ax.set_ylim(-pursuerRange - 1.0, pursuerRange + 1.0)

    agentInitialHeading = 0.0
    fast_pursuer.plotEngagementZone(
        agentInitialHeading,
        pursuerInitialPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
        ax,
    )
    ax.set_xlabel("X", fontsize=30)
    ax.set_ylabel("Y", fontsize=30)
    ax.set_title("Outside BEZ", fontsize=30)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
