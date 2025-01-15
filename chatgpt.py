import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
PI = math.pi


# Helper functions for angle normalization
def normalize_angle(angle):
    """Normalize angle to be within [-pi, pi]"""
    return (angle + PI) % (2 * PI) - PI


# Function to calculate the Euclidean distance between two points
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to compute the arc length given radius and angle
def arc_length(radius, theta):
    return radius * theta


# Function to draw an arc (circle segment)
def draw_arc(center, radius, start_angle, end_angle, num_points=100):
    angles = np.linspace(end_angle, start_angle, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return x, y


# Main function to generate a Dubins path (focusing on the LSL configuration)
def dubins_path(x_start, y_start, theta_start, x_goal, y_goal, theta_goal, r):
    # Normalize angles to be between -pi and pi
    theta_start = normalize_angle(theta_start)
    theta_goal = normalize_angle(theta_goal)

    # Step 1: Transform the goal position into the local frame of reference
    dx = x_goal - x_start
    dy = y_goal - y_start

    # Rotate by -theta_start
    x_prime = dx * math.cos(-theta_start) + dy * math.sin(-theta_start)
    y_prime = -dx * math.sin(-theta_start) + dy * math.cos(-theta_start)

    # Compute the angle between the start and goal positions
    alpha = math.atan2(y_prime, x_prime)

    # Calculate the Euclidean distance between the start and goal in the transformed frame
    d = math.hypot(x_prime, y_prime)

    # Compute the angle difference
    theta_diff = normalize_angle(theta_goal - theta_start - alpha)

    # Check if the LSL configuration is feasible
    if abs(theta_diff) > PI:
        raise ValueError("Path not feasible in the given configuration")

    # Calculate the arc lengths and straight line distance for the LSL configuration
    arc1_length = arc_length(r, theta_diff)
    arc2_length = arc_length(r, theta_diff)
    straight_length = d - 2 * r * math.sin(theta_diff)

    # Path points for LSL configuration (simplified path)
    return arc1_length, straight_length, arc2_length, alpha, theta_diff, r


# Function to plot the Dubins path
def plot_dubins_path(x_start, y_start, theta_start, x_goal, y_goal, theta_goal, r):
    arc1_length, straight_length, arc2_length, alpha, theta_diff, r = dubins_path(
        x_start, y_start, theta_start, x_goal, y_goal, theta_goal, r
    )

    # Calculate the centers of the arcs for LSL path
    arc1_center = (
        x_start + r * math.cos(theta_start),
        y_start + r * math.sin(theta_start),
    )
    arc2_center = (x_goal + r * math.cos(theta_goal), y_goal + r * math.sin(theta_goal))

    # Define the start and end angles for the arcs
    arc1_start_angle = theta_start
    arc1_end_angle = theta_start + theta_diff
    arc2_start_angle = theta_goal
    arc2_end_angle = theta_goal + theta_diff

    # Plot the start and goal points
    fig, ax = plt.subplots()
    ax.plot(x_start, y_start, "go", label="Start")
    ax.plot(x_goal, y_goal, "ro", label="Goal")

    # Plot the first arc
    arc1_x, arc1_y = draw_arc(arc1_center, r, arc1_start_angle, arc1_end_angle)
    ax.plot(arc1_x, arc1_y, "b-", label="Arc 1")

    # Plot the second arc
    arc2_x, arc2_y = draw_arc(arc2_center, r, arc2_start_angle, arc2_end_angle)
    ax.plot(arc2_x, arc2_y, "g-", label="Arc 2")

    # Plot the straight line
    ax.plot(
        [arc1_x[-1], arc2_x[0]], [arc1_y[-1], arc2_y[0]], "k-", label="Straight Line"
    )

    # Set up the plot
    ax.set_aspect("equal")
    ax.set_title("Dubins Path (LSL)")
    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


# Example usage
x_start, y_start, theta_start = 0, 0, 0  # Start position and heading
x_goal, y_goal, theta_goal = 10, 10, PI / 2  # Goal position and heading
r = 1  # Minimum turning radius

plot_dubins_path(x_start, y_start, theta_start, x_goal, y_goal, theta_goal, r)

