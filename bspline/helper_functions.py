"""Utility helpers shared by the B-spline evaluation code.

The surrounding research code uses two common control-point layouts:

- `(dimension, num_control_points)` for matrix-evaluation helpers
- `(num_control_points,)` for scalar splines

These helpers keep the original conventions intact while making the expected
shapes clearer for collaborators reading the code.
"""

import numpy as np


def get_dimension(control_points):
    """Return the spline output dimension for the given control-point array."""
    if control_points.ndim == 1:
        return 1
    return len(control_points)


def count_number_of_control_points(control_points):
    """Return the number of control points in either supported array layout."""
    if control_points.ndim == 1:
        return len(control_points)
    return len(control_points[0])


def calculate_number_of_control_points(order, knot_points):
    """Infer the control-point count from the knot vector and spline order."""
    return len(knot_points) - order - 1


def find_preceding_knot_index(time, order, knot_points):
    """Return the knot index immediately preceding ``time``."""
    number_of_control_points = calculate_number_of_control_points(order, knot_points)

    if time >= knot_points[number_of_control_points - 1]:
        return number_of_control_points - 1

    preceding_knot_index = number_of_control_points - 1
    for knot_index in range(order, number_of_control_points + 1):
        knot_point = knot_points[knot_index]
        next_knot_point = knot_points[knot_index + 1]
        if knot_point <= time < next_knot_point:
            preceding_knot_index = knot_index
            break

    return preceding_knot_index


def find_end_time(control_points, knot_points):
    """Return the spline end time implied by the current knot vector."""
    return knot_points[count_number_of_control_points(control_points)]


def get_time_to_point_correlation(points, start_time, end_time):
    """Evenly distribute sample times across a sequence of points."""
    number_of_points = count_number_of_control_points(points)
    return np.linspace(start_time, end_time, number_of_points)


def create_random_control_points_greater_than_angles(
    num_control_points, order, length, dimension
):
    """Create a simple 2D random walk with bounded turning angle by order."""
    if dimension != 2:
        raise ValueError("This helper currently supports only 2D control points.")

    if order in (1, 2):
        angle = np.pi / 2
    elif order == 3:
        angle = np.pi / 4
    elif order == 4:
        angle = np.pi / 6
    elif order == 5:
        angle = np.pi / 8
    else:
        raise ValueError(f"Unsupported spline order for random control points: {order}")

    control_points = np.zeros((dimension, num_control_points))
    for index in range(num_control_points):
        if index == 0:
            control_points[:, index][:, None] = np.array([[0], [0]])
        elif index == 1:
            random_vec = np.random.rand(2, 1)
            next_vec = length * random_vec / np.linalg.norm(random_vec)
            control_points[:, index][:, None] = (
                control_points[:, index - 1][:, None] + next_vec
            )
        else:
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            previous_vec = (
                control_points[:, index - 1][:, None]
                - control_points[:, index - 2][:, None]
            )
            unit_previous_vec = previous_vec / np.linalg.norm(previous_vec)
            next_vec = length * np.dot(rotation, unit_previous_vec)
            control_points[:, index][:, None] = (
                control_points[:, index - 1][:, None] + next_vec
            )

    return control_points
