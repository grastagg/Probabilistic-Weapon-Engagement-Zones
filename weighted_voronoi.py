import numpy as np
import matplotlib.pyplot as plt
from wevo_py import weighted_voronoi_diagram


def evaluate_arc(center, radius, theta1, theta2, spacing):
    # if theta1 < theta2:
    #     theta2 += 2*np.pi
    # dTheta = np.arccos((-spacing**2+2*radius**2)/(2*radius**2))
    dTheta = spacing / radius
    numTheta = int((abs(theta2 - theta1)) / dTheta)
    theta = np.linspace(theta1, theta2, numTheta)
    # theta = np.unwrap(theta)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))


def plot_arc(center, radius, theta1=0, theta2=2 * np.pi, ax=None, c="b"):
    # theta = np.linspace(theta1,theta2,100)
    # theta = np.unwrap(theta)
    out = evaluate_arc(center, radius, theta1, theta2, spacing=0.5)
    x = out[:, 0]
    y = out[:, 1]

    # x = center[0] + radius*np.cos(theta)
    # y = center[1] + radius*np.sin(theta)
    ax.plot(x, y, c=c, zorder=1000)


def plot_weighted_voronoi_arcs(arcs, boundarySegments, ax):
    # for arc in arcs:
    # arc = arcs[3]
    c = "g"
    for i, arc in enumerate(arcs):
        p1 = arc[0:2]
        p2 = arc[2:4]
        center = arc[4:6]
        radius = np.linalg.norm(p1 - center)
        # theta1 = minimize_angle(np.arctan2(p1[1]-center[1],p1[0]-center[0]))
        # theta2 = minimize_angle(np.arctan2(p2[1]-center[1],p2[0]-center[0]))
        theta1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
        theta2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
        if theta2 < theta1:
            theta2 += 2 * np.pi

        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c=c)
        # plot_arc(center, radius, np.min([theta1,theta2]), np.max([theta2,theta1]), ax)
        plot_arc(center, radius, theta1, theta2, ax, c=c)
    for seg in boundarySegments:
        ax.plot(seg[:, 0], seg[:, 1], c=c, marker="o")


def save_points_and_weights_to_file(pursuerPositions, weights, filename):
    generatorPoints = pursuerPositions
    print("Generator Points: ", generatorPoints.shape)
    weights = weights * 10000
    data = np.rint(np.hstack((generatorPoints, weights.reshape(-1, 1)))).astype(int)
    np.savetxt(filename, data, delimiter=" ", fmt="%i")


def load_weighted_voronoi_segments_from_file(filename):
    data = np.genfromtxt(filename, delimiter=",")
    # sourcePoints = data[:,0:2]
    # targetPoints = data[:,2:4]
    # centers = data[:,4:6]
    return data


def is_between_angles_radians(start_angle, stop_angle, angle):
    """
    Checks if a third angle lies between a start and stop angle counter-clockwise (radians).

    Args:
        start_angle: The starting angle in radians (0 to 2*pi).
        stop_angle: The stopping angle in radians (0 to 2*pi).
        angle: The angle to check if it lies between start and stop (0 to 2*pi).

    Returns:
        True if the angle lies between start and stop counter-clockwise, False otherwise.
    """

    # Normalize angles to 0-2*pi range
    start_angle = start_angle % (2 * np.pi)
    stop_angle = stop_angle % (2 * np.pi)
    angle = angle % (2 * np.pi)

    # Handle wraparound
    if stop_angle < start_angle:
        # return (angle > start_angle or angle < stop_angle)
        return np.logical_or(angle > start_angle, angle < stop_angle)
    else:
        # return start_angle < angle < stop_angle
        return np.logical_and(start_angle < angle, angle < stop_angle)


def combine_attached_arcs(arcs):
    # For some reason the c++ function splits arcs into two segments, this function will combine them
    for i in reversed(range(len(arcs))):
        for j in reversed(range(i)):
            if np.linalg.norm(arcs[i][4:6] - arcs[j][4:6]) < 1e-6:
                theta1 = np.arctan2(arcs[i][1] - arcs[i][5], arcs[i][0] - arcs[i][4])
                theta2 = np.arctan2(arcs[i][3] - arcs[i][5], arcs[i][2] - arcs[i][4])
                theta3 = np.arctan2(arcs[j][1] - arcs[j][5], arcs[j][0] - arcs[j][4])
                theta4 = np.arctan2(arcs[j][3] - arcs[j][5], arcs[j][2] - arcs[j][4])
                thetas = np.array([theta1, theta2, theta3, theta4])
                thetas = np.unwrap(thetas)
                # thetas[thetas < 0] += 2*np.pi

                roundedThetas = np.round(thetas, decimals=4)

                # Step 3: Find unique elements and their counts
                unique_elements, counts = np.unique(roundedThetas, return_counts=True)

                # Step 4: Filter elements that occur exactly once
                unique_elements_single_occurrence = unique_elements[counts == 1]

                # Step 5: Get indices of these unique elements in the original array
                indices = np.array(
                    [
                        index
                        for index, element in enumerate(roundedThetas)
                        if element in unique_elements_single_occurrence
                    ]
                )

                if len(indices) == 2:
                    middleAngle = unique_elements[counts == 2]
                    points = np.array(
                        [arcs[i][0:2], arcs[i][2:4], arcs[j][0:2], arcs[j][2:4]]
                    )
                    minIndex = np.argmin(thetas[indices])
                    maxIndex = np.argmax(thetas[indices])
                    startAngle = thetas[indices[minIndex]]
                    stopAngle = thetas[indices[maxIndex]]
                    if not is_between_angles_radians(
                        startAngle, stopAngle, middleAngle
                    ):
                        temp = maxIndex
                        maxIndex = minIndex
                        minIndex = temp
                    arcs[j][0:2] = points[indices[minIndex]]
                    arcs[j][2:4] = points[indices[maxIndex]]

                    arcs = np.delete(arcs, i, axis=0)

                    break

    return arcs


def circle_line_segment_intersection(
    circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9
):
    """Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx**2 + dy**2) ** 0.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius**2 * dr**2 - big_d**2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (
                cx
                + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**0.5)
                / dr**2,
                cy + (-big_d * dx + sign * abs(dy) * discriminant**0.5) / dr**2,
            )
            for sign in ((1, -1) if dy < 0 else (-1, 1))
        ]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [
                (xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy
                for xi, yi in intersections
            ]
            intersections = [
                pt
                for pt, frac in zip(intersections, fraction_along_segment)
                if 0 <= frac <= 1
            ]
        if (
            len(intersections) == 2 and abs(discriminant) <= tangent_tol
        ):  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections


def is_point_on_arc(point, center, p1, p2):
    angle_p1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    angle_p2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
    angle_point = np.arctan2(point[1] - center[1], point[0] - center[0])

    if angle_p1 < 0:
        angle_p1 += 2 * np.pi
    if angle_p2 < 0:
        angle_p2 += 2 * np.pi
    if angle_point < 0:
        angle_point += 2 * np.pi

    if angle_p1 > angle_p2:
        return angle_point >= angle_p1 or angle_point <= angle_p2
    else:
        return angle_p1 <= angle_point <= angle_p2


def is_point_on_segment(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Check if point is within the bounding box of the segment
    epsilon = 1e-6
    if (min(x1, x2) - epsilon <= x <= max(x1, x2) + epsilon) and (
        min(y1, y2) - epsilon <= y <= max(y1, y2) + epsilon
    ):
        return True
    return False


def arc_line_segment_intersection(center, radius, p1, p2, line_start, line_end):
    # intersections = circle_line_intersection(center, radius, line_start, line_end)
    intersections = circle_line_segment_intersection(
        center, radius, line_start, line_end, full_line=False, tangent_tol=1e-9
    )

    arc_intersections = [
        pt
        for pt in intersections
        if is_point_on_arc(pt, center, p1, p2)
        and is_point_on_segment(pt, line_start, line_end)
    ]
    return arc_intersections


def intersect_arcs_with_boundary(arcs, bounds):
    intersections = []
    arcsToDelete = []
    arcsToAdd = []
    for i, arc in enumerate(arcs):
        p1 = arc[0:2].copy()
        p2 = arc[2:4].copy()
        center = arc[4:6].copy()
        radius = np.linalg.norm(p1 - center)

        currentIntersections = []
        currentIntersections += arc_line_segment_intersection(
            center, radius, p1, p2, [0, 0], [0, bounds[1]]
        )
        currentIntersections += arc_line_segment_intersection(
            center, radius, p1, p2, [0, bounds[1]], [bounds[0], bounds[1]]
        )
        currentIntersections += arc_line_segment_intersection(
            center, radius, p1, p2, [bounds[0], bounds[1]], [bounds[0], 0]
        )
        currentIntersections += arc_line_segment_intersection(
            center, radius, p1, p2, [bounds[0], 0], [0, 0]
        )
        p1In = np.all(
            np.array([p1[0] >= 0, p1[0] <= bounds[0], p1[1] >= 0, p1[1] <= bounds[1]])
        )
        p2In = np.all(
            np.array([p2[0] >= 0, p2[0] <= bounds[0], p2[1] >= 0, p2[1] <= bounds[1]])
        )

        if len(currentIntersections) == 1:
            if p1In:
                arcs[i][2:4] = currentIntersections[0]
            elif p2In:
                arcs[i][0:2] = currentIntersections[0]
        if len(currentIntersections) == 2:
            if p1In and p2In:
                dist1 = np.linalg.norm(p1 - currentIntersections[0])
                dist2 = np.linalg.norm(p2 - currentIntersections[0])
                if dist1 < dist2:
                    arcs[i][0:2] = currentIntersections[0]
                    arcs[i][2:4] = p1
                    arcsToAdd.append(
                        np.array([p2, currentIntersections[1], center])
                        .reshape(-1, 6)
                        .flatten()
                    )
                else:
                    arcs[i][0:2] = p1
                    arcs[i][2:4] = currentIntersections[1]

                    arcsToAdd.append(
                        np.array([currentIntersections[0], p2, center])
                        .reshape(-1, 6)
                        .flatten()
                    )

            else:
                arcs[i][0:2] = currentIntersections[0]
                arcs[i][2:4] = currentIntersections[1]
        if not (p1In or p2In):
            arcsToDelete.append(i)
        intersections += currentIntersections

    for i in reversed(arcsToDelete):
        arcs = np.delete(arcs, i, axis=0)

    for arc in arcsToAdd:
        arcs = np.vstack((arcs, arc))

    boundarySegments = []
    intersections = np.array(intersections)
    # leftBoundaryIntersections = np.sort(intersections[intersections[:,0] == 0],axis=0)
    # rightBoundaryIntersections = np.sort(intersections[intersections[:,0] == bounds[0]],axis=0)
    # topBoundaryIntersections = np.sort(intersections[intersections[:,1] == bounds[1]],axis=0)
    # bottomBoundaryIntersections = np.sort(intersections[intersections[:,1] == 0],axis=0)

    leftBoundaryIntersections = np.sort(
        intersections[np.isclose(intersections[:, 0], 0)], axis=0
    )
    rightBoundaryIntersections = np.sort(
        intersections[np.isclose(intersections[:, 0], bounds[0])], axis=0
    )
    topBoundaryIntersections = np.sort(
        intersections[np.isclose(intersections[:, 1], bounds[1])], axis=0
    )
    bottomBoundaryIntersections = np.sort(
        intersections[np.isclose(intersections[:, 1], 0)], axis=0
    )

    if len(leftBoundaryIntersections) == 0:
        leftBoundaryIntersections = np.array([[0, 0], [0, bounds[1]]])
    else:
        for i in range(len(leftBoundaryIntersections) + 1):
            if i == 0:
                boundarySegments.append(
                    np.array([[0, 0], leftBoundaryIntersections[0]])
                )
            elif i == len(leftBoundaryIntersections):
                boundarySegments.append(
                    np.array([leftBoundaryIntersections[-1], [0, bounds[1]]])
                )
            else:
                boundarySegments.append(
                    np.array(
                        [leftBoundaryIntersections[i - 1], leftBoundaryIntersections[i]]
                    )
                )
    if len(rightBoundaryIntersections) == 0:
        rightBoundaryIntersections = np.array([[bounds[0], 0], [bounds[0], bounds[1]]])
    else:
        for i in range(len(rightBoundaryIntersections) + 1):
            if i == 0:
                boundarySegments.append(
                    np.array([[bounds[0], 0], rightBoundaryIntersections[0]])
                )
            elif i == len(rightBoundaryIntersections):
                boundarySegments.append(
                    np.array([rightBoundaryIntersections[-1], [bounds[0], bounds[1]]])
                )
            else:
                boundarySegments.append(
                    np.array(
                        [
                            rightBoundaryIntersections[i - 1],
                            rightBoundaryIntersections[i],
                        ]
                    )
                )

    if len(topBoundaryIntersections) == 0:
        topBoundaryIntersections = np.array([[0, bounds[1]], [bounds[0], bounds[1]]])
    else:
        for i in range(len(topBoundaryIntersections) + 1):
            if i == 0:
                boundarySegments.append(
                    np.array([[0, bounds[1]], topBoundaryIntersections[0]])
                )
            elif i == len(topBoundaryIntersections):
                boundarySegments.append(
                    np.array([topBoundaryIntersections[-1], [bounds[0], bounds[1]]])
                )
            else:
                boundarySegments.append(
                    np.array(
                        [topBoundaryIntersections[i - 1], topBoundaryIntersections[i]]
                    )
                )
    if len(bottomBoundaryIntersections) == 0:
        bottomBoundaryIntersections = np.array([[0, 0], [bounds[0], 0]])
    else:
        for i in range(len(bottomBoundaryIntersections) + 1):
            if i == 0:
                boundarySegments.append(
                    np.array([[0, 0], bottomBoundaryIntersections[0]])
                )
            elif i == len(bottomBoundaryIntersections):
                boundarySegments.append(
                    np.array([bottomBoundaryIntersections[-1], [bounds[0], 0]])
                )
            else:
                boundarySegments.append(
                    np.array(
                        [
                            bottomBoundaryIntersections[i - 1],
                            bottomBoundaryIntersections[i],
                        ]
                    )
                )
    return arcs, boundarySegments
