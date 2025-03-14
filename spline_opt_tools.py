import jax.numpy as jnp
from jax import jit, jacfwd
import jax
from functools import partial
import numpy as np

from bspline.matrix_evaluation import (
    matrix_bspline_evaluation_for_dataset,
    matrix_bspline_derivative_evaluation_for_dataset,
)


def evaluate_spline(controlPoints, knotPoints, numSamplesPerInterval):
    knotPoints = knotPoints.reshape((-1,))
    return matrix_bspline_evaluation_for_dataset(
        controlPoints.T, knotPoints, numSamplesPerInterval
    )


def evaluate_spline_derivative(
    controlPoints, knotPoints, splineOrder, derivativeOrder, numSamplesPerInterval
):
    scaleFactor = knotPoints[-splineOrder - 1] / (len(knotPoints) - 2 * splineOrder - 1)
    return matrix_bspline_derivative_evaluation_for_dataset(
        derivativeOrder, scaleFactor, controlPoints.T, knotPoints, numSamplesPerInterval
    )


@partial(jit, static_argnums=(2, 3))
def create_unclamped_knot_points(t0, tf, numControlPoints, splineOrder):
    internalKnots = jnp.linspace(t0, tf, numControlPoints - 2, endpoint=True)
    h = internalKnots[1] - internalKnots[0]
    knots = jnp.concatenate(
        (
            jnp.linspace(t0 - splineOrder * h, t0 - h, splineOrder),
            internalKnots,
            jnp.linspace(tf + h, tf + splineOrder * h, splineOrder),
        )
    )

    return knots


@partial(jit, static_argnums=(2, 3))
def get_spline_velocity(controlPoints, tf, splineOrder, numSamplesPerInterval):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 1, numSamplesPerInterval
    )
    return jnp.linalg.norm(out_d1, axis=1)


@partial(jit, static_argnums=(2, 3))
def get_spline_turn_rate(controlPoints, tf, splineOrder, numSamplesPerInterval):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 1, numSamplesPerInterval
    )
    out_d2 = evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 2, numSamplesPerInterval
    )
    v = jnp.linalg.norm(out_d1, axis=1)
    u = jnp.cross(out_d1, out_d2) / (v**2)
    return u


@partial(jit, static_argnums=(2, 3))
def get_spline_curvature(controlPoints, tf, splineOrder, numSamplesPerInterval):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 1, numSamplesPerInterval
    )
    out_d2 = evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 2, numSamplesPerInterval
    )
    v = jnp.linalg.norm(out_d1, axis=1)
    u = jnp.cross(out_d1, out_d2) / (v**2)
    return u / v


@partial(jit, static_argnums=(2, 3))
def get_spline_heading(controlPoints, tf, splineOrder, numSamplesPerInterval):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 1, numSamplesPerInterval
    )
    return jnp.arctan2(out_d1[:, 1], out_d1[:, 0])


def get_start_constraint(controlPoints):
    cp1 = controlPoints[0:2]
    cp2 = controlPoints[2:4]
    cp3 = controlPoints[4:6]
    return np.array((1 / 6) * cp1 + (2 / 3) * cp2 + (1 / 6) * cp3)


def get_end_constraint(controlPoints):
    cpnMinus2 = controlPoints[-6:-4]
    cpnMinus1 = controlPoints[-4:-2]
    cpn = controlPoints[-2:]
    return (1 / 6) * cpnMinus2 + (2 / 3) * cpnMinus1 + (1 / 6) * cpn


def get_start_constraint_jacobian(controlPoints):
    numControlPoints = int(controlPoints.shape[0] / 2)
    jac = np.zeros((2, 2 * numControlPoints))
    jac[0, 0] = 1 / 6
    jac[0, 2] = 2 / 3
    jac[0, 4] = 1 / 6
    jac[1, 1] = 1 / 6
    jac[1, 3] = 2 / 3
    jac[1, 5] = 1 / 6
    return jac


def get_end_constraint_jacobian(controlPoints):
    numControlPoints = int(controlPoints.shape[0] / 2)
    jac = np.zeros((2, 2 * numControlPoints))
    jac[0, -6] = 1 / 6
    jac[0, -4] = 2 / 3
    jac[0, -2] = 1 / 6
    jac[1, -5] = 1 / 6
    jac[1, -3] = 2 / 3
    jac[1, -1] = 1 / 6
    return jac


def get_turn_rate_velocity_and_headings(
    controlPoints, knotPoints, numSamplesPerInterval
):
    out_d1 = evaluate_spline_derivative(
        controlPoints, knotPoints, 3, 1, numSamplesPerInterval
    )
    out_d2 = evaluate_spline_derivative(
        controlPoints, knotPoints, 3, 2, numSamplesPerInterval
    )
    v = np.linalg.norm(out_d1, axis=1)
    u = np.cross(out_d1, out_d2) / (v**2)
    heading = np.arctan2(out_d1[:, 1], out_d1[:, 0])
    return u, v, heading


dVelocityDControlPoints = jacfwd(get_spline_velocity)
dVelocityDtf = jacfwd(get_spline_velocity, argnums=1)


dTurnRateDControlPoints = jacfwd(get_spline_turn_rate)
dTurnRateTf = jacfwd(get_spline_turn_rate, argnums=1)

dCurvatureDControlPoints = jacfwd(get_spline_curvature)
dCurvatureDtf = jacfwd(get_spline_curvature, argnums=1)


def assure_velocity_constraint(
    controlPoints,
    knotPoints,
    num_control_points,
    agentSpeed,
    velocityBounds,
    numSamplesPerInterval,
):
    splineOrder = 3
    tf = np.linalg.norm(controlPoints[0] - controlPoints[-1]) / agentSpeed
    v = get_spline_velocity(controlPoints, tf, splineOrder, numSamplesPerInterval)
    while np.max(v) > velocityBounds[1]:
        tf += 0.01
        # combined_knot_points = self.create_unclamped_knot_points(0, tf, num_control_points,params.splineOrder)
        v = get_spline_velocity(controlPoints, tf, splineOrder, numSamplesPerInterval)
        # pd, u, v, pos = self.spline_constraints(radarList, controlPoints, combined_knot_points,params.numConstraintSamples)
    print("max v", np.max(v))
    return tf


def move_first_control_point_so_spline_passes_through_start(
    controlPoints, knotPoints, start, startVelocity
):
    num_control_points = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((num_control_points, 2))
    dt = knotPoints[3] - knotPoints[0]
    A = np.array(
        [
            [1 / 6, 0, 2 / 3, 0],
            [0, 1 / 6, 0, 2 / 3],
            [-3 / (2 * dt), 0, 0, 0],
            [0, -3 / (2 * dt), 0, 0],
        ]
    )
    c3x = controlPoints[2, 0]
    c3y = controlPoints[2, 1]

    b = np.array(
        [
            [start[0] - (1 / 6) * c3x],
            [start[1] - (1 / 6) * c3y],
            [startVelocity[0] - 3 / (2 * dt) * c3x],
            [startVelocity[1] - 3 / (2 * dt) * c3y],
        ]
    )

    x = np.linalg.solve(A, b)
    controlPoints[0:2, 0:2] = x.reshape((2, 2))

    return controlPoints


def move_last_control_point_so_spline_passes_through_end(
    controlPoints, knotPoints, end, endVelocity
):
    num_control_points = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((num_control_points, 2))
    dt = knotPoints[3] - knotPoints[0]
    A = np.array(
        [
            [2 / 3, 0, 1 / 6, 0],
            [0, 2 / 3, 0, 1 / 6],
            [0, 0, 3 / (2 * dt), 0],
            [0, 0, 0, 3 / (2 * dt)],
        ]
    )
    cn_minus_2_x = controlPoints[-3, 0]
    cn_minus_2_y = controlPoints[-3, 1]

    b = np.array(
        [
            [end[0] - (1 / 6) * cn_minus_2_x],
            [end[1] - (1 / 6) * cn_minus_2_y],
            [endVelocity[0] + 3 / (2 * dt) * cn_minus_2_x],
            [endVelocity[1] + 3 / (2 * dt) * cn_minus_2_y],
        ]
    )

    x = np.linalg.solve(A, b)
    controlPoints[-2:, -2:] = x.reshape((2, 2))

    return controlPoints
