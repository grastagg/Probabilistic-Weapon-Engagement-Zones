import os


from operator import inv
from jax.profiler import start_trace, stop_trace

import os
from jax.lax import le
import numpy as np
import chaospy as cp

import jax.numpy as jnp
import jax
import time

from scipy.stats import zmap


import CSPEZ.csbez as csbez

import CSPEZ.nueral_network_cspez as nueral_network_cspez


jax.config.update("jax_enable_x64", True)

# Vectorized function using vmap
in_dubins_engagement_zone = jax.jit(
    jax.vmap(
        csbez.in_dubins_engagement_zone_single,
        in_axes=(
            0,  # pursuerPosition
            0,  # pursuerHeading
            0,  # minimumTurnRadius
            None,  # catureRadius
            0,  # pursuerRange
            0,  # pursuerSpeed
            None,  # evaderPosition
            None,  # evaderHeading
            None,  # evaderSpeed
        ),  # Vectorizing over evaderPosition & evaderHeading
    )
)

# Vectorized function using vmap
in_dubins_engagement_zone3 = jax.jit(
    jax.vmap(
        csbez.in_dubins_engagement_zone_single,
        in_axes=(
            0,  # pursuerPosition
            0,  # pursuerHeading
            0,  # minimumTurnRadius
            None,  # catureRadius
            0,  # pursuerRange
            0,  # pursuerSpeed
            None,  # evaderPosition
            None,  # evaderHeading
            None,  # evaderSpeed
        ),  # Vectorizing over evaderPosition & evaderHeading
    )
)


def new_in_dubins_engagement_zone(
    pursuerPosition,
    pursuerHeading,
    pursuerTurnRadius,
    captureRadius,
    pursuerRange,
    pursuerSpeed,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
):
    return in_dubins_engagement_zone3(
        pursuerPosition,
        pursuerHeading,
        pursuerTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )


new_in_dubins_engagement_zone3 = jax.jit(
    jax.vmap(
        new_in_dubins_engagement_zone,
        in_axes=(None, None, None, None, None, None, 0, 0, None),
    )
)


def mc_dubins_pez_single(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    pursuerRange,
    captureRadius,
    numSamples,
):
    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    # var = jnp.var(ez)
    # mean = jnp.mean(ez)
    return (
        jnp.sum(ez <= 0) / numSamples,
        ez,
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
    )


def mc_dubins_pez_single_differentiable(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    pursuerRange,
    captureRadius,
    numSamples,
):
    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    epsilon = 0.1
    Zsmooth = jax.nn.sigmoid(-ez / epsilon)
    return (
        jnp.sum(Zsmooth) / numSamples,
        ez,
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
    )


def mc_dubins_PEZ_single_mu_derivative(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    pursuerRange,
    captureRadius,
    numSamples,
    pursuerMean,
    pursuerCov,
):
    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    mask = ez <= 0
    numCaptured = jnp.sum(mask)
    p_capture = numCaptured / numSamples

    combinedSamples = jnp.stack(
        [
            pursuerPosition[..., 0],  # x
            pursuerPosition[..., 1],  # y
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ],
        axis=-1,
    )

    # Instead of combinedSamples[mask], use masking math:
    masked_combined = jnp.where(mask[:, None], combinedSamples, 0.0)
    masked_shift = masked_combined - pursuerMean  # (N, d)

    sum_shift = jnp.sum(masked_shift, axis=0)
    mean_shift = jnp.where(numCaptured > 0, sum_shift / numCaptured, 0.0)
    # mean_shift = sum_shift / numCaptured

    inv_Sigma = jnp.linalg.inv(pursuerCov)
    gradient = inv_Sigma @ (p_capture * mean_shift)

    return gradient


mc_dubins_pez = jax.jit(
    jax.vmap(
        mc_dubins_pez_single,
        in_axes=(0, 0, None, None, None, None, None, None, None, None),
    )
)
mc_dubins_pez_differentiable = jax.jit(
    jax.vmap(
        mc_dubins_pez_single_differentiable,
        in_axes=(0, 0, None, None, None, None, None, None, None, None),
    )
)


def generate_random_key():
    seed = np.random.randint(100000)
    key = jax.random.PRNGKey(seed)
    return jax.random.split(key)


def mc_dubins_PEZ(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    # start= time.time()
    numSamples = 10000

    key, subkey = generate_random_key()
    # Generate heading samples
    pursuerHeadingSamples = pursuerHeading + jnp.sqrt(
        pursuerHeadingVar
    ) * jax.random.normal(subkey, shape=(numSamples,))

    key, subkey = generate_random_key()

    # generate turn radius samples
    minimumTurnRadiusSamples = minimumTurnRadius + jnp.sqrt(
        minimumTurnRadiusVar
    ) * jax.random.normal(subkey, shape=(numSamples,))

    key, subkey = generate_random_key()
    # generate speed samples
    pursuerSpeedSamples = pursuerSpeed + jnp.sqrt(pursuerSpeedVar) * jax.random.normal(
        subkey, shape=(numSamples,)
    )

    key, subkey = generate_random_key()

    # generate position samples
    pursuerPositionSamples = jax.random.multivariate_normal(
        key, pursuerPosition, pursuerPositionCov, shape=(numSamples,)
    )

    key, subkey = generate_random_key()

    pursuerRangeSamples = pursuerRange + jnp.sqrt(pursuerRangeVar) * jax.random.normal(
        subkey, shape=(numSamples,)
    )

    # print("time taken to generate samples: ", time.time() - start)
    return mc_dubins_pez(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        pursuerPositionSamples,
        pursuerHeadingSamples,
        pursuerSpeedSamples,
        minimumTurnRadiusSamples,
        pursuerRangeSamples,
        captureRadius,
        numSamples,
    )


def mc_dubins_PEZ_Single(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    numSamples = 10000

    key, subkey = generate_random_key()
    # Generate heading samples
    pursuerHeadingSamples = pursuerHeading + jnp.sqrt(
        pursuerHeadingVar
    ) * jax.random.normal(subkey, shape=(numSamples,))

    key, subkey = generate_random_key()

    # generate turn radius samples
    minimumTurnRadiusSamples = minimumTurnRadius + jnp.sqrt(
        minimumTurnRadiusVar
    ) * jax.random.normal(subkey, shape=(numSamples,))

    key, subkey = generate_random_key()
    # generate speed samples
    pursuerSpeedSamples = pursuerSpeed + jnp.sqrt(pursuerSpeedVar) * jax.random.normal(
        subkey, shape=(numSamples,)
    )

    key, subkey = generate_random_key()

    # generate position samples
    pursuerPositionSamples = jax.random.multivariate_normal(
        key, pursuerPosition, pursuerPositionCov, shape=(numSamples,)
    )

    key, subkey = generate_random_key()

    pursuerRangeSamples = pursuerRange + jnp.sqrt(pursuerRangeVar) * jax.random.normal(
        subkey, shape=(numSamples,)
    )

    return mc_dubins_pez_single(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        pursuerPositionSamples,
        pursuerHeadingSamples,
        pursuerSpeedSamples,
        minimumTurnRadiusSamples,
        pursuerRangeSamples,
        captureRadius,
        numSamples,
    )


def mc_dubins_PEZ_differentiable(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    numSamples = 10000

    key, subkey = generate_random_key()
    # Generate heading samples
    pursuerHeadingSamples = pursuerHeading + jnp.sqrt(
        pursuerHeadingVar
    ) * jax.random.normal(subkey, shape=(numSamples,))

    key, subkey = generate_random_key()

    # generate turn radius samples
    minimumTurnRadiusSamples = minimumTurnRadius + jnp.sqrt(
        minimumTurnRadiusVar
    ) * jax.random.normal(subkey, shape=(numSamples,))

    key, subkey = generate_random_key()
    # generate speed samples
    pursuerSpeedSamples = pursuerSpeed + jnp.sqrt(pursuerSpeedVar) * jax.random.normal(
        subkey, shape=(numSamples,)
    )

    key, subkey = generate_random_key()

    # generate position samples
    pursuerPositionSamples = jax.random.multivariate_normal(
        key, pursuerPosition, pursuerPositionCov, shape=(numSamples,)
    )

    key, subkey = generate_random_key()

    pursuerRangeSamples = pursuerRange + jnp.sqrt(pursuerRangeVar) * jax.random.normal(
        subkey, shape=(numSamples,)
    )

    return mc_dubins_pez_differentiable(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        pursuerPositionSamples,
        pursuerHeadingSamples,
        pursuerSpeedSamples,
        minimumTurnRadiusSamples,
        pursuerRangeSamples,
        captureRadius,
        numSamples,
    )


def dubins_EZ_single_combined_input(pursuerParams, evaderParams):
    pursuerPosition = pursuerParams[:2]
    pursuerHeading = pursuerParams[2]
    pursuerSpeed = pursuerParams[3]
    minimumTurnRadius = pursuerParams[4]
    pursuerRange = pursuerParams[5]
    captureRadius = pursuerParams[6]
    evaderPosition = evaderParams[:2]
    evaderHeading = evaderParams[2]
    evaderSpeed = evaderParams[3]
    return csbez.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )


# find gradients/jacobians of dubins BEZ function
dDubinsEZ_dPursuerPosition = jax.jacfwd(csbez.in_dubins_engagement_zone_single, 0)
dDubinsEZ_dPursuerHeading = jax.jacfwd(csbez.in_dubins_engagement_zone_single, 1)
dDubinsEZ_dMinimumTurnRadius = jax.jacfwd(csbez.in_dubins_engagement_zone_single, 2)
dDubinsEZ_dCaptureRadius = jax.jacfwd(csbez.in_dubins_engagement_zone_single, 3)
dDubinsEZ_dPursuerRange = jax.jacfwd(csbez.in_dubins_engagement_zone_single, 4)
dDubinsEZ_dPursuerSpeed = jax.jacfwd(csbez.in_dubins_engagement_zone_single, 5)
dDubinsEZ_dEvaderPosition = jax.jacfwd(csbez.in_dubins_engagement_zone_single, 6)


# first order combined derivatives
dDubinsEZ_dPursuerParams = jax.jacfwd(dubins_EZ_single_combined_input, 0)

# sevond order combined derivatives
d2DubinsEZ_dPursuerParams = jax.jacfwd(dDubinsEZ_dPursuerParams, 0)

# # third order combined derivatives
# d3DubinsEZ_dPursuerParams = jax.jacfwd(d2DubinsEZ_dPursuerParams, 0)


def linear_dubins_PEZ_single(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    dDubinsEZ_dPursuerPositionValue = dDubinsEZ_dPursuerPosition(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    dDubinsEZ_dPursuerHeadingValue = dDubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    dDubinsEZ_dMinimumTurnRadiusValue = dDubinsEZ_dMinimumTurnRadius(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    dDubinsEZ_dPursuerRangeValue = dDubinsEZ_dPursuerRange(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    dDubinsEZ_dPursuerSpeedValue = dDubinsEZ_dPursuerSpeed(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    var = (
        dDubinsEZ_dPursuerPositionValue
        @ pursuerPositionCov
        @ dDubinsEZ_dPursuerPositionValue.T
        + dDubinsEZ_dPursuerHeadingValue**2 * pursuerHeadingVar
        + dDubinsEZ_dMinimumTurnRadiusValue**2 * minimumTurnRadiusVar
        + dDubinsEZ_dPursuerRangeValue**2 * pursuerRangeVar
        + dDubinsEZ_dPursuerSpeedValue**2 * pursuerSpeedVar
    )
    mean = csbez.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )

    return (
        jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var)),
        mean,
        var,
    )


linear_dubins_pez = jax.jit(
    jax.vmap(
        linear_dubins_PEZ_single,
        in_axes=(
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
            None,
            None,
        ),
    )
)


def stacked_cov(
    pursuerPositionCov,
    pursuerHeadingVar,
    pursuerSpeedVar,
    minimumTurnRadiusVar,
    pursuerRangeVar,
):
    heading_block = jnp.array([[pursuerHeadingVar]])
    speed_block = jnp.array([[pursuerSpeedVar]])
    radius_block = jnp.array([[minimumTurnRadiusVar]])
    range_block = jnp.array([[pursuerRangeVar]])

    # Assemble full covariance using jnp.block (block matrix layout)
    full_cov = jnp.block(
        [
            [
                pursuerPositionCov,
                jnp.zeros((2, 1)),
                jnp.zeros((2, 1)),
                jnp.zeros((2, 1)),
                jnp.zeros((2, 1)),
            ],
            [
                jnp.zeros((1, 2)),
                heading_block,
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
            ],
            [
                jnp.zeros((1, 2)),
                jnp.zeros((1, 1)),
                speed_block,
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
            ],
            [
                jnp.zeros((1, 2)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                radius_block,
                jnp.zeros((1, 1)),
            ],
            [
                jnp.zeros((1, 2)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
                range_block,
            ],
        ]
    )
    return full_cov


def closest_point_on_circle(point, center, radius):
    return center + 1.01 * radius * (point - center) / jnp.linalg.norm(point - center)


def quadratic_dubins_PEZ_single(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    pursuerParams = jnp.concatenate(
        [
            pursuerPosition,  # (2,)
            jnp.array([pursuerHeading]),  # (1,)
            jnp.array([pursuerSpeed]),  # (1,)
            jnp.array([minimumTurnRadius]),  # (1,)
            jnp.array([pursuerRange]),  # (1,)
        ]
    )
    evaderParams = jnp.concatenate(
        [evaderPositions, jnp.array([evaderHeadings]), jnp.array([evaderSpeed])]
    )
    val = csbez.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    dDubinsEZ_dPursuerParamsValue = dDubinsEZ_dPursuerParams(
        pursuerParams, evaderParams
    )
    d2DubinsEZ_dPursuerParamsValue = d2DubinsEZ_dPursuerParams(
        pursuerParams, evaderParams
    )
    full_cov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        pursuerSpeedVar,
        minimumTurnRadiusVar,
        pursuerRangeVar,
    )

    mean = val + 0.5 * jnp.trace(d2DubinsEZ_dPursuerParamsValue @ full_cov)
    var = (
        dDubinsEZ_dPursuerParamsValue @ full_cov @ dDubinsEZ_dPursuerParamsValue.T
        + 0.5
        * jnp.trace(
            d2DubinsEZ_dPursuerParamsValue
            @ full_cov
            @ d2DubinsEZ_dPursuerParamsValue
            @ full_cov
        )
    )
    return (
        jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var)),
        mean,
        var,
    )


quadratic_dubins_pez = jax.jit(
    jax.vmap(
        quadratic_dubins_PEZ_single,
        in_axes=(
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
            None,
            None,
        ),
    )
)


def quadratic_dubins_PEZ_full_cov_single(
    evaderPositions, evaderHeadings, evaderSpeed, pursuerParams, full_cov
):
    evaderParams = jnp.concatenate(
        [evaderPositions, jnp.array([evaderHeadings]), jnp.array([evaderSpeed])]
    )
    val = dubins_EZ_single_combined_input(pursuerParams, evaderParams)
    dDubinsEZ_dPursuerParamsValue = dDubinsEZ_dPursuerParams(
        pursuerParams, evaderParams
    )
    d2DubinsEZ_dPursuerParamsValue = d2DubinsEZ_dPursuerParams(
        pursuerParams, evaderParams
    )

    mean = val + 0.5 * jnp.trace(d2DubinsEZ_dPursuerParamsValue @ full_cov)
    var = (
        dDubinsEZ_dPursuerParamsValue @ full_cov @ dDubinsEZ_dPursuerParamsValue.T
        + 0.5
        * jnp.trace(
            d2DubinsEZ_dPursuerParamsValue
            @ full_cov
            @ d2DubinsEZ_dPursuerParamsValue
            @ full_cov
        )
    )
    return jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var))


quadratic_dubins_pez_full_cov = jax.jit(
    jax.vmap(
        quadratic_dubins_PEZ_full_cov_single,
        in_axes=(
            0,
            0,
            None,
            None,
            None,
        ),
    )
)


def cubic_dubins_PEZ_single(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    pursuerParams = jnp.concatenate(
        [
            pursuerPosition,  # (2,)
            jnp.array([pursuerHeading]),  # (1,)
            jnp.array([pursuerSpeed]),  # (1,)
            jnp.array([minimumTurnRadius]),  # (1,)
            jnp.array([pursuerRange]),  # (1,)
        ]
    )
    evaderParams = jnp.concatenate(
        [evaderPositions, jnp.array([evaderHeadings]), jnp.array([evaderSpeed])]
    )
    val = csbez.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    dDubinsEZ_dPursuerParamsValue = dDubinsEZ_dPursuerParams(
        pursuerParams, evaderParams
    )
    d2DubinsEZ_dPursuerParamsValue = d2DubinsEZ_dPursuerParams(
        pursuerParams, evaderParams
    )
    d3DubinsEZ_dPursuerParamsValue = d3DubinsEZ_dPursuerParams(
        pursuerParams, evaderParams
    )
    full_cov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        pursuerSpeedVar,
        minimumTurnRadiusVar,
        pursuerRangeVar,
    )

    mean, var = taylor3_mean_var_scalar_output(
        val,
        dDubinsEZ_dPursuerParamsValue,
        d2DubinsEZ_dPursuerParamsValue,
        d3DubinsEZ_dPursuerParamsValue,
        full_cov,
    )
    return (jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var)), mean, var)


# cubic_dubins_PEZ = jax.jit(
#     jax.vmap(
#         cubic_dubins_PEZ_single,
#         in_axes=(
#             0,
#             0,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#         ),
#     )
# )


def symmetric_sqrt(mat):
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = jnp.clip(eigvals, a_min=0.0)
    sqrt_eigvals = jnp.sqrt(eigvals)
    return eigvecs @ (sqrt_eigvals[..., None] * eigvecs.T)


@jax.jit
def generate_sigma_points(mean, cov, alpha=1e-3, beta=2.0, kappa=0.0):
    n = mean.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n

    # Compute square root of scaled covariance
    L = symmetric_sqrt((n + lambda_) * cov)

    # Compute sigma points
    offsets = L.T
    sigma_points = jnp.vstack([mean, mean + offsets, mean - offsets])

    # Compute weights
    weight = 1.0 / (2 * (n + lambda_))
    weights = jnp.full(2 * n, weight)
    weights_mean = jnp.concatenate([jnp.array([lambda_ / (n + lambda_)]), weights])
    weights_cov = jnp.concatenate(
        [jnp.array([lambda_ / (n + lambda_) + (1 - alpha**2 + beta)]), weights]
    )

    return sigma_points, weights_mean, weights_cov


def uncented_dubins_PEZ_single(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    # Construct covariance matrix
    cov = jax.scipy.linalg.block_diag(
        pursuerPositionCov,
        jnp.array([[pursuerHeadingVar]]),
        jnp.array([[pursuerSpeedVar]]),
        jnp.array([[minimumTurnRadiusVar]]),
        jnp.array([[pursuerRangeVar]]),
    )

    # Construct mean vector
    mean = jnp.array(
        [
            pursuerPosition[0],
            pursuerPosition[1],
            pursuerHeading,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
        ]
    )

    # Generate sigma points
    sigma_points, weights_mean, weights_cov = generate_sigma_points(mean, cov)

    # Extract individual sigma point components
    sigmaPointsPursuerPosition = sigma_points[:, :2]
    sigmaPointsPursuerHeading = sigma_points[:, 2]
    sigmaPointsPursuerSpeed = sigma_points[:, 3]
    sigmaPointsMinimumTurnRadius = sigma_points[:, 4]
    sigmaPointsPursuerRange = sigma_points[:, 5]

    # Compute transformed sigma points
    transformedSigmaPoints = in_dubins_engagement_zone(
        sigmaPointsPursuerPosition,
        sigmaPointsPursuerHeading,
        sigmaPointsMinimumTurnRadius,
        captureRadius,
        sigmaPointsPursuerRange,
        sigmaPointsPursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )

    # Compute mean and variance
    ezMean = jnp.dot(weights_mean, transformedSigmaPoints)
    ezVar = jnp.dot(weights_cov, (transformedSigmaPoints - ezMean) ** 2)

    return jax.scipy.stats.norm.cdf(0, ezMean, jnp.sqrt(ezVar)), ezMean, ezVar


uncented_dubins_pez = jax.jit(
    jax.vmap(
        uncented_dubins_PEZ_single,
        in_axes=(
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
            None,
            None,
        ),
    )
)


@jax.jit
def circle_intersection(c1, c2, r1, r2):
    x1, y1 = c1
    x2, y2 = c2

    # Distance between centers
    d = jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check for no intersection or identical circles
    no_intersection = (d > r1 + r2) | (d < jnp.abs(r1 - r2)) | (d == 0)

    def no_intersect_case():
        return jnp.full((2, 2), jnp.nan)  # Return NaN-filled array of shape (2,2)

    def intersect_case():
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = jnp.sqrt(r1**2 - a**2)

        xm = x1 + a * (x2 - x1) / d
        ym = y1 + a * (y2 - y1) / d

        x3_1 = xm + h * (y2 - y1) / d
        y3_1 = ym - h * (x2 - x1) / d
        x3_2 = xm - h * (y2 - y1) / d
        y3_2 = ym + h * (x2 - x1) / d

        return jnp.array([[x3_1, y3_1], [x3_2, y3_2]])

    return jax.lax.cond(no_intersection, no_intersect_case, intersect_case)


def create_linear_model_heading(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    minimumTurnRadius,
    pursuerRange,
    captureRadius,
):
    dDubinsEZ_dPursuerHeadingValue = dDubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    val = csbez.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    b = val - dDubinsEZ_dPursuerHeadingValue * pursuerHeading
    a = dDubinsEZ_dPursuerHeadingValue
    return a, b


def piecewise_linear_cdf_single_y(y, mu, sigma, breakpoints, slopes, intercepts):
    c = jnp.concatenate([jnp.array([-jnp.inf]), breakpoints, jnp.array([jnp.inf])])
    n_segments = slopes.shape[0]

    def body(i, total):
        a = slopes[i]
        b = intercepts[i]
        left = c[i]
        right = c[i + 1]

        def when_nonzero_slope(_):
            x_star = (y - b) / a
            interval_left = jnp.where(a > 0, left, x_star)
            interval_right = jnp.where(a > 0, x_star, right)
            interval_left = jnp.clip(interval_left, left, right)
            interval_right = jnp.clip(interval_right, left, right)
            contribution = jnp.maximum(
                0.0,
                jax.scipy.stats.norm.cdf((interval_right - mu) / sigma)
                - jax.scipy.stats.norm.cdf((interval_left - mu) / sigma),
            )
            return contribution

        def when_zero_slope(_):
            return jnp.where(
                y >= b,
                jax.scipy.stats.norm.cdf((right - mu) / sigma)
                - jax.scipy.stats.norm.cdf((left - mu) / sigma),
                0.0,
            )

        contribution = jax.lax.cond(
            a == 0.0, when_zero_slope, when_nonzero_slope, operand=None
        )
        return total + contribution

    return jax.lax.fori_loop(0, n_segments, body, 0.0)


def piecewise_linear_dubins_PEZ_single_heading_only(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    # crete evenly space bounding angles including min and max angles
    minAngle = pursuerHeading - 3.0 * jnp.sqrt(pursuerHeadingVar)
    maxAngle = pursuerHeading + 3.0 * jnp.sqrt(pursuerHeadingVar)
    numAngles = 51
    boundingAngles = jnp.linspace(minAngle, maxAngle, numAngles)

    linearizationPoints = (boundingAngles[:-1] + boundingAngles[1:]) / 2.0

    # slopes = []
    # intercepts = []
    # bounds = []

    # Vectorize the function over angle and index
    def linear_model_with_bounds(angle, i):
        a, b = create_linear_model_heading(
            evaderPositions,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            angle,
            pursuerSpeed,
            minimumTurnRadius,
            pursuerRange,
            captureRadius,
        )
        bound = (boundingAngles[i], boundingAngles[i + 1])
        return a, b

    # vmap over the indices of linearizationPoints
    # indices = jnp.arange(len(linearizationPoints))
    indices = jnp.arange(len(linearizationPoints))
    slopes, intercepts = jax.vmap(linear_model_with_bounds, in_axes=(0, 0))(
        linearizationPoints, indices
    )

    prob = piecewise_linear_cdf_single_y(
        0.0,
        pursuerHeading,
        jnp.sqrt(pursuerHeadingVar),
        boundingAngles[1:],
        slopes,
        intercepts,
    )
    return prob, slopes, intercepts, boundingAngles


# piecewise_linear_dubins_pez_heading_only = jax.jit(
#     jax.vmap(
#         piecewise_linear_dubins_PEZ_single_heading_only,
#         in_axes=(
#             0,
#             0,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#         ),
#     )
# )


def compute_region_weight(
    pursuerHeadingBounds,
    pursuerSpeedBounds,
    pursuerHeadingIndex,
    pursuerSpeedIndex,
    pursuerHeadingMean,
    pursuerHeadingVar,
    pursuerSpeedMean,
    pursuerSpeedVar,
):
    pursuerSpeedstd = jnp.sqrt(pursuerSpeedVar)
    pursuerHeadingstd = jnp.sqrt(pursuerHeadingVar)
    i = pursuerHeadingIndex
    j = pursuerSpeedIndex
    pursuerSpeedNormalizedLowerBound = (
        pursuerSpeedBounds[j] - pursuerSpeedMean
    ) / pursuerSpeedstd
    pursuerSpeedNormalizedUpperBound = (
        pursuerSpeedBounds[j + 1] - pursuerSpeedMean
    ) / pursuerSpeedstd
    pursuerHeadingNormalizedLowerBound = (
        pursuerHeadingBounds[i] - pursuerHeadingMean
    ) / pursuerHeadingstd
    pursuerHeadingNormalizedUpperBound = (
        pursuerHeadingBounds[i + 1] - pursuerHeadingMean
    ) / pursuerHeadingstd
    # Compute the probability of the region
    # This is the product of the two independent normal distributions
    return (
        jax.scipy.stats.norm.cdf(pursuerSpeedNormalizedUpperBound)
        - jax.scipy.stats.norm.cdf(pursuerSpeedNormalizedLowerBound)
    ) * (
        jax.scipy.stats.norm.cdf(pursuerHeadingNormalizedUpperBound)
        - jax.scipy.stats.norm.cdf(pursuerHeadingNormalizedLowerBound)
    )


# vectorize compute_region_weight
compute_region_weight_vmap = jax.vmap(
    compute_region_weight, in_axes=(None, None, 0, 0, None, None, None, None)
)


def piecewise_linear_dubins_PEZ_single_heading_and_speed_gmm(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerHeadingVar,
    boundingPursuerHeading,
    linearizationPursuerHeadings,
    pursuerHeadingIndices,
    pursuerSpeed,
    pursuerSpeedVar,
    boundingPursuerSpeed,
    linearizationPursuerSpeeds,
    pursuerSpeedIndices,
    minimumTurnRadius,
    pursuerRange,
    captureRadius,
    weights,
):
    pursuerHeadingSlopes, pursuerSpeedSlopes, intercepts = (
        create_linear_model_heading_and_speed_vmap(
            evaderPosition,
            evaderHeading,
            evaderSpeed,
            pursuerPosition,
            linearizationPursuerHeadings,
            linearizationPursuerSpeeds,
            minimumTurnRadius,
            pursuerRange,
            captureRadius,
            pursuerHeadingIndices,
            pursuerSpeedIndices,
        )
    )

    # mean of ez function
    mean = (
        pursuerHeadingSlopes * pursuerHeading
        + pursuerSpeedSlopes * pursuerSpeed
        + intercepts
    )
    # variance of ez function
    var = (
        pursuerHeadingSlopes**2 * pursuerHeadingVar
        + pursuerSpeedSlopes**2 * pursuerSpeedVar
    )
    # cdf of ez function
    probs = jax.scipy.stats.norm.cdf(0.0 - mean / jnp.sqrt(var))
    return jnp.sum(probs * weights), 0, 0, 0


piecewise_linear_dubins_PEZ_single_heading_and_speed_gmm_vmap = jax.vmap(
    piecewise_linear_dubins_PEZ_single_heading_and_speed_gmm,
    in_axes=(
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
        None,
        None,
        None,
        None,
        None,
        None,
    ),
)


@jax.jit
def piecewise_linear_dubins_pez_heading_and_speed(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    numBoundingPoints = 21
    maxPursuerHeading = pursuerHeading + 3 * jnp.sqrt(pursuerHeadingVar)
    minPursuerHeading = pursuerHeading - 3 * jnp.sqrt(pursuerHeadingVar)
    maxPursuerSpeed = pursuerSpeed + 3 * jnp.sqrt(pursuerSpeedVar)
    minPursuerSpeed = pursuerSpeed - 3 * jnp.sqrt(pursuerSpeedVar)

    boundingPursuerHeading = jnp.linspace(
        minPursuerHeading, maxPursuerHeading, numBoundingPoints
    )
    boundingPursuerSpeed = jnp.linspace(
        minPursuerSpeed, maxPursuerSpeed, numBoundingPoints
    )
    linearizationPursuerHeadings = (
        boundingPursuerHeading[:-1] + boundingPursuerHeading[1:]
    ) / 2.0
    linearizationPursuerSpeeds = (
        boundingPursuerSpeed[:-1] + boundingPursuerSpeed[1:]
    ) / 2.0

    # create meash grid of indices
    pursuerHeadingIndices = jnp.arange(len(linearizationPursuerHeadings))
    pursuerSpeedIndices = jnp.arange(len(linearizationPursuerSpeeds))
    pursuerHeadingIndices, pursuerSpeedIndices = jnp.meshgrid(
        pursuerHeadingIndices, pursuerSpeedIndices
    )
    pursuerHeadingIndices = pursuerHeadingIndices.ravel()
    pursuerSpeedIndices = pursuerSpeedIndices.ravel()
    weights = compute_region_weight_vmap(
        boundingPursuerHeading,
        boundingPursuerSpeed,
        pursuerHeadingIndices,
        pursuerSpeedIndices,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
    )
    prob, slopes, intercepts, bounds = (
        piecewise_linear_dubins_PEZ_single_heading_and_speed_gmm_vmap(
            evaderPositions,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerHeading,
            pursuerHeadingVar,
            boundingPursuerHeading,
            linearizationPursuerHeadings,
            pursuerHeadingIndices,
            pursuerSpeed,
            pursuerSpeedVar,
            boundingPursuerSpeed,
            linearizationPursuerSpeeds,
            pursuerSpeedIndices,
            minimumTurnRadius,
            pursuerRange,
            captureRadius,
            weights,
        )
    )
    return prob, slopes, intercepts, bounds


@jax.jit
def create_linear_model_heading_and_speed(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerHeadings,
    pursuerSpeeds,
    minimumTurnRadius,
    pursuerRange,
    captureRadius,
    pursuerHeadingIndex,
    pursuerSpeedIndex,
):
    pursuerHeading = pursuerHeadings[pursuerHeadingIndex]
    pursuerSpeed = pursuerSpeeds[pursuerSpeedIndex]
    dDubinsEZ_dPursuerHeadingValue = dDubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    dDubinsEZ_dPursuerSpeedValue = dDubinsEZ_dPursuerSpeed(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    val = csbez.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )
    c = (
        val
        - dDubinsEZ_dPursuerHeadingValue * pursuerHeading
        - dDubinsEZ_dPursuerSpeedValue * pursuerSpeed
    )
    a = dDubinsEZ_dPursuerHeadingValue
    b = dDubinsEZ_dPursuerSpeedValue
    return a, b, c


# vectorize for heading and speed
create_linear_model_heading_and_speed_vmap = jax.vmap(
    create_linear_model_heading_and_speed,
    in_axes=(None, None, None, None, None, None, None, None, None, 0, 0),
)


def insert_n_between_elements(arr, n):
    # Create a (m-1) x (n+2) array where each row is a linspace between consecutive elements
    expanded = jnp.linspace(arr[:-1], arr[1:], n + 2, axis=1)

    # Flatten and remove repeated values at segment boundaries
    result = expanded[:, :-1].ravel()

    # Append the last element of the original array
    return jnp.append(result, arr[-1])


def norm_logpdf(x, mean=0.0, std=1.0):
    var = std**2
    return -0.5 * (jnp.log(2 * jnp.pi * var) + ((x - mean) ** 2) / var)


def independent_gaussian_logpdf(x, mean, std):
    return jnp.sum(norm_logpdf(x, mean, std), axis=-1)


def independent_gaussian_pdf(x, mean, std):
    return jnp.exp(independent_gaussian_logpdf(x, mean, std))


def compute_average_pdf_single(
    pursuerPosition,
    pursuerPositionCov,
    pursuerPositionGridX,
    pursuerPositionGridY,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerHeadingGrid,
    pursuerSpeed,
    pursuerSpeedVar,
    pursuerSpeedGrid,
    pursuerTurnRadius,
    pursuerTurnRadiusVar,
    pursuerTurnRadiusGrid,
    pursuerPositionXIndex,
    pursuerPositionYIndex,
    pursuerHeadingIndex,
    pursuerSpeedIndex,
    pursuerTurnRadiusIndex,
):
    # Standard deviations
    heading_std = jnp.sqrt(pursuerHeadingVar)
    speed_std = jnp.sqrt(pursuerSpeedVar)
    pursuerTurnRadius_std = jnp.sqrt(pursuerTurnRadiusVar)

    # Grid corners

    x0 = pursuerPositionGridX[pursuerPositionXIndex]
    x1 = pursuerPositionGridX[pursuerPositionXIndex + 1]
    y0 = pursuerPositionGridY[pursuerPositionYIndex]
    y1 = pursuerPositionGridY[pursuerPositionYIndex + 1]
    h0 = pursuerHeadingGrid[pursuerHeadingIndex]
    h1 = pursuerHeadingGrid[pursuerHeadingIndex + 1]
    s0 = pursuerSpeedGrid[pursuerSpeedIndex]
    s1 = pursuerSpeedGrid[pursuerSpeedIndex + 1]
    r0 = pursuerTurnRadiusGrid[pursuerTurnRadiusIndex]
    r1 = pursuerTurnRadiusGrid[pursuerTurnRadiusIndex + 1]

    # Evaluate 1D PDFs at each coordinate
    xy_pdf_00 = jax.scipy.stats.multivariate_normal.pdf(
        jnp.array([x0, y0]), pursuerPosition, pursuerPositionCov
    )
    xy_pdf_01 = jax.scipy.stats.multivariate_normal.pdf(
        jnp.array([x0, y1]), pursuerPosition, pursuerPositionCov
    )
    xy_pdf_10 = jax.scipy.stats.multivariate_normal.pdf(
        jnp.array([x1, y0]), pursuerPosition, pursuerPositionCov
    )
    xy_pdf_11 = jax.scipy.stats.multivariate_normal.pdf(
        jnp.array([x1, y1]), pursuerPosition, pursuerPositionCov
    )

    h_pdf_0 = jax.scipy.stats.norm.pdf(h0, pursuerHeading, heading_std)
    h_pdf_1 = jax.scipy.stats.norm.pdf(h1, pursuerHeading, heading_std)
    s_pdf_0 = jax.scipy.stats.norm.pdf(s0, pursuerSpeed, speed_std)
    s_pdf_1 = jax.scipy.stats.norm.pdf(s1, pursuerSpeed, speed_std)
    r_pdf_0 = jax.scipy.stats.norm.pdf(r0, pursuerTurnRadius, pursuerTurnRadius_std)
    r_pdf_1 = jax.scipy.stats.norm.pdf(r1, pursuerTurnRadius, pursuerTurnRadius_std)

    # Compute 2D PDF at 32 corners
    p00000 = xy_pdf_00 * h_pdf_0 * s_pdf_0 * r_pdf_0  # (x0,y0,h0, s0, r0)
    p00001 = xy_pdf_00 * h_pdf_0 * s_pdf_0 * r_pdf_1  # (x0,y0,h0, s0, r1)
    p00010 = xy_pdf_00 * h_pdf_0 * s_pdf_1 * r_pdf_0  # (x0,y0,h0, s1, r0)
    p00011 = xy_pdf_00 * h_pdf_0 * s_pdf_1 * r_pdf_1  # (x0,y0,h0, s1, r1)
    p00100 = xy_pdf_00 * h_pdf_1 * s_pdf_0 * r_pdf_0  # (x0,y0,h1, s0, r0)
    p00101 = xy_pdf_00 * h_pdf_1 * s_pdf_0 * r_pdf_1  # (x0,y0,h1, s0, r1)
    p00110 = xy_pdf_00 * h_pdf_1 * s_pdf_1 * r_pdf_0  # (x0,y0,h1, s1, r0)
    p00111 = xy_pdf_00 * h_pdf_1 * s_pdf_1 * r_pdf_1  # (x0,y0,h1, s1, r1)
    p01000 = xy_pdf_01 * h_pdf_0 * s_pdf_0 * r_pdf_0  # (x0,y1,h0, s0, r0)
    p01001 = xy_pdf_01 * h_pdf_0 * s_pdf_0 * r_pdf_1  # (x0,y1,h0, s0, r1)
    p01010 = xy_pdf_01 * h_pdf_0 * s_pdf_1 * r_pdf_0  # (x0,y1,h0, s1, r0)
    p01011 = xy_pdf_01 * h_pdf_0 * s_pdf_1 * r_pdf_1  # (x0,y1,h0, s1, r1)
    p01100 = xy_pdf_01 * h_pdf_1 * s_pdf_0 * r_pdf_0  # (x0,y1,h1, s0, r0)
    p01101 = xy_pdf_01 * h_pdf_1 * s_pdf_0 * r_pdf_1  # (x0,y1,h1, s0, r1)
    p01110 = xy_pdf_01 * h_pdf_1 * s_pdf_1 * r_pdf_0  # (x0,y1,h1, s1, r0)
    p01111 = xy_pdf_01 * h_pdf_1 * s_pdf_1 * r_pdf_1  # (x0,y1,h1, s1, r1)
    p10000 = xy_pdf_10 * h_pdf_0 * s_pdf_0 * r_pdf_0  # (x1,y0,h0, s0, r0)
    p10001 = xy_pdf_10 * h_pdf_0 * s_pdf_0 * r_pdf_1  # (x1,y0,h0, s0, r1)
    p10010 = xy_pdf_10 * h_pdf_0 * s_pdf_1 * r_pdf_0  # (x1,y0,h0, s1, r0)
    p10011 = xy_pdf_10 * h_pdf_0 * s_pdf_1 * r_pdf_1  # (x1,y0,h0, s1, r1)
    p10100 = xy_pdf_10 * h_pdf_1 * s_pdf_0 * r_pdf_0  # (x1,y0,h1, s0, r0)
    p10101 = xy_pdf_10 * h_pdf_1 * s_pdf_0 * r_pdf_1  # (x1,y0,h1, s0, r1)
    p10110 = xy_pdf_10 * h_pdf_1 * s_pdf_1 * r_pdf_0  # (x1,y0,h1, s1, r0)
    p10111 = xy_pdf_10 * h_pdf_1 * s_pdf_1 * r_pdf_1  # (x1,y0,h1, s1, r1)
    p11000 = xy_pdf_11 * h_pdf_0 * s_pdf_0 * r_pdf_0  # (x1,y1,h0, s0, r0)
    p11001 = xy_pdf_11 * h_pdf_0 * s_pdf_0 * r_pdf_1  # (x1,y1,h0, s0, r1)
    p11010 = xy_pdf_11 * h_pdf_0 * s_pdf_1 * r_pdf_0  # (x1,y1,h0, s1, r0)
    p11011 = xy_pdf_11 * h_pdf_0 * s_pdf_1 * r_pdf_1  # (x1,y1,h0, s1, r1)
    p11100 = xy_pdf_11 * h_pdf_1 * s_pdf_0 * r_pdf_0  # (x1,y1,h1, s0, r0)
    p11101 = xy_pdf_11 * h_pdf_1 * s_pdf_0 * r_pdf_1  # (x1,y1,h1, s0, r1)
    p11110 = xy_pdf_11 * h_pdf_1 * s_pdf_1 * r_pdf_0  # (x1,y1,h1, s1, r0)
    p11111 = xy_pdf_11 * h_pdf_1 * s_pdf_1 * r_pdf_1  # (x1,y1,h1, s1, r1)

    cellVolume = (x1 - x0) * (y1 - y0) * (h1 - h0) * (s1 - s0) * (r1 - r0)

    # Average PDF over the cell
    average_pdf = (
        p00000
        + p00001
        + p00010
        + p00011
        + p00100
        + p00101
        + p00110
        + p00111
        + p01000
        + p01001
        + p01010
        + p01011
        + p01100
        + p01101
        + p01110
        + p01111
        + p10000
        + p10001
        + p10010
        + p10011
        + p10100
        + p10101
        + p10110
        + p10111
        + p11000
        + p11001
        + p11010
        + p11011
        + p11100
        + p11101
        + p11110
        + p11111
    ) / 32.0
    # average_pdf = (p000 + p001 + p010 + p011 + p100 + p101 + p110 + p111) / 8.0
    # average_pdf = (p00 + p01 + p10 + p11) / 4.0
    return average_pdf * cellVolume


compute_average_pdf = jax.jit(
    jax.vmap(
        compute_average_pdf_single,
        in_axes=(
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
            None,
            None,
            None,
            0,
            0,
            0,
            0,
            0,
        ),
    )
)


def sample_parameter(mean, var, num_samples):
    beta = 3.5
    std = jnp.sqrt(var)
    min_value = mean - beta * std
    max_value = mean + beta * std
    values = jnp.linspace(min_value, max_value, num_samples)
    # quantiles = jnp.linspace(0.0005, 0.9995, num_samples)
    # values = mean + std * jax.scipy.stats.norm.ppf(quantiles)
    return values


def compute_peicewise_approximate_pdf_heading_speed(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    pursuerTurnRadius,
    pursuerTurnRadiusVar,
    numSubdivisions,
):
    pursuerPositionXGrid = sample_parameter(
        pursuerPosition[0], pursuerPositionCov[0, 0], numSubdivisions
    )
    pursuerPositionYGrid = sample_parameter(
        pursuerPosition[1], pursuerPositionCov[1, 1], numSubdivisions
    )
    pursuerHeadingGrid = sample_parameter(
        pursuerHeading, pursuerHeadingVar, numSubdivisions
    )
    pursuerSpeedGrid = sample_parameter(pursuerSpeed, pursuerSpeedVar, numSubdivisions)
    pursuerTurnRadiusGrid = sample_parameter(
        pursuerTurnRadius, pursuerTurnRadiusVar, numSubdivisions
    )

    pursuerPositionXGridCenters = (
        pursuerPositionXGrid[:-1] + pursuerPositionXGrid[1:]
    ) / 2.0
    pursuerPositionYGridCenters = (
        pursuerPositionYGrid[:-1] + pursuerPositionYGrid[1:]
    ) / 2.0
    pursuerHeadingGridCenters = (pursuerHeadingGrid[:-1] + pursuerHeadingGrid[1:]) / 2.0
    pursuerSpeedGridCenters = (pursuerSpeedGrid[:-1] + pursuerSpeedGrid[1:]) / 2.0
    pursuerTurnRadiusGridCenters = (
        pursuerTurnRadiusGrid[:-1] + pursuerTurnRadiusGrid[1:]
    ) / 2.0
    # cellArea = (
    #     (pursuerHeadingGrid[1] - pursuerHeadingGrid[0])
    #     * (pursuerSpeedGrid[1] - pursuerSpeedGrid[0])
    #     * (pursuerTurnRadiusGrid[1] - pursuerTurnRadiusGrid[0])
    #     * (pursuerPositionXGrid[1] - pursuerPositionXGrid[0])
    #     * (pursuerPositionYGrid[1] - pursuerPositionYGrid[0])
    # )
    pursuerPositionXGridIndices = jnp.arange(len(pursuerPositionXGridCenters))
    pursuerPositionYGridIndices = jnp.arange(len(pursuerPositionYGridCenters))
    pursuerHeadingGridIndices = jnp.arange(len(pursuerHeadingGridCenters))
    pursuerSpeedGridIndices = jnp.arange(len(pursuerSpeedGridCenters))
    pursuerTurnRadiusGridIndices = jnp.arange(len(pursuerTurnRadiusGridCenters))
    (
        pursuerPositionXGridIndices,
        pursuerPositionYGridIndices,
        pursuerHeadingGridIndices,
        pursuerSpeedGridIndices,
        pursuerTurnRadiusGridIndices,
    ) = jnp.meshgrid(
        pursuerPositionXGridIndices,
        pursuerPositionYGridIndices,
        pursuerHeadingGridIndices,
        pursuerSpeedGridIndices,
        pursuerTurnRadiusGridIndices,
    )

    pursuerPositionXGridIndices = pursuerPositionXGridIndices.ravel()
    pursuerPositionYGridIndices = pursuerPositionYGridIndices.ravel()

    pursuerHeadingGridIndices = pursuerHeadingGridIndices.ravel()
    pursuerSpeedGridIndices = pursuerSpeedGridIndices.ravel()
    pursuerTurnRadiusGridIndices = pursuerTurnRadiusGridIndices.ravel()
    # compute the average pdf over the grid
    peicewiseAveragePdfTimesCellArea = compute_average_pdf(
        pursuerPosition,
        pursuerPositionCov,
        pursuerPositionXGrid,
        pursuerPositionYGrid,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerHeadingGrid,
        pursuerSpeed,
        pursuerSpeedVar,
        pursuerSpeedGrid,
        pursuerTurnRadius,
        pursuerTurnRadiusVar,
        pursuerTurnRadiusGrid,
        pursuerPositionXGridIndices,
        pursuerPositionYGridIndices,
        pursuerHeadingGridIndices,
        pursuerSpeedGridIndices,
        pursuerTurnRadiusGridIndices,
    )
    return (
        peicewiseAveragePdfTimesCellArea,
        pursuerPositionXGridCenters,
        pursuerPositionXGridIndices,
        pursuerPositionYGridCenters,
        pursuerPositionYGridIndices,
        pursuerHeadingGridCenters,
        pursuerHeadingGridIndices,
        pursuerSpeedGridCenters,
        pursuerSpeedGridIndices,
        pursuerTurnRadiusGridCenters,
        pursuerTurnRadiusGridIndices,
    )


@jax.jit
def dubins_pez_numerical_integration(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    numSubdivisions = 10
    # compute the average pdf over the ga
    (
        peicewiseAveragePdfTimesCellArea,
        pursuerPositionXGrid,
        pursuerPositionXGridIndices,
        pursuerPositionYGrid,
        pursuerPositionYGridIndices,
        pursuerHeadingGrid,
        pursuerHeadingGridIndices,
        pursuerSpeedGrid,
        pursuerSpeedGridIndices,
        pursuerTurnRadiusGrid,
        pursuerTurnRadiusGridIndices,
    ) = compute_peicewise_approximate_pdf_heading_speed(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        numSubdivisions,
    )
    jax.debug.print("length {x}", x=pursuerPositionXGridIndices.shape)
    pursuerPositions = jnp.column_stack(
        [
            pursuerPositionXGrid[pursuerPositionXGridIndices],
            pursuerPositionYGrid[pursuerPositionYGridIndices],
        ],
    )
    Z = new_in_dubins_engagement_zone3(
        pursuerPositions,
        pursuerHeadingGrid[pursuerHeadingGridIndices],
        pursuerTurnRadiusGrid[pursuerTurnRadiusGridIndices],
        captureRadius,
        pursuerRange,
        pursuerSpeedGrid[pursuerSpeedGridIndices],
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )

    epsilon = 1e-2
    weights = jax.nn.sigmoid(-Z / epsilon)
    cellProb = weights * peicewiseAveragePdfTimesCellArea
    jax.debug.print(
        "sum peicewiseAveragePdfTimesCellArea {x}",
        x=jnp.sum(peicewiseAveragePdfTimesCellArea),
    )
    # cellProb = jnp.where(Z < 0, peicewiseAveragePdf * cellArea, 0.0)
    probs = jnp.sum(cellProb, axis=1)

    return probs, 0, 0


def create_quadrature_nodes_and_weights(dim, order):
    # indep_dist = cp.MvNormal(np.zeros(dim), np.eye(dim))
    indep_dist = cp.J(*[cp.Normal(0, 1) for _ in range(dim)])

    # 3. Generate sparse grid nodes in the standard space
    nodes_z, weights = cp.generate_quadrature(
        order=order, dist=indep_dist, rule="gaussian", sparse=False, growth=True
    )  # nodes_z: shape (dim, n)
    return nodes_z, weights


@jax.jit
def transform_nodes_to_correlated_space(mu, cov, nodes_z, weights):
    # 4. TTrueransform Chaospy nodes to correlated space using Cholesky
    L = jnp.linalg.cholesky(cov)  # shape (dim, dim)
    nodes_z_jax = jnp.array(nodes_z)  # shape (dim, n)
    nodes = mu[:, None] + L @ nodes_z_jax  # shape (dim, n)
    return nodes, weights


nodes, weights = create_quadrature_nodes_and_weights(6, order=8)


@jax.jit
def dubins_pez_numerical_integration_sparse(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
    nodes,
    weights,
):
    mu = jnp.concatenate(
        [
            pursuerPosition,  # (2,)
            jnp.array([pursuerHeading]),  # (1,)
            jnp.array([minimumTurnRadius]),  # (1,)
            jnp.array([pursuerSpeed]),  # (1,)
            jnp.array([pursuerRange]),  # (1,)
        ]
    ).flatten()
    cov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        minimumTurnRadiusVar,
        pursuerSpeedVar,
        pursuerRangeVar,
    )
    nodes, weights = transform_nodes_to_correlated_space(mu, cov, nodes, weights)
    pursuerPositions = nodes[0:2, :].T  # shape (n, 2)
    pursuerHeadings = nodes[2, :]  # shape (n,)
    pursuerTurnRadii = nodes[3, :]  # shape (n,)
    pursuerSpeeds = nodes[4, :]  # shape (n,)
    pursuerRanges = nodes[5, :]  # shape (n,)

    Z = new_in_dubins_engagement_zone3(
        pursuerPositions,
        pursuerHeadings,
        pursuerTurnRadii,
        captureRadius,
        pursuerRanges,
        pursuerSpeeds,
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
    )

    # epsilon = 1e-3
    #
    # Zsmooth = jax.nn.sigmoid(-Z / epsilon)
    Zsmooth = jnp.where(Z < 0, 1.0, 0.0)
    probs = jnp.sum(weights * Zsmooth, axis=1)
    return probs, 0, 0


def dubins_pez_numerical_integration_surragate(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    mu = jnp.concatenate(
        [
            pursuerPosition,  # (2,)
            jnp.array([pursuerHeading]),  # (1,)
            jnp.array([minimumTurnRadius]),  # (1,)
            jnp.array([pursuerSpeed]),  # (1,)
            jnp.array([pursuerRange]),  # (1,)
        ]
    ).flatten()
    cov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        minimumTurnRadiusVar,
        pursuerSpeedVar,
        pursuerRangeVar,
    )
    nodes, weights = create_quadrature_nodes_and_weights(
        mu[0:5], cov[0:5, 0:5], order=2
    )
    coeffs, powers = polynomial_EZ.create_monomial_surragate(
        mu,
        cov,
        np.array([-3, 3, -3, 3 - np.pi, np.pi, 0, 0, 0, 0]),
        evaderSpeed,
        1000,
        degree=10,
    )

    pursuerRanges = np.ones((nodes.shape[1],)) * pursuerRange
    pursuerParams = np.hstack([nodes.T, pursuerRanges[:, None]])
    evaderParams = np.hstack([evaderPositions, evaderHeadings[:, None]])
    print("pursuerParams", pursuerParams.shape)
    print("evaderParams", evaderParams.shape)
    Z = polynomial_EZ.evaluate_monomial_grid(
        pursuerParams, evaderParams, powers, coeffs
    )

    epsilon = 1e-3
    # Zsmooth = jax.nn.sigmoid(-Z / epsilon)
    Zsmooth = jnp.where(Z < 0, 1.0, 0.0)
    probs = jnp.sum(weights * Zsmooth.T, axis=1)
    return probs, 0, 0


def create_bounding_and_linearization_points(mean, std, numBoundingPoints):
    # create bounding points
    maxBound = mean + 3 * std
    minBound = mean - 3 * std
    boundingPoints = jnp.linspace(minBound, maxBound, numBoundingPoints)
    linearizationPoints = (boundingPoints[:-1] + boundingPoints[1:]) / 2.0
    lineaizationIndex = jnp.arange(len(linearizationPoints))
    return boundingPoints, linearizationPoints, lineaizationIndex


# @jax.jit
# def create_linear_model_single(
#     evaderPositions,
#     evaderHeadings,
#     evaderSpeed,
#     pursuerPositionsX,
#     pursuerPositionsY,
#     pursuerHeadings,
#     pursuerSpeeds,
#     pursuerTurnRadii,
#     pursuerRanges,
#     captureRadius,
#     pursuerPositionXIndex,
#     pursuerPositionYIndex,
#     pursuerHeadingIndex,
#     pursuerTurnRadiusIndex,
#     pursuerRangeIndex,
#     pursuerSpeedIndex,
# ):
#     pursuerPositionX = pursuerPositionsX[pursuerPositionXIndex]
#     pursuerPositionY = pursuerPositionsY[pursuerPositionYIndex]
#     pursuerHeading = pursuerHeadings[pursuerHeadingIndex]
#     pursuerSpeed = pursuerSpeeds[pursuerSpeedIndex]
#     pursuerTurnRadius = pursuerTurnRadii[pursuerTurnRadiusIndex]
#     pursuerRange = pursuerRanges[pursuerRangeIndex]
#
#     pursuerParams = jnp.concatenate(
#         [
#             jnp.array([pursuerPositionX]),  # (1,)
#             jnp.array([pursuerPositionY]),  # (1,)
#             jnp.array([pursuerHeading]),  # (1,)
#             jnp.array([pursuerSpeed]),  # (1,)
#             jnp.array([pursuerTurnRadius]),  # (1,)
#             jnp.array([pursuerRange]),  # (1,)
#         ]
#     )
#     evaderParams = jnp.concatenate(
#         [evaderPositions, jnp.array([evaderHeadings]), jnp.array([evaderSpeed])]
#     )
#     pursuerPosition = jnp.array([pursuerPositionX, pursuerPositionY])
#     val = csbez.in_dubins_engagement_zone_right_single(
#         pursuerPosition,
#         pursuerHeading,
#         pursuerTurnRadius,
#         captureRadius,
#         pursuerRange,
#         pursuerSpeed,
#         evaderPositions,
#         evaderHeadings,
#         evaderSpeed,
#     )
#     dDubinsEZ_dPursuerParamsValue = dDubinsEZRight_dPursuerParams(
#         pursuerParams, evaderParams
#     )
#     M = dDubinsEZ_dPursuerParamsValue
#     b = val - jnp.dot(M, pursuerParams)
#
#     return M, b
#
#
# create_linear_model = jax.jit(
#     jax.vmap(
#         create_linear_model_single,
#         in_axes=(
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#         ),
#     )
# )


def multivariate_normal_pdf(x, mean, cov):
    return jax.scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=cov)


multivariate_normal_pdf_vmap = jax.vmap(
    multivariate_normal_pdf, in_axes=(0, None, None)
)


def compute_hypercube_verticies(
    pursuerPositionX0,
    pursuerPositionX1,
    pursuerPositionY0,
    pursuerPositionY1,
    pursuerHeading0,
    pursuerHeading1,
    pursuerTurnRadius0,
    pursuerTurnRadius1,
    pursuerRange0,
    pursuerRange1,
    pursuerSpeed0,
    pursuerSpeed1,
):
    # Step 1: Define low and high values
    lows = jnp.array(
        [
            pursuerPositionX0,
            pursuerPositionY0,
            pursuerHeading0,
            pursuerTurnRadius0,
            pursuerRange0,
            pursuerSpeed0,
        ]
    )

    highs = jnp.array(
        [
            pursuerPositionX1,
            pursuerPositionY1,
            pursuerHeading1,
            pursuerTurnRadius1,
            pursuerRange1,
            pursuerSpeed1,
        ]
    )

    # Step 2: Generate all binary combinations (64, 6) using broadcasting
    # Create binary numbers from 0 to 63 (shape: (64,))
    binary_indices = jnp.arange(64, dtype=np.uint8)[:, None]  # shape (64, 1)

    # Create powers of 2 for bitmasking: [32, 16, 8, 4, 2, 1]
    bit_weights = 2 ** jnp.arange(5, -1, -1)  # shape (6,)

    # Apply bitmasking to get binary matrix
    choices = ((binary_indices & bit_weights) > 0).astype(int)  # shape (64, 6)

    # Step 3: Interpolate between lows and highs
    vertices = lows + choices * (highs - lows)  # shape (64, 6)
    return jnp.array(vertices)


def compute_average_combined_pdf_single(
    pursuerParamsMean,
    pursuerParamsCov,
    pursuerPositionXGrid,
    pursuerPositionYGrid,
    pursuerHeadingGrid,
    pursuerTurnRadiusGrid,
    pursuerRangeGrid,
    pursuerSpeedGrid,
    pursuerPositionXIndex,
    pursuerPositionYIndex,
    pursuerHeadingIndex,
    pursuerTurnRadiusIndex,
    pursuerRangeIndex,
    pursuerSpeedIndex,
):
    # Grid corners
    pursuerPositionX0 = pursuerPositionXGrid[pursuerPositionXIndex]
    pursuerPositionX1 = pursuerPositionXGrid[pursuerPositionXIndex + 1]
    pursuerPositionY0 = pursuerPositionYGrid[pursuerPositionYIndex]
    pursuerPositionY1 = pursuerPositionYGrid[pursuerPositionYIndex + 1]
    pursuerHeading0 = pursuerHeadingGrid[pursuerHeadingIndex]
    pursuerHeading1 = pursuerHeadingGrid[pursuerHeadingIndex + 1]
    pursuerTurnRadius0 = pursuerTurnRadiusGrid[pursuerTurnRadiusIndex]
    pursuerTurnRadius1 = pursuerTurnRadiusGrid[pursuerTurnRadiusIndex + 1]
    pursuerRange0 = pursuerRangeGrid[pursuerRangeIndex]
    pursuerRange1 = pursuerRangeGrid[pursuerRangeIndex + 1]
    pursuerSpeed0 = pursuerSpeedGrid[pursuerSpeedIndex]
    pursuerSpeed1 = pursuerSpeedGrid[pursuerSpeedIndex + 1]

    vertices = compute_hypercube_verticies(
        pursuerPositionX0,
        pursuerPositionX1,
        pursuerPositionY0,
        pursuerPositionY1,
        pursuerHeading0,
        pursuerHeading1,
        pursuerTurnRadius0,
        pursuerTurnRadius1,
        pursuerRange0,
        pursuerRange1,
        pursuerSpeed0,
        pursuerSpeed1,
    )

    # Step 3: Compute the PDF for each corner
    pdfs = multivariate_normal_pdf_vmap(
        vertices, pursuerParamsMean, pursuerParamsCov
    )  # shape (64,)
    return jnp.mean(pdfs)
    # pursuerPositionX = (
    #     pursuerPositionXGrid[pursuerPositionXIndex]
    #     + pursuerPositionXGrid[pursuerPositionXIndex + 1]
    # ) / 2.0
    # pursuerPositionY = (
    #     pursuerPositionYGrid[pursuerPositionYIndex]
    #     + pursuerPositionYGrid[pursuerPositionYIndex + 1]
    # ) / 2.0
    # pursuerHeading = (
    #     pursuerHeadingGrid[pursuerHeadingIndex]
    #     + pursuerHeadingGrid[pursuerHeadingIndex + 1]
    # ) / 2.0
    # pursuerTurnRadius = (
    #     pursuerTurnRadiusGrid[pursuerTurnRadiusIndex]
    #     + pursuerTurnRadiusGrid[pursuerTurnRadiusIndex + 1]
    # ) / 2.0
    # pursuerRange = (
    #     pursuerRangeGrid[pursuerRangeIndex] + pursuerRangeGrid[pursuerRangeIndex + 1]
    # ) / 2.0
    # pursuerSpeed = (
    #     pursuerSpeedGrid[pursuerSpeedIndex] + pursuerSpeedGrid[pursuerSpeedIndex + 1]
    # ) / 2.0
    #
    # pursuerParams = jnp.array(
    #     [
    #         pursuerPositionX,
    #         pursuerPositionY,
    #         pursuerHeading,
    #         pursuerTurnRadius,
    #         pursuerRange,
    #         pursuerSpeed,
    #     ]
    # )
    # return multivariate_normal_pdf(pursuerParams, pursuerParamsMean, pursuerParamsCov)


compute_average_pdf_combined = jax.jit(
    jax.vmap(
        compute_average_combined_pdf_single,
        in_axes=(None, None, None, None, None, None, None, None, 0, 0, 0, 0, 0, 0),
    )
)


def compute_peicewise_approximate_pdf(
    pursuerParams,
    combinedCov,
    numSubdivisions,
):
    minPursuerPositionX = pursuerParams[0] - 3 * jnp.sqrt(combinedCov[0, 0])
    maxPursuerPositionX = pursuerParams[0] + 3 * jnp.sqrt(combinedCov[0, 0])
    minPursuerPositionY = pursuerParams[1] - 3 * jnp.sqrt(combinedCov[1, 1])
    maxPursuerPositionY = pursuerParams[1] + 3 * jnp.sqrt(combinedCov[1, 1])
    minPursuerHeading = pursuerParams[2] - 3 * jnp.sqrt(combinedCov[2, 2])
    maxPursuerHeading = pursuerParams[2] + 3 * jnp.sqrt(combinedCov[2, 2])
    minPursuerTurnRadius = pursuerParams[3] - 3 * jnp.sqrt(combinedCov[3, 3])
    maxPursuerTurnRadius = pursuerParams[3] + 3 * jnp.sqrt(combinedCov[3, 3])
    minPursuerRange = pursuerParams[4] - 3 * jnp.sqrt(combinedCov[4, 4])
    maxPursuerRange = pursuerParams[4] + 3 * jnp.sqrt(combinedCov[4, 4])
    minPursuerSpeed = pursuerParams[5] - 3 * jnp.sqrt(combinedCov[5, 5])
    maxPursuerSpeed = pursuerParams[5] + 3 * jnp.sqrt(combinedCov[5, 5])

    pursuerPositionXGrid = jnp.linspace(
        minPursuerPositionX, maxPursuerPositionX, numSubdivisions
    )
    pursuerPositionYGrid = jnp.linspace(
        minPursuerPositionY, maxPursuerPositionY, numSubdivisions
    )
    pursuerHeadingGrid = jnp.linspace(
        minPursuerHeading, maxPursuerHeading, numSubdivisions
    )
    pursuerTurnRadiusGrid = jnp.linspace(
        minPursuerTurnRadius, maxPursuerTurnRadius, numSubdivisions
    )
    pursuerRangeGrid = jnp.linspace(minPursuerRange, maxPursuerRange, numSubdivisions)
    pursuerSpeedGrid = jnp.linspace(minPursuerSpeed, maxPursuerSpeed, numSubdivisions)

    pursuerPositionXGridCenters = (
        pursuerPositionXGrid[:-1] + pursuerPositionXGrid[1:]
    ) / 2.0
    pursuerPositionYGridCenters = (
        pursuerPositionYGrid[:-1] + pursuerPositionYGrid[1:]
    ) / 2.0
    pursuerHeadingGridCenters = (pursuerHeadingGrid[:-1] + pursuerHeadingGrid[1:]) / 2.0
    pursuerTurnRadiusGridCenters = (
        pursuerTurnRadiusGrid[:-1] + pursuerTurnRadiusGrid[1:]
    ) / 2.0
    pursuerRangeGridCenters = (pursuerRangeGrid[:-1] + pursuerRangeGrid[1:]) / 2.0
    pursuerSpeedGridCenters = (pursuerSpeedGrid[:-1] + pursuerSpeedGrid[1:]) / 2.0

    pursuerPositionXGridIndices = jnp.arange(len(pursuerPositionXGridCenters))
    pursuerPositionYGridIndices = jnp.arange(len(pursuerPositionYGridCenters))
    pursuerHeadingGridIndices = jnp.arange(len(pursuerHeadingGridCenters))
    pursuerTurnRadiusGridIndices = jnp.arange(len(pursuerTurnRadiusGridCenters))
    pursuerRangeGridIndices = jnp.arange(len(pursuerRangeGridCenters))
    pursuerSpeedGridIndices = jnp.arange(len(pursuerSpeedGridCenters))

    (
        pursuerPositionXGridIndices,
        pursuerPositionYGridIndices,
        pursuerHeadingGridIndices,
        pursuerTurnRadiusGridIndices,
        pursuerRangeGridIndices,
        pursuerSpeedGridIndices,
    ) = jnp.meshgrid(
        pursuerPositionXGridIndices,
        pursuerPositionYGridIndices,
        pursuerHeadingGridIndices,
        pursuerTurnRadiusGridIndices,
        pursuerRangeGridIndices,
        pursuerSpeedGridIndices,
    )
    cellArea = (
        (pursuerPositionXGrid[1] - pursuerPositionXGrid[0])
        * (pursuerPositionYGrid[1] - pursuerPositionYGrid[0])
        * (pursuerHeadingGrid[1] - pursuerHeadingGrid[0])
        * (pursuerTurnRadiusGrid[1] - pursuerTurnRadiusGrid[0])
        * (pursuerRangeGrid[1] - pursuerRangeGrid[0])
        * (pursuerSpeedGrid[1] - pursuerSpeedGrid[0])
    )

    pursuerPositionXGridIndices = pursuerPositionXGridIndices.ravel()
    pursuerPositionYGridIndices = pursuerPositionYGridIndices.ravel()
    pursuerHeadingGridIndices = pursuerHeadingGridIndices.ravel()
    pursuerTurnRadiusGridIndices = pursuerTurnRadiusGridIndices.ravel()
    pursuerRangeGridIndices = pursuerRangeGridIndices.ravel()
    pursuerSpeedGridIndices = pursuerSpeedGridIndices.ravel()

    # compute the average pdf over the grid
    peicewiseAveragePdf = compute_average_pdf_combined(
        pursuerParams,
        combinedCov,
        pursuerPositionXGrid,
        pursuerPositionYGrid,
        pursuerHeadingGrid,
        pursuerTurnRadiusGrid,
        pursuerRangeGrid,
        pursuerSpeedGrid,
        pursuerPositionXGridIndices,
        pursuerPositionYGridIndices,
        pursuerHeadingGridIndices,
        pursuerTurnRadiusGridIndices,
        pursuerRangeGridIndices,
        pursuerSpeedGridIndices,
    )
    return (
        peicewiseAveragePdf,
        pursuerPositionXGridCenters,
        pursuerPositionXGridIndices,
        pursuerPositionYGridCenters,
        pursuerPositionYGridIndices,
        pursuerHeadingGridCenters,
        pursuerHeadingGridIndices,
        pursuerTurnRadiusGridCenters,
        pursuerTurnRadiusGridIndices,
        pursuerRangeGridCenters,
        pursuerRangeGridIndices,
        pursuerSpeedGridCenters,
        pursuerSpeedGridIndices,
        cellArea,
    )


def compute_probability_mass_in_cell_combined_single(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    peicewiseAveragePdf,
    pursuerPositionXGridCenters,
    pursuerPositionYGridCenters,
    pursuerHeadingGridCenters,
    pursuerTurnRadiusGridCenters,
    pursuerRangeGridCenters,
    pursuerSpeedGridCenters,
    pursuerPositionXGridIndex,
    pursuerPositionYGridIndex,
    pursuerHeadingGridIndex,
    pursuerTurnRadiusGridIndex,
    pursuerRangeGridIndex,
    pursuerSpeedGridIndex,
    cellArea,
):
    pursuerPosition = jnp.array(
        [
            pursuerPositionXGridCenters[pursuerPositionXGridIndex],
            pursuerPositionYGridCenters[pursuerPositionYGridIndex],
        ]
    )
    pursuerHeading = pursuerHeadingGridCenters[pursuerHeadingGridIndex]
    pursuerSpeed = pursuerSpeedGridCenters[pursuerSpeedGridIndex]
    pursuerTurnRadius = pursuerTurnRadiusGridCenters[pursuerTurnRadiusGridIndex]
    pursuerRange = pursuerRangeGridCenters[pursuerRangeGridIndex]

    z = csbez.in_dubins_engagement_zone_single(
        pursuerPosition,
        pursuerHeading,
        pursuerTurnRadius,
        0.0,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )

    probmass = jnp.where(
        z < 0,
        peicewiseAveragePdf[
            pursuerPositionXGridIndex,
            pursuerPositionYGridIndex,
            pursuerHeadingGridIndex,
            pursuerTurnRadiusGridIndex,
            pursuerRangeGridIndex,
            pursuerSpeedGridIndex,
        ]
        * cellArea,
        0.0,
    )

    return probmass, z


compute_probability_mass_in_cell_combined = jax.jit(
    jax.vmap(
        compute_probability_mass_in_cell_combined_single,
        in_axes=(
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
            0,
            0,
            0,
            0,
            0,
            0,
            None,
        ),
    )
)


def piecewise_linear_dubins_pez_pddf_single(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    peicewiseAveragePdf,
    pursuerPositionXGridCenters,
    pursuerPositionXGridIndices,
    pursuerPositionYGridCenters,
    pursuerPositionYGridIndices,
    pursuerHeadingGridCenters,
    pursuerHeadingGridIndices,
    pursuerTurnRadiusGridCenters,
    pursuerTurnRadiusGridIndices,
    pursuerRangeGridCenters,
    pursuerRangeGridIndices,
    pursuerSpeedGridCenters,
    pursuerSpeedGridIndices,
    cellArea,
):
    #
    probMass, z = compute_probability_mass_in_cell_combined(
        evaderPosition,
        evaderHeading,
        evaderSpeed,
        peicewiseAveragePdf,
        pursuerPositionXGridCenters,
        pursuerPositionYGridCenters,
        pursuerHeadingGridCenters,
        pursuerTurnRadiusGridCenters,
        pursuerRangeGridCenters,
        pursuerSpeedGridCenters,
        pursuerPositionXGridIndices,
        pursuerPositionYGridIndices,
        pursuerHeadingGridIndices,
        pursuerTurnRadiusGridIndices,
        pursuerRangeGridIndices,
        pursuerSpeedGridIndices,
        cellArea,
    )
    return jnp.sum(probMass), 0, 0, 0


piecewise_linear_dubins_pez_pddf_single_vmap = jax.jit(
    jax.vmap(
        piecewise_linear_dubins_pez_pddf_single,
        in_axes=(
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
            None,
            None,
            None,
            None,
            None,
        ),
    )
)


@jax.jit
def numerical_dubins_pez(
    evaderPositions,
    evaderHeadings,
    evaderSpeed,
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    pursuerRange,
    pursuerRangeVar,
    captureRadius,
):
    numSubdivisions = 5
    pursuerParams = jnp.concatenate(
        [
            pursuerPosition,  # (2,)
            jnp.array([pursuerHeading]),  # (1,)
            jnp.array([pursuerSpeed]),  # (1,)
            jnp.array([minimumTurnRadius]),  # (1,)
            jnp.array([pursuerRange]),  # (1,)
        ]
    )
    # jax.debug.print("pursuerParams: {x}", x=pursuerParams)
    combinedCov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        minimumTurnRadiusVar,
        pursuerSpeedVar,
        pursuerRangeVar,
    )
    # jax.debug.print("cov: {x}", x=combinedCov)
    (
        peicewiseAveragePdf,
        pursuerPositionXGridCenters,
        pursuerPositionXGridIndices,
        pursuerPositionYGridCenters,
        pursuerPositionYGridIndices,
        pursuerHeadingGridCenters,
        pursuerHeadingGridIndices,
        pursuerTurnRadiusGridCenters,
        pursuerTurnRadiusGridIndices,
        pursuerRangeGridCenters,
        pursuerRangeGridIndices,
        pursuerSpeedGridCenters,
        pursuerSpeedGridIndices,
        cellArea,
    ) = compute_peicewise_approximate_pdf(
        pursuerParams,
        combinedCov,
        numSubdivisions,
    )

    numGrid = len(pursuerHeadingGridCenters)
    peicewiseAveragePdf = peicewiseAveragePdf.reshape(
        numGrid, numGrid, numGrid, numGrid, numGrid, numGrid
    )

    prob, _, _, _ = piecewise_linear_dubins_pez_pddf_single_vmap(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        peicewiseAveragePdf,
        pursuerPositionXGridCenters,
        pursuerPositionXGridIndices,
        pursuerPositionYGridCenters,
        pursuerPositionYGridIndices,
        pursuerHeadingGridCenters,
        pursuerHeadingGridIndices,
        pursuerTurnRadiusGridCenters,
        pursuerTurnRadiusGridIndices,
        pursuerRangeGridCenters,
        pursuerRangeGridIndices,
        pursuerSpeedGridCenters,
        pursuerSpeedGridIndices,
        cellArea,
    )
    return prob, 0, 0, 0


# piecewise_linear_dubins_pez_pddf = jax.jit(
#     jax.vmap(
#         piecewise_linear_dubins_pez_pddf_single,
#         in_axes=(
#             0,
#             0,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#         ),
#     )
# )
