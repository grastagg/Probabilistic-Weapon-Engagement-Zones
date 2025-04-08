from jax.lax import with_sharding_constraint
import numpy as np

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from functools import partial
import time
import scipy


import fast_pursuer

import dubinsEZ

# Vectorized function using vmap
in_dubins_engagement_zone = jax.jit(
    jax.vmap(
        dubinsEZ.in_dubins_engagement_zone_single,
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
#
# in_dubins_engagement_zone = jax.jit(
#     jax.vmap(
#         dubinsEZ.in_dubins_engagement_zone_single,
#         in_axes=(
#             None,  # pursuerPosition
#             0,  # pursuerHeading
#             None,  # minimumTurnRadius
#             None,  # captureRadius
#             None,  # pursuerRange
#             None,  # pursuerSpeed
#             None,  # evaderPosition
#             None,  # evaderHeading
#             None,  # evaderSpeed
#         ),  # Vectorizing over evaderPosition & evaderHeading
#     )
# )


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


mc_dubins_pez = jax.jit(
    jax.vmap(
        mc_dubins_pez_single,
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
    numSamples = 5000

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
    return dubinsEZ.in_dubins_engagement_zone_single(
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


def dubins_EZ_single_right_combined_input(pursuerParams, evaderParams):
    pursuerPosition = pursuerParams[:2]
    pursuerHeading = pursuerParams[2]
    pursuerSpeed = pursuerParams[3]
    minimumTurnRadius = pursuerParams[4]
    pursuerRange = pursuerParams[5]
    captureRadius = pursuerParams[6]
    evaderPosition = evaderParams[:2]
    evaderHeading = evaderParams[2]
    evaderSpeed = evaderParams[3]
    return dubinsEZ.in_dubins_engagement_zone_right_single(
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


def dubins_EZ_single_left_combined_input(pursuerParams, evaderParams):
    pursuerPosition = pursuerParams[:2]
    pursuerHeading = pursuerParams[2]
    pursuerSpeed = pursuerParams[3]
    minimumTurnRadius = pursuerParams[4]
    pursuerRange = pursuerParams[5]
    captureRadius = pursuerParams[6]
    evaderPosition = evaderParams[:2]
    evaderHeading = evaderParams[2]
    evaderSpeed = evaderParams[3]
    return dubinsEZ.in_dubins_engagement_zone_left_single(
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
dDubinsEZ_dPursuerPosition = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 0)
dDubinsEZ_dPursuerHeading = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 1)
dDubinsEZ_dMinimumTurnRadius = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 2)
dDubinsEZ_dCaptureRadius = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 3)
dDubinsEZ_dPursuerRange = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 4)
dDubinsEZ_dPursuerSpeed = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 5)
dDubinsEZ_dEvaderPosition = jax.jacfwd(dubinsEZ.in_dubins_engagement_zone_single, 6)

# second order derivatives
d2DubinsEZ_dPursuerPosition = jax.jacfwd(dDubinsEZ_dPursuerPosition, 0)
d2DubinsEZ_dPursuerHeading = jax.jacfwd(dDubinsEZ_dPursuerHeading, 1)
d2DubinsEZ_dMinimumTurnRadius = jax.jacfwd(dDubinsEZ_dMinimumTurnRadius, 2)
d2DubinsEZ_dCaptureRadius = jax.jacfwd(dDubinsEZ_dCaptureRadius, 3)
d2DubinsEZ_dPursuerRange = jax.jacfwd(dDubinsEZ_dPursuerRange, 4)
d2DubinsEZ_dPursuerSpeed = jax.jacfwd(dDubinsEZ_dPursuerSpeed, 5)
d2DubinsEZ_dEvaderPosition = jax.jacfwd(dDubinsEZ_dEvaderPosition, 6)

d3DubinsEZ_dPursuerHeading = jax.jacfwd(d2DubinsEZ_dPursuerHeading, 1)
d4DubinsEZ_dPursuerHeading = jax.jacfwd(d3DubinsEZ_dPursuerHeading, 1)
d5DubinsEZ_dPursuerHeading = jax.jacfwd(d4DubinsEZ_dPursuerHeading, 1)
d6DubinsEZ_dPursuerHeading = jax.jacfwd(d5DubinsEZ_dPursuerHeading, 1)

# first order combined derivatives
dDubinsEZ_dPursuerParams = jax.jacfwd(dubins_EZ_single_combined_input, 0)
dDubinsEZRight_dPursuerParams = jax.jacfwd(dubins_EZ_single_right_combined_input, 0)
dDubinsEZLeft_dPursuerParams = jax.jacfwd(dubins_EZ_single_left_combined_input, 0)

# sevond order combined derivatives
d2DubinsEZ_dPursuerParams = jax.jacfwd(dDubinsEZ_dPursuerParams, 0)
d2DubinsEZRight_dPursuerParams = jax.jacfwd(dDubinsEZRight_dPursuerParams, 0)
d2DubinsEZLeft_dPursuerParams = jax.jacfwd(dDubinsEZLeft_dPursuerParams, 0)
# third order combined derivatives
d3DubinsEZ_dPursuerParams = jax.jacfwd(d2DubinsEZ_dPursuerParams, 0)


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
    mean = dubinsEZ.in_dubins_engagement_zone_single(
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


def find_closest_linearization_point_right(
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
    centerPoint = jnp.array(
        [
            pursuerPosition[0] + minimumTurnRadius * jnp.sin(pursuerHeading),
            pursuerPosition[1] - minimumTurnRadius * jnp.cos(pursuerHeading),
        ]
    )
    direction = jnp.array([jnp.cos(evaderHeadings), jnp.sin(evaderHeadings)])
    speedRatio = evaderSpeed / pursuerSpeed
    shiftedCenterPoint = centerPoint - speedRatio * pursuerRange * direction
    return closest_point_on_circle(
        evaderPositions, shiftedCenterPoint, minimumTurnRadius
    )


def linear_dubins_PEZ_single_right(
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
    val = dubinsEZ.in_dubins_engagement_zone_right_single(
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

    # def _use_original():
    #     return evaderPositions, val
    #
    # def _find_closest():
    #     closest = find_closest_linearization_point_right(
    #         evaderPositions,
    #         evaderHeadings,
    #         evaderSpeed,
    #         pursuerPosition,
    #         pursuerHeading,
    #         pursuerSpeed,
    #         minimumTurnRadius,
    #         pursuerRange,
    #         captureRadius,
    #     )
    #     val = dubinsEZ.in_dubins_engagement_zone_right_single(
    #         pursuerPosition,
    #         pursuerHeading,
    #         minimumTurnRadius,
    #         captureRadius,
    #         pursuerRange,
    #         pursuerSpeed,
    #         closest,
    #         evaderHeadings,
    #         evaderSpeed,
    #     )
    #     return closest, val
    #
    # evaderPositions, val = jax.lax.cond(jnp.isnan(val), _find_closest, _use_original)

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
    dDubinsEZ_dPursuerParamsValue = dDubinsEZRight_dPursuerParams(
        pursuerParams, evaderParams
    )
    full_cov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        pursuerSpeedVar,
        minimumTurnRadiusVar,
        pursuerRangeVar,
    )
    mean = val
    var = dDubinsEZ_dPursuerParamsValue @ full_cov @ dDubinsEZ_dPursuerParamsValue.T
    return (
        jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var)),
        mean,
        var,
        dDubinsEZ_dPursuerParamsValue,
    )


linear_dubins_pez_right = jax.jit(
    jax.vmap(
        linear_dubins_PEZ_single_right,
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


def find_closest_linearization_point_left(
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
    centerPoint = jnp.array(
        [
            pursuerPosition[0] - minimumTurnRadius * jnp.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * jnp.cos(pursuerHeading),
        ]
    )
    direction = jnp.array([jnp.cos(evaderHeadings), jnp.sin(evaderHeadings)])
    speedRatio = evaderSpeed / pursuerSpeed
    shiftedCenterPoint = centerPoint - speedRatio * pursuerRange * direction
    return closest_point_on_circle(
        evaderPositions, shiftedCenterPoint, minimumTurnRadius
    )


def linear_dubins_PEZ_single_left(
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
    val = dubinsEZ.in_dubins_engagement_zone_left_single(
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

    # def _use_original():
    #     return evaderPositions, val
    #
    # def _find_closest():
    #     closest = find_closest_linearization_point_left(
    #         evaderPositions,
    #         evaderHeadings,
    #         evaderSpeed,
    #         pursuerPosition,
    #         pursuerHeading,
    #         pursuerSpeed,
    #         minimumTurnRadius,
    #         pursuerRange,
    #         captureRadius,
    #     )
    #     val = dubinsEZ.in_dubins_engagement_zone_left_single(
    #         pursuerPosition,
    #         pursuerHeading,
    #         minimumTurnRadius,
    #         captureRadius,
    #         pursuerRange,
    #         pursuerSpeed,
    #         closest,
    #         evaderHeadings,
    #         evaderSpeed,
    #     )
    #     return closest, val
    #
    # evaderPositions, val = jax.lax.cond(jnp.isnan(val), _find_closest, _use_original)
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

    dDubinsEZ_dPursuerParamsValue = dDubinsEZLeft_dPursuerParams(
        pursuerParams, evaderParams
    )
    full_cov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        pursuerSpeedVar,
        minimumTurnRadiusVar,
        pursuerRangeVar,
    )
    mean = val
    var = dDubinsEZ_dPursuerParamsValue @ full_cov @ dDubinsEZ_dPursuerParamsValue.T
    return (
        jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var)),
        mean,
        var,
        dDubinsEZ_dPursuerParamsValue,
    )


linear_dubins_pez_left = jax.jit(
    jax.vmap(
        linear_dubins_PEZ_single_left,
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


def cdf_min_correlated_gaussians(x, mu1, sigma1, mu2, sigma2, rho):
    z1 = (x - mu1) / sigma1
    z2 = (x - mu2) / sigma2
    phi1 = jax.scipy.stats.norm.cdf(z1)
    phi2 = jax.scipy.stats.norm.cdf((z2 - rho * z1) / jnp.sqrt(1 - rho**2))
    return 1 - phi1 * phi2


def softmin_cdf(x, mu1, sigma1, mu2, sigma2, temperature=0.00001):
    # Smooth min: w1 = sigmoid((mu2 - mu1) / temp)
    w1 = jax.nn.sigmoid((mu2 - mu1) / temperature)
    w2 = 1.0 - w1
    cdf1 = jax.scipy.stats.norm.cdf(x, loc=mu1, scale=sigma1)
    cdf2 = jax.scipy.stats.norm.cdf(x, loc=mu2, scale=sigma2)
    return w1 * cdf1 + w2 * cdf2


def true_min_cdf(x, mu1, sigma1, mu2, sigma2, rho, eps=1e-6):
    mean = [mu1, mu2]
    cov = np.array(
        [
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2],
        ]
    )
    # Regularize covariance for stability
    cov += eps * np.eye(2)

    # Marginal CDFs
    p1 = scipy.stats.norm.cdf(x, loc=mu1, scale=sigma1)
    p2 = scipy.stats.norm.cdf(x, loc=mu2, scale=sigma2)

    # Joint CDF
    joint = scipy.stats.multivariate_normal.cdf([x, x], mean=mean, cov=cov)

    return p1 + p2 - joint


def combined_left_right_dubins_PEZ(
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
    right, rightMean, rightVar, dDubinsEZRight_dPursuerParamsValue = (
        linear_dubins_pez_right(
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
        )
    )

    left, leftMean, leftVar, dDubinsEZLeft_dPursuerParamsValue = linear_dubins_pez_left(
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
    )
    full_cov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        pursuerSpeedVar,
        minimumTurnRadiusVar,
        pursuerRangeVar,
    )
    # covariance = (
    #     dDubinsEZRight_dPursuerParamsValue
    #     @ full_cov
    #     @ dDubinsEZLeft_dPursuerParamsValue.T
    # )
    covariance = jnp.einsum(
        "ni,ij,nj->n",
        dDubinsEZRight_dPursuerParamsValue,
        full_cov,
        dDubinsEZLeft_dPursuerParamsValue,
    )

    correlation = covariance / jnp.sqrt(rightVar * leftVar)
    # uncorrelated version
    return 1 - (1 - right) * (1 - left)

    # true correlated version
    return true_min_cdf(
        0.0,
        rightMean[0],
        jnp.sqrt(rightVar[0]),
        leftMean[0],
        jnp.sqrt(leftVar[0]),
        correlation[0],
    )

    # softmin version correlated
    return softmin_cdf(
        0.0, rightMean, rightVar, leftMean, leftVar, correlation
    ).flatten()
    # approximate version correlated
    return cdf_min_correlated_gaussians(
        0.0, rightMean, rightVar, leftMean, leftVar, correlation
    )


def qaudratic_dubins_PEZ_single_right(
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
    val = dubinsEZ.in_dubins_engagement_zone_right_single(
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
    dDubinsEZ_dPursuerParamsValue = dDubinsEZRight_dPursuerParams(
        pursuerParams, evaderParams
    )
    d2DubinsEZ_dPursuerParamsValue = d2DubinsEZRight_dPursuerParams(
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
    return jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var))


quadratic_dubins_pez_right = jax.jit(
    jax.vmap(
        qaudratic_dubins_PEZ_single_right,
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


def qaudratic_dubins_PEZ_single_left(
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
    val = dubinsEZ.in_dubins_engagement_zone_left_single(
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
    dDubinsEZ_dPursuerParamsValue = dDubinsEZLeft_dPursuerParams(
        pursuerParams, evaderParams
    )
    d2DubinsEZ_dPursuerParamsValue = d2DubinsEZLeft_dPursuerParams(
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
    return jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var))


quadratic_dubins_pez_left = jax.jit(
    jax.vmap(
        qaudratic_dubins_PEZ_single_left,
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


def combined_left_right_quadratic_dubins_PEZ(
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
    right = quadratic_dubins_pez_right(
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
    )
    left = quadratic_dubins_pez_left(
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
    )
    return 1 - (1 - right) * (1 - left)


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
    val = dubinsEZ.in_dubins_engagement_zone_single(
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


def third_central_moment_taylor(J, H, Sigma):
    # First-order pieces
    JT_S_J = jnp.dot(J, jnp.dot(Sigma, J))  # Jᵗ Σ J
    tr_HS = jnp.trace(jnp.dot(H, Sigma))  # tr(H Σ)
    JTHSJ = jnp.dot(J, jnp.dot(H, jnp.dot(Sigma, J)))  # Jᵗ H Σ J

    # Higher-order terms
    HS = jnp.dot(H, Sigma)
    tr_HS2 = jnp.trace(jnp.dot(HS, HS))  # tr((H Σ)^2)
    tr_HS3 = jnp.trace(jnp.dot(HS, jnp.dot(HS, HS)))  # tr((H Σ)^3)

    # Combine terms
    skew = (3 / 2) * (JT_S_J * tr_HS + 2 * JTHSJ) + tr_HS3 - tr_HS * tr_HS2
    return skew


def edgeworth_pdf_skew_only(t, mean, var, third_moment):
    std = jnp.sqrt(var)
    z = (t - mean) / std
    phi = jax.scipy.stats.norm.pdf(z)
    skew = third_moment / (std**3)

    correction = (skew / 6) * (z**3 - 3 * z) * phi
    return (phi + correction) / std


edgeworth_pdf_skew_only_vec = jax.jit(
    jax.vmap(edgeworth_pdf_skew_only, in_axes=(0, None, None, None))
)


def edgeworth_cdf_skew_only(t, mean, var, third_moment):
    z = (t - mean) / jnp.sqrt(var)
    skew = third_moment / jnp.sqrt(var) ** 3
    return (
        jax.scipy.stats.norm.cdf(z)
        + (skew / 6) * (1 - z**2) * jax.scipy.stats.norm.pdf(z),
        mean,
        var,
    )


def second_order_taylor_expansion_edgeworth_dubins_PEZ_single(
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
    val = dubinsEZ.in_dubins_engagement_zone_single(
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
    third_moment = third_central_moment_taylor(
        dDubinsEZ_dPursuerParamsValue, d2DubinsEZ_dPursuerParamsValue, full_cov
    )
    std_dev = jnp.sqrt(var)
    skew = third_moment / std_dev**3
    skew = jnp.where(jnp.abs(skew) > 10, 0, skew)
    z = (0 - mean) / std_dev
    return (
        jax.scipy.stats.norm.cdf(z)
        + (skew / 6) * (1 - z**2) * jax.scipy.stats.norm.pdf(z),
        mean,
        var,
        third_moment,
    )
    # return (
    #     jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(var)),
    #     mean,
    #     var,    # )


second_order_taylor_expansion_edgeworth_dubins_PEZ = jax.jit(
    jax.vmap(
        second_order_taylor_expansion_edgeworth_dubins_PEZ_single,
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


def taylor3_mean_var_scalar_output(mu_f, J, H, T, Sigma):
    """
    Compute mean and variance of third-order Taylor approximation of f(x),
    where f: R^n -> R and x ~ N(mu, Sigma).

    Parameters:
    - mu_f: f(mu)           (scalar)
    - J: gradient (n,)       (∇f at mu)
    - H: Hessian (n, n)      (∇²f at mu)
    - T: third-deriv (n, n, n) (∇³f at mu)
    - Sigma: covariance (n, n)

    Returns:
    - mean: scalar
    - var: scalar
    """
    # Mean
    mean = mu_f + 0.5 * jnp.trace(H @ Sigma)

    # First-order variance term
    var1 = J @ Sigma @ J

    # Second-order variance term
    var2 = 0.5 * jnp.trace((H @ Sigma) @ (H @ Sigma))

    # Third-order variance term: T_{ijk} T_{pqr} Sigma_{ip} Sigma_{jq} Sigma_{kr}
    T_contracted = jnp.einsum("ijk,pqr,ip,jq,kr->", T, T, Sigma, Sigma, Sigma)
    var3 = (5 / 12) * T_contracted

    # Cross term: J_i T_{jkl} Σ_{ij} Σ_{kl}
    cross = jnp.einsum("i,jkl,ij,kl->", J, T, Sigma, Sigma)

    var = var1 + var2 + var3 + cross
    return mean, var


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
    val = dubinsEZ.in_dubins_engagement_zone_single(
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


cubic_dubins_PEZ = jax.jit(
    jax.vmap(
        cubic_dubins_PEZ_single,
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


def find_heading_discontinuity(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    minimumTurnRadius,
    pursuerRange,
    captureRadius,
):
    rightCenter = jnp.array(
        [
            pursuerPosition[0] + minimumTurnRadius * jnp.sin(pursuerHeading),
            pursuerPosition[1] - minimumTurnRadius * jnp.cos(pursuerHeading),
        ]
    )
    leftCenter = jnp.array(
        [
            pursuerPosition[0] - minimumTurnRadius * jnp.sin(pursuerHeading),
            pursuerPosition[1] + minimumTurnRadius * jnp.cos(pursuerHeading),
        ]
    )
    direction = jnp.array([jnp.cos(evaderHeading), jnp.sin(evaderHeading)])
    speedRatio = evaderSpeed / pursuerSpeed
    goalPosition = evaderPosition + speedRatio * pursuerRange * direction
    evaderRadius = jnp.linalg.norm(goalPosition - pursuerPosition)
    rightIntersection = circle_intersection(
        pursuerPosition, rightCenter, evaderRadius, minimumTurnRadius
    )
    leftIntersection = circle_intersection(
        pursuerPosition, leftCenter, evaderRadius, minimumTurnRadius
    )
    angleToGoal = jnp.arctan2(
        goalPosition[1] - pursuerPosition[1], goalPosition[0] - pursuerPosition[0]
    )
    rightAngles = angleToGoal - jnp.arctan2(
        rightIntersection[:, 1] - pursuerPosition[1],
        rightIntersection[:, 0] - pursuerPosition[0],
    )
    leftAngles = angleToGoal - jnp.arctan2(
        leftIntersection[:, 1] - pursuerPosition[1],
        leftIntersection[:, 0] - pursuerPosition[0],
    )
    rightAngles = pursuerHeading + rightAngles
    leftAngles = pursuerHeading + leftAngles
    allAnglesDisc = jnp.concatenate([rightAngles, leftAngles])
    return allAnglesDisc


def filter_angles(allAnglesDisc, minAngle, maxAngle):
    # Step 1: Augment angles with minAngle and maxAngle
    augmented = jnp.concatenate([allAnglesDisc, jnp.array([minAngle, maxAngle])])

    # Step 2: Create mask
    mask = jnp.logical_and(augmented >= minAngle, augmented <= maxAngle)

    # Step 3: Replace invalid with +inf and sort
    cleaned = jnp.where(mask, augmented, jnp.inf)
    sorted_cleaned = jnp.sort(cleaned)

    # Step 4: Replace infs with NaN
    filtered = jnp.where(sorted_cleaned < jnp.inf, sorted_cleaned, jnp.nan)

    # Step 5: Count how many are valid
    count_valid = jnp.sum(mask)

    return filtered, count_valid


def find_switch_angle(
    evaderPosition,
    evaderHeading,
    evaderSpeed,
    pursuerPosition,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    minimumTurnRadius,
    pursuerRange,
    captureRadius,
):
    direction = jnp.array([jnp.cos(evaderHeading), jnp.sin(evaderHeading)])
    speedRatio = evaderSpeed / pursuerSpeed
    goalPosition = evaderPosition + speedRatio * pursuerRange * direction
    goalAngle = jnp.arctan2(
        goalPosition[1] - pursuerPosition[1], goalPosition[0] - pursuerPosition[0]
    )
    return jnp.array([goalAngle - jnp.pi, goalAngle + jnp.pi])


def clockwise_angle_diff(angle, start):
    return (start - angle) % (2 * np.pi)


def find_piecwise_bounds(
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
    allAnglesDisc = find_heading_discontinuity(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        pursuerPosition,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
        captureRadius,
    )
    switchAngle = find_switch_angle(
        evaderPositions,
        evaderHeadings,
        evaderSpeed,
        pursuerPosition,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        minimumTurnRadius,
        pursuerRange,
        captureRadius,
    )
    # comnbine all angles
    allAngles = jnp.concatenate([allAnglesDisc, switchAngle])
    minAngle = pursuerHeading - 3.0 * jnp.sqrt(pursuerHeadingVar)
    maxAngle = pursuerHeading + 3.0 * jnp.sqrt(pursuerHeadingVar)
    sortedAngles, numValid = filter_angles(allAngles, minAngle, maxAngle)

    return sortedAngles, numValid


def create_linear_model(
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
    val = dubinsEZ.in_dubins_engagement_zone_single(
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
    # boundingAngles, numValid = find_piecwise_bounds(
    #     evaderPositions,
    #     evaderHeadings,
    #     evaderSpeed,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     captureRadius,
    # )
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
        a, b = create_linear_model(
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

    # for i, angle in enumerate(linearizationPoints):
    #     a, b = create_linear_model(
    #         evaderPositions,
    #         evaderHeadings,
    #         evaderSpeed,
    #         pursuerPosition,
    #         angle,
    #         pursuerSpeed,
    #         minimumTurnRadius,
    #         pursuerRange,
    #         captureRadius,
    #     )
    #     slopes.append(a)
    #     intercepts.append(b)
    #     bounds.append((boundingAngles[i], boundingAngles[i + 1]))

    prob = piecewise_linear_cdf_single_y(
        0.0,
        pursuerHeading,
        jnp.sqrt(pursuerHeadingVar),
        boundingAngles[1:],
        slopes,
        intercepts,
    )
    return prob, slopes, intercepts, boundingAngles


piecewise_linear_dubins_pez_heading_only = jax.jit(
    jax.vmap(
        piecewise_linear_dubins_PEZ_single_heading_only,
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
    val = dubinsEZ.in_dubins_engagement_zone_single(
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


def compute_peicewise_approximate_pdf(
    pursuerHeading,
    pursuerHeadingVar,
    pursuerHeadingEdges,
    pursuerSpeedEdges,
    slopes,
    intercepts,
):
    pass


def piecewise_linear_dubins_PEZ_single_heading_and_speed_pddf(
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
    numBoundingPoints = 11
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
    pursuerHeadingSlopes, pursuerSpeedSlopes, intercepts = (
        create_linear_model_heading_and_speed_vmap(
            evaderPositions,
            evaderHeadings,
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


piecewise_linear_dubins_pez_heading_and_speed_pddf = jax.jit(
    jax.vmap(
        piecewise_linear_dubins_PEZ_single_heading_and_speed_pddf,
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


def evaluate_linear_model(a, b, bounds, numSamples):
    x = jnp.linspace(bounds[0], bounds[1], numSamples)
    return a * x + b


def test_piecewise_linear_plot(
    evaderPosition,
    evaderHeading,
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
    evaderPosition = evaderPosition.flatten()
    evaderHeading = evaderHeading[0]
    prob, slopes, intercepts, bounds = piecewise_linear_dubins_PEZ_single_heading_only(
        evaderPosition,
        evaderHeading,
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
    )
    minHeading = pursuerHeading - 3 * jnp.sqrt(pursuerHeadingVar)
    maxHeading = pursuerHeading + 3 * jnp.sqrt(pursuerHeadingVar)
    numSamples = 200**2
    headingSamples = jnp.linspace(minHeading, maxHeading, numSamples).reshape(
        (numSamples,)
    )
    pursuerPositionSamples = jnp.tile(pursuerPosition, (numSamples, 1))
    turnRadiusSamples = minimumTurnRadius * jnp.ones((numSamples)).reshape(
        (numSamples,)
    )
    pursuerSpeedSamples = pursuerSpeed * jnp.ones((numSamples)).reshape((numSamples,))
    pursuerRangeSamples = pursuerRange * jnp.ones((numSamples)).reshape((numSamples,))
    ez = in_dubins_engagement_zone(
        pursuerPositionSamples,
        headingSamples,
        turnRadiusSamples,
        captureRadius,
        pursuerRangeSamples,
        pursuerSpeedSamples,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    fig2, ax2 = plt.subplots()

    for i in range(len(slopes)):
        a = slopes[i]
        b = intercepts[i]
        bound = (bounds[i], bounds[i + 1])
        y = evaluate_linear_model(a, b, bound, numSamples)
        ax2.plot(
            jnp.linspace(bound[0], bound[1], numSamples),
            y,
            label="Piecewise Linear Model",
        )

    ax2.scatter(headingSamples, ez, c="blue", label="EZ")

    fig3, ax3 = plt.subplots()
    pursuerHeadingSamples = pursuerHeading * jnp.ones((numSamples)).reshape(
        (numSamples,)
    )
    minPursuerSpeed = pursuerSpeed - 3 * jnp.sqrt(pursuerSpeedVar)
    maxPursuerSpeed = pursuerSpeed + 3 * jnp.sqrt(pursuerSpeedVar)
    pursuerSpeedSamples = jnp.linspace(minPursuerSpeed, maxPursuerSpeed, numSamples)
    ez = in_dubins_engagement_zone(
        pursuerPositionSamples,
        pursuerHeadingSamples,
        turnRadiusSamples,
        captureRadius,
        pursuerRangeSamples,
        pursuerSpeedSamples,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    ax3.scatter(pursuerSpeedSamples, ez, c="blue", label="EZ")

    fig4, ax4 = plt.subplots()
    sqrtNumSamples = int(np.sqrt(numSamples))
    pursuerHeadingSamples = jnp.linspace(
        minHeading, maxHeading, int(np.sqrt(numSamples))
    )
    pursuerSpeedSamples = jnp.linspace(
        minPursuerSpeed, maxPursuerSpeed, int(np.sqrt(numSamples))
    )
    print("pursuerHeadingSamples.shape", pursuerHeadingSamples.shape)
    print("pursuerSpeedSamples.shape", pursuerSpeedSamples.shape)
    [pursuerHeadingSamples, pursuerSpeedSamples] = jnp.meshgrid(
        pursuerHeadingSamples, pursuerSpeedSamples
    )
    print("pursuerHeadingSamples.shape", pursuerHeadingSamples.shape)
    print("pursuerSpeedSamples.shape", pursuerSpeedSamples.shape)
    ez = in_dubins_engagement_zone(
        pursuerPositionSamples,
        pursuerHeadingSamples.ravel(),
        turnRadiusSamples,
        captureRadius,
        pursuerRangeSamples,
        pursuerSpeedSamples.ravel(),
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    ez = ez.reshape(sqrtNumSamples, sqrtNumSamples)
    pursuerHeadingSamples = pursuerHeadingSamples.reshape(
        sqrtNumSamples, sqrtNumSamples
    )
    pursuerSpeedSamples = pursuerSpeedSamples.reshape(sqrtNumSamples, sqrtNumSamples)
    ax4.pcolormesh(pursuerHeadingSamples, pursuerSpeedSamples, ez)
    ax4.set_xlabel("Pursuer Heading")
    ax4.set_ylabel("Pursuer Speed")


def plot_dubins_PEZ(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
    ax,
    useLinear=False,
    useUnscented=False,
    useQuadratic=False,
    useMC=False,
    useEdgeworth=False,
    useCubic=False,
    useCombinedLinear=False,
    useCombinedQuadratic=False,
    usePiecewiseLinear=False,
):
    numPoints = 200
    if useLinear:
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        start = time.time()
        ZTrue, _, _ = linear_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        print("linear_dubins_pez time", time.time() - start)
        ax.set_title("Linear Dubins PEZ", fontsize=20)
    elif useUnscented:
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue, _, _ = uncented_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Unscented Dubins PEZ")
    elif useMC:
        rangeX = 2.0
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        start = time.time()
        ZTrue, _, _, _, _, _, _ = mc_dubins_PEZ(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        print("mc_dubins_PEZ time", time.time() - start)
        ax.set_title("Monte Carlo Dubins PEZ", fontsize=20)
    elif useQuadratic:
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue, _, _ = quadratic_dubins_pez(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Quadratic Dubins PEZ", fontsize=20)
    elif useEdgeworth:
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue, _, _, _ = second_order_taylor_expansion_edgeworth_dubins_PEZ(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
    elif useCubic:
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue, _, _ = cubic_dubins_PEZ(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Cubic Dubins PEZ", fontsize=20)
    elif useCombinedLinear:
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue = combined_left_right_dubins_PEZ(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Combined Linear Dubins PEZ", fontsize=20)
    elif useCombinedQuadratic:
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        ZTrue = combined_left_right_quadratic_dubins_PEZ(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Combined Quadratic Dubins PEZ", fontsize=20)
    elif usePiecewiseLinear:
        rangeX = 2
        x = jnp.linspace(-rangeX, rangeX, numPoints)
        y = jnp.linspace(-rangeX, rangeX, numPoints)
        [X, Y] = jnp.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        evaderHeadings = np.ones_like(X) * evaderHeading
        start = time.time()
        ZTrue, _, _, _ = piecewise_linear_dubins_pez_heading_and_speed(
            jnp.array([X, Y]).T,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        print("piecewise_linear_dubins_pez time", time.time() - start)
        ax.set_title("Piecewise Linear Dubins PEZ", fontsize=20)

    ZTrue = ZTrue.reshape(numPoints, numPoints)

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    c = ax.contour(
        X,
        Y,
        ZTrue,
        levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    plt.clabel(c, inline=True, fontsize=20)
    # c = ax.pcolormesh(X, Y, ZTrue)
    # if useUnscented:
    ax.set_xlabel("X", fontsize=26)
    ax.set_ylabel("Y", fontsize=26)
    # set tick size
    ax.tick_params(axis="both", which="major", labelsize=20)

    # ax.contour(X, Y, ZGeometric, cmap="summer")
    ax.scatter(*pursuerPosition, c="r")
    ax.set_aspect("equal", "box")
    ax.set_aspect("equal", "box")
    return ZTrue.flatten()


def plot_dubins_PEZ_diff(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadindgVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
    ax,
    useLinear=False,
    useUnscented=False,
    useQuadratic=False,
    useMC=False,
    useEdgeworth=False,
    useCubic=False,
    useCombinedLinear=False,
    useCombinedQuadratic=False,
    usePiecewiseLinear=False,
):
    numPoints = 200
    rangeX = 1.0
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    points = jnp.array([X, Y]).T
    evaderHeadings = np.ones_like(X) * evaderHeading
    ZMC, _, _, _, _, _, _ = mc_dubins_PEZ(
        points,
        evaderHeadings,
        evaderSpeed,
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadindgVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        pursuerRange,
        pursuerRangeVar,
        captureRadius,
    )
    if useLinear:
        print("Linear")
        ZTrue, _, _ = linear_dubins_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Linear Dubins PEZ", fontsize=20)
    elif useUnscented:
        print("Unscented")
        ZTrue, _, _ = uncented_dubins_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Unscented Dubins PEZ")
    elif useQuadratic:
        print("Quadratic")
        ZTrue, _, _ = quadratic_dubins_pez(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Quadratic Dubins PEZ", fontsize=20)
    elif useEdgeworth:
        print("Edgeworth")
        ZTrue, _, _, _ = second_order_taylor_expansion_edgeworth_dubins_PEZ(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
    elif useCubic:
        print("Cubic")
        ZTrue, _, _ = cubic_dubins_PEZ(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Cubic Dubins PEZ", fontsize=20)
    elif useCombinedLinear:
        print("Combined Linear")
        ZTrue = combined_left_right_dubins_PEZ(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Combined Linear Dubins PEZ", fontsize=20)
    elif useCombinedQuadratic:
        print("Combined Quadratic")
        ZTrue = combined_left_right_quadratic_dubins_PEZ(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Combined Quadratic Dubins PEZ", fontsize=20)
    elif usePiecewiseLinear:
        print("Piecewise Linear")
        ZTrue, _, _, _ = piecewise_linear_dubins_pez_heading_and_speed(
            points,
            evaderHeadings,
            evaderSpeed,
            pursuerPosition,
            pursuerPositionCov,
            pursuerHeading,
            pursuerHeadindgVar,
            pursuerSpeed,
            pursuerSpeedVar,
            minimumTurnRadius,
            minimumTurnRadiusVar,
            pursuerRange,
            pursuerRangeVar,
            captureRadius,
        )
        ax.set_title("Piecewise Linear Dubins PEZ", fontsize=20)

    rmse = jnp.sqrt(jnp.mean((ZTrue - ZMC) ** 2))
    print("RMSE", rmse)
    average_abs_diff = jnp.mean(jnp.abs(ZTrue - ZMC))
    print("Average Abs Diff", average_abs_diff)
    max_abs_diff = jnp.max(jnp.abs(ZTrue - ZMC))
    print("Max Abs Diff", max_abs_diff)
    print("points", points[jnp.argmax(jnp.abs(ZTrue - ZMC))])
    print("ztrue", ZTrue[jnp.argmax(jnp.abs(ZTrue - ZMC))])
    print("zmc", ZMC[jnp.argmax(jnp.abs(ZTrue - ZMC))])
    ZMC = ZMC.reshape(numPoints, numPoints)
    ZTrue = ZTrue.reshape(numPoints, numPoints)
    # write rmse on image
    ax.text(
        0.0,
        0.9,
        f"RMSE: {rmse:.4f}",
        fontsize=12,
        ha="center",
        va="center",
        color="white",
    )
    ax.text(
        0.0,
        0.7,
        f"Avg Abs Diff: {average_abs_diff:.4f}",
        fontsize=12,
        ha="center",
        va="center",
        color="white",
    )
    ax.text(
        0.0,
        0.5,
        f"Max Abs Diff: {max_abs_diff:.4f}",
        fontsize=12,
        ha="center",
        va="center",
        color="white",
    )

    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    c = ax.pcolormesh(X, Y, jnp.abs(ZTrue - ZMC), vmin=0, vmax=0.85)
    # make colorbar smaller
    cb = plt.colorbar(c, ax=ax, shrink=0.5)
    # if useUnscented:
    ax.set_xlabel("X", fontsize=26)
    ax.set_ylabel("Y", fontsize=26)
    # set tick size
    ax.tick_params(axis="both", which="major", labelsize=20)

    # ax.contour(X, Y, ZGeometric, cmap="summer")
    ax.scatter(*pursuerPosition, c="r")
    ax.set_aspect("equal", "box")
    ax.set_aspect("equal", "box")
    # dubinsEZ.plot_dubins_EZ(
    #     pursuerPosition,
    #     pursuerHeading,
    #     pursuerSpeed,
    #     minimumTurnRadius,
    #     captureRadius,
    #     pursuerRange,
    #     evaderHeading,
    #     evaderSpeed,
    #     ax,
    # )

    return ZTrue.flatten()


def plot_EZ_vs_pursuer_range(
    pursuerPosition,
    pursuerHeading,
    pursuerSpeed,
    pursuerRange,
    minimumTurnRadius,
    captureRadius,
    evaderPosition,
    evaderHeading,
    evaderSpeed,
):
    fig, ax = plt.subplots()
    # pursuerHeading = np.linspace(-np.pi, np.pi, 100)
    # turn radius derivat
    #
    dDubinsEZ_dRange = dDubinsEZ_dPursuerHeading(
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
    print("dDubinsEZ_dPursuerRange", dDubinsEZ_dRange)
    ezMean = in_dubins_engagement_zone(
        pursuerPosition,
        jnp.array([pursuerHeading]),
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )

    pursuerHeadingVec = np.linspace(-np.pi, np.pi, 1000)
    ez = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeadingVec,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )
    ax.scatter(pursuerHeadingVec, ez)
    # plot tangent point
    ax.plot(
        pursuerHeadingVec,
        dDubinsEZ_dRange * (pursuerHeadingVec - pursuerHeading) + ezMean,
    )


def plot_normal(mean, var, ax, label):
    x = np.linspace(mean - 3 * np.sqrt(var), mean + 3 * np.sqrt(var), 100)
    y = jax.scipy.stats.norm.pdf(x, mean, np.sqrt(var))
    ax.plot(x, y, label=label)


in_ez_right = jax.jit(
    jax.vmap(
        dubinsEZ.in_dubins_engagement_zone_right_single,
        in_axes=(None, 0, None, None, None, None, None, None, None),
    )
)
in_ez_left = jax.jit(
    jax.vmap(
        dubinsEZ.in_dubins_engagement_zone_left_single,
        in_axes=(None, 0, None, None, None, None, None, None, None),
    )
)


def compare_distribution(
    evaderPosition,
    evaderHeading,
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
    (
        inEZ,
        ez,
        pursuerPositionSamples,
        pursuerHeadingSamples,
        pursuerSpeedSamples,
        turnRadiusSamples,
        rangeSamples,
    ) = mc_dubins_PEZ(
        evaderPosition,
        evaderHeading,
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
    )
    print("mc", inEZ)
    print("mc mean", np.mean(ez))
    print("mc var", np.var(ez))
    inEZ, linMean, linVar = linear_dubins_pez(
        evaderPosition,
        evaderHeading,
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
    )
    print("lin ", inEZ)
    print("lin mean", linMean)
    print("lin var", linVar)
    inEZ, qMean, qVar = quadratic_dubins_pez(
        evaderPosition,
        evaderHeading,
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
    )
    print("quadratic", inEZ)
    print("quadratic mean", qMean)
    print("quadratic var", qVar)
    fig, ax = plt.subplots()
    inEZ, mean, var = cubic_dubins_PEZ(
        evaderPosition,
        evaderHeading,
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
    )
    print("cubic", inEZ)
    print("cubic mean", mean)
    print("cubic var", var)
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
        [
            evaderPosition.flatten(),
            jnp.array([evaderHeading[0]]),
            jnp.array([evaderSpeed]),
        ]
    )
    inEZ, meanrightlin, varrightlin, dDubinsEZRight_dPursuerParamsValue = (
        linear_dubins_pez_right(
            evaderPosition,
            evaderHeading,
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
        )
    )
    print("linear right", inEZ)
    print("linear right mean", meanrightlin)
    print("linear right var", varrightlin)
    inEZ, meanleftlin, varleftlin, dDubinsEZLeft_dPursuerParamsValue = (
        linear_dubins_pez_left(
            evaderPosition,
            evaderHeading,
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
        )
    )
    print("linear left", inEZ)
    print("linear left mean", meanleftlin)
    print("linear left var", varleftlin)
    inEZ = combined_left_right_dubins_PEZ(
        evaderPosition,
        evaderHeading,
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
    )
    print("combined linear", inEZ)
    inEZ, slopes, intercepts, bounds = piecewise_linear_dubins_pez_heading_only(
        evaderPosition,
        evaderHeading,
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
    )
    slopes = slopes.flatten()
    intercepts = intercepts.flatten()
    bounds = bounds.flatten()
    print("piecewise linear", inEZ)
    print("slopes", slopes)
    print("intercepts", intercepts)
    print("bounds", bounds)

    sortedEZindices = jnp.argsort(ez).flatten()
    ezSorted = ez.flatten()[sortedEZindices]

    plt.hist(ez, bins=1000, density=True)
    print("min turn radius", np.min(turnRadiusSamples))
    plot_normal(linMean, linVar, ax, "Linear")
    plot_normal(qMean, qVar, ax, "Quadratic")
    # plot_normal(meanrightlin, varrightlin, ax, "Right Linear")
    # plot_normal(meanleftlin, varleftlin, ax, "Left Linear")
    # plot vertical line at 0
    ax.axvline(0, color="k", linestyle="dashed", linewidth=1)
    # plot linear and unscented cdfs

    plt.legend()

    dDubinsEZ_dPursuerHeadingValue = dDubinsEZ_dPursuerHeading(
        pursuerPosition.flatten(),
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition.flatten(),
        evaderHeading[0],
        evaderSpeed,
    )
    d2DubinsEZ_dPursuerHeadingValue = d2DubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition.flatten(),
        evaderHeading[0],
        evaderSpeed,
    )
    d3DubinsEZ_dPursuerHeadingValue = d3DubinsEZ_dPursuerHeading(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition.flatten(),
        evaderHeading[0],
        evaderSpeed,
    )
    print("dDubinsEZ_dPursuerHeadingValue", dDubinsEZ_dPursuerHeadingValue)
    print("d2DubinsEZ_dPursuerHeadingValue", d2DubinsEZ_dPursuerHeadingValue)
    print("d3DubinsEZ_dPursuerHeadingValue", d3DubinsEZ_dPursuerHeadingValue)

    # right and left derivatives
    print("dDubinsEZright_dPursuerParamsValue", dDubinsEZRight_dPursuerParamsValue)
    dDubinsEZright_dPursuerHeadingValue = dDubinsEZRight_dPursuerParamsValue[0][2]
    dDubinsEZleft_dPursuerHeadingValue = dDubinsEZLeft_dPursuerParamsValue[0][2]

    print("dDubinsEZright_dPursuerHeadingValue", dDubinsEZright_dPursuerHeadingValue)
    print("dDubinsEZleft_dPursuerHeadingValue", dDubinsEZleft_dPursuerHeadingValue)

    sortIndex = np.argsort(pursuerHeadingSamples.flatten())
    ez = ez.flatten()[sortIndex]
    pursuerHeadingSamples = pursuerHeadingSamples.flatten()[sortIndex]
    rightEZ = in_ez_right(
        pursuerPosition.flatten(),
        pursuerHeadingSamples.flatten(),
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition.flatten(),
        evaderHeading[0],
        evaderSpeed,
    )
    leftEZ = in_ez_left(
        pursuerPosition.flatten(),
        pursuerHeadingSamples.flatten(),
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition.flatten(),
        evaderHeading[0],
        evaderSpeed,
    )

    fig1, ax1 = plt.subplots()
    ax1.scatter(pursuerHeadingSamples, ez, label="Monte Carlo")
    linApprox = (
        dDubinsEZ_dPursuerHeadingValue * (pursuerHeadingSamples - pursuerHeading)
        + linMean
    ).flatten()
    leftLinApprox = (
        dDubinsEZleft_dPursuerHeadingValue * (pursuerHeadingSamples - pursuerHeading)
        + meanleftlin
    ).flatten()
    rightLinApprox = (
        dDubinsEZright_dPursuerHeadingValue * (pursuerHeadingSamples - pursuerHeading)
        + meanrightlin
    ).flatten()
    quadApprox = (
        0.5
        * d2DubinsEZ_dPursuerHeadingValue
        * (pursuerHeadingSamples - pursuerHeading) ** 2
        + linApprox
    )
    cubicApprox = (
        (1 / 6)
        * d3DubinsEZ_dPursuerHeadingValue
        * (pursuerHeadingSamples - pursuerHeading) ** 3
        + quadApprox
    ).flatten()
    ax1.plot(
        pursuerHeadingSamples.flatten(),
        linApprox,
        label="Tangent",
    )
    ax1.plot(pursuerHeadingSamples.flatten(), quadApprox.flatten(), label="Quadratic")
    numSamples = 100

    for i in range(len(slopes)):
        if i % 2 == 0:
            color = "r"
        else:
            color = "lime"
        a = slopes[i]
        b = intercepts[i]
        bound = (bounds[i], bounds[i + 1])
        y = evaluate_linear_model(a, b, bound, numSamples)
        ax1.plot(
            jnp.linspace(bound[0], bound[1], numSamples),
            y,
            label="Piecewise Linear Model",
            color=color,
        )
    # ax1.scatter(pursuerHeadingSamples.flatten(), rightEZ, label="Right")
    # ax1.scatter(pursuerHeadingSamples.flatten(), leftEZ, label="Left")
    # ax1.plot(pursuerHeadingSamples.flatten(), leftLinApprox, label="Left Linear")
    # ax1.plot(pursuerHeadingSamples.flatten(), rightLinApprox, label="Right Linear")


def compare_PEZ(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
):
    evaderHeading = jnp.array([(0 / 20) * np.pi])
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    linEZ = plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[1],
        useLinear=True,
    )
    mcEZ = plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[0],
        useMC=True,
    )
    # quadEZ = plot_dubins_PEZ(
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[0][2],
    #     useQuadratic=True,
    # )
    # combined_linear = plot_dubins_PEZ(
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[1][0],
    #     useCombinedLinear=True,
    # )
    # combined_quadratic = plot_dubins_PEZ(
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[1][1],
    #     useCombinedQuadratic=True,
    # )
    piecewise_linear = plot_dubins_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[2],
        usePiecewiseLinear=True,
    )

    #
    # print("unscented rmse", np.sqrt(np.mean((usEZ - mcEZ) ** 2)))
    # print("quadratic rmse", np.sqrt(np.mean((quadEZ - mcEZ) ** 2)))
    # fig = plt.gcf()
    # fig.colorbar(c)


def plot_all_error(
    pursuerPosition,
    pursuerPositionCov,
    pursuerHeading,
    pursuerHeadingVar,
    pursuerSpeed,
    pursuerSpeedVar,
    minimumTurnRadius,
    minimumTurnRadiusVar,
    captureRadius,
    pursuerRange,
    pursuerRangeVar,
    evaderHeading,
    evaderSpeed,
):
    evaderHeading = jnp.array([(0 / 20) * np.pi])
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    linEZ = plot_dubins_PEZ_diff(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[0],
        useLinear=True,
    )
    usEZ = plot_dubins_PEZ_diff(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
        axes[1],
        usePiecewiseLinear=True,
    )
    # quadEZ = plot_dubins_PEZ_diff(
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[0][1],
    #     useQuadratic=True,
    # )
    # cubicEZ = plot_dubins_PEZ_diff(
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[0][2],
    #     useCubic=True,
    # )
    #
    # combined_linear = plot_dubins_PEZ_diff(
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[1][0],
    #     useCombinedLinear=True,
    # )
    # combined_quadratic = plot_dubins_PEZ_diff(
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     captureRadius,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     evaderHeading,
    #     evaderSpeed,
    #     axes[1][1],
    #     useCombinedQuadratic=True,
    # )


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
    pursuerPositionCov = np.array([[0.000000000001, 0.0], [0.0, 0.00000000001]])

    pursuerHeading = (0.0 / 4.0) * np.pi
    pursuerHeadingVar = 0.2

    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.3

    pursuerRange = 1.0
    pursuerRangeVar = 0.0

    minimumTurnRadius = 0.2
    minimumTurnRadiusVar = 0.0

    captureRadius = 0.0

    evaderHeading = jnp.array([(0.0 / 20.0) * np.pi])
    # evaderHeading = jnp.array((0.0 / 20.0) * np.pi)

    evaderSpeed = 0.5
    evaderPosition = np.array([[-0.25, 0.35]])
    # evaderPosition = np.array([[0.7487437185929648, 0.04522613065326636]])
    # evaderPosition = np.array([[0.205, -0.89]])

    # compare_distribution(
    #     evaderPosition,
    #     evaderHeading,
    #     evaderSpeed,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     captureRadius,
    # )
    plot_all_error(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
    )
    compare_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
    )
    compare_PEZ(
        pursuerPosition,
        pursuerPositionCov,
        pursuerHeading,
        pursuerHeadingVar,
        pursuerSpeed,
        pursuerSpeedVar,
        minimumTurnRadius,
        minimumTurnRadiusVar,
        captureRadius,
        pursuerRange,
        pursuerRangeVar,
        evaderHeading,
        evaderSpeed,
    )

    # test_piecewise_linear_plot(
    #     evaderPosition,
    #     evaderHeading,
    #     evaderSpeed,
    #     pursuerPosition,
    #     pursuerPositionCov,
    #     pursuerHeading,
    #     pursuerHeadingVar,
    #     pursuerSpeed,
    #     pursuerSpeedVar,
    #     minimumTurnRadius,
    #     minimumTurnRadiusVar,
    #     pursuerRange,
    #     pursuerRangeVar,
    #     captureRadius,
    # )

    plt.show()


if __name__ == "__main__":
    main()
