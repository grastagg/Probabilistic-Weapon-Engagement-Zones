import numpy as np

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import time


import dubinsEZ


jax.config.update("jax_enable_x64", False)

numPoints = 45
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
# Vectorized function using vmap
in_dubins_engagement_zone3 = jax.jit(
    jax.vmap(
        dubinsEZ.in_dubins_engagement_zone_single,
        in_axes=(
            0,  # pursuerPosition
            0,  # pursuerHeading
            0,  # minimumTurnRadius
            None,  # catureRadius
            None,  # pursuerRange
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
# with ko as target:
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
# d2DubinsEZ_dPursuerPosition = jax.jacfwd(dDubinsEZ_dPursuerPosition, 0)
# d2DubinsEZ_dPursuerHeading = jax.jacfwd(dDubinsEZ_dPursuerHeading, 1)
# d2DubinsEZ_dMinimumTurnRadius = jax.jacfwd(dDubinsEZ_dMinimumTurnRadius, 2)
# d2DubinsEZ_dCaptureRadius = jax.jacfwd(dDubinsEZ_dCaptureRadius, 3)
# d2DubinsEZ_dPursuerRange = jax.jacfwd(dDubinsEZ_dPursuerRange, 4)
# d2DubinsEZ_dPursuerSpeed = jax.jacfwd(dDubinsEZ_dPursuerSpeed, 5)
# d2DubinsEZ_dEvaderPosition = jax.jacfwd(dDubinsEZ_dEvaderPosition, 6)
#
# d3DubinsEZ_dPursuerHeading = jax.jacfwd(d2DubinsEZ_dPursuerHeading, 1)
# d4DubinsEZ_dPursuerHeading = jax.jacfwd(d3DubinsEZ_dPursuerHeading, 1)
# d5DubinsEZ_dPursuerHeading = jax.jacfwd(d4DubinsEZ_dPursuerHeading, 1)
# d6DubinsEZ_dPursuerHeading = jax.jacfwd(d5DubinsEZ_dPursuerHeading, 1)

# first order combined derivatives
dDubinsEZ_dPursuerParams = jax.jacfwd(dubins_EZ_single_combined_input, 0)
# dDubinsEZRight_dPursuerParams = jax.jacfwd(dubins_EZ_single_right_combined_input, 0)
# dDubinsEZLeft_dPursuerParams = jax.jacfwd(dubins_EZ_single_left_combined_input, 0)

# sevond order combined derivatives
# d2DubinsEZ_dPursuerParams = jax.jacfwd(dDubinsEZ_dPursuerParams, 0)
# d2DubinsEZRight_dPursuerParams = jax.jacfwd(dDubinsEZRight_dPursuerParams, 0)
# d2DubinsEZLeft_dPursuerParams = jax.jacfwd(dDubinsEZLeft_dPursuerParams, 0)
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


# quadratic_dubins_pez = jax.jit(
#     jax.vmap(
#         quadratic_dubins_PEZ_single,
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
    jax.debug.print(
        "pursuerPositionXGrid {pursuerPositionXGrid}",
        pursuerPositionXGrid=pursuerPositionXGrid,
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
def dubins_pez_numerical_integration_heading_and_speed(
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
    numSubdivisions = 13
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
    # cellProb = jnp.where(Z < 0, peicewiseAveragePdf * cellArea, 0.0)
    probs = jnp.sum(cellProb, axis=1)

    return probs, 0, 0, 0


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
#     val = dubinsEZ.in_dubins_engagement_zone_right_single(
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

    z = dubinsEZ.in_dubins_engagement_zone_single(
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
        ZTrue, _, _, _ = dubins_pez_numerical_integration_heading_and_speed(
            # ZTrue, _, _, _ = numerical_dubins_pez(
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
        print("numerical integration time", time.time() - start)
        ax.set_title("Numerical Integration Dubins PEZ", fontsize=20)

    ZTrue = np.array(ZTrue.block_until_ready())
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
    ZMC = np.array(ZMC.block_until_ready())
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
        ZTrue, _, _, _ = dubins_pez_numerical_integration_heading_and_speed(
            # ZTrue, _, _, _ = numerical_dubins_pez(
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
        ax.set_title("Numerical Dubins PEZ", fontsize=20)

    ZTrue = np.array(ZTrue.block_until_ready())
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
    # pursuerPositionCov = np.array([[0.000000000001, 0.0], [0.0, 0.00000000001]])

    pursuerHeading = (0.0 / 4.0) * np.pi
    pursuerHeadingVar = 0.5

    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.3

    pursuerRange = 1.0
    pursuerRangeVar = 0.1
    pursuerRangeVar = 0.0

    minimumTurnRadius = 0.2
    minimumTurnRadiusVar = 0.005
    # minimumTurnRadiusVar = 0.00000000001

    captureRadius = 0.0

    evaderHeading = jnp.array([(5.0 / 20.0) * np.pi])
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
    # compare_PEZ(
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
    # )
    # compare_PEZ(
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
    # )

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
    # numerical_dubins_pez(
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
