import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd, value_and_grad

from scipy.stats import norm
import time
from jax import jit
import jax
from jax import vmap

from PEZ import bez

jax.config.update("jax_enable_x64", True)

# jax.config.update("jax_platform_name", "cpu")
# jax.default_device(jax.devices("cpu")[0])
# jax.config.update("jax_platform_name", "cpu")


from math import erf, sqrt
from math import erfc
import matplotlib

import PEZ.bez as bez


def monte_carlo_probalistic_engagment_zone(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
    numMonteCarloTrials,
    pursurPositionSamples=None,
    pursuerRangeSamples=None,
    pursuerCaptureRangeSamples=None,
    agentHeadingSamples=None,
    pursuerSpeedSamples=None,
    agentPositionSamples=None,
):
    if pursurPositionSamples is None:
        # randomly sample from the pursuer position distribution
        pursurPositionSamples = np.random.multivariate_normal(
            pursuerPosition.squeeze(), pursuerPositionCov, numMonteCarloTrials
        )
    # randomly sample from the pursuer position distribution

    numInEngagementZone = 0
    for i, pursuerPositionSample in enumerate(pursurPositionSamples):
        pursuerPositionSample = pursuerPositionSample.reshape(-1, 1)

        ez = bez.inEngagementZone(
            agentPositionSamples[i].reshape(-1, 1),
            agentHeadingSamples[i],
            pursuerPositionSample,
            pursuerRangeSamples[i],
            pursuerCaptureRangeSamples[i],
            pursuerSpeedSamples[i],
            agentSpeed,
        )
        if ez < 0:
            numInEngagementZone += 1

    return numInEngagementZone / numMonteCarloTrials


def analytic_epsilonJac(agentPositionHat, pursuerPositionHat):
    y_p = pursuerPositionHat[1]
    x_p = pursuerPositionHat[0]
    y_a = agentPositionHat[1]
    x_a = agentPositionHat[0]
    epsilonJac = np.zeros((1, 2))
    epsilonJac[0][0] = (
        -(y_p - y_a) / ((x_p - x_a) ** 2 * ((y_p - y_a) ** 2 / (x_p - x_a) ** 2 + 1))
    )[0]
    epsilonJac[0][1] = (1 / ((x_p - x_a) * ((y_p - y_a) ** 2 / (x_p - x_a) ** 2 + 1)))[
        0
    ]

    return epsilonJac.squeeze()


def analytic_rhoJac(epsilon, speedRatio, pursuerRange, pursuerCaptureRange):
    return (
        pursuerRange
        * speedRatio
        * (
            -(np.cos(epsilon) * np.sin(epsilon))
            / np.sqrt(
                np.cos(epsilon) ** 2
                + (pursuerCaptureRange + pursuerRange) ** 2
                / (pursuerRange**2 * speedRatio**2)
                - 1
            )
            - np.sin(epsilon)
        )
    ).squeeze()


def analytic_distJac(agentPosition, pursuerPosition):
    return -(agentPosition - pursuerPosition) / np.linalg.norm(
        agentPosition - pursuerPosition
    )


def probabalisticEngagementZone(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    start = time.time()
    rotationMinusHeading = np.array(
        [
            [np.cos(agentHeading), np.sin(agentHeading)],
            [-np.sin(agentHeading), np.cos(agentHeading)],
        ]
    )

    pursuerPositionHat = rotationMinusHeading @ (pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading @ (agentPosition - pursuerPosition)
    epsilon = np.arctan2(
        pursuerPositionHat[1] - agentPositionHat[1],
        pursuerPositionHat[0] - agentPositionHat[0],
    )

    epsilonJac = analytic_epsilonJac(agentPositionHat, pursuerPositionHat)
    speedRatio = agentSpeed / pursuerSpeed

    rho = (
        speedRatio
        * pursuerRange
        * (
            np.cos(epsilon)
            + np.sqrt(
                np.cos(epsilon) ** 2
                - 1
                + (pursuerRange + pursuerCaptureRange) ** 2
                / (speedRatio**2 * pursuerRange**2)
            )
        )
    )

    rhoJac = analytic_rhoJac(epsilon, speedRatio, pursuerRange, pursuerCaptureRange)

    dist = np.linalg.norm(agentPosition - pursuerPosition)
    distJac = analytic_distJac(agentPosition, pursuerPosition)

    overallJac = (distJac).reshape((-1, 1)).squeeze() - (
        rotationMinusHeading.T @ epsilonJac * rhoJac
    ).squeeze()

    mean = dist - rho
    cov = overallJac @ pursuerPositionCov @ overallJac.T

    diffDistribution = norm(mean, np.sqrt(cov))

    return diffDistribution.cdf(0)


# @jit
def probabalisticEngagementZoneTemp(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
    dPezDPursuerPosition,
):
    mean = bez.inEngagementZoneJax(
        agentPosition,
        agentHeading,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
    ).squeeze()
    dPezDPursuerPositionJac = dPezDPursuerPosition(
        agentPosition,
        agentHeading,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
    ).squeeze()
    cov = dPezDPursuerPositionJac @ pursuerPositionCov @ dPezDPursuerPositionJac.T
    diffDistribution = norm(mean, np.sqrt(cov))
    return diffDistribution.cdf(0)


dPezDPursuerPosition = jacfwd(bez.inEngagementZoneJax, argnums=2)
dPezDAgentPosition = jacfwd(bez.inEngagementZoneJax, argnums=0)
dPezDPursuerRange = jacfwd(bez.inEngagementZoneJax, argnums=3)
dPezDPursuerCaptureRange = jacfwd(bez.inEngagementZoneJax, argnums=4)
dPezDAgentHeading = jacfwd(bez.inEngagementZoneJax, argnums=1)
dPezDPusuerSpeed = jacfwd(bez.inEngagementZoneJax, argnums=5)


def probabalisticEngagementZoneVectorizedTemp(
    agentPositions,
    agentPositionCov,
    agentHeadings,
    agentHeadingVar,
    pursuerPosition,
    pursuerPositionCov,
    pursuerRange,
    pursuerRangeVar,
    pursuerCaptureRange,
    pursuerCaptureRangeVar,
    pursuerSpeed,
    pusuerSpeedVar,
    agentSpeed,
):
    # Define vectorized operations with vmap
    def single_agent_prob(agentPosition, agentHeading):
        agentPosition = agentPosition.reshape(
            -1,
        )

        # Calculate the mean for the engagement zone
        mean = bez.inEngagementZoneJax(
            agentPosition,
            agentHeading,
            pursuerPosition.squeeze(),
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()

        # Calculate the Jacobian for the engagement zone
        dPezDPursuerPositionJac = dPezDPursuerPosition(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDAgentPositionJac = dPezDAgentPosition(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDPursuerRangeJac = dPezDPursuerRange(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDPursuerCaptureRangeJac = dPezDPursuerCaptureRange(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDAgentHeadingJac = dPezDAgentHeading(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()
        dPezDPursuerSpeedJac = dPezDPusuerSpeed(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        ).squeeze()

        # Compute the covariance matrix
        cov = (
            dPezDPursuerPositionJac @ pursuerPositionCov @ dPezDPursuerPositionJac.T
            + dPezDAgentPositionJac @ agentPositionCov @ dPezDAgentPositionJac.T
            + dPezDPursuerRangeJac**2 * pursuerRangeVar
            + dPezDPursuerCaptureRangeJac**2 * pursuerCaptureRangeVar
            + dPezDAgentHeadingJac**2 * agentHeadingVar
            + dPezDPursuerSpeedJac**2 * pusuerSpeedVar
        )

        # Return the CDF at 0
        return jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(cov))

    # Apply vectorization over agentPositions and agentHeadings
    return vmap(single_agent_prob)(agentPositions, agentHeadings)
