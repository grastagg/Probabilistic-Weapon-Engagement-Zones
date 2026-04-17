"""Probabilistic engagement-zone calculations built on the deterministic BEZ."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, vmap
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)

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
        pursurPositionSamples = np.random.multivariate_normal(
            pursuerPosition.squeeze(), pursuerPositionCov, numMonteCarloTrials
        )

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
