"""Deterministic BEZ boundary functions.

The repository uses these functions as the baseline deterministic engagement
zone model. Public names are preserved exactly because they are imported in
multiple other folders.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

jax.config.update("jax_enable_x64", True)


def inEngagementZoneJax(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    """Return the implicit BEZ boundary function evaluated with JAX."""
    x_p = pursuerPosition[0]
    y_p = pursuerPosition[1]
    x_e, y_e = agentPosition
    psi_e = agentHeading
    speed_ratio = agentSpeed / pursuerSpeed

    return (
        (x_e - x_p + speed_ratio * pursuerRange * jnp.cos(psi_e)) ** 2
        + (y_e - y_p + speed_ratio * pursuerRange * jnp.sin(psi_e)) ** 2
        - (pursuerRange + pursuerCaptureRange) ** 2
    )


def inEngagementZoneJaxVectorized(
    agentPositions,
    agentHeadings,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    """Vectorized JAX BEZ evaluation over many agent states."""

    def single_agent_value(agentPosition, agentHeading):
        return inEngagementZoneJax(
            agentPosition,
            agentHeading,
            pursuerPosition,
            pursuerRange,
            pursuerCaptureRange,
            pursuerSpeed,
            agentSpeed,
        )

    return vmap(single_agent_value)(agentPositions, agentHeadings)


def inEngagementZone(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    """Return ``distance - rho`` for the deterministic BEZ geometry."""
    rotation_minus_heading = np.array(
        [
            [np.cos(agentHeading), np.sin(agentHeading)],
            [-np.sin(agentHeading), np.cos(agentHeading)],
        ]
    )
    pursuerPositionHat = rotation_minus_heading @ (pursuerPosition - pursuerPosition)
    agentPositionHat = rotation_minus_heading @ (agentPosition - pursuerPosition)

    speedRatio = agentSpeed / pursuerSpeed
    distance = np.linalg.norm(agentPosition - pursuerPosition)
    epsilon = np.arctan2(
        pursuerPositionHat[1] - agentPositionHat[1],
        pursuerPositionHat[0] - agentPositionHat[0],
    )
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

    return distance - rho
