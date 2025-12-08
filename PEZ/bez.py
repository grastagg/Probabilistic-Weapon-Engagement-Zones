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

jax.config.update("jax_enable_x64", True)

# jax.config.update("jax_platform_name", "cpu")
# jax.default_device(jax.devices("cpu")[0])
# jax.config.update("jax_platform_name", "cpu")


from math import erf, sqrt
from math import erfc
import matplotlib


@jit
def inEngagementZoneJax(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    rotationMinusHeading = jnp.array(
        [
            [jnp.cos(agentHeading), jnp.sin(agentHeading)],
            [-jnp.sin(agentHeading), jnp.cos(agentHeading)],
        ]
    )
    pursuerPositionHat = rotationMinusHeading @ (pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading @ (agentPosition - pursuerPosition)

    speedRatio = agentSpeed / pursuerSpeed
    # jax.debug.print("agentPosition jax: {x}", x=agentPosition)
    # jax.debug.print("pursuerPosition jax: {x}", x=pursuerPosition)
    distance = jnp.linalg.norm(agentPosition - pursuerPosition)
    # epsilon = agentHeading + np.arctan2(agentPosition[1] - pursuerPosition[1], pursuerPosition[0] - agentPosition[0])
    # epsilon = np.arctan2(agentPositionHat[1] - pursuerPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    epsilon = jnp.arctan2(
        pursuerPositionHat[1] - agentPositionHat[1],
        pursuerPositionHat[0] - agentPositionHat[0],
    )
    # print(epsilon)
    rho = (
        speedRatio
        * pursuerRange
        * (
            jnp.cos(epsilon)
            + jnp.sqrt(
                jnp.cos(epsilon) ** 2
                - 1
                + (pursuerRange + pursuerCaptureRange) ** 2
                / (speedRatio**2 * pursuerRange**2)
            )
        )
    )

    # return distnce < rho
    # jax.debug.print("rho jax: {x}", x=rho)
    # jax.debug.print("distance jax: {x}", x=distance)
    # return distance - rho[0]
    return distance - rho


def inEngagementZoneJaxVectorized(
    agentPositions,
    agentHeadings,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    single_agent_prob = lambda agentPosition, agentHeading: inEngagementZoneJax(
        agentPosition,
        agentHeading,
        pursuerPosition,
        pursuerRange,
        pursuerCaptureRange,
        pursuerSpeed,
        agentSpeed,
    )
    # agentPosition = agentPosition.reshape(-1, 1)
    return vmap(single_agent_prob)(agentPositions, agentHeadings)


def inEngagementZone(
    agentPosition,
    agentHeading,
    pursuerPosition,
    pursuerRange,
    pursuerCaptureRange,
    pursuerSpeed,
    agentSpeed,
):
    rotationMinusHeading = np.array(
        [
            [np.cos(agentHeading), np.sin(agentHeading)],
            [-np.sin(agentHeading), np.cos(agentHeading)],
        ]
    )
    pursuerPositionHat = rotationMinusHeading @ (pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading @ (agentPosition - pursuerPosition)

    speedRatio = agentSpeed / pursuerSpeed
    # print("agentPosition not jax: ", agentPosition)
    # print("pursuerPosition not jax: ", pursuerPosition)
    distance = np.linalg.norm(agentPosition - pursuerPosition)
    # epsilon = agentHeading + np.arctan2(agentPosition[1] - pursuerPosition[1], pursuerPosition[0] - agentPosition[0])
    # epsilon = np.arctan2(agentPositionHat[1] - pursuerPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    epsilon = np.arctan2(
        pursuerPositionHat[1] - agentPositionHat[1],
        pursuerPositionHat[0] - agentPositionHat[0],
    )
    # print(epsilon)
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

    # return distance < rho
    # print("rho not jax: ", rho)
    # print("distance not jax: ", distance)
    return distance - rho
    # return rho
