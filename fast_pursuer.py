import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd, value_and_grad

from scipy.stats import norm
import time
from jax import jit
import jax
from jax import vmap
jax.config.update("jax_enable_x64", True)
from jax.lib import xla_bridge
jax.default_device(jax.devices("cpu")[0])
print(xla_bridge.get_backend().platform)


from math import erf, sqrt
from math import erfc



pursuerRange = .8
pursuerCaptureRange = 0.1
pursuerSpeed = 1
agentSpeed = .9


def inEngagementZone(agentPosition,agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    rotationMinusHeading = np.array([[np.cos(agentHeading), np.sin(agentHeading)], [-np.sin(agentHeading), np.cos(agentHeading)]])
    pursuerPositionHat = rotationMinusHeading@(pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading@(agentPosition - pursuerPosition)

    speedRatio = agentSpeed / pursuerSpeed
    distance = np.linalg.norm(agentPosition - pursuerPosition)
    # epsilon = agentHeading + np.arctan2(agentPosition[1] - pursuerPosition[1], pursuerPosition[0] - agentPosition[0])
    # epsilon = np.arctan2(agentPositionHat[1] - pursuerPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    epsilon = np.arctan2(pursuerPositionHat[1] - agentPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    # print(epsilon)
    rho = speedRatio*pursuerRange*(np.cos(epsilon) + np.sqrt(np.cos(epsilon)**2 - 1+(pursuerRange+pursuerCaptureRange)**2/(speedRatio**2*pursuerRange**2)))
    
    # return distance < rho
    return distance - rho
    # return rho

def plotMalhalanobisDistance(pursuerPosition, pursuerPositionCov, ax):
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    [X, Y] = np.meshgrid(x, y)

    malhalanobisDistance = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            malhalanobisDistance[i,j] = np.linalg.norm((np.array([[X[i,j]],[Y[i,j]]]) - pursuerPosition).T@np.linalg.inv(pursuerPositionCov)@(np.array([[X[i,j]],[Y[i,j]]]) - pursuerPosition))
    c = ax.contourf(X, Y, malhalanobisDistance, levels=[0,1,2,3])
    ax.scatter(pursuerPosition[0], pursuerPosition[1], color='red')

def plotEngagementZone(agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax):
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    [X, Y] = np.meshgrid(x, y)
    agentPositions = jnp.vstack([X.ravel(), Y.ravel()]).T
    agentHeadings = jnp.ones(agentPositions.shape[0]) * agentHeading
    engagementZonePlot = inEngagementZoneJaxVectorized(agentPositions, agentHeadings, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)

    # engagementZonePlot = np.zeros(X.shape)

    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         engagementZonePlot[i, j] = inEngagementZone(np.array([[X[i, j]], [Y[i, j]]]), agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
    # c = plt.pcolormesh(X, Y, engagementZonePlot)
    # plt.colorbar(c, ax=ax)
    # c = plt.Circle([0,0], pursuerRange+pursuerCaptureRange, fill=False)
    # ax.add_artist(c)
    # plt.contour(X, Y, engagementZonePlot, levels=[0, 1], colors=['red'])
    ax.contour(X, Y, engagementZonePlot.reshape((50,50)), levels=[0], colors=['red'])
    ax.scatter(pursuerPosition[0], pursuerPosition[1], color='red')

    
    return


def analytic_epsilonJac(agentPositionHat, pursuerPositionHat):
    y_p = pursuerPositionHat[1]
    x_p = pursuerPositionHat[0]
    y_a = agentPositionHat[1]
    x_a = agentPositionHat[0]
    epsilonJac = np.zeros((1,2))
    epsilonJac[0][0] = (-(y_p-y_a)/((x_p-x_a)**2*((y_p-y_a)**2/(x_p-x_a)**2+1)))[0]
    epsilonJac[0][1] = (1/((x_p-x_a)*((y_p-y_a)**2/(x_p-x_a)**2+1)))[0]
    
    return epsilonJac.squeeze() 

def analytic_rhoJac(epsilon, speedRatio, pursuerRange, pursuerCaptureRange):
    return (pursuerRange*speedRatio*(-(np.cos(epsilon)*np.sin(epsilon))/np.sqrt(np.cos(epsilon)**2+(pursuerCaptureRange+pursuerRange)**2/(pursuerRange**2*speedRatio**2)-1)-np.sin(epsilon))).squeeze()

def analytic_distJac(agentPosition, pursuerPosition):
    return -(agentPosition - pursuerPosition)/np.linalg.norm(agentPosition - pursuerPosition)


def probabalisticEngagementZone(agentPosition, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    start = time.time()
    rotationMinusHeading = np.array([[np.cos(agentHeading), np.sin(agentHeading)], [-np.sin(agentHeading), np.cos(agentHeading)]])

    pursuerPositionHat = rotationMinusHeading@(pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading@(agentPosition - pursuerPosition)
    epsilon = np.arctan2(pursuerPositionHat[1] - agentPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])

    epsilonJac = analytic_epsilonJac(agentPositionHat, pursuerPositionHat)
    speedRatio = agentSpeed / pursuerSpeed
    
    rho = speedRatio*pursuerRange*(np.cos(epsilon) + np.sqrt(np.cos(epsilon)**2 - 1+(pursuerRange+pursuerCaptureRange)**2/(speedRatio**2*pursuerRange**2)))
    
    rhoJac = analytic_rhoJac(epsilon, speedRatio, pursuerRange, pursuerCaptureRange)

    dist = np.linalg.norm(agentPosition - pursuerPosition)
    distJac = analytic_distJac(agentPosition, pursuerPosition)
    
    
    overallJac = (distJac).reshape((-1,1)).squeeze() -(rotationMinusHeading.T@epsilonJac*rhoJac).squeeze()

    mean = dist - rho
    cov = overallJac@pursuerPositionCov@overallJac.T

    diffDistribution = norm(mean, np.sqrt(cov))
    
    return diffDistribution.cdf(0)

@jit
def inEngagementZoneJax(agentPosition,agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    rotationMinusHeading = jnp.array([[jnp.cos(agentHeading), jnp.sin(agentHeading)], [-jnp.sin(agentHeading), jnp.cos(agentHeading)]])
    pursuerPositionHat = rotationMinusHeading@(pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading@(agentPosition - pursuerPosition)

    speedRatio = agentSpeed / pursuerSpeed
    distance = jnp.linalg.norm(agentPosition - pursuerPosition)
    # epsilon = agentHeading + np.arctan2(agentPosition[1] - pursuerPosition[1], pursuerPosition[0] - agentPosition[0])
    # epsilon = np.arctan2(agentPositionHat[1] - pursuerPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    epsilon = jnp.arctan2(pursuerPositionHat[1] - agentPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    # print(epsilon)
    rho = speedRatio*pursuerRange*(jnp.cos(epsilon) + jnp.sqrt(jnp.cos(epsilon)**2 - 1+(pursuerRange+pursuerCaptureRange)**2/(speedRatio**2*pursuerRange**2)))
    
    # return distnce < rho
    # jax.debug.print("agen: {x}", x=rho)
    return (distance - rho[0])

def inEngagementZoneJaxVectorized(agentPositions,agentHeadings, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    single_agent_prob = lambda agentPosition, agentHeading: inEngagementZoneJax(agentPosition.reshape(-1,1), agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
        # agentPosition = agentPosition.reshape(-1, 1)
    return vmap(single_agent_prob)(agentPositions, agentHeadings)


    
# @jit
def probabalisticEngagementZoneTemp(agentPosition, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,dPezDPursuerPosition):
    mean = inEngagementZoneJax(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
    dPezDPursuerPositionJac = dPezDPursuerPosition(agentPosition,agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
    cov = dPezDPursuerPositionJac@pursuerPositionCov@dPezDPursuerPositionJac.T
    diffDistribution = norm(mean, np.sqrt(cov))
    return diffDistribution.cdf(0)

dPezDPursuerPosition = jacfwd(inEngagementZoneJax, argnums=2)
dPezDAgentPosition = jacfwd(inEngagementZoneJax, argnums=0)
dPezDPursuerRange = jacfwd(inEngagementZoneJax, argnums=3)
dPezDPursuerCaptureRange = jacfwd(inEngagementZoneJax, argnums=4)
dPezDAgentHeading = jacfwd(inEngagementZoneJax, argnums=1)
def probabalisticEngagementZoneVectorizedTemp(agentPositions, agentPositionCov, agentHeadings, agentHeadingVar, 
                                              pursuerPosition, pursuerPositionCov, pursuerRange, pursuerRangeVar, 
                                              pursuerCaptureRange, pursuerCaptureRangeVar, pursuerSpeed, agentSpeed):
    
    # Define vectorized operations with vmap
    def single_agent_prob(agentPosition, agentHeading):
        agentPosition = agentPosition.reshape(-1, 1)
        
        # Calculate the mean for the engagement zone
        mean = inEngagementZoneJax(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()

        # Calculate the Jacobian for the engagement zone
        dPezDPursuerPositionJac = dPezDPursuerPosition(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
        dPezDAgentPositionJac = dPezDAgentPosition(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
        dPezDPursuerRangeJac = dPezDPursuerRange(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
        dPezDPursuerCaptureRangeJac = dPezDPursuerCaptureRange(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
        dPezDAgentHeadingJac = dPezDAgentHeading(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()

        # Compute the covariance matrix
        cov = (dPezDPursuerPositionJac @ pursuerPositionCov @ dPezDPursuerPositionJac.T + 
               dPezDAgentPositionJac @ agentPositionCov @ dPezDAgentPositionJac.T + 
               dPezDPursuerRangeJac**2 * pursuerRangeVar + 
               dPezDPursuerCaptureRangeJac**2 * pursuerCaptureRangeVar + 
               dPezDAgentHeadingJac**2 * agentHeadingVar)

        # Return the CDF at 0
        return jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(cov))
    
    # Apply vectorization over agentPositions and agentHeadings
    # print("agentPositions: ", agentPositions.shape)
    # print("agentHeadings: ", agentHeadings.shape)
    return vmap(single_agent_prob)(agentPositions, agentHeadings)

# def  probabalisticEngagementZoneVectorizedTemp(agentPositions,agentPositionCov, agentHeading,agentHeadingVar, pursuerPosition, pursuerPositionCov, pursuerRange,pursuerRangeVar, pursuerCaptureRange,pursuerCaptureRangeVar, pursuerSpeed, agentSpeed, dPezDPursuerPosition, dPezDAgentPosition,dPezDPursuerRange,dPezDPursuerCaptureRange,dPezDAgentHeading):
#     # Define vectorized operations with vmap
#     def single_agent_prob(agentPosition):
#         agentPosition = agentPosition.reshape(-1,1)
#         # Calculate the mean for the engagement zone
#         mean = inEngagementZoneJax(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()

#         # Calculate the Jacobian for the engagement zone
#         dPezDPursuerPositionJac = dPezDPursuerPosition(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
#         dPezDAgentPositionJac = dPezDAgentPosition(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
#         dPezDPursuerRangeJac = dPezDPursuerRange(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
#         dPezDPursuerCaptureRangeJac = dPezDPursuerCaptureRange(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
#         dPezDAgentHeadingJac = dPezDAgentHeading(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()

#         # Compute the covariance matrix
#         cov = dPezDPursuerPositionJac @ pursuerPositionCov @ dPezDPursuerPositionJac.T + dPezDAgentPositionJac @ agentPositionCov @ dPezDAgentPositionJac.T+ dPezDPursuerRangeJac**2 * pursuerRangeVar + dPezDPursuerCaptureRangeJac**2 * pursuerCaptureRangeVar+ dPezDAgentHeadingJac**2 * agentHeadingVar

#         # Define the normal distribution based on mean and covariance
#         # diffDistribution = jax.scipy.stats.norm(mean, jnp.sqrt(cov))
#         # jax.scipy.stats.norm.
        
#         # # Return the CDF at 0
#         # return diffDistribution.cdf(0)
#         return jax.scipy.stats.norm.cdf(0, mean, jnp.sqrt(cov))
    

#     # Apply vectorization over agentPositions
#     return vmap(single_agent_prob)(agentPositions)

# def plotProbablisticEngagemenpedalck([X.ravel(), Y.ravel()]).T


#     dPezDPursuerPosition = jacfwd(inEngagementZoneJax, argnums=2)

    
#     engagementZonePlot = np.zeros(X.shape)
    

#     totalTime = 0
    
#     engagementZonePlot = probabalisticEngagementZoneVectorizedTemp(agentPositions, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,dPezDPursuerPosition)
#     # for i in range(X.shape[0]):
#     #     print(i)
#     #     for j in range(X.shape[1]):
#     #         start = time.time()
#     #         # engagementZonePlot[i, j] = probabalisticEngagementZone(np.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
#     #         engagementZonePlot[i, j] = probabalisticEngagementZoneTemp(jnp.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,dPezDPursuerPosition)
#             # totalTime += time.time() - start
#     # print(totalTime)
#     # c = ax.pcolormesh(X, Y, engagementZonePlot)
#     # c = plt.Circle(pursuerPosition, pursuerRange+pursuerCaptureRange, fill=False)
#     # ax.add_artist(c)
#     c = ax.contour(X, Y, engagementZonePlot.reshape(50,50), levels=np.linspace(0,1,11))
#     ax.clabel(c, inline=True, fontsize=8)
    
#     return
def plotProbablisticEngagementZone(agentPositionCov,agentHeading,agentHeadingVar, pursuerPosition, pursuerPositionCov, pursuerRange, pursuerRangeVar,pursuerCaptureRange,pursuerCaptureRangeVar, pursuerSpeed, agentSpeed, ax):
    ax.set_aspect('equal')
    ax.set_title("Linearized Probabilistic Engagement Zone")

    # Define the grid
    x = jnp.linspace(-2, 2, 50)
    y = jnp.linspace(-2, 2, 50)
    X, Y = jnp.meshgrid(x, y)
    agentPositions = jnp.vstack([X.ravel(), Y.ravel()]).T
    agentHeadings = jnp.ones(agentPositions.shape[0]) * agentHeading

    # Compute Jacobian of engagement zone function
    # dPezDPursuerPosition = jacfwd(inEngagementZoneJax, argnums=2)
    # dPezDAgentPosition = jacfwd(inEngagementZoneJax, argnums=0)
    # dPezDPursuerRange = jacfwd(inEngagementZoneJax, argnums=3)
    # dPezDPursuerCaptureRange = jacfwd(inEngagementZoneJax, argnums=4)
    # dPezDAgentHeading = jacfwd(inEngagementZoneJax, argnums=1)

    start = time.time()
    # Compute engagement zone probabilities
    engagementZonePlot = probabalisticEngagementZoneVectorizedTemp(
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
        agentSpeed
    )
    print("total time: ", time.time() - start)

    # Convert result to NumPy array for plotting
    engagementZonePlot_np = np.array(engagementZonePlot)

    # Reshape for contour plotting
    engagementZonePlot_reshaped = engagementZonePlot_np.reshape(X.shape)

    # Plotting
    c = ax.contour(X, Y, engagementZonePlot_reshaped, levels=np.linspace(0, 1, 11))
    ax.clabel(c, inline=True, fontsize=8)
    
    # Add circle representing pursuer's range
    # for i in range(3):
    #     c = plt.Circle(pursuerPosition, pursuerRange + i*np.sqrt(pursuerRangeVar) + pursuerCaptureRange, fill=False, color='r', linestyle='--')
    #     ax.add_artist(c)
    c = plt.Circle(pursuerPosition, pursuerRange, fill=False, color='r', linestyle='--')
    ax.add_artist(c)

    return

def plotMCProbablisticEngagementZone(agentPositionCov,agentHeading,agentHeadingVar, pursuerPosition,pursuerPositionCov, pursuerRange,pursuerRangeCov, pursuerCaptureRange,pursuerCaptureRangeVar, pursuerSpeed, agentSpeed,ax):
    ax.set_title("Monte Carlo Probabalistic Engagement Zone")
    ax.set_aspect('equal')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    [X, Y] = np.meshgrid(x, y)

    
    malhalanobisDistance = np.zeros(X.shape)
    engagementZonePlot = np.zeros(X.shape)

    numMcTrials = 1000

    pursuerPositionSamples = np.random.multivariate_normal(pursuerPosition.squeeze(), pursuerPositionCov, numMcTrials)
    pursuerRangeSamples = np.random.normal(pursuerRange, np.sqrt(pursuerRangeCov), numMcTrials)
    pursuerCaptureRangeSamples = np.random.normal(pursuerCaptureRange, np.sqrt(pursuerCaptureRangeVar), numMcTrials)
    agentHeadingSamples = np.random.normal(agentHeading, np.sqrt(agentHeadingVar), numMcTrials)
    

    for i in range(X.shape[0]):
        print(i)
        for j in range(X.shape[1]):
            # engagementZonePlot[i, j] = probabalisticEngagementZone(np.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
            engagementZonePlot[i, j] = monte_carlo_probalistic_engagment_zone(np.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, numMcTrials, pursuerPositionSamples,pursuerRangeSamples,pursuerCaptureRangeSamples,agentHeadingSamples)
    # c = ax.pcolormesh(X, Y, engagementZonePlot)
    # c = plt.Circle(pursuerPosition, pursuerRange+pursuerCaptureRange, fill=False)
    # ax.add_artist(c)
    c = ax.contour(X, Y, engagementZonePlot, levels=np.linspace(0,1,11))
    ax.clabel(c, inline=True, fontsize=8)
    
    return c

def monte_carlo_probalistic_engagment_zone(agentPosition, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, numMonteCarloTrials, pursurPositionSamples=None,pursuerRangeSamples=None,pursuerCaptureRangeSamples=None,agentHeadingSamples=None):
    if pursurPositionSamples is None:
        #randomly sample from the pursuer position distribution
        pursurPositionSamples = np.random.multivariate_normal(pursuerPosition.squeeze(), pursuerPositionCov, numMonteCarloTrials)
    #randomly sample from the pursuer position distribution
    
    numInEngagementZone = 0
    for i,pursuerPositionSample in enumerate(pursurPositionSamples):
        pursuerPositionSample = pursuerPositionSample.reshape(-1,1)
        if inEngagementZone(agentPosition, agentHeadingSamples[i], pursuerPositionSample, pursuerRangeSamples[i], pursuerCaptureRangeSamples[i], pursuerSpeed, agentSpeed) < 0:
            numInEngagementZone += 1
    
    return numInEngagementZone/numMonteCarloTrials
    
    
    


# probabalisticEngagementZone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, np.array([[0.1, 0], [0, 0.1]]), pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
# inEngagementZone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)

def main():
    pursuerRange = .8
    pursuerRangeVar = 0.2
    pursuerCaptureRange = 0.2
    pursuerCaptureRangeVar = 0.1
    pursuerSpeed = 1
    agentSpeed = .8

    agentPositionCov = np.array([[0.0, 0.0], [0.0, 0.0]])
    pursuerPositionCov = np.array([[0.1, 0.0], [0.1, 0.0]])
    # pursuerInitialPosition = jnp.array([-.5, .5])
    pursuerInitialPosition = np.array([[0.0], [0.0]])
    agentInitialPosition = np.array([[0.0], [2.0]])

    agentInitialHeading = 0.0
    agentHeadingVar = 0.0
    

    

    # mcpez = monte_carlo_probalistic_engagment_zone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, 100)
    # print(mcpez)

    # fig, ax = plt.subplots()
    # plotMCProbablisticEngagementZone(agentPositionCov,agentInitialHeading,agentHeadingVar, pursuerInitialPosition, pursuerPositionCov, pursuerRange,pursuerRangeVar, pursuerCaptureRange,pursuerCaptureRangeVar, pursuerSpeed, agentSpeed, ax)
    # plotEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax)   
    # plotMalhalanobisDistance(pursuerInitialPosition, pursuerPositionCov, ax)


    fig1, ax1 = plt.subplots()
    plotProbablisticEngagementZone(agentPositionCov,agentInitialHeading,agentHeadingVar, pursuerInitialPosition, pursuerPositionCov, pursuerRange, pursuerRangeVar, pursuerCaptureRange,pursuerCaptureRangeVar, pursuerSpeed, agentSpeed,ax1)
    plotEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax1)   
    # plotMalhalanobisDistance(pursuerInitialPosition, pursuerPositionCov, ax1)
    plt.show()

    

if __name__ == "__main__":
    main()