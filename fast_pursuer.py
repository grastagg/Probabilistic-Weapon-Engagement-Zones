import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd
from scipy.stats import norm



pursuerRange = .8
pursuerCaptureRange = 0.1
pursuerSpeed = 1
agentSpeed = .9


# pursuerInitialPosition = jnp.array([-.5, .5])
pursuerInitialPosition = jnp.array([[0.0], [0.0]])
agentInitialPosition = jnp.array([[0.0], [2.0]])
agentInitialHeading = 0

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

def plotMalhalanobisDistance(agentPosition, agentHeading, pursuerPosition, pursuerPositionCov, ax):
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

    engagementZonePlot = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            engagementZonePlot[i, j] = inEngagementZone(np.array([[X[i, j]], [Y[i, j]]]), agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
    # c = plt.pcolormesh(X, Y, engagementZonePlot)
    # plt.colorbar(c, ax=ax)
    c = plt.Circle([0,0], pursuerRange+pursuerCaptureRange, fill=False)
    ax.add_artist(c)
    # plt.contour(X, Y, engagementZonePlot, levels=[0, 1], colors=['red'])
    ax.contour(X, Y, engagementZonePlot, levels=[0], colors=['red'])
    ax.scatter(pursuerPosition[0], pursuerPosition[1], color='red')

    
    return


def probabalisticEngagementZone(agentPosition, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    rotationMinusHeading = np.array([[np.cos(agentHeading), np.sin(agentHeading)], [-np.sin(agentHeading), np.cos(agentHeading)]])

    pursuerPositionHat = rotationMinusHeading@(pursuerPosition - pursuerPosition)
    agentPositionHat = rotationMinusHeading@(agentPosition - pursuerPosition)
    pursuerPositionCovHat = rotationMinusHeading.T@pursuerPositionCov@rotationMinusHeading
    epsilon = np.arctan2(pursuerPositionHat[1] - agentPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    # epsilon = np.arctan2(agentPositionHat[1] - pursuerPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    # dEpsilondpursuerPosition = jacfwd(lambda x: jnp.arctan2(agentPositionHat[1] - x[1], x[0] - agentPositionHat[0]))(pursuerPositionHat).squeeze()
    dEpsilondpursuerPosition = jacfwd(lambda x: jnp.arctan2(x[1] - agentPositionHat[1], x[0] - agentPositionHat[0]))(pursuerPositionHat).squeeze()
    epsilonCov = dEpsilondpursuerPosition.T@pursuerPositionCovHat@dEpsilondpursuerPosition

    speedRatio = agentSpeed / pursuerSpeed
    
    rho = speedRatio*pursuerRange*(np.cos(epsilon) + np.sqrt(np.cos(epsilon)**2 - 1+(pursuerRange+pursuerCaptureRange)**2/(speedRatio**2*pursuerRange**2)))
    rhoJac = np.array(jacfwd(lambda x: speedRatio*pursuerRange*(jnp.cos(x) + jnp.sqrt(jnp.cos(x)**2 - 1+(pursuerRange+pursuerCaptureRange)**2/(speedRatio**2*pursuerRange**2))))(epsilon))
    rhoCov = rhoJac**2*epsilonCov
    # dist = np.linalg.norm(agentPositionHat - pursuerPositionHat)
    # ddistdpursuerPosition = jacfwd(lambda x: jnp.linalg.norm(agentPositionHat - x))(pursuerPositionHat)
    # dist = np.linalg.norm(pursuerPositionHat - agentPositionHat)
    # ddistdpursuerPosition = jacfwd(lambda x: jnp.linalg.norm(x - agentPositionHat))(pursuerPositionHat)
    dist = np.linalg.norm(agentPosition - pursuerPosition)
    ddistdpursuerPosition = jacfwd(lambda x: jnp.linalg.norm(agentPosition - x))(pursuerPosition)
    distCov = ddistdpursuerPosition.T@pursuerPositionCov@ddistdpursuerPosition

    differenceMean = dist - rho.squeeze()
    differenceCov = distCov.squeeze() + rhoCov.squeeze()

    diffDistribution = norm(differenceMean, np.sqrt(differenceCov))
    
    return diffDistribution.cdf(0)

def plotProbablisticEngagementZone(agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax):
    ax.set_aspect('equal')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    [X, Y] = np.meshgrid(x, y)

    
    malhalanobisDistance = np.zeros(X.shape)
    engagementZonePlot = np.zeros(X.shape)

    for i in range(X.shape[0]):
        print(i)
        for j in range(X.shape[1]):
            engagementZonePlot[i, j] = probabalisticEngagementZone(np.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
    # c = ax.pcolormesh(X, Y, engagementZonePlot)
    # c = plt.Circle(pursuerPosition, pursuerRange+pursuerCaptureRange, fill=False)
    # ax.add_artist(c)
    c = ax.contour(X, Y, engagementZonePlot, levels=np.linspace(0,1,11))
    ax.clabel(c, inline=True, fontsize=8)
    
    return

def plotMCProbablisticEngagementZone(agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax):
    ax.set_aspect('equal')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    [X, Y] = np.meshgrid(x, y)

    
    malhalanobisDistance = np.zeros(X.shape)
    engagementZonePlot = np.zeros(X.shape)

    numMcTrials = 1000

    pursuerPositionSamples = np.random.multivariate_normal(pursuerPosition.squeeze(), pursuerPositionCov, numMcTrials)

    for i in range(X.shape[0]):
        print(i)
        for j in range(X.shape[1]):
            # engagementZonePlot[i, j] = probabalisticEngagementZone(np.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
            engagementZonePlot[i, j] = monte_carlo_probalistic_engagment_zone(np.array([[X[i,j]],[Y[i,j]]]), agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, numMcTrials, pursuerPositionSamples)
    # c = ax.pcolormesh(X, Y, engagementZonePlot)
    # c = plt.Circle(pursuerPosition, pursuerRange+pursuerCaptureRange, fill=False)
    # ax.add_artist(c)
    c = ax.contour(X, Y, engagementZonePlot, levels=np.linspace(0,1,11))
    ax.clabel(c, inline=True, fontsize=8)
    
    return c

def monte_carlo_probalistic_engagment_zone(agentPosition, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, numMonteCarloTrials, pursurPositionSamples=None):
    if pursurPositionSamples is None:
        #randomly sample from the pursuer position distribution
        pursurPositionSamples = np.random.multivariate_normal(pursuerPosition.squeeze(), pursuerPositionCov, numMonteCarloTrials)
    #randomly sample from the pursuer position distribution
    
    numInEngagementZone = 0
    for pursuerPositionSample in pursurPositionSamples:
        pursuerPositionSample = pursuerPositionSample.reshape(-1,1)
        if inEngagementZone(agentPosition, agentHeading, pursuerPositionSample, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed) < 0:
            numInEngagementZone += 1
    
    return numInEngagementZone/numMonteCarloTrials
    
    
    


# probabalisticEngagementZone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, np.array([[0.1, 0], [0, 0.1]]), pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
# inEngagementZone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)



pursuerPositionCov = np.array([[0.5, 0.0], [0.0, 0.1]])


# mcpez = monte_carlo_probalistic_engagment_zone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, 100)
# print(mcpez)

fig, ax = plt.subplots()
plotMCProbablisticEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, ax)
plotEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax)   
plotMalhalanobisDistance(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, pursuerPositionCov, ax)


fig1, ax1 = plt.subplots()
plotProbablisticEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax1)
plotEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax1)   
plotMalhalanobisDistance(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, pursuerPositionCov, ax1)
plt.show()

