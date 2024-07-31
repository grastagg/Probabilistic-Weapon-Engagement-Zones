import numpy as np
import matplotlib.pyplot as plt
# import jax.numpy as jnp
# from jax import jacfwd, value_and_grad
from scipy.stats import norm
import time
# from jax import jit

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

#@jit
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
    return distance - rho

# @jit
def get_grad_and_value(agentPosition, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    fun = value_and_grad(lambda x: inEngagementZoneJax(agentPosition, agentHeading, x, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed))
    return fun
    

    
def probabalisticEngagementZoneTemp(agentPosition, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    # start = time.time()
    mean = inEngagementZoneJax(agentPosition, agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed).squeeze()
    # print("mean time: ", time.time() - start)
    # start = time.time()
    jac = jacfwd(lambda x: inEngagementZoneJax(agentPosition, agentHeading, x, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed))(pursuerPosition).squeeze()
    jac = jacfwd(lambda x: inEngagementZoneJax(agentPosition, agentHeading, x, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed))(pursuerPosition).squeeze()
    # print("jac time: ", time.time() - start)
    # start = time.time()
    cov = jac@pursuerPositionCov@jac.T
    # print("cov time: ", time.time() - start)

    #'Cumulative distribution function for the standard normal distribution'
    # print("temp")
    # print("mean: ", mean)
    # print("cov: ", cov)

    # start = time.time()
    diffDistribution = norm(mean, np.sqrt(cov))
    # print("dist time: ", time.time() - start)
    return diffDistribution.cdf(0)

def plotProbablisticEngagementZone(agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax):
    ax.set_aspect('equal')
    ax.set_title("Linearized Probabalistic Engagement Zone")
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    [X, Y] = np.meshgrid(x, y)

    
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
    ax.set_title("Monte Carlo Probabalistic Engagement Zone")
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


if __name__ == "__main__":
    pursuerRange = .8
    pursuerCaptureRange = 0.1
    pursuerSpeed = 1
    agentSpeed = .9

    pursuerPositionCov = np.array([[0.1, 0.0], [0.0, 0.1]])
    # pursuerInitialPosition = jnp.array([-.5, .5])
    pursuerInitialPosition = np.array([[0.0], [0.0]])
    agentInitialPosition = np.array([[0.0], [2.0]])
    # agentInitialHeading = 9*np.pi/11
    agentInitialHeading = 0

    

    # mcpez = monte_carlo_probalistic_engagment_zone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, 100)
    # print(mcpez)

    # fig, ax = plt.subplots()
    # plotMCProbablisticEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed, ax)
    # plotEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax)   
    # plotMalhalanobisDistance(pursuerInitialPosition, pursuerPositionCov, ax)


    fig1, ax1 = plt.subplots()
    # plotProbablisticEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax1)
    # plotEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed,ax1)   
    ax1.set_aspect('equal')
    plotMalhalanobisDistance(pursuerInitialPosition, pursuerPositionCov, ax1)
    plt.show()

