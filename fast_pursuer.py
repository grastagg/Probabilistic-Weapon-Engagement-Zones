import numpy as np
import matplotlib.pyplot as plt



pursuerRange = 1
pursuerCaptureRange = 0.1
pursuerSpeed = 1
agentSpeed = .9


pursuerInitialPosition = np.array([.0, 0])
agentInitialPosition = np.array([0, -1])
agentInitialHeading = np.pi/2

def inEngagementZone(agentPosition,agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    speedRatio = agentSpeed / pursuerSpeed
    distance = np.linalg.norm(agentPosition - pursuerPosition)
    epsilon = agentHeading + np.arctan2(agentPosition[1] - pursuerPosition[1], pursuerPosition[0] - agentPosition[0])
    rho = speedRatio*pursuerRange*(np.cos(epsilon) + np.sqrt(np.cos(epsilon)**2 - 1+(pursuerRange+pursuerCaptureRange)**2/(speedRatio**2*pursuerRange**2)))
    print(epsilon)
    
    # return distance < rho
    return rho

    
    

def plotEngagementZone(agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 100)
    [X, Y] = np.meshgrid(x, y)

    engagementZonePlot = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            engagementZonePlot[i, j] = inEngagementZone(np.array([X[i, j], Y[i, j]]), agentHeading, pursuerPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
    c = plt.Circle(pursuerPosition, pursuerRange+pursuerCaptureRange, fill=False)
    ax.add_artist(c)
    # plt.contour(X, Y, engagementZonePlot, levels=[0, 1], colors=['red'])
    c = plt.pcolormesh(X, Y, engagementZonePlot)
    plt.colorbar(c, ax=ax)
    plt.show()
    
    return

def probabalisticEngagementZone(agentPosition, agentHeading, pursuerPosition,pursuerPositionCov, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed):
    rotationMinusHeading = np.array([[np.cos(agentHeading), -np.sin(agentHeading)], [np.sin(agentHeading), np.cos(agentHeading)]])

    pursuerPositionHat = (pursuerPosition - pursuerPosition)@rotationMinusHeading
    agentPositionHat = (agentPosition - pursuerPosition)@rotationMinusHeading
    pursuerPositionCovHat = rotationMinusHeading@pursuerPositionCov@rotationMinusHeading.T
    epsilon = np.arctan2(agentPositionHat[1] - pursuerPositionHat[1], pursuerPositionHat[0] - agentPositionHat[0])
    print(epsilon)


probabalisticEngagementZone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, np.array([[0.1, 0], [0, 0.1]]), pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)
inEngagementZone(agentInitialPosition, agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)

# plotEngagementZone(agentInitialHeading, pursuerInitialPosition, pursuerRange, pursuerCaptureRange, pursuerSpeed, agentSpeed)   
