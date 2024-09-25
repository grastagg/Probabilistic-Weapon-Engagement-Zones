import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plotMahalanobisDistance(pursuerPosition, pursuerPositionCov, ax):
    # Define the grid
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Stack X, Y coordinates into a single array
    points = np.stack([X.ravel(), Y.ravel()]).T
    
    # Compute the inverse of the covariance matrix
    inv_cov = np.linalg.inv(pursuerPositionCov)
    
    # Compute Mahalanobis distance for each point (vectorized)
    delta = points - pursuerPosition.T
    malhalanobisDistance = np.sqrt(np.einsum('ij,jk,ik->i', delta, inv_cov, delta))
    malhalanobisDistance = malhalanobisDistance.reshape(X.shape)
    
    # Specify darker shades of red
    colors = ['#CC0000', '#FF6666','#FFCCCC' ] 
    cmap = ListedColormap(colors)
    
    # Plot filled contours for Mahalanobis distance with darker red shades
    c = ax.contourf(X, Y, malhalanobisDistance, levels=[0, 1, 2, 3], colors=colors, alpha=0.75)
    
    # Mark the pursuer position with a dark red dot
    ax.scatter(pursuerPosition[0], pursuerPosition[1], color='darkred', label='Pursuer Position', s=100)
    
    # Add a color bar and increase font size
    cbar = plt.colorbar(c, ax=ax, ticks=[0, 1, 2, 3])
    cbar.set_label("Mahalanobis Distance", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Set larger fonts for axis labels and legend
    ax.set_xlabel('X-axis', fontsize=14)
    ax.set_ylabel('Y-axis', fontsize=14)
    ax.legend(fontsize=12)

    # Increase the size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)

# Example of usage
fig, ax = plt.subplots(figsize=(8, 8))
pursuerPosition = np.array([[0], [0]])  # Centered at origin
pursuerPositionCov = np.array([[1, 0.5], [0.5, 1]])  # Covariance matrix

plotMahalanobisDistance(pursuerPosition, pursuerPositionCov, ax)
plt.show()
