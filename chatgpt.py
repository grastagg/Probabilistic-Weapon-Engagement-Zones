import numpy as np
import matplotlib.pyplot as plt


def minkowski_sum(A, B):
    """
    Computes the Minkowski sum of two sets A and B.

    Parameters:
        A (ndarray): An (m, n) array representing m points in n-dimensional space.
        B (ndarray): A (p, n) array representing p points in n-dimensional space.

    Returns:
        ndarray: The Minkowski sum of A and B.
    """
    A = A[:, np.newaxis, :]  # Reshape A to (m,1,n)
    B = B[np.newaxis, :, :]  # Reshape B to (1,p,n)
    return (A + B).reshape(-1, A.shape[2])  # Compute sum and reshape to (m*p, n)


# Example: Minkowski Sum of Two 2D Shapes
square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])  # A square
diamond = np.array([[0, 2], [2, 0], [0, -2], [-2, 0]])  # A diamond (rotated square)

result = minkowski_sum(square, diamond)

# Plot the original shapes and their Minkowski sum
plt.figure(figsize=(6, 6))
plt.scatter(square[:, 0], square[:, 1], color="blue", label="Square")
plt.scatter(diamond[:, 0], diamond[:, 1], color="red", label="Diamond")
plt.scatter(result[:, 0], result[:, 1], color="green", alpha=0.5, label="Minkowski Sum")
plt.legend()
plt.grid(True)
plt.show()

