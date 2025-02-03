import jax.numpy as jnp
from jax import jit, lax


@jit
def circle_segment_intersection_jax(circle_center, radius, A, B):
    """
    Computes the single intersection point of a line segment AB with a circle using JAX.

    Args:
        circle_center (tuple): (h, k) center of the circle.
        radius (float): Radius of the circle.
        A (tuple): (x1, y1) first point of the segment.
        B (tuple): (x2, y2) second point of the segment.

    Returns:
        jnp.ndarray: Intersection point (x, y) or empty array if no intersection.
    """
    h, k = jnp.array(circle_center, dtype=jnp.float32)
    r = radius
    A = jnp.array(A, dtype=jnp.float32)
    B = jnp.array(B, dtype=jnp.float32)

    # Direction vector of the line segment
    d = B - A

    # Quadratic coefficients
    A_coeff = jnp.dot(d, d)
    B_coeff = 2 * jnp.dot(d, A - jnp.array([h, k]))
    C_coeff = jnp.dot(A - jnp.array([h, k]), A - jnp.array([h, k])) - r**2

    # Compute discriminant
    discriminant = B_coeff**2 - 4 * A_coeff * C_coeff

    def compute_intersection():
        sqrt_disc = jnp.sqrt(discriminant)
        t = (-B_coeff + sqrt_disc) / (
            2 * A_coeff
        )  # Only considering one root (single intersection)

        # Check if t is within the segment range [0, 1]
        valid_t = jnp.logical_and(0 <= t, t <= 1)

        # Compute the intersection point
        intersection_point = A + t * d

        # Ensure the shape is consistent for both branches
        intersection_point = jnp.expand_dims(intersection_point, axis=0)  # (1, 2)
        empty_point = jnp.empty((0, 2))  # (0, 2) for no intersection

        # Use jax.lax.select for conditional result based on valid_t
        return lax.select(valid_t, intersection_point, empty_point)

    # Use JAX conditional execution: return the intersection if discriminant >= 0
    return lax.cond(discriminant >= 0, compute_intersection, lambda: jnp.empty((0, 2)))


# Example usage
circle_center = (0, 0)
radius = 5
A = (-6, 0)
B = (6, 0)

intersection = circle_segment_intersection_jax(circle_center, radius, A, B)
print("Intersection point within segment:\n", intersection)
