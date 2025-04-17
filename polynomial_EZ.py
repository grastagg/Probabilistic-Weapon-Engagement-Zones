import numpy as np
from scipy.special import hermitenorm
from itertools import combinations_with_replacement, permutations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax


import dubinsEZ

in_dubins_engagement_zone = jax.jit(
    jax.vmap(
        dubinsEZ.in_dubins_engagement_zone_single,
        in_axes=(
            0,  # pursuerPosition
            0,  # pursuerHeading
            0,  # minimumTurnRadius
            None,  # catureRadius
            0,  # pursuerRange
            0,  # pursuerSpeed
            0,  # evaderPosition
            0,  # evaderHeading
            None,  # evaderSpeed
        ),  # Vectorizing over evaderPosition & evaderHeading
    )
)


def stacked_cov(
    pursuerPositionCov,
    pursuerHeadingVar,
    pursuerSpeedVar,
    minimumTurnRadiusVar,
    pursuerRangeVar,
):
    heading_block = np.array([[pursuerHeadingVar]])
    speed_block = np.array([[pursuerSpeedVar]])
    radius_block = np.array([[minimumTurnRadiusVar]])
    range_block = np.array([[pursuerRangeVar]])

    # Assemble full covariance using np.block (block matrix layout)
    full_cov = np.block(
        [
            [
                pursuerPositionCov,
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                np.zeros((2, 1)),
            ],
            [
                np.zeros((1, 2)),
                heading_block,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
            ],
            [
                np.zeros((1, 2)),
                np.zeros((1, 1)),
                speed_block,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
            ],
            [
                np.zeros((1, 2)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                radius_block,
                np.zeros((1, 1)),
            ],
            [
                np.zeros((1, 2)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                range_block,
            ],
        ]
    )
    return full_cov


def generate_multi_indices(d, p):
    """
    Generate all multi-indices alpha in N^d with |alpha| <= p,
    sorted in graded lex order.
    """
    indices = [(0,) * d]
    for total in range(1, p + 1):
        for c in combinations_with_replacement(range(d), total):
            # count occurrences
            alpha = [0] * d
            for idx in c:
                alpha[idx] += 1
            indices.append(tuple(alpha))
    return indices


def build_hermite_design_matrix(X, indices):
    """
    Build the design matrix Phi for multivariate probabilists' Hermite polynomials.

    X: (n_samples, d)
    indices: list of tuples of length d (multi-indices)

    Returns: Phi of shape (n_samples, n_terms)
    """
    X = np.asarray(X)
    n, d = X.shape
    m = len(indices)
    Phi = np.empty((n, m), dtype=float)
    for j, alpha in enumerate(indices):
        # compute product of univariate Hermite polynomials He_{alpha[k]}(X[:,k])
        term = np.ones(n, dtype=float)
        for k, ak in enumerate(alpha):
            if ak > 0:
                # hermitenorm gives probabilists' Hermite H_n
                term *= hermitenorm(ak)(X[:, k])
        Phi[:, j] = term
    return Phi


def fit_hermite_surrogate(X, y, degree):
    """
    Fit a global multivariate Hermite (polynomial chaos) surrogate by least squares.

    X: (n_samples, d)
    y: (n_samples,)
    degree: max total degree

    Returns:
      indices: list of multi-indices
      coeffs: array of shape (n_terms,)
    """
    d = X.shape[1]
    indices = generate_multi_indices(d, degree)
    Phi = build_hermite_design_matrix(X, indices)
    # Solve least-squares: Phi @ coeffs = y
    coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return indices, coeffs


def eval_hermite_surrogate(X, indices, coeffs):
    """
    Evaluate the fitted Hermite surrogate at new points X.
    """
    Phi = build_hermite_design_matrix(X, indices)
    return Phi.dot(coeffs)


# Example usage:
# X = np.random.randn(500, 5)        # 500 training points in 5 dims
# y = compute_z_soft(X)              # your smoothed z(theta)
# inds, cfs = fit_hermite_surrogate(X, y, degree=3)
# X_new = np.random.randn(10, 5)
# y_new = eval_hermite_surrogate(X_new, inds, cfs)


@jit
def hermite_prob(n: int, x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute probabilists' Hermite polynomial He_n(x) via recurrence:
      He_0(x) = 1
      He_1(x) = x
      He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)
    """

    def body(carry, i):
        H_nm2, H_nm1 = carry
        H_n = x * H_nm1 - (i - 1) * H_nm2
        return (H_nm1, H_n), H_n

    # handle n=0 and n=1 as special cases
    H0 = jnp.ones_like(x)
    if n == 0:
        return H0
    H1 = x
    if n == 1:
        return H1

    # use lax.scan to iterate from 2..n
    (_, Hn), _ = lax.scan(body, (H0, H1), jnp.arange(2, n + 1))
    return Hn


@jax.jit
def eval_hermite_surrogate_jax(
    X: jnp.ndarray, indices: jnp.ndarray, coeffs: jnp.ndarray
) -> jnp.ndarray:
    """
    Evaluate multivariate Hermite surrogate in JAX.

    Args:
      X: array of shape (m, d) – new input points
      indices: array of shape (n_terms, d) – multi-indices
      coeffs: array of shape (n_terms,) – Hermite coefficients

    Returns:
      y: array of shape (m,) – surrogate values
    """

    def eval_point(theta):
        # theta: (d,)
        def eval_term(alpha, c):
            # alpha: (d,), c: scalar coefficient
            # compute product of univariate Hermite polynomials
            H_vals = [
                hermite_prob(int(alpha[k]), theta[k]) for k in range(alpha.shape[0])
            ]
            return c * jnp.prod(jnp.stack(H_vals))

        # vectorize over terms
        terms = vmap(eval_term, in_axes=(0, 0))(indices, coeffs)  # (n_terms,)
        return jnp.sum(terms)

    # vectorize over sample points
    return vmap(eval_point, in_axes=(0,))(X)  # (m,)


def create_input_output_data(
    n_samples, pursuerParams, pursuerParamsCov, evaderBounds, evaderSpeed
):
    # draw n_samples samples from the multivariate normal distribution
    pursuerParams_samples = np.random.multivariate_normal(
        pursuerParams, pursuerParamsCov, n_samples
    )
    evaderX = np.random.uniform(evaderBounds[0], evaderBounds[1], n_samples)
    evaderY = np.random.uniform(evaderBounds[2], evaderBounds[3], n_samples)
    evaderHeading = np.random.uniform(evaderBounds[4], evaderBounds[5], n_samples)
    # stack the samples into a single array
    X = np.column_stack((pursuerParams_samples, evaderX, evaderY, evaderHeading))
    pursuerPosition = pursuerParams_samples[:, :2]
    pursuerHeading = pursuerParams_samples[:, 2]
    minimumTurnRadius = pursuerParams_samples[:, 3]
    captureRadius = 0.0
    pursuerRange = pursuerParams_samples[:, 4]
    pursuerSpeed = pursuerParams_samples[:, 5]
    evaderPosition = np.column_stack((evaderX, evaderY))
    evaderHeading = evaderHeading
    y = in_dubins_engagement_zone(
        pursuerPosition,
        pursuerHeading,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        pursuerSpeed,
        evaderPosition,
        evaderHeading,
        evaderSpeed,
    )

    return X, y


def create_hermite_surragate(
    pursuerParams, pursuerParamsCov, evaderBounds, evaderSpeed, numSamples, degree
):
    X, y = create_input_output_data(
        numSamples, pursuerParams, pursuerParamsCov, evaderBounds, evaderSpeed
    )
    indicies, coeffs = fit_hermite_surrogate(X, y, degree)
    return indicies, coeffs


def main():
    pursuerPosition = np.array([0.0, 0.0])
    pursuerPositionCov = np.array([[0.025, -0.04], [-0.04, 0.1]])
    # pursuerPositionCov = np.array([[0.000000000001, 0.0], [0.0, 0.00000000001]])

    pursuerHeading = (0.0 / 4.0) * np.pi
    pursuerHeadingVar = 0.5

    pursuerSpeed = 2.0
    pursuerSpeedVar = 0.3

    pursuerRange = 1.0
    pursuerRangeVar = 0.1
    pursuerRangeVar = 0.0

    minimumTurnRadius = 0.2
    minimumTurnRadiusVar = 0.005
    # minimumTurnRadiusVar = 0.00000000001

    captureRadius = 0.0

    evaderHeading = np.array([(5.0 / 20.0) * np.pi])
    # evaderHeading = np.array((0.0 / 20.0) * np.pi)

    evaderSpeed = 0.5
    evaderPosition = np.array([[-0.25, 0.35]])

    numberOfSamples = 1000
    pursuerParams = np.concatenate(
        [
            pursuerPosition,  # (2,)
            np.array([pursuerHeading]),  # (1,)
            np.array([pursuerSpeed]),  # (1,)
            np.array([minimumTurnRadius]),  # (1,)
            np.array([pursuerRange]),  # (1,)
        ]
    )
    full_cov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        pursuerSpeedVar,
        minimumTurnRadiusVar,
        pursuerRangeVar,
    )
    evaderBounds = np.array([-3, 3, -3, 3, -np.pi, np.pi])


if __name__ == "__main__":
    main()
