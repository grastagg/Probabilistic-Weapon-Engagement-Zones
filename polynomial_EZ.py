import numpy as np
from scipy.special import hermitenorm
from itertools import combinations_with_replacement, permutations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import matplotlib.pyplot as plt
from functools import partial


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
in_dubins_engagement_zone_ev = jax.jit(
    jax.vmap(
        dubinsEZ.in_dubins_engagement_zone_single,
        in_axes=(
            None,  # pursuerPosition
            None,  # pursuerHeading
            None,  # minimumTurnRadius
            None,  # catureRadius
            None,  # pursuerRange
            None,  # pursuerSpeed
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


def hermite_prob(n: int, x: jnp.ndarray) -> jnp.ndarray:
    """
    Probabilists' Hermite polynomial He_n(x) via recurrence:
      He_0(x) = 1
      He_1(x) = x
      He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)
    """

    def body(carry, i):
        H_nm2, H_nm1 = carry
        H_n = x * H_nm1 - (i - 1) * H_nm2
        return (H_nm1, H_n), H_n

    H0 = jnp.ones_like(x)
    if n == 0:
        return H0
    H1 = x
    if n == 1:
        return H1

    (_, Hn), _ = lax.scan(body, (H0, H1), jnp.arange(2, n + 1))
    return Hn


def make_hermite_evaluator(indices: jnp.ndarray, coeffs_np: jnp.ndarray):
    """
    Create a JAX-jitted surrogate evaluator for multivariate Hermite chaos.

    Args:
      indices_np: np.ndarray of shape (n_terms, d) -- integer multi-indices
      coeffs_np:  np.ndarray of shape (n_terms,)    -- Hermite coefficients

    Returns:
      A function eval_H(X) that maps X of shape (m, d) -> (m,) surrogate values.
    """
    # Convert to Python tuple of tuples of ints (static for JIT)
    # indices_list = tuple(map(tuple, indices_np.tolist()))
    # Static JAX array of coefficients
    coeffs_jax = jnp.array(coeffs_np)

    @jit
    def eval_H(X: jnp.ndarray) -> jnp.ndarray:
        # X: (m, d)
        def eval_point(theta):
            acc = 0.0
            for alpha, c in zip(indices, coeffs_jax):
                term = c
                for k, ak in enumerate(alpha):
                    # ak is a Python int here
                    term = term * hermite_prob(ak, theta[k])
                acc = acc + term
            return acc

        # vectorize over m samples
        return vmap(eval_point)(X)

    return eval_H


def evaluate_hermite_grid(
    pursuer_params: jnp.ndarray,  # shape (P, d1)
    evader_params: jnp.ndarray,  # shape (E, d2)
    indices_np: jnp.ndarray,  # shape (n_terms, d1+d2)
    coeffs_np: jnp.ndarray,  # shape (n_terms,)
) -> jnp.ndarray:
    """
    Evaluate the surrogate for all (pursuer, evader) pairs via nested vmap.

    Returns:
      Z_matrix of shape (P, E).
    """
    # Build the base surrogate evaluator: input X of shape (m, d1+d2) -> (m,)
    eval_H = make_hermite_evaluator(indices_np, coeffs_np)

    @jit
    def eval_for_p(p: jnp.ndarray) -> jnp.ndarray:
        # p: (d1,)
        def eval_e(e: jnp.ndarray):
            # build single row and evaluate
            X = jnp.concatenate([p, e])[None, :]
            return eval_H(X)[0]

        return vmap(eval_e)(evader_params)  # -> (E,)

    @jit
    def eval_all(Ps: jnp.ndarray) -> jnp.ndarray:
        # Ps: (P, d1) -> (P, E)
        return vmap(eval_for_p)(Ps)

    return eval_all(pursuer_params)  # shape (P, E)


def fit_monomial_surrogate(X: np.ndarray, y: np.ndarray, degree: int):
    """
    Fit a global monomial surrogate z_poly(X) to data (X, y) using sklearn.

    Args:
        X: numpy.ndarray of shape (n_samples, d) — input samples.
        y: numpy.ndarray of shape (n_samples,)   — target z(theta) values.
        degree: int — maximum total degree of monomials.

    Returns:
        coeffs: numpy.ndarray of shape (n_terms,) — fitted coefficients.
        powers: numpy.ndarray of shape (n_terms, d) — each row gives exponents for one term.
    """
    # 1) Build monomial features up to total degree
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(X)  # shape (n_samples, n_terms)

    # 2) Fit linear model (no intercept, bias in features)
    model = LinearRegression(fit_intercept=True)
    model.fit(Phi, y)

    # 3) Extract coefficients and exponent patterns
    coeffs = model.coef_  # shape (n_terms,)
    powers = poly.powers_  # shape (n_terms, d)

    return coeffs, powers


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


def make_monomial_evaluator(powers_np: jnp.ndarray, coeffs_np: jnp.ndarray):
    """
    Build a JAX-jitted evaluator for a monomial surrogate.

    Args:
      powers_np: array of shape (n_terms, d) of integer exponents
      coeffs_np: array of shape (n_terms,) of float coefficients

    Returns:
      eval_mono: function mapping X (m, d) -> (m,) surrogate values
    """
    powers = jnp.array(powers_np)  # (n_terms, d)
    coeffs = jnp.array(coeffs_np)  # (n_terms,)

    @jit
    def eval_mono(X: jnp.ndarray) -> jnp.ndarray:
        # X: (m, d)
        # Compute X^(powers) -> shape (m, n_terms, d)
        X_exp = X[:, None, :] ** powers[None, :, :]
        # Multiply across d to get each monomial term: (m, n_terms)
        monom = jnp.prod(X_exp, axis=-1)
        # Sum weighted terms: (m,)
        return monom @ coeffs

    return eval_mono


def evaluate_monomial_grid(
    pursuer_params: jnp.ndarray,  # shape (P, d1)
    evader_params: jnp.ndarray,  # shape (E, d2)
    powers_np: jnp.ndarray,  # shape (n_terms, d1+d2)
    coeffs_np: jnp.ndarray,  # shape (n_terms,)
) -> jnp.ndarray:
    """
    Evaluate the monomial surrogate for all combinations of pursuer and evader.

    Returns:
      Z_matrix of shape (P, E).
    """
    # Build the core surrogate evaluator
    eval_mono = make_monomial_evaluator(powers_np, coeffs_np)

    @jit
    def eval_for_p(p: jnp.ndarray) -> jnp.ndarray:
        # p: (d1,)
        def eval_e(e: jnp.ndarray):
            # Concatenate pursuer and evader parameters
            x = jnp.concatenate([p, e])[None, :]  # shape (1, d1+d2)
            # Evaluate surrogate on single point
            return eval_mono(x)[0]  # scalar

        # Vectorize over all evaders: returns (E,)
        return vmap(eval_e)(evader_params)

    @jit
    def eval_all(Ps: jnp.ndarray) -> jnp.ndarray:
        # Ps: (P, d1) -> returns (P, E)
        return vmap(eval_for_p)(Ps)

    # Compute the full grid
    return eval_all(pursuer_params)  # shape (P, E)


def create_monomial_surragate(
    pursuerParams, pursuerParamsCov, evaderBounds, evaderSpeed, numSamples, degree
):
    X, y = create_input_output_data(
        numSamples, pursuerParams, pursuerParamsCov, evaderBounds, evaderSpeed
    )
    coeffs, powers = fit_monomial_surrogate(X, y, degree)
    return coeffs, powers


def plot_surruagate(
    pursuerParams, pursuerParamsCov, evaderBounds, evaderHeading, evaderSpeed
):
    coeffs, powers = create_monomial_surragate(
        pursuerParams, pursuerParamsCov, evaderBounds, evaderSpeed, 2000, 11
    )

    rangeX = 1.5
    numPoints = 100
    x = jnp.linspace(-rangeX, rangeX, numPoints)
    y = jnp.linspace(-rangeX, rangeX, numPoints)
    [X, Y] = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    evaderHeadings = np.ones_like(X) * evaderHeading
    evaderPositions = np.column_stack((X, Y))

    evaderParams = np.hstack([evaderPositions, evaderHeadings[:, None]])
    Z = evaluate_monomial_grid(np.array([pursuerParams]), evaderParams, powers, coeffs)
    Z = Z.reshape(numPoints, numPoints)
    fig, ax = plt.subplots()
    X = X.reshape(numPoints, numPoints)
    Y = Y.reshape(numPoints, numPoints)
    c = ax.pcolormesh(
        X,
        Y,
        Z,
    )
    colors = ["red"]
    ax.contour(X, Y, Z, levels=[0], colors=colors, zorder=10000)
    cbar = plt.colorbar(c)
    pursuerPosition = pursuerParams[:2]
    pursuerHeading = pursuerParams[2]
    minimumTurnRadius = pursuerParams[3]
    captureRadius = 0.0
    pursuerRange = pursuerParams[4]
    pursuerSpeed = pursuerParams[5]
    evaderPosition = evaderPositions
    evaderHeading = evaderHeadings
    y = in_dubins_engagement_zone_ev(
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
    print(evaderHeadings)
    fig2, ax2 = plt.subplots()
    c = ax2.pcolormesh(X, Y, y.reshape(numPoints, numPoints))
    dubinsEZ.plot_dubins_EZ(
        pursuerPosition,
        pursuerHeading,
        pursuerSpeed,
        minimumTurnRadius,
        captureRadius,
        pursuerRange,
        evaderHeading[0],
        evaderSpeed,
        ax2,
    )
    cbar = plt.colorbar(c)
    plt.show()


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
            np.array([minimumTurnRadius]),  # (1,)
            np.array([pursuerRange]),  # (1,)
            np.array([pursuerSpeed]),  # (1,)
        ]
    )
    full_cov = stacked_cov(
        pursuerPositionCov,
        pursuerHeadingVar,
        minimumTurnRadiusVar,
        pursuerRangeVar,
        pursuerSpeedVar,
    )
    evaderBounds = np.array([-3, 3, -3, 3, -np.pi, np.pi])
    plot_surruagate(pursuerParams, full_cov, evaderBounds, evaderHeading, evaderSpeed)


if __name__ == "__main__":
    main()
