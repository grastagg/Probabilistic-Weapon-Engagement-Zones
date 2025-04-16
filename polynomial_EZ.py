import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def fit_polynomial_surrogate(X, y, degree):
    """
    Fit a global polynomial surrogate z_poly(theta) to data (X, y).

    Args:
        X: numpy.ndarray of shape (n_samples, d) — input parameter samples.
        y: numpy.ndarray of shape (n_samples,)   — output z(theta) values.
        degree: int — total degree of the polynomial surrogate.

    Returns:
        model: trained LinearRegression model (no intercept, bias is in features).
        poly:  fitted PolynomialFeatures transformer, so you can transform new θ.
    """
    # 1) Build polynomial features up to total degree
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(X)  # shape (n_samples, n_terms)

    # 2) Fit linear model without additional intercept
    model = LinearRegression(fit_intercept=False)
    model.fit(Phi, y)

    return model, poly


def create_input_output_data(n_samples, pursuerParams, pursuerParamsCov, evaderBounds):
    # draw n_samples samples from the multivariate normal distribution
    pursuerParams_samples = np.random.multivariate_normal(
        pursuerParams, pursuerParamsCov, n_samples
    )
    evaderX = np.random.uniform(evaderBounds[0], evaderBounds[1], n_samples)
    evaderY = np.random.uniform(evaderBounds[2], evaderBounds[3], n_samples)
    evaderHeading = np.random.uniform(evaderBounds[4], evaderBounds[5], n_samples)
    # stack the samples into a single array
    X = np.column_stack((pursuerParams_samples, evaderX, evaderY, evaderHeading))

    return X
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

    evaderHeading = jnp.array([(5.0 / 20.0) * np.pi])
    # evaderHeading = jnp.array((0.0 / 20.0) * np.pi)

    evaderSpeed = 0.5
    evaderPosition = np.array([[-0.25, 0.35]])

if __name__ '__main__':
    main()
