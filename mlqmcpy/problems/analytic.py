import numpy as np

from .utils import multilevel


@multilevel
def analytic(level, samples=None):
    """
    Analytic test function

        Q_\\ell = sin(X) + 0.5^\\ell * sin(Y),

    where `samples` is an array-like of shape (n_samples, 2) with
      - samples[:, 0] is interpreted as X ~ Uniform(0, 1)
      - samples[:, 1] is interpreted as Y ~ Uniform(0, 1)

    For this example, statistics are know analytically as

        E[Q_\\ell] = (1 - cos(1)) * (1 + 0.5^\\ell)

        V[Q_\\ell] = [1/2 - sin(2)/4 - (1 - cos(1))^2] * (1 + 0.5^(2 * \\ell))

    The multilevel difference is

        Y_\\ell = Q_\\ell - Q_{\\ell - 1} = -0.5^\\ell * sin(Y)

    with statistics

        E[Y_\\ell] = -0.5^\\ell * (1 - cos(1))

        V[Y_\\ell] = 0.5^(2 * \\ell) [1/2 - sin(2)/4 - (1 - cos(1))^2]

    Returns:
      Quantity of interest Q_\\ell.
    """

    samples = np.atleast_2d(samples)
    return np.sin(samples[:, 0]) + 0.5**level * np.sin(samples[:, 1])


# Define analytic solutions
analytic.exact = lambda: None

analytic.exact.Q = lambda: None
analytic.exact.Q.mean = lambda level: (1 - np.cos(1)) * (1 + 0.5**level)
analytic.exact.Q.variance = lambda level: (
    0.5 - np.sin(2) / 4 - (1 - np.cos(1)) ** 2
) * (1 + 0.5 ** (2 * level))

analytic.exact.Y = lambda: None
analytic.exact.Y.mean = lambda level: -(0.5**level) * (1 - np.cos(1))
analytic.exact.Y.variance = lambda level: 0.5 ** (2 * level) * (
    0.5 - np.sin(2) / 4 - (1 - np.cos(1)) ** 2
)
