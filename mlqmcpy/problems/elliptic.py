import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

from .utils import multilevel


# Elliptic PDE example
def solve_elliptic_pde(level=5, coeffs=None):
    """
    Solve the 1D elliptic PDE with a spatially varying diffusion coefficient.

    Parameters:
    - level (int): Controls the mesh size, h = 2^(-level - 1).
    - coeffs (list or np.array): Coefficients for the sine expansion of the diffusion coefficient.

    Returns:
    - u_0_5 (float): The numerical solution at x = 0.5.
    """
    # Define grid
    N = 2 ** (level + 2)  # Number of intervals
    h = 1.0 / N  # Mesh spacing
    x = np.linspace(h, 1 - h, N - 1)  # Interior points

    # Compute diffusion coefficient a(x)
    coeffs = np.random.rand(8) if coeffs is None else coeffs
    coeffs = norm.ppf(coeffs)  # Transform from uniform to iid Gaussian
    a_x = np.exp(
        sum(c / (k + 1) * np.sin(np.pi * (k + 1) * x) for k, c in enumerate(coeffs))
    )

    # Compute a at half-grid points (needed for flux terms)
    a_half = np.zeros(N)
    a_half[1:-1] = (a_x[:-1] + a_x[1:]) / 2  # Midpoint values for flux approximation
    a_half[0] = a_x[0]  # At the first midpoint
    a_half[-1] = a_x[-1]  # At the last midpoint

    # Construct the finite difference matrix
    lower_diag = -a_half[1:-1] / h**2
    upper_diag = -a_half[1:-1] / h**2
    main_diag = (a_half[:-1] + a_half[1:]) / h**2

    # Sparse matrix
    A = diags(
        [main_diag, upper_diag, lower_diag],
        [0, 1, -1],
        shape=(N - 1, N - 1),
        format="csc",
    )

    # Right-hand side (forcing term)
    b = np.ones(N - 1)  # Constant source term (1)

    # Solve the system
    u = spsolve(A, b)

    # Find index closest to x = 0.5
    idx = np.argmin(np.abs(x - 0.5))
    return u[idx], x, a_x, u


# Define wrapper function
@multilevel
def elliptic(level, sample):
    """Wrapper that just returns the quantity of interest."""
    return solve_elliptic_pde(level, sample)[0]
