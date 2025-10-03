import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

from .utils import multilevel

def _spsolve(main_diag, upper_diag, lower_diag, b):
    N = main_diag.shape[-1]
    assert main_diag.shape==(N,) and upper_diag.shape==(N-1,) and lower_diag.shape==(N-1,) and b.shape==(N,)
    # Sparse matrix
    A = diags(
        [main_diag, upper_diag, lower_diag],
        [0, 1, -1],
        shape=[N,N],
        format="csc",
    )

    # Solve the system
    u = spsolve(A, b)

    return u

vec_spsolve = np.vectorize(_spsolve,signature="(n),(m),(m),(n)->(n)")

# Elliptic PDE example
def solve_elliptic_pde(level=5, coeffs=None):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> coeffs = rng.uniform(low=0,high=1,size=(2,8))
    >>> sol,x,a,u = solve_elliptic_pde(level=1,coeffs=coeffs)
    >>> sol
    array([0.0874314 , 0.09523068])
    >>> x 
    array([0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875])
    >>> a
    array([[1.66918312, 3.25086801, 1.95703413, 1.38673674, 0.41874112,
            1.34205836, 0.75966799],
           [0.8576812 , 1.71783066, 2.68539861, 2.38600689, 2.18915356,
            1.77799639, 1.25820725]])
    >>> u
    array([[0.04033278, 0.06134792, 0.07520102, 0.0874314 , 0.0927738 ,
            0.08050413, 0.05535604],
           [0.05850198, 0.08533241, 0.09392885, 0.09523068, 0.08984335,
            0.07575316, 0.04705024]])

    >>> for i in range(coeffs.shape[0]):
    ...     sol_i,x_i,a_i,u_i = solve_elliptic_pde(level=1,coeffs=coeffs[i])
    ...     assert (sol_i==sol[i]).all()
    ...     assert (x_i==x).all()
    ...     assert (a_i==a[i]).all()
    ...     assert (u_i==u[i]).all()

    Solve the 1D elliptic PDE with a spatially varying diffusion coefficient.

    Parameters:
    - level (int): Controls the mesh size, h = 2^(-level - 1).
    - coeffs (np.array): Coefficients for the sine expansion of the diffusion coefficient.

    Returns:
    - u_0_5 (float): The numerical solution at x = 0.5.
    """
    # Define grid
    N = 2 ** (level + 2)  # Number of intervals
    h = 1.0 / N  # Mesh spacing
    x = np.linspace(h, 1 - h, N - 1)  # Interior points

    # Compute diffusion coefficient a(x)
    coeffs = np.random.rand(8) if coeffs is None else coeffs
    assert isinstance(coeffs,np.ndarray)
    coeffs = norm.ppf(coeffs)  # Transform from uniform to iid Gaussian

    batch_shape = list(coeffs.shape)[:-1]

    k = np.arange(1,coeffs.shape[-1]+1)
    a_x = np.exp((coeffs[...,None] / k[:,None]  *np.sin(np.pi * k[:,None] * x)).sum(-2))

    # Compute a at half-grid points (needed for flux terms)
    a_half = np.zeros(batch_shape+[N])
    a_half[...,1:-1] = (a_x[...,:-1] + a_x[...,1:]) / 2  # Midpoint values for flux approximation
    a_half[...,0] = a_x[...,0]  # At the first midpoint
    a_half[...,-1] = a_x[...,-1]  # At the last midpoint

    # Construct the finite difference matrix
    lower_diag = -a_half[...,1:-1] / h**2
    upper_diag = -a_half[...,1:-1] / h**2
    main_diag = (a_half[...,:-1] + a_half[...,1:]) / h**2

    # Right-hand side (forcing term)
    b = np.ones([N-1])  # Constant source term (1)

    u = vec_spsolve(main_diag, upper_diag, lower_diag, b)
    # Find index closest to x = 0.5
    idx = np.argmin(np.abs(x - 0.5))
    return u[...,idx], x, a_x, u

# Define wrapper function
@multilevel
def elliptic(level, sample):
    """
    Wrapper that just returns the quantity of interest.
    
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> qmean = 0
    >>> for l in range(4):
    ...     x = rng.uniform(size=(2**12,8))
    ...     y = elliptic.ml(level=l,sample=x)
    ...     ymean_l = y.mean()
    ...     ystd_l = y.std(ddof=1)
    ...     qmean += ymean_l
    ...     print("Qmean[l] = %-10.3f Ymean[l] = %-15.3e Ystd[l] = %.3e"%(qmean,ymean_l,ystd_l))
    Qmean[l] = 0.156      Ymean[l] = 1.557e-01       Ystd[l] = 1.400e-01
    Qmean[l] = 0.144      Ymean[l] = -1.182e-02      Ystd[l] = 6.286e-02
    Qmean[l] = 0.146      Ymean[l] = 2.504e-03       Ystd[l] = 1.076e-02
    Qmean[l] = 0.148      Ymean[l] = 1.534e-03       Ystd[l] = 3.531e-03
    """
    return solve_elliptic_pde(level, sample)[0]
