"""
Shared numerical model for the transformer heat equation
in Bragone et al. (2022).

Both the FVM and PINN notebooks import the same coefficients
from this file.

Numerical reproduction of the printed scaled Eq. 12:
    1.8*u_t = 0.05*u_xx + 15 + 83*K(t)**2 - (u - Ta(t))

General notation:
    a*u_t = k*u_xx + P0 + mu*K(t)**2 - h*(u - Ta(t))

Boundary conditions:
    u(0, t) = Ta(t)
    u(H, t) = To(t)

PINN residual:
    f = a*u_t - k*u_xx - (P0 + mu*K(t)**2 - h*(u - Ta(t)))

The numerical time coordinate advances by 1 per hourly observation.
These coefficients reproduce Eq. 12 literally. We do not claim that
their scaling is a dimensionally consistent conversion of Table 1's
SI coefficients to hours.

The height coordinate spans 0 to 1, following the paper.
"""
import numpy as np
from scipy.linalg import solve_banded

# --------------------------------------------------------------------------
# Coefficients copied literally from the printed scaled Eq. 12.
# Shared by FVM and PINN for the same numerical formulation.
# Time advances by 1 per hourly observation; physical unit equivalence
# with Table 1 has not been established.
# --------------------------------------------------------------------------

PARAMS = dict(
    a=0.9 * 2.0,   # rho*c_p  (900 kg/m^3 * 2000 J/kgK, scaled)
    k=0.05,        # effective thermal conductivity (50, scaled)
    h=1.0,         # convective heat transfer coefficient (1000, scaled)
    P0=15.0,       # no-load loss (15000 W, scaled)
    mu=83.0,       # rated load loss (83000 W, scaled); P_K = mu*K^2
    H=1.0,         # height [m]
)


def fvm_heat(Ta, To, K, a, k, h, P0, mu, H=1.0, N=50, dt=1.0,
             substeps=1, u0=None):
    """
    Finite volume (cell-centred) + backward Euler solver.

    Ta, To, K : arrays of length n_t, one value per data time point (hourly).
    N         : number of cells; unknowns live at cell centres.
    dt : numerical time spacing between observations. Default 1.0 represents one hourly observation interval.
     
    substeps  : internal time steps per data interval; inputs are linearly
                interpolated in between. Output is still returned at the data
                time points. Increase this to reduce time-discretisation error.
    u0        : initial profile at t = 0 (length N). The paper gives none;
                default = linear between the two boundary values (ASSUMPTION).

    Returns x (cell centres, shape (N,)) and U (shape (n_t, N)).
    """
    Ta, To, K = (np.asarray(v, dtype=float) for v in (Ta, To, K))
    n_t = len(Ta)
    dx = H / N
    x = (np.arange(N) + 0.5) * dx
    c = k / dx**2
    tau = dt / substeps

    U = np.zeros((n_t, N))
    U[0] = u0 if u0 is not None else Ta[0] + (To[0] - Ta[0]) * x / H

    # Constant tridiagonal matrix in banded form
    diag = np.full(N, a / tau + 2 * c + h)
    diag[0] += c        # boundary face is dx/2 from the first/last cell centre
    diag[-1] += c
    ab = np.zeros((3, N))
    ab[0, 1:] = -c
    ab[1, :] = diag
    ab[2, :-1] = -c

    u = U[0].copy()
    for n in range(1, n_t):
        for s in range(1, substeps + 1):
            w = s / substeps                       # interpolation weight
            Ta_s = (1 - w) * Ta[n - 1] + w * Ta[n]
            To_s = (1 - w) * To[n - 1] + w * To[n]
            K_s = (1 - w) * K[n - 1] + w * K[n]
            rhs = a / tau * u + P0 + mu * K_s**2 + h * Ta_s
            rhs[0] += 2 * c * Ta_s
            rhs[-1] += 2 * c * To_s
            u = solve_banded((1, 1), ab, rhs)
        U[n] = u
    return x, U


def steady_state_exact(x, Ta, To, K, k, h, P0, mu, H=1.0, **_):
    """Exact steady solution of k*u'' - h*u + (P0 + mu*K^2 + h*Ta) = 0
    with u(0) = Ta, u(H) = To (constant inputs). Used to verify the FVM."""
    m = np.sqrt(h / k)
    up = (P0 + mu * K**2 + h * Ta) / h          # particular solution
    A = Ta - up
    B = (To - up - A * np.cosh(m * H)) / np.sinh(m * H)
    return up + A * np.cosh(m * x) + B * np.sinh(m * x)