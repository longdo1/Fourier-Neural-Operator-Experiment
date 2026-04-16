import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_poisson_fdm(f, x):
    dx = x[1] - x[0]
    n = len(x) - 2

    bands = np.zeros((3, n))
    bands[0, 1:] = -1
    bands[1, :] = 2
    bands[2, :-1] = -1

    rhs = f[1:-1] * dx**2
    u_interior = solve_banded((1,1), bands, rhs)

    u = np.zeros_like(f)
    u[1:-1] = u_interior
    return u


def solve_poisson_fem(f, x):
    dx = x[1] - x[0]
    n = len(x) - 2

    main = 2 * np.ones(n)
    off = -1 * np.ones(n-1)

    A = diags([off, main, off], [-1, 0, 1]) / dx
    b = f[1:-1] * dx

    v_interior = spsolve(A, b)

    v = np.zeros_like(f)
    v[1:-1] = v_interior
    return v

f_data = np.load("poisson_1d_f_train.npy")

x = np.load("poisson_1d_x_train.npy")

N_SAMPLES = f_data.shape[0]

u_data = []
v_data = []

for i in range(N_SAMPLES):
    f = f_data[i]

    u = solve_poisson_fdm(f, x)
    v = solve_poisson_fem(f, x)

    u_data.append(u)
    v_data.append(v)

u_data = np.array(u_data)
v_data = np.array(v_data)