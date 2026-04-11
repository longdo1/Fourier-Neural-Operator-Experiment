"""
Generate training data for 1D Poisson equation: -u''(x) = f(x)
Domain: [0, 1], BCs: u(0) = u(1) = 0

Source terms f(x) are sampled from a Gaussian Random Field (GRF).
Solutions u(x) are computed via finite differences (tridiagonal solve).

Outputs:
  poisson_1d_f.npy  — shape [N_samples, resolution]
  poisson_1d_u.npy  — shape [N_samples, resolution]
  poisson_1d_x.npy  — shape [resolution]  (grid coordinates)
"""

import numpy as np
from scipy.linalg import solve_banded

# ──────────────────────────────────────────────
# Parameters — adjust these as needed
# ──────────────────────────────────────────────
N_SAMPLES = 2000          # number of (f, u) pairs
RESOLUTION = 256          # grid points (interior + boundary)
ALPHA = 4.0               # GRF smoothness (higher = smoother f)
TAU = 3.0                 # GRF length-scale parameter
SEED = 42                 # reproducibility

np.random.seed(SEED)

# ──────────────────────────────────────────────
# 1. Grid setup
# ──────────────────────────────────────────────
x = np.linspace(0, 1, RESOLUTION)       # includes boundary points
dx = x[1] - x[0]
n_interior = RESOLUTION - 2             # solve only at interior nodes

# ──────────────────────────────────────────────
# 2. Sample f(x) from a Gaussian Random Field
# ──────────────────────────────────────────────
def sample_grf_batch(n_samples, n_points, alpha, tau):
    """
    Draw n_samples realisations of a GRF on [0,1] with n_points grid points.

    Power spectrum:  S(k) = (tau^2 + k^2)^(-alpha/2)
    Higher alpha  → smoother samples.
    Higher tau    → suppresses low-frequency content.
    """
    # Frequencies for real FFT
    k = np.fft.rfftfreq(n_points, d=1.0 / n_points)   # 0, 1, 2, ...

    # Power spectrum (amplitude envelope)
    amplitude = (tau**2 + k**2) ** (-alpha / 2)
    amplitude[0] = 0.0   # zero mean

    # Random Fourier coefficients (complex Gaussian)
    coeffs = np.random.randn(n_samples, len(k)) + \
             1j * np.random.randn(n_samples, len(k))
    coeffs *= amplitude[None, :]

    # Transform to physical space (real part)
    f = np.fft.irfft(coeffs, n=n_points)

    # Normalise so samples have unit variance on average
    f /= (np.std(f) + 1e-8)

    return f

f_all = sample_grf_batch(N_SAMPLES, RESOLUTION, ALPHA, TAU)

# ──────────────────────────────────────────────
# 3. Solve -u''(x) = f(x) via finite differences
# ──────────────────────────────────────────────
def build_tridiag_bands(n):
    """
    Build the banded representation of the tridiagonal matrix A for
    -u'' discretised with central differences:
        A[i,i] = 2,  A[i,i±1] = -1   (divided by dx² later)

    Returns shape (3, n) for scipy.linalg.solve_banded with (1, 1).
    """
    bands = np.zeros((3, n))
    bands[0, 1:] = -1.0    # upper diagonal
    bands[1, :]  =  2.0    # main diagonal
    bands[2, :-1] = -1.0   # lower diagonal
    return bands

bands = build_tridiag_bands(n_interior)

u_all = np.zeros_like(f_all)   # includes boundary (zeros)

for i in range(N_SAMPLES):
    rhs = f_all[i, 1:-1] * dx**2          # interior RHS
    u_interior = solve_banded((1, 1), bands, rhs)
    u_all[i, 1:-1] = u_interior
    # u_all[i, 0] and u_all[i, -1] remain 0 (Dirichlet BCs)

# ──────────────────────────────────────────────
# 4. Quick sanity check
# ──────────────────────────────────────────────
# Verify one sample: compute -u'' numerically and compare to f
u_test = u_all[0]
f_test = f_all[0]
u_pp = -(np.roll(u_test, -1) - 2*u_test + np.roll(u_test, 1)) / dx**2
residual = np.max(np.abs(u_pp[1:-1] - f_test[1:-1]))
print(f"Sanity check — max residual |−u'' − f| at interior points: {residual:.2e}")

# ──────────────────────────────────────────────
# 5. Save
# ──────────────────────────────────────────────
np.save("poisson_1d_f_train.npy", f_all)
np.save("poisson_1d_u_train.npy", u_all)
np.save("poisson_1d_x_train.npy", x)

print(f"Saved {N_SAMPLES} samples at resolution {RESOLUTION}")
print(f"  f shape: {f_all.shape}")
print(f"  u shape: {u_all.shape}")
print(f"  x shape: {x.shape}")

# ──────────────────────────────────────────────
# 6. Optional: plot a few samples
# ──────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    for row in range(3):
        axes[row, 0].plot(x, f_all[row], color="steelblue")
        axes[row, 0].set_ylabel(f"f_{row}(x)")
        axes[row, 1].plot(x, u_all[row], color="coral")
        axes[row, 1].set_ylabel(f"u_{row}(x)")
    axes[0, 0].set_title("Source term f(x)")
    axes[0, 1].set_title("Solution u(x)")
    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("x")
    plt.tight_layout()
    plt.savefig("poisson_1d_samples.png", dpi=150)
    print("Saved plot: poisson_1d_samples.png")
except ImportError:
    print("matplotlib not available — skipping plot")