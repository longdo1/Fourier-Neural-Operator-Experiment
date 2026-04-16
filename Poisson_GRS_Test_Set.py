"""
Generate test set for 1D Poisson equation: -u''(x) = f(x)
Domain: [0, 1], BCs: u(0) = u(1) = 0

Source terms f(x) are sampled from a Gaussian Random Field (GRF).

Outputs:
  poisson_1d_f.npy  — shape [N_samples, resolution]
  poisson_1d_x.npy  — shape [resolution]  (grid coordinates)
"""

import numpy as np
from scipy.linalg import solve_banded

# ──────────────────────────────────────────────
# Parameters — adjust these as needed
# ──────────────────────────────────────────────
N_SAMPLES = 2000          # number of f samples
RESOLUTION = 256          # grid points (interior + boundary)
ALPHA = 4.0               # GRF smoothness (higher = smoother f)
TAU = 3.0                 # GRF length-scale parameter
SEED = 43                 # reproducibility

np.random.seed(SEED)

# ──────────────────────────────────────────────
# 1. Sample f(x) from a Gaussian Random Field
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

np.save("data/poisson_1d_f_test.npy", f_all)
x = np.linspace(0, 1, RESOLUTION)
np.save("data/poisson_1d_x_test.npy", x)

print(f"Saved {N_SAMPLES} samples at resolution {RESOLUTION}")
print(f"  f shape: {f_all.shape}")
print(f"  x shape: {x.shape}")


