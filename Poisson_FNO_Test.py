"""
Run the trained FNO1d on the held-out test set and compare against FDM/FEM.

Outputs:
  poisson_1d_u_FNO.npy      -- FNO predictions, shape (N_SAMPLES, RESOLUTION)
  fno_vs_fdm_samples.png    -- side-by-side curves + pointwise error for 6 samples
  fno_vs_fem_samples.png    -- same layout vs FEM
  fno_vs_fdm_error_dist.png -- relative L2 error histograms FNO vs FDM
  fno_vs_fem_error_dist.png -- relative L2 error histograms FNO vs FEM

Requires:
  fno1d_weights.pt
  poisson_1d_f_test.npy / poisson_1d_x_test.npy
  poisson_1d_u_FDM.npy / poisson_1d_u_FEM.npy
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from Poisson_FNO_Train import FNO1d

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SAMPLE_INDICES = [0, 40, 200, 500, 1000, 1500]

f_test = np.load("poisson_1d_f_test.npy")   # (N_SAMPLES, RESOLUTION)
x_grid = np.load("poisson_1d_x_test.npy")   # (RESOLUTION,)
u_fdm  = np.load("poisson_1d_u_FDM.npy")    # (N_SAMPLES, RESOLUTION)
u_fem  = np.load("poisson_1d_u_FEM.npy")    # (N_SAMPLES, RESOLUTION)

N_SAMPLES, RESOLUTION = f_test.shape

# ---------------------------------------------------------------------------
# Build model input tensor: (N_SAMPLES, RESOLUTION, 2) = [f(x), x]
# ---------------------------------------------------------------------------
f_tensor = torch.as_tensor(f_test, dtype=torch.float32)
x_tensor = torch.as_tensor(x_grid, dtype=torch.float32).unsqueeze(0).expand(N_SAMPLES, -1)
x_in_all = torch.stack([f_tensor, x_tensor], dim=-1)  # (N_SAMPLES, RESOLUTION, 2)

# ---------------------------------------------------------------------------
# Load model and run inference in batches
# ---------------------------------------------------------------------------
model = FNO1d(modes=16, width=64, in_channels=2, out_channels=1, n_layers=4)
model.load_state_dict(torch.load("fno1d_weights.pt", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

preds = []
with torch.no_grad():
    for start in range(0, N_SAMPLES, BATCH_SIZE):
        batch = x_in_all[start : start + BATCH_SIZE].to(DEVICE)
        out = model(batch)               # (batch, RESOLUTION, 1)
        preds.append(out.cpu().numpy())

u_fno = np.concatenate(preds, axis=0).squeeze(-1)   # (N_SAMPLES, RESOLUTION)

np.save("poisson_1d_u_FNO.npy", u_fno)
print(f"Saved poisson_1d_u_FNO.npy  shape={u_fno.shape}")

# ---------------------------------------------------------------------------
# Helper: relative L2 error per sample
# ---------------------------------------------------------------------------
def rel_l2(pred, ref):
    diff = np.linalg.norm(pred - ref, axis=1)
    norm = np.linalg.norm(ref, axis=1)
    return diff / (norm + 1e-8)


# ---------------------------------------------------------------------------
# Helper: 6-panel comparison figure (curves + pointwise error)
# ---------------------------------------------------------------------------
def plot_comparison(u_a, u_b, label_a, label_b, color_a, color_b, filename):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    fig.suptitle(f"{label_a} vs {label_b} — test samples", fontsize=13)

    for col, idx in enumerate(SAMPLE_INDICES[:3]):
        ax_top = axes[col, 0]
        ax_top.plot(x_grid, u_a[idx], color=color_a, linewidth=1.8, label=label_a)
        ax_top.plot(x_grid, u_b[idx], color=color_b, linewidth=1.5, linestyle="--", label=label_b)
        ax_top.set_title(f"sample {idx}", fontsize=9)
        ax_top.set_ylabel("u(x)", fontsize=8)
        ax_top.tick_params(labelsize=7)
        if col == 0:
            ax_top.legend(fontsize=7)

        ax_err = axes[col, 1]
        ax_err.plot(x_grid, np.abs(u_a[idx] - u_b[idx]), color="grey", linewidth=1.2)
        ax_err.set_title(f"|{label_a} − {label_b}|  s={idx}", fontsize=9)
        ax_err.set_ylabel("|error|", fontsize=8)
        ax_err.tick_params(labelsize=7)

    for col, idx in enumerate(SAMPLE_INDICES[3:]):
        ax_top = axes[col, 2]
        ax_top.plot(x_grid, u_a[idx], color=color_a, linewidth=1.8, label=label_a)
        ax_top.plot(x_grid, u_b[idx], color=color_b, linewidth=1.5, linestyle="--", label=label_b)
        ax_top.set_title(f"sample {idx}", fontsize=9)
        ax_top.tick_params(labelsize=7)

        ax_err = axes[col, 3]
        ax_err.plot(x_grid, np.abs(u_a[idx] - u_b[idx]), color="grey", linewidth=1.2)
        ax_err.set_title(f"|{label_a} − {label_b}|  s={idx}", fontsize=9)
        ax_err.tick_params(labelsize=7)

    for ax in axes[-1, :]:
        ax.set_xlabel("x", fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


# ---------------------------------------------------------------------------
# Helper: error-distribution histogram
# ---------------------------------------------------------------------------
def plot_error_dist(errors, label_a, label_b, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(np.median(errors), color="tomato",    linestyle="--", linewidth=1.5,
               label=f"Median {np.median(errors):.4f}")
    ax.axvline(np.mean(errors),   color="darkorange", linestyle=":",  linewidth=1.5,
               label=f"Mean   {np.mean(errors):.4f}")
    ax.axvline(np.percentile(errors, 90), color="purple", linestyle="-.", linewidth=1.2,
               label=f"90th pct {np.percentile(errors, 90):.4f}")
    ax.set_xlabel(f"Relative L2 error  ({label_a} vs {label_b})")
    ax.set_ylabel("Count")
    ax.set_title(f"{label_a} vs {label_b} — error distribution  (n={len(errors)})")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


# ---------------------------------------------------------------------------
# FNO vs FDM
# ---------------------------------------------------------------------------
plot_comparison(u_fno, u_fdm, "FNO", "FDM", "tomato", "steelblue", "fno_vs_fdm_samples.png")
errors_fdm = rel_l2(u_fno, u_fdm)
plot_error_dist(errors_fdm, "FNO", "FDM", "fno_vs_fdm_error_dist.png")

# ---------------------------------------------------------------------------
# FNO vs FEM
# ---------------------------------------------------------------------------
plot_comparison(u_fno, u_fem, "FNO", "FEM", "tomato", "seagreen",  "fno_vs_fem_samples.png")
errors_fem = rel_l2(u_fno, u_fem)
plot_error_dist(errors_fem, "FNO", "FEM", "fno_vs_fem_error_dist.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nFNO vs FDM — relative L2:")
print(f"  mean={np.mean(errors_fdm):.4f}  median={np.median(errors_fdm):.4f}  90th={np.percentile(errors_fdm,90):.4f}")
print("FNO vs FEM — relative L2:")
print(f"  mean={np.mean(errors_fem):.4f}  median={np.median(errors_fem):.4f}  90th={np.percentile(errors_fem,90):.4f}")
