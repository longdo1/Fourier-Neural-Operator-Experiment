"""
Analysis of the trained FNO1d model for the 1D Poisson equation.

Produces three figures:
  1. fno_loss_history.png      — training and validation loss curves
  2. fno_val_predictions.png   — predicted vs true u(x) on sample validation cases
  3. fno_error_distribution.png — per-sample relative L2 error histogram over the full val set

Requires:
  fno1d_weights.pt   — saved by Poisson_FNO_Train.py
  fno1d_history.json — saved by Poisson_FNO_Train.py (re-run training if missing)
  poisson_1d_f_train.npy / poisson_1d_u_train.npy
"""

from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Poisson_FNO_Train import FNO1d, FUPairDataset, LpLoss

# ---------------------------------------------------------------------------
# Reproducible validation split — must match Poisson-FNO-Train exactly
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 20
TRAIN_FRAC = 0.8
SPLIT_SEED = 42

f_all = np.load("poisson_1d_f_train.npy")   # (2000, 256)
u_all = np.load("poisson_1d_u_train.npy")   # (2000, 256)

dataset = FUPairDataset(f_all, u_all, domain=(0.0, 1.0))
n_train = int(TRAIN_FRAC * len(dataset))
n_val = len(dataset) - n_train
_, val_set = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SPLIT_SEED),
)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Grid coordinates (same for every sample)
N = f_all.shape[-1]
x_grid = np.linspace(0.0, 1.0, N)

# ---------------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------------
model = FNO1d(modes=16, width=64, in_channels=2, out_channels=1, n_layers=4)
model.load_state_dict(torch.load("fno1d_weights.pt", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------------------------------------------------------------------
# Figure 1: Loss history
# ---------------------------------------------------------------------------
with open("fno1d_history.json") as fh:
    history = json.load(fh)

train_loss = history["train"]
val_loss = history["val"]
epochs = np.arange(1, len(train_loss) + 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(epochs, train_loss, label="Train", linewidth=1.5)
ax.semilogy(epochs, val_loss,   label="Validation", linewidth=1.5, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Relative L2 loss")
ax.set_title("FNO1d — Training history")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("fno_loss_history.png", dpi=150)
plt.close(fig)
print("Saved fno_loss_history.png")

# ---------------------------------------------------------------------------
# Collect all validation predictions for figures 2 & 3
# ---------------------------------------------------------------------------
all_preds, all_targets = [], []
with torch.no_grad():
    for x_in, u_true in val_loader:
        x_in = x_in.to(DEVICE)
        pred = model(x_in)                     # (batch, N, 1)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(u_true.numpy())

all_preds   = np.concatenate(all_preds,   axis=0).squeeze(-1)   # (n_val, N)
all_targets = np.concatenate(all_targets, axis=0).squeeze(-1)   # (n_val, N)

# Recover the source terms for the val split in the same order
val_indices = val_set.indices
f_val = f_all[val_indices]   # (n_val, N)

# ---------------------------------------------------------------------------
# Figure 2: Predictions vs truth on 6 validation samples
# ---------------------------------------------------------------------------
SAMPLE_INDICES = [0, 40, 80, 120, 160, 200]

fig = plt.figure(figsize=(14, 8))
fig.suptitle("FNO1d — Predictions vs truth (validation samples)", fontsize=13)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

for plot_idx, sample_idx in enumerate(SAMPLE_INDICES):
    row, col = divmod(plot_idx, 3)
    ax = fig.add_subplot(gs[row, col])

    u_true = all_targets[sample_idx]
    u_pred = all_preds[sample_idx]
    f_i    = f_val[sample_idx]

    rel_err = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)

    ax.plot(x_grid, u_true, color="steelblue",  linewidth=1.8, label="True u")
    ax.plot(x_grid, u_pred, color="tomato",     linewidth=1.5, linestyle="--", label="FNO pred")
    ax.plot(x_grid, f_i,    color="grey",       linewidth=0.9, linestyle=":",  alpha=0.6, label="f(x)")
    ax.set_title(f"val[{sample_idx}]  rel-L2={rel_err:.3f}", fontsize=9)
    ax.set_xlabel("x", fontsize=8)
    if col == 0:
        ax.set_ylabel("u(x)", fontsize=8)
    ax.tick_params(labelsize=7)
    if plot_idx == 0:
        ax.legend(fontsize=7, loc="upper right")

fig.savefig("fno_val_predictions.png", dpi=150)
plt.close(fig)
print("Saved fno_val_predictions.png")

# ---------------------------------------------------------------------------
# Figure 3: Per-sample relative L2 error distribution (full val set)
# ---------------------------------------------------------------------------
errors = np.linalg.norm(all_preds - all_targets, axis=1) / (
    np.linalg.norm(all_targets, axis=1) + 1e-8
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(errors, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
ax.axvline(np.median(errors), color="tomato",    linestyle="--", linewidth=1.5,
           label=f"Median {np.median(errors):.4f}")
ax.axvline(np.mean(errors),   color="darkorange", linestyle=":",  linewidth=1.5,
           label=f"Mean   {np.mean(errors):.4f}")
ax.axvline(np.percentile(errors, 90), color="purple", linestyle="-.", linewidth=1.2,
           label=f"90th pct {np.percentile(errors, 90):.4f}")

ax.set_xlabel("Per-sample relative L2 error")
ax.set_ylabel("Count")
ax.set_title(f"FNO1d — Error distribution  (n={len(errors)} val samples)")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("fno_error_distribution.png", dpi=150)
plt.close(fig)
print("Saved fno_error_distribution.png")

print(f"\nVal set summary:")
print(f"  n_samples    : {len(errors)}")
print(f"  mean  rel-L2 : {np.mean(errors):.4f}")
print(f"  median rel-L2: {np.median(errors):.4f}")
print(f"  std          : {np.std(errors):.4f}")
print(f"  90th pct     : {np.percentile(errors, 90):.4f}")
print(f"  max          : {np.max(errors):.4f}")
