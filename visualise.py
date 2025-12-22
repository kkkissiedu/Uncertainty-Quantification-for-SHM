"""
visualise.py — Uncertainty interval plots for deep ensemble and GP predictions.

Usage:
    conda activate cuda_pt
    python visualise.py
"""

# %% Imports
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import load_concrete_data, split_raw
from model import EnsembleModel, load_ensemble


# %% Config
DATA_PATH     = "Concrete_Data.xls"
SAVE_DIR      = "saved_models"
OUT_DIR       = "visualisations"
INPUT_DIM     = 8
HIDDEN_SIZE   = 64
ENSEMBLE_SIZE = 5
TEST_SPLIT    = 0.20
RANDOM_SEED   = 42
CI_Z          = 1.96
N_PLOT        = 50     # samples shown in ensemble error-bar plot


# %% Plot functions
def plot_ensemble(
    y_true:    np.ndarray,
    pred_mean: np.ndarray,
    pred_std:  np.ndarray,
    out_path:  str,
    ci_z:      float,
    n_plot:    int,
) -> None:
    idx = np.arange(n_plot)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(idx, pred_mean[:n_plot], yerr=ci_z * pred_std[:n_plot],
                fmt='o', markersize=4, capsize=3, label=f'Prediction ±{ci_z}σ')
    ax.plot(idx, y_true[:n_plot], 'x', color='tab:red', markersize=5, label='True')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Compressive strength (MPa)')
    ax.set_title('Deep Ensemble — Prediction intervals on test set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_gp(
    y_true:    np.ndarray,
    pred_mean: np.ndarray,
    pred_std:  np.ndarray,
    out_path:  str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, pred_mean, alpha=0.6, edgecolors='k', linewidths=0.4, label='Predictions')
    ax.errorbar(y_true, pred_mean, yerr=CI_Z * pred_std,
                fmt='none', ecolor='lightcoral', alpha=0.4, label='95% CI')
    lims = [y_true.min(), y_true.max()]
    ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('True compressive strength (MPa)')
    ax.set_ylabel('Predicted compressive strength (MPa)')
    ax.set_title('Gaussian Process — True vs predicted with uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# %% Main function
def main(config: dict) -> None:
    os.makedirs(config["out_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Raw test split ────────────────────────────────────────────────────────
    X, y = load_concrete_data(config["data_path"])
    _, X_test_raw, _, y_test = split_raw(
        X, y,
        test_size   = config["test_split"],
        random_seed = config["random_seed"],
    )

    # ── Ensemble plot ─────────────────────────────────────────────────────────
    with open(os.path.join(config["save_dir"], "scaler_ensemble.pkl"), "rb") as f:
        scaler_ens = pickle.load(f)
    X_ens    = scaler_ens.transform(X_test_raw)
    ensemble = load_ensemble(config["save_dir"], config, device)
    with torch.no_grad():
        mean_t, var_t = ensemble(torch.tensor(X_ens, dtype=torch.float32).to(device))
    mean_ens = mean_t.cpu().numpy().flatten()
    std_ens  = np.sqrt(var_t.cpu().numpy().flatten())

    plot_ensemble(
        y_test, mean_ens, std_ens,
        os.path.join(config["out_dir"], "ensemble_uncertainty.png"),
        ci_z   = config["ci_z"],
        n_plot = config["n_plot"],
    )

    # ── GP plot ───────────────────────────────────────────────────────────────
    with open(os.path.join(config["save_dir"], "gp_model.pkl"), "rb") as f:
        gp = pickle.load(f)["model"]

    with open(os.path.join(config["save_dir"], "scaler_gp.pkl"), "rb") as f:
        scaler_gp = pickle.load(f)
    X_gp          = scaler_gp.transform(X_test_raw)
    y_gp, std_gp  = gp.predict(X_gp, return_std=True)

    plot_gp(
        y_test, y_gp, std_gp,
        os.path.join(config["out_dir"], "gp_uncertainty.png"),
    )


# %% Entry point
if __name__ == "__main__":

    config = dict(
        data_path     = DATA_PATH,
        save_dir      = SAVE_DIR,
        out_dir       = OUT_DIR,
        input_dim     = INPUT_DIM,
        hidden_size   = HIDDEN_SIZE,
        ensemble_size = ENSEMBLE_SIZE,
        test_split    = TEST_SPLIT,
        random_seed   = RANDOM_SEED,
        ci_z          = CI_Z,
        n_plot        = N_PLOT,
    )

    main(config)
