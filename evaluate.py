"""
evaluate.py — Compare deep ensemble and GP uncertainty models on the test set.

Usage:
    conda activate cuda_pt
    python evaluate.py
"""

# %% Imports
import os
import pickle

import numpy as np
import torch
from sklearn.metrics import r2_score

from dataset import load_concrete_data, split_raw
from model import EnsembleModel, load_ensemble


# %% Config
DATA_PATH     = "Concrete_Data.xls"
SAVE_DIR      = "saved_models"
INPUT_DIM     = 8
HIDDEN_SIZE   = 64
ENSEMBLE_SIZE = 5
TEST_SPLIT    = 0.20
RANDOM_SEED   = 42
CI_Z          = 1.96    # 95% confidence interval multiplier


# %% Metric functions
def picp_mpiw(
    y_true:    np.ndarray,
    pred_mean: np.ndarray,
    pred_std:  np.ndarray,
    z:         float,
) -> tuple:
    lower = pred_mean - z * pred_std
    upper = pred_mean + z * pred_std
    picp  = float(np.mean((y_true >= lower) & (y_true <= upper))) * 100.0
    mpiw  = float(np.mean(upper - lower))
    return picp, mpiw


# =============================================================================
# Per-model evaluation
# =============================================================================

def evaluate_ensemble(
    config: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> dict:
    # ── Load scaler ───────────────────────────────────────────────────────────
    with open(os.path.join(config["save_dir"], "scaler_ensemble.pkl"), "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_test)

    # ── Load ensemble ─────────────────────────────────────────────────────────
    ensemble = load_ensemble(config["save_dir"], config, device)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        mean_t, var_t = ensemble(X_tensor)

    mean_np = mean_t.cpu().numpy().flatten()
    std_np  = np.sqrt(var_t.cpu().numpy().flatten())

    r2           = float(r2_score(y_test, mean_np))
    rmse         = float(np.sqrt(np.mean((y_test - mean_np) ** 2)))
    picp, mpiw   = picp_mpiw(y_test, mean_np, std_np, config["ci_z"])
    return {"R²": r2, "RMSE (MPa)": rmse, "PICP 95% (%)": picp, "MPIW (MPa)": mpiw}


def evaluate_gp(
    config: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    with open(os.path.join(config["save_dir"], "gp_model.pkl"), "rb") as f:
        payload = pickle.load(f)
    gp = payload["model"]

    with open(os.path.join(config["save_dir"], "scaler_gp.pkl"), "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_test)

    y_pred, y_std = gp.predict(X_scaled, return_std=True)
    r2            = float(r2_score(y_test, y_pred))
    rmse          = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    picp, mpiw    = picp_mpiw(y_test, y_pred, y_std, config["ci_z"])
    return {"R²": r2, "RMSE (MPa)": rmse, "PICP 95% (%)": picp, "MPIW (MPa)": mpiw}


# =============================================================================
# Table printer
# =============================================================================

FMT = {
    "R²"          : ".4f",
    "RMSE (MPa)"  : ".4f",
    "PICP 95% (%)": ".2f",
    "MPIW (MPa)"  : ".4f",
}


def print_table(ens: dict, gp: dict) -> None:
    col = 18
    sep = "+" + "-" * 22 + "+" + "-" * col + "+" + "-" * col + "+"
    print(sep)
    print(f"| {'Metric':<20} | {'Deep Ensemble':>{col - 2}} | {'GP':>{col - 2}} |")
    print(sep)
    for metric, fmt in FMT.items():
        print(f"| {metric:<20} | {ens[metric]:{col - 2}{fmt}} | {gp[metric]:{col - 2}{fmt}} |")
    print(sep)


# %% Main function
def main(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Raw test split — each evaluator applies its own saved scaler ──────────
    X, y = load_concrete_data(config["data_path"])
    _, X_test_raw, _, y_test = split_raw(
        X, y,
        test_size   = config["test_split"],
        random_seed = config["random_seed"],
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Evaluating deep ensemble...")
    ens_metrics = evaluate_ensemble(config, X_test_raw, y_test, device)

    print("Evaluating GP...")
    gp_metrics  = evaluate_gp(config, X_test_raw, y_test)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\nTest Set Results")
    print_table(ens_metrics, gp_metrics)


# %% Entry point
if __name__ == "__main__":

    config = dict(
        data_path     = DATA_PATH,
        save_dir      = SAVE_DIR,
        input_dim     = INPUT_DIM,
        hidden_size   = HIDDEN_SIZE,
        ensemble_size = ENSEMBLE_SIZE,
        test_split    = TEST_SPLIT,
        random_seed   = RANDOM_SEED,
        ci_z          = CI_Z,
    )

    main(config)
