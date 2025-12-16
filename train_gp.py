"""
train_gp.py — Gaussian process surrogate for concrete strength prediction.

Usage:
    conda activate cuda_pt
    python train_gp.py
"""

# %% Imports
import os
import pickle

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score

from dataset import create_splits, load_concrete_data


# %% Config
DATA_PATH          = "Concrete_Data.xls"
SAVE_DIR           = "saved_models"
TEST_SPLIT         = 0.20
RANDOM_SEED        = 42
GP_CONSTANT        = 1.0
GP_RBF_LENGTH      = 1.0
GP_NOISE_LEVEL     = 1.0
GP_N_RESTARTS      = 10


# %% Model factory
def build_gp(config: dict) -> GaussianProcessRegressor:
    kernel = (
        C(config["gp_constant"]) *
        RBF(length_scale=config["gp_rbf_length"]) +
        WhiteKernel(noise_level=config["gp_noise_level"])
    )
    return GaussianProcessRegressor(
        kernel               = kernel,
        n_restarts_optimizer = config["gp_n_restarts"],   # marginal likelihood optimisation replaces a val loop
        random_state         = config["random_seed"],
    )


# %% Main function
def main(config: dict) -> None:
    np.random.seed(config["random_seed"])
    os.makedirs(config["save_dir"], exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    X, y = load_concrete_data(config["data_path"])
    X_train, X_test, y_train, y_test, scaler = create_splits(
        X, y,
        test_size   = config["test_split"],
        random_seed = config["random_seed"],
    )

    # ── Save scaler ───────────────────────────────────────────────────────────
    scaler_path = os.path.join(config["save_dir"], "scaler_gp.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {scaler_path}")

    # ── Train ─────────────────────────────────────────────────────────────────
    gp = build_gp(config)
    print("Fitting GP...")
    gp.fit(X_train, y_train)
    print(f"Optimised kernel: {gp.kernel_}")

    # ── Quick eval ────────────────────────────────────────────────────────────
    y_pred, _ = gp.predict(X_test, return_std=True)
    r2        = r2_score(y_test, y_pred)
    rmse      = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    print(f"R²: {r2:.4f}  RMSE: {rmse:.4f} MPa")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(config["save_dir"], "gp_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": gp, "config": config}, f)
    print(f"GP saved → {model_path}")


# %% Entry point
if __name__ == "__main__":

    config = dict(
        data_path      = DATA_PATH,
        save_dir       = SAVE_DIR,
        test_split     = TEST_SPLIT,
        random_seed    = RANDOM_SEED,
        gp_constant    = GP_CONSTANT,
        gp_rbf_length  = GP_RBF_LENGTH,
        gp_noise_level = GP_NOISE_LEVEL,
        gp_n_restarts  = GP_N_RESTARTS,
    )

    main(config)
