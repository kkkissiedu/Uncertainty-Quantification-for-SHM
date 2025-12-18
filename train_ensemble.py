"""
train_ensemble.py — Deep ensemble with aleatoric UQ for concrete strength prediction.

Usage:
    conda activate cuda_pt
    python train_ensemble.py
"""

# %% Imports
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from dataset import create_splits, load_concrete_data
from model import AleatoricNet, gaussian_nll_loss


# %% Config
DATA_PATH     = "Concrete_Data.xls"
SAVE_DIR      = "saved_models"
INPUT_DIM     = 8
HIDDEN_SIZE   = 64
ENSEMBLE_SIZE = 5
NUM_EPOCHS    = 300
BATCH_SIZE    = 64
LR            = 1e-3
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.20
RANDOM_SEED   = 42
MODEL_NAME    = "deep-ensemble"


# %% DataLoader factory
def to_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).view(-1, 1),
    )


def make_loaders(
    X_tr:       np.ndarray,
    y_tr:       np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    batch_size: int,
) -> tuple:
    kw       = dict(batch_size=batch_size, num_workers=0)
    train_dl = DataLoader(to_dataset(X_tr,  y_tr),  shuffle=True,  **kw)
    val_dl   = DataLoader(to_dataset(X_val, y_val), shuffle=False, **kw)
    return train_dl, val_dl


# %% Training function
def train_member(
    model:      AleatoricNet,
    train_dl:   DataLoader,
    val_dl:     DataLoader,
    config:     dict,
    member_idx: int,
    device:     torch.device,
    writer:     SummaryWriter,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    best_val  = float("inf")
    save_path = os.path.join(config["save_dir"], f"ensemble_member_{member_idx}.pth")

    for epoch in range(1, config["num_epochs"] + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            mu, log_var = model(X_b)
            loss = gaussian_nll_loss(mu, log_var, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                mu, log_var = model(X_b)
                val_loss += gaussian_nll_loss(mu, log_var, y_b).item()
        val_loss /= len(val_dl)

        # ── TensorBoard log ───────────────────────────────────────────────────
        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar(f"member_{member_idx}/train_loss", train_loss, epoch)
        writer.add_scalar(f"member_{member_idx}/val_loss",   val_loss,   epoch)
        writer.add_scalar(f"member_{member_idx}/lr",         lr,         epoch)

        new_best = val_loss < best_val
        if new_best:
            best_val = val_loss
            torch.save({
                "epoch"            : epoch,
                "model_state_dict" : model.state_dict(),
                "best_val_loss"    : best_val,
                "config"           : config,
            }, save_path)

        flag = " *" if new_best else ""
        print(
            f"  Epoch {epoch:3d}/{config['num_epochs']}"
            f"  member {member_idx + 1}/{config['ensemble_size']}"
            f"  train {train_loss:.4f}"
            f"  val {val_loss:.4f}{flag}"
        )

    return best_val


# %% Main function
def main(config: dict) -> None:

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    os.makedirs(config["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    X, y = load_concrete_data(config["data_path"])
    X_train, _, y_train, _, scaler = create_splits(
        X, y,
        test_size   = config["test_split"],
        random_seed = config["random_seed"],
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size    = config["val_split"],
        random_state = config["random_seed"],
    )

    # ── Save scaler ───────────────────────────────────────────────────────────
    scaler_path = os.path.join(config["save_dir"], "scaler_ensemble.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {scaler_path}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dl, val_dl = make_loaders(X_tr, y_tr, X_val, y_val, config["batch_size"])

    # ── TensorBoard ───────────────────────────────────────────────────────────
    run_dir = f"runs/{config['model_name']}_{datetime.now():%Y%m%d_%H%M%S}"
    writer  = SummaryWriter(run_dir)
    print(f"TensorBoard: {run_dir}")

    # ── Train ensemble ────────────────────────────────────────────────────────
    for i in range(config["ensemble_size"]):
        # Distinct random init per member, deterministic across runs
        torch.manual_seed(config["random_seed"] + i)
        model    = AleatoricNet(config["input_dim"], config["hidden_size"]).to(device)
        best_val = train_member(model, train_dl, val_dl, config, i, device, writer)
        print(f"Member {i + 1}/{config['ensemble_size']} best val loss: {best_val:.4f}\n")

    writer.close()
    print(f"All {config['ensemble_size']} members trained. Checkpoints in {config['save_dir']}/")


# %% Entry point
if __name__ == "__main__":

    config = dict(
        data_path     = DATA_PATH,
        save_dir      = SAVE_DIR,
        input_dim     = INPUT_DIM,
        hidden_size   = HIDDEN_SIZE,
        ensemble_size = ENSEMBLE_SIZE,
        num_epochs    = NUM_EPOCHS,
        batch_size    = BATCH_SIZE,
        lr            = LR,
        val_split     = VAL_SPLIT,
        test_split    = TEST_SPLIT,
        random_seed   = RANDOM_SEED,
        model_name    = MODEL_NAME,
    )

    main(config)
