# %% Imports
import os

import torch
import torch.nn as nn


# =============================================================================
# Single ensemble member
# =============================================================================

class AleatoricNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,   hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.mu      = nn.Linear(hidden_size, 1)
        self.log_var = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.net(x)
        return self.mu(h), self.log_var(h)


# =============================================================================
# Ensemble wrapper
# =============================================================================

class EnsembleModel(nn.Module):
    def __init__(self, models: list) -> None:
        super().__init__()
        self.members = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> tuple:
        mus, vars_ = [], []
        for m in self.members:
            mu, log_var = m(x)
            mus.append(mu)
            vars_.append(torch.exp(log_var))

        mus   = torch.stack(mus,   dim=0)   # (M, N, 1)
        vars_ = torch.stack(vars_, dim=0)

        mean_pred = mus.mean(dim=0)
        epistemic = mus.var(dim=0, unbiased=False)
        aleatoric = vars_.mean(dim=0)
        return mean_pred, epistemic + aleatoric


# =============================================================================
# Loss
# =============================================================================

def gaussian_nll_loss(
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    target:  torch.Tensor,
) -> torch.Tensor:
    precision = torch.exp(-log_var)
    return torch.mean(precision * (target - mu) ** 2 + log_var)


# =============================================================================
# Checkpoint loader
# =============================================================================

def load_ensemble(
    checkpoint_dir: str,
    config:         dict,
    device:         torch.device,
) -> EnsembleModel:
    members = []
    for i in range(config["ensemble_size"]):
        ckpt = torch.load(
            os.path.join(checkpoint_dir, f"ensemble_member_{i}.pth"),
            map_location=device,
        )
        m = AleatoricNet(config["input_dim"], config["hidden_size"]).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        members.append(m)
    return EnsembleModel(members).to(device)
