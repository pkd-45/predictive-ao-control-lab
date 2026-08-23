from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .dataset import load_split, make_windows
from .models import PredictorMLP


def fit_linear(cfg: Config, dataset_path: str | Path, output: str | Path, ridge: float = 1e-3) -> Path:
    train_phase, _ = load_split(dataset_path, "train")
    x, y = make_windows(train_phase, cfg.history, cfg.wfs_noise_sigma, cfg.seed + 31)
    x2 = x.reshape(len(x), -1).astype(np.float64)
    y2 = y.astype(np.float64)
    x_aug = np.column_stack([x2, np.ones(len(x2))])
    reg = ridge * np.eye(x_aug.shape[1])
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, weights=beta[:-1], bias=beta[-1])
    return output


def train_mlp(cfg: Config, dataset_path: str | Path, output: str | Path) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train_phase, _ = load_split(dataset_path, "train")
    val_phase, _ = load_split(dataset_path, "val")
    x_train, y_train = make_windows(train_phase, cfg.history, cfg.wfs_noise_sigma, cfg.seed + 41)
    x_val, y_val = make_windows(val_phase, cfg.history, cfg.wfs_noise_sigma, cfg.seed + 42)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    model = PredictorMLP(cfg.history, cfg.n_modes, cfg.training.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state = None
    history: list[dict[str, float]] = []
    xv = torch.from_numpy(x_val)
    yv = torch.from_numpy(y_val)
    for epoch in range(cfg.training.epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().item()) * len(xb)
            count += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv))
        train_loss = total / max(count, 1)
        history.append({"epoch": epoch + 1, "train_mse": train_loss, "val_mse": val_loss})
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "history": cfg.history,
            "n_modes": cfg.n_modes,
            "hidden": cfg.training.hidden,
            "best_val_mse": best,
        },
        output,
    )
    output.with_suffix(".training.json").write_text(json.dumps(history, indent=2))
    return output


def load_mlp(path: str | Path) -> PredictorMLP:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = PredictorMLP(int(payload["history"]), int(payload["n_modes"]), int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
