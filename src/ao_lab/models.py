from __future__ import annotations

import torch
from torch import nn


class PredictorMLP(nn.Module):
    def __init__(self, history: int, n_modes: int, hidden: int = 48) -> None:
        super().__init__()
        self.history = history
        self.n_modes = n_modes
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history * n_modes, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_modes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
