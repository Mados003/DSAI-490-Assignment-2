from __future__ import annotations

import torch
import torch.nn as nn

from utils.tokenizer import CONDITION_DIM

NOISE_DIM: int = 64


class CGANGenerator(nn.Module):
    def __init__(self, noise_dim: int = NOISE_DIM) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim + CONDITION_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 41),
        )

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([noise, cond], dim=1))
        return out[:, :10], out[:, 10:]


class CGANDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONDITION_DIM + 41, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, cond: torch.Tensor, year_soft: torch.Tensor, day_soft: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, year_soft, day_soft], dim=1))
