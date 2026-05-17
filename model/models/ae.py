from __future__ import annotations

import torch
import torch.nn as nn

from utils.tokenizer import CONDITION_DIM

LATENT_DIM_AE: int = 32


class ConditionalAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM_AE) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(CONDITION_DIM + 41, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(CONDITION_DIM + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.year_head = nn.Linear(128, 10)
        self.day_head  = nn.Linear(128, 31)

    def encode(self, cond: torch.Tensor, year_oh: torch.Tensor, day_oh: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([cond, year_oh, day_oh], dim=1))

    def decode(self, cond: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder_net(torch.cat([cond, z], dim=1))
        return self.year_head(h), self.day_head(h)

    def forward(self, cond: torch.Tensor, year_oh: torch.Tensor, day_oh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode(cond, self.encode(cond, year_oh, day_oh))

    def generate(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(cond.size(0), self.latent_dim, device=cond.device)
        return self.decode(cond, z)
