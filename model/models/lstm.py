from __future__ import annotations

import torch
import torch.nn as nn

from utils.tokenizer import NUM_DECADES


class ConditionalLSTM(nn.Module):
    def __init__(self, d_embed: int = 32, hidden: int = 128, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.decade_emb = nn.Embedding(NUM_DECADES, d_embed)
        self.leap_emb   = nn.Embedding(2, d_embed)
        self.mon_emb    = nn.Embedding(12, d_embed)
        self.day_emb    = nn.Embedding(7, d_embed)
        self.lstm = nn.LSTM(
            input_size=d_embed,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.year_head = nn.Linear(hidden, 10)
        self.day_head  = nn.Linear(hidden, 31)

    def forward(self, day_tok: torch.Tensor, mon_tok: torch.Tensor,
                leap_tok: torch.Tensor, decade_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = torch.stack([
            self.decade_emb(decade_tok),
            self.leap_emb(leap_tok),
            self.mon_emb(mon_tok),
            self.day_emb(day_tok),
        ], dim=1)
        _, (h_n, _) = self.lstm(seq)
        h = h_n[-1]
        return self.year_head(h), self.day_head(h)
