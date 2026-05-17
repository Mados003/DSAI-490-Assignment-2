from __future__ import annotations

import torch
import torch.nn as nn

from utils.tokenizer import NUM_DECADES


class DateTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.day_emb    = nn.Embedding(7, d_model)
        self.mon_emb    = nn.Embedding(12, d_model)
        self.leap_emb   = nn.Embedding(2, d_model)
        self.decade_emb = nn.Embedding(NUM_DECADES, d_model)
        self.pos_emb    = nn.Embedding(4, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.year_head = nn.Linear(d_model * 4, 10)
        self.day_head  = nn.Linear(d_model * 4, 31)

    def forward(self, day_tok: torch.Tensor, mon_tok: torch.Tensor,
                leap_tok: torch.Tensor, decade_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = day_tok.size(0)
        pos = torch.arange(4, device=day_tok.device).unsqueeze(0).expand(B, -1)
        tokens = torch.stack([
            self.day_emb(day_tok)       + self.pos_emb(pos[:, 0]),
            self.mon_emb(mon_tok)       + self.pos_emb(pos[:, 1]),
            self.leap_emb(leap_tok)     + self.pos_emb(pos[:, 2]),
            self.decade_emb(decade_tok) + self.pos_emb(pos[:, 3]),
        ], dim=1)
        out = self.transformer(tokens).flatten(1)
        return self.year_head(out), self.day_head(out)
