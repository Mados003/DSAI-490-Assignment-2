from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from utils.date_utils import parse_date
from utils.tokenizer import DAYS_IDX, MIN_DECADE, MONTHS_IDX, encode_conditions, encode_targets, parse_full_line


class DateDataset(Dataset):
    def __init__(self, filepath: str | Path) -> None:
        self.records: list[dict] = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                day_s, mon_s, leap, decade, date_str = parse_full_line(line)
                d, m, y = parse_date(date_str)
                year_t, day_t = encode_targets(y, d, decade)
                self.records.append({
                    "cond":         encode_conditions(day_s, mon_s, leap, decade),
                    "day_tok":      DAYS_IDX[day_s],
                    "mon_tok":      MONTHS_IDX[mon_s],
                    "leap_tok":     1 if leap else 0,
                    "decade_tok":   decade - MIN_DECADE,
                    "year_target":  year_t,
                    "day_target":   day_t,
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        return {
            "cond":         r["cond"],
            "day_tok":      torch.tensor(r["day_tok"],     dtype=torch.long),
            "mon_tok":      torch.tensor(r["mon_tok"],     dtype=torch.long),
            "leap_tok":     torch.tensor(r["leap_tok"],    dtype=torch.long),
            "decade_tok":   torch.tensor(r["decade_tok"],  dtype=torch.long),
            "year_target":  torch.tensor(r["year_target"], dtype=torch.long),
            "day_target":   torch.tensor(r["day_target"],  dtype=torch.long),
        }


def get_dataloaders(
    filepath: str | Path,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    dataset = DateDataset(filepath)
    n_train = int(len(dataset) * train_ratio)
    n_test  = len(dataset) - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=gen)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
