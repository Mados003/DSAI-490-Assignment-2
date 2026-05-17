from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.date_utils import DAYS, MONTHS, MONTH_NUM

DAYS_IDX: dict[str, int]   = {d: i for i, d in enumerate(DAYS)}
MONTHS_IDX: dict[str, int] = {m: i for i, m in enumerate(MONTHS)}

MIN_DECADE: int  = 180
MAX_DECADE: int  = 220
NUM_DECADES: int = MAX_DECADE - MIN_DECADE + 1

CONDITION_DIM: int = 7 + 12 + 1 + NUM_DECADES


def encode_conditions(day: str, month: str, leap: bool, decade: int) -> torch.Tensor:
    day_oh = F.one_hot(torch.tensor(DAYS_IDX[day]), 7).float()
    mon_oh = F.one_hot(torch.tensor(MONTHS_IDX[month]), 12).float()
    leap_f = torch.tensor([1.0 if leap else 0.0])
    dec_oh = F.one_hot(torch.tensor(decade - MIN_DECADE), NUM_DECADES).float()
    return torch.cat([day_oh, mon_oh, leap_f, dec_oh])


def encode_targets(year: int, day: int, decade: int) -> tuple[int, int]:
    return year - decade * 10, day - 1


def parse_condition_line(line: str) -> tuple[str, str, bool, int]:
    t = line.strip().split()
    return t[0][1:-1], t[1][1:-1], t[2][1:-1] == "True", int(t[3][1:-1])


def parse_full_line(line: str) -> tuple[str, str, bool, int, str]:
    t = line.strip().split()
    return t[0][1:-1], t[1][1:-1], t[2][1:-1] == "True", int(t[3][1:-1]), t[4]


def format_output_line(
    day_str: str, month_str: str, leap: bool, decade: int,
    out_day: int, out_month: int, out_year: int,
) -> str:
    return f"[{day_str}] [{month_str}] [{'True' if leap else 'False'}] [{decade}] {out_day}-{out_month}-{out_year}"
