from __future__ import annotations

import datetime

DAYS: list[str]   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTHS: list[str] = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
MONTH_NUM: dict[str, int] = {m: i + 1 for i, m in enumerate(MONTHS)}


def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_in_month(month: int, year: int) -> int:
    if month == 2:
        return 29 if is_leap_year(year) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def get_day_of_week(day: int, month: int, year: int) -> str:
    return DAYS[datetime.date(year, month, day).weekday()]


def parse_date(date_str: str) -> tuple[int, int, int]:
    d, m, y = date_str.strip().split("-")
    return int(d), int(m), int(y)


def format_date(day: int, month: int, year: int) -> str:
    return f"{day}-{month}-{year}"


def check_conditions(
    day: int, month: int, year: int,
    req_day: str, req_month: str, req_leap: bool, req_decade: int,
) -> dict[str, bool]:
    return {
        "day_of_week": get_day_of_week(day, month, year) == req_day,
        "month":       MONTHS[month - 1] == req_month,
        "leap_year":   is_leap_year(year) == req_leap,
        "decade":      year // 10 == req_decade,
    }


def snap_to_valid(
    year_pred: int, day_pred: int, month_str: str, decade: int, req_day: str,
) -> tuple[int, int, int]:
    month = MONTH_NUM[month_str]
    base = decade * 10
    yr_offset_0 = year_pred - base

    for i in range(10):
        yr_offset = (yr_offset_0 + i) % 10
        year = base + yr_offset
        if year < 1800 or year > 2200:
            continue
        max_d = days_in_month(month, year)
        valid_days = [d for d in range(1, max_d + 1)
                      if get_day_of_week(d, month, year) == req_day]
        if valid_days:
            target  = max(1, min(day_pred, max_d))
            closest = min(valid_days, key=lambda d: abs(d - target))
            return closest, month, year

    year = max(1800, min(2200, base + yr_offset_0))
    return 1, month, year
