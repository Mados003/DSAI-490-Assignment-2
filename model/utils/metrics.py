from __future__ import annotations

from utils.date_utils import check_conditions


def condition_satisfaction_rate(
    predictions: list[tuple[int, int, int]],
    conditions: list[tuple[str, str, bool, int]],
) -> dict[str, float]:
    keys = ["day_of_week", "month", "leap_year", "decade", "all"]
    totals: dict[str, int] = {k: 0 for k in keys}
    n = len(predictions)

    if n == 0:
        return {k: 0.0 for k in keys}

    for (day, month, year), (req_day, req_month, req_leap, req_decade) in zip(predictions, conditions):
        checks = check_conditions(day, month, year, req_day, req_month, req_leap, req_decade)
        for k, v in checks.items():
            if v:
                totals[k] += 1
        if all(checks.values()):
            totals["all"] += 1

    return {k: totals[k] / n for k in keys}
