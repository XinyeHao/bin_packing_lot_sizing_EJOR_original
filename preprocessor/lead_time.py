"""由 config_wait_steps 预处理占罐时段表 lt_it。"""

from __future__ import annotations

import math


def delta_from_wait_steps(wait_steps: list[list[int]] | list[tuple[int, int]], omega: float) -> float:
    """等待阶梯取档、不累加：取满足 threshold ≤ ω 的最大档位的 delta。"""
    if not wait_steps:
        return 0.0
    delta = 0.0
    for thresh, extra in sorted(wait_steps, key=lambda x: x[0]):
        if omega >= thresh:
            delta = float(extra)
    return delta


def compute_lt_it(
    cured_items: list[str],
    item_config: dict[str, str],
    config_lead_times: dict[str, int],
    config_wait_steps: dict[str, list[list[int]]],
    earliest_starts: dict[str, int],
    deadlines: dict[str, int],
    num_periods: int,
    period_hours: float = 1.0,
    item_wait_steps: dict[str, list[list[int]]] | None = None,
) -> list[list[int]]:
    """
    计算 lt_it[i_idx][t_idx]（1-based t 对应下标 t-1）。
    在 rel_i ≤ t ≤ end_i 内随等待分段阶跃；窗外置为基准 l_u（X 已被时间窗约束为 0）。
    """
    item_wait_steps = item_wait_steps or {}
    lt_it: list[list[int]] = []
    for item in cured_items:
        u = item_config[item]
        rel = earliest_starts.get(item, 1)
        end = deadlines.get(item, num_periods)
        base = config_lead_times[u]
        steps = item_wait_steps.get(item) or config_wait_steps.get(u, [])
        row: list[int] = []
        for t in range(1, num_periods + 1):
            if t < rel or t > end:
                row.append(base)
            else:
                omega = (t - rel) * period_hours
                extra = delta_from_wait_steps(steps, omega)
                row.append(base + math.ceil(extra / period_hours))
        lt_it.append(row)
    return lt_it
