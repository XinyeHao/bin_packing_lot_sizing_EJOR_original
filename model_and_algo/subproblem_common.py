"""定价子问题共享数据结构。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from common.data_models import ProcessedInstance, ScheduleColumn


@dataclass
class SubproblemResult:
    reduced_cost: float
    x: dict[tuple[int, int], int]
    y: dict[tuple[int, int], int]
    production_cost: float
    # False means the pricing oracle stopped before proving its optimum.  Such
    # a result may provide a feasible improving column, but it must never be
    # used to certify CG convergence or a DW lower bound.
    is_optimal: bool = True


def scale_lengths(values: list[float], q: float) -> tuple[int, list[int]]:
    scale = 100
    capacity = max(1, round(q * scale))
    weights = [max(1, round(v * scale)) for v in values]
    return capacity, weights


def column_fingerprint(col: ScheduleColumn) -> tuple[tuple, tuple]:
    """列 (x,y) 结构签名，用于去重。"""
    x_sig = tuple(sorted((k, v) for k, v in col.x.items() if v))
    y_sig = tuple(sorted((k, v) for k, v in col.y.items() if v))
    return (x_sig, y_sig)


def subproblem_to_column(
    data: ProcessedInstance,
    machine_idx: int,
    column_id: int,
    sp_result: SubproblemResult,
) -> ScheduleColumn:
    machine = data.machines[machine_idx]
    x_named: dict[tuple[str, int], int] = {}
    y_named: dict[tuple[str, int], int] = {}

    for (i_idx, t_idx), qty in sp_result.x.items():
        x_named[(data.cured_items[i_idx], data.periods[t_idx])] = qty
    for (u_idx, t_idx), val in sp_result.y.items():
        if val:
            y_named[(data.configs[u_idx], data.periods[t_idx])] = 1

    return ScheduleColumn(
        machine=machine,
        column_id=column_id,
        x=x_named,
        y=y_named,
        production_cost=sp_result.production_cost,
        reduced_cost=sp_result.reduced_cost,
    )
