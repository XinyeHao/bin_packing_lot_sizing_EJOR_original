"""Branch-and-price 辅助：列与 setup 分支兼容性。"""

from __future__ import annotations

from common.data_models import ProcessedInstance, ScheduleColumn

SetupFix = tuple[int, int, int]  # (u_idx, m_idx, t_idx) -> 0/1


def machine_fixed_y(
    fixed_y: dict[SetupFix, int],
    machine_idx: int,
) -> dict[tuple[int, int], int]:
    """提取单台机器子问题所需的 Y 固定 {(u_idx, t_idx): 0|1}。"""
    out: dict[tuple[int, int], int] = {}
    for (u_idx, m_idx, t_idx), val in fixed_y.items():
        if m_idx == machine_idx:
            out[(u_idx, t_idx)] = val
    return out


def column_has_setup(col: ScheduleColumn, data: ProcessedInstance, u_idx: int, t_idx: int) -> bool:
    u_name = data.configs[u_idx]
    period = data.periods[t_idx]
    return col.y.get((u_name, period), 0) > 0.5


def column_compatible(
    col: ScheduleColumn,
    machine_idx: int,
    data: ProcessedInstance,
    fixed_y: dict[SetupFix, int],
) -> bool:
    """列是否满足当前节点 setup 分支固定。"""
    if not fixed_y:
        return True
    is_empty = not col.y and not col.x
    for (u_idx, m_idx, t_idx), val in fixed_y.items():
        if m_idx != machine_idx:
            continue
        has_setup = column_has_setup(col, data, u_idx, t_idx)
        if val == 0 and has_setup:
            return False
        if val == 1 and not has_setup and not is_empty:
            return False
    return True


def filter_columns(
    columns: dict[str, list[ScheduleColumn]],
    data: ProcessedInstance,
    fixed_y: dict[SetupFix, int],
) -> dict[str, list[ScheduleColumn]]:
    """保留与 fixed_y 兼容的列（按机器过滤）。"""
    filtered: dict[str, list[ScheduleColumn]] = {}
    for m_idx, machine in enumerate(data.machines):
        filtered[machine] = [
            col
            for col in columns.get(machine, [])
            if column_compatible(col, m_idx, data, fixed_y)
        ]
    return filtered


def copy_column_pool(columns: dict[str, list[ScheduleColumn]]) -> dict[str, list[ScheduleColumn]]:
    return {m: list(cols) for m, cols in columns.items()}
