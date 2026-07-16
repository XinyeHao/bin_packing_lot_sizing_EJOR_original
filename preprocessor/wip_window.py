"""WIP 可开始加工时间窗 [E_i, D_i] 工具函数。"""

from __future__ import annotations

from common.data_models import ProcessedInstance


def wip_window(data: ProcessedInstance, i_idx: int) -> tuple[int, int]:
    """返回物料 i 的 (最早开始加工时段 E_i, 最晚开始加工时段 D_i)，均为 1-based。"""
    return data.earliest_starts[i_idx], data.deadlines[i_idx]


def can_start_processing(data: ProcessedInstance, i_idx: int, period: int) -> bool:
    """时段 period（1-based）是否允许对物料 i 开始加工（进罐）。"""
    e_i, d_i = wip_window(data, i_idx)
    return e_i <= period <= d_i
