"""CGFO 启发式：基于 LRMP 列的 Fix-and-Optimize。"""

from __future__ import annotations

import time

from common.data_models import ProcessedInstance, ScheduleColumn, SolutionResult
from configuration import (
    CGFO_THRESHOLD_I,
    CGFO_THRESHOLD_II,
    DEFAULT_CG_INIT_TIME_LIMIT,
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_MIP_GAP,
    DEFAULT_TIME_LIMIT,
)
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_im


def _build_allowed_setup(
    data: ProcessedInstance,
    selected_columns: dict[str, list[ScheduleColumn]],
) -> dict[tuple[int, int, int], int]:
    """从选中列构造允许 setup 的 (u_idx, m_idx, t_idx) 集合。"""
    allowed: set[tuple[int, int, int]] = set()
    for m_idx, machine in enumerate(data.machines):
        for col in selected_columns.get(machine, []):
            for (u_name, period), val in col.y.items():
                if val:
                    allowed.add((data.config_index[u_name], m_idx, period - 1))
    return {key: 1 for key in allowed}


def _build_fixed_y(
    data: ProcessedInstance,
    allowed: dict[tuple[int, int, int], int],
) -> dict[tuple[int, int, int], int]:
    """不在允许集合内的 Y 固定为 0。"""
    num_u = len(data.configs)
    num_m = len(data.machines)
    num_t = len(data.periods)
    fixed: dict[tuple[int, int, int], int] = {}
    allowed_set = set(allowed.keys())
    for m_idx in range(num_m):
        for u_idx in range(num_u):
            for t_idx in range(num_t):
                if (u_idx, m_idx, t_idx) not in allowed_set:
                    fixed[(u_idx, m_idx, t_idx)] = 0
    return fixed


def solve_cgfo(
    data: ProcessedInstance,
    variant: str = "cgfo_ii",
    cg_max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
    init_time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
    fo_time_limit: float = DEFAULT_TIME_LIMIT,
    mip_gap: float = DEFAULT_MIP_GAP,
) -> SolutionResult:
    """CGFO-I / CGFO-II 启发式。"""
    start = time.perf_counter()
    threshold = CGFO_THRESHOLD_I if variant == "cgfo_i" else CGFO_THRESHOLD_II

    cg = ColumnGenerationSolver(
        data,
        max_iterations=cg_max_iterations,
        init_time_limit=init_time_limit,
    )
    lb, _, _ = cg.run()
    selected = cg.get_selected_columns(threshold=threshold)

    allowed = _build_allowed_setup(data, selected)
    fixed_y = _build_fixed_y(data, allowed) if allowed else None

    fo_result = solve_im(
        data,
        time_limit=fo_time_limit,
        mip_gap=mip_gap,
        fixed_y=fixed_y,
        method=variant,
    )
    fo_result.lower_bound = lb
    fo_result.runtime_sec = time.perf_counter() - start
    fo_result.columns_selected = [col.column_id for cols in selected.values() for col in cols]
    fo_result.extra.setdefault("instance_seed", data.raw.seed)
    fo_result.extra.setdefault("set_type", data.raw.set_type)
    fo_result.extra.setdefault("num_periods", data.raw.num_periods)
    fo_result.extra["cg_lower_bound"] = lb
    return fo_result
