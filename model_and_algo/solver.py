"""求解入口：IM-Gurobi、列生成、CGFO。"""

from __future__ import annotations

from typing import Any

from common.data_models import ProcessedInstance, SolutionResult
from configuration import (
    DEFAULT_BP_MAX_NODES,
    DEFAULT_BP_NODE_IM_TIME_LIMIT,
    DEFAULT_CG_INIT_TIME_CAP,
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_MIP_GAP,
    DEFAULT_TIME_LIMIT,
)
from model_and_algo.branch_and_price import solve_branch_and_price
from model_and_algo.cgfo import solve_cgfo
from model_and_algo.column_generation import solve_column_generation
from model_and_algo.im_model import solve_im


def solve(
    data: ProcessedInstance,
    method: str = "im_gurobi",
    time_limit: float = DEFAULT_TIME_LIMIT,
    mip_gap: float = DEFAULT_MIP_GAP,
    cg_max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
    **kwargs: Any,
) -> SolutionResult:
    """统一求解接口。"""
    method = method.lower()
    if method in {"im", "im_gurobi", "sim"}:
        return solve_im(data, time_limit=time_limit, mip_gap=mip_gap, method="im_gurobi")
    if method in {"cg", "column_generation"}:
        return solve_column_generation(
            data,
            max_iterations=cg_max_iterations,
            init_time_limit=min(time_limit, DEFAULT_CG_INIT_TIME_CAP),
        )
    if method in {"cgfo_i", "cgfo-i", "cgfo1"}:
        return solve_cgfo(
            data,
            variant="cgfo_i",
            fo_time_limit=time_limit,
            mip_gap=mip_gap,
            cg_max_iterations=cg_max_iterations,
        )
    if method in {"cgfo_ii", "cgfo-ii", "cgfo2"}:
        return solve_cgfo(
            data,
            variant="cgfo_ii",
            fo_time_limit=time_limit,
            mip_gap=mip_gap,
            cg_max_iterations=cg_max_iterations,
        )
    if method in {"branch_and_price", "bp"}:
        return solve_branch_and_price(
            data,
            max_nodes=kwargs.get("max_nodes", DEFAULT_BP_MAX_NODES),
            cg_max_iterations=cg_max_iterations,
            im_time_limit=min(time_limit, DEFAULT_BP_NODE_IM_TIME_LIMIT),
            mip_gap=mip_gap,
        )
    raise ValueError(f"Unknown method: {method}")
