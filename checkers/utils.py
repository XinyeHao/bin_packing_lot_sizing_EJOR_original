"""Solution checkers 公共工具。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from common.data_models import SolutionResult
from configuration import CHECKER_TOL
from common.paths import INSTANCES_DIR
from preprocessor.preprocess import load_instance, preprocess


def parse_period_key(key: str | int) -> int:
    return int(key)


def nested_get(mapping: dict, *keys: str | int, default: float = 0.0) -> float:
    current: Any = mapping
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            if key in current:
                current = current[key]
                continue
            str_key = str(key)
            if str_key in current:
                current = current[str_key]
                continue
            return default
        return default
    if current is None:
        return default
    return float(current)


def load_solution_result(path: str | Path) -> SolutionResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SolutionResult(
        method=data["method"],
        instance_name=data["instance_name"],
        status=data["status"],
        objective=data.get("objective"),
        lower_bound=data.get("lower_bound"),
        mip_gap=data.get("mip_gap"),
        runtime_sec=float(data.get("runtime_sec", 0)),
        holding_cost=data.get("holding_cost"),
        backorder_cost=data.get("backorder_cost"),
        production_cost=data.get("production_cost"),
        inventory=data.get("inventory", {}),
        backorder=data.get("backorder", {}),
        assembly=data.get("assembly", {}),
        packing=data.get("packing", {}),
        setup=data.get("setup", {}),
        columns_selected=data.get("columns_selected", []),
        extra=data.get("extra", {}),
    )


def resolve_instance_path(instance_name: str) -> Path:
    path = INSTANCES_DIR / f"{instance_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Instance not found: {path}")
    return path


def load_processed_instance(instance_name: str):
    return preprocess(load_instance(resolve_instance_path(instance_name)))


def setup_window_start(period: int, l_u: int) -> int:
    return max(period - l_u + 1, 1)


def derive_z_from_y(
    setup: dict[str, dict[str, dict[int | str, float]]],
    configs: list[str],
    machines: list[str],
    periods: list[int],
    config_lead_times: dict[str, int],
    tol: float = CHECKER_TOL["setup_derive"],
) -> dict[tuple[str, str, int], int]:
    """由 Y 按约束 (13) 推导 Z。"""
    z: dict[tuple[str, str, int], int] = {}
    for m in machines:
        for u in configs:
            l_u = config_lead_times[u]
            for t in periods:
                t_start = setup_window_start(t, l_u)
                y_sum = sum(
                    1
                    for tp in range(t_start, t + 1)
                    if nested_get(setup, m, u, tp, default=0.0) >= tol
                )
                z[(m, u, t)] = y_sum
    return z


def is_integer(val: float, tol: float = CHECKER_TOL["integer"]) -> bool:
    return abs(val - round(val)) <= tol


def almost_equal(a: float, b: float, tol: float = CHECKER_TOL["almost_equal"]) -> bool:
    return abs(a - b) <= tol
