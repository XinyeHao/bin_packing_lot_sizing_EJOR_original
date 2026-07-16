"""计算结果后处理与输出。"""

from __future__ import annotations

import json
from pathlib import Path

from common.data_models import SolutionResult
from common.paths import RESULT_DIR


def enrich_result(result: SolutionResult) -> SolutionResult:
    """补充成本占比等派生指标。"""
    if result.objective and result.objective > 0:
        result.extra["cost_share"] = {
            "holding": (result.holding_cost or 0) / result.objective,
            "backorder": (result.backorder_cost or 0) / result.objective,
            "production": (result.production_cost or 0) / result.objective,
            "scrap": (result.extra.get("scrap_cost") or 0) / result.objective,
        }
    if result.objective is not None and result.lower_bound is not None and result.objective > 0:
        result.extra["optimality_gap"] = (result.objective - result.lower_bound) / result.objective
    return result


def save_result(result: SolutionResult, output_dir: Path | None = None) -> Path:
    """将求解结果写入 result 目录。"""
    out_dir = output_dir or RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.instance_name}_{result.method}.json"
    path.write_text(json.dumps(enrich_result(result).to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def format_summary(result: SolutionResult) -> str:
    """生成可读摘要。"""
    lines = [
        f"算例: {result.instance_name}",
        f"方法: {result.method}",
        f"状态: {result.status}",
        f"目标值: {result.objective}",
        f"下界: {result.lower_bound}",
        f"MIP Gap: {result.mip_gap}",
        f"运行时间(s): {result.runtime_sec:.2f}",
    ]
    if result.holding_cost is not None:
        lines.append(f"  库存成本: {result.holding_cost:.2f}")
    if result.backorder_cost is not None:
        lines.append(f"  延期成本: {result.backorder_cost:.2f}")
    if result.production_cost is not None:
        lines.append(f"  生产成本: {result.production_cost:.2f}")
    if result.method in {"cg", "column_generation"}:
        extra = result.extra or {}
        lines.append(f"LRMP 下界: {result.lower_bound:.2f}" if result.lower_bound is not None else "LRMP 下界: —")
        lines.append(f"迭代次数: {extra.get('num_iterations', '—')}")
        lines.append(f"收敛: {extra.get('converged', False)}  停止原因: {extra.get('stop_reason') or '—'}")
        if extra.get("last_min_rc") is not None:
            lines.append(f"末轮 min_RC: {extra['last_min_rc']:.6f}")
        ncol = extra.get("num_columns")
        if ncol:
            lines.append(f"列数: {sum(ncol.values())}  (各机 {ncol})")
        if extra.get("total_scrap_qty") is not None:
            lines.append(f"LRMP 报废量: {extra.get('total_scrap_qty')}")
    return "\n".join(lines)
