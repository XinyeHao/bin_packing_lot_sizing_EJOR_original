"""目标函数与成本分解检查。"""

from __future__ import annotations

from common.data_models import ProcessedInstance, SolutionResult
from checkers.report import CheckReport
from checkers.utils import almost_equal, nested_get
from configuration import CHECKER_TOL


def recompute_costs(data: ProcessedInstance, result: SolutionResult) -> tuple[float, float, float, float, float]:
    holding = sum(
        data.h_c[i_idx] * nested_get(result.inventory, data.cured_items[i_idx], t)
        for i_idx in range(len(data.cured_items))
        for t in data.periods
    )
    backorder = sum(
        data.bc_j[i_idx] * nested_get(result.backorder, data.cured_items[i_idx], t)
        for i_idx in range(len(data.cured_items))
        for t in data.periods
    )
    production = sum(
        data.p_cum[m_idx][u_idx] * nested_get(result.setup, data.machines[m_idx], data.configs[u_idx], t)
        for m_idx in range(len(data.machines))
        for u_idx in range(len(data.configs))
        for t in data.periods
        if nested_get(result.setup, data.machines[m_idx], data.configs[u_idx], t) >= 0.5
    )
    scrap_qty = result.extra.get("scrap") or {}
    scrap = sum(
        data.sc_i[i_idx] * float(scrap_qty.get(data.cured_items[i_idx], 0.0))
        for i_idx in range(len(data.cured_items))
    )
    total = holding + backorder + production + scrap
    return holding, backorder, production, scrap, total


def check_objective(
    data: ProcessedInstance,
    result: SolutionResult,
    tol: float = CHECKER_TOL["objective"],
) -> CheckReport:
    report = CheckReport(name=f"objective:{result.instance_name}:{result.method}")

    if result.objective is None:
        report.metrics["skipped"] = True
        return report

    holding, backorder, production, scrap, total = recompute_costs(data, result)
    report.metrics["recomputed_total"] = round(total, 4)
    report.metrics["reported_total"] = round(result.objective, 4)
    report.metrics["recomputed_scrap"] = round(scrap, 4)

    if not almost_equal(total, result.objective, tol):
        report.add_violation(
            "objective",
            "objective_mismatch",
            f"重算目标 {total:.4f} != 报告目标 {result.objective:.4f}",
        )

    if result.holding_cost is not None and not almost_equal(holding, result.holding_cost, tol):
        report.add_violation(
            "objective",
            "holding_mismatch",
            f"重算 holding {holding:.4f} != 报告 {result.holding_cost:.4f}",
        )

    if result.backorder_cost is not None and not almost_equal(backorder, result.backorder_cost, tol):
        report.add_violation(
            "objective",
            "backorder_mismatch",
            f"重算 backorder {backorder:.4f} != 报告 {result.backorder_cost:.4f}",
        )

    if result.production_cost is not None and not almost_equal(production, result.production_cost, tol):
        report.add_violation(
            "objective",
            "production_mismatch",
            f"重算 production {production:.4f} != 报告 {result.production_cost:.4f}",
        )

    scrap_cost_reported = result.extra.get("scrap_cost")
    if scrap_cost_reported is not None and not almost_equal(scrap, float(scrap_cost_reported), tol):
        report.add_violation(
            "objective",
            "scrap_mismatch",
            f"重算 scrap {scrap:.4f} != 报告 {scrap_cost_reported:.4f}",
        )

    return report
