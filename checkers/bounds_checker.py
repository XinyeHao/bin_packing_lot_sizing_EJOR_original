"""上下界一致性检查。"""

from __future__ import annotations

from common.data_models import SolutionResult
from checkers.report import CheckReport
from configuration import CHECKER_TOL


def check_bounds(result: SolutionResult, tol: float = CHECKER_TOL["bounds"]) -> CheckReport:
    report = CheckReport(name=f"bounds:{result.instance_name}:{result.method}")

    if result.objective is None:
        report.metrics["skipped"] = "no_upper_bound"
        return report

    if result.lower_bound is not None and result.objective + tol < result.lower_bound:
        report.add_violation(
            "bounds",
            "ub_lt_lb",
            f"上界 {result.objective:.4f} < 下界 {result.lower_bound:.4f}",
        )

    if result.mip_gap is not None and result.lower_bound is not None and result.objective > 0:
        expected_gap = abs(result.objective - result.lower_bound) / abs(result.objective)
        if abs(expected_gap - result.mip_gap) > CHECKER_TOL["mip_gap_report"]:
            report.metrics["mip_gap_note"] = (
                f"报告 gap={result.mip_gap:.4f}, 重算 gap={expected_gap:.4f}（求解器与重算可能略有差异）"
            )

    if result.method in {"cgfo_i", "cgfo_ii"} and result.extra.get("cg_lower_bound") is not None:
        cg_lb = float(result.extra["cg_lower_bound"])
        if result.objective + tol < cg_lb:
            report.add_violation(
                "bounds",
                "cgfo_ub_lt_cg_lb",
                f"CGFO 上界 {result.objective:.4f} < CG 下界 {cg_lb:.4f}",
            )

    return report
