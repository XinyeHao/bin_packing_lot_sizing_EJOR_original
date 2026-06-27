"""各环节系统自检。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkers.bounds_checker import check_bounds
from checkers.im_solution_checker import check_im_feasibility
from checkers.instance_checker import check_instance_raw, check_processed
from checkers.objective_checker import check_objective
from checkers.report import CheckReport
from checkers.subproblem_checker import check_subproblem_solver
from checkers.utils import load_processed_instance, load_solution_result
from common.paths import INSTANCES_DIR, RESULT_DIR
from configuration import (
    AUDIT_CG_INSTANCE_NAMES,
    AUDIT_INSTANCE_NAMES,
    AUDIT_OUTPUT_FILE,
    AUDIT_RESULT_PATTERN,
    AUDIT_SUBPROBLEM_INSTANCES,
    CHECKER_TOL,
    DEFAULT_CG_MAX_ITERATIONS,
    IM_FEASIBILITY_METHODS,
    SET_B_TEST_LOG,
)
from model_and_algo.column_generation import ColumnGenerationSolver
from preprocessor.preprocess import load_instance, preprocess


def audit_instances(names: list[str]) -> list[CheckReport]:
    reports = []
    for name in names:
        data = preprocess(load_instance(INSTANCES_DIR / f"{name}.json"))
        r = CheckReport(name=f"instance:{name}")
        r.merge(check_instance_raw(data.raw))
        r.merge(check_processed(data))
        reports.append(r)
    return reports


def audit_bounds_from_csv(csv_path: Path) -> CheckReport:
    import csv

    report = CheckReport(name="bound_ordering")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    by_inst: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("status") == "error" or not row.get("value"):
            continue
        by_inst.setdefault(row["instance"], {})[row["method"]] = float(row["value"])

    tol = CHECKER_TOL["bound_order"]
    for inst, vals in by_inst.items():
        lim = vals.get("lim")
        cg = vals.get("cg")
        im = vals.get("im_gurobi")
        if lim and cg and lim > cg + tol:
            report.add_violation("bounds", "lim_gt_cg", f"{inst}: LIM={lim:.1f} > CG={cg:.1f}")
        if cg and im and cg > im + tol:
            report.add_violation(
                "bounds",
                "cg_gt_im",
                f"{inst}: CG下界={cg:.1f} > IM上界={im:.1f}（IM为time_limit时可能正常）",
            )
    report.metrics["instances_checked"] = len(by_inst)
    return report


def audit_cg_convergence(name: str) -> CheckReport:
    report = CheckReport(name=f"cg_convergence:{name}")
    data = load_processed_instance(name)
    cg = ColumnGenerationSolver(data, max_iterations=DEFAULT_CG_MAX_ITERATIONS)
    lb, rcs, cols = cg.run()
    rc_tol = CHECKER_TOL["cg_convergence"]
    report.metrics["iterations"] = len(rcs)
    report.metrics["columns"] = sum(len(v) for v in cols.values())
    report.metrics["last_sum_rc"] = round(rcs[-1], 4) if rcs else None
    report.metrics["converged"] = bool(rcs and rcs[-1] > rc_tol)
    report.metrics["lb"] = round(lb, 2)
    if rcs and rcs[-1] <= rc_tol:
        report.add_violation(
            "cg",
            "not_converged",
            f"达到 max_iterations 仍未满足 sum(rc)>{rc_tol}，last_rc={rcs[-1]:.4f}",
        )
    return report


def audit_saved_results(pattern: str) -> list[CheckReport]:
    reports = []
    for path in sorted(RESULT_DIR.glob(pattern)):
        if "summary" in path.name or path.suffix != ".json":
            continue
        try:
            result = load_solution_result(path)
            if result.objective is None or result.method not in IM_FEASIBILITY_METHODS:
                continue
            data = load_processed_instance(result.instance_name)
            r = CheckReport(name=f"result:{path.name}")
            r.merge(check_im_feasibility(data, result))
            r.merge(check_objective(data, result))
            r.merge(check_bounds(result))
            reports.append(r)
        except Exception as exc:
            r = CheckReport(name=f"result:{path.name}")
            r.add_violation("load", "error", str(exc))
            reports.append(r)
    return reports


def main() -> None:
    all_reports: list[CheckReport] = []

    print("=== 1. 算例数据 ===")
    for r in audit_instances(list(AUDIT_INSTANCE_NAMES)):
        all_reports.append(r)
        print(r.summary(), "\n")

    print("=== 2. 子问题 BLDP ===")
    for name in AUDIT_SUBPROBLEM_INSTANCES:
        data = load_processed_instance(name)
        r = check_subproblem_solver(data)
        all_reports.append(r)
        print(r.summary(), "\n")

    print("=== 3. CG 收敛性（抽样）===")
    for name in AUDIT_CG_INSTANCE_NAMES:
        r = audit_cg_convergence(name)
        all_reports.append(r)
        print(r.summary(), "\n")

    print("=== 4. 已保存可行解 ===")
    for r in audit_saved_results(AUDIT_RESULT_PATTERN):
        all_reports.append(r)
        print(r.summary(), "\n")

    csv_path = RESULT_DIR / SET_B_TEST_LOG
    if csv_path.exists():
        print("=== 5. Set-B 下界/上界序 ===")
        r = audit_bounds_from_csv(csv_path)
        all_reports.append(r)
        print(r.summary(), "\n")

    passed = sum(1 for r in all_reports if r.passed)
    print(f"=== 总计: {passed}/{len(all_reports)} 项通过 ===")

    out = RESULT_DIR / AUDIT_OUTPUT_FILE
    out.write_text(
        json.dumps(
            [{"name": r.name, "passed": r.passed, "violations": len(r.violations), "metrics": r.metrics} for r in all_reports],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
