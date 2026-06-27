"""统一运行全部 checker。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkers.bounds_checker import check_bounds
from checkers.column_checker import check_columns_from_solver
from checkers.im_solution_checker import check_im_feasibility
from checkers.instance_checker import check_processed
from checkers.instance_match_checker import check_instance_match
from checkers.objective_checker import check_objective
from checkers.report import CheckReport
from checkers.subproblem_checker import check_subproblem_solver
from checkers.utils import load_processed_instance, load_solution_result
from common.paths import INSTANCES_DIR, RESULT_DIR
from configuration import (
    CHECK_COLUMN_CG_ITERATIONS,
    CHECK_COLUMN_INIT_TIME,
    CHECK_FRESH_CG_ITERATIONS,
    CHECK_FRESH_SOLVE_TIME_LIMIT,
    DEFAULT_METHOD,
    DEFAULT_MIP_GAP,
    IM_FEASIBILITY_METHODS,
    RESULT_JSON_SKIP,
)
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.solver import solve


def check_solution_file(result_path: Path, run_subproblem: bool = False) -> CheckReport:
    result = load_solution_result(result_path)
    data = load_processed_instance(result.instance_name)

    report = CheckReport(name=f"all:{result.instance_name}:{result.method}")
    report.merge(check_processed(data))
    report.merge(check_instance_match(data, result))

    if result.objective is not None and result.method in IM_FEASIBILITY_METHODS:
        report.merge(check_im_feasibility(data, result))
        report.merge(check_objective(data, result))
        report.merge(check_bounds(result))

    if run_subproblem:
        report.merge(check_subproblem_solver(data))

    return report


def check_fresh_solve(instance_name: str, method: str = DEFAULT_METHOD) -> CheckReport:
    data = load_processed_instance(instance_name)
    result = solve(
        data,
        method=method,
        time_limit=CHECK_FRESH_SOLVE_TIME_LIMIT,
        mip_gap=DEFAULT_MIP_GAP,
        cg_max_iterations=CHECK_FRESH_CG_ITERATIONS,
    )
    report = CheckReport(name=f"fresh:{instance_name}:{method}")
    report.merge(check_processed(data))
    if result.objective is not None and result.method in IM_FEASIBILITY_METHODS:
        report.merge(check_im_feasibility(data, result))
        report.merge(check_objective(data, result))
        report.merge(check_bounds(result))
    report.merge(check_subproblem_solver(data))
    return report


def check_column_generation(instance_name: str) -> CheckReport:
    data = load_processed_instance(instance_name)
    cg = ColumnGenerationSolver(
        data,
        max_iterations=CHECK_COLUMN_CG_ITERATIONS,
        init_time_limit=CHECK_COLUMN_INIT_TIME,
    )
    _, _, columns = cg.run()
    report = check_columns_from_solver(data, columns)
    report.merge(check_subproblem_solver(data))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 solution checkers")
    parser.add_argument("--result", type=str, help="结果 JSON 路径")
    parser.add_argument("--instance", type=str, help="算例名，触发 fresh solve + check")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--cg", action="store_true", help="检查列生成产生的列")
    parser.add_argument("--all-results", action="store_true", help="检查 result/ 下全部 JSON")
    parser.add_argument("--subproblem", action="store_true", help="额外检查子问题求解器")
    args = parser.parse_args()

    reports: list[CheckReport] = []

    if args.all_results:
        for path in sorted(RESULT_DIR.glob("*.json")):
            if path.name in RESULT_JSON_SKIP:
                continue
            reports.append(check_solution_file(path, run_subproblem=args.subproblem))
    elif args.result:
        reports.append(check_solution_file(Path(args.result), run_subproblem=args.subproblem))
    elif args.cg and args.instance:
        reports.append(check_column_generation(args.instance))
    elif args.instance:
        reports.append(check_fresh_solve(args.instance, args.method))
    else:
        demo = INSTANCES_DIR / "demo_small.json"
        if demo.exists():
            reports.append(check_fresh_solve("demo_small", DEFAULT_METHOD))
            reports.append(check_column_generation("demo_small"))
        else:
            parser.error("请指定 --result、--instance 或 --all-results")

    passed = 0
    for rep in reports:
        print(rep.summary())
        print()
        if rep.passed:
            passed += 1

    print(f"总计: {passed}/{len(reports)} 通过")
    sys.exit(0 if passed == len(reports) else 1)


if __name__ == "__main__":
    main()
