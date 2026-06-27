"""子问题 BLDP / Gurobi 解检查。"""

from __future__ import annotations

import random

from common.data_models import ProcessedInstance
from checkers.column_checker import check_schedule_column
from checkers.report import CheckReport
from configuration import CHECKER_TOL
from model_and_algo.bldp import solve_subproblem_bldp, subproblem_to_column
from model_and_algo.subproblem_gurobi import solve_subproblem_gurobi


def check_subproblem_solver(
    data: ProcessedInstance,
    machine_idx: int = 0,
    seed: int = 0,
) -> CheckReport:
    report = CheckReport(name=f"subproblem:{data.raw.name}:m{machine_idx}")
    rng = random.Random(seed)
    num_t = len(data.periods)
    num_i = len(data.cured_items)
    alpha = [[rng.uniform(0, 10) for _ in range(num_t)] for _ in range(num_i)]
    beta = rng.uniform(0, 5)

    sp_bldp = solve_subproblem_bldp(data, machine_idx, alpha, beta)
    col_bldp = subproblem_to_column(data, machine_idx, 0, sp_bldp)
    report.merge(check_schedule_column(data, machine_idx, col_bldp))
    report.metrics["bldp_reduced_cost"] = round(sp_bldp.reduced_cost, 4)

    sp_gurobi = solve_subproblem_gurobi(data, machine_idx, alpha, beta, time_limit=60)
    col_gurobi = subproblem_to_column(data, machine_idx, 1, sp_gurobi)
    report.merge(check_schedule_column(data, machine_idx, col_gurobi))
    report.metrics["gurobi_reduced_cost"] = round(sp_gurobi.reduced_cost, 4)

    if sp_gurobi.reduced_cost < sp_bldp.reduced_cost - CHECKER_TOL["subproblem"]:
        report.add_violation(
            "subproblem",
            "bldp_not_optimal",
            f"BLDP rc={sp_bldp.reduced_cost:.4f} > Gurobi rc={sp_gurobi.reduced_cost:.4f}",
        )

    return report
