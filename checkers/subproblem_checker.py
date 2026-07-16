"""子问题 EBRP / Gurobi §3.3 解检查。"""

from __future__ import annotations

import random

from common.data_models import ProcessedInstance
from checkers.column_checker import check_schedule_column
from checkers.report import CheckReport
from configuration import CHECKER_TOL, EBRP_EXACT_MAX_ITEMS
from model_and_algo.pricing_ebrp import solve_subproblem_ebrp
from model_and_algo.pricing_im_gurobi import solve_subproblem_im_gurobi
from model_and_algo.subproblem_common import subproblem_to_column


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
    rho = [rng.uniform(0, 2) for _ in range(num_i)]

    sp_gurobi = solve_subproblem_im_gurobi(
        data, machine_idx, alpha, beta, rho=rho, time_limit=120.0,
    )
    col_gurobi = subproblem_to_column(data, machine_idx, 0, sp_gurobi)
    report.merge(check_schedule_column(data, machine_idx, col_gurobi))
    report.metrics["gurobi_reduced_cost"] = round(sp_gurobi.reduced_cost, 4)

    if num_i <= EBRP_EXACT_MAX_ITEMS:
        sp_ebrp = solve_subproblem_ebrp(data, machine_idx, alpha, beta, rho=rho)
        col_ebrp = subproblem_to_column(data, machine_idx, 1, sp_ebrp)
        report.merge(check_schedule_column(data, machine_idx, col_ebrp))
        report.metrics["ebrp_reduced_cost"] = round(sp_ebrp.reduced_cost, 4)

        diff = abs(sp_gurobi.reduced_cost - sp_ebrp.reduced_cost)
        if diff > CHECKER_TOL["subproblem"]:
            report.add_violation(
                "subproblem",
                "ebrp_gurobi_mismatch",
                f"EBRP rc={sp_ebrp.reduced_cost:.4f} vs Gurobi rc={sp_gurobi.reduced_cost:.4f}",
            )

    return report
