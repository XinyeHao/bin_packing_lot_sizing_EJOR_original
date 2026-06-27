"""列（子问题解）可行性检查。"""

from __future__ import annotations

from common.data_models import ProcessedInstance, ScheduleColumn
from checkers.report import CheckReport
from checkers.utils import setup_window_start
from configuration import CHECKER_TOL
from preprocessor.preprocess import max_tray_count


def _column_y(col: ScheduleColumn, u: str, t: int) -> int:
    return col.y.get((u, t), 0)


def _column_x(col: ScheduleColumn, i: str, t: int) -> int:
    return col.x.get((i, t), 0)


def check_schedule_column(
    data: ProcessedInstance,
    machine_idx: int,
    column: ScheduleColumn,
    tol: float = CHECKER_TOL["column"],
) -> CheckReport:
    report = CheckReport(name=f"column:{column.machine}:{column.column_id}")
    m = data.machines[machine_idx]
    q = data.q_m[machine_idx]
    periods = data.periods
    num_u = len(data.configs)

    # 推导 Z
    z: dict[tuple[str, int], int] = {}
    for u_idx, u in enumerate(data.configs):
        l_u = data.l_u[u_idx]
        for t in periods:
            t_start = setup_window_start(t, l_u)
            z[(u, t)] = sum(_column_y(column, u, tp) for tp in range(t_start, t + 1))

    for t in periods:
        if sum(z[(u, t)] for u in data.configs) > 1:
            report.add_violation("column", "one_config", f"t={t} 多配置同时运行")

    for u_idx, u in enumerate(data.configs):
        l_u = data.l_u[u_idx]
        for t in periods:
            t_start = setup_window_start(t, l_u)
            y_sum = sum(_column_y(column, u, tp) for tp in range(t_start, t + 1))
            if y_sum != z[(u, t)]:
                report.add_violation("column", "link_yz", f"{u},t={t}: sum(Y)={y_sum} != Z={z[(u,t)]}")

    for i_idx, i in enumerate(data.cured_items):
        cap = max_tray_count(q, data.v_i[i_idx])
        for t in periods:
            x = _column_x(column, i, t)
            cfg = sum(data.b_iu[i_idx][u_idx] * _column_y(column, data.configs[u_idx], t) for u_idx in range(num_u))
            if x > cap * cfg + tol:
                report.add_violation("column", "config_packing", f"X[{i},{t}]={x} 超配置上限")

    for t in periods:
        used = sum(data.v_i[i_idx] * _column_x(column, data.cured_items[i_idx], t) for i_idx in range(len(data.cured_items)))
        if used > q + tol:
            report.add_violation("column", "capacity", f"t={t} 占用 {used:.4f} > {q:.4f}")

    recomputed_prod = sum(
        data.p_cum[machine_idx][u_idx] * _column_y(column, data.configs[u_idx], t)
        for u_idx in range(num_u)
        for t in periods
    )
    if abs(recomputed_prod - column.production_cost) > CHECKER_TOL["column_cost"]:
        report.add_violation(
            "column",
            "production_cost",
            f"列生产成本 {column.production_cost:.4f} != 重算 {recomputed_prod:.4f}",
        )

    return report


def check_columns_from_solver(
    data: ProcessedInstance,
    columns: dict[str, list[ScheduleColumn]],
) -> CheckReport:
    report = CheckReport(name=f"columns:{data.raw.name}")
    failed = 0
    checked = 0
    for m_idx, machine in enumerate(data.machines):
        for col in columns.get(machine, []):
            checked += 1
            sub = check_schedule_column(data, m_idx, col)
            if not sub.passed:
                failed += 1
                report.merge(sub)
    report.metrics["columns_checked"] = checked
    report.metrics["columns_failed"] = failed
    if failed == 0:
        report.metrics["result"] = "all_columns_feasible"
    return report
