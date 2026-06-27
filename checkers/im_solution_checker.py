"""SIM 可行解约束检查。"""

from __future__ import annotations

import math

from common.data_models import ProcessedInstance, SolutionResult
from checkers.report import CheckReport
from checkers.utils import (
    almost_equal,
    derive_z_from_y,
    is_integer,
    nested_get,
    setup_window_start,
)
from configuration import CHECKER_TOL
from preprocessor.preprocess import max_tray_count


def _get_x(result: SolutionResult, m: str, i: str, t: int) -> float:
    return nested_get(result.packing, m, i, t, default=0.0)


def _get_y(result: SolutionResult, m: str, u: str, t: int) -> float:
    return nested_get(result.setup, m, u, t, default=0.0)


def check_im_feasibility(
    data: ProcessedInstance,
    result: SolutionResult,
    tol: float = CHECKER_TOL["feasibility"],
) -> CheckReport:
    report = CheckReport(name=f"im_feasibility:{result.instance_name}:{result.method}")

    if result.objective is None:
        report.add_violation("im_feasibility", "no_solution", "结果不含可行目标值，跳过约束检查")
        return report

    periods = data.periods
    num_i = len(data.cured_items)
    end_offset = num_i

    # --- 非负性与整数性 ---
    for item in data.all_items:
        for t in periods:
            inv = nested_get(result.inventory, item, t)
            if inv < -tol:
                report.add_violation("im_feasibility", "negative_inventory", f"{item} 在 t={t} 库存为负: {inv}")
            if not is_integer(inv, tol):
                report.add_violation("im_feasibility", "fractional_inventory", f"{item} 在 t={t} 库存非整数: {inv}")

    for j in data.end_items:
        for t in periods:
            bo = nested_get(result.backorder, j, t)
            asm = nested_get(result.assembly, j, t)
            if bo < -tol:
                report.add_violation("im_feasibility", "negative_backorder", f"{j} 在 t={t} 延期为负: {bo}")
            if asm < -tol:
                report.add_violation("im_feasibility", "negative_assembly", f"{j} 在 t={t} 装配量为负: {asm}")

    for m in data.machines:
        for u in data.configs:
            for t in periods:
                y = _get_y(result, m, u, t)
                if y < -tol or y > 1 + tol:
                    report.add_violation("im_feasibility", "setup_range", f"Y[{m},{u},{t}]={y} 不在 [0,1]")
                elif abs(y) > tol and abs(y - 1) > tol:
                    report.add_violation("im_feasibility", "fractional_setup", f"Y[{m},{u},{t}]={y} 非 0/1")

    for m in data.machines:
        for i in data.cured_items:
            for t in periods:
                x = _get_x(result, m, i, t)
                if x < -tol:
                    report.add_violation("im_feasibility", "negative_packing", f"X[{m},{i},{t}]={x} 为负")
                if not is_integer(x, tol):
                    report.add_violation("im_feasibility", "fractional_packing", f"X[{m},{i},{t}]={x} 非整数")

    # --- 约束 (2): 成品流量守恒 ---
    for j_idx, j in enumerate(data.end_items):
        item_idx = end_offset + j_idx
        for t_idx, t in enumerate(periods):
            prev_inv = nested_get(result.inventory, j, periods[t_idx - 1]) if t_idx > 0 else 0.0
            prev_bo = nested_get(result.backorder, j, periods[t_idx - 1]) if t_idx > 0 else 0.0
            lhs = prev_inv - prev_bo + nested_get(result.assembly, j, t)
            rhs = data.d_jt[j_idx][t_idx] + nested_get(result.inventory, j, t) - nested_get(result.backorder, j, t)
            if not almost_equal(lhs, rhs, tol):
                report.add_violation(
                    "im_feasibility",
                    "flow_end",
                    f"成品 {j} t={t} 流量不守恒: lhs={lhs:.4f}, rhs={rhs:.4f}",
                    j=j,
                    t=t,
                )

    # --- 约束 (3): 固化物料库存平衡 ---
    for i_idx, i in enumerate(data.cured_items):
        lti = int(data.l_ti[i_idx])
        for t_idx, t in enumerate(periods):
            prev_inv = nested_get(result.inventory, i, periods[t_idx - 1]) if t_idx > 0 else 0.0
            prod_in = sum(
                _get_x(result, m, i, periods[t_idx - lti])
                for m in data.machines
                if t_idx - lti >= 0
            )
            demand = sum(data.r_ij[i_idx][j_idx] * nested_get(result.assembly, data.end_items[j_idx], t) for j_idx in range(len(data.end_items)))
            lhs = prev_inv + prod_in
            rhs = demand + nested_get(result.inventory, i, t)
            if not almost_equal(lhs, rhs, tol):
                report.add_violation(
                    "im_feasibility",
                    "flow_cured",
                    f"固化物料 {i} t={t} 流量不守恒: lhs={lhs:.4f}, rhs={rhs:.4f}",
                    i=i,
                    t=t,
                )

    z_vals = derive_z_from_y(result.setup, data.configs, data.machines, periods, data.raw.config_lead_times)

    # --- 约束 (5): 每时段至多一种配置运行 ---
    for m in data.machines:
        for t in periods:
            z_sum = sum(1 for u in data.configs if z_vals.get((m, u, t), 0) == 1)
            if z_sum > 1:
                report.add_violation("im_feasibility", "one_config", f"{m} t={t} 同时运行 {z_sum} 种配置")

    # --- 约束 (13): Y-Z 关联 ---
    for m in data.machines:
        for u in data.configs:
            l_u = data.raw.config_lead_times[u]
            for t in periods:
                t_start = setup_window_start(t, l_u)
                y_sum = sum(1 for tp in range(t_start, t + 1) if _get_y(result, m, u, tp) >= 0.5)
                z = z_vals.get((m, u, t), 0)
                if y_sum != z:
                    report.add_violation(
                        "im_feasibility",
                        "link_yz",
                        f"{m},{u},t={t}: sum(Y)={y_sum} != Z={z}",
                    )

    # --- 约束 (8): 配置-装箱关联 ---
    for i_idx, i in enumerate(data.cured_items):
        for m_idx, m in enumerate(data.machines):
            cap_count = max_tray_count(data.q_m[m_idx], data.v_i[i_idx])
            for t in periods:
                x = _get_x(result, m, i, t)
                cfg_match = sum(data.b_iu[i_idx][u_idx] * _get_y(result, m, data.configs[u_idx], t) for u_idx in range(len(data.configs)))
                if x > cap_count * cfg_match + tol:
                    report.add_violation(
                        "im_feasibility",
                        "config_packing",
                        f"X[{m},{i},{t}]={x} 超过配置允许上限 {cap_count * cfg_match}",
                    )
                if cfg_match < 0.5 and x > tol:
                    report.add_violation(
                        "im_feasibility",
                        "packing_without_setup",
                        f"{m},{i},t={t}: X={x} 但无匹配 setup",
                    )

    # --- 约束 (9): 热压罐容量 ---
    for m_idx, m in enumerate(data.machines):
        q = data.q_m[m_idx]
        for t in periods:
            used = sum(data.v_i[i_idx] * _get_x(result, m, data.cured_items[i_idx], t) for i_idx in range(num_i))
            if used > q + tol:
                report.add_violation(
                    "im_feasibility",
                    "capacity",
                    f"{m} t={t} 占用长度 {used:.4f} > 容量 {q:.4f}",
                )

    report.metrics["violations_count"] = len(report.violations)
    return report
