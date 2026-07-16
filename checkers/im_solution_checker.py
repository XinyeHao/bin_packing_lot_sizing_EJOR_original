"""IM 可行解约束检查（单件物料、lt_it 阶跃占罐）。"""

from __future__ import annotations

from common.data_models import ProcessedInstance, SolutionResult
from checkers.report import CheckReport
from checkers.utils import almost_equal, is_integer, nested_get
from configuration import CHECKER_TOL


def _get_x(result: SolutionResult, m: str, i: str, t: int) -> float:
    return nested_get(result.packing, m, i, t, default=0.0)


def _get_y(result: SolutionResult, m: str, u: str, t: int) -> float:
    return nested_get(result.setup, m, u, t, default=0.0)


def _get_l(result: SolutionResult, m: str, u: str, t: int) -> float:
    return nested_get(result.extra.get("span", {}), m, u, t, default=0.0)


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
    t_max = periods[-1]
    num_i = len(data.cured_items)
    scrap_qty = result.extra.get("scrap") or {}

    for i in data.cured_items:
        for t in periods:
            inv = nested_get(result.inventory, i, t)
            bo = nested_get(result.backorder, i, t)
            if inv < -tol:
                report.add_violation("im_feasibility", "negative_inventory", f"{i} t={t} 库存为负: {inv}")
            if bo < -tol:
                report.add_violation("im_feasibility", "negative_backorder", f"{i} t={t} 延期为负: {bo}")

    for m in data.machines:
        for u in data.configs:
            for t in periods:
                y = _get_y(result, m, u, t)
                if y < -tol or y > 1 + tol:
                    report.add_violation("im_feasibility", "setup_range", f"Y[{m},{u},{t}]={y} 不在 [0,1]")
                elif y > tol and abs(y - 1) > tol:
                    report.add_violation("im_feasibility", "fractional_setup", f"Y[{m},{u},{t}]={y} 非 0/1")

    for m in data.machines:
        for i in data.cured_items:
            for t in periods:
                x = _get_x(result, m, i, t)
                if x < -tol:
                    report.add_violation("im_feasibility", "negative_packing", f"X[{m},{i},{t}]={x} 为负")
                if x > tol and abs(x - 1) > tol:
                    report.add_violation("im_feasibility", "fractional_packing", f"X[{m},{i},{t}]={x} 非 0/1")

    # 成品流平衡
    for i_idx, i in enumerate(data.cured_items):
        r_i = float(scrap_qty.get(i, 0.0))
        for t_idx, t in enumerate(periods):
            prev_inv = nested_get(result.inventory, i, periods[t_idx - 1]) if t_idx > 0 else 0.0
            prev_bo = nested_get(result.backorder, i, periods[t_idx - 1]) if t_idx > 0 else 0.0
            prod_in = sum(
                _get_x(result, m, i, periods[prod_t_idx])
                for m in data.machines
                for prod_t_idx in range(len(periods))
                if periods[prod_t_idx] + data.lt_it[i_idx][prod_t_idx] - 1 == periods[t_idx]
            )
            dem = data.d_it[i_idx][t_idx] * (1.0 - r_i)
            lhs = prev_inv - prev_bo + prod_in
            rhs = dem + nested_get(result.inventory, i, t) - nested_get(result.backorder, i, t)
            if not almost_equal(lhs, rhs, tol):
                report.add_violation(
                    "im_feasibility",
                    "flow_item",
                    f"{i} t={t} 流量不守恒: lhs={lhs:.4f}, rhs={rhs:.4f}",
                )

    # 单件去向
    for i_idx, i in enumerate(data.cured_items):
        e_i = data.earliest_starts[i_idx]
        d_i = data.deadlines[i_idx]
        x_sum = sum(
            _get_x(result, m, i, t) for m in data.machines for t in periods if e_i <= t <= d_i
        )
        r_i = float(scrap_qty.get(i, 0.0))
        if r_i < -tol:
            report.add_violation("im_feasibility", "negative_scrap", f"R[{i}]={r_i} 为负")
        if r_i > tol and abs(r_i - 1) > tol:
            report.add_violation("im_feasibility", "fractional_scrap", f"R[{i}]={r_i} 非 0/1")
        if not almost_equal(x_sum + r_i, 1.0, tol):
            report.add_violation(
                "im_feasibility",
                "fate",
                f"{i}: 窗内 X 总和 {x_sum:.4f} + R {r_i:.4f} != 1",
            )

    # 时间窗与期末可行
    for i_idx, i in enumerate(data.cured_items):
        e_i = data.earliest_starts[i_idx]
        d_i = data.deadlines[i_idx]
        for m in data.machines:
            for t_idx, t in enumerate(periods):
                x = _get_x(result, m, i, t)
                if x <= tol:
                    continue
                if t < e_i or t > d_i:
                    report.add_violation(
                        "im_feasibility",
                        "wip_window",
                        f"X[{m},{i},{t}]={x} 超出窗口 [{e_i},{d_i}]",
                    )
                if t + data.lt_it[i_idx][t_idx] - 1 > t_max:
                    report.add_violation(
                        "im_feasibility",
                        "horizon",
                        f"X[{m},{i},{t}]={x} 占罐超出计划期",
                    )

    # 每期至多一种 setup
    for m in data.machines:
        for t in periods:
            y_sum = sum(1 for u in data.configs if _get_y(result, m, u, t) >= 0.5)
            if y_sum > 1:
                report.add_violation("im_feasibility", "one_setup", f"{m} t={t} 有 {y_sum} 个 setup")

    # 开始生产与配置匹配
    for u_idx, u in enumerate(data.configs):
        i_list = data.items_by_config[u_idx]
        for m in data.machines:
            for t in periods:
                x_sum = sum(_get_x(result, m, data.cured_items[i_idx], t) for i_idx in i_list)
                y = _get_y(result, m, u, t)
                if x_sum > len(i_list) * y + tol:
                    report.add_violation("im_feasibility", "x_le_y", f"{m},{u},t={t}: X_sum={x_sum} > |I_u| Y")
                if y >= 0.5 and x_sum < 1 - tol:
                    report.add_violation("im_feasibility", "y_le_x", f"{m},{u},t={t}: Y=1 但无生产")
                l_val = _get_l(result, m, u, t)
                if y < 0.5 and l_val > tol:
                    report.add_violation("im_feasibility", "l_without_y", f"{m},{u},t={t}: L={l_val} 但 Y=0")

    for i_idx, i in enumerate(data.cured_items):
        u = data.configs[data.config_of_item[i_idx]]
        for m_idx, m in enumerate(data.machines):
            if data.b_im[i_idx][m_idx] == 0:
                for t in periods:
                    if _get_x(result, m, i, t) >= 0.5:
                        report.add_violation(
                            "im_feasibility",
                            "item_machines",
                            f"X[{m},{i},{t}]=1 但 m∉M_i",
                        )
                continue
            for t in periods:
                x = _get_x(result, m, i, t)
                if x >= 0.5 and _get_y(result, m, u, t) < 0.5:
                    report.add_violation(
                        "im_feasibility",
                        "item_config_mismatch",
                        f"X[{m},{i},{t}]=1 但 Y[{m},{u},{t}]=0",
                    )

    # 跨期下界
    for u_idx, u in enumerate(data.configs):
        for i_idx in data.items_by_config[u_idx]:
            i = data.cured_items[i_idx]
            for m in data.machines:
                for t_idx, t in enumerate(periods):
                    x = _get_x(result, m, i, t)
                    if x < 0.5:
                        continue
                    l_need = data.lt_it[i_idx][t_idx]
                    l_val = _get_l(result, m, u, t)
                    if l_val + tol < l_need:
                        report.add_violation(
                            "im_feasibility",
                            "span_short",
                            f"{m},{u},t={t}: L={l_val} < lt={l_need} (item {i})",
                        )

    # 机台不重叠
    for m in data.machines:
        setups = []
        for u in data.configs:
            for t in periods:
                if _get_y(result, m, u, t) >= 0.5:
                    setups.append((t, u, _get_l(result, m, u, t)))
        setups.sort(key=lambda x: x[0])
        for k in range(len(setups) - 1):
            t1, u1, l1 = setups[k]
            t2, _, _ = setups[k + 1]
            if t1 + l1 > t2 + tol:
                report.add_violation(
                    "im_feasibility",
                    "machine_overlap",
                    f"{m}: setup ({u1},{t1}) 占罐 {l1} 与 t={t2} 重叠",
                )

    # 容量
    for m_idx, m in enumerate(data.machines):
        q = data.q_m[m_idx]
        for t in periods:
            used = sum(data.v_i[i_idx] * _get_x(result, m, data.cured_items[i_idx], t) for i_idx in range(num_i))
            if used > q + tol:
                report.add_violation(
                    "im_feasibility",
                    "capacity",
                    f"{m} t={t} 占用 {used:.4f} > 容量 {q:.4f}",
                )

    report.metrics["violations_count"] = len(report.violations)
    return report
