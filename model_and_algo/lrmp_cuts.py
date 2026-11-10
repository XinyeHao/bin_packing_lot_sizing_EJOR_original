"""LRMP 有效不等式：加强根 CG / B&P 节点松弛。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gurobipy as gp

from common.data_models import ProcessedInstance, ScheduleColumn

if TYPE_CHECKING:
    from gurobipy import GRB

FRAC_TOL = 1e-4

DEFAULT_LRMP_CUTS: dict[str, bool | int | float] = {
    "deadline_cover": True,
    "sri": False,
    "sri_max_cuts": 3,
    "sri_violation_tol": 1e-6,
    "same_mt_setup": False,
    "wip_process_lb": False,
    "tight_deadline": True,
    "setup_pair_static": False,
    "setup_pair_separate": False,
    "setup_pair_sep_max": 40,
    "setup_pair_viol_tol": 1e-4,
}

# 与 DEFAULT 相同：只启用 Deadline Cover（tight RHS）
LRMP_CUTS_STANDARD: dict[str, bool | int | float] = {
    "deadline_cover": True,
    "sri": False,
    "sri_max_cuts": 3,
    "sri_violation_tol": 1e-6,
    "same_mt_setup": False,
    "wip_process_lb": False,
    "tight_deadline": True,
    "setup_pair_static": False,
    "setup_pair_separate": False,
    "setup_pair_sep_max": 30,
    "setup_pair_viol_tol": 1e-4,
}

LRMP_CUTS_STRONG: dict[str, bool | int | float] = {
    "deadline_cover": True,
    "sri": False,
    "sri_max_cuts": 3,
    "sri_violation_tol": 1e-6,
    "same_mt_setup": False,
    "wip_process_lb": False,
    "tight_deadline": True,
    "setup_pair_static": False,
    "setup_pair_separate": False,
    "setup_pair_sep_max": 60,
    "setup_pair_viol_tol": 1e-4,
}


def merge_cut_options(opts: dict | str | None) -> dict:
    """合并切平面开关。非 deadline 类割已停用：即使外部传入 True 也强制关掉。"""
    if opts in ("none", "off"):
        return {
            "deadline_cover": False,
            "sri": False,
            "sri_max_cuts": 0,
            "sri_violation_tol": 1e-6,
            "same_mt_setup": False,
            "wip_process_lb": False,
            "tight_deadline": False,
            "setup_pair_static": False,
            "setup_pair_separate": False,
            "setup_pair_sep_max": 0,
            "setup_pair_viol_tol": 1e-4,
        }
    if opts is None:
        out = dict(DEFAULT_LRMP_CUTS)
    elif opts == "standard":
        out = dict(LRMP_CUTS_STANDARD)
    elif opts == "strong":
        out = dict(LRMP_CUTS_STRONG)
    elif opts == "sri":
        out = dict(LRMP_CUTS_STANDARD)
        out["sri"] = True
    else:
        out = dict(DEFAULT_LRMP_CUTS)
        out.update(opts)
    # 只保留论文中的 deadline-cover；其余结构割一律关闭。
    # 对于显式的 none/off，前面已直接返回。
    out["same_mt_setup"] = False
    out["wip_process_lb"] = False
    out["setup_pair_static"] = False
    out["setup_pair_separate"] = False
    out["tight_deadline"] = True
    return out


def cuts_enabled(opts: dict) -> bool:
    # Deadline Cover 在主问题中始终添加；此处仅表示是否还有额外结构割
    keys = ("same_mt_setup", "wip_process_lb", "setup_pair_static", "setup_pair_separate")
    return any(bool(opts.get(k)) for k in keys)


def column_setup_indicator(col: ScheduleColumn, data: ProcessedInstance, u_idx: int, t_idx: int) -> float:
    u_name = data.configs[u_idx]
    period = data.periods[t_idx]
    return 1.0 if col.y.get((u_name, period), 0) > 0.5 else 0.0


def _setup_interval(data: ProcessedInstance, u_idx: int, t_idx: int) -> tuple[int, int]:
    return t_idx, t_idx + data.l_u[u_idx] - 1


def _intervals_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return s1 <= e2 and s2 <= e1


def max_setups_in_window(data: ProcessedInstance, e_i: int, d_i: int) -> int:
    """窗 [e_i,d_i] 内不相交 setup 个数的贪心上界（1-based period）。"""
    min_l = max(1, min(data.l_u) if data.l_u else 1)
    count = 0
    t = e_i
    while t <= d_i:
        count += 1
        t += min_l
    return count


def max_process_for_item(data: ProcessedInstance, i_idx: int) -> float:
    """物料 i 在窗内可加工量的有效上界。"""
    e_i = data.earliest_starts[i_idx]
    d_i = data.deadlines[i_idx]
    v_i = data.v_i[i_idx]
    setups = max_setups_in_window(data, e_i, d_i)
    cap = 0.0
    for q in data.q_m:
        trays = int(q / max(v_i, 1e-6))
        cap += setups * trays
    return float(cap)


def deadline_cover_rhs(data: ProcessedInstance, tau: int, I: list[int]) -> float:
    """RHS of paper (37)--(38), computed for the batch model.

    ``I`` is E_tau={i: e_i <= tau}.  For each autoclave m, b_mt is the
    maximum cardinality of one feasible batch among items in E_tau that are
    eligible for both m and a common configuration u.  Since every item has
    unit value in this cardinality maximisation, taking the shortest eligible
    items first is exact.  The number of loads that can unload by tau is
    floor(tau / lt_min), where lt_min=min_i l_ti, exactly as in (38).
    """
    if not I:
        return 0.0
    lt_vals = [float(v) for v in data.l_ti if float(v) > 0]
    if not lt_vals:
        return 0.0
    min_l = max(1, int(min(lt_vals)))
    max_setups = max(0, int(tau) // min_l)
    rhs = 0.0
    for m_idx, q in enumerate(data.q_m):
        best = 0
        for u_idx, cfg_items in enumerate(data.items_by_config):
            eligible = [
                i for i in cfg_items
                if i in I and bool(data.b_im[i][m_idx])
            ]
            if not eligible:
                continue
            used = 0.0
            count = 0
            for i in sorted(eligible, key=lambda j: float(data.v_i[j])):
                vi = float(data.v_i[i])
                if used + vi <= float(q) + 1e-9:
                    used += vi
                    count += 1
                else:
                    break
            best = max(best, count)
        rhs += max_setups * best
    return float(rhs)


def enumerate_overlap_pairs(data: ProcessedInstance, m_idx: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """同机时间重叠 setup 对 (u1,t1),(u2,t2)，t1-based。"""
    num_t = len(data.periods)
    num_u = len(data.configs)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    keys: list[tuple[int, int]] = []
    for u_idx in range(num_u):
        for t_idx in range(num_t):
            keys.append((u_idx, t_idx))
    for i, (u1, t1) in enumerate(keys):
        s1, e1 = _setup_interval(data, u1, t1)
        for u2, t2 in keys[i + 1 :]:
            s2, e2 = _setup_interval(data, u2, t2)
            if _intervals_overlap(s1, e1, s2, e2):
                pairs.append(((u1, t1), (u2, t2)))
    return pairs


def build_static_setup_pair_cuts(data: ProcessedInstance) -> list[dict]:
    """静态 setup 冲突对：y_a + y_b <= 1。"""
    cuts: list[dict] = []
    for m_idx in range(len(data.machines)):
        for (u1, t1), (u2, t2) in enumerate_overlap_pairs(data, m_idx):
            cuts.append(
                {
                    "kind": "setup_pair",
                    "m_idx": m_idx,
                    "a": (u1, t1),
                    "b": (u2, t2),
                    "name": f"setup_pair_m{m_idx}_{u1}_{t1}_{u2}_{t2}",
                }
            )
    return cuts


def pair_cut_lhs_value(
    data: ProcessedInstance,
    machine: str,
    m_idx: int,
    columns: list[ScheduleColumn],
    q_vars: list,
    cut: dict,
) -> float:
    u1, t1 = cut["a"]
    u2, t2 = cut["b"]
    val = 0.0
    for col_idx, col in enumerate(columns):
        try:
            qv = q_vars[col_idx].X
        except Exception:
            qv = 0.0
        if qv <= FRAC_TOL:
            continue
        y1 = column_setup_indicator(col, data, u1, t1)
        y2 = column_setup_indicator(col, data, u2, t2)
        val += (y1 + y2) * qv
    return val


def find_violated_setup_pairs(
    data: ProcessedInstance,
    q_vars: dict[str, list],
    columns: dict[str, list[ScheduleColumn]],
    existing_names: set[str],
    *,
    viol_tol: float = 1e-4,
    max_cuts: int = 40,
) -> list[dict]:
    """从当前 LP 解分离 violated setup pair 不等式。"""
    violated: list[tuple[float, dict]] = []
    for m_idx, machine in enumerate(data.machines):
        cols = columns.get(machine, [])
        qv_list = q_vars[machine]
        for (u1, t1), (u2, t2) in enumerate_overlap_pairs(data, m_idx):
            name = f"setup_pair_m{m_idx}_{u1}_{t1}_{u2}_{t2}"
            if name in existing_names:
                continue
            cut = {"kind": "setup_pair", "m_idx": m_idx, "a": (u1, t1), "b": (u2, t2), "name": name}
            lhs = pair_cut_lhs_value(data, machine, m_idx, cols, qv_list, cut)
            if lhs > 1.0 + viol_tol:
                violated.append((lhs - 1.0, cut))
    violated.sort(key=lambda x: -x[0])
    return [c for _, c in violated[:max_cuts]]


def add_setup_expr(
    expr: gp.LinExpr,
    col: ScheduleColumn,
    data: ProcessedInstance,
    m_idx: int,
    u_idx: int,
    t_idx: int,
    q_var,
) -> None:
    coeff = column_setup_indicator(col, data, u_idx, t_idx)
    if coeff > 0:
        expr.addTerms(coeff, q_var)


def add_pair_cut_coeffs(
    expr: gp.LinExpr,
    col: ScheduleColumn,
    data: ProcessedInstance,
    cut: dict,
    q_var,
) -> None:
    u1, t1 = cut["a"]
    u2, t2 = cut["b"]
    add_setup_expr(expr, col, data, cut["m_idx"], u1, t1, q_var)
    add_setup_expr(expr, col, data, cut["m_idx"], u2, t2, q_var)
