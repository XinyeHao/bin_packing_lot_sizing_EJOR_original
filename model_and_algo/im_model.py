"""集成模型 IM：批次按最长固化时间共同出罐、共同进入库存。"""

from __future__ import annotations

import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from common.data_models import ProcessedInstance, SolutionResult
from configuration import (
    CHECKER_TOL,
    DEFAULT_LIM_TIME_LIMIT,
    DEFAULT_MIP_GAP,
    DEFAULT_TIME_LIMIT,
)


def _big_m(data: ProcessedInstance) -> int:
    num_t = len(data.periods)
    max_lt = max(max(row) for row in data.lt_it) if data.lt_it else 1
    return num_t + max_lt


def _item_start_feasible(data: ProcessedInstance, i_idx: int, t_idx: int, t_max: int) -> bool:
    period = data.periods[t_idx]
    e_i = data.earliest_starts[i_idx]
    d_i = data.deadlines[i_idx]
    return e_i <= period <= d_i and period + data.lt_it[i_idx][t_idx] - 1 <= t_max


def _item_on_machine(data: ProcessedInstance, i_idx: int, m_idx: int) -> bool:
    return data.b_im[i_idx][m_idx] == 1


def _batch_span_values(
    data: ProcessedInstance, u_idx: int, m_idx: int, t_idx: int, t_max: int
) -> list[int]:
    """给定配置、设备和开工期，可能出现的批次最大固化跨期。"""
    return sorted({
        data.lt_it[i_idx][t_idx]
        for i_idx in data.items_by_config[u_idx]
        if _item_on_machine(data, i_idx, m_idx)
        and _item_start_feasible(data, i_idx, t_idx, t_max)
    })


def build_im_model(
    data: ProcessedInstance,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    fixed_y: dict[tuple[int, int, int], int] | None = None,
    linear_relaxation: bool = False,
    verbose: bool = False,
) -> tuple[gp.Model, dict[str, Any]]:
    """构建 IM。fixed_y 键为 (u_idx, m_idx, t_idx)，值为 0/1。"""
    model = gp.Model("IM")
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    T = data.periods
    num_t = len(T)
    num_m = len(data.machines)
    num_u = len(data.configs)
    num_i = len(data.cured_items)
    bar_m = _big_m(data)

    int_type = GRB.CONTINUOUS if linear_relaxation else GRB.INTEGER
    bin_type = GRB.CONTINUOUS if linear_relaxation else GRB.BINARY
    bin_ub = 1.0

    S_plus = model.addVars(num_i, num_t, vtype=int_type, lb=0, ub=1, name="S_plus")
    S_minus = model.addVars(num_i, num_t, vtype=int_type, lb=0, ub=1, name="S_minus")
    R = model.addVars(num_i, vtype=bin_type, lb=0, ub=bin_ub, name="R")
    X = model.addVars(num_i, num_m, num_t, vtype=bin_type, lb=0, ub=bin_ub, name="X")
    Y = model.addVars(num_u, num_m, num_t, vtype=bin_type, lb=0, ub=bin_ub, name="Y")
    # L is a derived batch-span variable.  It must be relaxed together with
    # the remaining decision variables when constructing LIM; otherwise the
    # purported LP relaxation is still a mixed-integer program.
    Lspan = model.addVars(num_u, num_m, num_t, vtype=int_type, lb=0, ub=num_t, name="L")

    # H 选择实际批次跨度；Z 将工件开工与所选跨度线性化。
    # 因此同一批内每件工件均在 t + L - 1 时共同进入库存。
    span_values = {
        (u_idx, m_idx, t_idx): _batch_span_values(data, u_idx, m_idx, t_idx, T[-1])
        for u_idx in range(num_u)
        for m_idx in range(num_m)
        for t_idx in range(num_t)
    }
    h_keys = [
        (u_idx, m_idx, t_idx, ell)
        for (u_idx, m_idx, t_idx), values in span_values.items()
        for ell in values
    ]
    H = model.addVars(h_keys, vtype=bin_type, lb=0, ub=bin_ub, name="H")
    z_keys = [
        (i_idx, m_idx, t_idx, ell)
        for i_idx in range(num_i)
        for m_idx in range(num_m)
        if _item_on_machine(data, i_idx, m_idx)
        for t_idx in range(num_t)
        if _item_start_feasible(data, i_idx, t_idx, T[-1])
        for ell in span_values[(data.config_of_item[i_idx], m_idx, t_idx)]
        if ell >= data.lt_it[i_idx][t_idx]
    ]
    Z = model.addVars(z_keys, vtype=bin_type, lb=0, ub=bin_ub, name="Z")

    if fixed_y is not None:
        for (u_idx, m_idx, t_idx), val in fixed_y.items():
            Y[u_idx, m_idx, t_idx].lb = val
            Y[u_idx, m_idx, t_idx].ub = val

    holding = gp.quicksum(data.h_c[i] * S_plus[i, t] for i in range(num_i) for t in range(num_t))
    backorder = gp.quicksum(data.bc_j[i] * S_minus[i, t] for i in range(num_i) for t in range(num_t))
    production = gp.quicksum(
        data.p_cum[m_idx][u_idx] * Y[u_idx, m_idx, t_idx]
        for u_idx in range(num_u)
        for m_idx in range(num_m)
        for t_idx in range(num_t)
    )
    scrap = gp.quicksum(data.sc_i[i] * R[i] for i in range(num_i))
    model.setObjective(holding + backorder + production + scrap, GRB.MINIMIZE)

    # 成品流平衡：批次内所有工件在最长固化跨度结束时共同到达。
    for i_idx in range(num_i):
        for t_idx in range(num_t):
            prev_inv = 0 if t_idx == 0 else S_plus[i_idx, t_idx - 1]
            prev_bo = 0 if t_idx == 0 else S_minus[i_idx, t_idx - 1]
            prod_in = gp.quicksum(
                Z[i_idx, m_idx, prod_t_idx, ell]
                for m_idx in range(num_m)
                if _item_on_machine(data, i_idx, m_idx)
                for prod_t_idx in range(num_t)
                if _item_start_feasible(data, i_idx, prod_t_idx, T[-1])
                for ell in span_values[(data.config_of_item[i_idx], m_idx, prod_t_idx)]
                if ell >= data.lt_it[i_idx][prod_t_idx]
                and T[prod_t_idx] + ell - 1 == T[t_idx]
            )
            dem = data.d_it[i_idx][t_idx]
            model.addConstr(
                prev_inv - prev_bo + prod_in
                == dem * (1 - R[i_idx]) + S_plus[i_idx, t_idx] - S_minus[i_idx, t_idx],
                name=f"flow_{i_idx}_{t_idx}",
            )

    # 每期至多一种新开配置
    for m_idx in range(num_m):
        for t_idx in range(num_t):
            model.addConstr(
                gp.quicksum(Y[u_idx, m_idx, t_idx] for u_idx in range(num_u)) <= 1,
                name=f"one_setup_{m_idx}_{t_idx}",
            )

    # 开始生产：有生产必有 setup，有 setup 必有生产
    for u_idx in range(num_u):
        i_list = data.items_by_config[u_idx]
        cap = max(1, len(i_list))
        for m_idx in range(num_m):
            for t_idx in range(num_t):
                x_sum = gp.quicksum(
                    X[i_idx, m_idx, t_idx]
                    for i_idx in i_list
                    if _item_on_machine(data, i_idx, m_idx)
                )
                model.addConstr(x_sum <= cap * Y[u_idx, m_idx, t_idx], name=f"x_le_y_{u_idx}_{m_idx}_{t_idx}")
                model.addConstr(Y[u_idx, m_idx, t_idx] <= x_sum, name=f"y_le_x_{u_idx}_{m_idx}_{t_idx}")

    # 单件须匹配自身配置 setup（仅可行开工点）
    for i_idx in range(num_i):
        u_idx = data.config_of_item[i_idx]
        for m_idx in range(num_m):
            if not _item_on_machine(data, i_idx, m_idx):
                continue
            for t_idx in range(num_t):
                if not _item_start_feasible(data, i_idx, t_idx, T[-1]):
                    continue
                model.addConstr(
                    X[i_idx, m_idx, t_idx] <= Y[u_idx, m_idx, t_idx],
                    name=f"x_match_y_{i_idx}_{m_idx}_{t_idx}",
                )

    # 实际跨度选择、锚点和 X--H 线性化。
    for u_idx in range(num_u):
        for m_idx in range(num_m):
            for t_idx in range(num_t):
                values = span_values[(u_idx, m_idx, t_idx)]
                model.addConstr(
                    gp.quicksum(H[u_idx, m_idx, t_idx, ell] for ell in values) == Y[u_idx, m_idx, t_idx],
                    name=f"choose_span_{u_idx}_{m_idx}_{t_idx}",
                )
                model.addConstr(
                    Lspan[u_idx, m_idx, t_idx]
                    == gp.quicksum(ell * H[u_idx, m_idx, t_idx, ell] for ell in values),
                    name=f"span_value_{u_idx}_{m_idx}_{t_idx}",
                )
                for ell in values:
                    anchors = [
                        i_idx for i_idx in data.items_by_config[u_idx]
                        if _item_on_machine(data, i_idx, m_idx)
                        and _item_start_feasible(data, i_idx, t_idx, T[-1])
                        and data.lt_it[i_idx][t_idx] == ell
                    ]
                    model.addConstr(
                        H[u_idx, m_idx, t_idx, ell]
                        <= gp.quicksum(X[i_idx, m_idx, t_idx] for i_idx in anchors),
                        name=f"span_anchor_{u_idx}_{m_idx}_{t_idx}_{ell}",
                    )

    for i_idx, m_idx, t_idx, ell in z_keys:
        u_idx = data.config_of_item[i_idx]
        model.addConstr(Z[i_idx, m_idx, t_idx, ell] <= X[i_idx, m_idx, t_idx])
        model.addConstr(Z[i_idx, m_idx, t_idx, ell] <= H[u_idx, m_idx, t_idx, ell])
        model.addConstr(
            Z[i_idx, m_idx, t_idx, ell] >= X[i_idx, m_idx, t_idx] + H[u_idx, m_idx, t_idx, ell] - 1
        )

    # 共同出罐的关键链接：已开工的工件必须随一个不短于自身固化时长的
    # 批次跨度共同完工。没有这条等式时，X 可以取 1 而所有对应 Z 取 0，
    # 从而错误地把长固化工件放入较短批次，并使流平衡与定价逻辑不一致。
    for i_idx in range(num_i):
        u_idx = data.config_of_item[i_idx]
        for m_idx in range(num_m):
            if not _item_on_machine(data, i_idx, m_idx):
                continue
            for t_idx in range(num_t):
                if not _item_start_feasible(data, i_idx, t_idx, T[-1]):
                    continue
                compatible_z = [
                    Z[i_idx, m_idx, t_idx, ell]
                    for ell in span_values[(u_idx, m_idx, t_idx)]
                    if (i_idx, m_idx, t_idx, ell) in Z
                ]
                model.addConstr(
                    X[i_idx, m_idx, t_idx] == gp.quicksum(compatible_z),
                    name=f"x_choose_span_{i_idx}_{m_idx}_{t_idx}",
                )

    # 机台不重叠（批量添加）
    model.addConstrs(
        (
            Lspan[u_idx, m_idx, t_idx]
            - bar_m * (2 - Y[u_idx, m_idx, t_idx] - Y[u2_idx, m_idx, t2_idx])
            <= T[t2_idx] - T[t_idx]
            for m_idx in range(num_m)
            for u_idx in range(num_u)
            for u2_idx in range(num_u)
            for t_idx in range(num_t)
            for t2_idx in range(num_t)
            if T[t2_idx] > T[t_idx]
        ),
        name="no_overlap",
    )

    # 容量（开工时刻装箱）
    for m_idx in range(num_m):
        for t_idx in range(num_t):
            model.addConstr(
                gp.quicksum(
                    data.v_i[i_idx] * X[i_idx, m_idx, t_idx]
                    for i_idx in data.items_by_machine[m_idx]
                )
                <= data.q_m[m_idx],
                name=f"capacity_{m_idx}_{t_idx}",
            )

    # 单件去向：进罐或报废（仅专属/允许机台）
    for i_idx in range(num_i):
        e_i = data.earliest_starts[i_idx]
        d_i = data.deadlines[i_idx]
        model.addConstr(
            gp.quicksum(
                X[i_idx, m_idx, t_idx]
                for m_idx in range(num_m)
                if _item_on_machine(data, i_idx, m_idx)
                for t_idx in range(num_t)
                if e_i <= T[t_idx] <= d_i
            )
            + R[i_idx]
            == 1,
            name=f"fate_{i_idx}",
        )

    # 时间窗、专属机台：不可行开工点直接固定 X=0
    t_max = T[-1]
    for i_idx in range(num_i):
        for m_idx in range(num_m):
            for t_idx in range(num_t):
                if not _item_on_machine(data, i_idx, m_idx) or not _item_start_feasible(
                    data, i_idx, t_idx, t_max
                ):
                    X[i_idx, m_idx, t_idx].ub = 0.0

    vars_bundle = {
        "S_plus": S_plus,
        "S_minus": S_minus,
        "R": R,
        "X": X,
        "Y": Y,
        "L": Lspan,
        "H": H,
        "Z": Z,
    }
    return model, vars_bundle


def solve_im(
    data: ProcessedInstance,
    time_limit: float | None = DEFAULT_TIME_LIMIT,
    mip_gap: float | None = DEFAULT_MIP_GAP,
    fixed_y: dict[tuple[int, int, int], int] | None = None,
    method: str = "im_gurobi",
    verbose: bool = False,
) -> SolutionResult:
    start = time.perf_counter()
    model, vars_bundle = build_im_model(
        data,
        time_limit=time_limit,
        mip_gap=mip_gap,
        fixed_y=fixed_y,
        verbose=verbose,
    )
    model.optimize()
    runtime = time.perf_counter() - start

    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INTERRUPTED: "interrupted",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
    }
    status = status_map.get(model.Status, f"status_{model.Status}")

    if model.SolCount == 0:
        result = SolutionResult(
            method=method,
            instance_name=data.raw.name,
            status=status,
            objective=None,
            lower_bound=model.ObjBound if model.Status != GRB.INFEASIBLE else None,
            mip_gap=None,
            runtime_sec=runtime,
            extra={"mip_node_count": model.NodeCount},
        )
        model.dispose()
        return result

    obj = model.ObjVal
    lb = model.ObjBound
    gap = None
    if obj != 0:
        gap = abs(obj - lb) / max(abs(obj), CHECKER_TOL["objective_divisor"])

    num_t = len(data.periods)
    num_i = len(data.cured_items)

    holding_cost = sum(
        data.h_c[i] * vars_bundle["S_plus"][i, t].X for i in range(num_i) for t in range(num_t)
    )
    backorder_cost = sum(
        data.bc_j[i] * vars_bundle["S_minus"][i, t].X for i in range(num_i) for t in range(num_t)
    )
    production_cost = sum(
        data.p_cum[m_idx][u_idx] * vars_bundle["Y"][u_idx, m_idx, t_idx].X
        for u_idx in range(len(data.configs))
        for m_idx in range(len(data.machines))
        for t_idx in range(num_t)
    )
    scrap_cost = sum(data.sc_i[i] * vars_bundle["R"][i].X for i in range(num_i))
    scrap_qty = {data.cured_items[i]: vars_bundle["R"][i].X for i in range(num_i)}

    inventory = {
        data.cured_items[i]: {data.periods[t]: vars_bundle["S_plus"][i, t].X for t in range(num_t)}
        for i in range(num_i)
    }
    backorder = {
        data.cured_items[i]: {data.periods[t]: vars_bundle["S_minus"][i, t].X for t in range(num_t)}
        for i in range(num_i)
    }

    packing = {
        data.machines[m_idx]: {
            data.cured_items[i_idx]: {
                data.periods[t_idx]: vars_bundle["X"][i_idx, m_idx, t_idx].X for t_idx in range(num_t)
            }
            for i_idx in range(num_i)
        }
        for m_idx in range(len(data.machines))
    }
    setup = {
        data.machines[m_idx]: {
            data.configs[u_idx]: {
                data.periods[t_idx]: int(round(vars_bundle["Y"][u_idx, m_idx, t_idx].X))
                for t_idx in range(num_t)
            }
            for u_idx in range(len(data.configs))
        }
        for m_idx in range(len(data.machines))
    }
    span = {
        data.machines[m_idx]: {
            data.configs[u_idx]: {
                data.periods[t_idx]: int(round(vars_bundle["L"][u_idx, m_idx, t_idx].X))
                for t_idx in range(num_t)
            }
            for u_idx in range(len(data.configs))
        }
        for m_idx in range(len(data.machines))
    }

    result = SolutionResult(
        method=method,
        instance_name=data.raw.name,
        status=status,
        objective=obj,
        lower_bound=lb,
        mip_gap=gap,
        runtime_sec=runtime,
        holding_cost=holding_cost,
        backorder_cost=backorder_cost,
        production_cost=production_cost,
        inventory=inventory,
        backorder=backorder,
        assembly={},
        packing=packing,
        setup=setup,
        extra={
            "instance_seed": data.raw.seed,
            "set_type": data.raw.set_type,
            "num_periods": data.raw.num_periods,
            "scrap_cost": scrap_cost,
            "scrap": scrap_qty,
            "total_scrap_qty": sum(scrap_qty.values()),
            "span": span,
            "mip_node_count": model.NodeCount,
        },
    )
    model.dispose()
    return result


def solve_lim(
    data: ProcessedInstance,
    time_limit: float | None = DEFAULT_LIM_TIME_LIMIT,
    verbose: bool = False,
) -> SolutionResult:
    start = time.perf_counter()
    model, _ = build_im_model(data, time_limit=time_limit, linear_relaxation=True, verbose=verbose)
    # A LIM must contain no integer or binary variables.  Keep this explicit
    # check so a future formulation change cannot silently turn the reported
    # LP relaxation into a MIP again.
    model.update()
    if any(var.VType != GRB.CONTINUOUS for var in model.getVars()):
        model.dispose()
        raise RuntimeError("solve_lim built a model containing non-continuous variables")
    model.optimize()
    runtime = time.perf_counter() - start
    # ObjBound remains a valid minimisation lower bound when a time limit is reached.
    obj = model.ObjVal if model.Status == GRB.OPTIMAL else None
    lower_bound = (
        float(model.ObjBound)
        if model.Status not in {GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED}
        else None
    )
    result = SolutionResult(
        method="lim",
        instance_name=data.raw.name,
        status="optimal" if model.Status == GRB.OPTIMAL else f"status_{model.Status}",
        objective=obj,
        lower_bound=lower_bound,
        mip_gap=0.0 if model.Status == GRB.OPTIMAL else None,
        runtime_sec=runtime,
    )
    model.dispose()
    return result

