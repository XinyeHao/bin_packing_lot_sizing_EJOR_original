"""集成模型 SIM（IM）的 Gurobi 实现。"""

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
from preprocessor.preprocess import max_tray_count


def _setup_window_start(t: int, l_u: int) -> int:
    return max(t - l_u + 1, 1)


def build_im_model(
    data: ProcessedInstance,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    fixed_y: dict[tuple[int, int, int], int] | None = None,
    linear_relaxation: bool = False,
    verbose: bool = False,
) -> tuple[gp.Model, dict[str, Any]]:
    """构建 SIM 模型。fixed_y 的键为 (u_idx, m_idx, t_idx)，值为 0/1。"""
    model = gp.Model("IM_SIM")
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
    num_j = len(data.end_items)
    num_all = len(data.all_items)

    int_type = GRB.CONTINUOUS if linear_relaxation else GRB.INTEGER
    bin_type = GRB.CONTINUOUS if linear_relaxation else GRB.BINARY
    bin_ub = 1.0

    S_plus = model.addVars(num_all, num_t, vtype=int_type, lb=0, name="S_plus")
    S_minus = model.addVars(num_j, num_t, vtype=int_type, lb=0, name="S_minus")
    A = model.addVars(num_j, num_t, vtype=int_type, lb=0, name="A")
    X = model.addVars(num_i, num_m, num_t, vtype=int_type, lb=0, name="X")
    Y = model.addVars(num_u, num_m, num_t, vtype=bin_type, lb=0, ub=bin_ub, name="Y")
    Z = model.addVars(num_u, num_m, num_t, vtype=bin_type, lb=0, ub=bin_ub, name="Z")

    if fixed_y is not None:
        for (u_idx, m_idx, t_idx), val in fixed_y.items():
            Y[u_idx, m_idx, t_idx].lb = val
            Y[u_idx, m_idx, t_idx].ub = val

    holding = gp.quicksum(data.h_c[item_idx] * S_plus[item_idx, t_idx] for item_idx in range(num_all) for t_idx in range(num_t))
    backorder = gp.quicksum(data.bc_j[j_idx] * S_minus[j_idx, t_idx] for j_idx in range(num_j) for t_idx in range(num_t))
    production = gp.quicksum(
        data.p_cum[m_idx][u_idx] * Y[u_idx, m_idx, t_idx]
        for u_idx in range(num_u)
        for m_idx in range(num_m)
        for t_idx in range(num_t)
    )
    model.setObjective(holding + backorder + production, GRB.MINIMIZE)

    cured_offset = 0
    end_offset = num_i

    for j_idx in range(num_j):
        item_idx = end_offset + j_idx
        for t_idx in range(num_t):
            prev_inv = 0 if t_idx == 0 else S_plus[item_idx, t_idx - 1]
            prev_bo = 0 if t_idx == 0 else S_minus[j_idx, t_idx - 1]
            model.addConstr(
                prev_inv - prev_bo + A[j_idx, t_idx]
                == data.d_jt[j_idx][t_idx] + S_plus[item_idx, t_idx] - S_minus[j_idx, t_idx],
                name=f"flow_end_{j_idx}_{t_idx}",
            )

    for i_idx in range(num_i):
        for t_idx in range(num_t):
            prev_inv = 0 if t_idx == 0 else S_plus[i_idx, t_idx - 1]
            prod_in = gp.quicksum(
                X[i_idx, m_idx, prod_t]
                for m_idx in range(num_m)
                for prod_t in [t_idx - int(data.l_ti[i_idx])]
                if prod_t >= 0
            )
            demand = gp.quicksum(data.r_ij[i_idx][j_idx] * A[j_idx, t_idx] for j_idx in range(num_j))
            model.addConstr(prev_inv + prod_in == demand + S_plus[i_idx, t_idx], name=f"flow_cured_{i_idx}_{t_idx}")

    for m_idx in range(num_m):
        for t_idx in range(num_t):
            model.addConstr(gp.quicksum(Z[u_idx, m_idx, t_idx] for u_idx in range(num_u)) <= 1, name=f"one_cfg_{m_idx}_{t_idx}")

    for u_idx in range(num_u):
        l_u = data.l_u[u_idx]
        for m_idx in range(num_m):
            for t_idx in range(num_t):
                t_start = _setup_window_start(T[t_idx], l_u) - 1
                model.addConstr(
                    gp.quicksum(Y[u_idx, m_idx, tp] for tp in range(t_start, t_idx + 1)) == Z[u_idx, m_idx, t_idx],
                    name=f"link_yz_{u_idx}_{m_idx}_{t_idx}",
                )

    for i_idx in range(num_i):
        cap_count = [max_tray_count(data.q_m[m_idx], data.v_i[i_idx]) for m_idx in range(num_m)]
        for m_idx in range(num_m):
            for t_idx in range(num_t):
                cfg_match = gp.quicksum(data.b_iu[i_idx][u_idx] * Y[u_idx, m_idx, t_idx] for u_idx in range(num_u))
                model.addConstr(X[i_idx, m_idx, t_idx] <= cap_count[m_idx] * cfg_match, name=f"cfg_pack_{i_idx}_{m_idx}_{t_idx}")

    for m_idx in range(num_m):
        for t_idx in range(num_t):
            model.addConstr(
                gp.quicksum(data.v_i[i_idx] * X[i_idx, m_idx, t_idx] for i_idx in range(num_i)) <= data.q_m[m_idx],
                name=f"capacity_{m_idx}_{t_idx}",
            )

    vars_bundle = {
        "S_plus": S_plus,
        "S_minus": S_minus,
        "A": A,
        "X": X,
        "Y": Y,
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
    """求解 SIM 并返回结构化结果。"""
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
        return SolutionResult(
            method=method,
            instance_name=data.raw.name,
            status=status,
            objective=None,
            lower_bound=model.ObjBound if model.Status != GRB.INFEASIBLE else None,
            mip_gap=None,
            runtime_sec=runtime,
        )

    obj = model.ObjVal
    lb = model.ObjBound
    gap = None
    if obj != 0:
        gap = abs(obj - lb) / max(abs(obj), CHECKER_TOL["objective_divisor"])

    num_t = len(data.periods)
    num_i = len(data.cured_items)
    num_j = len(data.end_items)
    num_all = len(data.all_items)
    end_offset = num_i

    holding_cost = sum(
        data.h_c[item_idx] * vars_bundle["S_plus"][item_idx, t_idx].X
        for item_idx in range(num_all)
        for t_idx in range(num_t)
    )
    backorder_cost = sum(
        data.bc_j[j_idx] * vars_bundle["S_minus"][j_idx, t_idx].X
        for j_idx in range(num_j)
        for t_idx in range(num_t)
    )
    production_cost = sum(
        data.p_cum[m_idx][u_idx] * vars_bundle["Y"][u_idx, m_idx, t_idx].X
        for u_idx in range(len(data.configs))
        for m_idx in range(len(data.machines))
        for t_idx in range(num_t)
    )

    inventory = {
        data.all_items[item_idx]: {data.periods[t_idx]: vars_bundle["S_plus"][item_idx, t_idx].X for t_idx in range(num_t)}
        for item_idx in range(num_all)
    }
    backorder = {
        data.end_items[j_idx]: {data.periods[t_idx]: vars_bundle["S_minus"][j_idx, t_idx].X for t_idx in range(num_t)}
        for j_idx in range(num_j)
    }
    assembly = {
        data.end_items[j_idx]: {data.periods[t_idx]: vars_bundle["A"][j_idx, t_idx].X for t_idx in range(num_t)}
        for j_idx in range(num_j)
    }
    packing = {
        data.machines[m_idx]: {
            data.cured_items[i_idx]: {data.periods[t_idx]: vars_bundle["X"][i_idx, m_idx, t_idx].X for t_idx in range(num_t)}
            for i_idx in range(num_i)
        }
        for m_idx in range(len(data.machines))
    }
    setup = {
        data.machines[m_idx]: {
            data.configs[u_idx]: {data.periods[t_idx]: int(round(vars_bundle["Y"][u_idx, m_idx, t_idx].X)) for t_idx in range(num_t)}
            for u_idx in range(len(data.configs))
        }
        for m_idx in range(len(data.machines))
    }

    return SolutionResult(
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
        assembly=assembly,
        packing=packing,
        setup=setup,
        extra={
            "instance_seed": data.raw.seed,
            "set_type": data.raw.set_type,
            "num_periods": data.raw.num_periods,
        },
    )


def solve_lim(
    data: ProcessedInstance,
    time_limit: float | None = DEFAULT_LIM_TIME_LIMIT,
    verbose: bool = False,
) -> SolutionResult:
    """求解线性松弛集成模型 LIM，提供下界 LBLIM。"""
    start = time.perf_counter()
    model, _ = build_im_model(data, time_limit=time_limit, linear_relaxation=True, verbose=verbose)
    model.optimize()
    runtime = time.perf_counter() - start
    obj = model.ObjVal if model.Status == GRB.OPTIMAL else None
    return SolutionResult(
        method="lim",
        instance_name=data.raw.name,
        status="optimal" if model.Status == GRB.OPTIMAL else f"status_{model.Status}",
        objective=obj,
        lower_bound=obj,
        mip_gap=0.0,
        runtime_sec=runtime,
    )
