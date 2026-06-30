"""Gurobi 求解子问题以生成初始列。"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from common.data_models import ProcessedInstance
from configuration import DEFAULT_CG_INIT_TIME_LIMIT
from model_and_algo.bldp import SubproblemResult
from preprocessor.preprocess import max_tray_count


def solve_subproblem_gurobi(
    data: ProcessedInstance,
    machine_idx: int,
    alpha: list[list[float]],
    beta: float,
    rho: list[float] | None = None,
    time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
) -> SubproblemResult:
    """用 Gurobi 精确求解单台热压罐子问题（用于初始列）。"""
    model = gp.Model("subproblem")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    num_t = len(data.periods)
    num_u = len(data.configs)
    num_i = len(data.cured_items)
    q = data.q_m[machine_idx]
    p_cum = data.p_cum[machine_idx]
    rho = rho or [0.0] * num_i

    X = model.addVars(num_i, num_t, vtype=GRB.INTEGER, lb=0, name="X")
    Y = model.addVars(num_u, num_t, vtype=GRB.BINARY, name="Y")
    Z = model.addVars(num_u, num_t, vtype=GRB.BINARY, name="Z")

    obj = gp.quicksum(p_cum[u_idx] * Y[u_idx, t_idx] for u_idx in range(num_u) for t_idx in range(num_t))
    for i_idx in range(num_i):
        dl = data.deadlines[i_idx] if hasattr(data, "deadlines") and data.deadlines else num_t
        for t_idx in range(num_t):
            if (t_idx + 1) > dl:
                continue
            arrival = t_idx + int(data.l_ti[i_idx])
            if arrival < num_t:
                obj -= alpha[i_idx][arrival] * X[i_idx, t_idx]
            obj -= rho[i_idx] * X[i_idx, t_idx]
    obj -= beta
    model.setObjective(obj, GRB.MINIMIZE)

    for t_idx in range(num_t):
        model.addConstr(gp.quicksum(Z[u_idx, t_idx] for u_idx in range(num_u)) <= 1)

    for u_idx in range(num_u):
        l_u = data.l_u[u_idx]
        for t_idx in range(num_t):
            t_start = max(t_idx - l_u + 1, 0)
            model.addConstr(gp.quicksum(Y[u_idx, tp] for tp in range(t_start, t_idx + 1)) == Z[u_idx, t_idx])

    for i_idx in range(num_i):
        cap = max_tray_count(q, data.v_i[i_idx])
        for t_idx in range(num_t):
            model.addConstr(
                X[i_idx, t_idx]
                <= cap * gp.quicksum(data.b_iu[i_idx][u_idx] * Y[u_idx, t_idx] for u_idx in range(num_u))
            )

    for t_idx in range(num_t):
        model.addConstr(gp.quicksum(data.v_i[i_idx] * X[i_idx, t_idx] for i_idx in range(num_i)) <= q)

    if hasattr(data, "deadlines") and data.deadlines:
        for i_idx in range(num_i):
            dl = data.deadlines[i_idx]
            for t_idx in range(num_t):
                if (t_idx + 1) > dl:
                    model.addConstr(X[i_idx, t_idx] == 0, name=f"dl_{i_idx}_{t_idx}")

    model.optimize()

    x: dict[tuple[int, int], int] = {}
    y: dict[tuple[int, int], int] = {}
    production_cost = 0.0
    if model.SolCount > 0:
        for i_idx in range(num_i):
            for t_idx in range(num_t):
                qty = int(round(X[i_idx, t_idx].X))
                if qty > 0:
                    x[(i_idx, t_idx)] = qty
        for u_idx in range(num_u):
            for t_idx in range(num_t):
                if Y[u_idx, t_idx].X > 0.5:
                    y[(u_idx, t_idx)] = 1
                    production_cost += p_cum[u_idx]

    rc = model.ObjVal if model.SolCount > 0 else 0.0
    return SubproblemResult(reduced_cost=rc, x=x, y=y, production_cost=production_cost)
