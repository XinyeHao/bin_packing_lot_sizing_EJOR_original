"""Gurobi 精确求解规范 §3.3 单机定价子问题（lt_it 阶跃、动态 L、0/1 合批）。"""

from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB

from common.data_models import ProcessedInstance
from configuration import CHECKER_TOL, DEFAULT_CG_INIT_TIME_LIMIT, DEFAULT_PROGRESS_INTERVAL_SEC
from model_and_algo.subproblem_common import SubproblemResult
from model_and_algo.im_model import _big_m, _item_start_feasible
from model_and_algo.progress_monitor import heartbeat_while

def solve_subproblem_im_gurobi(
    data: ProcessedInstance,
    machine_idx: int,
    alpha: list[list[float]],
    beta: float,
    rho: list[float] | None = None,
    time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
    fixed_y: dict[tuple[int, int], int] | None = None,
    progress_interval_sec: float = DEFAULT_PROGRESS_INTERVAL_SEC,
) -> SubproblemResult:
    """规范 IM 单机定价 MIP；用于小算例对照验证。"""
    machine = data.machines[machine_idx]
    build_start = time.perf_counter()
    model = gp.Model("pricing_im")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    num_t = len(data.periods)
    num_u = len(data.configs)
    num_i = len(data.cured_items)
    t_max = data.periods[-1]
    bar_m = _big_m(data)
    q = data.q_m[machine_idx]
    p_cum = data.p_cum[machine_idx]
    rho = rho or [0.0] * num_i
    local_items = data.items_by_machine[machine_idx]

    X = model.addVars(num_i, num_t, vtype=GRB.BINARY, name="X")
    Y = model.addVars(num_u, num_t, vtype=GRB.BINARY, name="Y")
    Lspan = model.addVars(num_u, num_t, vtype=GRB.INTEGER, lb=0, ub=num_t, name="L")

    obj = gp.quicksum(p_cum[u_idx] * Y[u_idx, t_idx] for u_idx in range(num_u) for t_idx in range(num_t))
    for i_idx in range(num_i):
        if data.b_im[i_idx][machine_idx] == 0:
            for t_idx in range(num_t):
                X[i_idx, t_idx].ub = 0.0
            continue
        for t_idx in range(num_t):
            if not _item_start_feasible(data, i_idx, t_idx, t_max):
                continue
            u_idx = data.config_of_item[i_idx]
            span = data.lt_it[i_idx][t_idx]
            comp_idx = t_idx + span - 1
            if comp_idx < num_t:
                obj -= alpha[i_idx][comp_idx] * X[i_idx, t_idx]
            obj -= rho[i_idx] * X[i_idx, t_idx]
    obj -= beta
    model.setObjective(obj, GRB.MINIMIZE)

    for t_idx in range(num_t):
        model.addConstr(gp.quicksum(Y[u_idx, t_idx] for u_idx in range(num_u)) <= 1)

    for u_idx in range(num_u):
        i_list = [i for i in data.items_by_config[u_idx] if data.b_im[i][machine_idx]]
        for t_idx in range(num_t):
            x_sum = gp.quicksum(X[i_idx, t_idx] for i_idx in i_list)
            model.addConstr(x_sum <= max(1, len(i_list)) * Y[u_idx, t_idx])
            model.addConstr(Y[u_idx, t_idx] <= x_sum)

    for i_idx in local_items:
        u_idx = data.config_of_item[i_idx]
        for t_idx in range(num_t):
            if not _item_start_feasible(data, i_idx, t_idx, t_max):
                model.addConstr(X[i_idx, t_idx] == 0)
                continue
            model.addConstr(X[i_idx, t_idx] <= Y[u_idx, t_idx])

    for u_idx in range(num_u):
        for i_idx in data.items_by_config[u_idx]:
            if data.b_im[i_idx][machine_idx] == 0:
                continue
            for t_idx in range(num_t):
                if not _item_start_feasible(data, i_idx, t_idx, t_max):
                    continue
                model.addConstr(
                    Lspan[u_idx, t_idx] >= data.lt_it[i_idx][t_idx] * X[i_idx, t_idx],
                )
        for t_idx in range(num_t):
            model.addConstr(Lspan[u_idx, t_idx] <= bar_m * Y[u_idx, t_idx])

    for u_idx in range(num_u):
        for u2_idx in range(num_u):
            for t_idx in range(num_t):
                for t2_idx in range(num_t):
                    if data.periods[t2_idx] <= data.periods[t_idx]:
                        continue
                    model.addConstr(
                        Lspan[u_idx, t_idx]
                        - bar_m * (2 - Y[u_idx, t_idx] - Y[u2_idx, t2_idx])
                        <= data.periods[t2_idx] - data.periods[t_idx],
                    )

    for t_idx in range(num_t):
        model.addConstr(
            gp.quicksum(data.v_i[i_idx] * X[i_idx, t_idx] for i_idx in local_items) <= q
        )

    for i_idx in local_items:
        model.addConstr(gp.quicksum(X[i_idx, t_idx] for t_idx in range(num_t)) <= 1)

    for i_idx in range(num_i):
        if i_idx not in local_items:
            for t_idx in range(num_t):
                X[i_idx, t_idx].ub = 0.0

    if fixed_y:
        for (u_idx, t_idx), val in fixed_y.items():
            Y[u_idx, t_idx].lb = val
            Y[u_idx, t_idx].ub = val

    build_sec = time.perf_counter() - build_start
    opt_start = time.perf_counter()
    state = {"incumbent": None, "gap": None, "nodes": 0}

    def _gurobi_status() -> str:
        parts = [f"build={build_sec:.1f}s", f"opt={time.perf_counter() - opt_start:.1f}s"]
        if state["incumbent"] is not None:
            parts.append(f"incumbent={state['incumbent']:.4f}")
        if state["gap"] is not None:
            parts.append(f"gap={state['gap']:.2%}")
        if state["nodes"]:
            parts.append(f"nodes={state['nodes']}")
        return "  ".join(parts)

    def _callback(model: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIP:
            try:
                objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                state["incumbent"] = objbst
                if abs(objbst) > CHECKER_TOL["objective_divisor"]:
                    state["gap"] = abs(objbst - objbnd) / abs(objbst)
                state["nodes"] = int(model.cbGet(GRB.Callback.MIP_NODCNT))
            except gp.GurobiError:
                pass

    with heartbeat_while(
        f"IM-pricing {machine}",
        _gurobi_status,
        interval_sec=progress_interval_sec,
        enabled=True,
    ):
        model.optimize(_callback)
    x: dict[tuple[int, int], int] = {}
    y: dict[tuple[int, int], int] = {}
    production_cost = 0.0
    if model.SolCount > 0:
        for i_idx in range(num_i):
            for t_idx in range(num_t):
                if X[i_idx, t_idx].X > 0.5:
                    x[(i_idx, t_idx)] = 1
        for u_idx in range(num_u):
            for t_idx in range(num_t):
                if Y[u_idx, t_idx].X > 0.5:
                    y[(u_idx, t_idx)] = 1
                    production_cost += p_cum[u_idx]

    rc = model.ObjVal if model.SolCount > 0 else float("inf")
    if model.Status == GRB.INFEASIBLE:
        rc = float("inf")
    elif model.SolCount == 0 and model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        rc = float("inf")
    return SubproblemResult(reduced_cost=rc, x=x, y=y, production_cost=production_cost)
