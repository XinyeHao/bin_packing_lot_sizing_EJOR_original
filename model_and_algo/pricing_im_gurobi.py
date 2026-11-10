"""Gurobi 精确单机定价：批次按最长固化时间共同出罐。"""

from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB

from common.data_models import ProcessedInstance
from configuration import CHECKER_TOL, DEFAULT_CG_INIT_TIME_LIMIT, DEFAULT_PROGRESS_INTERVAL_SEC
from model_and_algo.subproblem_common import SubproblemResult
from model_and_algo.im_model import _big_m, _item_start_feasible, _batch_span_values
from model_and_algo.progress_monitor import heartbeat_while

def solve_subproblem_im_gurobi(
    data: ProcessedInstance,
    machine_idx: int,
    alpha: list[list[float]],
    beta: float,
    rho: list[float] | None = None,
    time_limit: float | None = DEFAULT_CG_INIT_TIME_LIMIT,
    fixed_y: dict[tuple[int, int], int] | None = None,
    progress_interval_sec: float = DEFAULT_PROGRESS_INTERVAL_SEC,
    setup_cost_delta: list[list[float]] | None = None,
) -> SubproblemResult:
    """规范 IM 单机定价 MIP；用于小算例对照验证。"""
    machine = data.machines[machine_idx]
    build_start = time.perf_counter()
    model = gp.Model("pricing_im")
    model.Params.OutputFlag = 0
    # A root-DW lower-bound experiment needs an exact pricing certificate.
    # ``None`` therefore means no artificial per-pricing-MIP limit; callers
    # that need a computational safeguard may still pass a positive value.
    if time_limit is not None:
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
    span_values = {
        (u_idx, t_idx): _batch_span_values(data, u_idx, machine_idx, t_idx, t_max)
        for u_idx in range(num_u)
        for t_idx in range(num_t)
    }
    h_keys = [
        (u_idx, t_idx, ell)
        for (u_idx, t_idx), values in span_values.items()
        for ell in values
    ]
    H = model.addVars(h_keys, vtype=GRB.BINARY, name="H")
    z_keys = [
        (i_idx, t_idx, ell)
        for i_idx in local_items
        for t_idx in range(num_t)
        if _item_start_feasible(data, i_idx, t_idx, t_max)
        for ell in span_values[(data.config_of_item[i_idx], t_idx)]
        if ell >= data.lt_it[i_idx][t_idx]
    ]
    Z = model.addVars(z_keys, vtype=GRB.BINARY, name="Z")

    obj = gp.quicksum(p_cum[u_idx] * Y[u_idx, t_idx] for u_idx in range(num_u) for t_idx in range(num_t))
    if setup_cost_delta is not None:
        obj = obj + gp.quicksum(
            setup_cost_delta[u_idx][t_idx] * Y[u_idx, t_idx]
            for u_idx in range(num_u)
            for t_idx in range(num_t)
            if abs(setup_cost_delta[u_idx][t_idx]) > 1e-12
        )
    for i_idx in range(num_i):
        if data.b_im[i_idx][machine_idx] == 0:
            for t_idx in range(num_t):
                X[i_idx, t_idx].ub = 0.0
            continue
        for t_idx in range(num_t):
            if not _item_start_feasible(data, i_idx, t_idx, t_max):
                continue
            u_idx = data.config_of_item[i_idx]
            obj -= rho[i_idx] * X[i_idx, t_idx]
    obj -= gp.quicksum(
        alpha[i_idx][t_idx + ell - 1] * Z[i_idx, t_idx, ell]
        for i_idx, t_idx, ell in z_keys
        if t_idx + ell - 1 < num_t
    )
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
        for t_idx in range(num_t):
            values = span_values[(u_idx, t_idx)]
            model.addConstr(gp.quicksum(H[u_idx, t_idx, ell] for ell in values) == Y[u_idx, t_idx])
            model.addConstr(Lspan[u_idx, t_idx] == gp.quicksum(ell * H[u_idx, t_idx, ell] for ell in values))
            for ell in values:
                anchors = [
                    i_idx for i_idx in data.items_by_config[u_idx]
                    if data.b_im[i_idx][machine_idx]
                    and _item_start_feasible(data, i_idx, t_idx, t_max)
                    and data.lt_it[i_idx][t_idx] == ell
                ]
                model.addConstr(H[u_idx, t_idx, ell] <= gp.quicksum(X[i_idx, t_idx] for i_idx in anchors))

    for i_idx, t_idx, ell in z_keys:
        u_idx = data.config_of_item[i_idx]
        model.addConstr(Z[i_idx, t_idx, ell] <= X[i_idx, t_idx])
        model.addConstr(Z[i_idx, t_idx, ell] <= H[u_idx, t_idx, ell])
        model.addConstr(Z[i_idx, t_idx, ell] >= X[i_idx, t_idx] + H[u_idx, t_idx, ell] - 1)

    # If an item is loaded, it must be paired with the selected batch span.
    # This prevents a long-curing item from being assigned X=1 while the
    # pricing objective evaluates the batch with a shorter span.
    for i_idx in local_items:
        u_idx = data.config_of_item[i_idx]
        for t_idx in range(num_t):
            if not _item_start_feasible(data, i_idx, t_idx, t_max):
                continue
            compatible_z = [
                Z[i_idx, t_idx, ell]
                for ell in span_values[(u_idx, t_idx)]
                if (i_idx, t_idx, ell) in Z
            ]
            model.addConstr(
                X[i_idx, t_idx] == gp.quicksum(compatible_z),
                name=f"x_choose_span_{i_idx}_{t_idx}",
            )

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
        enabled=progress_interval_sec > 0,
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
    return SubproblemResult(
        reduced_cost=rc,
        x=x,
        y=y,
        production_cost=production_cost,
        is_optimal=(model.Status == GRB.OPTIMAL),
    )
