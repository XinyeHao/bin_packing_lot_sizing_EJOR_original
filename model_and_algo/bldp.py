"""子问题定价：无界背包 + 路由（BLDP）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from common.data_models import ProcessedInstance, ScheduleColumn
from configuration import CHECKER_TOL


@dataclass
class SubproblemResult:
    reduced_cost: float
    x: dict[tuple[int, int], int]
    y: dict[tuple[int, int], int]
    production_cost: float


def _scale_lengths(values: list[float], q: float) -> tuple[int, list[int]]:
    scale = 100
    capacity = max(1, int(math.floor(q * scale)))
    weights = [max(1, int(math.ceil(v * scale))) for v in values]
    return capacity, weights


def _solve_ukp(
    item_indices: list[int],
    delta: list[float],
    pcu: float,
    q: float,
    v_i: list[float],
) -> tuple[float, dict[int, int]]:
    """求解 min pcu - sum delta_i X_i, s.t. sum v_i X_i <= q。"""
    if not item_indices:
        return pcu, {}

    capacity, weights = _scale_lengths([v_i[i] for i in item_indices], q)
    dp = [0.0] * (capacity + 1)
    parent: list[tuple[int, int] | None] = [None] * (capacity + 1)

    for idx, item in enumerate(item_indices):
        weight = weights[idx]
        value = delta[item]
        for w in range(weight, capacity + 1):
            candidate = dp[w - weight] + value
            if candidate > dp[w] + CHECKER_TOL["bldp_dp"]:
                dp[w] = candidate
                parent[w] = (w - weight, idx)

    best_w = max(range(capacity + 1), key=lambda w: dp[w])
    counts: dict[int, int] = {}
    w = best_w
    while parent[w] is not None:
        prev_w, idx = parent[w]
        item = item_indices[idx]
        counts[item] = counts.get(item, 0) + 1
        w = prev_w

    eta = pcu - dp[best_w]
    return eta, counts


def solve_subproblem_bldp(
    data: ProcessedInstance,
    machine_idx: int,
    alpha: list[list[float]],
    beta: float,
    rho: list[float] | None = None,
) -> SubproblemResult:
    """对单台热压罐求解定价子问题。"""
    num_t = len(data.periods)
    num_u = len(data.configs)
    num_i = len(data.cured_items)
    q = data.q_m[machine_idx]
    p_cum = data.p_cum[machine_idx]
    rho = rho or [0.0] * num_i

    eta: list[list[float]] = [[math.inf] * num_t for _ in range(num_u)]
    pack: list[list[dict[int, int]]] = [[{} for _ in range(num_t)] for _ in range(num_u)]

    for u_idx in range(num_u):
        items = data.items_by_config[u_idx]
        pcu = p_cum[u_idx]
        for t_idx in range(num_t):
            delta = [0.0] * num_i
            for i_idx in items:
                if hasattr(data, "deadlines") and (t_idx + 1) > data.deadlines[i_idx]:
                    continue
                arrival = t_idx + int(data.l_ti[i_idx])
                if arrival < num_t:
                    delta[i_idx] = alpha[i_idx][arrival] + rho[i_idx]
            eta[u_idx][t_idx], pack[u_idx][t_idx] = _solve_ukp(items, delta, pcu, q, data.v_i)

    phi = [0.0] * (num_t + 1)
    phi[0] = -beta
    prev = [-1] * (num_t + 1)
    choice_u = [-1] * (num_t + 1)
    choice_t = [-1] * (num_t + 1)

    for t_idx in range(1, num_t + 1):
        phi[t_idx] = phi[t_idx - 1]
        prev[t_idx] = t_idx - 1
        choice_u[t_idx] = -1
        choice_t[t_idx] = -1

        for u_idx in range(num_u):
            l_u = data.l_u[u_idx]
            tau = t_idx - l_u
            if tau < 0:
                continue
            candidate = phi[tau] + eta[u_idx][tau]
            if candidate < phi[t_idx] - CHECKER_TOL["bldp_dp"]:
                phi[t_idx] = candidate
                prev[t_idx] = tau
                choice_u[t_idx] = u_idx
                choice_t[t_idx] = tau

    y: dict[tuple[int, int], int] = {}
    x: dict[tuple[int, int], int] = {}
    t_idx = num_t
    production_cost = 0.0

    while t_idx > 0:
        if choice_u[t_idx] >= 0:
            u_idx = choice_u[t_idx]
            tau_idx = choice_t[t_idx]
            y[(u_idx, tau_idx)] = 1
            production_cost += p_cum[u_idx]
            for i_idx, qty in pack[u_idx][tau_idx].items():
                x[(i_idx, tau_idx)] = x.get((i_idx, tau_idx), 0) + qty
            t_idx = tau_idx
        else:
            t_idx = prev[t_idx]

    reduced_cost = phi[num_t]
    return SubproblemResult(reduced_cost=reduced_cost, x=x, y=y, production_cost=production_cost)


def subproblem_to_column(
    data: ProcessedInstance,
    machine_idx: int,
    column_id: int,
    sp_result: SubproblemResult,
) -> ScheduleColumn:
    machine = data.machines[machine_idx]
    x_named: dict[tuple[str, int], int] = {}
    y_named: dict[tuple[str, int], int] = {}

    for (i_idx, t_idx), qty in sp_result.x.items():
        x_named[(data.cured_items[i_idx], data.periods[t_idx])] = qty
    for (u_idx, t_idx), val in sp_result.y.items():
        if val:
            y_named[(data.configs[u_idx], data.periods[t_idx])] = 1

    return ScheduleColumn(
        machine=machine,
        column_id=column_id,
        x=x_named,
        y=y_named,
        production_cost=sp_result.production_cost,
        reduced_cost=sp_result.reduced_cost,
    )
