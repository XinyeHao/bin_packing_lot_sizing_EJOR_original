"""EBRP-H：1 秒硬限时的批路由定价启发式（§3.3 可行列，非精确最优）。

算法（EBRP-H2）：
1. 与精确 EBRP 相同构建批目录 C（共享 EBRPContext）；
2. 多策略贪心路由（δ 最小 / δ密度 / 窗紧迫度加权），取约化成本最低者；
3. 支持 fixed_y 分支约束；墙钟默认 ≤1s。
"""

from __future__ import annotations

import math
import time
from typing import Callable

from common.data_models import ProcessedInstance
from configuration import CHECKER_TOL, EBRP_HEURISTIC_TIME_LIMIT
from model_and_algo.pricing_ebrp import (
    Batch,
    EBRPContext,
    EBRPStats,
    _index_batches_by_t,
    _next_event_t,
)
from model_and_algo.subproblem_common import SubproblemResult

_BatchScore = Callable[[Batch], float]


def _score_delta(batch: Batch) -> float:
    return batch.delta


def _score_density(batch: Batch) -> float:
    span = max(1, batch.span)
    return batch.delta / span


def _make_urgency_scorer(deadlines: list[int], periods: list[int]) -> _BatchScore:
    """窗越紧的物料所在批，优先选更负的加权 δ。"""

    def _score(batch: Batch) -> float:
        if not batch.items:
            return batch.delta
        urg = 0.0
        for i_idx in batch.items:
            d_i = deadlines[i_idx]
            t = periods[batch.t_idx]
            slack = max(1, d_i - t)
            urg += 1.0 / slack
        return batch.delta - 0.05 * urg * abs(batch.delta)

    return _score


def _greedy_batch_route(
    batches_by_t: list[list[Batch]],
    num_t: int,
    p_cum: list[float],
    *,
    deadline: float | None,
    start_t: int = 0,
    used_mask: int = 0,
    score_fn: _BatchScore = _score_delta,
) -> tuple[float, dict[tuple[int, int], int], dict[tuple[int, int], int], float, int]:
    """按事件时刻贪心选评分最低批次；每件物料至多加工一次。"""
    tol = CHECKER_TOL["pricing_dp"]
    total = 0.0
    x: dict[tuple[int, int], int] = {}
    y: dict[tuple[int, int], int] = {}
    production_cost = 0.0
    t = start_t

    while t < num_t:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if not batches_by_t[t]:
            t = _next_event_t(batches_by_t, t, num_t)
            continue

        best_batch: Batch | None = None
        best_score = 0.0
        for batch in batches_by_t[t]:
            bits = batch.item_mask
            if bits & used_mask:
                continue
            sc = score_fn(batch)
            if sc < best_score - tol:
                best_score = sc
                best_batch = batch

        if best_batch is None:
            t = _next_event_t(batches_by_t, t, num_t)
            continue

        bits = best_batch.item_mask
        total += best_batch.delta
        y[(best_batch.u_idx, best_batch.t_idx)] = 1
        production_cost += p_cum[best_batch.u_idx]
        for i_idx in best_batch.items:
            x[(i_idx, best_batch.t_idx)] = 1
        used_mask |= bits
        t = best_batch.t_idx + best_batch.span

    return total, x, y, production_cost, used_mask


def _solve_forced_heuristic(
    catalog: list[list[list[Batch]]],
    batches_by_t: list[list[Batch]],
    forced: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    beta: float,
    p_cum: list[float],
    num_t: int,
    deadline: float | None,
    score_fn: _BatchScore,
) -> SubproblemResult:
    """分支固定 Y=1 时：段间贪心路由 + 强制批。"""
    forced = sorted(forced, key=lambda x: x[1])
    x: dict[tuple[int, int], int] = {}
    y: dict[tuple[int, int], int] = {}
    production_cost = 0.0
    total = 0.0
    used_mask = 0
    cursor = 0

    for u_idx, tau in forced:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if (u_idx, tau) in forbidden or tau < cursor:
            return SubproblemResult(float("inf"), {}, {}, 0.0)
        if tau > cursor:
            seg_cost, seg_x, seg_y, seg_prod, used_mask = _greedy_batch_route(
                batches_by_t,
                num_t,
                p_cum,
                deadline=deadline,
                start_t=cursor,
                used_mask=used_mask,
                score_fn=score_fn,
            )
            total += seg_cost
            production_cost += seg_prod
            x.update(seg_x)
            y.update(seg_y)
            cursor = tau

        cands = [
            b
            for b in catalog[u_idx][tau]
            if not (b.item_mask & used_mask)
        ]
        if not cands:
            return SubproblemResult(float("inf"), {}, {}, 0.0)
        best_b = min(cands, key=score_fn)
        if tau + best_b.span > num_t:
            return SubproblemResult(float("inf"), {}, {}, 0.0)

        y[(u_idx, tau)] = 1
        production_cost += p_cum[u_idx]
        total += best_b.delta
        bits = best_b.item_mask
        for i_idx in best_b.items:
            x[(i_idx, tau)] = 1
        used_mask |= bits
        cursor = tau + best_b.span

    if cursor < num_t and (deadline is None or time.perf_counter() < deadline):
        seg_cost, seg_x, seg_y, seg_prod, _ = _greedy_batch_route(
            batches_by_t,
            num_t,
            p_cum,
            deadline=deadline,
            start_t=cursor,
            used_mask=used_mask,
            score_fn=score_fn,
        )
        total += seg_cost
        production_cost += seg_prod
        x.update(seg_x)
        y.update(seg_y)

    return SubproblemResult(
        reduced_cost=total - beta,
        x=x,
        y=y,
        production_cost=production_cost,
        is_optimal=False,
    )


def _multi_start_greedy(
    catalog: list[list[list[Batch]]],
    batches_by_t: list[list[Batch]],
    forced: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    beta: float,
    p_cum: list[float],
    num_t: int,
    deadline: float | None,
    data: ProcessedInstance,
) -> SubproblemResult:
    """多策略贪心，取约化成本最低的路线。"""
    scorers: list[_BatchScore] = [_score_delta, _score_density]
    if data.deadlines:
        scorers.append(_make_urgency_scorer(data.deadlines, data.periods))

    best: SubproblemResult | None = None
    for score_fn in scorers:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if forced:
            cand = _solve_forced_heuristic(
                catalog, batches_by_t, forced, forbidden, beta, p_cum, num_t, deadline, score_fn
            )
        else:
            cost, x, y, prod, _ = _greedy_batch_route(
                batches_by_t, num_t, p_cum, deadline=deadline, score_fn=score_fn
            )
            cand = SubproblemResult(
                reduced_cost=cost - beta if math.isfinite(cost) else float("inf"),
                x=x,
                y=y,
                production_cost=prod,
                is_optimal=False,
            )
        if not math.isfinite(cand.reduced_cost):
            continue
        if best is None or cand.reduced_cost < best.reduced_cost - CHECKER_TOL["pricing_dp"]:
            best = cand

    return best or SubproblemResult(float("inf"), {}, {}, 0.0, is_optimal=False)


def solve_subproblem_ebrp_heuristic(
    data: ProcessedInstance,
    machine_idx: int,
    alpha: list[list[float]],
    beta: float,
    rho: list[float] | None = None,
    fixed_y: dict[tuple[int, int], int] | None = None,
    time_limit: float = EBRP_HEURISTIC_TIME_LIMIT,
    stats: EBRPStats | None = None,
    ctx: EBRPContext | None = None,
    alpha_flat=None,
    num_t_for_flat: int | None = None,
    setup_cost_delta: list[list[float]] | None = None,
) -> SubproblemResult:
    """EBRP-H2 多策略贪心定价；墙钟不超过 time_limit（默认 1s）。"""
    num_t = len(data.periods)
    num_u = len(data.configs)
    rho = rho or [0.0] * len(data.cured_items)

    st = stats or EBRPStats()
    st.exact = False
    st.timed_out = False

    solve_start = time.perf_counter()
    deadline = solve_start + time_limit

    if ctx is None:
        ctx = EBRPContext.build(data, machine_idx)

    catalog = ctx.build_catalog(
        data,
        alpha,
        rho,
        alpha_flat=alpha_flat,
        num_t=num_t_for_flat or num_t,
        deadline=deadline,
        setup_cost_delta=setup_cost_delta,
    )

    forbidden: set[tuple[int, int]] = set()
    forced: list[tuple[int, int]] = []
    for (u, t_idx), val in (fixed_y or {}).items():
        if val == 0:
            forbidden.add((u, t_idx))
        elif val == 1:
            forced.append((u, t_idx))
            for other_u in range(num_u):
                if other_u != u:
                    forbidden.add((other_u, t_idx))

    p_cum = data.p_cum[machine_idx]
    batches_by_t = _index_batches_by_t(catalog, num_u, num_t, forbidden)

    sp = _multi_start_greedy(
        catalog, batches_by_t, forced, forbidden, beta, p_cum, num_t, deadline, data
    )

    if time.perf_counter() >= deadline:
        st.timed_out = True

    return sp
