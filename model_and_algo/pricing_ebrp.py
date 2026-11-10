"""EBRP：精确定价子问题求解（批目录 + 记忆化 DP 路由）。

四阶段：
  1. 静态子集枚举 — 每机首轮，穷举容量可行物料子集（与对偶无关）
  2. 批目录生成   — 每轮 CG，计算 δ + Pareto 支配剪枝
  3. DP 路由      — 记忆化 DP，不做前瞻剪枝
  4. 解码         — 从 memo 回溯还原 X/Y 赋值
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

from common.data_models import ProcessedInstance
from configuration import CHECKER_TOL, DEFAULT_PROGRESS_INTERVAL_SEC
from model_and_algo.subproblem_common import SubproblemResult, scale_lengths
from model_and_algo.im_model import _item_start_feasible

ActionKind = Literal["end", "idle", "batch"]


# ──────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Batch:
    u_idx: int
    t_idx: int
    span: int
    items: frozenset[int]
    delta: float
    item_mask: int = 0


@dataclass
class EBRPStats:
    memo_hits: int = 0
    memo_misses: int = 0
    nodes_expanded: int = 0
    pareto_prunes: int = 0
    bound_prunes: int = 0
    timed_out: bool = False
    exact: bool = True


@dataclass
class _MemoEntry:
    cost: float
    kind: ActionKind
    next_t: int = 0
    batch: Batch | None = None


@dataclass
class _PoolSpec:
    u_idx: int
    t_idx: int
    span: int
    pool: list[int]            # 全局物料索引
    weights: list[int]         # 缩放后的整数体积


# ──────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────

def _item_mask_global(items: frozenset[int] | list[int], g2l: dict[int, int]) -> int:
    mask = 0
    for i in items:
        loc = g2l.get(i)
        if loc is not None:
            mask |= 1 << loc
    return mask


# ──────────────────────────────────────────────────────────────────────
# 第一阶段：静态子集枚举
# ──────────────────────────────────────────────────────────────────────

def _build_subset_cache(
    pool_specs: list[_PoolSpec],
    int_capacity: int,
    float_capacity: float,
    item_volumes: list[float],
    lt_at_t: dict[tuple[int, int], int],
    global_to_local: dict[int, int],
) -> list[list[tuple[frozenset[int], int]]]:
    """预计算每 (u, t, s) 的全部容量可行物料子集。

    每个子集恰好被生成一次：对候选池全体运行一次 0/1 背包 DP，
    然后只保留至少包含一个 lt_it == s 物料的子集。
    item_mask 使用机器本地索引（global_to_local 映射）。
    """
    cache: list[list[tuple[frozenset[int], int]]] = []
    for spec in pool_specs:
        n = len(spec.pool)
        # 背包 DP：dp[c] = 容量恰好为 c 的全部子集
        dp: list[set[frozenset[int]]] = [set() for _ in range(int_capacity + 1)]
        dp[0].add(frozenset())

        for loc in range(n):
            w = spec.weights[loc]
            if w > int_capacity:
                continue
            item = spec.pool[loc]
            # 浮点预筛：单件体积超容则跳过
            if item_volumes[item] > float_capacity + 1e-9:
                continue
            for cap in range(int_capacity, w - 1, -1):
                if not dp[cap - w]:
                    continue
                for prev in dp[cap - w]:
                    dp[cap].add(prev | frozenset({item}))

        # 收集全部子集 (subset, item_mask)，浮点校验 + 跨期过滤
        subsets: list[tuple[frozenset[int], int]] = []
        tol = 1e-9
        for cap in range(int_capacity + 1):
            for subset in dp[cap]:
                # 浮点容量校验
                if sum(item_volumes[i] for i in subset) > float_capacity + tol:
                    continue
                # 必须包含至少一个跨期 == span 的物料
                if any(
                    lt_at_t.get((i, spec.t_idx), 0) == spec.span
                    for i in subset
                ):
                    mask = 0
                    for i in subset:
                        loc = global_to_local.get(i)
                        if loc is not None:
                            mask |= 1 << loc
                    subsets.append((subset, mask))

        cache.append(subsets)
    return cache


# ──────────────────────────────────────────────────────────────────────
# EBRPContext
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EBRPContext:
    """单机 EBRP 静态结构缓存：对偶变化时仅重算批目录 delta。"""

    machine_idx: int
    num_t: int
    num_u: int
    capacity: int
    p_cum: list[float]
    local_items: list[int] = field(default_factory=list)
    global_to_local: dict[int, int] = field(default_factory=dict)
    pool_specs: list[_PoolSpec] = field(default_factory=list)
    event_times: list[int] = field(default_factory=list)
    _subset_cache: list[list[tuple[frozenset[int], int]]] = field(default_factory=list)
    _float_capacity: float = 0.0
    _item_volumes: list[float] = field(default_factory=list)
    _lt_at_t: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def build(cls, data: ProcessedInstance, machine_idx: int) -> EBRPContext:
        num_t = len(data.periods)
        num_u = len(data.configs)
        t_max = data.periods[-1]
        q = data.q_m[machine_idx]
        capacity, _ = scale_lengths([1.0], q)
        local_items = list(data.items_by_machine[machine_idx])
        g2l = {g: i for i, g in enumerate(local_items)}
        lt_at_t: dict[tuple[int, int], int] = {}

        ctx = cls(
            machine_idx=machine_idx,
            num_t=num_t,
            num_u=num_u,
            capacity=capacity,
            p_cum=data.p_cum[machine_idx],
            local_items=local_items,
            global_to_local=g2l,
            _float_capacity=data.q_m[machine_idx],
            _item_volumes=list(data.v_i),
            _lt_at_t=lt_at_t,
        )

        has_batch = [False] * num_t
        pool_specs: list[_PoolSpec] = []

        for u_idx in range(num_u):
            for t_idx in range(num_t):
                # 收集该 (u, t) 的可行物料
                eligible: list[int] = []
                levels: set[int] = set()
                for i_idx in data.items_by_config[u_idx]:
                    if data.b_im[i_idx][machine_idx] == 0:
                        continue
                    if not _item_start_feasible(data, i_idx, t_idx, t_max):
                        continue
                    eligible.append(i_idx)
                    lt = data.lt_it[i_idx][t_idx]
                    levels.add(lt)
                    lt_at_t[(i_idx, t_idx)] = lt

                if not eligible:
                    continue

                # 按跨期 s 升序构造候选池
                for span in sorted(levels):
                    if t_idx + span - 1 >= num_t:
                        continue
                    pool: list[int] = []
                    for i_idx in eligible:
                        if data.lt_it[i_idx][t_idx] <= span:
                            pool.append(i_idx)
                    if not pool:
                        continue
                    # 至少存在一个物料跨期 == span
                    if not any(data.lt_it[i][t_idx] == span for i in pool):
                        continue

                    _, weights_pool = scale_lengths(
                        [data.v_i[i] for i in pool], q
                    )
                    pool_specs.append(
                        _PoolSpec(u_idx, t_idx, span, pool, weights_pool)
                    )
                    has_batch[t_idx] = True

        ctx.pool_specs = pool_specs
        ctx._subset_cache = _build_subset_cache(
            pool_specs,
            capacity,
            ctx._float_capacity,
            ctx._item_volumes,
            lt_at_t,
            g2l,
        )
        ctx.event_times = [t for t, flag in enumerate(has_batch) if flag]
        return ctx

    def build_catalog(
        self,
        data: ProcessedInstance,
        alpha: list[list[float]],
        rho: list[float],
        *,
        alpha_flat=None,
        num_t: int | None = None,
        deadline: float | None = None,
        setup_cost_delta: list[list[float]] | None = None,
        required_setups: set[tuple[int, int]] | None = None,
    ) -> list[list[list[Batch]]]:
        """第二阶段：批目录生成。

        将当前对偶 (α, ρ) 代入缓存子集，计算 δ(B)，滤除 δ≥0，
        经 Pareto 支配剪枝后产出批目录。
        """
        catalog: list[list[list[Batch]]] = [
            [[] for _ in range(self.num_t)] for _ in range(self.num_u)
        ]

        for spec_idx, spec in enumerate(self.pool_specs):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            u_idx, t_idx, span = spec.u_idx, spec.t_idx, spec.span
            p_cum_u = self.p_cum[u_idx]
            if setup_cost_delta is not None:
                p_cum_u = p_cum_u + setup_cost_delta[u_idx][t_idx]
            for subset, item_mask in self._subset_cache[spec_idx]:
                # 同一热压罐批次中的所有工件均在批次最长固化时间结束时出罐。
                comp_t = t_idx + span - 1
                if alpha_flat is not None and num_t is not None:
                    arrival_dual = lambda i: alpha_flat[i * num_t + comp_t] if 0 <= comp_t < num_t else 0.0
                else:
                    arrival_dual = lambda i: alpha[i][comp_t] if 0 <= comp_t < self.num_t else 0.0
                dual_sum_val = sum(rho[i] + arrival_dual(i) for i in subset)
                delta = p_cum_u - dual_sum_val
                # Without a Y=1 branch, a nonnegative batch can never improve
                # the pricing objective and may be discarded.  At a forced
                # setup (u,t), however, one batch must be chosen to satisfy
                # Y[u,t]=1.  Such a mandatory batch can have a nonnegative
                # individual delta while the complete schedule still has a
                # negative reduced cost, so it must remain in the catalog.
                if (
                    delta >= 0 - CHECKER_TOL["pricing_dp"]
                    and (required_setups is None or (u_idx, t_idx) not in required_setups)
                ):
                    continue

                catalog[u_idx][t_idx].append(
                    Batch(
                        u_idx=u_idx,
                        t_idx=t_idx,
                        span=span,
                        items=subset,
                        delta=delta,
                        item_mask=item_mask,
                    )
                )

        # Pareto 支配剪枝（per (u,t)）
        for u_idx in range(self.num_u):
            for t_idx in range(self.num_t):
                catalog[u_idx][t_idx] = _prune_batch_cell(catalog[u_idx][t_idx])

        return catalog


# ──────────────────────────────────────────────────────────────────────
# Pareto 支配剪枝
# ──────────────────────────────────────────────────────────────────────

def _prune_batch_cell(batches: list[Batch]) -> list[Batch]:
    """Pareto 支配剪枝：若 A.items ⊆ B.items 且 A.δ ≤ B.δ 且 A.span ≤ B.span，
    则 B 被 A 支配（用更少物料达相同或更低成本、占相同或更短时段，永不劣势）。
    """
    if not batches:
        return batches
    tol = CHECKER_TOL["pricing_dp"]

    # 按 (span, |items|, delta) 升序，优先处理更小跨期 + 更少物料的批
    sorted_batches = sorted(batches, key=lambda b: (b.span, len(b.items), b.delta))
    kept: list[Batch] = []
    for b in sorted_batches:
        dominated = False
        for o in kept:
            if o.span <= b.span and o.items <= b.items and o.delta <= b.delta + tol:
                dominated = True
                break
        if not dominated:
            kept.append(b)
    return kept


# ──────────────────────────────────────────────────────────────────────
# 第三阶段：DP 路由
# ──────────────────────────────────────────────────────────────────────

def _index_batches_by_t(
    catalog: list[list[list[Batch]]],
    num_u: int,
    num_t: int,
    forbidden: set[tuple[int, int]],
) -> list[list[Batch]]:
    """将三维 catalog[u][t] 展平为按 t 索引的批列表。"""
    by_t: list[list[Batch]] = [[] for _ in range(num_t)]
    for u_idx in range(num_u):
        for t_idx in range(num_t):
            if (u_idx, t_idx) in forbidden:
                continue
            by_t[t_idx].extend(catalog[u_idx][t_idx])
    for t_idx in range(num_t):
        if not by_t[t_idx]:
            continue
        by_t[t_idx] = _prune_batch_cell(by_t[t_idx])
    return by_t


def _next_event_t(
    batches_by_t: list[list[Batch]],
    t: int,
    num_t: int,
) -> int:
    """返回 >= t 的下一个有批次的时刻。"""
    while t < num_t:
        if batches_by_t[t]:
            return t
        t += 1
    return num_t


def _solve_routing(
    batches_by_t: list[list[Batch]],
    num_t: int,
    *,
    deadline: float | None,
    st: EBRPStats,
    progress_interval_sec: float,
    machine: str,
    start_t: int = 0,
    start_mask: int = 0,
    forced_at: dict[int, int] | None = None,
    sri_terms: tuple[tuple[int, float], ...] = (),
) -> tuple[float, dict[tuple[int, int], _MemoEntry]]:
    """记忆化 DP 路由（递归，精确）。

    状态：(t, used_mask)，t 为当前时刻，used_mask 为已用物料位掩码。
    forced_at: 若给出 {t: u}，则时刻 t 必须开工配置 u（Y=1 分支），不可空闲跳过。
    """
    # 预排序：每时刻的批按 delta 升序（最负优先），尽早收紧 best 便于递归分支剪枝
    for t in range(num_t):
        if batches_by_t[t]:
            batches_by_t[t].sort(key=lambda b: b.delta)

    # 预计算 next_event[t] = 下一个有批次的时刻（>= t）
    next_event = [num_t] * (num_t + 1)
    nxt = num_t
    for t in range(num_t - 1, -1, -1):
        next_event[t] = nxt
        if batches_by_t[t]:
            nxt = t
    next_event[num_t] = num_t

    forced_times = sorted(forced_at.keys()) if forced_at else []

    # 打包键：单整数替代 tuple，减少 Python 分配与哈希开销
    # shift 必须 ≥ max_item_index，否则 (t, mask) 会与 (t-1, mask + 2^shift) 碰撞
    # 保守取 64 位，足以覆盖所有实测算例
    shift = 64
    def _pack(t_idx: int, mask: int) -> int:
        return (t_idx << shift) | mask

    memo: dict[int, _MemoEntry] = {}
    solve_start = time.perf_counter()
    last_progress = solve_start
    tol = CHECKER_TOL["pricing_dp"]
    _next_event = next_event  # local binding
    _batches_by_t = batches_by_t
    _num_t = num_t
    _deadline = deadline
    _forced_at = forced_at or {}
    _sri_terms = sri_terms

    def _sri_transition_cost(used_mask: int, batch_mask: int) -> float:
        """Reduced-cost contribution of rank-1 SRI duals for one new batch."""
        next_mask = used_mask | batch_mask
        return sum(
            -pi * (((next_mask & item_mask).bit_count() // 2) - ((used_mask & item_mask).bit_count() // 2))
            for item_mask, pi in _sri_terms
        )

    # 预计算乐观下界：lb_sum[t] = sum_{t'>=t} min(0, min_delta_at_t')
    # 忽略物料冲突，是真实未来成本的下界（更负 = 更乐观）
    # 注意：若某时刻所有批的 delta >= 0，最优选择是空闲（成本 0）
    lb_sum = [0.0] * (_num_t + 1)
    for t in range(_num_t - 1, -1, -1):
        batches = _batches_by_t[t]
        best_at_t = 0.0
        if t in _forced_at:
            u_req = _forced_at[t]
            forced_deltas = [b.delta for b in batches if b.u_idx == u_req]
            best_at_t = min(forced_deltas) if forced_deltas else 0.0
        elif batches and batches[0].delta < 0.0:
            best_at_t = batches[0].delta  # batches 已按 delta 升序排列
        lb_sum[t] = best_at_t + lb_sum[t + 1]
    lb_sum[_num_t] = 0.0

    _node_counter = [0]

    def _progress_report(force: bool = False) -> None:
        nonlocal last_progress
        if progress_interval_sec <= 0:
            return
        now = time.perf_counter()
        if not force and now - last_progress < progress_interval_sec:
            return
        last_progress = now
        elapsed = now - solve_start
        print(
            f"[EBRP {machine}] +{elapsed:6.0f}s  "
            f"nnodes={st.nodes_expanded}  memo={len(memo)}  exact={st.exact}",
            flush=True,
            file=sys.stderr,
        )

    def timed_out() -> bool:
        if _deadline is not None and time.perf_counter() >= _deadline:
            st.timed_out = True
            st.exact = False
            return True
        return False

    def _idle_target(t_idx: int) -> int:
        """空闲跳转的下一时刻；不得越过未处理的强制开工点。"""
        nxt = _next_event[t_idx]
        for ft in forced_times:
            if t_idx <= ft < nxt:
                return ft
        return nxt

    def solve(t_idx: int, used_mask: int) -> float:
        _node_counter[0] += 1
        if _node_counter[0] & 63 == 0 and timed_out():
            return math.inf
        if t_idx >= _num_t:
            return 0.0

        key = _pack(t_idx, used_mask)
        cached = memo.get(key)
        if cached is not None:
            st.memo_hits += 1
            return cached.cost

        st.memo_misses += 1
        st.nodes_expanded += 1
        _progress_report()

        u_req = _forced_at.get(t_idx)
        if u_req is not None:
            # Y=1：本时刻必须开工 u_req，禁止空闲
            best = math.inf
            entry = _MemoEntry(best, "end")
            for b in _batches_by_t[t_idx]:
                if b.u_idx != u_req:
                    continue
                if b.item_mask & used_mask:
                    continue
                et = t_idx + b.span
                if et > _num_t:
                    continue
                # 不得跨越后续强制开工点
                if any(t_idx < ft < et for ft in forced_times):
                    continue
                if b.delta + lb_sum[et] >= best - tol:
                    continue
                cand = b.delta + _sri_transition_cost(used_mask, b.item_mask) + solve(et, used_mask | b.item_mask)
                if cand < best - tol:
                    best = cand
                    entry = _MemoEntry(best, "batch", next_t=et, batch=b)
            memo[key] = entry
            return best

        nxt = _idle_target(t_idx)
        idle_best = solve(nxt, used_mask)
        best = idle_best
        entry = _MemoEntry(best, "idle", next_t=nxt)

        for b in _batches_by_t[t_idx]:
            if b.item_mask & used_mask:
                continue
            et = t_idx + b.span
            if et > _num_t:
                continue
            # 不得跨越强制开工点（否则会跳过 Y=1）
            if any(t_idx < ft < et for ft in forced_times):
                continue

            # 乐观下界剪枝：即使未来没有任何冲突，也无法胜过 best
            if b.delta + lb_sum[et] >= best - tol:
                continue

            cand = b.delta + _sri_transition_cost(used_mask, b.item_mask) + solve(et, used_mask | b.item_mask)
            if cand < best - tol:
                best = cand
                entry = _MemoEntry(best, "batch", next_t=et, batch=b)

        memo[key] = entry
        return best

    batch_cost = solve(start_t, start_mask)
    _progress_report(force=True)

    # 将打包键还原为 tuple 键，供 _decode_plan 使用
    unpacked_memo: dict[tuple[int, int], _MemoEntry] = {}
    for packed_key, val in memo.items():
        t_val = packed_key >> shift
        m_val = packed_key & ((1 << shift) - 1)
        unpacked_memo[(t_val, m_val)] = val

    return batch_cost, unpacked_memo


# ──────────────────────────────────────────────────────────────────────
# 第四阶段：解码
# ──────────────────────────────────────────────────────────────────────

def _decode_plan(
    memo: dict[tuple[int, int], _MemoEntry],
    t_idx: int,
    used_mask: int,
    num_t: int,
    p_cum: list[float],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], float]:
    x: dict[tuple[int, int], int] = {}
    y: dict[tuple[int, int], int] = {}
    production_cost = 0.0

    t = t_idx
    used = used_mask
    while t < num_t:
        key = (t, used)
        if key not in memo:
            break
        entry = memo[key]
        if entry.kind == "end":
            break
        if entry.kind == "idle":
            t = entry.next_t
            continue
        batch = entry.batch
        if batch is None:
            break
        y[(batch.u_idx, batch.t_idx)] = 1
        production_cost += p_cum[batch.u_idx]
        for i_idx in batch.items:
            x[(i_idx, batch.t_idx)] = 1
        used |= batch.item_mask
        t = entry.next_t

    return x, y, production_cost


# ──────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────

def build_ebrp_contexts(data: ProcessedInstance) -> list[EBRPContext]:
    return [EBRPContext.build(data, m_idx) for m_idx in range(len(data.machines))]


def solve_subproblem_ebrp(
    data: ProcessedInstance,
    machine_idx: int,
    alpha: list[list[float]],
    beta: float,
    rho: list[float] | None = None,
    catalog: list[list[list[Batch]]] | None = None,
    fixed_y: dict[tuple[int, int], int] | None = None,
    time_limit: float | None = None,
    max_nodes: int | None = None,
    stats: EBRPStats | None = None,
    progress_interval_sec: float = DEFAULT_PROGRESS_INTERVAL_SEC,
    ctx: EBRPContext | None = None,
    alpha_flat=None,
    num_t_for_flat: int | None = None,
    skip_cross_check: bool = False,
    setup_cost_delta: list[list[float]] | None = None,
    sri_terms: tuple[tuple[int, float], ...] = (),
) -> SubproblemResult:
    num_t = len(data.periods)
    num_i = len(data.cured_items)
    num_u = len(data.configs)
    p_cum = data.p_cum[machine_idx]
    rho_val = rho or [0.0] * num_i

    st = stats or EBRPStats()
    solve_start = time.perf_counter()
    deadline = solve_start + time_limit if time_limit else None
    machine = data.machines[machine_idx]

    if ctx is None:
        ctx = EBRPContext.build(data, machine_idx)

    # Process branch fixes before building the catalog.  Y=1 fixes change the
    # admissible pricing problem: the corresponding setup must retain even
    # nonnegative reduced-cost batches (see build_catalog).
    forbidden: set[tuple[int, int]] = set()
    forced_at: dict[int, int] = {}
    if fixed_y:
        for (u, t), val in fixed_y.items():
            if val == 0:
                forbidden.add((u, t))
            elif val == 1:
                if t in forced_at and forced_at[t] != u:
                    return SubproblemResult(
                        reduced_cost=float("inf"),
                        x={},
                        y={},
                        production_cost=0.0,
                        is_optimal=False,
                    )
                forced_at[t] = u
                for other_u in range(num_u):
                    if other_u != u:
                        forbidden.add((other_u, t))

    # A cached catalog has already discarded nonnegative-delta batches.  It is
    # valid only when no setup is forced.  Rebuild under Y=1 fixes so mandatory
    # batches at the forced setups are present.
    if catalog is None or forced_at:
        catalog = ctx.build_catalog(
            data,
            alpha,
            rho_val,
            alpha_flat=alpha_flat,
            num_t=num_t_for_flat or num_t,
            deadline=deadline,
            setup_cost_delta=setup_cost_delta,
            required_setups={(u, t) for t, u in forced_at.items()},
        )
        if deadline is not None and time.perf_counter() >= deadline:
            st.timed_out = True
            st.exact = False
            return SubproblemResult(
                reduced_cost=float("inf"),
                x={},
                y={},
                production_cost=0.0,
                is_optimal=False,
            )

    batches_by_t = _index_batches_by_t(catalog, num_u, num_t, forbidden)

    if progress_interval_sec > 0:
        total_batches = sum(len(bt) for bt in batches_by_t)
        print(
            f"[EBRP {machine}] search start  pools={len(ctx.pool_specs)}  "
            f"batches={total_batches}  events={len(ctx.event_times)}",
            flush=True,
            file=sys.stderr,
        )

    batch_cost, memo = _solve_routing(
        batches_by_t,
        num_t,
        deadline=deadline,
        st=st,
        progress_interval_sec=progress_interval_sec,
        machine=machine,
        forced_at=forced_at or None,
        sri_terms=sri_terms,
    )
    if st.timed_out and not math.isfinite(batch_cost):
        batch_cost = math.inf

    reduced_cost = batch_cost - beta if math.isfinite(batch_cost) else float("inf")
    x, y, production_cost = _decode_plan(memo, 0, 0, num_t, p_cum)

    # Gurobi 交叉验证（仅在 EBRP 认为无负 RC 列时，兜底防止实现瑕疵）
    if not skip_cross_check and not sri_terms and reduced_cost > -1e-3 and not fixed_y and not st.timed_out:
        try:
            from model_and_algo.pricing_im_gurobi import solve_subproblem_im_gurobi

            if alpha_flat is not None and num_t_for_flat is not None:
                gb_alpha = [[0.0] * num_t_for_flat for _ in range(num_i)]
                for i_idx in range(num_i):
                    base = i_idx * num_t_for_flat
                    row = gb_alpha[i_idx]
                    for t_idx in range(num_t_for_flat):
                        row[t_idx] = alpha_flat[base + t_idx]
            else:
                gb_alpha = alpha if alpha else [[]]

            sp_gb = solve_subproblem_im_gurobi(
                data, machine_idx, gb_alpha, beta, rho_val,
                time_limit=5.0, progress_interval_sec=0.0,
                setup_cost_delta=setup_cost_delta,
            )
            if sp_gb.reduced_cost < reduced_cost - CHECKER_TOL["pricing_dp"]:
                if progress_interval_sec > 0:
                    print(
                        f"[EBRP {machine}] Gurobi found better "
                        f"RC={sp_gb.reduced_cost:.4f} vs EBRP={reduced_cost:.4f}",
                        flush=True, file=sys.stderr,
                    )
                return SubproblemResult(
                    reduced_cost=sp_gb.reduced_cost,
                    x=sp_gb.x,
                    y=sp_gb.y,
                    production_cost=sp_gb.production_cost,
                    is_optimal=sp_gb.is_optimal,
                )
        except Exception as exc:
            print(
                f"[EBRP {machine}] Gurobi cross-check failed: {exc}",
                flush=True, file=sys.stderr,
            )

    return SubproblemResult(
        reduced_cost=reduced_cost,
        x=x,
        y=y,
        production_cost=production_cost,
        is_optimal=st.exact and not st.timed_out,
    )
