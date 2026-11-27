"""EBRP：精确定价子问题求解（批目录 + 记忆化 DP 路由）。

四阶段：
  1. 静态子集枚举 — 每机首轮，穷举容量可行物料子集（与对偶无关）
  2. 批目录生成   — 每轮 CG，计算 δ + Pareto 支配剪枝
  3. DP 路由      — 时间 DAG 下界 + DSSR，必要时回退完整记忆化 DP
  4. 解码         — 从 memo 回溯还原 X/Y 赋值
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numba import njit

from common.data_models import ProcessedInstance
from configuration import CHECKER_TOL, DEFAULT_PROGRESS_INTERVAL_SEC
from model_and_algo.subproblem_common import SubproblemResult, scale_lengths
from model_and_algo.im_model import _item_start_feasible
from model_and_algo.pricing_route_dssr import solve_dssr

ActionKind = Literal["end", "idle", "batch"]

_EBRP_TIME_LOCK = threading.Lock()
EBRP_TIME = {
    "catalog": 0.0, "route": 0.0, "calls": 0,
    "dssr_calls": 0, "dssr_rounds": 0, "dssr_nodes": 0, "dssr_fallbacks": 0,
}


def reset_ebrp_time() -> None:
    with _EBRP_TIME_LOCK:
        for key in EBRP_TIME:
            EBRP_TIME[key] = 0


def _add_ebrp_time(catalog_sec: float, route_sec: float) -> None:
    with _EBRP_TIME_LOCK:
        EBRP_TIME["catalog"] += catalog_sec
        EBRP_TIME["route"] += route_sec
        EBRP_TIME["calls"] += 1


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
    dssr_rounds: int = 0
    critical_items: int = 0


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
    _subset_items: list[np.ndarray] = field(default_factory=list)
    _subset_ptr: list[np.ndarray] = field(default_factory=list)
    _subset_mask: list[np.ndarray] = field(default_factory=list)
    _subset_n: list[np.ndarray] = field(default_factory=list)
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
        ctx._subset_items = []
        ctx._subset_ptr = []
        ctx._subset_mask = []
        ctx._subset_n = []
        for subsets in ctx._subset_cache:
            item_idx: list[int] = []
            ptr = [0]
            for subset, _mask in subsets:
                item_idx.extend(subset)
                ptr.append(len(item_idx))
            ctx._subset_items.append(np.asarray(item_idx, dtype=np.int32))
            ctx._subset_ptr.append(np.asarray(ptr, dtype=np.int32))
            ctx._subset_mask.append(
                np.asarray([mask for _subset, mask in subsets], dtype=np.uint64)
            )
            ctx._subset_n.append(
                np.asarray([len(subset) for subset, _mask in subsets], dtype=np.int16)
            )
        ctx.event_times = [t for t, flag in enumerate(has_batch) if flag]
        ctx._score_pack = None
        return ctx

    def _pareto_cells(
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
    ) -> list[list[tuple[np.ndarray, ...]]]:
        """Score cached subsets and Pareto-prune each (u, t) without Batch objects.

        Each cell is ``(span, nitem, delta, mask, spec_idx, sub_idx)``.
        """
        chunks: list[list[list[tuple]]] = [
            [[] for _ in range(self.num_t)] for _ in range(self.num_u)
        ]
        tol = CHECKER_TOL["pricing_dp"]
        rho_vec = np.asarray(rho, dtype=np.float64)
        num_i = int(rho_vec.shape[0])
        flat = None
        alpha_arr = None
        if alpha_flat is not None and num_t is not None:
            flat = np.asarray(alpha_flat, dtype=np.float64)
        elif alpha:
            alpha_arr = np.asarray(alpha, dtype=np.float64)

        for spec_idx, spec in enumerate(self.pool_specs):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            nsub = int(self._subset_n[spec_idx].shape[0])
            if nsub == 0:
                continue
            u_idx, t_idx, span = spec.u_idx, spec.t_idx, spec.span
            p_cum_u = self.p_cum[u_idx]
            if setup_cost_delta is not None:
                p_cum_u = p_cum_u + setup_cost_delta[u_idx][t_idx]
            comp_t = t_idx + span - 1
            if flat is not None and 0 <= comp_t < int(num_t):
                coeff = rho_vec + flat[comp_t:comp_t + num_i * int(num_t):int(num_t)]
            elif alpha_arr is not None and 0 <= comp_t < alpha_arr.shape[1]:
                coeff = rho_vec + alpha_arr[:, comp_t]
            else:
                coeff = rho_vec
            dual_sum = np.add.reduceat(coeff[self._subset_items[spec_idx]], self._subset_ptr[spec_idx][:-1])
            deltas = p_cum_u - dual_sum
            keep_nonnegative = required_setups is not None and (u_idx, t_idx) in required_setups
            chosen = _negative_indices(deltas, keep_nonnegative)
            if chosen.size == 0:
                continue
            chunks[u_idx][t_idx].append((span, spec_idx, chosen, deltas))

        empty = (
            np.empty(0, np.int64),
            np.empty(0, np.int16),
            np.empty(0, np.float64),
            np.empty(0, np.uint64),
            np.empty(0, np.int32),
            np.empty(0, np.int32),
        )
        cells: list[list[tuple[list, ...]]] = [
            [empty for _ in range(self.num_t)] for _ in range(self.num_u)
        ]
        for u_idx in range(self.num_u):
            for t_idx in range(self.num_t):
                parts = chunks[u_idx][t_idx]
                if not parts:
                    continue
                total = sum(part[2].size for part in parts)
                span_arr = np.empty(total, dtype=np.int64)
                n_arr = np.empty(total, dtype=np.int16)
                d_arr = np.empty(total, dtype=np.float64)
                m_arr = np.empty(total, dtype=np.uint64)
                spec_arr = np.empty(total, dtype=np.int32)
                sub_arr = np.empty(total, dtype=np.int32)
                pos = 0
                for span, spec_idx, chosen, deltas in parts:
                    n = int(chosen.size)
                    span_arr[pos:pos + n] = span
                    spec_arr[pos:pos + n] = spec_idx
                    sub_arr[pos:pos + n] = chosen
                    d_arr[pos:pos + n] = deltas[chosen]
                    m_arr[pos:pos + n] = self._subset_mask[spec_idx][chosen]
                    n_arr[pos:pos + n] = self._subset_n[spec_idx][chosen]
                    pos += n
                order = np.lexsort((np.arange(total), d_arr, n_arr, span_arr))
                keep = _pareto_keep_numba(order, span_arr, d_arr, m_arr)
                cells[u_idx][t_idx] = (
                    span_arr[keep],
                    n_arr[keep],
                    d_arr[keep],
                    m_arr[keep],
                    spec_arr[keep],
                    sub_arr[keep],
                )
        return cells

    def _ensure_score_pack(self) -> None:
        """Concatenate per-spec subset tables so one compiled pass can score them."""
        if self._score_pack is not None:
            return
        nspec = len(self.pool_specs)
        spec_u = np.empty(nspec, dtype=np.int32)
        spec_t = np.empty(nspec, dtype=np.int32)
        spec_span = np.empty(nspec, dtype=np.int32)
        spec_comp = np.empty(nspec, dtype=np.int32)
        spec_p = np.empty(nspec, dtype=np.float64)
        spec_ptr = np.zeros(nspec + 1, dtype=np.int32)
        masks: list[np.ndarray] = []
        ns: list[np.ndarray] = []
        iptr = [0]
        item_chunks: list[np.ndarray] = []
        for s, spec in enumerate(self.pool_specs):
            spec_u[s] = spec.u_idx
            spec_t[s] = spec.t_idx
            spec_span[s] = spec.span
            spec_comp[s] = spec.t_idx + spec.span - 1
            spec_p[s] = self.p_cum[spec.u_idx]
            nsub = int(self._subset_mask[s].shape[0])
            spec_ptr[s + 1] = spec_ptr[s] + nsub
            masks.append(self._subset_mask[s])
            ns.append(self._subset_n[s])
            local_ptr = self._subset_ptr[s]
            local_items = self._subset_items[s]
            base = iptr[-1]
            item_chunks.append(local_items)
            for j in range(nsub):
                iptr.append(base + int(local_ptr[j + 1]))
        self._score_pack = {
            "u": spec_u,
            "t": spec_t,
            "span": spec_span,
            "comp": spec_comp,
            "p": spec_p,
            "spec_ptr": spec_ptr,
            "iptr": np.asarray(iptr, dtype=np.int32),
            "items": np.concatenate(item_chunks) if item_chunks else np.empty(0, dtype=np.int32),
            "mask": np.concatenate(masks) if masks else np.empty(0, dtype=np.uint64),
            "n": np.concatenate(ns) if ns else np.empty(0, dtype=np.int16),
        }

    def score_route_batches(
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
        forbidden: set[tuple[int, int]] | None = None,
        preserve_all: bool = False,
    ) -> "_FlatRoute":
        """Score every cached subset once, then Pareto-prune and delta-sort each time."""
        del data
        self._ensure_score_pack()
        pack = self._score_pack
        nspec = int(pack["u"].shape[0])
        periods = self.num_t if num_t is None else int(num_t)
        num_i = len(rho)
        alpha_mat = np.zeros((num_i, periods), dtype=np.float64)
        if alpha_flat is not None:
            flat_arr = np.asarray(alpha_flat, dtype=np.float64)
            alpha_mat[:, :] = flat_arr[: num_i * periods].reshape(num_i, periods)
        elif alpha:
            alpha_mat[:, :] = np.asarray(alpha, dtype=np.float64)[:, :periods]
        rho_vec = np.asarray(rho, dtype=np.float64)
        spec_p = pack["p"].copy()
        if setup_cost_delta is not None:
            for s in range(nspec):
                u_idx = int(pack["u"][s])
                t_idx = int(pack["t"][s])
                spec_p[s] += setup_cost_delta[u_idx][t_idx]
        skip = np.zeros(nspec, dtype=np.uint8)
        keep_all = np.zeros(nspec, dtype=np.uint8)
        # RF-same obligations may need a positive-cost companion batch.
        # Mandatory setups may invalidate ordinary subset dominance, too.
        if preserve_all:
            keep_all[:] = 1
        if forbidden:
            for s in range(nspec):
                if (int(pack["u"][s]), int(pack["t"][s])) in forbidden:
                    skip[s] = 1
        if required_setups:
            for s in range(nspec):
                if (int(pack["u"][s]), int(pack["t"][s])) in required_setups:
                    keep_all[s] = 1
        ptr, delta_b, mask_b, span_b, n_b, u_b, spec_b, sub_b = _emit_negative_buckets(
            pack["u"],
            pack["t"],
            pack["span"],
            pack["comp"],
            spec_p,
            pack["spec_ptr"],
            skip,
            keep_all,
            pack["iptr"],
            pack["items"],
            pack["mask"],
            pack["n"],
            rho_vec,
            alpha_mat,
            self.num_t,
        )
        pieces: list[list[np.ndarray]] = [[] for _ in range(7)]
        offsets = [0]
        for t_idx in range(self.num_t):
            a = int(ptr[t_idx])
            b = int(ptr[t_idx + 1])
            if deadline is not None and time.perf_counter() >= deadline:
                while len(offsets) < self.num_t + 1:
                    offsets.append(offsets[-1])
                break
            if a == b:
                offsets.append(offsets[-1])
                continue
            span = span_b[a:b]
            delta = delta_b[a:b]
            mask = mask_b[a:b]
            nitem = n_b[a:b]
            order = np.lexsort((np.arange(b - a), delta, nitem, span))
            keep = (np.arange(b - a, dtype=np.int64) if preserve_all
                    else _pareto_keep_numba(order, span, delta, mask))
            delta_k = delta[keep]
            delta_order = np.argsort(delta_k, kind="mergesort")
            pieces[0].append(delta_k[delta_order])
            pieces[1].append(mask[keep][delta_order])
            pieces[2].append(span[keep][delta_order])
            pieces[3].append(u_b[a:b][keep][delta_order])
            pieces[4].append(np.full(delta_order.shape[0], t_idx, dtype=np.int64))
            pieces[5].append(spec_b[a:b][keep][delta_order])
            pieces[6].append(sub_b[a:b][keep][delta_order])
            offsets.append(offsets[-1] + int(delta_order.shape[0]))
        if pieces[0]:
            stacked = [np.concatenate(col) for col in pieces]
        else:
            stacked = [
                np.empty(0, np.float64),
                np.empty(0, np.uint64),
                np.empty(0, np.int64),
                np.empty(0, np.int64),
                np.empty(0, np.int64),
                np.empty(0, np.int32),
                np.empty(0, np.int32),
            ]
        return _FlatRoute(
            delta=stacked[0],
            mask=stacked[1],
            span=stacked[2],
            u_idx=stacked[3],
            t_idx=stacked[4],
            spec_idx=stacked[5],
            sub_idx=stacked[6],
            offsets=np.asarray(offsets, dtype=np.int64),
        )

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
        经 Pareto 支配剪枝后才为留下的批次建立对象。
        """
        cells = self._pareto_cells(
            data,
            alpha,
            rho,
            alpha_flat=alpha_flat,
            num_t=num_t,
            deadline=deadline,
            setup_cost_delta=setup_cost_delta,
            required_setups=required_setups,
        )
        catalog: list[list[list[Batch]]] = [
            [[] for _ in range(self.num_t)] for _ in range(self.num_u)
        ]
        for u_idx in range(self.num_u):
            for t_idx in range(self.num_t):
                span, _nitem, delta, mask, spec, sub = cells[u_idx][t_idx]
                cell = catalog[u_idx][t_idx]
                for i in range(len(span)):
                    subset = self._subset_cache[int(spec[i])][int(sub[i])][0]
                    cell.append(
                        Batch(
                            u_idx=u_idx,
                            t_idx=t_idx,
                            span=int(span[i]),
                            items=subset,
                            delta=float(delta[i]),
                            item_mask=int(mask[i]),
                        )
                    )
        return catalog


# ──────────────────────────────────────────────────────────────────────
# Pareto 支配剪枝
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _FlatRoute:
    delta: np.ndarray
    mask: np.ndarray
    span: np.ndarray
    u_idx: np.ndarray
    t_idx: np.ndarray
    spec_idx: np.ndarray
    sub_idx: np.ndarray
    offsets: np.ndarray


@njit(cache=True, nogil=True)
def _time_dag_bound(deltas, spans, u_idxs, offsets, forced_u, num_t):
    """Time-feasible shortest path, relaxing only item/RF restrictions."""
    values = np.zeros(num_t + 1, dtype=np.float64)
    next_forced = np.full(num_t, num_t, dtype=np.int64)
    nxt = num_t
    for t in range(num_t - 1, -1, -1):
        next_forced[t] = nxt
        if forced_u[t] >= 0:
            nxt = t
    for t in range(num_t - 1, -1, -1):
        required = forced_u[t]
        best = np.inf if required >= 0 else values[t + 1]
        for b in range(offsets[t], offsets[t + 1]):
            if required >= 0 and u_idxs[b] != required:
                continue
            end = t + spans[b]
            if end > num_t or next_forced[t] < end:
                continue
            candidate = deltas[b] + values[end]
            if candidate < best:
                best = candidate
        values[t] = best
    return values


def _routing_lower_bounds(deltas, spans, u_idxs, offsets, num_t, forced_at, sri_terms=()):
    forced_u = np.full(num_t, -1, dtype=np.int64)
    for t, u in (forced_at or {}).items():
        if 0 <= t < num_t:
            forced_u[t] = u
    values = _time_dag_bound(deltas, spans, u_idxs, offsets, forced_u, num_t)
    # Normally SRI multipliers of <= rows are nonpositive and their routing
    # increments are nonnegative. This floor also covers positive multipliers
    # conservatively, including a possibly negative immediate transition.
    sri_floor = sum(min(0.0, -pi) * (int(mask).bit_count() // 2) for mask, pi in sri_terms)
    if sri_floor:
        values = values + sri_floor
    return values.tolist()


@njit(cache=True)
def _emit_negative_buckets(
    spec_u,
    spec_t,
    spec_span,
    spec_comp,
    spec_p,
    spec_ptr,
    spec_skip,
    spec_keep_all,
    iptr,
    items,
    masks,
    nitems,
    rho,
    alpha,
    num_t,
):
    """One pass over every cached subset. Negatives are bucketed by start time."""
    nspec = spec_u.shape[0]
    counts = np.zeros(num_t, dtype=np.int32)
    for s in range(nspec):
        if spec_skip[s] != 0:
            continue
        comp = spec_comp[s]
        p_cum = spec_p[s]
        t_idx = spec_t[s]
        keep_all = spec_keep_all[s] != 0
        in_horizon = 0 <= comp < alpha.shape[1]
        for j in range(spec_ptr[s], spec_ptr[s + 1]):
            acc = 0.0
            for q in range(iptr[j], iptr[j + 1]):
                i_idx = items[q]
                acc += rho[i_idx]
                if in_horizon:
                    acc += alpha[i_idx, comp]
            if keep_all or p_cum - acc < -1e-9:
                if 0 <= t_idx < num_t:
                    counts[t_idx] += 1
    ptr = np.empty(num_t + 1, dtype=np.int32)
    ptr[0] = 0
    for t_idx in range(num_t):
        ptr[t_idx + 1] = ptr[t_idx] + counts[t_idx]
    total = int(ptr[num_t])
    head = ptr.copy()
    out_delta = np.empty(total, dtype=np.float64)
    out_mask = np.empty(total, dtype=np.uint64)
    out_span = np.empty(total, dtype=np.int64)
    out_n = np.empty(total, dtype=np.int16)
    out_u = np.empty(total, dtype=np.int64)
    out_spec = np.empty(total, dtype=np.int32)
    out_sub = np.empty(total, dtype=np.int32)
    for s in range(nspec):
        if spec_skip[s] != 0:
            continue
        comp = spec_comp[s]
        p_cum = spec_p[s]
        t_idx = spec_t[s]
        keep_all = spec_keep_all[s] != 0
        in_horizon = 0 <= comp < alpha.shape[1]
        local = 0
        for j in range(spec_ptr[s], spec_ptr[s + 1]):
            acc = 0.0
            for q in range(iptr[j], iptr[j + 1]):
                i_idx = items[q]
                acc += rho[i_idx]
                if in_horizon:
                    acc += alpha[i_idx, comp]
            delta = p_cum - acc
            if (keep_all or delta < -1e-9) and 0 <= t_idx < num_t:
                w = head[t_idx]
                head[t_idx] = w + 1
                out_delta[w] = delta
                out_mask[w] = masks[j]
                out_span[w] = spec_span[s]
                out_n[w] = nitems[j]
                out_u[w] = spec_u[s]
                out_spec[w] = s
                out_sub[w] = local
            local += 1
    return ptr, out_delta, out_mask, out_span, out_n, out_u, out_spec, out_sub


@njit(cache=True)
def _negative_indices(deltas, keep_nonnegative):
    n = deltas.shape[0]
    sel = np.empty(n, dtype=np.int32)
    nn = 0
    if keep_nonnegative:
        for i in range(n):
            sel[nn] = i
            nn += 1
    else:
        for i in range(n):
            if deltas[i] < -1e-9:
                sel[nn] = i
                nn += 1
    return sel[:nn]


@njit(cache=True)
def _pareto_keep_numba(order, span, delta, mask):
    """Pareto indices. ``order`` is already sorted by span, size, delta."""
    n = order.shape[0]
    kept_pos = np.empty(n, dtype=np.int32)
    ks = np.empty(n, dtype=np.int64)
    km = np.empty(n, dtype=np.uint64)
    kd = np.empty(n, dtype=np.float64)
    nkeep = 0
    tol = 1e-9
    for p in range(n):
        i = order[p]
        si = span[i]
        mi = mask[i]
        di = delta[i]
        dominated = False
        for k in range(nkeep):
            if ks[k] <= si and (km[k] & mi) == km[k] and kd[k] <= di + tol:
                dominated = True
                break
        if not dominated:
            kept_pos[nkeep] = i
            ks[nkeep] = si
            km[nkeep] = mi
            kd[nkeep] = di
            nkeep += 1
    return kept_pos[:nkeep]


def _prune_batch_cell(batches: list[Batch]) -> list[Batch]:
    """Pareto 支配剪枝：若 A.items ⊆ B.items 且 A.δ ≤ B.δ 且 A.span ≤ B.span，
    则 B 被 A 支配（用更少物料达相同或更低成本、占相同或更短时段，永不劣势）。
    """
    if not batches:
        return batches
    tol = CHECKER_TOL["pricing_dp"]

    # 按 (span, |items|, delta) 升序，优先处理更小跨期 + 更少物料的批。
    # 掩码包含关系与 items 的子集关系相同。
    sorted_batches = sorted(batches, key=lambda b: (b.span, len(b.items), b.delta))
    kept: list[Batch] = []
    kept_span: list[int] = []
    kept_mask: list[int] = []
    kept_delta: list[float] = []
    for b in sorted_batches:
        dominated = False
        b_mask = b.item_mask
        b_span = b.span
        b_delta = b.delta
        for k in range(len(kept)):
            if (
                kept_span[k] <= b_span
                and (kept_mask[k] & b_mask) == kept_mask[k]
                and kept_delta[k] <= b_delta + tol
            ):
                dominated = True
                break
        if not dominated:
            kept.append(b)
            kept_span.append(b_span)
            kept_mask.append(b_mask)
            kept_delta.append(b_delta)
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


_ROUTE_SHIFT = 48
_COMPILED_ROUTE_FAILED = False
_MASK_DOMINANCE = True


def _compiled_route_supported(batches_by_t: list[list[Batch]], num_t: int, start_mask: int) -> bool:
    if num_t >= (1 << (64 - _ROUTE_SHIFT)) or start_mask.bit_length() > _ROUTE_SHIFT:
        return False
    for batches in batches_by_t:
        for batch in batches:
            if batch.item_mask.bit_length() > _ROUTE_SHIFT:
                return False
    return True


@njit(cache=True, nogil=True)
def _route_allows(used, batch_mask, rf_differ, rf_same):
    new_mask = used | batch_mask
    for k in range(rf_differ.shape[0]):
        if (new_mask & rf_differ[k, 0]) and (new_mask & rf_differ[k, 1]):
            return False
    if batch_mask == 0:
        for k in range(rf_same.shape[0]):
            has_i = (new_mask & rf_same[k, 0]) != 0
            has_j = (new_mask & rf_same[k, 1]) != 0
            if has_i != has_j:
                return False
    return True


@njit(cache=True, nogil=True)
def _route_sri_cost(used, batch_mask, sri_masks, sri_pi):
    n = sri_masks.shape[0]
    if n == 0:
        return 0.0
    nxt = used | batch_mask
    total = 0.0
    one = np.uint64(1)
    for k in range(n):
        item_mask = sri_masks[k]
        before = 0
        after = 0
        left = used & item_mask
        right = nxt & item_mask
        while left:
            left &= left - one
            before += 1
        while right:
            right &= right - one
            after += 1
        total += -sri_pi[k] * ((after // 2) - (before // 2))
    return total


@njit(cache=True, nogil=True)
def _route_idle_target(t_idx, next_event, forced_times):
    nxt = next_event[t_idx]
    for k in range(forced_times.shape[0]):
        ft = forced_times[k]
        if t_idx <= ft < nxt:
            return ft
    return nxt


@njit(cache=True, nogil=True)
def _route_search(
    deltas,
    masks,
    spans,
    u_idxs,
    offsets,
    next_event,
    lb_sum,
    forced_u,
    forced_times,
    rf_differ,
    rf_same,
    sri_masks,
    sri_pi,
    tol,
    stop,
    start_t,
    start_mask,
    memo_cost,
    memo_dec,
    counter,
    hits,
    timed,
):
    """Iterative exact DP. A timed-out branch returns +inf and is not cached."""
    num_t = forced_u.shape[0]
    shift = np.uint64(48)
    cap = num_t + 3
    st_t = np.empty(cap, dtype=np.int64)
    st_used = np.empty(cap, dtype=np.uint64)
    st_phase = np.empty(cap, dtype=np.int64)
    st_cursor = np.empty(cap, dtype=np.int64)
    st_best = np.empty(cap, dtype=np.float64)
    st_kind = np.empty(cap, dtype=np.uint64)
    st_next = np.empty(cap, dtype=np.uint64)
    st_batch = np.empty(cap, dtype=np.uint64)
    st_wait = np.empty(cap, dtype=np.int8)
    st_launch_et = np.empty(cap, dtype=np.int64)
    st_launch_bi = np.empty(cap, dtype=np.int64)
    res = np.empty(cap, dtype=np.float64)
    st_t[0] = start_t
    st_used[0] = start_mask
    st_phase[0] = 0
    sp = 0
    rp = 0

    while sp >= 0:
        phase = st_phase[sp]
        t_idx = st_t[sp]
        used = st_used[sp]
        if phase == 0:
            counter[0] += 1
            if (counter[0] & 63) == 0 and stop[0] != 0:
                timed[0] = 1
                res[rp] = np.inf
                rp += 1
                sp -= 1
                continue
            if t_idx >= num_t:
                if _route_allows(used, np.uint64(0), rf_differ, rf_same):
                    res[rp] = 0.0
                else:
                    res[rp] = np.inf
                rp += 1
                sp -= 1
                continue
            key = (np.uint64(t_idx) << shift) | used
            if key in memo_cost:
                hits[0] += 1
                res[rp] = memo_cost[key]
                rp += 1
                sp -= 1
                continue
            if forced_u[t_idx] >= 0:
                st_phase[sp] = 2
                st_cursor[sp] = offsets[t_idx]
                st_best[sp] = np.inf
                st_kind[sp] = 0
                st_next[sp] = 0
                st_batch[sp] = 0
                st_wait[sp] = 0
            else:
                st_phase[sp] = 1
                child = sp + 1
                st_t[child] = _route_idle_target(t_idx, next_event, forced_times)
                st_used[child] = used
                st_phase[child] = 0
                sp = child
            continue

        if phase == 1:
            rp -= 1
            st_best[sp] = res[rp]
            st_kind[sp] = 1
            st_next[sp] = np.uint64(_route_idle_target(t_idx, next_event, forced_times))
            st_batch[sp] = 0
            st_phase[sp] = 2
            st_cursor[sp] = offsets[t_idx]
            st_wait[sp] = 0
            continue

        end = offsets[t_idx + 1]
        u_req = forced_u[t_idx]
        if st_wait[sp]:
            rp -= 1
            launched_bi = st_launch_bi[sp]
            cand = (
                deltas[launched_bi]
                + _route_sri_cost(used, masks[launched_bi], sri_masks, sri_pi)
                + res[rp]
            )
            st_wait[sp] = 0
            if cand < st_best[sp] - tol:
                st_best[sp] = cand
                st_kind[sp] = 2
                st_next[sp] = np.uint64(st_launch_et[sp])
                st_batch[sp] = np.uint64(st_launch_bi[sp])
            st_cursor[sp] += 1

        cursor = st_cursor[sp]
        launched = False
        while cursor < end:
            bi = cursor
            if u_req >= 0 and u_idxs[bi] != u_req:
                cursor += 1
                continue
            batch_mask = masks[bi]
            if batch_mask & used:
                cursor += 1
                continue
            if not _route_allows(used, batch_mask, rf_differ, rf_same):
                cursor += 1
                continue
            et = t_idx + spans[bi]
            if et > num_t:
                cursor += 1
                continue
            crossed = False
            for fk in range(forced_times.shape[0]):
                ft = forced_times[fk]
                if t_idx < ft < et:
                    crossed = True
                    break
            if crossed or deltas[bi] + lb_sum[et] >= st_best[sp] - tol:
                cursor += 1
                continue
            st_cursor[sp] = cursor
            st_wait[sp] = 1
            st_launch_et[sp] = et
            st_launch_bi[sp] = bi
            child = sp + 1
            st_t[child] = et
            st_used[child] = used | batch_mask
            st_phase[child] = 0
            sp = child
            launched = True
            break
        if launched:
            continue
        key = (np.uint64(t_idx) << shift) | used
        best = st_best[sp]
        memo_cost[key] = best
        memo_dec[key] = (st_kind[sp] << np.uint64(48)) | (st_next[sp] << np.uint64(32)) | st_batch[sp]
        res[rp] = best
        rp += 1
        sp -= 1

    if rp <= 0:
        return np.inf
    return res[rp - 1]


def _pairs_to_array(pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = np.zeros((len(pairs), 2), dtype=np.uint64)
    for i, (a, b) in enumerate(pairs):
        out[i, 0] = a
        out[i, 1] = b
    return out


def _try_compiled_route(
    batches_by_t: list[list[Batch]],
    num_t: int,
    *,
    deadline: float | None,
    start_t: int,
    start_mask: int,
    forced_at: dict[int, int],
    sri_terms: tuple[tuple[int, float], ...],
    rf_differ: tuple[tuple[int, int], ...],
    rf_same: tuple[tuple[int, int], ...],
    next_event: list[int],
    lb_sum: list[float],
) -> tuple[float, dict[tuple[int, int], _MemoEntry], int, int, bool] | None:
    """Compiled exact route. None means the Python recursion must be used."""
    global _COMPILED_ROUTE_FAILED
    if _COMPILED_ROUTE_FAILED or not _compiled_route_supported(batches_by_t, num_t, start_mask):
        return None
    flat: list[Batch] = []
    for batches in batches_by_t:
        flat.extend(batches)
    for batch in flat:
        if batch.item_mask < 0:
            return None
    for pairs in (rf_differ, rf_same):
        for a, b in pairs:
            if a < 0 or b < 0 or a.bit_length() > _ROUTE_SHIFT or b.bit_length() > _ROUTE_SHIFT:
                return None
    n = len(flat)
    deltas = np.empty(n, dtype=np.float64)
    masks = np.empty(n, dtype=np.uint64)
    spans = np.empty(n, dtype=np.int64)
    u_idxs = np.empty(n, dtype=np.int64)
    offsets = np.empty(num_t + 1, dtype=np.int64)
    offsets[0] = 0
    cursor = 0
    for t, batches in enumerate(batches_by_t):
        for batch in batches:
            deltas[cursor] = batch.delta
            masks[cursor] = batch.item_mask
            spans[cursor] = batch.span
            u_idxs[cursor] = batch.u_idx
            cursor += 1
        offsets[t + 1] = cursor
    forced_u = np.full(num_t, -1, dtype=np.int64)
    for t, u in forced_at.items():
        if 0 <= t < num_t:
            forced_u[t] = u
    forced_times = np.asarray(sorted(forced_at), dtype=np.int64)
    sri_masks = np.asarray([m for m, _pi in sri_terms], dtype=np.uint64)
    sri_pi = np.asarray([pi for _m, pi in sri_terms], dtype=np.float64)
    from numba.typed import Dict
    from numba import types

    memo_cost = Dict.empty(types.uint64, types.float64)
    memo_dec = Dict.empty(types.uint64, types.uint64)
    counter = np.zeros(1, dtype=np.int64)
    hits = np.zeros(1, dtype=np.int64)
    timed = np.zeros(1, dtype=np.int8)
    stop = np.zeros(1, dtype=np.int8)
    timer = None
    if deadline is not None:
        delay = max(0.0, float(deadline) - time.perf_counter())

        def _trip_deadline(flag=stop) -> None:
            flag[0] = 1

        timer = threading.Timer(delay, _trip_deadline)
        timer.daemon = True
        timer.start()
    try:
        cost = _route_search(
            deltas,
            masks,
            spans,
            u_idxs,
            offsets,
            np.asarray(next_event, dtype=np.int64),
            np.asarray(lb_sum, dtype=np.float64),
            forced_u,
            forced_times,
            _pairs_to_array(rf_differ),
            _pairs_to_array(rf_same),
            sri_masks,
            sri_pi,
            float(CHECKER_TOL["pricing_dp"]),
            stop,
            int(start_t),
            np.uint64(start_mask),
            memo_cost,
            memo_dec,
            counter,
            hits,
            timed,
        )
    except Exception as exc:
        _COMPILED_ROUTE_FAILED = True
        print(f"[EBRP] compiled routing disabled: {exc}", file=sys.stderr, flush=True)
        return None
    finally:
        if timer is not None:
            timer.cancel()
    memo: dict[tuple[int, int], _MemoEntry] = {}
    t_idx = start_t
    used = start_mask
    for _guard in range(num_t + 1):
        if t_idx >= num_t:
            break
        key = (np.uint64(t_idx) << np.uint64(_ROUTE_SHIFT)) | np.uint64(used)
        if key not in memo_dec:
            break
        dec = int(memo_dec[key])
        kind_i = dec >> 48
        next_t = (dec >> 32) & 0xFFFF
        batch_i = dec & 0xFFFFFFFF
        if kind_i == 2 and batch_i < len(flat):
            batch = flat[batch_i]
            memo[(t_idx, used)] = _MemoEntry(float(cost), "batch", next_t=next_t, batch=batch)
            used |= batch.item_mask
            t_idx = next_t
        elif kind_i == 1:
            memo[(t_idx, used)] = _MemoEntry(float(cost), "idle", next_t=next_t)
            t_idx = next_t
        else:
            memo[(t_idx, used)] = _MemoEntry(float(cost), "end")
            break
    return float(cost), memo, int(counter[0]), int(hits[0]), bool(timed[0])


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
    rf_differ: tuple[tuple[int, int], ...] = (),
    rf_same: tuple[tuple[int, int], ...] = (),
) -> tuple[float, dict[tuple[int, int], _MemoEntry]]:
    """记忆化 DP 路由（递归，精确）。

    状态：(t, used_mask)，t 为当前时刻，used_mask 为已用物料位掩码。
    forced_at: 若给出 {t: u}，则时刻 t 必须开工配置 u（Y=1 分支），不可空闲跳过。
    rf_differ / rf_same: Ryan–Foster bit-pairs (bit_i, bit_j).
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
    _rf_differ = rf_differ
    _rf_same = rf_same

    def _sri_transition_cost(used_mask: int, batch_mask: int) -> float:
        """Reduced-cost contribution of rank-1 SRI duals for one new batch."""
        next_mask = used_mask | batch_mask
        return sum(
            -pi * (((next_mask & item_mask).bit_count() // 2) - ((used_mask & item_mask).bit_count() // 2))
            for item_mask, pi in _sri_terms
        )

    def _rf_allows(used_mask: int, batch_mask: int = 0) -> bool:
        new_mask = used_mask | batch_mask
        for bi, bj in _rf_differ:
            if (new_mask & bi) and (new_mask & bj):
                return False
        if batch_mask == 0:
            for bi, bj in _rf_same:
                has_i = bool(new_mask & bi)
                has_j = bool(new_mask & bj)
                if has_i != has_j:
                    return False
        return True

    flat_batches = [b for batches in _batches_by_t for b in batches]
    offsets = np.zeros(_num_t + 1, dtype=np.int64)
    for t in range(_num_t):
        offsets[t + 1] = offsets[t] + len(_batches_by_t[t])
    lb_sum = _routing_lower_bounds(
        np.asarray([b.delta for b in flat_batches], dtype=np.float64),
        np.asarray([b.span for b in flat_batches], dtype=np.int64),
        np.asarray([b.u_idx for b in flat_batches], dtype=np.int64),
        offsets, _num_t, _forced_at, _sri_terms,
    )

    if progress_interval_sec <= 0:
        compiled = _try_compiled_route(
            batches_by_t,
            num_t,
            deadline=deadline,
            start_t=start_t,
            start_mask=start_mask,
            forced_at=_forced_at,
            sri_terms=_sri_terms,
            rf_differ=_rf_differ,
            rf_same=_rf_same,
            next_event=next_event,
            lb_sum=lb_sum,
        )
        if compiled is not None:
            cost, memo_out, nodes, hits, timed_out = compiled
            st.nodes_expanded += nodes
            st.memo_hits += hits
            if timed_out:
                st.timed_out = True
                st.exact = False
            return cost, memo_out

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
            return 0.0 if _rf_allows(used_mask) else math.inf

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
                if not _rf_allows(used_mask, b.item_mask):
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
            if not _rf_allows(used_mask, b.item_mask):
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

def _materialize_route(ctx: EBRPContext, flat: _FlatRoute) -> list[list[Batch]]:
    num_t = int(flat.offsets.shape[0]) - 1
    by_t: list[list[Batch]] = [[] for _ in range(num_t)]
    for i in range(flat.delta.shape[0]):
        spec = int(flat.spec_idx[i])
        sub = int(flat.sub_idx[i])
        subset = ctx._subset_cache[spec][sub][0]
        t_idx = int(flat.t_idx[i])
        by_t[t_idx].append(
            Batch(
                u_idx=int(flat.u_idx[i]),
                t_idx=t_idx,
                span=int(flat.span[i]),
                items=subset,
                delta=float(flat.delta[i]),
                item_mask=int(flat.mask[i]),
            )
        )
    return by_t


def _batch_on_path(ctx: EBRPContext, flat: _FlatRoute, batch_i: int) -> Batch:
    spec = int(flat.spec_idx[batch_i])
    sub = int(flat.sub_idx[batch_i])
    subset = ctx._subset_cache[spec][sub][0]
    t_idx = int(flat.t_idx[batch_i])
    return Batch(
        u_idx=int(flat.u_idx[batch_i]),
        t_idx=t_idx,
        span=int(flat.span[batch_i]),
        items=subset,
        delta=float(flat.delta[batch_i]),
        item_mask=int(flat.mask[batch_i]),
    )


def _route_flat(
    ctx: EBRPContext,
    flat: _FlatRoute,
    num_t: int,
    *,
    deadline: float | None,
    st: EBRPStats,
    progress_interval_sec: float,
    machine: str,
    forced_at: dict[int, int] | None,
    sri_terms: tuple[tuple[int, float], ...],
    rf_differ: tuple[tuple[int, int], ...],
    rf_same: tuple[tuple[int, int], ...],
) -> tuple[float, dict[tuple[int, int], _MemoEntry]]:
    """Route on the array catalog. Batch objects are built only if the compiled search cannot run."""
    forced = forced_at or {}
    next_event = [num_t] * (num_t + 1)
    nxt = num_t
    for t_idx in range(num_t - 1, -1, -1):
        next_event[t_idx] = nxt
        if int(flat.offsets[t_idx + 1]) > int(flat.offsets[t_idx]):
            nxt = t_idx
    lb_sum = _routing_lower_bounds(
        flat.delta, flat.span, flat.u_idx, flat.offsets, num_t, forced, sri_terms,
    )

    # SRI increments depend on the complete used set; retain the full-state
    # exact oracle there. RF endpoints, by contrast, are tracked by DSSR.
    if not sri_terms:
        result = solve_dssr(
            flat.delta, flat.mask, flat.span, flat.u_idx, flat.offsets, num_t,
            forced_at=forced, rf_differ=rf_differ, rf_same=rf_same,
            lower_bounds=lb_sum, deadline=deadline,
        )
        with _EBRP_TIME_LOCK:
            EBRP_TIME["dssr_calls"] += 1
            if result is None:
                EBRP_TIME["dssr_fallbacks"] += 1
            else:
                EBRP_TIME["dssr_rounds"] += result.rounds
                EBRP_TIME["dssr_nodes"] += result.nodes
        if result is not None:
            st.nodes_expanded += result.nodes
            st.dssr_rounds += result.rounds
            st.critical_items = max(st.critical_items, result.critical_count)
            st.exact = st.exact and result.exact
            st.timed_out = st.timed_out or result.timed_out
            memo: dict[tuple[int, int], _MemoEntry] = {}
            t_idx = used = 0
            for batch_i in result.path:
                batch = _batch_on_path(ctx, flat, batch_i)
                if batch.t_idx > t_idx:
                    memo[t_idx, used] = _MemoEntry(result.cost, "idle", next_t=batch.t_idx)
                    t_idx = batch.t_idx
                end = t_idx + batch.span
                memo[t_idx, used] = _MemoEntry(result.cost, "batch", next_t=end, batch=batch)
                used |= batch.item_mask
                t_idx = end
            memo[t_idx, used] = _MemoEntry(0.0, "end")
            return result.cost, memo

    if progress_interval_sec > 0:
        return _solve_routing(
            _materialize_route(ctx, flat), num_t, deadline=deadline, st=st,
            progress_interval_sec=progress_interval_sec, machine=machine,
            forced_at=forced, sri_terms=sri_terms, rf_differ=rf_differ, rf_same=rf_same,
        )

    compiled = _compiled_on_flat(
        ctx,
        flat,
        num_t,
        deadline=deadline,
        start_t=0,
        start_mask=0,
        forced_at=forced,
        sri_terms=sri_terms,
        rf_differ=rf_differ,
        rf_same=rf_same,
        next_event=next_event,
        lb_sum=lb_sum,
    )
    if compiled is None:
        return _solve_routing(
            _materialize_route(ctx, flat),
            num_t,
            deadline=deadline,
            st=st,
            progress_interval_sec=0.0,
            machine=machine,
            forced_at=forced_at,
            sri_terms=sri_terms,
            rf_differ=rf_differ,
            rf_same=rf_same,
        )
    cost, memo, nodes, hits, timed_out = compiled
    st.nodes_expanded += nodes
    st.memo_hits += hits
    if timed_out:
        st.timed_out = True
        st.exact = False
    return cost, memo


@njit(cache=True, nogil=True)
def _forward_search(
    deltas,
    masks,
    spans,
    u_idxs,
    offsets,
    next_event,
    lb_sum,
    forced_u,
    forced_times,
    rf_differ,
    rf_same,
    same_bits,
    subset_dom,
    tol,
    stop,
    num_t,
):
    """Forward exact DP.

    A label is dropped only when another label at the same time has used a
    subset of its items, costs no more, and the extra items are not part of a
    Ryan–Foster same-pair.  Identical masks are always reduced to the cheaper
    one.  Large buckets skip the subset test and keep every distinct mask.
    """
    cap = 250000
    lab_t = np.empty(cap, dtype=np.int64)
    lab_mask = np.empty(cap, dtype=np.uint64)
    lab_g = np.empty(cap, dtype=np.float64)
    lab_pred = np.empty(cap, dtype=np.int32)
    lab_kind = np.empty(cap, dtype=np.int8)
    lab_batch = np.empty(cap, dtype=np.int32)
    lab_next = np.empty(cap, dtype=np.int64)
    lab_alive = np.empty(cap, dtype=np.uint8)
    bucket_head = np.full(num_t + 1, -1, dtype=np.int32)
    bucket_next = np.empty(cap, dtype=np.int32)
    lab_t[0] = 0
    lab_mask[0] = 0
    lab_g[0] = 0.0
    lab_pred[0] = -1
    lab_kind[0] = 0
    lab_batch[0] = -1
    lab_next[0] = 0
    lab_alive[0] = 1
    bucket_head[0] = 0
    bucket_next[0] = -1
    nlab = 1
    best_g = 1e300
    best_from = -1
    best_kind = np.int8(1)
    best_batch = np.int32(-1)
    best_next = num_t
    found = 0
    overflow = 0
    counter = 0
    ids = np.empty(cap, dtype=np.int32)

    for t_idx in range(num_t):
        if stop[0] != 0:
            break
        ids_n = 0
        j = bucket_head[t_idx]
        while j >= 0:
            if lab_alive[j] != 0:
                ids[ids_n] = j
                ids_n += 1
            j = bucket_next[j]
        if ids_n <= 4000:
            for a in range(ids_n):
                ia = ids[a]
                if lab_alive[ia] == 0:
                    continue
                ma = lab_mask[ia]
                ga = lab_g[ia]
                for b in range(ids_n):
                    if a == b:
                        continue
                    ib = ids[b]
                    if lab_alive[ib] == 0:
                        continue
                    mb = lab_mask[ib]
                    gb = lab_g[ib]
                    if gb > ga + tol or (mb & ma) != mb:
                        continue
                    if mb == ma:
                        if gb < ga or ib < ia:
                            lab_alive[ia] = 0
                            break
                    elif subset_dom != 0 and ids_n <= 800 and ((ma ^ mb) & same_bits) == 0:
                        lab_alive[ia] = 0
                        break
        begin = offsets[t_idx]
        end = offsets[t_idx + 1]
        u_req = forced_u[t_idx]
        for a in range(ids_n):
            ia = ids[a]
            if lab_alive[ia] == 0:
                continue
            counter += 1
            if (counter & 63) == 0 and stop[0] != 0:
                break
            used = lab_mask[ia]
            g = lab_g[ia]
            if u_req < 0:
                nxt = _route_idle_target(t_idx, next_event, forced_times)
                if nxt >= num_t:
                    if g < best_g and _route_allows(used, np.uint64(0), rf_differ, rf_same):
                        best_g = g
                        best_from = ia
                        best_kind = np.int8(1)
                        best_batch = np.int32(-1)
                        best_next = nxt
                        found = 1
                elif nlab < cap:
                    lab_t[nlab] = nxt
                    lab_mask[nlab] = used
                    lab_g[nlab] = g
                    lab_pred[nlab] = ia
                    lab_kind[nlab] = 1
                    lab_batch[nlab] = -1
                    lab_next[nlab] = nxt
                    lab_alive[nlab] = 1
                    bucket_next[nlab] = bucket_head[nxt]
                    bucket_head[nxt] = nlab
                    nlab += 1
                else:
                    overflow = 1
            for bi in range(begin, end):
                if u_req >= 0 and u_idxs[bi] != u_req:
                    continue
                batch_mask = masks[bi]
                if (batch_mask & used) != 0:
                    continue
                if not _route_allows(used, batch_mask, rf_differ, rf_same):
                    continue
                et = t_idx + spans[bi]
                if et > num_t:
                    continue
                crossed = False
                for fk in range(forced_times.shape[0]):
                    ft = forced_times[fk]
                    if t_idx < ft < et:
                        crossed = True
                        break
                if crossed:
                    continue
                g2 = g + deltas[bi]
                if g2 + lb_sum[et] >= best_g - tol:
                    continue
                if et >= num_t:
                    new_mask = used | batch_mask
                    if g2 < best_g and _route_allows(new_mask, np.uint64(0), rf_differ, rf_same):
                        best_g = g2
                        best_from = ia
                        best_kind = np.int8(2)
                        best_batch = np.int32(bi)
                        best_next = et
                        found = 1
                    continue
                if nlab >= cap:
                    overflow = 1
                    continue
                new_mask = used | batch_mask
                lab_t[nlab] = et
                lab_mask[nlab] = new_mask
                lab_g[nlab] = g2
                lab_pred[nlab] = ia
                lab_kind[nlab] = 2
                lab_batch[nlab] = bi
                lab_next[nlab] = et
                lab_alive[nlab] = 1
                bucket_next[nlab] = bucket_head[et]
                bucket_head[et] = nlab
                nlab += 1
        if overflow != 0:
            break
    return (
        found,
        overflow,
        best_g,
        best_from,
        best_kind,
        best_batch,
        best_next,
        counter,
        nlab,
        lab_pred,
        lab_kind,
        lab_batch,
        lab_next,
        lab_t,
        lab_mask,
    )


def _same_pair_bits(rf_same: tuple[tuple[int, int], ...]) -> int:
    bits = 0
    for left, right in rf_same:
        bits |= int(left) | int(right)
    return bits


def _forward_route(
    ctx: EBRPContext,
    flat: _FlatRoute,
    num_t: int,
    *,
    deadline: float | None,
    forced_at: dict[int, int],
    rf_differ: tuple[tuple[int, int], ...],
    rf_same: tuple[tuple[int, int], ...],
    subset_dom: bool,
    next_event: list[int],
    lb_sum: list[float],
) -> tuple[float, dict[tuple[int, int], _MemoEntry], int, int, bool] | None:
    """Exact forward route with subset dominance. None asks the caller to use the plain DP."""
    stop = np.zeros(1, dtype=np.int8)
    timer = None
    if deadline is not None:
        delay = max(0.0, float(deadline) - time.perf_counter())

        def _trip(flag=stop) -> None:
            flag[0] = 1

        timer = threading.Timer(delay, _trip)
        timer.daemon = True
        timer.start()
    forced_u = np.full(num_t, -1, dtype=np.int64)
    for t_idx, u_idx in forced_at.items():
        if 0 <= t_idx < num_t:
            forced_u[t_idx] = u_idx
    try:
        (
            found,
            overflow,
            best_g,
            best_from,
            best_kind,
            best_batch,
            best_next,
            counter,
            _nlab,
            lab_pred,
            lab_kind,
            lab_batch,
            lab_next,
            _lab_t,
            _lab_mask,
        ) = _forward_search(
            flat.delta,
            flat.mask,
            flat.span,
            flat.u_idx,
            flat.offsets,
            np.asarray(next_event, dtype=np.int64),
            np.asarray(lb_sum, dtype=np.float64),
            forced_u,
            np.asarray(sorted(forced_at), dtype=np.int64),
            _pairs_to_array(rf_differ),
            _pairs_to_array(rf_same),
            np.uint64(_same_pair_bits(rf_same)),
            np.uint8(1 if subset_dom else 0),
            float(CHECKER_TOL["pricing_dp"]),
            stop,
            num_t,
        )
    except Exception as exc:
        print(f"[EBRP] forward routing disabled: {exc}", file=sys.stderr, flush=True)
        return None
    finally:
        if timer is not None:
            timer.cancel()
    if overflow:
        return None
    if not found:
        return float("inf"), {}, int(counter), 0, bool(stop[0])
    chain: list[int] = []
    cursor = int(best_from)
    while cursor >= 0:
        chain.append(cursor)
        cursor = int(lab_pred[cursor])
    chain.reverse()
    memo: dict[tuple[int, int], _MemoEntry] = {}
    t_idx = 0
    used = 0
    for label in chain[1:]:
        kind_i = int(lab_kind[label])
        next_t = int(lab_next[label])
        if kind_i == 2:
            batch = _batch_on_path(ctx, flat, int(lab_batch[label]))
            memo[(t_idx, used)] = _MemoEntry(float(best_g), "batch", next_t=next_t, batch=batch)
            used |= batch.item_mask
            t_idx = next_t
        else:
            memo[(t_idx, used)] = _MemoEntry(float(best_g), "idle", next_t=next_t)
            t_idx = next_t
    if int(best_kind) == 2:
        batch = _batch_on_path(ctx, flat, int(best_batch))
        memo[(t_idx, used)] = _MemoEntry(float(best_g), "batch", next_t=int(best_next), batch=batch)
    else:
        memo[(t_idx, used)] = _MemoEntry(float(best_g), "idle", next_t=int(best_next))
    return float(best_g), memo, int(counter), 0, bool(stop[0])


def _compiled_on_flat(
    ctx: EBRPContext,
    flat: _FlatRoute,
    num_t: int,
    *,
    deadline: float | None,
    start_t: int,
    start_mask: int,
    forced_at: dict[int, int],
    sri_terms: tuple[tuple[int, float], ...],
    rf_differ: tuple[tuple[int, int], ...],
    rf_same: tuple[tuple[int, int], ...],
    next_event: list[int],
    lb_sum: list[float],
) -> tuple[float, dict[tuple[int, int], _MemoEntry], int, int, bool] | None:
    global _COMPILED_ROUTE_FAILED
    if _COMPILED_ROUTE_FAILED or num_t >= (1 << (64 - _ROUTE_SHIFT)) or start_mask.bit_length() > _ROUTE_SHIFT:
        return None
    if flat.mask.size and int(flat.mask.max()) >= (1 << _ROUTE_SHIFT):
        return None
    for pairs in (rf_differ, rf_same):
        for a, b in pairs:
            if a < 0 or b < 0 or a.bit_length() > _ROUTE_SHIFT or b.bit_length() > _ROUTE_SHIFT:
                return None
    from numba.typed import Dict
    from numba import types

    memo_cost = Dict.empty(types.uint64, types.float64)
    memo_dec = Dict.empty(types.uint64, types.uint64)
    counter = np.zeros(1, dtype=np.int64)
    hits = np.zeros(1, dtype=np.int64)
    timed = np.zeros(1, dtype=np.int8)
    stop = np.zeros(1, dtype=np.int8)
    timer = None
    if deadline is not None:
        delay = max(0.0, float(deadline) - time.perf_counter())

        def _trip_deadline(flag=stop) -> None:
            flag[0] = 1

        timer = threading.Timer(delay, _trip_deadline)
        timer.daemon = True
        timer.start()
    forced_u = np.full(num_t, -1, dtype=np.int64)
    for t_idx, u_idx in forced_at.items():
        if 0 <= t_idx < num_t:
            forced_u[t_idx] = u_idx
    try:
        if _MASK_DOMINANCE:
            forwarded = _forward_route(
                ctx,
                flat,
                num_t,
                deadline=deadline,
                forced_at=forced_at,
                rf_differ=rf_differ,
                rf_same=rf_same,
                subset_dom=len(sri_terms) == 0,
                next_event=next_event,
                lb_sum=lb_sum,
            )
            if forwarded is not None:
                return forwarded
        cost = _route_search(
            flat.delta,
            flat.mask,
            flat.span,
            flat.u_idx,
            flat.offsets,
            np.asarray(next_event, dtype=np.int64),
            np.asarray(lb_sum, dtype=np.float64),
            forced_u,
            np.asarray(sorted(forced_at), dtype=np.int64),
            _pairs_to_array(rf_differ),
            _pairs_to_array(rf_same),
            np.asarray([m for m, _pi in sri_terms], dtype=np.uint64),
            np.asarray([pi for _m, pi in sri_terms], dtype=np.float64),
            float(CHECKER_TOL["pricing_dp"]),
            stop,
            int(start_t),
            np.uint64(start_mask),
            memo_cost,
            memo_dec,
            counter,
            hits,
            timed,
        )
    except Exception as exc:
        _COMPILED_ROUTE_FAILED = True
        print(f"[EBRP] compiled routing disabled: {exc}", file=sys.stderr, flush=True)
        return None
    finally:
        if timer is not None:
            timer.cancel()
    memo: dict[tuple[int, int], _MemoEntry] = {}
    t_idx = start_t
    used = start_mask
    n_batch = int(flat.delta.shape[0])
    for _guard in range(num_t + 1):
        if t_idx >= num_t:
            break
        key = (np.uint64(t_idx) << np.uint64(_ROUTE_SHIFT)) | np.uint64(used)
        if key not in memo_dec:
            break
        dec = int(memo_dec[key])
        kind_i = dec >> 48
        next_t = (dec >> 32) & 0xFFFF
        batch_i = dec & 0xFFFFFFFF
        if kind_i == 2 and batch_i < n_batch:
            batch = _batch_on_path(ctx, flat, batch_i)
            memo[(t_idx, used)] = _MemoEntry(float(cost), "batch", next_t=next_t, batch=batch)
            used |= batch.item_mask
            t_idx = next_t
        elif kind_i == 1:
            memo[(t_idx, used)] = _MemoEntry(float(cost), "idle", next_t=next_t)
            t_idx = next_t
        else:
            memo[(t_idx, used)] = _MemoEntry(float(cost), "end")
            break
    return float(cost), memo, int(counter[0]), int(hits[0]), bool(timed[0])


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
    fixed_rf: dict[tuple[int, int], int] | None = None,
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
    catalog_sec = 0.0
    flat_route: _FlatRoute | None = None
    batches_by_t: list[list[Batch]] | None = None
    if catalog is None or forced_at or fixed_rf:
        catalog_t0 = time.perf_counter()
        flat_route = ctx.score_route_batches(
            data,
            alpha,
            rho_val,
            alpha_flat=alpha_flat,
            num_t=num_t_for_flat or num_t,
            deadline=deadline,
            setup_cost_delta=setup_cost_delta,
            required_setups={(u, t) for t, u in forced_at.items()},
            forbidden=forbidden,
            preserve_all=bool(forced_at or any(val == 1 for val in (fixed_rf or {}).values())),
        )
        catalog_sec = time.perf_counter() - catalog_t0
        if deadline is not None and time.perf_counter() >= deadline:
            _add_ebrp_time(catalog_sec, 0.0)
            st.timed_out = True
            st.exact = False
            return SubproblemResult(
                reduced_cost=float("inf"),
                x={},
                y={},
                production_cost=0.0,
                is_optimal=False,
            )
    else:
        batches_by_t = _index_batches_by_t(catalog, num_u, num_t, forbidden)

    if progress_interval_sec > 0:
        total_batches = (
            int(flat_route.delta.shape[0])
            if flat_route is not None
            else sum(len(bt) for bt in batches_by_t or [])
        )
        print(
            f"[EBRP {machine}] search start  pools={len(ctx.pool_specs)}  "
            f"batches={total_batches}  events={len(ctx.event_times)}",
            flush=True,
            file=sys.stderr,
        )

    rf_differ: list[tuple[int, int]] = []
    rf_same: list[tuple[int, int]] = []
    if fixed_rf:
        for (i, j), val in fixed_rf.items():
            # Batch masks use machine-local indices. A missing item has
            # bit zero; a same-pair then forbids its present partner.
            local_i = ctx.global_to_local.get(i)
            local_j = ctx.global_to_local.get(j)
            pair = (
                0 if local_i is None else 1 << local_i,
                0 if local_j is None else 1 << local_j,
            )
            if val == 0:
                rf_differ.append(pair)
            elif val == 1:
                rf_same.append(pair)

    route_t0 = time.perf_counter()
    if flat_route is not None:
        batch_cost, memo = _route_flat(
            ctx,
            flat_route,
            num_t,
            deadline=deadline,
            st=st,
            progress_interval_sec=progress_interval_sec,
            machine=machine,
            forced_at=forced_at or None,
            sri_terms=sri_terms,
            rf_differ=tuple(rf_differ),
            rf_same=tuple(rf_same),
        )
    else:
        batch_cost, memo = _solve_routing(
            batches_by_t or [],
            num_t,
            deadline=deadline,
            st=st,
            progress_interval_sec=progress_interval_sec,
            machine=machine,
            forced_at=forced_at or None,
            sri_terms=sri_terms,
            rf_differ=tuple(rf_differ),
            rf_same=tuple(rf_same),
        )
    _add_ebrp_time(catalog_sec, time.perf_counter() - route_t0)
    if st.timed_out and not math.isfinite(batch_cost):
        batch_cost = math.inf

    reduced_cost = batch_cost - beta if math.isfinite(batch_cost) else float("inf")
    x, y, production_cost = _decode_plan(memo, 0, 0, num_t, p_cum)

    # Gurobi 交叉验证（仅在 EBRP 认为无负 RC 列时，兜底防止实现瑕疵）
    if not skip_cross_check and not sri_terms and reduced_cost > -1e-3 and not fixed_y and not fixed_rf and not st.timed_out:
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
