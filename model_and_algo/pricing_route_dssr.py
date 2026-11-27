"""Exact selective-elementarity routing on a time-indexed batch catalog.

Only the items needed to exclude repeated use are initially remembered.  An
optimal relaxed path is inspected and every repeated item is added to that
memory.  A relaxed optimum without repeated items is an optimum of the original
pricing problem.  Budget exhaustion returns ``None`` for a full-memory fallback;
an incomplete relaxation is never reported as an exact pricing certificate.

The caller must supply a catalog whose filtering is valid under its branch
constraints.  In particular, RF-same and forced setups may require positive-cost
batches that ordinary root pricing can discard.  SRI transition costs are not
handled by this module; callers with those terms should use their full-state DP.
"""

from __future__ import annotations

import bisect
import functools
import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DSSRResult:
    cost: float
    path: list[int]
    nodes: int
    rounds: int
    critical_count: int
    exact: bool = True
    timed_out: bool = False


class _DSSRLimit(Exception):
    pass


def solve_dssr(
    deltas: Sequence[float],
    masks: Sequence[int],
    spans: Sequence[int],
    u_idxs: Sequence[int],
    offsets: Sequence[int],
    num_t: int,
    *,
    forced_at: Mapping[int, int] | None = None,
    rf_differ: Sequence[tuple[int, int]] = (),
    rf_same: Sequence[tuple[int, int]] = (),
    lower_bounds: Sequence[float] | None = None,
    deadline: float | None = None,
    max_states: int = 100_000,
) -> DSSRResult | None:
    """Return an exact route, exact infeasibility, or ``None`` for fallback.

    Times and flat batch indices are zero based.  ``rf_*`` pairs contain item
    bit masks, including zero for an endpoint not eligible on this machine.
    RF endpoints are remembered from the first relaxation, so grouping batches
    by their remembered items preserves every branch restriction.  Forced setup
    times are enforced on every relaxation, including idle transitions.

    ``lower_bounds[t]`` must lower-bound every feasible residual route at time
    ``t``.  Without one, a time-DAG bound is computed from this catalog.  The
    state cap counts cache misses across all refinement rounds, not paths.
    """
    if num_t < 0:
        raise ValueError("num_t must be nonnegative")
    if len(offsets) != num_t + 1:
        raise ValueError("offsets must contain num_t + 1 entries")
    nb = len(deltas)
    if not (len(masks) == len(spans) == len(u_idxs) == nb):
        raise ValueError("batch arrays have different lengths")
    if int(offsets[0]) != 0 or int(offsets[-1]) != nb:
        raise ValueError("offsets do not cover the batch arrays")
    if max_states <= 0 or (deadline is not None and time.perf_counter() >= deadline):
        return None

    forced = {int(t): int(u) for t, u in (forced_at or {}).items()}
    differ = tuple((int(a), int(b)) for a, b in rf_differ)
    same = tuple((int(a), int(b)) for a, b in rf_same)
    critical = 0
    for a, b in differ + same:
        critical |= a | b
    if any(t < 0 or t >= num_t for t in forced):
        return DSSRResult(math.inf, [], 0, 0, critical.bit_count())
    forced_times = sorted(forced)
    total_nodes = 0
    rounds = 0

    def check_budget() -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise _DSSRLimit

    # Branch filtering is independent of the remembered item set.  Retain all
    # batch costs, including nonnegative ones: forcing a setup or an RF partner
    # can make such a batch necessary for a feasible improving column.
    arcs_by_t: list[list[tuple[int, int, float, int]]] = [[] for _ in range(num_t)]
    future_masks = [0] * (num_t + 1)
    try:
        for t in range(num_t):
            check_budget()
            lo, hi = int(offsets[t]), int(offsets[t + 1])
            if lo > hi or lo < 0 or hi > nb:
                raise ValueError("offsets must be monotone and in range")
            pos = bisect.bisect_right(forced_times, t)
            next_forced = forced_times[pos] if pos < len(forced_times) else num_t
            required_u = forced.get(t)
            for b in range(lo, hi):
                if b & 255 == 0:
                    check_budget()
                span = int(spans[b])
                if span <= 0:
                    raise ValueError("batch spans must be positive")
                end = t + span
                if end > num_t or end > next_forced:
                    continue
                if required_u is not None and int(u_idxs[b]) != required_u:
                    continue
                mask = int(masks[b])
                cost = float(deltas[b])
                if mask < 0 or not math.isfinite(cost):
                    raise ValueError("batch masks must be nonnegative and costs finite")
                if any((mask & a) and (mask & c) for a, c in differ):
                    continue
                # A missing RF-same endpoint cannot be supplied later.
                if any((a == 0 and mask & c) or (c == 0 and mask & a) for a, c in same):
                    continue
                arcs_by_t[t].append((end, mask, cost, b))
                future_masks[t] |= mask
        for t in range(num_t - 1, -1, -1):
            future_masks[t] |= future_masks[t + 1]

        if lower_bounds is None:
            lower = [0.0] * (num_t + 1)
            for t in range(num_t - 1, -1, -1):
                value = math.inf if t in forced else lower[t + 1]
                for end, _mask, cost, _b in arcs_by_t[t]:
                    value = min(value, cost + lower[end])
                lower[t] = value
        else:
            if len(lower_bounds) != num_t + 1:
                raise ValueError("lower_bounds must contain num_t + 1 entries")
            lower = [float(v) for v in lower_bounds]
            if any(math.isnan(v) for v in lower):
                raise ValueError("lower bounds must not contain NaN")

        while True:
            check_budget()
            rounds += 1
            # Unremembered filler items have no effect on future relaxed
            # feasibility.  For a fixed start, end and remembered item set,
            # retaining only the cheapest filler is therefore exact.
            grouped: list[list[tuple[int, int, float, int]]] = []
            for original in arcs_by_t:
                check_budget()
                cheapest: dict[tuple[int, int], tuple[int, int, float, int]] = {}
                for k, (end, mask, cost, b) in enumerate(original):
                    if k & 255 == 0:
                        check_budget()
                    tracked = mask & critical
                    key = (end, tracked)
                    old = cheapest.get(key)
                    if old is None or cost < old[2]:
                        cheapest[key] = (end, tracked, cost, b)
                arcs = list(cheapest.values())
                arcs.sort(key=lambda arc: arc[2] + lower[arc[0]])
                grouped.append(arcs)

            decisions: dict[tuple[int, int], tuple[int, int, int]] = {}

            @functools.lru_cache(maxsize=None)
            def solve(t: int, used: int) -> float:
                nonlocal total_nodes
                total_nodes += 1
                if total_nodes > max_states:
                    raise _DSSRLimit
                if total_nodes & 127 == 0:
                    check_budget()
                # An RF-same obligation that cannot be completed is already
                # infeasible; do not wait until the horizon to discover it.
                future = future_masks[t]
                for a, c in same:
                    has_a, has_c = bool(used & a), bool(used & c)
                    if has_a != has_c:
                        missing = c if has_a else a
                        if not future & missing:
                            return math.inf
                if t >= num_t:
                    return 0.0

                if t in forced:
                    best = math.inf
                    decision = None
                else:
                    best = solve(t + 1, used)
                    decision = (t + 1, used, -1)
                for end, tracked, cost, b in grouped[t]:
                    if used & tracked:
                        continue
                    new_used = used | tracked
                    if any((new_used & a) and (new_used & c) for a, c in differ):
                        continue
                    # This is a state-local residual bound.  No global prefix
                    # incumbent is used to cache a falsely optimal residual.
                    if cost + lower[end] > best + 1e-9:
                        continue
                    value = cost + solve(end, new_used)
                    if value < best:
                        best = value
                        decision = (end, new_used, b)
                if decision is not None:
                    decisions[t, used] = decision
                return best

            try:
                value = solve(0, 0)
            finally:
                # Pricing is called thousands of times.  Do not leave a large
                # self-referential lru-cache waiting for cyclic GC, including
                # when a round exhausts its state or wall-clock budget.
                solve.cache_clear()
            check_budget()
            if math.isinf(value):
                return DSSRResult(value, [], total_nodes, rounds, critical.bit_count())
            t = used = seen = repeated = 0
            path = []
            while t < num_t:
                end, new_used, b = decisions[t, used]
                if b >= 0:
                    mask = int(masks[b])
                    repeated |= seen & mask
                    seen |= mask
                    path.append(b)
                t, used = end, new_used
            if repeated == 0:
                return DSSRResult(value, path, total_nodes, rounds, critical.bit_count())
            # Remembered resources cannot repeat on a feasible relaxed path.
            # Strict growth gives finite termination at the full-memory DP.
            if repeated & critical:
                raise RuntimeError("DSSR repeated a remembered item")
            critical |= repeated
    except (_DSSRLimit, RecursionError):
        return None
