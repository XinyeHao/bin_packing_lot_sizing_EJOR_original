"""Validate exact DSSR against independent exhaustive tiny-path enumeration.

The catalogs include positive batches: forced setups and RF-same decisions can
make these necessary.  No Gurobi or existing EBRP route is used as the reference.
Run from the project root: python -B experiments/validate_ebrp_dssr.py
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Batch:
    start: int
    span: int
    mask: int
    cost: float
    config: int = 0


def arrays(batches: list[Batch], horizon: int):
    ordered = sorted(batches, key=lambda b: (b.start, b.cost, b.span, b.mask))
    offsets = [0]
    for t in range(horizon):
        offsets.append(offsets[-1] + sum(b.start == t for b in ordered))
    return ordered, (
        np.asarray([b.cost for b in ordered], dtype=np.float64),
        np.asarray([b.mask for b in ordered], dtype=np.uint64),
        np.asarray([b.span for b in ordered], dtype=np.int64),
        np.asarray([b.config for b in ordered], dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
    )


def rf_allows(used: int, differ, same, *, terminal: bool):
    if any(used & left and used & right for left, right in differ):
        return False
    return not terminal or all(bool(used & left) == bool(used & right) for left, right in same)


def exhaustive(batches, horizon, forced, differ, same):
    """Enumerate feasible paths without memoization or relaxed-state merging."""
    by_start = [[(i, b) for i, b in enumerate(batches) if b.start == t] for t in range(horizon)]
    best, best_path = math.inf, []

    def visit(t, used, cost, path):
        nonlocal best, best_path
        if t == horizon:
            if rf_allows(used, differ, same, terminal=True) and cost < best:
                best, best_path = cost, list(path)
            return
        if t not in forced:
            visit(t + 1, used, cost, path)
        for i, b in by_start[t]:
            end = t + b.span
            if end > horizon or used & b.mask:
                continue
            if t in forced and b.config != forced[t]:
                continue
            if any(t < ft < end for ft in forced):
                continue
            new_used = used | b.mask
            if rf_allows(new_used, differ, same, terminal=False):
                visit(end, new_used, cost + b.cost, path + [i])

    visit(0, 0, 0.0, [])
    return best, best_path


def time_lower_bound(batches, horizon, forced):
    """Exact time-only relaxation, ignoring every item-uniqueness/RF row."""
    bound = [0.0] * (horizon + 1)
    for t in range(horizon - 1, -1, -1):
        values = [] if t in forced else [bound[t + 1]]
        for b in batches:
            if b.start != t or t + b.span > horizon:
                continue
            if t in forced and b.config != forced[t]:
                continue
            if any(t < ft < t + b.span for ft in forced):
                continue
            values.append(b.cost + bound[t + b.span])
        bound[t] = min(values, default=math.inf)
    return bound


def check_path(path, batches, horizon, forced, differ, same):
    used, previous_end, value = 0, 0, 0.0
    setup_starts = {}
    for i in path:
        b = batches[i]
        assert b.start >= previous_end, ("overlap", path)
        assert b.start + b.span <= horizon
        assert not used & b.mask, ("repeated item", path)
        used |= b.mask
        value += b.cost
        previous_end = b.start + b.span
        setup_starts[b.start] = b.config
    assert all(setup_starts.get(t) == config for t, config in forced.items()), ("missing forced setup", path)
    assert rf_allows(used, differ, same, terminal=True), ("RF violation", path)
    return value


def check_case(name, batches, horizon, forced=None, differ=(), same=()):
    from model_and_algo.pricing_route_dssr import solve_dssr

    forced = forced or {}
    ordered, data = arrays(batches, horizon)
    expected, _ = exhaustive(ordered, horizon, forced, differ, same)
    lower = time_lower_bound(ordered, horizon, forced)
    for use_bound in (False, True):
        result = solve_dssr(*data, horizon, forced_at=forced, rf_differ=differ, rf_same=same,
                            lower_bounds=lower if use_bound else None, deadline=time.perf_counter() + 10)
        assert result is not None and result.exact and not result.timed_out, (name, "not exact", result)
        if math.isinf(expected):
            assert math.isinf(result.cost) and result.cost > 0, (name, expected, result.cost)
        else:
            assert abs(result.cost - expected) <= 1e-7, (name, use_bound, expected, result.cost)
            value = check_path(result.path, ordered, horizon, forced, differ, same)
            assert abs(value - result.cost) <= 1e-7, (name, value, result.cost)


def random_catalog(rng, horizon, nitems):
    """Build real feasible same-family subsets with stepped item durations."""
    families = [rng.randrange(2) for _ in range(nitems)]
    lengths = [rng.randint(1, 3) for _ in range(nitems)]
    base = [rng.randint(1, 2) for _ in range(nitems)]
    releases = [rng.randrange(max(1, horizon - 1)) for _ in range(nitems)]
    deadlines = [rng.randint(r, horizon - 1) for r in releases]
    thresholds = [rng.randint(1, horizon) for _ in range(nitems)]
    gains = [rng.randint(-20, 100) / 10 for _ in range(nitems)]
    arrival_prices = [[rng.randint(-10, 20) / 10 for _ in range(horizon)] for _ in range(nitems)]
    setups = [rng.randint(0, 40) / 10 for _ in range(2)]
    capacity = rng.randint(3, 6)
    result = []
    for t in range(horizon):
        for family in (0, 1):
            eligible = [i for i in range(nitems) if families[i] == family and releases[i] <= t <= deadlines[i]]
            for count in range(1, len(eligible) + 1):
                for subset in itertools.combinations(eligible, count):
                    if sum(lengths[i] for i in subset) > capacity:
                        continue
                    span = max(base[i] + int(t - releases[i] >= thresholds[i]) for i in subset)
                    end = t + span
                    if end <= horizon:
                        value = setups[family] - sum(gains[i] + arrival_prices[i][end - 1] for i in subset)
                        result.append(Batch(t, span, sum(1 << i for i in subset), value, family))
    return result


def production_checks():
    """Exercise real context scoring, branch filtering and X/Y decoding.

    The reference enumerates subsets directly from item data, independently
    of both the cached subsets and the production score/Pareto routines.
    """
    from common.data_models import InstanceData
    from model_and_algo.pricing_ebrp import EBRPContext, solve_subproblem_ebrp
    from preprocessor.preprocess import preprocess

    def tiny(capacity):
        raw = InstanceData(
            name="dssr_validation", set_type="validation", num_periods=4,
            config_ids=["u0", "u1"], machine_ids=["m0", "m1"],
            cured_item_ids=["A", "B", "C"], config_lead_times={"u0": 1, "u1": 1},
            item_config={"A": "u0", "B": "u0", "C": "u1"},
            item_lead_times={"A": 1, "B": 1, "C": 1},
            tray_lengths={"A": 1.0, "B": 1.0, "C": 1.0},
            machine_capacities={"m0": float(capacity), "m1": 2.0},
            production_costs={m: {"u0": 5.0, "u1": 7.0} for m in ("m0", "m1")},
            holding_costs={i: 1.0 for i in ("A", "B", "C")},
            backorder_costs={i: 1.0 for i in ("A", "B", "C")},
            demand={i: {4: 1} for i in ("A", "B", "C")},
            earliest_starts={i: 1 for i in ("A", "B", "C")},
            deadlines={i: 4 for i in ("A", "B", "C")},
            item_machines={"A": ["m0", "m1"], "B": ["m0", "m1"], "C": ["m1"]},
        )
        data = preprocess(raw)
        return data, EBRPContext.build(data, 0)

    cases = [
        ("root full scoring", 1, [12.0, 1.0, 0.0], {}, {}),
        ("forced positive full scoring", 1, [0.0, 0.0, 0.0], {(0, 1): 1}, {}),
        ("RF same positive companion full scoring", 1, [12.0, 1.0, 0.0], {}, {(0, 1): 1}),
        ("RF same prevents subset dominance", 2, [12.0, 0.0, 0.0], {}, {(0, 1): 1}),
        ("RF differ full scoring", 2, [12.0, 10.0, 0.0], {}, {(0, 1): 0}),
        ("RF missing partner full scoring", 1, [12.0, 1.0, 0.0], {}, {(0, 2): 1}),
        ("forbidden setup full scoring", 1, [12.0, 1.0, 0.0], {(0, t): 0 for t in range(4)}, {}),
        ("infeasible forced setup full scoring", 1, [12.0, 1.0, 0.0], {(1, 1): 1}, {}),
    ]
    contexts = {cap: tiny(cap) for cap in (1, 2)}
    comparisons = 0
    for name, capacity, rho, fixed_y, fixed_rf in cases:
        data, ctx = contexts[capacity]
        horizon = len(data.periods)
        alpha = [[0.0] * horizon for _ in data.cured_items]
        beta = -0.75
        forced = {t: u for (u, t), value in fixed_y.items() if value == 1}
        forbidden = {(u, t) for (u, t), value in fixed_y.items() if value == 0}
        full = []
        for t in range(horizon):
            for u, config_items in enumerate(data.items_by_config):
                if (u, t) in forbidden:
                    continue
                eligible = [i for i in config_items if data.b_im[i][0]
                            and data.earliest_starts[i] <= t + 1 <= data.deadlines[i]]
                for count in range(1, len(eligible) + 1):
                    for subset in itertools.combinations(eligible, count):
                        span = max(data.lt_it[i][t] for i in subset)
                        if sum(data.v_i[i] for i in subset) > data.q_m[0] or t + span > horizon:
                            continue
                        mask = sum(1 << ctx.global_to_local[i] for i in subset)
                        cost = data.p_cum[0][u] - sum(rho[i] + alpha[i][t + span - 1] for i in subset)
                        full.append(Batch(t, span, mask, cost, u))
        differ, same = [], []
        for (i, j), value in fixed_rf.items():
            bits = tuple(0 if k not in ctx.global_to_local else 1 << ctx.global_to_local[k] for k in (i, j))
            (same if value else differ).append(bits)
        expected, _ = exhaustive(full, horizon, forced, differ, same)
        # Both the flat/no-progress route and the progress-report legacy route
        # must preserve the same full-subproblem optimum.
        for progress in (0.0, 1000.0):
            result = solve_subproblem_ebrp(data, 0, alpha, beta, rho=rho,
                                           fixed_y=fixed_y, fixed_rf=fixed_rf,
                                           ctx=ctx, progress_interval_sec=progress,
                                           skip_cross_check=True, time_limit=10.0)
            comparisons += 1
            assert result.is_optimal, (name, progress, "not exact")
            if math.isinf(expected):
                assert math.isinf(result.reduced_cost) and result.reduced_cost > 0, (name, result)
                continue
            assert abs(result.reduced_cost - (expected - beta)) <= 1e-7, (name, progress, expected - beta, result.reduced_cost)
            starts = {}
            for (i, t), quantity in result.x.items():
                if quantity:
                    assert quantity == 1 and i in ctx.global_to_local, (name, "invalid X", i, t, quantity)
                    starts.setdefault(t, []).append(i)
            decoded = []
            for t, subset in sorted(starts.items()):
                configs = {data.config_of_item[i] for i in subset}
                assert len(configs) == 1, (name, "mixed config")
                u = configs.pop()
                mask = sum(1 << ctx.global_to_local[i] for i in subset)
                span = max(data.lt_it[i][t] for i in subset)
                match = [idx for idx, batch in enumerate(full)
                         if (batch.start, batch.span, batch.mask, batch.config) == (t, span, mask, u)]
                assert len(match) == 1, (name, "invalid decoded batch", t, subset)
                assert result.y.get((u, t)) == 1, (name, "missing decoded setup")
                decoded.append(match[0])
            value = check_path(decoded, full, horizon, forced, differ, same)
            assert abs(value - beta - result.reduced_cost) <= 1e-7, (name, "decoded RC mismatch")
            production = sum(data.p_cum[0][u] * quantity for (u, _t), quantity in result.y.items())
            assert abs(production - result.production_cost) <= 1e-7, (name, "production cost mismatch")
    return comparisons


def main():
    special = [
        ("empty", [], 4, {}, (), ()),
        ("only positive", [Batch(0, 1, 1, 2), Batch(2, 1, 2, 3)], 4, {}, (), ()),
        ("must remove repeated filler", [Batch(0, 1, 1, -4), Batch(1, 1, 1, -4), Batch(2, 1, 2, -3)], 3, {}, (), ()),
        ("positive forced setup", [Batch(1, 1, 1, 5)], 3, {1: 0}, (), ()),
        ("crossing forced epoch", [Batch(0, 3, 1, -10), Batch(1, 1, 2, 2)], 3, {1: 0}, (), ()),
        ("wrong forced family", [Batch(1, 1, 1, -10, 0), Batch(1, 1, 2, 3, 1)], 3, {1: 1}, (), ()),
        ("infeasible forced setup", [], 3, {1: 0}, (), ()),
        ("RF same needs positive partner", [Batch(0, 1, 1, -8), Batch(1, 1, 2, 3)], 3, {}, (), ((1, 2),)),
        ("RF same absent partner", [Batch(0, 1, 1, -8)], 3, {}, (), ((1, 0),)),
        ("RF differ separate batches", [Batch(0, 1, 1, -8), Batch(1, 1, 2, -9)], 3, {}, ((1, 2),), ()),
        ("RF differ inside batch", [Batch(0, 1, 3, -20), Batch(0, 1, 1, -8), Batch(1, 1, 2, -9)], 3, {}, ((1, 2),), ()),
        ("RF same infeasible forced", [Batch(0, 1, 1, -8)], 2, {0: 0}, (), ((1, 0),)),
        ("RF transitive same", [Batch(0, 1, 1, -8), Batch(1, 1, 2, 3), Batch(2, 1, 4, 2)], 3, {}, (), ((1, 2), (2, 4))),
    ]
    for case in special:
        check_case(*case)
    rng = random.Random(40101)
    random_count = 100
    for idx in range(random_count):
        horizon, nitems = rng.randint(3, 6), rng.randint(2, 5)
        batches = random_catalog(rng, horizon, nitems)
        forced = {}
        if idx % 3 == 0:
            forced[rng.randrange(horizon)] = rng.randrange(2)
        if idx % 11 == 0:
            forced[rng.randrange(horizon)] = rng.randrange(2)
        pair = tuple(1 << i for i in rng.sample(range(nitems), 2))
        differ = (pair,) if idx % 4 == 0 else ()
        same = (pair,) if idx % 4 == 1 else ()
        check_case(f"random {idx}", batches, horizon, forced, differ, same)

    from model_and_algo.pricing_route_dssr import solve_dssr
    ordered, data = arrays([Batch(t, 1, 1 << (t % 3), -3.0) for t in range(6)], 6)
    expired = solve_dssr(*data, 6, deadline=time.perf_counter() - 1)
    assert expired is None or not expired.exact, "Expired budget falsely claimed exactness"
    capped = solve_dssr(*data, 6, max_states=1)
    assert capped is None or not capped.exact, "State cap falsely claimed exactness"
    production_comparisons = production_checks()
    print(json.dumps(dict(special_cases=len(special), randomized_cases=random_count,
                          bound_modes=2, exact_comparisons=(len(special) + random_count) * 2,
                          interrupted_cases=2, production_comparisons=production_comparisons,
                          status="passed")))


if __name__ == "__main__":
    main()
