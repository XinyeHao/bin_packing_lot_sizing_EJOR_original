"""Bounded root-pricing probes; does not change the production pricing oracle.

All methods use the same initial-RMP duals and Pareto-filtered batch catalog.
These probes have no branch fixes or SRI terms. Catalog filtering is therefore
valid for every method below. Results are not full-BP performance measurements.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gurobipy as gp
import numpy as np
from gurobipy import GRB

import configuration as cfg
from experiments.paper_experiment_config import SCALES, make_set_config
from experiments.run_exp3_ebrp_capacity_sensitivity import initial_duals
from instance_generator.generator import generate_instance
from model_and_algo.pricing_ebrp import EBRPStats, _compiled_on_flat, _route_flat
from preprocessor.preprocess import preprocess


def time_bounds(flat, nt):
    weak = [0.0] * (nt + 1)
    strong = [0.0] * (nt + 1)
    next_event = [nt] * (nt + 1)
    nxt = nt
    for t in range(nt - 1, -1, -1):
        lo, hi = int(flat.offsets[t]), int(flat.offsets[t + 1])
        next_event[t] = nxt
        if hi > lo:
            nxt = t
        weak[t] = weak[t + 1] + min([0.0] + [float(flat.delta[b]) for b in range(lo, hi)])
        strong[t] = min([strong[t + 1]] + [float(flat.delta[b]) + strong[t + int(flat.span[b])] for b in range(lo, hi)])
    return weak, strong, next_event


def batch_mip(ctx, flat, nt, limit):
    started = time.perf_counter()
    model = gp.Model("batch_pricing_probe")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = limit
    model.Params.MIPGap = 0
    model.Params.MIPGapAbs = 1e-7
    nb = len(flat.delta)
    x = model.addVars(nb, vtype=GRB.BINARY)
    model.setObjective(gp.quicksum(float(flat.delta[b]) * x[b] for b in range(nb)), GRB.MINIMIZE)
    period_rows = [[] for _ in range(nt)]
    item_rows = [[] for _ in ctx.local_items]
    for b in range(nb):
        t, span, mask = int(flat.t_idx[b]), int(flat.span[b]), int(flat.mask[b])
        for tau in range(t, t + span):
            period_rows[tau].append(b)
        while mask:
            bit = mask & -mask
            item_rows[bit.bit_length() - 1].append(b)
            mask -= bit
    for ids in period_rows + item_rows:
        if ids:
            model.addConstr(gp.quicksum(x[b] for b in ids) <= 1)
    build_sec = time.perf_counter() - started
    started = time.perf_counter()
    model.optimize()
    result = dict(build_sec=build_sec, solve_sec=time.perf_counter() - started,
                  objective=model.ObjVal if model.SolCount else None,
                  lower_bound=model.ObjBound, status=int(model.Status), exact=model.Status == GRB.OPTIMAL)
    model.dispose()
    return result


def dssr(flat, nt, lower, limit):
    """Exact selective-elementarity on a fixed catalog, with a time safeguard.

    Arcs with equal (start, end, tracked items) share the cheapest filler.
    Only complete relaxed solves contribute a lower bound. Untracked repeated
    items are added to the critical set; an elementary relaxed optimum is exact.
    """
    started = time.perf_counter()
    deadline = started + limit
    critical = 0
    rounds = []
    last_bound = float(lower[0])
    while True:
        if time.perf_counter() >= deadline:
            return dict(exact=False, lower_bound=last_bound, rounds=rounds, solve_sec=time.perf_counter() - started)
        best_arcs = {}
        for b in range(len(flat.delta)):
            key = (int(flat.t_idx[b]), int(flat.span[b]), int(flat.mask[b]) & critical)
            old = best_arcs.get(key)
            if old is None or float(flat.delta[b]) < float(flat.delta[old]):
                best_arcs[key] = b
        by_t = [[] for _ in range(nt)]
        for (t, span, mask), b in best_arcs.items():
            by_t[t].append((t + span, mask, float(flat.delta[b]), b))
        for arcs in by_t:
            arcs.sort(key=lambda a: a[2] + lower[a[0]])
        decisions = {}
        calls = 0

        @functools.lru_cache(None)
        def solve(t, used):
            nonlocal calls
            calls += 1
            if calls % 256 == 0 and time.perf_counter() >= deadline:
                raise TimeoutError
            if t >= nt:
                return 0.0
            best = solve(t + 1, used)
            decision = (t + 1, used, -1)
            for end, mask, cost, b in by_t[t]:
                if used & mask or cost + lower[end] >= best - 1e-9:
                    continue
                value = cost + solve(end, used | mask)
                if value < best - 1e-9:
                    best, decision = value, (end, used | mask, b)
            decisions[t, used] = decision
            return best

        try:
            value = solve(0, 0)
        except TimeoutError:
            return dict(exact=False, lower_bound=last_bound, rounds=rounds, solve_sec=time.perf_counter() - started)
        last_bound = value
        t = used = seen = repeated = 0
        path = []
        while t < nt:
            end, new_used, b = decisions[t, used]
            if b >= 0:
                mask = int(flat.mask[b])
                repeated |= seen & mask
                seen |= mask
                path.append(b)
            t, used = end, new_used
        rounds.append(dict(critical_items=critical.bit_count(), relaxed_objective=value,
                           states=calls, arcs=len(best_arcs), repeated_items=repeated.bit_count()))
        if not repeated:
            assert abs(sum(float(flat.delta[b]) for b in path) - value) < 1e-5
            return dict(exact=True, objective=value, lower_bound=value, rounds=rounds, solve_sec=time.perf_counter() - started)
        assert repeated & critical == 0
        critical |= repeated


def run_case(seed, mi, limit, warm):
    data = preprocess(generate_instance(make_set_config(SCALES["large"]), seed % 1000, seed))
    cg, alpha, rho, beta = initial_duals(data)
    ctx = cg._ebrp_contexts[mi]
    started = time.perf_counter()
    flat = ctx.score_route_batches(data, alpha, rho)
    score_sec = time.perf_counter() - started
    nt = len(data.periods)
    started = time.perf_counter()
    weak, strong, next_event = time_bounds(flat, nt)
    bounds_sec = time.perf_counter() - started
    warm_sec = 0.0
    if warm:
        started = time.perf_counter()
        _route_flat(ctx, flat, nt, deadline=started + limit, st=EBRPStats(), progress_interval_sec=0.0,
                    machine=data.machines[mi], forced_at=None, sri_terms=(), rf_differ=(), rf_same=())
        warm_sec = time.perf_counter() - started
    st = EBRPStats()
    started = time.perf_counter()
    cost, _ = _route_flat(ctx, flat, nt, deadline=started + limit, st=st, progress_interval_sec=0.0,
                          machine=data.machines[mi], forced_at=None, sri_terms=(), rf_differ=(), rf_same=())
    baseline = dict(solve_sec=time.perf_counter() - started, objective=cost,
                    exact=st.exact and not st.timed_out, states=st.nodes_expanded)
    started = time.perf_counter()
    routed = _compiled_on_flat(ctx, flat, nt, deadline=started + limit, start_t=0, start_mask=0,
                              forced_at={}, sri_terms=(), rf_differ=(), rf_same=(),
                              next_event=next_event, lb_sum=strong)
    stronger = dict(solve_sec=time.perf_counter() - started, supported=routed is not None)
    if routed is not None:
        stronger.update(objective=routed[0], exact=not routed[4], states=routed[2])
    mip = batch_mip(ctx, flat, nt, limit)
    selective = dssr(flat, nt, strong, limit)
    exact_values = [r["objective"] for r in (baseline, stronger, mip, selective) if r.get("exact")]
    if exact_values and max(exact_values) - min(exact_values) > 1e-4:
        raise AssertionError(f"Exact oracle mismatch: {exact_values}")
    result = dict(seed=seed, machine=mi, items=len(ctx.local_items), batches=len(flat.delta), beta=beta[mi],
                  objective_convention="batch reduced-cost sum; subtract beta for column RC",
                  shared_score_sec=score_sec, bounds_sec=bounds_sec, warmup_sec=warm_sec,
                  weak_lower_bound=weak[0], time_dag_lower_bound=strong[0],
                  baseline=baseline, stronger_time_bound=stronger, batch_mip=mip, dssr=selective)
    cg._model.dispose()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=float, default=8.0)
    parser.add_argument("--output", default="result/diagnostics/pricing_alternatives_root_probe.json")
    args = parser.parse_args()
    cfg.apply_bench_generation()
    rows = []
    for idx, (seed, mi) in enumerate(((41301, 0), (41601, 0), (40101, 4))):
        row = run_case(seed, mi, args.limit, warm=idx == 0)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope="three initial-RMP single-machine root probes; no branch fixes or SRI",
                                     gurobi=gp.gurobi.version(), limit_per_method=args.limit, rows=rows),
                                 ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
