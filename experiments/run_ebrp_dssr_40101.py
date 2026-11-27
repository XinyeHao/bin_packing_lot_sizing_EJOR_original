"""Run one large RF-BP instance with the current exact EBRP oracle.

The settings match run_rf_small.py --scale large --seeds SEED. Historical
rf_large_SEED.json is read as a baseline, never overwritten. The default seed
remains 40101; --seed selects another member of the large benchmark set.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import gurobipy as gp
import configuration as cfg
from experiments.paper_experiment_config import (
    MODEL_VERSION, PROCESS_ASSUMPTION, generation_metadata,
    make_set_config, scale_spec,
)
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from model_and_algo.pricing_ebrp import EBRP_TIME
from preprocessor.preprocess import preprocess


def _gap(ub, lb):
    if ub is None or lb is None or abs(ub) < 1e-9:
        return None
    return 100.0 * (ub - lb) / abs(ub)


def _source_hashes():
    result = {}
    for name in ("model_and_algo/pricing_ebrp.py", "model_and_algo/pricing_route_dssr.py",
                 "model_and_algo/branch_and_price.py"):
        path = ROOT / name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description="Large instance: EBRP time-DAG bound and DSSR")
    parser.add_argument("--seed", type=int, choices=scale_spec("large").seeds, default=40101)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    seed = args.seed
    output = ROOT / (args.output or f"result/diagnostics/ebrp_dssr_{seed}.json")
    baseline_path = ROOT / f"result/rf_large_{seed}.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    spec = scale_spec("large")
    cfg.apply_bench_generation()
    raw = generate_instance(make_set_config(spec), seed % 1000, seed)
    data = preprocess(raw)
    cg_kwargs = {
        "pricing_mode": "ebrp", "ebrp_force": True,
        "use_gurobi_init": True, "pricing_inprocess": True,
        "skip_cross_check": True, "lrmp_cuts": "none",
        "max_iterations": 120, "ebrp_time_limit": 20.0, "verbose": False,
    }
    row = {
        "experiment": "ebrp_time_dag_dssr_large",
        "scale": spec.name, "seed": seed, "T": spec.periods, "I": spec.items,
        "machines": len(data.machines), "configurations": len(data.configs),
        "branch_strategy": "ryan_foster", "cuts": "none",
        "upper_bound_method": "root_irmp_cgfo_local_repair",
        "time_limit_sec": args.time_limit, "target_mip_gap": 0.01,
        "max_nodes": 8000, "strict_cg_certification": True,
        "cg_kwargs": cg_kwargs,
        "model_version": MODEL_VERSION, "process_assumption": PROCESS_ASSUMPTION,
        "generation": generation_metadata(),
        "environment": {"python": platform.python_version(), "gurobi": gp.gurobi.version(), "pid": os.getpid()},
        "source_sha256": _source_hashes(),
        "certificate_safeguards": [
            "retain every live open/lazy node; no queue truncation",
            "compute global bounds from certified node/parent bounds, not scheduling priorities",
            "close integer nodes locally without stopping pending siblings",
            "retain unresolved-node bounds and block uncertified gap/optimal status",
            "do not use an unconverged restricted-master objective as a fallback lower bound",
        ],
        "comparison_scope": "Historical comparison includes exactness safeguards; not an isolated pricing ablation.",
        "baseline_path": str(baseline_path.relative_to(ROOT)), "baseline": baseline,
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Large instance seed={seed}; limit={args.time_limit:.0f}s; RF-BP gap=1%", flush=True)
    print(f"Baseline retained at {baseline_path}; new result -> {output}", flush=True)
    solver = BranchAndPriceSolver(
        data, max_nodes=8000, time_limit=args.time_limit, mip_gap=0.01,
        branch_strategy="ryan_foster", use_node_im=False,
        root_im=False, final_im=False, strict_cg_certification=True,
        cg_kwargs=cg_kwargs,
    )
    started = time.perf_counter()
    try:
        result = solver.solve()
        row.update(
            status=result.status, ub=result.objective, lb=result.lower_bound,
            gap_pct=_gap(result.objective, result.lower_bound),
            runtime_sec=round(time.perf_counter() - started, 6),
            core_runtime_sec=result.runtime_sec, audit_runtime_sec=result.audit_runtime_sec,
            best_node_lb=result.best_node_lb, incumbent=result.incumbent,
            nodes=result.nodes_explored, nodes_created=result.nodes_created,
            nodes_pruned=result.nodes_pruned, nodes_infeasible=result.nodes_infeasible,
            nodes_cg_failed=result.nodes_cg_failed, max_depth=result.max_depth,
            nodes_unresolved=result.nodes_unresolved,
            root_lb=result.root_valid_lb, root_lrmp=result.root_lrmp,
            root_converged=result.root_cg_converged, root_cg_stop=result.root_cg_stop,
            y1_lazy_created=result.y1_lazy_created, lazy_popped=result.lazy_popped,
            dfs_popped=result.dfs_popped,
            profiling={
                name.removeprefix("_time_"): float(getattr(solver, name, 0.0))
                for name in ("_time_cg", "_time_master", "_time_pricing", "_time_irmp", "_time_cgfo")
            },
            primal_history=result.primal_history, primal_stats=result.primal_stats,
            ebrp_accumulated=dict(EBRP_TIME),
        )
        # Capture optional counters added by the DSSR integration without
        # imposing an extra public API on the production oracle.
        from model_and_algo import pricing_ebrp
        for name in ("DSSR_TIME", "DSSR_STATS"):
            counters = getattr(pricing_ebrp, name, None)
            if isinstance(counters, dict):
                row[name.lower()] = dict(counters)
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                   runtime_sec=round(time.perf_counter() - started, 6))
        raise
    finally:
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        print(f"Saved to {output}", flush=True)
        gc.collect()
        try:
            gp.disposeDefaultEnv()
        except Exception:
            pass


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
