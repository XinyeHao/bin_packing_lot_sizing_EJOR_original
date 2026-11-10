"""Experiment 4 BP worker with the legacy non-strict CG rule.

BP uses EBRP pricing and no LRMP cuts.  A partial CG node may still be
used to search for a feasible incumbent, matching the legacy experiment
configuration; it is not treated as a certified lower bound.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import gurobipy as gp

import configuration as cfg
from experiments.paper_experiment_config import (
    MODEL_VERSION,
    PROCESS_ASSUMPTION,
    make_set_config,
    scale_spec,
)
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess


def gap(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return 100.0 * max(0.0, (float(ub) - float(lb)) / float(ub))


def cleanup(*objects):
    for obj in objects:
        try:
            obj.dispose()
        except Exception:
            pass
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scale", choices=("small", "medium", "large"))
    p.add_argument("seed", type=int)
    p.add_argument("output")
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--ebrp-time-limit", type=float, default=20.0)
    p.add_argument("--init-mode", choices=("gurobi", "ebrp"), default="gurobi")
    p.add_argument("--strict-cg", action="store_true")
    args = p.parse_args()
    spec = scale_spec(args.scale)
    limit = args.limit or spec.time_limit_sec
    raw = data = solver = result = None
    row = {
        "experiment": "new_exp4_bp_nonstrict_cg",
        "method": "BP",
        "scale": args.scale,
        "seed": args.seed,
        "T": spec.periods,
        "I": spec.items,
        "time_limit_sec": limit,
        "target_mip_gap": 0.0 if args.scale == "small" else 0.01,
        "model_version": MODEL_VERSION,
        "process_assumption": PROCESS_ASSUMPTION,
        "pricing_mode": "ebrp",
        "cuts": "none",
        "strict_cg_certification": args.strict_cg,
    }
    try:
        cfg.apply_bench_generation()
        raw = generate_instance(make_set_config(spec), args.seed % 1000, args.seed)
        data = preprocess(raw)
        t0 = time.perf_counter()
        solver = BranchAndPriceSolver(
            data,
            max_nodes=8000,
            time_limit=limit,
            mip_gap=0.0 if args.scale == "small" else 0.01,
            branch_strategy="frac_sb",
            use_node_im=False,
            root_im=False,
            final_im=False,
            strict_cg_certification=args.strict_cg,
            cg_kwargs={
                "pricing_mode": "ebrp",
                "ebrp_force": True,
                "use_gurobi_init": args.init_mode == "gurobi",
                "pricing_inprocess": True,
                "skip_cross_check": True,
                "lrmp_cuts": "none",
                "max_iterations": 120,
                "ebrp_time_limit": args.ebrp_time_limit,
                "verbose": False,
            },
        )
        result = solver.solve()
        runtime = time.perf_counter() - t0
        row.update(
            status=result.status,
            ub=result.objective,
            lb=result.lower_bound,
            gap_pct=gap(result.objective, result.lower_bound),
            runtime_sec=round(runtime, 3),
            nodes=result.nodes_explored,
            nodes_created=result.nodes_created,
            nodes_pruned=result.nodes_pruned,
            nodes_cg_failed=result.nodes_cg_failed,
            root_lb=result.root_valid_lb,
            root_converged=result.root_cg_converged,
            root_stop=result.root_cg_stop,
            lower_bound_certified=(result.nodes_cg_failed == 0 and result.lower_bound is not None),
            termination_rule="pending-node heap empty OR raw gap <= 1% OR time limit",
        )
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        cleanup(result, solver, data, raw)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
