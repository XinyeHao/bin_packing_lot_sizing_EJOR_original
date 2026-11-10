"""Diagnostic BPC run for small seed 10102.

This test keeps EBRP as the pricing method, but relaxes only the
implementation control that can cause premature CG failures: the per-node
EBRP budget is increased from the formal experiment value to 40 seconds.
The formal experiment scripts are not modified.
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
from experiments.paper_experiment_config import PROCESS_ASSUMPTION, MODEL_VERSION, make_set_config, scale_spec
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=10102)
    p.add_argument("--output", default=None)
    p.add_argument("--limit", type=float, default=900.0)
    p.add_argument("--ebrp-time-limit", type=float, default=40.0)
    p.add_argument("--cg-iterations", type=int, default=120)
    a = p.parse_args()

    seed = a.seed
    spec = scale_spec("small")
    raw = data = solver = None
    started = time.perf_counter()
    row = {
        "method": "bpc_test",
        "scale": "small",
        "seed": seed,
        "T": spec.periods,
        "I": spec.items,
        "time_limit_sec": a.limit,
        "target_mip_gap": 0.0,
        "pricing_mode": "ebrp",
        "ebrp_force": True,
        "ebrp_time_limit_sec": a.ebrp_time_limit,
        "cg_max_iterations": a.cg_iterations,
        "cuts": "standard",
        "strict_cg_certification": True,
        "process_assumption": PROCESS_ASSUMPTION,
        "model_version": MODEL_VERSION,
    }
    try:
        cfg.apply_bench_generation()
        raw = generate_instance(make_set_config(spec), seed % 1000, seed)
        data = preprocess(raw)
        solver = BranchAndPriceSolver(
            data,
            max_nodes=8000,
            time_limit=a.limit,
            mip_gap=0.0,
            branch_strategy="frac_sb",
            use_node_im=False,
            root_im=False,
            final_im=False,
            strict_cg_certification=True,
            cg_kwargs={
                "pricing_mode": "ebrp",
                "ebrp_force": True,
                "use_gurobi_init": True,
                "pricing_inprocess": True,
                "skip_cross_check": True,
                "lrmp_cuts": "standard",
                "max_iterations": a.cg_iterations,
                "ebrp_time_limit": a.ebrp_time_limit,
                "verbose": False,
            },
        )
        result = solver.solve()
        ub, lb = result.objective, result.lower_bound
        row.update(
            status=result.status,
            ub=ub,
            lb=lb,
            gap_pct=None if ub is None or lb is None else max(0.0, 100.0 * (ub - lb) / ub),
            lower_bound_certified=(result.nodes_cg_failed == 0 and lb is not None),
            nodes=result.nodes_explored,
            nodes_created=result.nodes_created,
            nodes_pruned=result.nodes_pruned,
            nodes_cg_failed=result.nodes_cg_failed,
            root_lb=result.root_valid_lb,
            root_converged=result.root_cg_converged,
            runtime_sec=round(time.perf_counter() - started, 3),
        )
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}", runtime_sec=round(time.perf_counter() - started, 3))
    finally:
        for obj in (solver, data, raw):
            try:
                obj.dispose()
            except Exception:
                pass
        gc.collect()
        try:
            gp.disposeDefaultEnv()
        except Exception:
            pass

    output = Path(a.output or f"result/test_bpc_small_{seed}_ebrp.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
