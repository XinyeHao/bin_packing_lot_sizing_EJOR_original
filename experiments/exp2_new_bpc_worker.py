"""One isolated, paper-aligned BPC run for Experiment 2.

The compact IM is not solved here.  BPC uses EBRP for every pricing call;
Gurobi is used only as the RMP LP solver and inside the CGFO incumbent
heuristic implemented by the branch-and-price solver.
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


def _gap(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return max(0.0, 100.0 * (ub - lb) / ub)


def _cleanup(*objects):
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


def solve_one(scale: str, seed: int, limit: float, max_nodes: int, ebrp_limit: float | None) -> dict:
    spec = scale_spec(scale)
    raw = data = solver = result = None
    row = {
        "experiment": "exp2_new_bpc",
        "method": "BPC",
        "scale": scale,
        "seed": seed,
        "T": spec.periods,
        "I": spec.items,
        "time_limit_sec": limit,
        "model_version": MODEL_VERSION,
        "process_assumption": PROCESS_ASSUMPTION,
        "pricing_mode": "ebrp",
        "cuts": "standard_deadline_cover",
    }
    try:
        cfg.apply_bench_generation()
        raw = generate_instance(make_set_config(spec), seed % 1000, seed)
        data = preprocess(raw)
        started = time.perf_counter()
        solver = BranchAndPriceSolver(
            data,
            max_nodes=max_nodes,
            time_limit=limit,
            mip_gap=0.0 if scale == "small" else 0.01,
            branch_strategy="frac_sb",
            use_node_im=False,
            root_im=False,
            final_im=False,
            strict_cg_certification=True,
            cg_kwargs={
                "pricing_mode": "ebrp",
                "ebrp_force": True,
                "use_gurobi_init": False,
                "pricing_inprocess": True,
                "skip_cross_check": True,
                "lrmp_cuts": "standard",
                "max_iterations": 120,
                "ebrp_time_limit": ebrp_limit,
                "verbose": False,
            },
        )
        result = solver.solve()
        runtime = time.perf_counter() - started
        ub, lb = result.objective, result.lower_bound
        row.update(
            status=result.status,
            ub=ub,
            lb=lb,
            gap_pct=_gap(ub, lb),
            runtime_sec=round(runtime, 3),
            nodes=result.nodes_explored,
            nodes_created=result.nodes_created,
            nodes_pruned=result.nodes_pruned,
            nodes_cg_failed=result.nodes_cg_failed,
            root_lb=result.root_valid_lb,
            root_cg_converged=result.root_cg_converged,
            root_cg_stop=result.root_cg_stop,
            lower_bound_certified=(result.lower_bound is not None and result.nodes_cg_failed == 0),
            termination_rule="pending-node heap empty OR 1% gap OR time limit",
        )
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        _cleanup(result, solver, data, raw)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scale", choices=("small", "medium", "large"))
    parser.add_argument("seed", type=int)
    parser.add_argument("output")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--max-nodes", type=int, default=8000)
    parser.add_argument("--ebrp-time-limit", type=float, default=None)
    args = parser.parse_args()
    spec = scale_spec(args.scale)
    row = solve_one(args.scale, args.seed, args.limit or spec.time_limit_sec, args.max_nodes, args.ebrp_time_limit)
    Path(args.output).write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
