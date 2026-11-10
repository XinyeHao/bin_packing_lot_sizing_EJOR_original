"""One controlled legacy BP/BPC cut-ablation solve.

The worker deliberately follows the legacy BPC settings used for the paper's
original medium- and large-scale runs.  ``method`` is the only switch:

* BP  : no LRMP deadline-cover cuts;
* BPC : the legacy ``standard`` LRMP deadline-cover cuts.

Everything else (instance generation, EBRP pricing, initial columns,
branching, budgets, and non-strict CG treatment) is held fixed.
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
    PROCESS_ASSUMPTION,
    make_set_config,
    scale_spec,
)
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess


# The suffix distinguishes results produced after aligning E_tau with the
# manuscript definition E_tau={i: e_i <= tau}.
MODEL_VERSION = "common_unload_v2_span_link_legacy_cut_ablation_deadline_e"


def relative_gap(ub: float | None, lb: float | None) -> float | None:
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return 100.0 * max(0.0, (float(ub) - float(lb)) / float(ub))


def cleanup(*objects) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4 worker: legacy BP/BPC cut ablation."
    )
    parser.add_argument("method", choices=("bp", "bpc"))
    parser.add_argument("scale", choices=("small", "medium", "large"))
    parser.add_argument("seed", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--ebrp-time-limit", type=float, default=20.0)
    args = parser.parse_args()

    spec = scale_spec(args.scale)
    limit = float(args.limit if args.limit is not None else spec.time_limit_sec)
    target_mip_gap = 0.0 if args.scale == "small" else 0.01
    cuts = "none" if args.method == "bp" else "standard"
    raw = data = solver = result = None
    row = {
        "experiment": "Exp4 legacy BP versus BPC deadline-cover-cut ablation",
        "method": args.method.upper(),
        "scale": args.scale,
        "seed": args.seed,
        "T": spec.periods,
        "I": spec.items,
        "time_limit_sec": limit,
        "target_mip_gap": target_mip_gap,
        "process_assumption": PROCESS_ASSUMPTION,
        "model_version": MODEL_VERSION,
        "pricing_mode": "ebrp",
        "cuts": cuts,
        "strict_cg_certification": False,
        "controlled_difference": "lrmp_cuts only",
        "algorithm_parameters": {
            "branch_strategy": "frac_sb",
            "use_gurobi_init": True,
            "pricing_inprocess": True,
            "ebrp_force": True,
            "ebrp_time_limit": args.ebrp_time_limit,
            "max_iterations": 120,
            "max_nodes": 8000,
            "use_node_im": False,
            "root_im": False,
            "final_im": False,
        },
        "termination_rule": "pending-node heap empty OR target relative gap OR time limit",
    }

    try:
        # Same generator and seed mapping as the original BPC worker.
        cfg.apply_bench_generation()
        raw = generate_instance(make_set_config(spec), args.seed % 1000, args.seed)
        data = preprocess(raw)

        t0 = time.perf_counter()
        solver = BranchAndPriceSolver(
            data,
            max_nodes=8000,
            time_limit=limit,
            mip_gap=target_mip_gap,
            branch_strategy="frac_sb",
            use_node_im=False,
            root_im=False,
            final_im=False,
            # This is intentionally the old, practical BPC convention.
            strict_cg_certification=False,
            cg_kwargs={
                "pricing_mode": "ebrp",
                "ebrp_force": True,
                "use_gurobi_init": True,
                "skip_cross_check": True,
                "pricing_inprocess": True,
                "verbose": False,
                "progress_interval_sec": 120.0,
                "lrmp_cuts": cuts,
                "max_iterations": 120,
                "ebrp_time_limit": args.ebrp_time_limit,
            },
        )
        result = solver.solve()
        elapsed = time.perf_counter() - t0
        raw_gap = relative_gap(result.objective, result.lower_bound)
        certified = result.nodes_cg_failed == 0 and result.lower_bound is not None
        row.update(
            status=result.status,
            ub=result.objective,
            lb=result.lower_bound,
            raw_gap_pct=raw_gap,
            certified_gap_pct=raw_gap if certified else None,
            lower_bound_certified=certified,
            runtime_sec=round(elapsed, 3),
            nodes=result.nodes_explored,
            nodes_created=result.nodes_created,
            nodes_pruned=result.nodes_pruned,
            nodes_cg_failed=result.nodes_cg_failed,
            root_lb=result.root_valid_lb,
            root_converged=result.root_cg_converged,
            root_stop=result.root_cg_stop,
        )
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        cleanup(result, solver, data, raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
