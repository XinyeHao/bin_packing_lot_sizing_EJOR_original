"""One practical BPC solve for Experiment 6.

This intentionally follows the legacy practical BPC protocol: EBRP pricing
with a 20-second per-call cap and no strict CG certification.  It is not used
to claim a certified optimum; the result reports the best incumbent, the
available tree lower bound, the raw gap, runtime, nodes, and CG-failed nodes.
"""
from __future__ import annotations

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
from experiments.paper_experiment_config import MODEL_VERSION, PROCESS_ASSUMPTION, make_set_config, scale_spec
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess


def gap_pct(ub, lb):
    if ub is None or lb is None or abs(float(ub)) <= 1e-9:
        return None
    return 100.0 * max(0.0, (float(ub) - float(lb)) / abs(float(ub)))


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


def main() -> None:
    # argv: scale seed time_limit output_json
    scale = sys.argv[1]
    seed = int(sys.argv[2])
    limit = float(sys.argv[3])
    output = Path(sys.argv[4])
    spec = scale_spec(scale)
    raw = data = solver = result = None
    started = time.perf_counter()
    try:
        cfg.apply_bench_generation()
        raw = generate_instance(make_set_config(spec), seed % 1000, seed)
        data = preprocess(raw)
        solver = BranchAndPriceSolver(
            data,
            max_nodes=8000,
            time_limit=limit,
            mip_gap=0.01 if scale != "small" else 0.0,
            branch_strategy="frac_sb",
            use_node_im=False,
            root_im=False,
            final_im=False,
            # Deliberately disabled for Exp6: this is the legacy practical
            # protocol requested by the user.
            strict_cg_certification=False,
            cg_kwargs={
                "pricing_mode": "ebrp",
                "ebrp_force": True,
                "use_gurobi_init": True,
                "pricing_inprocess": True,
                "skip_cross_check": True,
                "lrmp_cuts": "standard",
                "max_iterations": 120,
                "ebrp_time_limit": 20.0,
                "verbose": False,
            },
        )
        result = solver.solve()
        runtime = time.perf_counter() - started
        ub = result.objective
        lb = result.lower_bound
        row = {
            "scale": scale,
            "seed": seed,
            "T": spec.periods,
            "I": spec.items,
            "time_limit_sec": limit,
            "status": result.status,
            "ub": ub,
            "lb": lb,
            "gap_pct": gap_pct(ub, lb),
            "runtime_sec": round(runtime, 3),
            "nodes": result.nodes_explored,
            "nodes_cg_failed": result.nodes_cg_failed,
            "model_version": MODEL_VERSION,
            "process_assumption": PROCESS_ASSUMPTION,
        }
    except Exception as exc:
        row = {
            "scale": scale,
            "seed": seed,
            "time_limit_sec": limit,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_sec": round(time.perf_counter() - started, 3),
        }
    finally:
        cleanup(result, solver, data, raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    os._exit(0 if row.get("status") != "error" else 1)


if __name__ == "__main__":
    main()
