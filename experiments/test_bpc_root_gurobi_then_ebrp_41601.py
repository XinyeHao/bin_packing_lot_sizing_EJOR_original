from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["GRB_CS_QUIET"] = "1"

import gurobipy as gp
import configuration as cfg
from experiments.paper_experiment_config import MODEL_VERSION, PROCESS_ASSUMPTION, make_set_config, scale_spec
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess


def main():
    seed = 41601
    spec = scale_spec("large")
    raw = data = solver = result = None
    t0 = time.perf_counter()
    try:
        cfg.apply_bench_generation()
        raw = generate_instance(make_set_config(spec), seed % 1000, seed)
        data = preprocess(raw)
        solver = BranchAndPriceSolver(
            data, max_nodes=8000, time_limit=3600.0, mip_gap=0.01,
            branch_strategy="frac_sb", strict_cg_certification=True,
            use_node_im=False, root_im=False, final_im=False,
            cg_kwargs={
                "pricing_mode": "ebrp",
                "initial_pricing_mode": "im_gurobi",
                "ebrp_force": True,
                "use_gurobi_init": True,
                "pricing_inprocess": True,
                "skip_cross_check": True,
                "lrmp_cuts": "standard",
                "max_iterations": 120,
                "ebrp_time_limit": 20.0,
                "im_pricing_time_limit": 90.0,
                "verbose": False,
            },
        )
        result = solver.solve()
        ub, lb = result.objective, result.lower_bound
        row = {
            "method": "bpc_root_gurobi_init_then_ebrp",
            "scale": "large", "seed": seed, "T": spec.periods, "I": spec.items,
            "time_limit_sec": 3600.0,
            "initial_pricing_mode": "im_gurobi",
            "pricing_mode_after_initial": "ebrp",
            "status": result.status, "ub": ub, "lb": lb,
            "gap_pct": (100.0 * (ub - lb) / ub) if ub and lb is not None else None,
            "runtime_sec": round(time.perf_counter() - t0, 3),
            "nodes": result.nodes_explored,
            "nodes_cg_failed": result.nodes_cg_failed,
            "root_converged": result.root_cg_converged,
            "root_stop": result.root_cg_stop,
            "model_version": MODEL_VERSION,
            "process_assumption": PROCESS_ASSUMPTION,
        }
    except Exception as exc:
        row = {"method": "bpc_root_gurobi_init_then_ebrp", "scale": "large", "seed": seed,
               "status": "error", "error": f"{type(exc).__name__}: {exc}",
               "runtime_sec": round(time.perf_counter() - t0, 3)}
    finally:
        for obj in (result, solver, data, raw):
            try:
                obj.dispose()
            except Exception:
                pass
        gc.collect()
        try:
            gp.disposeDefaultEnv()
        except Exception:
            pass
    out = ROOT / "result" / "exp2" / "test_bpc_root_gurobi_then_ebrp_41601.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
