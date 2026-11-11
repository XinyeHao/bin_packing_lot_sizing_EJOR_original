"""Medium seed 31801: Ryan–Foster BP with no Deadline/SRI cuts."""
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
from experiments.paper_experiment_config import make_set_config, scale_spec
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess

SEED = 31801
OUT = ROOT / "result" / "rf_31801.json"


def gap(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return round(100.0 * max(0.0, (ub - lb) / ub), 4)


def main():
    spec = scale_spec("medium")
    cfg.apply_bench_generation()
    data = preprocess(generate_instance(make_set_config(spec), SEED % 1000, SEED))
    print(
        f"======== Ryan-Foster BP seed={SEED} T={spec.periods} I={spec.items} "
        f"cuts=none limit={spec.time_limit_sec:.0f}s ========",
        flush=True,
    )
    t0 = time.perf_counter()
    bp = BranchAndPriceSolver(
        data,
        max_nodes=8000,
        time_limit=spec.time_limit_sec,
        mip_gap=0.01,
        branch_strategy="ryan_foster",
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
            "lrmp_cuts": "none",
            "max_iterations": 120,
            "ebrp_time_limit": 20.0,
            "verbose": False,
        },
    )
    res = bp.solve()
    row = {
        "method": "bp_ryan_foster",
        "scale": "medium",
        "seed": SEED,
        "T": spec.periods,
        "I": spec.items,
        "cuts": "none",
        "branch_strategy": "ryan_foster",
        "status": res.status,
        "ub": res.objective,
        "lb": res.lower_bound,
        "gap_pct": gap(res.objective, res.lower_bound),
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "nodes": res.nodes_explored,
        "nodes_created": res.nodes_created,
        "nodes_pruned": res.nodes_pruned,
        "nodes_cg_failed": res.nodes_cg_failed,
        "root_lb": res.root_valid_lb,
        "root_lrmp": res.root_lrmp,
        "root_converged": res.root_cg_converged,
    }
    OUT.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)
    del bp, res, data
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass


if __name__ == "__main__":
    main()
