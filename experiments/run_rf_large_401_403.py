"""Large seeds 40101, 40301: Ryan–Foster BP with no Deadline/SRI cuts (3600s)."""
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

SEEDS = [40101, 40301]


def gap(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return round(100.0 * max(0.0, (ub - lb) / ub), 4)


def run_one(seed: int, spec) -> dict:
    out = ROOT / "result" / f"rf_{seed}.json"
    data = preprocess(generate_instance(make_set_config(spec), seed % 1000, seed))
    print(
        f"\n======== Ryan-Foster BP seed={seed} T={spec.periods} I={spec.items} "
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
        "scale": "large",
        "seed": seed,
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
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    del bp, res, data
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass
    return row


def main():
    spec = scale_spec("large")
    cfg.apply_bench_generation()
    summary = []
    for seed in SEEDS:
        try:
            summary.append(run_one(seed, spec))
        except Exception as e:
            err = {"seed": seed, "error": repr(e)}
            summary.append(err)
            print(f"FAILED seed={seed}: {e!r}", flush=True)
            try:
                gp.disposeDefaultEnv()
            except Exception:
                pass
            gc.collect()
    summary_path = ROOT / "result" / "rf_large_401_403_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n======== batch done → {summary_path} ========", flush=True)


if __name__ == "__main__":
    main()
