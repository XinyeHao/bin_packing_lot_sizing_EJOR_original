"""Retest medium seed=30301 with frac_sb after CG-fail / scrap_r0 fixes."""
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
from configuration import BENCH_MEDIUM, FRAC_SB_REFINE_K, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess

SEED = 30301
LIMIT = 1800.0
OUT = ROOT / "supplement_materials" / "bpc_frac_30301_retest.json"


def main():
    cfg.apply_bench_generation()
    T, I = BENCH_MEDIUM
    print(
        f"=== retest seed={SEED} T={T} I={I} limit={LIMIT}s "
        f"frac_sb k={FRAC_SB_REFINE_K} ===",
        flush=True,
    )
    raw = generate_instance(SetConfig("small_cg", T, 5, 0, I, 5), SEED % 1000, SEED)
    data = preprocess(raw)
    bp = BranchAndPriceSolver(
        data,
        max_nodes=8000,
        time_limit=LIMIT,
        mip_gap=0.0,
        branch_strategy="frac_sb",
        use_node_im=False,
        root_im=False,
        final_im=False,
        cg_kwargs=dict(
            pricing_mode="ebrp",
            ebrp_force=True,
            use_gurobi_init=True,
            skip_cross_check=True,
            pricing_inprocess=True,
            verbose=False,
            progress_interval_sec=120.0,
            lrmp_cuts="standard",
            max_iterations=120,
            ebrp_time_limit=20.0,
        ),
    )
    t0 = time.perf_counter()
    res = bp.solve()
    elapsed = time.perf_counter() - t0
    ub, lb = res.objective, res.lower_bound
    gap = None
    if ub is not None and lb is not None and ub > 1e-9:
        gap = 100.0 * max(0.0, (ub - lb) / ub)
    row = {
        "seed": SEED,
        "T": T,
        "I": I,
        "time_limit": LIMIT,
        "status": res.status,
        "ub": ub,
        "lb": lb,
        "gap_pct": None if gap is None else round(gap, 4),
        "runtime_sec": round(elapsed, 2),
        "nodes_explored": res.nodes_explored,
        "nodes_pruned": res.nodes_pruned,
        "nodes_cg_failed": res.nodes_cg_failed,
        "proved": res.status == "optimal"
        or (ub is not None and lb is not None and abs(ub - lb) <= 1e-4),
        "note": "after scrap_r0 inherit + CG-fail retry + larger budgets",
    }
    print(
        f"-> status={row['status']} ub={ub} lb={lb} gap={row['gap_pct']} "
        f"nodes={row['nodes_explored']} cg_fail={row['nodes_cg_failed']} "
        f"sec={row['runtime_sec']} proved={row['proved']}",
        flush=True,
    )
    OUT.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    del bp, res, data, raw
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass


if __name__ == "__main__":
    main()
