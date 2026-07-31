"""Batch medium-scale BPC+ML (no IM). Spec: T=30, I=40, M=U=5."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
from configuration import BENCH_MEDIUM, ML_SB_REFINE_K, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess

SEEDS = [8201, 8301, 8401, 8501, 8601]
LIMIT = 1800.0
OUT = ROOT / "supplement_materials" / "bpc_ml_medium_batch.json"


def run_one(seed: int) -> dict:
    T, I = BENCH_MEDIUM
    scfg = SetConfig("small_cg", T, 5, 0, I, 5)
    raw = generate_instance(scfg, instance_id=seed % 1000, seed=seed)
    data = preprocess(raw)
    print(
        f"\n=== seed={seed} T={len(data.periods)} I={len(data.cured_items)} "
        f"refine_k={ML_SB_REFINE_K} limit={LIMIT:.0f}s ===",
        flush=True,
    )
    bp = BranchAndPriceSolver(
        data,
        max_nodes=8000,
        time_limit=LIMIT,
        mip_gap=0.0,
        branch_strategy="ml",
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
            ebrp_time_limit=25.0,
        ),
    )
    t0 = time.perf_counter()
    res = bp.solve()
    elapsed = time.perf_counter() - t0
    ub, lb = res.objective, res.lower_bound
    gap = None
    if ub is not None and lb is not None and ub > 1e-9:
        gap = 100.0 * max(0.0, (ub - lb) / ub)
    proved = res.status == "optimal" or (
        ub is not None and lb is not None and abs(ub - lb) <= 1e-4
    )
    row = {
        "seed": seed,
        "T": T,
        "I": I,
        "status": res.status,
        "ub": ub,
        "lb": lb,
        "gap_pct": None if gap is None else round(gap, 4),
        "runtime_sec": round(elapsed, 2),
        "nodes_explored": res.nodes_explored,
        "nodes_pruned": res.nodes_pruned,
        "proved": proved,
        "ml_sb_refine_k": ML_SB_REFINE_K,
    }
    print(
        f"  -> status={row['status']} ub={ub} lb={lb} gap={row['gap_pct']} "
        f"nodes={row['nodes_explored']} sec={row['runtime_sec']} proved={proved}",
        flush=True,
    )
    try:
        data.dispose()
    except Exception:
        pass
    return row


def main():
    cfg.apply_bench_generation()
    rows = []
    t_all = time.perf_counter()
    for seed in SEEDS:
        try:
            rows.append(run_one(seed))
        except Exception as e:
            print(f"  FAIL seed={seed}: {e}", flush=True)
            rows.append({"seed": seed, "error": str(e), "proved": False})
        OUT.write_text(
            json.dumps(
                {
                    "scale": "medium",
                    "no_im": True,
                    "time_limit": LIMIT,
                    "branch_strategy": "ml",
                    "ml_sb_refine_k": ML_SB_REFINE_K,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    n_ok = sum(1 for r in rows if r.get("proved"))
    times = [r["runtime_sec"] for r in rows if "runtime_sec" in r]
    print("\n=== SUMMARY ===", flush=True)
    print(
        f"proved {n_ok}/{len(rows)} runtime_sum={sum(times):.1f}s "
        f"wall={time.perf_counter()-t_all:.1f}s",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
