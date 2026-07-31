"""Round-N smoke: a few small + medium BPC+ML runs with current ml_branch_model.txt."""
from __future__ import annotations

import argparse
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
from configuration import BENCH_MEDIUM, BENCH_SMALL, ML_SB_REFINE_K, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess


def run_one(scale: str, seed: int, limit: float) -> dict:
    T, I = BENCH_SMALL if scale == "small" else BENCH_MEDIUM
    scfg = SetConfig("small_cg", T, 5, 0, I, 5)
    raw = generate_instance(scfg, instance_id=seed % 1000, seed=seed)
    data = preprocess(raw)
    print(
        f"\n=== [{scale}] seed={seed} T={len(data.periods)} I={len(data.cured_items)} "
        f"refine_k={ML_SB_REFINE_K} limit={limit:.0f}s ===",
        flush=True,
    )
    bp = BranchAndPriceSolver(
        data,
        max_nodes=8000,
        time_limit=limit,
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
    proved = res.status == "optimal" or (
        ub is not None and lb is not None and abs(ub - lb) <= 1e-4
    )
    row = {
        "scale": scale,
        "seed": seed,
        "T": T,
        "I": I,
        "time_limit": limit,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--small-seeds", type=str, default="7401,7501")
    ap.add_argument("--medium-seeds", type=str, default="8201")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    cfg.apply_bench_generation()
    model = ROOT / "ml_branch_model.txt"
    assert model.exists(), f"missing {model}"

    small_seeds = [int(x) for x in args.small_seeds.split(",") if x.strip()]
    med_seeds = [int(x) for x in args.medium_seeds.split(",") if x.strip()]
    out = Path(args.out) if args.out else (
        ROOT / "supplement_materials" / f"bpc_ml_iter_round{args.round}.json"
    )

    rows = []
    meta = {
        "round": args.round,
        "model": str(model),
        "model_mtime": model.stat().st_mtime,
        "branch_strategy": "ml",
        "ml_sb_refine_k": ML_SB_REFINE_K,
        "window": dict(cfg.BENCH_WINDOW),
        "scales": {"small": list(BENCH_SMALL), "medium": list(BENCH_MEDIUM)},
        "rows": rows,
    }
    t_all = time.perf_counter()
    for seed in small_seeds:
        try:
            rows.append(run_one("small", seed, 900.0))
        except Exception as e:
            print(f"  FAIL small seed={seed}: {e}", flush=True)
            rows.append({"scale": "small", "seed": seed, "error": str(e)})
        out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for seed in med_seeds:
        if not seed:
            continue
        try:
            rows.append(run_one("medium", seed, 1800.0))
        except Exception as e:
            print(f"  FAIL medium seed={seed}: {e}", flush=True)
            rows.append({"scale": "medium", "seed": seed, "error": str(e)})
        out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"\n=== ROUND {args.round} DONE wall={time.perf_counter() - t_all:.1f}s wrote {out} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
