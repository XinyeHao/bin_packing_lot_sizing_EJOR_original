"""BPC frac_sb (frac Top-5 + short SB) vs IM on 5 medium + 2 large instances.

Cleans Gurobi / Python memory after each solve to reduce OOM risk.
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
from configuration import (
    BENCH_LARGE,
    BENCH_MEDIUM,
    FRAC_SB_REFINE_K,
    SetConfig,
)
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import preprocess

# 5 medium + 2 large
JOBS = [
    ("medium", 30101, 1800.0),
    ("medium", 30201, 1800.0),
    ("medium", 30301, 1800.0),
    ("medium", 30401, 1800.0),
    ("medium", 30501, 1800.0),
    ("large", 40101, 3600.0),
    ("large", 40201, 3600.0),
]
OUT = ROOT / "supplement_materials" / "bpc_frac_vs_im.json"


def _gap(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return round(100.0 * max(0.0, (ub - lb) / ub), 4)


def _cleanup(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass
    gc.collect()


def run_bpc(data, limit: float) -> dict:
    bp = BranchAndPriceSolver(
        data,
        max_nodes=8000,
        time_limit=limit,
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
    row = {
        "status": res.status,
        "ub": ub,
        "lb": lb,
        "gap_pct": _gap(ub, lb),
        "runtime_sec": round(elapsed, 2),
        "nodes_explored": res.nodes_explored,
        "nodes_pruned": res.nodes_pruned,
        "proved": res.status == "optimal"
        or (ub is not None and lb is not None and abs(ub - lb) <= 1e-4),
    }
    _cleanup(bp, res)
    return row


def run_im(data, limit: float) -> dict:
    t0 = time.perf_counter()
    im = solve_im(data, time_limit=limit, mip_gap=0.0, verbose=False)
    elapsed = time.perf_counter() - t0
    ub, lb = im.objective, im.lower_bound
    row = {
        "status": im.status,
        "ub": ub,
        "lb": lb,
        "gap_pct": _gap(ub, lb),
        "runtime_sec": round(elapsed, 2),
        "proved": im.status == "optimal"
        or (ub is not None and lb is not None and abs(ub - lb) <= 1e-4),
    }
    _cleanup(im)
    return row


def main():
    cfg.apply_bench_generation()
    rows = []
    meta = {
        "note": "BPC frac_sb (frac Top-K + short SB refine) vs IM",
        "frac_sb_refine_k": FRAC_SB_REFINE_K,
        "window": dict(cfg.BENCH_WINDOW),
        "scales": {"medium": list(BENCH_MEDIUM), "large": list(BENCH_LARGE)},
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for scale, seed, limit in JOBS:
        T, I = BENCH_MEDIUM if scale == "medium" else BENCH_LARGE
        print(
            f"\n======== {scale} seed={seed} T={T} I={I} limit={limit:.0f}s "
            f"frac_sb k={FRAC_SB_REFINE_K} ========",
            flush=True,
        )
        raw = generate_instance(SetConfig("small_cg", T, 5, 0, I, 5), seed % 1000, seed)
        data = preprocess(raw)
        print(
            f"generated periods={len(data.periods)} items={len(data.cured_items)}",
            flush=True,
        )

        print("--- BPC frac_sb ---", flush=True)
        bpc = run_bpc(data, limit)
        print(
            f"  BPC -> {bpc['status']} ub={bpc['ub']} lb={bpc['lb']} "
            f"gap={bpc['gap_pct']} nodes={bpc.get('nodes_explored')} "
            f"sec={bpc['runtime_sec']}",
            flush=True,
        )

        # regenerate fresh preprocess for IM (BPC may leave state); safer: new data
        _cleanup(data, raw)
        raw = generate_instance(SetConfig("small_cg", T, 5, 0, I, 5), seed % 1000, seed)
        data = preprocess(raw)

        print("--- IM ---", flush=True)
        im = run_im(data, limit)
        print(
            f"  IM  -> {im['status']} ub={im['ub']} lb={im['lb']} "
            f"gap={im['gap_pct']} sec={im['runtime_sec']}",
            flush=True,
        )

        row = {
            "scale": scale,
            "seed": seed,
            "T": T,
            "I": I,
            "time_limit": limit,
            "bpc": bpc,
            "im": im,
        }
        rows.append(row)
        OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _cleanup(data, raw)
        print(f"[checkpoint] wrote {OUT} ({len(rows)}/{len(JOBS)})", flush=True)

    print("\n=== ALL DONE ===", flush=True)
    for r in rows:
        print(
            f"{r['scale']} seed={r['seed']}: "
            f"BPC gap={r['bpc'].get('gap_pct')} / IM gap={r['im'].get('gap_pct')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
