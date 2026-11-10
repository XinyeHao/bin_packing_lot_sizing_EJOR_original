"""Diagnostic only: compare root CG bounds for seed 10110.

This script does not alter benchmark results.  A valid LP/Lagrangian lower
bound must not exceed the compact-model optimum 27987.8 for this instance.
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import configuration as cfg
from configuration import SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.column_generation import ColumnGenerationSolver
from preprocessor.preprocess import preprocess

SEED = 10110
IM_OPTIMUM = 27987.8


def run_case(name: str, *, pricing_mode: str, cuts, limit: float) -> dict:
    raw = generate_instance(SetConfig("small_cg", 30, 5, 0, 40, 5), SEED % 1000, SEED)
    data = preprocess(raw)
    cg = ColumnGenerationSolver(
        data,
        max_iterations=1000,
        time_limit=limit,
        init_time_limit=90.0,
        pricing_mode=pricing_mode,
        ebrp_force=(pricing_mode == "ebrp"),
        ebrp_time_limit=None,
        im_pricing_time_limit=120.0,
        use_gurobi_init=True,
        # Do not suppress the built-in EBRP-versus-Gurobi pricing check.
        # This is the key diagnostic for an allegedly converged root node.
        skip_cross_check=False,
        pricing_inprocess=True,
        verbose=False,
        lrmp_cuts=cuts,
    )
    t0 = time.perf_counter()
    cg.run()
    elapsed = time.perf_counter() - t0
    lb = cg.last_valid_lb
    delta = None if lb is None else lb - IM_OPTIMUM
    delta_text = "None" if delta is None else f"{delta:.6f}"
    print(
        f"{name}: converged={cg.converged}, stop={cg.stop_reason}, "
        f"lrmp={cg.lrmp_obj}, valid_lb={lb}, min_rc={cg.last_min_rc}, "
        f"gap_to_IMopt={delta_text}, "
        f"time={elapsed:.2f}s",
        flush=True,
    )
    return {
        "case": name,
        "pricing_mode": pricing_mode,
        "cuts": cuts,
        "converged": cg.converged,
        "stop_reason": cg.stop_reason,
        "lrmp_obj": cg.lrmp_obj,
        "last_valid_lb": lb,
        "last_min_rc": cg.last_min_rc,
        "delta_to_im_optimum": delta,
        "runtime_sec": elapsed,
    }


def main() -> None:
    cfg.apply_bench_generation()
    rows = [
        run_case("EBRP cross-checked + standard cuts", pricing_mode="ebrp", cuts="standard", limit=180.0),
        run_case("EBRP cross-checked + no cuts", pricing_mode="ebrp", cuts=None, limit=180.0),
    ]
    out = ROOT / "result" / "diagnose_seed_10110.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
