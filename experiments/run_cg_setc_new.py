"""Set-C 新算例 CG 效率对比（默认 Gurobi 定价 vs EBRP 限时）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuration import SET_C_CG, SET_C_CG_TIME_LIMIT
from model_and_algo.column_generation import solve_column_generation
from preprocessor.preprocess import load_instance, preprocess


def run_cg(label: str, instance: str, extra: dict) -> dict:
    data = preprocess(load_instance(instance))
    opts = {
        **SET_C_CG,
        "time_limit": SET_C_CG_TIME_LIMIT,
        "verbose": True,
        **extra,
    }
    print(f"\n{'='*60}\n=== CG [{label}] instance={instance} ===\n{'='*60}", flush=True)
    t0 = time.perf_counter()
    result = solve_column_generation(data, **opts)
    elapsed = time.perf_counter() - t0
    extra_info = result.extra or {}
    summary = {
        "label": label,
        "status": result.status,
        "lb": result.lower_bound,
        "lrmp": extra_info.get("lrmp_obj"),
        "iters": extra_info.get("num_iterations"),
        "min_rc": extra_info.get("last_min_rc"),
        "converged": extra_info.get("converged"),
        "time": elapsed,
        "cols": sum((extra_info.get("num_columns") or {}).values()),
    }
    print(f"\n--- {label} DONE ---", summary, flush=True)
    return summary


def main() -> None:
    instance = "set_c_01_new"
    runs = [
        (
            "default_gurobi",
            {},
        ),
        (
            "ebrp_force_1s",
            {
                "pricing_mode": "ebrp_force",
                "ebrp_force": True,
                "ebrp_time_limit": 1.0,
            },
        ),
    ]
    summaries = []
    for label, extra in runs:
        summaries.append(run_cg(label, instance, extra))

    print("\n=== CG 对比摘要 ===")
    for s in summaries:
        print(
            f"{s['label']:20s}  iters={s['iters']:3d}  "
            f"LRMP={s['lrmp'] or s['lb']:.1f}  min_RC={s['min_rc']:.2f}  "
            f"time={s['time']:.0f}s  converged={s['converged']}"
        )


if __name__ == "__main__":
    main()
