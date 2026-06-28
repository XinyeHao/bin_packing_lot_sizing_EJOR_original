"""10 个 Set-B 算例：并行 CG 重跑（仅 CG，不跑 IM）。"""

from __future__ import annotations

import csv
import json
import statistics as stats
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import INSTANCES_DIR, RESULT_DIR
from configuration import DEFAULT_CG_MAX_ITERATIONS, SET_B, SET_EXPERIMENT_LIMITS
from model_and_algo.column_generation import ColumnGenerationSolver
from preprocessor.preprocess import load_instance, preprocess

COUNT = 10
SET_B_LIMITS = SET_EXPERIMENT_LIMITS["B"]
OUTPUT_JSON = RESULT_DIR / "set_b_cg_10_parallel.json"
OUTPUT_CSV = RESULT_DIR / "set_b_cg_10_parallel.csv"


def run_cg_only() -> dict:
    rows: list[dict] = []
    for idx in range(1, COUNT + 1):
        name = f"set_b_{idx:02d}"
        print(f"\n[{name}] CG ...", flush=True)
        data = preprocess(load_instance(INSTANCES_DIR / f"{name}.json"))

        t0 = time.perf_counter()
        cg = ColumnGenerationSolver(
            data,
            max_iterations=DEFAULT_CG_MAX_ITERATIONS,
            init_time_limit=SET_B_LIMITS["cg_init"],
        )
        lb, rcs, cols = cg.run()
        cg_time = time.perf_counter() - t0
        ncol = sum(len(v) for v in cols.values())
        pricing_times = [d["pricing"] for d in cg.iteration_time_detail]
        iter_times = cg.iteration_times

        row = {
            "instance": name,
            "cg_converged": cg.converged,
            "cg_iters": len(rcs),
            "cg_columns": ncol,
            "lrmp": round(cg.lrmp_obj, 2) if cg.lrmp_obj is not None else None,
            "cg_sec": round(cg_time, 2),
            "sec_per_iter": round(cg_time / len(rcs), 3) if rcs else None,
            "pricing_mean_sec": round(stats.mean(pricing_times), 4) if pricing_times else None,
            "pricing_max_sec": round(max(pricing_times), 4) if pricing_times else None,
            "iter_mean_sec": round(stats.mean(iter_times), 4) if iter_times else None,
            "last_min_rc": cg.last_min_rc,
        }
        rows.append(row)
        print(
            f"  LRMP={row['lrmp']} iters={row['cg_iters']} cols={row['cg_columns']} "
            f"time={row['cg_sec']}s pricing_mean={row['pricing_mean_sec']}s converged={row['cg_converged']}",
            flush=True,
        )

    ok = [r for r in rows if r["cg_converged"]]
    summary = {
        "set_type": "B",
        "parallel_pricing": True,
        "count": len(rows),
        "instances": rows,
        "summary": {
            "cg_converged": len(ok),
            "avg_cg_sec": round(stats.mean([r["cg_sec"] for r in ok]), 2),
            "avg_cg_iters": round(stats.mean([r["cg_iters"] for r in ok]), 1),
            "avg_cg_columns": round(stats.mean([r["cg_columns"] for r in ok]), 1),
            "avg_sec_per_iter": round(stats.mean([r["sec_per_iter"] for r in ok]), 3),
            "avg_pricing_wall_sec": round(stats.mean([r["pricing_mean_sec"] for r in ok]), 4),
            "total_cg_sec": round(sum(r["cg_sec"] for r in rows), 2),
        },
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {OUTPUT_JSON}", flush=True)
    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run_cg_only()
