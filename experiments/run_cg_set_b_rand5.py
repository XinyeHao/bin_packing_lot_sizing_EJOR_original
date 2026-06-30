"""随机生成 5 个 Set-B 算例，测试 CG（LRMP）与 IM 的质量和效率。"""

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
from configuration import (
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_MIP_GAP,
    SET_B,
    SET_EXPERIMENT_LIMITS,
)
from instance_generator.generator import generate_instance
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess

COUNT = 5
BASE_SEED = 9000
NAME_PREFIX = "set_b_rand"
SET_B_LIMITS = SET_EXPERIMENT_LIMITS["B"]
IM_TIME_LIMIT = SET_B_LIMITS["im_time"]
OUTPUT_JSON = RESULT_DIR / "set_b_cg_rand5_benchmark.json"
OUTPUT_CSV = RESULT_DIR / "set_b_cg_rand5_benchmark.csv"


def generate_instances() -> list[str]:
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for idx in range(1, COUNT + 1):
        name = f"{NAME_PREFIX}_{idx:02d}"
        seed = BASE_SEED + idx
        inst = generate_instance(SET_B, idx, seed=seed)
        inst.name = name
        path = INSTANCES_DIR / f"{name}.json"
        path.write_text(json.dumps(inst.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"generated {name} (seed={seed})", flush=True)
        names.append(name)
    return names


def run_benchmark() -> dict:
    names = generate_instances()
    rows: list[dict] = []

    for name in names:
        print(f"\n[{name}] ...", flush=True)
        data = preprocess(load_instance(INSTANCES_DIR / f"{name}.json"))
        total_demand = sum(sum(row) for row in data.d_it)

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

        t1 = time.perf_counter()
        im = solve_im(
            data,
            time_limit=IM_TIME_LIMIT,
            mip_gap=DEFAULT_MIP_GAP,
            method="im_gurobi",
        )
        im_time = time.perf_counter() - t1

        lrmp = cg.lrmp_obj
        im_ub = im.objective
        gap_pct = None
        if im_ub is not None and lrmp is not None and im_ub > 1e-6:
            gap_pct = (im_ub - lrmp) / im_ub * 100.0

        row = {
            "instance": name,
            "seed": data.raw.seed,
            "total_demand": int(total_demand),
            "cg_converged": cg.converged,
            "cg_iters": len(rcs),
            "cg_columns": ncol,
            "lrmp": round(lrmp, 2) if lrmp is not None else None,
            "cg_sec": round(cg_time, 2),
            "sec_per_iter": round(cg_time / len(rcs), 3) if rcs else None,
            "pricing_mean_sec": round(stats.mean(pricing_times), 4) if pricing_times else None,
            "im_ub": round(im_ub, 2) if im_ub is not None else None,
            "im_status": im.status,
            "im_sec": round(im_time, 2),
            "im_gap_pct": round(im.mip_gap * 100, 2) if im.mip_gap is not None else None,
            "lrmp_gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
            "last_min_rc": cg.last_min_rc,
        }
        rows.append(row)
        gap_s = f"{gap_pct:.2f}%" if gap_pct is not None else "n/a"
        print(
            f"  CG: LRMP={lrmp:.2f} iters={len(rcs)} cols={ncol} "
            f"time={cg_time:.1f}s converged={cg.converged}",
            flush=True,
        )
        print(
            f"  IM: UB={im_ub} status={im.status} time={im_time:.1f}s "
            f"mip_gap={row['im_gap_pct']}% lrmp_gap={gap_s}",
            flush=True,
        )

    ok = [r for r in rows if r["cg_converged"]]
    with_gap = [r for r in rows if r["lrmp_gap_pct"] is not None]
    optimal_im = [r for r in rows if r["im_status"] == "optimal"]

    summary = {
        "set_type": "B",
        "count": len(rows),
        "base_seed": BASE_SEED,
        "scale": {
            "periods": SET_B.num_periods,
            "machines": SET_B.num_machines,
            "items": SET_B.num_cured_items,
            "configs": SET_B.num_configs,
        },
        "cg_init_limit": SET_B_LIMITS["cg_init"],
        "im_time_limit": IM_TIME_LIMIT,
        "instances": rows,
        "summary": {
            "cg_converged": len(ok),
            "im_optimal": len(optimal_im),
            "avg_cg_sec": round(stats.mean([r["cg_sec"] for r in ok]), 2) if ok else None,
            "avg_cg_iters": round(stats.mean([r["cg_iters"] for r in ok]), 1) if ok else None,
            "avg_cg_columns": round(stats.mean([r["cg_columns"] for r in ok]), 1) if ok else None,
            "avg_sec_per_iter": round(stats.mean([r["sec_per_iter"] for r in ok]), 3) if ok else None,
            "avg_lrmp_gap_pct": round(stats.mean([r["lrmp_gap_pct"] for r in with_gap]), 2) if with_gap else None,
            "avg_lrmp_gap_optimal_im": round(
                stats.mean([r["lrmp_gap_pct"] for r in optimal_im if r["lrmp_gap_pct"] is not None]), 2
            )
            if optimal_im
            else None,
            "max_lrmp_gap_pct": max((r["lrmp_gap_pct"] for r in with_gap), default=None),
            "min_lrmp_gap_pct": min((r["lrmp_gap_pct"] for r in with_gap), default=None),
            "avg_im_sec": round(stats.mean([r["im_sec"] for r in rows]), 2),
            "total_cg_sec": round(sum(r["cg_sec"] for r in rows), 2),
            "total_im_sec": round(sum(r["im_sec"] for r in rows), 2),
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
    run_benchmark()
