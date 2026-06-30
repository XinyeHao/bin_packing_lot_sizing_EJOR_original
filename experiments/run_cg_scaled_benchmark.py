"""列生成规模测试：在 Set-B 基础上放大计划期 / 产品数 / 机器数。"""

from __future__ import annotations

import csv
import json
import statistics as stats
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import INSTANCES_DIR, RESULT_DIR
from configuration import DEFAULT_CG_MAX_ITERATIONS, SET_B, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess

OUTPUT_JSON = RESULT_DIR / "set_b_cg_scaled_benchmark.json"
OUTPUT_CSV = RESULT_DIR / "set_b_cg_scaled_benchmark.csv"

# 相对 Set-B (25期, 35品, 7机, 10配置) 的放大方案
SCALED_CONFIGS: dict[str, SetConfig] = {
    "baseline": SET_B,
    "periods_40": SetConfig("B", 40, 10, 7, 35, 7),
    "items_50": SetConfig("B", 25, 10, 7, 50, 7),
    "machines_10": SetConfig("B", 25, 10, 7, 35, 10),
    "large": SetConfig("B", 40, 12, 8, 50, 10),
}

BASELINE_INSTANCE = "set_b_01"
IM_TIME_LIMIT = 600.0
CG_INIT_TIME = 180.0


@dataclass(frozen=True)
class ScaleSpec:
    label: str
    config: SetConfig
    instance_name: str
    seed: int
    use_existing: bool = False


def _specs() -> list[ScaleSpec]:
    return [
        ScaleSpec("baseline", SET_B, BASELINE_INSTANCE, 43, use_existing=True),
        ScaleSpec("periods_40", SCALED_CONFIGS["periods_40"], "scale_periods_40_01", 1040),
        ScaleSpec("items_50", SCALED_CONFIGS["items_50"], "scale_items_50_01", 1050),
        ScaleSpec("machines_10", SCALED_CONFIGS["machines_10"], "scale_machines_10_01", 1010),
        ScaleSpec("large", SCALED_CONFIGS["large"], "scale_large_01", 1100),
    ]


def _ensure_instance(spec: ScaleSpec) -> None:
    path = INSTANCES_DIR / f"{spec.instance_name}.json"
    if spec.use_existing:
        if not path.exists():
            raise FileNotFoundError(path)
        return
    instance = generate_instance(spec.config, 1, seed=spec.seed)
    instance.name = spec.instance_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _run_cg(data, init_time_limit: float) -> dict:
    t0 = time.perf_counter()
    cg = ColumnGenerationSolver(
        data,
        max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        init_time_limit=init_time_limit,
    )
    lb, rcs, cols = cg.run()
    elapsed = time.perf_counter() - t0
    pricing_times = [d["pricing"] for d in cg.iteration_time_detail]
    return {
        "cg_converged": cg.converged,
        "cg_iters": len(rcs),
        "cg_columns": sum(len(v) for v in cols.values()),
        "lrmp": round(cg.lrmp_obj, 2) if cg.lrmp_obj is not None else None,
        "cg_sec": round(elapsed, 2),
        "sec_per_iter": round(elapsed / len(rcs), 3) if rcs else None,
        "pricing_mean_sec": round(stats.mean(pricing_times), 4) if pricing_times else None,
        "last_min_rc": cg.last_min_rc,
    }


def _run_im(data, time_limit: float) -> dict:
    t0 = time.perf_counter()
    result = solve_im(data, time_limit=time_limit, mip_gap=0.05, method="im_gurobi")
    elapsed = time.perf_counter() - t0
    return {
        "im_status": result.status,
        "im_obj": round(result.objective, 2) if result.objective is not None else None,
        "im_sec": round(elapsed, 2),
        "im_gap_pct": round(result.mip_gap * 100, 2) if result.mip_gap is not None else None,
    }


def run_benchmark() -> dict:
    rows: list[dict] = []
    for spec in _specs():
        print(f"\n=== {spec.label} ({spec.instance_name}) ===", flush=True)
        _ensure_instance(spec)
        data = preprocess(load_instance(INSTANCES_DIR / f"{spec.instance_name}.json"))
        cfg = spec.config
        row = {
            "scale": spec.label,
            "instance": spec.instance_name,
            "periods": cfg.num_periods,
            "items": cfg.num_cured_items,
            "machines": cfg.num_machines,
            "configs": cfg.num_configs,
        }
        print(
            f"  size: T={row['periods']} items={row['items']} "
            f"machines={row['machines']} configs={row['configs']}",
            flush=True,
        )

        cg_row = _run_cg(data, CG_INIT_TIME)
        row.update(cg_row)
        print(
            f"  CG: LRMP={row['lrmp']} iters={row['cg_iters']} cols={row['cg_columns']} "
            f"time={row['cg_sec']}s converged={row['cg_converged']}",
            flush=True,
        )

        print(f"  IM (limit={IM_TIME_LIMIT}s) ...", flush=True)
        im_row = _run_im(data, IM_TIME_LIMIT)
        row.update(im_row)
        if row["lrmp"] is not None and row["im_obj"] is not None:
            row["lrmp_gap_pct"] = round((row["im_obj"] - row["lrmp"]) / row["im_obj"] * 100, 2)
        else:
            row["lrmp_gap_pct"] = None
        print(
            f"  IM: obj={row['im_obj']} time={row['im_sec']}s gap={row['im_gap_pct']}% "
            f"LRMP gap={row['lrmp_gap_pct']}%",
            flush=True,
        )
        rows.append(row)

    converged = [r for r in rows if r["cg_converged"]]
    summary = {
        "cg_init_time_limit": CG_INIT_TIME,
        "im_time_limit": IM_TIME_LIMIT,
        "instances": rows,
        "summary": {
            "count": len(rows),
            "cg_converged": len(converged),
            "avg_cg_sec": round(stats.mean([r["cg_sec"] for r in converged]), 2) if converged else None,
            "max_cg_sec": max(r["cg_sec"] for r in rows),
            "avg_lrmp_gap_pct": round(
                stats.mean([r["lrmp_gap_pct"] for r in rows if r["lrmp_gap_pct"] is not None]), 2
            )
            if any(r["lrmp_gap_pct"] is not None for r in rows)
            else None,
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
