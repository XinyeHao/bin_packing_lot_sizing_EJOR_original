"""Set-B 小规模测试（5 个算例）。"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import INSTANCES_DIR, RESULT_DIR
from configuration import (
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_LIM_TIME_LIMIT,
    DEFAULT_MIP_GAP,
    PAPER_REFERENCE,
    SET_B_TEST_COUNT,
    SET_B_TEST_LOG,
    SET_B_TEST_METHODS,
    SET_B_TEST_SUMMARY,
    SET_EXPERIMENT_LIMITS,
)
from model_and_algo.cgfo import solve_cgfo
from model_and_algo.column_generation import solve_column_generation
from model_and_algo.im_model import solve_im, solve_lim
from postprocessor.postprocess import save_result
from preprocessor.preprocess import load_instance, preprocess

SET_B_LIMITS = SET_EXPERIMENT_LIMITS["B"]


def run_set_b_test(count: int = SET_B_TEST_COUNT) -> Path:
    log_path = RESULT_DIR / SET_B_TEST_LOG
    summary_path = RESULT_DIR / SET_B_TEST_SUMMARY
    rows: list[dict] = []

    for idx in range(1, count + 1):
        name = f"set_b_{idx:02d}"
        path = INSTANCES_DIR / f"{name}.json"
        data = preprocess(load_instance(path))
        print(f"[{name}] start", flush=True)

        for method in SET_B_TEST_METHODS:
            t0 = time.perf_counter()
            try:
                if method == "lim":
                    r = solve_lim(data, time_limit=DEFAULT_LIM_TIME_LIMIT)
                    value = r.lower_bound
                elif method == "cg":
                    r = solve_column_generation(
                        data,
                        max_iterations=DEFAULT_CG_MAX_ITERATIONS,
                        init_time_limit=SET_B_LIMITS["cg_init"],
                    )
                    value = r.lower_bound
                elif method == "im_gurobi":
                    r = solve_im(
                        data,
                        time_limit=SET_B_LIMITS["im_time"],
                        mip_gap=DEFAULT_MIP_GAP,
                        method="im_gurobi",
                    )
                    value = r.objective
                elif method == "cgfo_ii":
                    r = solve_cgfo(
                        data,
                        variant="cgfo_ii",
                        init_time_limit=SET_B_LIMITS["cg_init"],
                        fo_time_limit=SET_B_LIMITS["cgfo_time"],
                        cg_max_iterations=DEFAULT_CG_MAX_ITERATIONS,
                    )
                    value = r.objective
                else:
                    continue
                save_result(r, RESULT_DIR)
            except Exception as exc:
                row = {
                    "instance": name,
                    "set_type": "B",
                    "method": method,
                    "status": "error",
                    "value": "",
                    "lower_bound": "",
                    "runtime_sec": round(time.perf_counter() - t0, 2),
                    "error": str(exc),
                }
                rows.append(row)
                print(f"  {method} ERROR: {exc}", flush=True)
                continue

            runtime = time.perf_counter() - t0
            row = {
                "instance": name,
                "set_type": "B",
                "method": method,
                "status": r.status,
                "value": value,
                "lower_bound": r.lower_bound,
                "runtime_sec": round(runtime, 2),
                "extra_iters": r.extra.get("num_iterations", ""),
                "extra_cols": sum(r.extra.get("num_columns", {}).values()) if r.extra.get("num_columns") else "",
            }
            rows.append(row)
            print(f"  {method}: value={value}, time={runtime:.1f}s, status={r.status}", flush=True)

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows, count)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return log_path


def _summarize(rows: list[dict], count: int) -> dict:
    from statistics import mean

    by_method: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") == "error":
            continue
        by_method.setdefault(row["method"], []).append(row)

    summary: dict = {"instances": count, "methods": {}}
    for method, items in by_method.items():
        vals = [float(r["value"]) for r in items if r["value"] not in ("", None)]
        times = [float(r["runtime_sec"]) for r in items]
        summary["methods"][method] = {
            "avg_value": round(mean(vals), 2) if vals else None,
            "avg_runtime_sec": round(mean(times), 2),
            "count": len(items),
        }

    if "im_gurobi" in summary["methods"] and "cgfo_ii" in summary["methods"]:
        gaps = []
        for idx in range(1, count + 1):
            name = f"set_b_{idx:02d}"
            im = next(float(r["value"]) for r in rows if r["instance"] == name and r["method"] == "im_gurobi")
            c2 = next(float(r["value"]) for r in rows if r["instance"] == name and r["method"] == "cgfo_ii")
            gaps.append((c2 - im) / min(c2, im) * 100)
        summary["cgfo_ii_vs_im_gap_pct"] = round(mean(gaps), 2)

    if "lim" in summary["methods"] and "cg" in summary["methods"]:
        gaps = []
        for idx in range(1, count + 1):
            name = f"set_b_{idx:02d}"
            lim = next(float(r["value"]) for r in rows if r["instance"] == name and r["method"] == "lim")
            cg = next(float(r["value"]) for r in rows if r["instance"] == name and r["method"] == "cg")
            gaps.append((cg - lim) / max(cg, lim) * 100)
        summary["cg_vs_lim_gap_pct"] = round(mean(gaps), 2)

    summary["paper_set_b_reference"] = PAPER_REFERENCE["set_b_quick_test"]
    return summary


if __name__ == "__main__":
    run_set_b_test(SET_B_TEST_COUNT)
