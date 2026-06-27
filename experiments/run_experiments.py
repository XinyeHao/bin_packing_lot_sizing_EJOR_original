"""批量实验：复现论文 Section 5 数值实验并汇总对比。"""

from __future__ import annotations

import argparse
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
    EXPERIMENT_DEFAULT_COUNT,
    EXPERIMENT_DEFAULT_SETS,
    EXPERIMENT_LOG_FILE,
    EXPERIMENT_METHODS,
    EXPERIMENT_SUMMARY_FILE,
    INSTANCE_BASE_SEED,
    PAPER_REFERENCE,
    SET_EXPERIMENT_LIMITS,
)
from instance_generator.generator import generate_instance_set
from model_and_algo.cgfo import solve_cgfo
from model_and_algo.column_generation import solve_column_generation
from model_and_algo.im_model import solve_im, solve_lim
from preprocessor.preprocess import load_instance, preprocess


def _gap_lrmp_lim(lblim: float, lblrmp: float) -> float:
    return (lblrmp - lblim) / max(lblrmp, lblim) * 100.0


def _ub_gap_cgfo_im(ub_cgfo: float, ub_im: float) -> float:
    return (ub_cgfo - ub_im) / min(ub_cgfo, ub_im) * 100.0


def _load_done_keys(log_path: Path) -> set[tuple[str, str]]:
    if not log_path.exists():
        return set()
    done = set()
    with log_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["instance"], row["method"]))
    return done


def _append_log(log_path: Path, row: dict) -> None:
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_single(instance_path: Path, method: str, set_type: str) -> dict:
    data = preprocess(load_instance(instance_path))
    limits = SET_EXPERIMENT_LIMITS[set_type]
    t0 = time.perf_counter()

    if method == "lim":
        result = solve_lim(data, time_limit=DEFAULT_LIM_TIME_LIMIT)
        value = result.lower_bound
    elif method == "cg":
        result = solve_column_generation(
            data,
            max_iterations=DEFAULT_CG_MAX_ITERATIONS,
            init_time_limit=limits["cg_init"],
        )
        value = result.lower_bound
    elif method == "im_gurobi":
        result = solve_im(
            data,
            time_limit=limits["im_time"],
            mip_gap=DEFAULT_MIP_GAP,
            method="im_gurobi",
        )
        value = result.objective
    elif method == "cgfo_i":
        result = solve_cgfo(
            data,
            variant="cgfo_i",
            init_time_limit=limits["cg_init"],
            fo_time_limit=limits["cgfo_time"],
            mip_gap=DEFAULT_MIP_GAP,
            cg_max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        )
        value = result.objective
    elif method == "cgfo_ii":
        result = solve_cgfo(
            data,
            variant="cgfo_ii",
            init_time_limit=limits["cg_init"],
            fo_time_limit=limits["cgfo_time"],
            mip_gap=DEFAULT_MIP_GAP,
            cg_max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        )
        value = result.objective
    else:
        raise ValueError(method)

    runtime = time.perf_counter() - t0
    return {
        "instance": data.raw.name,
        "set_type": set_type,
        "method": method,
        "status": result.status,
        "value": value,
        "lower_bound": result.lower_bound,
        "runtime_sec": round(runtime, 2),
    }


def run_batch(
    set_types: list[str],
    methods: list[str],
    count: int = EXPERIMENT_DEFAULT_COUNT,
    generate: bool = True,
) -> Path:
    log_path = RESULT_DIR / EXPERIMENT_LOG_FILE
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done_keys(log_path)

    if generate:
        for st in set_types:
            generate_instance_set(st, count=count, base_seed=INSTANCE_BASE_SEED)

    for set_type in set_types:
        for idx in range(1, count + 1):
            instance_path = INSTANCES_DIR / f"set_{set_type.lower()}_{idx:02d}.json"
            for method in methods:
                key = (instance_path.stem, method)
                if key in done:
                    continue
                print(f"[RUN] {instance_path.stem} / {method}", flush=True)
                row = run_single(instance_path, method, set_type)
                _append_log(log_path, row)
                done.add(key)
                print(f"  -> value={row['value']}, time={row['runtime_sec']}s", flush=True)
    return log_path


def summarize(log_path: Path) -> dict:
    rows = list(csv.DictReader(log_path.open(encoding="utf-8")))
    by_set_method: dict[str, dict[str, list[float]]] = {}

    for row in rows:
        if row["value"] in ("", "None", None):
            continue
        st = row["set_type"]
        method = row["method"]
        by_set_method.setdefault(st, {}).setdefault(method, [])
        by_set_method[st][method].append(float(row["value"]))
        by_set_method[st].setdefault(f"{method}_time", []).append(float(row["runtime_sec"]))

    summary: dict = {"sets": {}, "comparison_with_paper": {}}

    for st in ("A", "B"):
        if st not in by_set_method:
            continue
        sdata = by_set_method[st]
        set_summary = {}
        for method in EXPERIMENT_METHODS:
            if method in sdata:
                vals = sdata[method]
                set_summary[method] = {
                    "avg_value": round(sum(vals) / len(vals), 2),
                    "avg_runtime": round(sum(sdata.get(f"{method}_time", [])) / len(vals), 2),
                    "count": len(vals),
                }

        if "lim" in set_summary and "cg" in set_summary:
            lblim = set_summary["lim"]["avg_value"]
            lblrmp = set_summary["cg"]["avg_value"]
            set_summary["cg_vs_lim"] = {
                "avg_lblim": lblim,
                "avg_lblrmp": lblrmp,
                "avg_cg_runtime": set_summary["cg"]["avg_runtime"],
                "gap_lrmp_lim_pct": round(_gap_lrmp_lim(lblim, lblrmp), 2),
            }

        if all(m in set_summary for m in ("im_gurobi", "cgfo_i", "cgfo_ii")):
            ub_im = set_summary["im_gurobi"]["avg_value"]
            ub_i = set_summary["cgfo_i"]["avg_value"]
            ub_ii = set_summary["cgfo_ii"]["avg_value"]
            set_summary["upper_bounds"] = {
                "avg_ub_im": ub_im,
                "avg_ub_cgfo_i": ub_i,
                "avg_ub_cgfo_ii": ub_ii,
                "avg_time_im": set_summary["im_gurobi"]["avg_runtime"],
                "avg_time_cgfo_i": set_summary["cgfo_i"]["avg_runtime"],
                "avg_time_cgfo_ii": set_summary["cgfo_ii"]["avg_runtime"],
                "ub_gap_cgfo_i_pct": round(_ub_gap_cgfo_im(ub_i, ub_im), 2),
                "ub_gap_cgfo_ii_pct": round(_ub_gap_cgfo_im(ub_ii, ub_im), 2),
            }

        summary["sets"][st] = set_summary

        paper_t2 = PAPER_REFERENCE["table2"].get(st, {})
        paper_ub = PAPER_REFERENCE["table_ub_a" if st == "A" else "table_ub_b"]
        paper_t3 = PAPER_REFERENCE["table3"].get(st, {})
        cmp_entry = {}

        if "cg_vs_lim" in set_summary:
            ours = set_summary["cg_vs_lim"]
            cmp_entry["table2_cg"] = {
                "ours": ours,
                "paper": paper_t2,
                "delta_lblim_pct": round((ours["avg_lblim"] - paper_t2.get("lblim", 0)) / paper_t2.get("lblim", 1) * 100, 2),
                "delta_lblrmp_pct": round((ours["avg_lblrmp"] - paper_t2.get("lblrmp", 0)) / paper_t2.get("lblrmp", 1) * 100, 2),
                "delta_gap_pct_points": round(ours["gap_lrmp_lim_pct"] - paper_t2.get("gap", 0), 2),
                "delta_cpu_sec": round(ours["avg_cg_runtime"] - paper_t2.get("cpu", 0), 2),
            }

        if "upper_bounds" in set_summary:
            ours_ub = set_summary["upper_bounds"]
            cmp_entry["table_ub"] = {
                "ours": ours_ub,
                "paper": paper_ub,
                "delta_ub_im_pct": round((ours_ub["avg_ub_im"] - paper_ub["im"]["ub"]) / paper_ub["im"]["ub"] * 100, 2),
                "delta_ub_cgfo_i_gap_points": round(ours_ub["ub_gap_cgfo_i_pct"] - paper_ub["cgfo_i"]["ub_gap"], 2),
                "delta_ub_cgfo_ii_gap_points": round(ours_ub["ub_gap_cgfo_ii_pct"] - paper_ub["cgfo_ii"]["ub_gap"], 2),
            }
            cmp_entry["table3_improvement"] = {
                "paper": paper_t3,
                "ours_cgfo_i_improve_pct": round(-ours_ub["ub_gap_cgfo_i_pct"], 2),
                "ours_cgfo_ii_improve_pct": round(-ours_ub["ub_gap_cgfo_ii_pct"], 2),
            }

        summary["comparison_with_paper"][st] = cmp_entry

    out_path = RESULT_DIR / EXPERIMENT_SUMMARY_FILE
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", default=list(EXPERIMENT_DEFAULT_SETS), choices=["A", "B"])
    parser.add_argument("--methods", nargs="+", default=list(EXPERIMENT_METHODS))
    parser.add_argument("--count", type=int, default=EXPERIMENT_DEFAULT_COUNT)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    log_path = RESULT_DIR / EXPERIMENT_LOG_FILE
    if not args.summarize_only:
        run_batch(args.sets, args.methods, count=args.count, generate=not args.no_generate)

    if log_path.exists():
        summary = summarize(log_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("No experiment log found.")


if __name__ == "__main__":
    main()
