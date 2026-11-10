"""Experiment 1: compact IM-LP versus root DW-CG lower bounds.

This follows the stopping design used in Hao et al. (2022): initial columns
receive scale-dependent time budgets, then root CG stops at 500 iterations or
when the sum of pricing reduced costs exceeds -0.1.  The Lagrangian correction
``LRMP + sum(min(RC_m, 0))`` is recorded as a valid CG lower bound at every
completed pricing round; full CG convergence is reported separately.
"""
from __future__ import annotations

import argparse
import csv
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
from experiments.paper_experiment_config import (
    MODEL_VERSION,
    PROCESS_ASSUMPTION,
    SCALES,
    generation_metadata,
    make_set_config,
)
from instance_generator.generator import generate_instance
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_lim
from preprocessor.preprocess import preprocess


INITIAL_COLUMN_TIME_LIMIT = {"small": 60.0, "medium": 120.0, "large": 180.0}
E1_EXPERIMENT_VERSION = "root_cg_vs_all_continuous_im_lp_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E1: paper-style IM-LP versus root DW-CG lower-bound comparison."
    )
    parser.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Optional seed subset. Use only seeds belonging to the selected scale(s).",
    )
    parser.add_argument("--output-prefix", default="exp1_root_cg_vs_im_lp")
    parser.add_argument(
        "--im-lp-time-limit", type=float, default=None,
        help="Optional common IM-LP limit. Default: no limit, because LIM is a pure LP.",
    )
    parser.add_argument(
        "--max-cg-iterations", type=int, default=500,
        help="CG iteration cap; default 500 follows the reference experiment.",
    )
    parser.add_argument(
        "--rerun-incomplete", action="store_true",
        help="Retry rows lacking either valid lower bound.",
    )
    return parser.parse_args()


def cleanup(*objects: object) -> None:
    for obj in objects:
        try:
            if hasattr(obj, "dispose"):
                obj.dispose()
        except Exception:
            pass
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass


def run_one(scale: str, seed: int, args: argparse.Namespace) -> dict:
    spec = SCALES[scale]
    # LIM is a genuine all-continuous LP.  It is solved to optimality by
    # default; an explicit limit is available only as a diagnostic option.
    im_lp_time_limit = args.im_lp_time_limit
    row: dict = {
        "scale": scale, "seed": seed, "T": spec.periods, "I": spec.items,
        "im_lp_time_limit_sec": im_lp_time_limit,
        "im_lp_all_continuous": True,
        "im_lp_status": None, "im_lp_lb": None, "im_lp_time_sec": None,
        "root_cg_status": None, "root_cg_valid_lb": None, "root_cg_rmp_value": None,
        "root_cg_time_sec": None,
        "root_cg_iterations": None, "root_cg_columns": None,
        "root_cg_last_min_rc": None, "root_cg_last_valid_lb": None,
        "both_valid_bounds": False, "both_exact": False, "dw_minus_im_lp": None,
        "dw_over_im_lp_pct": None, "error": None,
    }
    raw = data = cg = None
    try:
        cfg.apply_bench_generation()
        raw = generate_instance(make_set_config(spec), seed % 1000, seed)
        data = preprocess(raw)

        start = time.perf_counter()
        im = solve_lim(data, time_limit=im_lp_time_limit, verbose=False)
        row["im_lp_time_sec"] = round(time.perf_counter() - start, 3)
        row["im_lp_status"] = im.status
        # ObjBound is a valid minimisation lower bound even if the LP solve is
        # interrupted.  The status separately reveals whether it is exact.
        row["im_lp_lb"] = im.lower_bound
        row["im_lp_exact"] = im.status == "optimal"

        cg = ColumnGenerationSolver(
            data,
            max_iterations=args.max_cg_iterations,
            init_time_limit=INITIAL_COLUMN_TIME_LIMIT[scale],
            time_limit=None,
            pricing_mode="ebrp",
            ebrp_force=True,
            use_gurobi_init=True,
            pricing_inprocess=True,
            skip_cross_check=True,
            lrmp_cuts="none",
            ebrp_time_limit=None,
            sum_rc_stop_threshold=-0.1,
            verbose=True,
            progress_interval_sec=30.0,
        )
        start = time.perf_counter()
        _, rcs, columns = cg.run()
        row["root_cg_time_sec"] = round(time.perf_counter() - start, 3)
        row["root_cg_status"] = "converged" if cg.converged else (cg.stop_reason or "not_converged")
        row["root_cg_rmp_value"] = cg.lrmp_obj
        row["root_cg_valid_lb"] = getattr(cg, "last_valid_lb", None)
        row["root_cg_last_valid_lb"] = getattr(cg, "last_valid_lb", None)
        row["root_cg_iterations"] = len(rcs)
        row["root_cg_columns"] = sum(len(cols) for cols in columns.values())
        row["root_cg_last_min_rc"] = cg.last_min_rc

        if row["im_lp_lb"] is not None and row["root_cg_valid_lb"] is not None:
            delta = float(row["root_cg_valid_lb"]) - float(row["im_lp_lb"])
            row["both_valid_bounds"] = True
            row["both_exact"] = bool(row["im_lp_exact"] and cg.converged)
            row["dw_minus_im_lp"] = delta
            # Follow the paper-style comparison: the stronger of the two
            # lower bounds is the denominator.  A positive value means that
            # root DW-CG yields a tighter relaxation than compact IM-LP.
            row["dw_over_im_lp_pct"] = 100.0 * delta / max(
                abs(float(row["root_cg_valid_lb"])), abs(float(row["im_lp_lb"])), 1e-6
            )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if cg is not None and cg._model is not None:
                cg._model.dispose()
        except Exception:
            pass
        cleanup(data, raw)
    return row


def write_results(rows: list[dict], args: argparse.Namespace, json_path: Path, csv_path: Path) -> None:
    comparable = [row for row in rows if row.get("both_valid_bounds")]
    gains = [float(row["dw_over_im_lp_pct"]) for row in comparable]
    payload = {
        "experiment": "E1: compact IM-LP versus paper-style root DW-CG lower bounds",
        "experiment_version": E1_EXPERIMENT_VERSION,
        "purpose": "Compare compact-LP and DW lower bounds under the reference CG stopping rule.",
        "process_assumption": PROCESS_ASSUMPTION,
        "model_version": MODEL_VERSION,
        "generation_settings": generation_metadata(),
        "scales": {
            name: {"T": spec.periods, "I": spec.items, "seeds": list(spec.seeds)}
            for name, spec in SCALES.items() if name in args.scales
        },
        "termination_rules": {
            "initial_columns": "Gurobi initial columns receive 60/120/180 seconds for small/medium/large.",
            "root_cg": "Stop at sum of pricing reduced costs > -0.1 or 500 iterations.",
            "dw_bound": "max over rounds of LRMP + sum(min(RC_m, 0)).",
            "comparison": "Gap=(LB_CG-LB_IM)/max(LB_CG,LB_IM)*100%; positive means that root DW-CG is tighter.",
            "im_lp_time_limit_sec": (
                args.im_lp_time_limit if args.im_lp_time_limit is not None
                else "none (pure LP solved to optimality)"
            ),
            "max_cg_iterations": args.max_cg_iterations,
        },
        "summary": {
            "instances": len(rows),
            "both_valid_bounds": len(comparable),
            "both_exact": sum(bool(row.get("both_exact")) for row in rows),
            "mean_dw_over_im_lp_pct": sum(gains) / len(gains) if gains else None,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({field for row in rows for field in row}) if rows else ["scale", "seed"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "result"
    result_dir.mkdir(exist_ok=True)
    json_path = result_dir / f"{args.output_prefix}.json"
    csv_path = result_dir / f"{args.output_prefix}.csv"
    old_rows: dict[tuple[str, int], dict] = {}
    if json_path.exists():
        try:
            old = json.loads(json_path.read_text(encoding="utf-8"))
            if (
                old.get("experiment_version") == E1_EXPERIMENT_VERSION
                and old.get("model_version") == MODEL_VERSION
                and old.get("process_assumption") == PROCESS_ASSUMPTION
            ):
                old_rows = {(row["scale"], int(row["seed"])): row for row in old.get("rows", [])}
        except (OSError, ValueError, KeyError, TypeError):
            pass

    rows: list[dict] = []
    seed_sets = {
        scale: tuple(seed for seed in SCALES[scale].seeds if args.seeds is None or seed in args.seeds)
        for scale in args.scales
    }
    unknown = set(args.seeds or ()) - {seed for seeds in seed_sets.values() for seed in seeds}
    if unknown:
        raise SystemExit(f"Seeds not present in the selected scale(s): {sorted(unknown)}")
    total = sum(len(seeds) for seeds in seed_sets.values())
    index = 0
    for scale in args.scales:
        for seed in seed_sets[scale]:
            index += 1
            key = (scale, seed)
            old = old_rows.get(key)
            if old is not None and (old.get("both_valid_bounds") or not args.rerun_incomplete):
                rows.append(old)
                print(f"[{index}/{total}] {scale} seed={seed}: reuse existing row", flush=True)
                continue
            print(f"[{index}/{total}] {scale} seed={seed}: IM-LP then paper-style root DW-CG", flush=True)
            row = run_one(scale, seed, args)
            rows.append(row)
            print(
                f"  IM-LP={row['im_lp_lb']} ({row['im_lp_status']}); "
                f"root-CG valid LB={row['root_cg_valid_lb']} ({row['root_cg_status']}); "
                f"valid-comparison={row['both_valid_bounds']}",
                flush=True,
            )
            write_results(rows, args, json_path, csv_path)

    write_results(rows, args, json_path, csv_path)
    print(f"Finished. Results: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
