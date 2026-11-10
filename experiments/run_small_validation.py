"""Small-scale exact-validation experiment: Gurobi IM versus BPC.

For each seed, the script generates exactly the same T=30, I=40 instance
for both methods.  It then solves the compact formulation (IM) and the
branch-and-price formulation (BPC) with a zero target relative gap.

For the paper-aligned run, BPC uses EBRP for every pricing call.  Gurobi is
used only by the compact IM benchmark, never as the BPC pricing oracle.

The output files are written incrementally, so completed instances are kept
if a later run is interrupted:
    result/small_validation_common_unload_v2.json
    result/small_validation_common_unload_v2.csv
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
from configuration import BENCH_SMALL, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import preprocess


DEFAULT_SEEDS = list(range(10101, 10116))
T, I = BENCH_SMALL
MODEL_VERSION = "common_unload_v2_span_link"


def _gap_pct(ub: float | None, lb: float | None) -> float | None:
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return round(100.0 * max(0.0, (ub - lb) / ub), 4)


def _proved(status: str, ub: float | None, lb: float | None) -> bool:
    return status == "optimal" or (
        ub is not None and lb is not None and abs(ub - lb) <= 1e-4
    )


def _cleanup(*objects) -> None:
    """Release Gurobi/native objects before starting the next instance."""
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
    gc.collect()


def _solve_im(seed: int, time_limit: float) -> dict:
    raw = data = im = None
    try:
        raw = generate_instance(SetConfig("small_cg", T, 5, 0, I, 5), seed % 1000, seed)
        data = preprocess(raw)
        t0 = time.perf_counter()
        im = solve_im(data, time_limit=time_limit, mip_gap=0.0, verbose=False)
        elapsed = time.perf_counter() - t0
        ub, lb = im.objective, im.lower_bound
        return {
            "status": im.status,
            "objective": ub,
            "lower_bound": lb,
            "gap_pct": _gap_pct(ub, lb),
            "runtime_sec": round(elapsed, 2),
            "proved": _proved(im.status, ub, lb),
        }
    finally:
        _cleanup(im, data, raw)


def _solve_bpc(seed: int, time_limit: float, max_nodes: int, pricing_mode: str) -> dict:
    raw = data = bp = result = None
    try:
        raw = generate_instance(SetConfig("small_cg", T, 5, 0, I, 5), seed % 1000, seed)
        data = preprocess(raw)
        bp = BranchAndPriceSolver(
            data,
            max_nodes=max_nodes,
            time_limit=time_limit,
            mip_gap=0.0,
            branch_strategy="frac_sb",
            use_node_im=False,
            root_im=False,
            final_im=False,
            strict_cg_certification=True,
            cg_kwargs={
                # This validation entry point is intentionally EBRP-only.
                # Do not allow Gurobi to generate initial or fallback columns.
                "pricing_mode": "ebrp",
                "ebrp_force": True,
                "use_gurobi_init": False,
                "skip_cross_check": True,
                "pricing_inprocess": True,
                "verbose": False,
                "progress_interval_sec": 120.0,
                "lrmp_cuts": "standard",
                "max_iterations": 120,
                # Small-scale exact validation: do not truncate EBRP pricing.
                "ebrp_time_limit": None,
            },
        )
        t0 = time.perf_counter()
        result = bp.solve()
        elapsed = time.perf_counter() - t0
        ub, lb = result.objective, result.lower_bound
        return {
            "status": result.status,
            "objective": ub,
            "lower_bound": lb,
            "gap_pct": _gap_pct(ub, lb),
            "runtime_sec": round(elapsed, 2),
            "nodes_explored": result.nodes_explored,
            "nodes_pruned": result.nodes_pruned,
            "nodes_cg_failed": result.nodes_cg_failed,
            "root_cg_converged": result.root_cg_converged,
            "root_cg_stop": result.root_cg_stop,
            "proved": _proved(result.status, ub, lb) and result.nodes_cg_failed == 0,
        }
    finally:
        _cleanup(result, bp, data, raw)


def _write_results(rows: list[dict], json_path: Path, csv_path: Path, args: argparse.Namespace) -> None:
    summary = {
        "instances": len(rows),
        "both_proved": sum(row["both_proved"] for row in rows),
        "same_objective": sum(row["same_objective"] for row in rows),
        "validation_passed": sum(row["validation_passed"] for row in rows),
    }
    payload = {
        "experiment": "Small-scale IM (Gurobi) versus BPC validation",
        "process_assumption": "All items in a batch leave the autoclave and enter inventory together at the batch maximum curing duration.",
        "model_version": MODEL_VERSION,
        "scale": {"T": T, "I": I, "U": 5, "M": 5},
        "seeds": args.seeds,
        "time_limit_sec_per_method": args.time_limit,
        "max_bpc_nodes": args.max_nodes,
        "bpc_branch_strategy": "frac_sb",
        "pricing_mode": "ebrp",
        "pricing_note": "BPC uses EBRP for all pricing calls; Gurobi is used only for the compact IM benchmark.",
        "criterion": "Both methods prove optimality, BPC has zero CG failures, and their objectives are equal within 1e-4.",
        "summary": summary,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "seed", "T", "I", "im_status", "im_objective", "im_lower_bound",
        "im_gap_pct", "im_runtime_sec", "im_proved", "bpc_status", "bpc_objective",
        "bpc_lower_bound", "bpc_gap_pct", "bpc_runtime_sec", "bpc_nodes_explored",
        "bpc_nodes_pruned", "bpc_nodes_cg_failed", "bpc_root_cg_converged",
        "bpc_root_cg_stop", "bpc_proved", "same_objective", "both_proved",
        "validation_passed", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BPC against the Gurobi compact model on small instances.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Instance seeds; default: 10101 to 10115.")
    parser.add_argument("--time-limit", type=float, default=900.0, help="Seconds allowed for each method on each instance.")
    parser.add_argument("--max-nodes", type=int, default=8000, help="Maximum BPC nodes per instance.")
    parser.add_argument(
        "--pricing-mode", choices=("ebrp",), default="ebrp",
        help="BPC pricing oracle. This small-scale validation entry point always uses EBRP.",
    )
    parser.add_argument("--output-prefix", default="small_validation_common_unload_v2", help="Filename prefix under result/.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg.apply_bench_generation()
    result_dir = ROOT / "result"
    result_dir.mkdir(exist_ok=True)
    json_path = result_dir / f"{args.output_prefix}.json"
    csv_path = result_dir / f"{args.output_prefix}.csv"
    # Reuse completed rows from an interrupted run with the same output prefix.
    # A row with an error is deliberately re-run; a complete IM+BPC pair is not.
    old_rows: dict[int, dict] = {}
    if json_path.exists():
        try:
            old_payload = json.loads(json_path.read_text(encoding="utf-8"))
            if old_payload.get("process_assumption") != "All items in a batch leave the autoclave and enter inventory together at the batch maximum curing duration.":
                raise ValueError("result file uses a different process assumption")
            if old_payload.get("model_version") != MODEL_VERSION:
                raise ValueError("result file was created before the common-unload span-link correction")
            for old_row in old_payload.get("rows", []):
                if (
                    old_row.get("error") is None
                    and old_row.get("im_status") is not None
                    and old_row.get("bpc_status") is not None
                ):
                    old_rows[int(old_row["seed"])] = old_row
        except (OSError, ValueError, KeyError, TypeError):
            # A malformed old file should not prevent a new experiment run.
            old_rows = {}
    rows: list[dict] = [old_rows[seed] for seed in args.seeds if seed in old_rows]

    print(f"Small validation: T={T}, I={I}, U=M=5; seeds={args.seeds}", flush=True)
    print(f"Per-method time limit: {args.time_limit:.0f}s; BPC max nodes: {args.max_nodes}", flush=True)
    for index, seed in enumerate(args.seeds, start=1):
        if seed in old_rows:
            print(f"\n[{index}/{len(args.seeds)}] seed={seed}: reuse completed result from {json_path.name}", flush=True)
            continue
        row = {"seed": seed, "T": T, "I": I, "error": None}
        print(f"\n[{index}/{len(args.seeds)}] seed={seed}: IM (Gurobi)", flush=True)
        try:
            im = _solve_im(seed, args.time_limit)
            row.update({f"im_{key}": value for key, value in im.items()})
            print(f"  IM: status={im['status']}, obj={im['objective']}, gap={im['gap_pct']}%, time={im['runtime_sec']}s", flush=True)

            print(f"[{index}/{len(args.seeds)}] seed={seed}: BPC (frac_sb, pricing={args.pricing_mode})", flush=True)
            bpc = _solve_bpc(seed, args.time_limit, args.max_nodes, args.pricing_mode)
            row.update({f"bpc_{key}": value for key, value in bpc.items()})
            print(f"  BPC: status={bpc['status']}, obj={bpc['objective']}, gap={bpc['gap_pct']}%, nodes={bpc['nodes_explored']}, cg_fail={bpc['nodes_cg_failed']}, time={bpc['runtime_sec']}s", flush=True)

            row["same_objective"] = (
                im["objective"] is not None
                and bpc["objective"] is not None
                and abs(im["objective"] - bpc["objective"]) <= 1e-4
            )
            row["both_proved"] = im["proved"] and bpc["proved"]
            row["validation_passed"] = row["same_objective"] and row["both_proved"]
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["same_objective"] = False
            row["both_proved"] = False
            row["validation_passed"] = False
            print(f"  FAILED: {row['error']}", flush=True)
        rows.append(row)
        _write_results(rows, json_path, csv_path, args)

    passed = sum(row["validation_passed"] for row in rows)
    print(f"\nFinished: {passed}/{len(rows)} instances passed exact validation.", flush=True)
    print(f"JSON: {json_path}\nCSV:  {csv_path}", flush=True)


if __name__ == "__main__":
    main()
