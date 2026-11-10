"""Experiment 2: fixed-budget upper-bound comparison of IM (Gurobi) and BPC.

The primary outcome is the best feasible integer solution (UB) held by each
method when the *same scale-specific budget* expires.  A positive reported
improvement means that BPC has a lower cost than Gurobi's compact IM.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.paper_experiment_config import MODEL_VERSION, PROCESS_ASSUMPTION, SCALES


# Bump this whenever E2's reported fields or solver protocol changes, so that
# incomplete older result files cannot silently be reused.
E2_EXPERIMENT_VERSION = "im_vs_full_bpc_ub_lb_nodes_practical_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2: compare best IM and BPC feasible solutions at identical time limits."
    )
    parser.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    parser.add_argument("--output-prefix", default="exp2_im_vs_full_bpc_ub_nodes")
    parser.add_argument(
        "--rerun-incomplete", action="store_true",
        help="Retry rows where either method did not return a usable result.",
    )
    return parser.parse_args()


def run_worker(method: str, scale: str, seed: int, limit: float, tmp_dir: Path) -> dict:
    result_path = tmp_dir / f"{method}_{scale}_{seed}.json"
    log_path = tmp_dir / f"{method}_{scale}_{seed}.log"
    command = [
        sys.executable, "-B", "experiments/paper_experiment_worker.py",
        method, scale, str(seed), str(result_path), "--limit", str(limit),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            # The extra buffer is solely for instance/model construction and
            # result serialization.  Both solvers themselves receive exactly
            # ``limit`` seconds through the worker.
            subprocess.run(
                command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                timeout=limit + 180.0, check=False,
            )
        except subprocess.TimeoutExpired:
            pass
    if result_path.exists():
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"status": "worker_error", "error": f"cannot read worker result: {exc}"}
    return {
        "method": method, "scale": scale, "seed": seed,
        "status": "wall_time_limit",
        "error": "worker did not return after solver budget plus setup buffer",
    }


def as_number(value):
    return float(value) if value is not None else None


def as_count(value):
    return int(value) if value is not None else None


def combine(scale: str, seed: int, im: dict, bpc: dict) -> dict:
    """Make one publication-table row from the two isolated solver runs."""
    spec = SCALES[scale]
    im_ub = as_number(im.get("ub"))
    bpc_ub = as_number(bpc.get("ub"))
    # Positive means BPC's feasible solution is cheaper and is therefore better.
    improvement = (
        100.0 * (im_ub - bpc_ub) / abs(im_ub)
        if im_ub is not None and bpc_ub is not None and abs(im_ub) > 1e-9
        else None
    )
    # Positive means that BPC's incumbent is worse (larger) than IM's.
    ub_gap = (
        100.0 * (bpc_ub - im_ub) / abs(im_ub)
        if im_ub is not None and bpc_ub is not None and abs(im_ub) > 1e-9
        else None
    )
    return {
        "scale": scale,
        "seed": seed,
        "T": spec.periods,
        "I": spec.items,
        "time_limit_sec": spec.time_limit_sec,
        "gurobi_status": im.get("status"),
        "gurobi_ub": im_ub,
        # Keep the incumbent and certified lower bound separately.  The LB is
        # still reported when Gurobi stops at the time limit, provided it has
        # a valid bound, so E2 can be reused for the BP--BPC ablation.
        "gurobi_lb": as_number(im.get("lb")),
        "gurobi_nodes": as_count(im.get("gurobi_nodes")),
        "gurobi_runtime_sec": as_number(im.get("runtime_sec")),
        "gurobi_solver_gap_pct": as_number(im.get("gap_pct")),
        "bpc_status": bpc.get("status"),
        "bpc_ub": bpc_ub,
        "bpc_lb": as_number(bpc.get("lb")),
        "bpc_runtime_sec": as_number(bpc.get("runtime_sec")),
        # In the practical experiment this is the raw tree gap used by the
        # stopping rule.  It may be uncertified when CG-failed nodes remain.
        "bpc_raw_gap_pct": as_number(bpc.get("raw_gap_pct", bpc.get("gap_pct"))),
        "bpc_certified_gap_pct": as_number(bpc.get("certified_gap_pct")),
        "bpc_solver_gap_pct": as_number(bpc.get("raw_gap_pct", bpc.get("gap_pct"))),
        "bpc_lb_certified": bool(bpc.get("lower_bound_certified", False)),
        # ``nodes`` is the number of BPC nodes actually processed, including
        # the root.  It is the transparent branch-and-price tree-size metric
        # requested for E2; it is not a Gurobi MIP-node count.
        "bpc_nodes_explored": as_count(bpc.get("nodes")),
        "bpc_nodes_created": as_count(bpc.get("nodes_created")),
        "bpc_nodes_pruned": as_count(bpc.get("nodes_pruned")),
        "bpc_nodes_cg_failed": as_count(bpc.get("nodes_cg_failed")),
        "bpc_termination_rule": bpc.get("termination_rule"),
        "bpc_ub_improvement_vs_gurobi_pct": improvement,
        "bpc_ub_gap_vs_gurobi_pct": ub_gap,
        "both_methods_found_feasible_solution": im_ub is not None and bpc_ub is not None,
        "both_methods_proved_optimal": im.get("status") == "optimal" and bpc.get("status") == "optimal",
        "gurobi_error": im.get("error"),
        "bpc_error": bpc.get("error"),
        # Retain the raw lower-bound information in JSON for auditability,
        # while keeping UB quality as the primary publication comparison.
        "raw_im": im,
        "raw_bpc": bpc,
    }


def write_results(rows: list[dict], args: argparse.Namespace, json_path: Path, csv_path: Path) -> None:
    valid = [row for row in rows if row["both_methods_found_feasible_solution"]]
    improvements = [row["bpc_ub_improvement_vs_gurobi_pct"] for row in valid]
    payload = {
        "experiment": "E2: fixed-budget IM (Gurobi) versus BPC upper-bound comparison",
        "experiment_version": E2_EXPERIMENT_VERSION,
        "purpose": "Compare the best feasible integer objective obtained under equal time budgets.",
        "process_assumption": PROCESS_ASSUMPTION,
        "model_version": MODEL_VERSION,
        "scales": {
            scale: {"T": SCALES[scale].periods, "I": SCALES[scale].items,
                    "seeds": list(SCALES[scale].seeds), "time_limit_sec": SCALES[scale].time_limit_sec}
            for scale in args.scales
        },
        "metric_definition": {
            "gurobi_ub": "Best feasible integer objective from compact IM.",
            "gurobi_lb": "Gurobi's best bound (ObjBound), including time-limit runs when available.",
            "gurobi_nodes": "Gurobi MIP node count reported by the compact IM solve.",
            "bpc_ub": "Best feasible integer objective from BPC.",
            "bpc_lb": "BPC's global lower bound from the branch-and-price tree when available.",
            "bpc_lb_certified": "True only when every processed BPC node has a valid certified lower bound.",
            "bpc_raw_gap_pct": "Raw (UB-tree LB)/UB used by the practical stopping rule; it may be uncertified.",
            "bpc_certified_gap_pct": "Same gap, reported only when all processed BPC nodes are CG-certified.",
            "bpc_ub_gap_vs_gurobi_pct": "100*(BPC_UB-Gurobi_UB)/Gurobi_UB; positive means BPC is worse.",
            "bpc_ub_improvement_vs_gurobi_pct": "100*(Gurobi_UB-BPC_UB)/Gurobi_UB; positive means BPC is better.",
            "bpc_nodes_explored": "Number of BPC nodes processed, including the root node.",
            "bpc_nodes_created": "Total BPC nodes created, including the root node.",
            "bpc_nodes_pruned": "BPC nodes pruned by bound, infeasibility, or incumbent dominance.",
        "solver_gap_pct": "Gurobi's certified gap; BPC's corresponding field is the practical raw tree gap.",
        },
        "summary": {
            "instances": len(rows),
            "both_methods_found_feasible_solution": len(valid),
            "mean_bpc_ub_improvement_vs_gurobi_pct": sum(improvements) / len(improvements) if improvements else None,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "scale", "seed", "T", "I", "time_limit_sec",
        "gurobi_ub", "gurobi_lb", "gurobi_nodes", "gurobi_runtime_sec", "gurobi_solver_gap_pct",
        # Publication table: statuses remain in the auditable JSON/raw rows,
        # but are intentionally omitted from the compact CSV table.
        "bpc_ub", "bpc_lb", "bpc_lb_certified", "bpc_runtime_sec", "bpc_raw_gap_pct", "bpc_certified_gap_pct",
        "bpc_nodes_explored", "bpc_nodes_created",
        "bpc_nodes_pruned", "bpc_nodes_cg_failed",
        "bpc_ub_improvement_vs_gurobi_pct", "bpc_ub_gap_vs_gurobi_pct",
        "both_methods_found_feasible_solution",
        "gurobi_error", "bpc_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "result" / "exp2"
    result_dir.mkdir(exist_ok=True)
    json_path = result_dir / f"{args.output_prefix}.json"
    csv_path = result_dir / f"{args.output_prefix}.csv"
    tmp_dir = result_dir / f"_{args.output_prefix}_tmp"
    tmp_dir.mkdir(exist_ok=True)

    old_rows: dict[tuple[str, int], dict] = {}
    if json_path.exists():
        try:
            old = json.loads(json_path.read_text(encoding="utf-8"))
            if (old.get("experiment_version") == E2_EXPERIMENT_VERSION
                    and old.get("model_version") == MODEL_VERSION
                    and old.get("process_assumption") == PROCESS_ASSUMPTION):
                old_rows = {(row["scale"], int(row["seed"])): row for row in old.get("rows", [])}
        except (OSError, ValueError, KeyError, TypeError):
            pass

    rows: list[dict] = []
    total = sum(len(SCALES[scale].seeds) for scale in args.scales)
    index = 0
    for scale in args.scales:
        spec = SCALES[scale]
        for seed in spec.seeds:
            index += 1
            key = (scale, seed)
            old = old_rows.get(key)
            if old is not None and (old.get("both_methods_found_feasible_solution") or not args.rerun_incomplete):
                rows.append(old)
                print(f"[{index}/{total}] {scale} seed={seed}: reuse existing row", flush=True)
                continue
            print(f"[{index}/{total}] {scale} seed={seed}: IM and BPC, budget={spec.time_limit_sec:.0f}s each", flush=True)
            im = run_worker("im", scale, seed, spec.time_limit_sec, tmp_dir)
            bpc = run_worker("bpc", scale, seed, spec.time_limit_sec, tmp_dir)
            row = combine(scale, seed, im, bpc)
            rows.append(row)
            print(
                f"  Gurobi UB={row['gurobi_ub']}; BPC UB={row['bpc_ub']}; "
                f"BPC improvement={row['bpc_ub_improvement_vs_gurobi_pct']}", flush=True,
            )
            write_results(rows, args, json_path, csv_path)

    write_results(rows, args, json_path, csv_path)
    print(f"Finished. Results: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
