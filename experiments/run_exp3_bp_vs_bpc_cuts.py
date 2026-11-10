"""Experiment 3: fixed-budget ablation of BP versus BPC (with cuts).

BP and BPC use the same instances, time budget, branching rule and exact
pricing.  The only intended algorithmic difference is ``lrmp_cuts``: ``none``
for BP and ``standard`` for BPC.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3: BP versus BPC cut ablation under equal budgets.")
    parser.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    parser.add_argument(
        "--output-prefix", default="exp3_bp_vs_bpc_cuts_certified",
        help="New default avoids reusing rows produced before CG-certification repair.",
    )
    parser.add_argument("--rerun-incomplete", action="store_true")
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
            subprocess.run(
                command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                # Setup/serialization buffer only; the solver budget is exactly ``limit``.
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


def number(value):
    return float(value) if value is not None else None


def relative_improvement(base: float | None, improved: float | None) -> float | None:
    """Return 100*(base-improved)/base; positive means the latter is better."""
    if base is None or improved is None or abs(base) <= 1e-9:
        return None
    return 100.0 * (base - improved) / abs(base)


def combine(scale: str, seed: int, bp: dict, bpc: dict) -> dict:
    spec = SCALES[scale]
    bp_lb = number(bp.get("lb")) if bp.get("lower_bound_certified") else None
    bpc_lb = number(bpc.get("lb")) if bpc.get("lower_bound_certified") else None
    bp_gap = number(bp.get("gap_pct")) if bp.get("lower_bound_certified") else None
    bpc_gap = number(bpc.get("gap_pct")) if bpc.get("lower_bound_certified") else None
    bp_time = number(bp.get("runtime_sec"))
    bpc_time = number(bpc.get("runtime_sec"))
    bp_optimal = bp.get("status") == "optimal"
    bpc_optimal = bpc.get("status") == "optimal"
    return {
        "scale": scale, "seed": seed, "T": spec.periods, "I": spec.items,
        "time_limit_sec": spec.time_limit_sec,
        "bp_status": bp.get("status"), "bp_ub": number(bp.get("ub")),
        "bp_lb": bp_lb, "bp_gap_pct": bp_gap, "bp_runtime_sec": bp_time,
        "bp_nodes": bp.get("nodes"), "bp_nodes_pruned": bp.get("nodes_pruned"),
        "bp_nodes_cg_failed": bp.get("nodes_cg_failed"),
        "bpc_status": bpc.get("status"), "bpc_ub": number(bpc.get("ub")),
        "bpc_lb": bpc_lb, "bpc_gap_pct": bpc_gap, "bpc_runtime_sec": bpc_time,
        "bpc_nodes": bpc.get("nodes"), "bpc_nodes_pruned": bpc.get("nodes_pruned"),
        "bpc_nodes_cg_failed": bpc.get("nodes_cg_failed"),
        # Positive values consistently mean BPC is better.
        "bpc_ub_improvement_vs_bp_pct": relative_improvement(number(bp.get("ub")), number(bpc.get("ub"))),
        "bpc_lb_improvement_vs_bp_pct": (
            100.0 * (bpc_lb - bp_lb) / max(abs(bp_lb), 1e-9)
            if bp_lb is not None and bpc_lb is not None else None
        ),
        "bpc_gap_reduction_vs_bp_pct": relative_improvement(bp_gap, bpc_gap),
        # A time saving is meaningful only if both methods have proven optimal.
        "bpc_time_saving_when_both_optimal_pct": (
            relative_improvement(bp_time, bpc_time) if bp_optimal and bpc_optimal else None
        ),
        "both_methods_found_feasible_solution": bp.get("ub") is not None and bpc.get("ub") is not None,
        "both_lower_bounds_certified": bp_lb is not None and bpc_lb is not None,
        "both_methods_proved_optimal": bp_optimal and bpc_optimal,
        "bp_error": bp.get("error"), "bpc_error": bpc.get("error"),
        "raw_bp": bp, "raw_bpc": bpc,
    }


def mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def write_results(rows: list[dict], args: argparse.Namespace, json_path: Path, csv_path: Path) -> None:
    payload = {
        "experiment": "E3: BP versus BPC cut ablation under equal budgets",
        "purpose": "Measure the contribution of LRMP cuts to bound quality, gap, search and time.",
        "process_assumption": PROCESS_ASSUMPTION,
        "model_version": MODEL_VERSION,
        "controlled_difference": "BP uses lrmp_cuts=none; BPC uses lrmp_cuts=standard. All other worker settings are identical.",
        "scales": {
            scale: {"T": SCALES[scale].periods, "I": SCALES[scale].items,
                    "seeds": list(SCALES[scale].seeds), "time_limit_sec": SCALES[scale].time_limit_sec}
            for scale in args.scales
        },
        "metric_definition": {
            "positive_improvement": "Positive UB/LB improvement means BPC is better.",
            "bpc_gap_reduction_vs_bp_pct": "100*(BP_gap-BPC_gap)/BP_gap; reported only for certified gaps.",
            "bpc_time_saving_when_both_optimal_pct": "100*(BP_time-BPC_time)/BP_time; reported only when both prove optimal.",
        },
        "summary": {
            "instances": len(rows),
            "mean_bpc_ub_improvement_vs_bp_pct": mean([r["bpc_ub_improvement_vs_bp_pct"] for r in rows]),
            "mean_bpc_lb_improvement_vs_bp_pct": mean([r["bpc_lb_improvement_vs_bp_pct"] for r in rows]),
            "mean_bpc_gap_reduction_vs_bp_pct": mean([r["bpc_gap_reduction_vs_bp_pct"] for r in rows]),
            "mean_bpc_time_saving_when_both_optimal_pct": mean([r["bpc_time_saving_when_both_optimal_pct"] for r in rows]),
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "scale", "seed", "T", "I", "time_limit_sec",
        "bp_status", "bp_ub", "bp_lb", "bp_gap_pct", "bp_runtime_sec", "bp_nodes", "bp_nodes_pruned", "bp_nodes_cg_failed",
        "bpc_status", "bpc_ub", "bpc_lb", "bpc_gap_pct", "bpc_runtime_sec", "bpc_nodes", "bpc_nodes_pruned", "bpc_nodes_cg_failed",
        "bpc_ub_improvement_vs_bp_pct", "bpc_lb_improvement_vs_bp_pct", "bpc_gap_reduction_vs_bp_pct",
        "bpc_time_saving_when_both_optimal_pct", "both_methods_found_feasible_solution",
        "both_lower_bounds_certified", "both_methods_proved_optimal", "bp_error", "bpc_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "result"; result_dir.mkdir(exist_ok=True)
    json_path = result_dir / f"{args.output_prefix}.json"
    csv_path = result_dir / f"{args.output_prefix}.csv"
    tmp_dir = result_dir / f"_{args.output_prefix}_tmp"; tmp_dir.mkdir(exist_ok=True)
    old_rows: dict[tuple[str, int], dict] = {}
    if json_path.exists():
        try:
            old = json.loads(json_path.read_text(encoding="utf-8"))
            if old.get("model_version") == MODEL_VERSION and old.get("process_assumption") == PROCESS_ASSUMPTION:
                old_rows = {(r["scale"], int(r["seed"])): r for r in old.get("rows", [])}
        except (OSError, ValueError, KeyError, TypeError):
            pass

    rows: list[dict] = []
    total = sum(len(SCALES[scale].seeds) for scale in args.scales); index = 0
    for scale in args.scales:
        spec = SCALES[scale]
        for seed in spec.seeds:
            index += 1; key = (scale, seed); old = old_rows.get(key)
            if old is not None and (old.get("both_methods_found_feasible_solution") or not args.rerun_incomplete):
                rows.append(old); print(f"[{index}/{total}] {scale} seed={seed}: reuse existing row", flush=True); continue
            print(f"[{index}/{total}] {scale} seed={seed}: BP and BPC, budget={spec.time_limit_sec:.0f}s each", flush=True)
            row = combine(scale, seed, run_worker("bp", scale, seed, spec.time_limit_sec, tmp_dir), run_worker("bpc", scale, seed, spec.time_limit_sec, tmp_dir))
            rows.append(row)
            print(f"  BP gap={row['bp_gap_pct']}; BPC gap={row['bpc_gap_pct']}; BPC UB improvement={row['bpc_ub_improvement_vs_bp_pct']}", flush=True)
            write_results(rows, args, json_path, csv_path)
    write_results(rows, args, json_path, csv_path)
    print(f"Finished. Results: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
