"""Four-way BP/BPC cut ablation under controlled, equal-budget settings.

Variants:
  * bp_no_cut:        BP without LRMP cuts;
  * bpc_deadline:     deadline-cover cuts only;
  * bpc_sri_only:     root subset-row inequalities only;
  * bpc_deadline_sri: deadline-cover cuts plus root SRIs.

Example
-------
python -B experiments/run_exp3_fourway_cut_ablation.py \
    --scales medium --seeds 30101 31301 31601
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import configuration as cfg
from experiments.paper_experiment_config import MODEL_VERSION, PROCESS_ASSUMPTION, SCALES, make_set_config
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess


VARIANTS: dict[str, dict | str] = {
    "bp_no_cut": "none",
    "bpc_deadline": "standard",
    "bpc_sri_only": {
        "deadline_cover": False,
        "sri": True,
        "sri_max_cuts": 3,
        "sri_violation_tol": 1e-6,
    },
    "bpc_deadline_sri": "sri",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled BP / deadline-cover / SRI four-way ablation."
    )
    parser.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=["small"])
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=None,
        help="Optional method subset; default runs all four variants.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Optional seed subset. Every supplied seed must belong to a selected scale.",
    )
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="Override the per-instance time limit in seconds.",
    )
    parser.add_argument(
        "--pricing-time-limit", type=float, default=None,
        help="Per-machine EBRP cap; omit for exact uncapped pricing.",
    )
    parser.add_argument("--max-nodes", type=int, default=8000)
    parser.add_argument(
        "--strict-cg-node-limit", type=float, default=120.0,
        help="Per-attempt CG cap (s) in strict-certification mode.",
    )
    parser.add_argument("--output-prefix", default="exp3_fourway_cut_ablation")
    parser.add_argument("--rerun", action="store_true", help="Recompute existing result files.")
    return parser.parse_args()


def relative_gap(ub: float | None, lb: float | None) -> float | None:
    if ub is None or lb is None or abs(ub) <= 1e-9:
        return None
    return 100.0 * (ub - lb) / abs(ub)


def output_paths(prefix: str) -> tuple[Path, Path]:
    # Keep experiment outputs writable when the source checkout is read-only.
    out_dir = Path(os.environ.get("BPC_RESULT_DIR", ROOT / "result" / "exp3_fourway"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{prefix}.json", out_dir / f"{prefix}.csv"


def validate_seed_selection(scales: list[str], requested: list[int] | None) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    permitted = {seed for scale in scales for seed in SCALES[scale].seeds}
    if requested is not None and any(seed not in permitted for seed in requested):
        raise ValueError("Every supplied seed must belong to one of the selected scales.")
    for scale in scales:
        result[scale] = [seed for seed in SCALES[scale].seeds if requested is None or seed in requested]
    return result


def run_one(scale: str, seed: int, variant: str, cuts: dict | str, args: argparse.Namespace) -> dict:
    spec = SCALES[scale]
    raw = generate_instance(make_set_config(spec), seed % 1000, seed)
    data = preprocess(raw)
    limit = float(args.time_limit if args.time_limit is not None else spec.time_limit_sec)
    started = time.perf_counter()
    solver = BranchAndPriceSolver(
        data,
        max_nodes=args.max_nodes,
        time_limit=limit,
        mip_gap=0.0 if scale == "small" else 0.01,
        branch_strategy="frac_sb",
        use_node_im=False,
        root_im=False,
        # New SRI variants receive an independent compact-IM audit before an
        # apparent B&P optimum is recorded.  This prevents an unsound tree
        # certificate from entering the ablation table as an optimal result.
        final_im=variant in {"bpc_sri_only", "bpc_deadline_sri"},
        strict_cg_certification=True,
        strict_cg_node_time_limit=args.strict_cg_node_limit,
        cg_kwargs={
            "pricing_mode": "ebrp",
            "ebrp_force": True,
            "use_gurobi_init": True,
            "pricing_inprocess": True,
            "skip_cross_check": True,
            "lrmp_cuts": cuts,
            "max_iterations": 120,
            "ebrp_time_limit": args.pricing_time_limit,
            "verbose": False,
        },
    )
    result = solver.solve()
    certified = bool(result.nodes_cg_failed == 0 and result.lower_bound is not None)
    return {
        "scale": scale,
        "seed": seed,
        "T": spec.periods,
        "I": spec.items,
        "variant": variant,
        "cuts": cuts,
        "time_limit_sec": limit,
        "pricing_time_limit_sec": args.pricing_time_limit,
        "strict_cg_node_limit_sec": args.strict_cg_node_limit,
        "status": result.status,
        "ub": result.objective,
        "lb": result.lower_bound,
        "gap_pct": relative_gap(result.objective, result.lower_bound),
        "lower_bound_certified": certified,
        # Main-table time excludes the optional independent IM audit.
        # ``total_runtime_sec`` preserves the end-to-end replication cost.
        "runtime_sec": round(result.runtime_sec, 3),
        "audit_runtime_sec": round(result.audit_runtime_sec, 3),
        "total_runtime_sec": round(time.perf_counter() - started, 3),
        "nodes": result.nodes_explored,
        "nodes_created": result.nodes_created,
        "nodes_pruned": result.nodes_pruned,
        "nodes_cg_failed": result.nodes_cg_failed,
        "root_lb": result.root_valid_lb,
        "root_cg_converged": result.root_cg_converged,
        "root_cg_stop": result.root_cg_stop,
    }


def mean(values: list[float | None]) -> float | None:
    observed = [float(value) for value in values if value is not None]
    return sum(observed) / len(observed) if observed else None


def summarise(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for scale in sorted({row["scale"] for row in rows}):
        summary[scale] = {}
        for variant in VARIANTS:
            group = [row for row in rows if row["scale"] == scale and row["variant"] == variant]
            certified = [row for row in group if row["lower_bound_certified"]]
            summary[scale][variant] = {
                "n": len(group),
                "n_certified": len(certified),
                "n_gap_at_or_below_1pct": sum(
                    row["gap_pct"] is not None and row["gap_pct"] <= 1.0 for row in certified
                ),
                "mean_ub": mean([row["ub"] for row in group]),
                "mean_certified_lb": mean([row["lb"] for row in certified]),
                "mean_certified_gap_pct": mean([row["gap_pct"] for row in certified]),
                "mean_runtime_sec": mean([row["runtime_sec"] for row in group]),
                "mean_nodes": mean([row["nodes"] for row in group]),
                "mean_root_lb": mean([row["root_lb"] for row in group]),
            }
    return summary


def write_outputs(rows: list[dict], args: argparse.Namespace) -> tuple[Path, Path]:
    json_path, csv_path = output_paths(args.output_prefix)
    report = {
        "experiment": "controlled four-way BP/BPC cut ablation",
        "model_version": MODEL_VERSION,
        "process_assumption": PROCESS_ASSUMPTION,
        "variants": VARIANTS,
        "settings": {
            "strict_cg_certification": True,
            "pricing_mode": "ebrp",
            "pricing_time_limit_sec": args.pricing_time_limit,
            "strict_cg_node_limit_sec": args.strict_cg_node_limit,
            "branch_strategy": "frac_sb",
        },
        "summary": summarise(rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # The CSV is the paper-facing result table.  Keep validation and
    # end-to-end replication timings in JSON, but do not mix them into the
    # reported algorithm-time columns.
    hidden_csv_fields = {"audit_runtime_sec", "total_runtime_sec"}
    fields = (
        [field for field in rows[0] if field not in hidden_csv_fields]
        if rows else ["scale", "seed", "variant"]
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            # Audit and total timings stay in the JSON record only.
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    cfg.apply_bench_generation()
    seed_map = validate_seed_selection(args.scales, args.seeds)
    json_path, _ = output_paths(args.output_prefix)
    rows: list[dict] = []
    # Always retain prior rows.  ``--rerun`` means recompute the selected
    # jobs below, not discard every unselected result from the report.
    if json_path.exists():
        prior = json.loads(json_path.read_text(encoding="utf-8"))
        rows = list(prior.get("rows", []))

    existing = {(row["scale"], row["seed"], row["variant"]) for row in rows}
    selected_variants = args.variants or list(VARIANTS)
    jobs = [(scale, seed, variant, VARIANTS[variant]) for scale, seeds in seed_map.items()
            for seed in seeds for variant in selected_variants]
    for index, (scale, seed, variant, cuts) in enumerate(jobs, 1):
        key = (scale, seed, variant)
        if key in existing and not args.rerun:
            print(f"[{index}/{len(jobs)}] reuse {scale} seed={seed} {variant}", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] {scale} seed={seed} {variant}", flush=True)
        rows = [row for row in rows if (row["scale"], row["seed"], row["variant"]) != key]
        rows.append(run_one(scale, seed, variant, cuts, args))
        json_out, csv_out = write_outputs(rows, args)
        print(f"  saved {json_out.name}, {csv_out.name}", flush=True)

    json_out, csv_out = write_outputs(rows, args)
    print(f"Completed {len(rows)} rows: {json_out}\n{csv_out}", flush=True)


if __name__ == "__main__":
    main()
