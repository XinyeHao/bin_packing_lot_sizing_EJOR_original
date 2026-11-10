"""Experiment 4: controlled comparison of legacy BP and legacy BPC.

Usage examples (run from the project root):

    python experiments/run_exp4_legacy_bp_vs_bpc.py --scales small
    python experiments/run_exp4_legacy_bp_vs_bpc.py --scales medium large
    python experiments/run_exp4_legacy_bp_vs_bpc.py --scales medium --seeds 30101 30201

The program never overwrites a completed pair unless ``--rerun`` is supplied.
All output is stored in ``result/exp4/legacy_bp_vs_bpc_deadline_e``.
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

from experiments.paper_experiment_config import PROCESS_ASSUMPTION, SCALES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp4: legacy BP versus legacy BPC; only deadline-cover cuts differ."
    )
    parser.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        help="Optional subset of seeds; every supplied seed must belong to a selected scale.",
    )
    parser.add_argument("--ebrp-time-limit", type=float, default=20.0)
    parser.add_argument("--rerun", action="store_true", help="Recompute completed BP/BPC results.")
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_one(method: str, scale: str, seed: int, limit: float, ebrp_limit: float, out_dir: Path, rerun: bool) -> dict:
    output = out_dir / f"{method}_{scale}_{seed}.json"
    log = out_dir / f"{method}_{scale}_{seed}.log"
    old = read_json(output)
    if old is not None and not rerun and old.get("status") != "error":
        return old

    command = [
        sys.executable, "-u", "experiments/exp4_legacy_cut_worker.py",
        method, scale, str(seed), str(output),
        "--limit", str(limit), "--ebrp-time-limit", str(ebrp_limit),
    ]
    with log.open("w", encoding="utf-8") as handle:
        try:
            subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                timeout=limit + 180.0, check=False,
            )
        except subprocess.TimeoutExpired:
            output.write_text(
                json.dumps({
                    "method": method.upper(), "scale": scale, "seed": seed,
                    "status": "worker_timeout",
                    "error": "worker exceeded solver budget plus 180-second cleanup allowance",
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return read_json(output) or {
        "method": method.upper(), "scale": scale, "seed": seed,
        "status": "worker_error", "error": "worker produced no readable JSON result",
    }


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    return float(value) if value is not None else None


def pct_change(base: float | None, changed: float | None, direction: str) -> float | None:
    if base is None or changed is None or abs(base) <= 1e-9:
        return None
    # Positive means BPC is preferable in all reported improvement columns.
    return 100.0 * ((base - changed) if direction == "smaller" else (changed - base)) / abs(base)


def combine(scale: str, seed: int, bp: dict, bpc: dict) -> dict:
    bp_cert = bool(bp.get("lower_bound_certified"))
    bpc_cert = bool(bpc.get("lower_bound_certified"))
    bp_lb = number(bp, "lb")
    bpc_lb = number(bpc, "lb")
    return {
        "scale": scale,
        "seed": seed,
        "T": SCALES[scale].periods,
        "I": SCALES[scale].items,
        "time_limit_sec": SCALES[scale].time_limit_sec,
        "bp_status": bp.get("status"),
        "bp_ub": number(bp, "ub"),
        "bp_lb": bp_lb,
        "bp_gap_pct": number(bp, "raw_gap_pct"),
        "bp_runtime_sec": number(bp, "runtime_sec"),
        "bp_nodes": bp.get("nodes"),
        "bp_nodes_pruned": bp.get("nodes_pruned"),
        "bp_nodes_cg_failed": bp.get("nodes_cg_failed"),
        "bp_lb_certified": bp_cert,
        "bpc_status": bpc.get("status"),
        "bpc_ub": number(bpc, "ub"),
        "bpc_lb": bpc_lb,
        "bpc_gap_pct": number(bpc, "raw_gap_pct"),
        "bpc_runtime_sec": number(bpc, "runtime_sec"),
        "bpc_nodes": bpc.get("nodes"),
        "bpc_nodes_pruned": bpc.get("nodes_pruned"),
        "bpc_nodes_cg_failed": bpc.get("nodes_cg_failed"),
        "bpc_lb_certified": bpc_cert,
        "bpc_ub_improvement_vs_bp_pct": pct_change(number(bp, "ub"), number(bpc, "ub"), "smaller"),
        "bpc_lb_improvement_vs_bp_pct": (
            pct_change(bp_lb, bpc_lb, "larger") if bp_cert and bpc_cert else None
        ),
        "bpc_gap_reduction_vs_bp_pct": (
            pct_change(number(bp, "raw_gap_pct"), number(bpc, "raw_gap_pct"), "smaller")
            if bp_cert and bpc_cert else None
        ),
        "bpc_node_reduction_vs_bp_pct": pct_change(number(bp, "nodes"), number(bpc, "nodes"), "smaller"),
        "bpc_error": bpc.get("error"),
        "bp_error": bp.get("error"),
    }


def write_outputs(rows: list[dict], args: argparse.Namespace, out_dir: Path) -> None:
    meta = {
        "experiment": "Exp4: legacy BP versus legacy BPC deadline-cover-cut ablation",
        "purpose": "Isolate the empirical effect of deadline-cover cuts on solution quality and search efficiency.",
        "controlled_difference": "BP has lrmp_cuts=none; BPC has lrmp_cuts=standard. All other settings match.",
        "strict_cg_certification": False,
        "warning": (
            "When nodes_cg_failed is positive, UB remains a valid feasible value, "
            "but LB and gap are reported as non-certified search values."
        ),
        "process_assumption": PROCESS_ASSUMPTION,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "scale", "seed", "T", "I", "time_limit_sec",
        "bp_status", "bp_ub", "bp_lb", "bp_gap_pct", "bp_runtime_sec", "bp_nodes", "bp_nodes_pruned", "bp_nodes_cg_failed", "bp_lb_certified",
        "bpc_status", "bpc_ub", "bpc_lb", "bpc_gap_pct", "bpc_runtime_sec", "bpc_nodes", "bpc_nodes_pruned", "bpc_nodes_cg_failed", "bpc_lb_certified",
        "bpc_ub_improvement_vs_bp_pct", "bpc_lb_improvement_vs_bp_pct", "bpc_gap_reduction_vs_bp_pct", "bpc_node_reduction_vs_bp_pct",
        "bp_error", "bpc_error",
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    # Keep the earlier diagnostic runs (which used the pre-correction cut
    # definition) separate from results aligned with manuscript Eq. (37).
    out_dir = ROOT / "result" / "exp4" / "legacy_bp_vs_bpc_deadline_e"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Preserve rows already generated in earlier, smaller batches.  This lets
    # the user run one scale (or a few seeds) at a time without losing the
    # cumulative table.
    prior = read_json(out_dir / "summary.json") or {}
    row_by_key = {
        (row.get("scale"), row.get("seed")): row
        for row in prior.get("rows", [])
        if row.get("scale") is not None and row.get("seed") is not None
    }
    chosen = set(args.seeds) if args.seeds else None
    jobs = [
        (scale, seed) for scale in args.scales for seed in SCALES[scale].seeds
        if chosen is None or seed in chosen
    ]
    if chosen is not None:
        unknown = chosen - {seed for _, seed in jobs}
        if unknown:
            raise ValueError(f"Seeds do not belong to selected scales: {sorted(unknown)}")

    for index, (scale, seed) in enumerate(jobs, start=1):
        limit = SCALES[scale].time_limit_sec
        print(f"[{index}/{len(jobs)}] {scale} seed={seed}: BP then BPC; budget={limit:.0f}s each", flush=True)
        bp = run_one("bp", scale, seed, limit, args.ebrp_time_limit, out_dir, args.rerun)
        bpc = run_one("bpc", scale, seed, limit, args.ebrp_time_limit, out_dir, args.rerun)
        row = combine(scale, seed, bp, bpc)
        row_by_key[(scale, seed)] = row
        ordered_rows = [row_by_key[key] for key in sorted(row_by_key)]
        write_outputs(ordered_rows, args, out_dir)
        print(
            f"  BP: UB={row['bp_ub']}, gap={row['bp_gap_pct']}, nodes={row['bp_nodes']}; "
            f"BPC: UB={row['bpc_ub']}, gap={row['bpc_gap_pct']}, nodes={row['bpc_nodes']}",
            flush=True,
        )

    ordered_rows = [row_by_key[key] for key in sorted(row_by_key)]
    write_outputs(ordered_rows, args, out_dir)
    print(f"Finished. Summary: {out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
