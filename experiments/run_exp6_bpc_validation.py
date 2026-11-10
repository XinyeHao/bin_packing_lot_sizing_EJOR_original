"""Experiment 6: legacy-practical BPC validation.

Runs BPC only, using EBRP pricing with a fixed 20-second pricing cap and
strict CG certification disabled.  Each row contains only the requested
metrics: UB, LB, raw gap, runtime, processed nodes, and CG-failed nodes.

Examples:
  python experiments/run_exp6_bpc_validation.py --scales small
  python experiments/run_exp6_bpc_validation.py --scales medium --seeds 30101 30201
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

EXP6_VERSION = "legacy_practical_bpc_ebrp20_v1"


def parse_args():
    p = argparse.ArgumentParser(description="Exp6: practical BPC validation with EBRP cap 20s.")
    p.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=["small"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--time-limit", type=float, default=None,
                   help="Override the scale time limit for every selected seed.")
    p.add_argument("--output-prefix", default="exp6_bpc_validation")
    p.add_argument("--rerun", action="store_true", help="Ignore existing rows and rerun them.")
    return p.parse_args()


def run_one(scale, seed, limit, tmp_dir):
    out = tmp_dir / f"{scale}_{seed}.json"
    log = tmp_dir / f"{scale}_{seed}.log"
    command = [sys.executable, "-B", "experiments/_exp6_bpc_worker.py",
               scale, str(seed), str(limit), str(out)]
    with log.open("w", encoding="utf-8") as handle:
        try:
            subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                           timeout=limit + 180.0, check=False)
        except subprocess.TimeoutExpired:
            return {"scale": scale, "seed": seed, "status": "worker_timeout",
                    "error": "worker exceeded solver budget plus 180-second buffer"}
    if out.exists():
        try:
            return json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"scale": scale, "seed": seed, "status": "worker_error", "error": str(exc)}
    return {"scale": scale, "seed": seed, "status": "worker_error",
            "error": f"no output JSON; inspect {log.name}"}


def write(rows, args, json_path, csv_path):
    payload = {
        "experiment": "Exp6: practical BPC validation",
        "experiment_version": EXP6_VERSION,
        "protocol": {
            "pricing": "EBRP",
            "ebrp_time_limit_sec": 20.0,
            "strict_cg_certification": False,
            "cuts": "standard",
            "branch_strategy": "frac_sb",
        },
        "process_assumption": PROCESS_ASSUMPTION,
        "model_version": MODEL_VERSION,
        "scales": args.scales,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["scale", "seed", "T", "I", "time_limit_sec", "status", "ub", "lb",
              "gap_pct", "runtime_sec", "nodes", "nodes_cg_failed", "error"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main():
    args = parse_args()
    result_dir = ROOT / "result" / "exp6"
    result_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = result_dir / f"_{args.output_prefix}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / f"{args.output_prefix}.json"
    csv_path = result_dir / f"{args.output_prefix}.csv"
    jobs = []
    for scale in args.scales:
        seeds = args.seeds if args.seeds is not None else list(SCALES[scale].seeds)
        limit = float(args.time_limit or SCALES[scale].time_limit_sec)
        jobs.extend((scale, int(seed), limit) for seed in seeds)
    rows = []
    for i, (scale, seed, limit) in enumerate(jobs, 1):
        cache = tmp_dir / f"{scale}_{seed}.json"
        if cache.exists() and not args.rerun:
            row = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[{i}/{len(jobs)}] reuse {scale} seed={seed}: {row.get('status')}", flush=True)
        else:
            print(f"[{i}/{len(jobs)}] {scale} seed={seed}, limit={limit:.0f}s, EBRP=20s", flush=True)
            row = run_one(scale, seed, limit, tmp_dir)
        rows.append(row)
        write(rows, args, json_path, csv_path)
        print(f"  UB={row.get('ub')} LB={row.get('lb')} gap={row.get('gap_pct')}% "
              f"time={row.get('runtime_sec')}s nodes={row.get('nodes')} "
              f"cg_failed={row.get('nodes_cg_failed')}", flush=True)
    write(rows, args, json_path, csv_path)
    print(f"Finished. CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
