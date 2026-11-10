"""Experiment 2: run the repaired paper-aligned BPC on one or more scales.

Outputs are written incrementally to ``result/exp2_new`` so an interrupted
run can be resumed.  Each instance is solved in a fresh worker process.
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

from experiments.paper_experiment_config import SCALES, MODEL_VERSION, PROCESS_ASSUMPTION


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="Optional seed filter; omitted means all configured seeds for the selected scales.")
    p.add_argument("--output-prefix", default="exp2_new_bpc_repaired")
    p.add_argument("--rerun", action="store_true", help="Recompute existing rows.")
    p.add_argument("--max-nodes", type=int, default=8000)
    p.add_argument("--ebrp-time-limit", type=float, default=None)
    return p.parse_args()


def _write(rows, out_json: Path, out_csv: Path, args):
    payload = {
        "experiment": "Experiment 2: repaired BPC upper/lower bound comparison",
        "model_version": MODEL_VERSION,
        "process_assumption": PROCESS_ASSUMPTION,
        "pricing": "EBRP for every pricing call; no Gurobi pricing fallback",
        "cut": "paper deadline-cover inequalities (equations 37-39)",
        "metrics": ["ub", "lb", "gap_pct", "runtime_sec", "nodes"],
        "scales": {s: {"T": SCALES[s].periods, "I": SCALES[s].items,
                       "seeds": list(SCALES[s].seeds), "time_limit_sec": SCALES[s].time_limit_sec}
                   for s in args.scales},
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["scale", "seed", "T", "I", "status", "ub", "lb", "gap_pct", "runtime_sec",
              "nodes", "nodes_created", "nodes_pruned", "nodes_cg_failed", "root_lb",
              "lower_bound_certified", "error"]
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def main():
    args = _parse_args()
    result_dir = ROOT / "result" / "exp2_new"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_json = result_dir / f"{args.output_prefix}.json"
    out_csv = result_dir / f"{args.output_prefix}.csv"
    tmp_dir = result_dir / f"_{args.output_prefix}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    old = {}
    if out_json.exists() and not args.rerun:
        try:
            old = {(r["scale"], int(r["seed"])): r
                   for r in json.loads(out_json.read_text(encoding="utf-8")).get("rows", [])}
        except (OSError, ValueError, KeyError, TypeError):
            old = {}

    jobs = []
    for scale in args.scales:
        seeds = list(SCALES[scale].seeds)
        if args.seeds is not None:
            seeds = [s for s in seeds if s in args.seeds]
        jobs.extend((scale, seed) for seed in seeds)

    rows = []
    for idx, (scale, seed) in enumerate(jobs, 1):
        key = (scale, seed)
        if key in old and not args.rerun:
            row = old[key]
            rows.append(row)
            print(f"[{idx}/{len(jobs)}] reuse {scale} seed={seed}", flush=True)
            continue
        spec = SCALES[scale]
        one = tmp_dir / f"{scale}_{seed}.json"
        log = tmp_dir / f"{scale}_{seed}.log"
        cmd = [sys.executable, "-B", "experiments/exp2_new_bpc_worker.py", scale, str(seed), str(one),
               "--limit", str(spec.time_limit_sec), "--max-nodes", str(args.max_nodes)]
        if args.ebrp_time_limit is not None:
            cmd += ["--ebrp-time-limit", str(args.ebrp_time_limit)]
        print(f"[{idx}/{len(jobs)}] run {scale} seed={seed} limit={spec.time_limit_sec:.0f}s", flush=True)
        try:
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                               timeout=spec.time_limit_sec + 180.0, check=False)
            row = json.loads(one.read_text(encoding="utf-8")) if one.exists() else {
                "scale": scale, "seed": seed, "status": "worker_timeout",
                "error": "worker did not return after solver budget plus setup buffer",
            }
        except subprocess.TimeoutExpired:
            row = {"scale": scale, "seed": seed, "status": "worker_timeout",
                   "error": "worker exceeded wall-clock buffer"}
        except Exception as exc:
            row = {"scale": scale, "seed": seed, "status": "runner_error", "error": str(exc)}
        rows.append(row)
        _write(rows, out_json, out_csv, args)
        print(f"  status={row.get('status')} UB={row.get('ub')} LB={row.get('lb')} "
              f"gap={row.get('gap_pct')} time={row.get('runtime_sec')} nodes={row.get('nodes')}", flush=True)
    _write(rows, out_json, out_csv, args)
    print(f"Finished: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
