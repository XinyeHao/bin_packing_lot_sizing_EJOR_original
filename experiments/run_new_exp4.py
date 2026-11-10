"""Run the non-strict-CG BP validation for selected Experiment-4 seeds."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=("small", "medium", "large"), default="large")
    p.add_argument("--seeds", nargs="+", type=int, default=[40401, 41301, 41601])
    p.add_argument("--limit", type=float, default=3600.0)
    p.add_argument("--ebrp-time-limit", type=float, default=20.0)
    args = p.parse_args()
    out_dir = ROOT / "result" / "exp4_new"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        out = out_dir / f"bp_{seed}_nonstrict.json"
        log = out_dir / f"bp_{seed}_nonstrict.log"
        cmd = [sys.executable, "-u", "experiments/new_exp4_bp_worker.py", args.scale,
               str(seed), str(out), "--limit", str(args.limit),
               "--ebrp-time-limit", str(args.ebrp_time_limit)]
        print(f"Run {args.scale} seed={seed} limit={args.limit:.0f}s", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            try:
                subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                               timeout=args.limit + 180.0, check=False)
            except subprocess.TimeoutExpired:
                out.write_text(json.dumps({"scale": args.scale, "seed": seed,
                                           "status": "worker_timeout"}, indent=2), encoding="utf-8")
        if out.exists():
            rows.append(json.loads(out.read_text(encoding="utf-8")))
    (out_dir / f"bp_{args.scale}_nonstrict_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Finished: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
