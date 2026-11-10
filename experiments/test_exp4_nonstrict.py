"""Test the legacy non-strict-CG BP configuration on three difficult seeds.

Usage from the project root:
    python experiments/test_exp4_nonstrict.py

The test runs large seeds 40401, 41301 and 41601 with BP, EBRP pricing,
no cuts, and strict_cg_certification=False.  Results are written under
result/exp4_test_nonstrict.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (40401, 41301, 41601)
OUT_DIR = ROOT / "result" / "exp4_test_nonstrict"
LIMIT = 3600.0
EBRP_LIMIT = 20.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in SEEDS:
        out = OUT_DIR / f"bp_{seed}.json"
        log = OUT_DIR / f"bp_{seed}.log"
        cmd = [
            sys.executable,
            "-u",
            "experiments/new_exp4_bp_worker.py",
            "large",
            str(seed),
            str(out),
            "--limit",
            str(LIMIT),
            "--ebrp-time-limit",
            str(EBRP_LIMIT),
        ]
        print(f"[{seed}] start: non-strict CG BP", flush=True)
        try:
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    cmd,
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=LIMIT + 180.0,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            out.write_text(
                json.dumps(
                    {"method": "BP", "scale": "large", "seed": seed,
                     "status": "worker_timeout"},
                    indent=2,
                ),
                encoding="utf-8",
            )
        if out.exists():
            row = json.loads(out.read_text(encoding="utf-8"))
            summary.append(row)
            print(
                f"[{seed}] status={row.get('status')} UB={row.get('ub')} "
                f"LB={row.get('lb')} gap={row.get('gap_pct')} "
                f"time={row.get('runtime_sec')} nodes={row.get('nodes')}",
                flush=True,
            )
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Finished. Results: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
