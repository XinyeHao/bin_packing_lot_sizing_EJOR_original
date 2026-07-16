"""Run med_cg with EBRP only. Gurobi pricing OOM on I=40/T=28."""
from __future__ import annotations

import json, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessor.preprocess import load_instance, preprocess
from model_and_algo.column_generation import solve_column_generation

INSTANCE_NAMES = ["set_med_cg_01", "set_med_cg_02", "set_med_cg_03"]


def main():
    out_path = ROOT / "result" / "batch_17_cg_compare.json"
    if out_path.exists():
        rows: list[dict] = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        rows = []

    for name in INSTANCE_NAMES:
        data = preprocess(load_instance(name))
        print(f"{name}: I={len(data.cured_items)} T={len(data.periods)} M={len(data.machines)}", flush=True)

        # EBRP only (Gurobi pricing OOM on this scale)
        t0 = time.perf_counter()
        result = solve_column_generation(
            data,
            pricing_mode="ebrp",
            use_gurobi_init=False,
            ebrp_force=True,
            max_iterations=5000,
            verbose=False,
            progress_interval_sec=0.0,
            time_limit=300.0,
            ebrp_time_limit=None,
            pricing_workers=0,
        )
        elapsed = time.perf_counter() - t0
        ex = result.extra or {}
        conv = ex.get("converged")
        iters = ex.get("num_iterations")
        cols = sum((ex.get("num_columns") or {}).values())
        lrmp = result.lower_bound
        print(f"  EBRP: LRMP={lrmp:.1f} it={iters} cols={cols} conv={conv} {elapsed:.1f}s", flush=True)
        rows.append({
            "instance": name, "pricing": "ebrp",
            "T": len(data.periods), "I": len(data.cured_items), "M": len(data.machines),
            "LRMP": lrmp, "iters": iters, "cols": cols, "converged": conv,
            "stop_reason": ex.get("stop_reason"), "last_min_rc": ex.get("last_min_rc"),
            "time_sec": round(elapsed, 1), "status": result.status, "note": "Gurobi_pricing_OOM",
        })
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
