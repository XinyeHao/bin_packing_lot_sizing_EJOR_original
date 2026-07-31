"""IM on the same 5 medium seeds as bpc_ml_medium_batch (T=30,I=40), 1800s, no BPC."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
from configuration import BENCH_MEDIUM, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import preprocess

SEEDS = [8201, 8301, 8401, 8501, 8601]
LIMIT = 1800.0
OUT = ROOT / "supplement_materials" / "im_medium_batch_8201.json"


def run_one(seed: int) -> dict:
    T, I = BENCH_MEDIUM
    scfg = SetConfig("small_cg", T, 5, 0, I, 5)
    raw = generate_instance(scfg, instance_id=seed % 1000, seed=seed)
    data = preprocess(raw)
    print(
        f"\n=== IM seed={seed} T={len(data.periods)} I={len(data.cured_items)} "
        f"limit={LIMIT:.0f}s ===",
        flush=True,
    )
    t0 = time.perf_counter()
    im = solve_im(data, time_limit=LIMIT, mip_gap=0.0, verbose=False)
    elapsed = time.perf_counter() - t0
    ub, lb = im.objective, im.lower_bound
    gap = None
    if ub is not None and lb is not None and ub > 1e-9:
        gap = 100.0 * max(0.0, (ub - lb) / ub)
    proved = im.status == "optimal" or (
        ub is not None and lb is not None and abs(ub - lb) <= 1e-4
    )
    row = {
        "seed": seed,
        "T": T,
        "I": I,
        "status": im.status,
        "ub": ub,
        "lb": lb,
        "gap_pct": None if gap is None else round(gap, 4),
        "runtime_sec": round(elapsed, 2),
        "proved": proved,
    }
    print(
        f"  -> status={row['status']} ub={ub} lb={lb} gap={row['gap_pct']} "
        f"sec={row['runtime_sec']} proved={proved}",
        flush=True,
    )
    try:
        data.dispose()
    except Exception:
        pass
    return row


def main():
    cfg.apply_bench_generation()
    rows = []
    t_all = time.perf_counter()
    for seed in SEEDS:
        try:
            rows.append(run_one(seed))
        except Exception as e:
            print(f"  FAIL seed={seed}: {e}", flush=True)
            rows.append({"seed": seed, "error": str(e), "proved": False})
        OUT.write_text(
            json.dumps(
                {
                    "scale": "medium",
                    "method": "im_gurobi",
                    "time_limit": LIMIT,
                    "mip_gap": 0.0,
                    "seeds": SEEDS,
                    "note": "same seeds as bpc_ml_medium_batch.json",
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    n_ok = sum(1 for r in rows if r.get("proved"))
    times = [r["runtime_sec"] for r in rows if "runtime_sec" in r]
    print("\n=== SUMMARY ===", flush=True)
    print(
        f"proved {n_ok}/{len(rows)} runtime_sum={sum(times):.1f}s "
        f"wall={time.perf_counter()-t_all:.1f}s",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
