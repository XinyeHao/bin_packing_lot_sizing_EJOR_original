"""One random small/med/large instance each; IM with 900/1800/3600s."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
from configuration import (
    BENCH_LARGE,
    BENCH_MEDIUM,
    BENCH_SMALL,
    BENCH_WINDOW,
    BENCH_LEAD_TIME_CHOICES,
    SetConfig,
)
from instance_generator.generator import generate_instance
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import preprocess

OUT = ROOT / "supplement_materials" / "im_bench_one_each.json"
SCALES = [
    ("small", *BENCH_SMALL, 900.0),
    ("medium", *BENCH_MEDIUM, 1800.0),
    ("large", *BENCH_LARGE, 3600.0),
]


def run_one(name: str, T: int, I: int, limit: float, seed: int) -> dict:
    scfg = SetConfig("small_cg", T, 5, 0, I, 5)
    raw = generate_instance(scfg, instance_id=seed % 1000, seed=seed)
    data = preprocess(raw)
    print(
        f"\n=== IM {name} seed={seed} T={len(data.periods)} I={len(data.cured_items)} "
        f"M={len(data.machines)} limit={limit:.0f}s "
        f"window={BENCH_WINDOW} LT={BENCH_LEAD_TIME_CHOICES} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    im = solve_im(data, time_limit=limit, mip_gap=0.0, verbose=False)
    elapsed = time.perf_counter() - t0
    ub, lb = im.objective, im.lower_bound
    gap = None
    if ub is not None and lb is not None and ub > 1e-9:
        gap = 100.0 * max(0.0, (ub - lb) / ub)
    proved = im.status == "optimal" or (
        ub is not None and lb is not None and abs(ub - lb) <= 1e-4
    )
    row = {
        "scale": name,
        "seed": seed,
        "T": T,
        "I": I,
        "M": 5,
        "U": 5,
        "time_limit": limit,
        "status": im.status,
        "ub": ub,
        "lb": lb,
        "gap_pct": None if gap is None else round(gap, 4),
        "runtime_sec": round(elapsed, 2),
        "proved": proved,
        "window": dict(BENCH_WINDOW),
        "LT": list(BENCH_LEAD_TIME_CHOICES),
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
    rng = random.Random()
    # one fresh seed per scale
    base = rng.randint(100000, 999999)
    seeds = {"small": base, "medium": base + 1, "large": base + 2}
    print(f"random base seed={base}", flush=True)
    rows = []
    t_all = time.perf_counter()
    for name, T, I, limit in SCALES:
        rows.append(run_one(name, T, I, limit, seeds[name]))
        OUT.write_text(
            json.dumps(
                {
                    "note": "new BENCH window [12,16], one random instance per scale",
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print("\n=== SUMMARY ===", flush=True)
    for r in rows:
        print(
            f"{r['scale']:7s} seed={r['seed']} proved={r['proved']} "
            f"gap={r['gap_pct']} sec={r['runtime_sec']}/{r['time_limit']}",
            flush=True,
        )
    print(f"wall={time.perf_counter()-t_all:.1f}s wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
