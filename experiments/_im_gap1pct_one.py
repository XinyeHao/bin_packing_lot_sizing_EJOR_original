"""Worker: one IM solve (mip_gap=1% or time limit), hard cleanup, exit."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import gurobipy as gp

import configuration as cfg
from configuration import BENCH_LARGE, BENCH_MEDIUM, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import preprocess

PROCESS_ASSUMPTION = (
    "All items in a batch leave the autoclave and enter inventory together "
    "at the batch maximum curing duration."
)


def _gap(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return round(100.0 * max(0.0, (ub - lb) / ub), 4)


def cleanup(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass
    gc.collect()


def main() -> None:
    # argv: scale seed limit mip_gap out_json
    scale = sys.argv[1]
    seed = int(sys.argv[2])
    limit = float(sys.argv[3])
    mip_gap = float(sys.argv[4])
    out_json = Path(sys.argv[5])

    cfg.apply_bench_generation()
    T, I = BENCH_MEDIUM if scale == "medium" else BENCH_LARGE
    print(
        f"=== IM {scale} seed={seed} T={T} I={I} "
        f"limit={limit:.0f}s mip_gap={100 * mip_gap:.0f}% ===",
        flush=True,
    )

    raw = data = im = None
    row: dict
    try:
        raw = generate_instance(SetConfig("small_cg", T, 5, 0, I, 5), seed % 1000, seed)
        data = preprocess(raw)
        t0 = time.perf_counter()
        im = solve_im(data, time_limit=limit, mip_gap=mip_gap, verbose=False)
        elapsed = time.perf_counter() - t0
        ub, lb = im.objective, im.lower_bound
        row = {
            "process_assumption": PROCESS_ASSUMPTION,
            "model_version": "common_unload_v2_span_link",
            "scale": scale,
            "seed": seed,
            "T": T,
            "I": I,
            "M": 5,
            "time_limit": limit,
            "mip_gap": mip_gap,
            "status": im.status,
            "ub": ub,
            "lb": lb,
            "gap_pct": _gap(ub, lb),
            "runtime_sec": round(elapsed, 2),
            "proved": im.status == "optimal"
            or (ub is not None and lb is not None and abs(ub - lb) <= 1e-4),
        }
        print(
            f"  -> status={row['status']} ub={ub} lb={lb} gap={row['gap_pct']}% "
            f"sec={row['runtime_sec']}",
            flush=True,
        )
    except Exception as e:
        row = {"scale": scale, "seed": seed, "error": str(e)}
        print(f"  FAIL {scale} seed={seed}: {e}", flush=True)
    finally:
        cleanup(im, data, raw)

    out_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
    cleanup()
    os._exit(0 if "error" not in row else 1)


if __name__ == "__main__":
    main()
