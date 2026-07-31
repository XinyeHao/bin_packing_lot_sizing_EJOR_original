"""Worker: one large seed, BPC+ML, then hard cleanup and exit."""
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

import configuration as cfg
from configuration import ML_BRANCH_MODEL_PATH, SetConfig
from instance_generator import generator as genmod
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import solve_branch_and_price
from preprocessor.preprocess import preprocess


def cleanup(*objs) -> None:
    for o in objs:
        try:
            if o is not None and hasattr(o, "dispose"):
                o.dispose()
        except Exception:
            pass
    gc.collect()
    try:
        import gurobipy as gp

        gp.disposeDefaultEnv()
    except Exception:
        pass
    gc.collect()


def main() -> None:
    seed = int(sys.argv[1])
    limit = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
    assert ML_BRANCH_MODEL_PATH.exists(), f"missing {ML_BRANCH_MODEL_PATH}"

    cfg.apply_bench_generation()

    scfg = SetConfig("small_cg", 35, 5, 0, 50, 5)
    print(
        f"=== BPC+ML large seed={seed} T=35 I=50 M=5 limit={limit:.0f}s ===",
        flush=True,
    )
    raw = None
    data = None
    result = None
    try:
        raw = generate_instance(scfg, instance_id=seed % 100, seed=seed)
        data = preprocess(raw)
        print(
            f"T={len(data.periods)} I={len(data.cured_items)} M={len(data.machines)}",
            flush=True,
        )
        cg = dict(
            pricing_mode="ebrp",
            ebrp_force=True,
            use_gurobi_init=False,
            skip_cross_check=True,
            pricing_inprocess=True,
            verbose=False,
            progress_interval_sec=60.0,
            time_limit=limit,
            lrmp_cuts="standard",
        )
        t0 = time.perf_counter()
        result = solve_branch_and_price(
            data,
            max_nodes=8000,
            im_time_limit=90.0,
            time_limit=limit,
            use_node_im=False,
            root_im=True,
            final_im=True,
            branch_strategy="ml",
            cg_kwargs=cg,
        )
        elapsed = time.perf_counter() - t0
        proved = result.status == "optimal" or (
            result.objective is not None
            and result.lower_bound is not None
            and abs(result.objective - result.lower_bound) <= 1e-4
        )
        gap = None
        if result.objective and result.lower_bound is not None and result.objective > 1e-9:
            gap = 100.0 * max(0.0, (result.objective - result.lower_bound) / result.objective)
        nodes = result.extra.get("nodes_explored") if result.extra else None
        row = {
            "seed": seed,
            "strategy": "ml",
            "status": result.status,
            "proved": proved,
            "ub": result.objective,
            "lb": result.lower_bound,
            "gap": gap,
            "nodes": nodes,
            "time": elapsed,
            "T": 35,
            "I": 50,
            "M": 5,
            "U": 5,
            "LT": list(lt),
            "window": dict(window),
        }
        print(
            f"DONE seed={seed}: proved={proved} status={result.status} "
            f"UB={result.objective} LB={result.lower_bound} gap={gap} "
            f"nodes={nodes} t={elapsed:.1f}s",
            flush=True,
        )
        out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else (
            ROOT / "supplement_materials" / f"_bpc_ml_large_{seed}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"WROTE {out_path}", flush=True)
        print("RESULT_JSON=" + json.dumps(row, ensure_ascii=True), flush=True)
    finally:
        cleanup(result, data, raw)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
