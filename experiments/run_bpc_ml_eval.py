"""Evaluate BPC+ML on small (900s) and medium (1800s) instance sets."""
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

from configuration import ML_BRANCH_MODEL_PATH
from model_and_algo.branch_and_price import solve_branch_and_price
from preprocessor.instance_db import load_instance_from_db
from preprocessor.preprocess import preprocess

SMALL = [f"set_im_opt900_{i:02d}" for i in range(1, 6)]
MEDIUM = [f"set_im_med_{i:02d}" for i in range(1, 6)]
OUT = ROOT / "supplement_materials" / "bpc_ml_eval.json"

# Prior references (IM / ordinary BPC best_bound where known)
REF = {
    "set_im_opt900_01": {"im": 68.0, "bb": 778.0, "bb_proved": True},
    "set_im_opt900_02": {"im": 40.0},
    "set_im_opt900_03": {"im": 63.4, "bb": 907.5, "bb_proved": False},
    "set_im_opt900_04": {"im": 30.2, "bb": 706.9, "bb_proved": True},
    "set_im_opt900_05": {"im": 252.0},
}


def run_one(name: str, limit: float) -> dict:
    data = preprocess(load_instance_from_db(name))
    print(
        f"\n=== {name} T={len(data.periods)} I={len(data.cured_items)} "
        f"M={len(data.machines)} ML {limit:.0f}s ===",
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
    r = solve_branch_and_price(
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
    proved = r.status == "optimal" or (
        r.objective is not None
        and r.lower_bound is not None
        and abs(r.objective - r.lower_bound) <= 1e-4
    )
    gap = None
    if r.objective and r.lower_bound is not None and r.objective > 1e-9:
        gap = 100.0 * max(0.0, (r.objective - r.lower_bound) / r.objective)
    nodes = r.extra.get("nodes_explored") if r.extra else None
    row = {
        "name": name,
        "strategy": "ml",
        "status": r.status,
        "proved": proved,
        "ub": r.objective,
        "lb": r.lower_bound,
        "gap": gap,
        "time": elapsed,
        "nodes": nodes,
        "ref": REF.get(name),
    }
    print(
        f"  ML: proved={proved} UB={r.objective} LB={r.lower_bound} "
        f"gap={gap} t={elapsed:.1f}s nodes={nodes} ref={REF.get(name)}",
        flush=True,
    )
    try:
        data.dispose()
    except Exception:
        pass
    gc.collect()
    time.sleep(0.5)
    return row


def main():
    assert ML_BRANCH_MODEL_PATH.exists(), f"missing model {ML_BRANCH_MODEL_PATH}"
    rows = []
    print(f"Model: {ML_BRANCH_MODEL_PATH}", flush=True)
    for name in SMALL:
        rows.append(run_one(name, 900.0))
    for name in MEDIUM:
        rows.append(run_one(name, 1800.0))
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}", flush=True)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
