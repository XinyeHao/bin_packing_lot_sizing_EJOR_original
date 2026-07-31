"""Ordinary BPC (best_bound) on small opt900 03/04 vs prior BPC+ML."""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

from model_and_algo.branch_and_price import solve_branch_and_price
from preprocessor.instance_db import load_instance_from_db
from preprocessor.preprocess import preprocess

ML_REF = {
    "set_im_opt900_03": {"time": 900.7, "nodes": 67, "status": "optimal", "ub": 17165.30},
    "set_im_opt900_04": {"time": 581.7, "nodes": 33, "status": "optimal", "ub": 13460.00},
}
IM_REF = {"set_im_opt900_03": 63.4, "set_im_opt900_04": 30.2}
LIMIT = 900.0


def run(name: str) -> dict:
    data = preprocess(load_instance_from_db(name))
    print(
        f"\n=== {name} T={len(data.periods)} I={len(data.cured_items)} "
        f"M={len(data.machines)} ===",
        flush=True,
    )
    cg = dict(
        pricing_mode="ebrp",
        ebrp_force=True,
        use_gurobi_init=False,
        skip_cross_check=True,
        pricing_inprocess=True,
        verbose=False,
        progress_interval_sec=30.0,
        time_limit=LIMIT,
        lrmp_cuts="standard",
    )
    t0 = time.perf_counter()
    r = solve_branch_and_price(
        data,
        max_nodes=8000,
        im_time_limit=90.0,
        time_limit=LIMIT,
        use_node_im=False,
        root_im=True,
        final_im=True,
        branch_strategy="best_bound",
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
    ml = ML_REF[name]
    print(
        f"  best_bound: status={r.status} proved={proved} UB={r.objective} "
        f"LB={r.lower_bound} gap={gap} t={elapsed:.1f}s nodes={nodes}",
        flush=True,
    )
    print(
        f"  vs ML: status={ml['status']} t={ml['time']:.1f}s "
        f"nodes={ml['nodes']} UB={ml['ub']}",
        flush=True,
    )
    print(f"  vs IM: t={IM_REF[name]:.1f}s", flush=True)

    if proved and ml["status"] == "optimal":
        winner = "best_bound" if elapsed < ml["time"] else "ML"
    elif not proved and ml["status"] == "optimal":
        winner = "ML"
    elif proved and ml["status"] != "optimal":
        winner = "best_bound"
    else:
        winner = "tie/unclear"
    print(f"  winner={winner}", flush=True)

    try:
        data.dispose()
    except Exception:
        pass
    gc.collect()
    time.sleep(0.5)
    return {
        "name": name,
        "proved": proved,
        "status": r.status,
        "ub": r.objective,
        "lb": r.lower_bound,
        "gap": gap,
        "time": elapsed,
        "nodes": nodes,
        "winner": winner,
    }


def main() -> None:
    rows = []
    for n in ["set_im_opt900_03", "set_im_opt900_04"]:
        rows.append(run(n))
    print("\n=== SUMMARY ===", flush=True)
    for r in rows:
        ml_better = r["winner"] == "ML"
        print(
            f"{r['name']}: best_bound t={r['time']:.1f}s proved={r['proved']} "
            f"| ML_better={ml_better} winner={r['winner']}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
