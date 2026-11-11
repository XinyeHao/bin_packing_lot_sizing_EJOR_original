"""Medium seed 31101: Ryan–Foster BP + odd SRI k=3 (no Deadline)."""
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
from experiments.paper_experiment_config import make_set_config, scale_spec
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess

SEED = 31101
OUT = ROOT / "result" / "rf_sri3_31101.json"
SRI_MAX = 12
CAND_MAX = 16


def gap(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return round(100.0 * max(0.0, (ub - lb) / ub), 4)


def main():
    spec = scale_spec("medium")
    cfg.apply_bench_generation()
    data = preprocess(generate_instance(make_set_config(spec), SEED % 1000, SEED))
    cuts = {
        "deadline_cover": False,
        "sri": True,
        "sri_max_cuts": SRI_MAX,
        "sri_violation_tol": 1e-6,
        "sri_cardinalities": (3,),
        "sri_cand_max": CAND_MAX,
        "tight_deadline": False,
    }
    print(
        f"======== RF+SRI3 seed={SEED} T={spec.periods} I={spec.items} "
        f"sri_max={SRI_MAX} limit={spec.time_limit_sec:.0f}s ========",
        flush=True,
    )
    t0 = time.perf_counter()
    bp = BranchAndPriceSolver(
        data,
        max_nodes=8000,
        time_limit=spec.time_limit_sec,
        mip_gap=0.01,
        branch_strategy="ryan_foster",
        use_node_im=False,
        root_im=False,
        final_im=False,
        strict_cg_certification=True,
        cg_kwargs={
            "pricing_mode": "ebrp",
            "ebrp_force": True,
            "use_gurobi_init": True,
            "pricing_inprocess": True,
            "skip_cross_check": True,
            "lrmp_cuts": cuts,
            "max_iterations": 120,
            "ebrp_time_limit": 20.0,
            "verbose": False,
        },
    )
    res = bp.solve()
    n_sri = 0
    by_r: dict[int, int] = {}
    # root / last CG may keep sri on nodes; report root_lb vs pure RF baseline
    row = {
        "method": "bp_ryan_foster_sri3",
        "scale": "medium",
        "seed": SEED,
        "T": spec.periods,
        "I": spec.items,
        "cuts": "sri_k3",
        "sri_max_cuts": SRI_MAX,
        "sri_cand_max": CAND_MAX,
        "branch_strategy": "ryan_foster",
        "status": res.status,
        "ub": res.objective,
        "lb": res.lower_bound,
        "gap_pct": gap(res.objective, res.lower_bound),
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "nodes": res.nodes_explored,
        "nodes_created": res.nodes_created,
        "nodes_pruned": res.nodes_pruned,
        "nodes_cg_failed": res.nodes_cg_failed,
        "root_lb": res.root_valid_lb,
        "root_lrmp": res.root_lrmp,
        "root_converged": res.root_cg_converged,
    }
    rf_ref = ROOT / "result" / "rf_31101.json"
    if rf_ref.exists():
        ref = json.loads(rf_ref.read_text(encoding="utf-8"))
        row["rf_no_cut_reference"] = {
            "ub": ref.get("ub"),
            "lb": ref.get("lb"),
            "gap_pct": ref.get("gap_pct"),
            "runtime_sec": ref.get("runtime_sec"),
            "nodes": ref.get("nodes"),
            "root_lb": ref.get("root_lb"),
        }
        if row["root_lb"] is not None and ref.get("root_lb") is not None:
            row["root_lb_delta_vs_rf"] = round(
                float(row["root_lb"]) - float(ref["root_lb"]), 4
            )
    OUT.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)
    del bp, res, data
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass


if __name__ == "__main__":
    main()
