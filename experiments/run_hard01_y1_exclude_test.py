"""hard_01：Y=0 only vs exclude_conflict vs fix_y1 对比。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import INSTANCES_DIR, RESULT_DIR
from configuration import (
    DEFAULT_BP_IM_TIME_RATIO,
    DEFAULT_BP_Y0_DEPTH_BUDGET,
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_MIP_GAP,
    SET_B_CG_FAST,
    SET_B_GENERATION,
    SET_EXPERIMENT_LIMITS,
)
from experiments.run_set_b_hard_bp_test import (
    REF_IM_SEC,
    REF_IM_UB,
    bp_time_cap_for_instance,
    ensure_instances,
)
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import load_instance, preprocess

CG_INIT = SET_EXPERIMENT_LIMITS["B"]["cg_init"]
OUTPUT = RESULT_DIR / "set_b_hard_y1_exclude_results.json"

SPRINTS = (
    {"label": "y0_only", "branch_strategy": "new_policy", "y1_sibling_mode": "off"},
    {"label": "exclude_conflict", "branch_strategy": "new_policy", "y1_sibling_mode": "exclude_conflict"},
    {"label": "fix_y1_lazy", "branch_strategy": "new_policy", "y1_sibling_mode": "fix_y1"},
)


def run_sprint(
    name: str,
    *,
    label: str,
    branch_strategy: str,
    y1_sibling_mode: str,
    bp_time_limit: float,
    bp_max_nodes: int,
) -> dict:
    data = preprocess(load_instance(INSTANCES_DIR / f"{name}.json"))
    print(
        f"\n=== {name} [{label}] y1_sibling={y1_sibling_mode} BP={bp_time_limit}s ===",
        flush=True,
    )
    t0 = time.perf_counter()
    bp = BranchAndPriceSolver(
        data,
        max_nodes=bp_max_nodes,
        cg_max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        cg_init_time_limit=CG_INIT,
        mip_gap=DEFAULT_MIP_GAP,
        time_limit=bp_time_limit,
        cg_kwargs=SET_B_CG_FAST,
        use_node_im=False,
        root_im=False,
        final_im=False,
        branch_strategy=branch_strategy,
        y0_depth_budget=DEFAULT_BP_Y0_DEPTH_BUDGET,
        y1_sibling_mode=y1_sibling_mode,
    )
    result = bp.solve()
    bp_sec = time.perf_counter() - t0
    ref_ub = REF_IM_UB.get(name)
    row = {
        "instance": name,
        "label": label,
        "y1_sibling_mode": y1_sibling_mode,
        "bp_time_limit": bp_time_limit,
        "bp_lb_root": round(result.root_lrmp, 2) if result.root_lrmp else None,
        "bp_lb_best": round(result.best_node_lb, 2) if result.best_node_lb else None,
        "bp_sec": round(bp_sec, 2),
        "bp_status": result.status,
        "bp_nodes": result.nodes_explored,
        "bp_cg_failed": result.nodes_cg_failed,
        "bp_y1_exclude_created": result.y1_exclude_created,
        "bp_y1_lazy_created": result.y1_lazy_created,
        "bp_dfs_popped": result.dfs_popped,
        "bp_lazy_popped": result.lazy_popped,
        "ref_im_ub": ref_ub,
        "bp_gap_pct": round((ref_ub - result.best_node_lb) / ref_ub * 100, 4)
        if ref_ub and result.best_node_lb
        else None,
    }
    print(
        f"  LB_best={row['bp_lb_best']} gap={row['bp_gap_pct']}% nodes={row['bp_nodes']} "
        f"exclude={row['bp_y1_exclude_created']} lazy_y1={row['bp_y1_lazy_created']} "
        f"cg_fail={row['bp_cg_failed']} ({row['bp_sec']}s)",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="set_b_hard_01")
    parser.add_argument("--bp-time", type=float, default=None)
    parser.add_argument("--bp-nodes", type=int, default=5000)
    parser.add_argument("--bp-im-ratio", type=float, default=DEFAULT_BP_IM_TIME_RATIO)
    args = parser.parse_args()

    ensure_instances()
    bp_limit = args.bp_time or bp_time_cap_for_instance(args.instance, ratio=args.bp_im_ratio)
    rows = [
        run_sprint(
            args.instance,
            label=s["label"],
            branch_strategy=s["branch_strategy"],
            y1_sibling_mode=s["y1_sibling_mode"],
            bp_time_limit=bp_limit,
            bp_max_nodes=args.bp_nodes,
        )
        for s in SPRINTS
    ]
    payload = {
        "generation": SET_B_GENERATION,
        "ref_im_sec": REF_IM_SEC.get(args.instance),
        "bp_time_limit_sec": bp_limit,
        "runs": rows,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
