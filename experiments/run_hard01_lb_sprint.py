"""hard_01 下界冲刺：B&P 时限默认 < 50% IM 耗时。"""

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
OUTPUT = RESULT_DIR / "set_b_hard_01_lb_sprint.json"

SPRINTS = (
    {
        "label": "baseline_y0_no_y1",
        "branch_strategy": "baseline_y0_dfs",
        "lazy_y1_enabled": False,
        "y0_depth_budget": DEFAULT_BP_Y0_DEPTH_BUDGET,
    },
    {
        "label": "new_policy_layer2",
        "branch_strategy": "new_policy",
        "lazy_y1_enabled": False,
        "y0_depth_budget": DEFAULT_BP_Y0_DEPTH_BUDGET,
    },
    {
        "label": "new_policy_full_lazy",
        "branch_strategy": "new_policy",
        "lazy_y1_enabled": True,
        "y0_depth_budget": DEFAULT_BP_Y0_DEPTH_BUDGET,
    },
)


def run_sprint(
    name: str,
    *,
    label: str,
    branch_strategy: str,
    lazy_y1_enabled: bool,
    y0_depth_budget: int,
    bp_time_limit: float,
    bp_max_nodes: int,
    root_im: bool,
    final_im: bool,
) -> dict:
    data = preprocess(load_instance(INSTANCES_DIR / f"{name}.json"))
    im_ref = REF_IM_SEC.get(name)
    print(
        f"\n=== {name} [{label}] strategy={branch_strategy} "
        f"BP={bp_time_limit}s (IM_ref={im_ref}s) lazy_y1={lazy_y1_enabled} ===",
        flush=True,
    )

    t0 = time.perf_counter()
    bp = BranchAndPriceSolver(
        data,
        max_nodes=bp_max_nodes,
        cg_max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        cg_init_time_limit=CG_INIT,
        im_time_limit=min(60.0, bp_time_limit * 0.1),
        mip_gap=DEFAULT_MIP_GAP,
        time_limit=bp_time_limit,
        cg_kwargs=SET_B_CG_FAST,
        use_node_im=False,
        root_im=root_im,
        final_im=final_im,
        branch_strategy=branch_strategy,
        y0_depth_budget=y0_depth_budget,
        lazy_y1_enabled=lazy_y1_enabled,
    )
    result = bp.solve()
    bp_sec = time.perf_counter() - t0

    ref_ub = REF_IM_UB.get(name)
    row = {
        "instance": name,
        "label": label,
        "branch_strategy": branch_strategy,
        "lazy_y1_enabled": lazy_y1_enabled,
        "y0_depth_budget": y0_depth_budget,
        "ref_im_sec": im_ref,
        "bp_time_limit": bp_time_limit,
        "bp_lb_root": round(result.root_lrmp, 2) if result.root_lrmp is not None else None,
        "bp_lb_best": round(result.best_node_lb, 2) if result.best_node_lb is not None else None,
        "bp_ub": round(result.objective, 2) if result.objective is not None else None,
        "bp_sec": round(bp_sec, 2),
        "bp_status": result.status,
        "bp_nodes": result.nodes_explored,
        "bp_nodes_created": result.nodes_created,
        "bp_cg_failed": result.nodes_cg_failed,
        "bp_infeasible": result.nodes_infeasible,
        "bp_max_depth": result.max_depth,
        "bp_y1_lazy_created": result.y1_lazy_created,
        "bp_lazy_popped": result.lazy_popped,
        "bp_dfs_popped": result.dfs_popped,
        "ref_im_ub": ref_ub,
        "bp_gap_pct": round((ref_ub - result.best_node_lb) / ref_ub * 100, 4)
        if ref_ub and result.best_node_lb
        else None,
        "bp_proven_optimal": result.status == "optimal",
    }
    print(
        f"  LB_root={row['bp_lb_root']} LB_best={row['bp_lb_best']} gap={row['bp_gap_pct']}% "
        f"status={row['bp_status']} nodes={row['bp_nodes']} dfs={row['bp_dfs_popped']} "
        f"lazy={row['bp_lazy_popped']} cg_fail={row['bp_cg_failed']} ({row['bp_sec']}s)",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="hard LB sprint: BP time < 50% IM time")
    parser.add_argument("--instance", default="set_b_hard_01")
    parser.add_argument(
        "--bp-time",
        type=float,
        default=None,
        help="显式 B&P 时限；默认 auto = ratio × REF_IM_SEC",
    )
    parser.add_argument("--bp-nodes", type=int, default=5000)
    parser.add_argument(
        "--bp-im-ratio",
        type=float,
        default=DEFAULT_BP_IM_TIME_RATIO,
        help="auto 时限比例（须 < 0.5，默认 0.45）",
    )
    parser.add_argument(
        "--sprint",
        choices=[s["label"] for s in SPRINTS] + ["all"],
        default="all",
    )
    parser.add_argument("--root-im", action="store_true", help="默认关闭，避免占满短时限")
    parser.add_argument("--final-im", action="store_true")
    args = parser.parse_args()

    ensure_instances()
    sprints = SPRINTS if args.sprint == "all" else [s for s in SPRINTS if s["label"] == args.sprint]
    bp_limit = (
        args.bp_time
        if args.bp_time is not None
        else bp_time_cap_for_instance(args.instance, ratio=args.bp_im_ratio)
    )
    im_ref = REF_IM_SEC.get(args.instance)
    print(
        f"LB sprint | instance={args.instance} IM_ref={im_ref}s "
        f"BP_limit={bp_limit}s (ratio={args.bp_im_ratio}) runs={[s['label'] for s in sprints]}",
        flush=True,
    )

    rows = [
        run_sprint(
            args.instance,
            label=s["label"],
            branch_strategy=s["branch_strategy"],
            lazy_y1_enabled=s["lazy_y1_enabled"],
            y0_depth_budget=s["y0_depth_budget"],
            bp_time_limit=bp_limit,
            bp_max_nodes=args.bp_nodes,
            root_im=args.root_im,
            final_im=args.final_im,
        )
        for s in sprints
    ]

    payload = {
        "generation": SET_B_GENERATION,
        "bp_im_ratio": args.bp_im_ratio,
        "ref_im_sec": im_ref,
        "bp_time_limit_sec": bp_limit,
        "bp_max_nodes": args.bp_nodes,
        "pricing": "EBRP+fixed_y",
        "cg_fast": SET_B_CG_FAST,
        "runs": rows,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
