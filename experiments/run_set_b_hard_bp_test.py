"""Set-B hard_01–03：三策略 B&P 对比 + CG/IM 基准。"""

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
    BP_STRATEGIES,
    DEFAULT_BP_FINAL_IM,
    DEFAULT_BP_IM_TIME_RATIO,
    DEFAULT_BP_ROOT_IM,
    DEFAULT_BP_STRATEGY,
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_MIP_GAP,
    SET_B,
    SET_B_CG_FAST,
    SET_B_GENERATION,
    SET_EXPERIMENT_LIMITS,
)
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from model_and_algo.column_generation import solve_column_generation
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess

SEEDS = (10001, 10002, 10003)
CG_INIT = SET_EXPERIMENT_LIMITS["B"]["cg_init"]
IM_LIMIT = min(SET_EXPERIMENT_LIMITS["B"]["im_time"], 1200.0)
BP_CG_KWARGS = SET_B_CG_FAST
OUTPUT = RESULT_DIR / "set_b_hard_bp_results.json"

REF_IM_UB = {
    "set_b_hard_01": 50061.53,
    "set_b_hard_02": 35822.64,
    "set_b_hard_03": 40030.13,
}

# 实测 IM 耗时（秒），用于 B&P 时限 = ratio × im_sec（须 < 50%）
REF_IM_SEC = {
    "set_b_hard_01": 298.0,
    "set_b_hard_02": 430.0,
    "set_b_hard_03": 1065.0,
}


def bp_time_cap_for_instance(
    instance_name: str,
    ratio: float | None = None,
    fallback_im_sec: float | None = None,
) -> float:
    """B&P 时限：不超过同算例 IM 耗时的一半以上（默认 45%）。"""
    from configuration import DEFAULT_BP_IM_TIME_RATIO

    r = DEFAULT_BP_IM_TIME_RATIO if ratio is None else ratio
    im_sec = REF_IM_SEC.get(instance_name)
    if im_sec is None:
        im_sec = fallback_im_sec if fallback_im_sec is not None else SET_EXPERIMENT_LIMITS["B"]["im_time"]
    return round(im_sec * r, 1)


def ensure_instances() -> list[str]:
    names: list[str] = []
    for idx, seed in enumerate(SEEDS, 1):
        name = f"set_b_hard_{idx:02d}"
        path = INSTANCES_DIR / f"{name}.json"
        if not path.exists():
            inst = generate_instance(SET_B, idx, seed=seed)
            inst.name = name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(inst.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        names.append(name)
    return names


def run_case(
    name: str,
    *,
    bp_time_limit: float,
    bp_max_nodes: int,
    branch_strategy: str,
    instances_filter: list[str] | None = None,
    skip_cg_im: bool = False,
    root_im: bool = DEFAULT_BP_ROOT_IM,
    final_im: bool = DEFAULT_BP_FINAL_IM,
    lazy_y1_enabled: bool | None = None,
    y1_sibling_mode: str | None = None,
) -> dict | None:
    if instances_filter and name not in instances_filter:
        return None

    data = preprocess(load_instance(INSTANCES_DIR / f"{name}.json"))
    print(
        f"\n=== {name} strategy={branch_strategy} BP_limit={bp_time_limit}s ===",
        flush=True,
    )

    cg_lb = None
    cg_sec = 0.0
    cg_converged = None
    if not skip_cg_im:
        t0 = time.perf_counter()
        cg = solve_column_generation(
            data,
            max_iterations=DEFAULT_CG_MAX_ITERATIONS,
            init_time_limit=CG_INIT,
            fast=True,
        )
        cg_sec = time.perf_counter() - t0
        cg_lb = cg.lower_bound
        cg_converged = cg.extra.get("converged")

    if y1_sibling_mode is None:
        if lazy_y1_enabled is False:
            y1_sibling_mode = "off"
        elif lazy_y1_enabled is True:
            y1_sibling_mode = "fix_y1"
    t1 = time.perf_counter()
    bp = BranchAndPriceSolver(
        data,
        max_nodes=bp_max_nodes,
        cg_max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        cg_init_time_limit=CG_INIT,
        im_time_limit=min(600.0, bp_time_limit * 0.2),
        mip_gap=DEFAULT_MIP_GAP,
        time_limit=bp_time_limit,
        cg_kwargs=BP_CG_KWARGS,
        use_node_im=False,
        root_im=root_im,
        final_im=final_im,
        branch_strategy=branch_strategy,
        y1_sibling_mode=y1_sibling_mode,
        lazy_y1_enabled=lazy_y1_enabled,
    )
    bp_result = bp.solve()
    bp_sec = time.perf_counter() - t1

    im_ub = None
    im_status = None
    im_sec = 0.0
    if not skip_cg_im:
        t2 = time.perf_counter()
        im = solve_im(data, time_limit=IM_LIMIT, mip_gap=DEFAULT_MIP_GAP)
        im_sec = time.perf_counter() - t2
        im_ub = im.objective
        im_status = im.status

    ref_ub = REF_IM_UB.get(name)
    bp_ub = bp_result.objective
    best_ub = min(v for v in (bp_ub, im_ub) if v is not None) if (bp_ub or im_ub) else None

    row = {
        "instance": name,
        "branch_strategy": branch_strategy,
        "cg_lb": round(cg_lb, 2) if cg_lb is not None else None,
        "cg_sec": round(cg_sec, 2),
        "cg_converged": cg_converged,
        "bp_lb_root": round(bp_result.root_lrmp, 2) if bp_result.root_lrmp is not None else None,
        "bp_lb_best": round(bp_result.best_node_lb, 2) if bp_result.best_node_lb is not None else None,
        "bp_ub": round(bp_ub, 2) if bp_ub is not None else None,
        "bp_sec": round(bp_sec, 2),
        "bp_status": bp_result.status,
        "bp_nodes": bp_result.nodes_explored,
        "bp_nodes_created": bp_result.nodes_created,
        "bp_pruned": bp_result.nodes_pruned,
        "bp_infeasible": bp_result.nodes_infeasible,
        "bp_cg_failed": bp_result.nodes_cg_failed,
        "bp_max_depth": bp_result.max_depth,
        "bp_y1_lazy_created": bp_result.y1_lazy_created,
        "bp_y1_exclude_created": bp_result.y1_exclude_created,
        "bp_lazy_popped": bp_result.lazy_popped,
        "bp_dfs_popped": bp_result.dfs_popped,
        "im_ub": round(im_ub, 2) if im_ub is not None else None,
        "im_status": im_status,
        "im_sec": round(im_sec, 2),
        "ref_im_ub": ref_ub,
        "best_ub": round(best_ub, 2) if best_ub is not None else None,
        "bp_gap_pct": round((best_ub - bp_result.best_node_lb) / best_ub * 100, 4)
        if best_ub and bp_result.best_node_lb
        else None,
        "bp_proven_optimal": bp_result.status == "optimal",
    }
    print(
        f"  CG LB={row['cg_lb']} | BP LB_best={row['bp_lb_best']} status={row['bp_status']} "
        f"nodes={row['bp_nodes']} dfs={row['bp_dfs_popped']} lazy={row['bp_lazy_popped']} "
        f"cg_fail={row['bp_cg_failed']} infeas={row['bp_infeasible']} ({row['bp_sec']}s)",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Set-B hard BP strategy benchmark")
    parser.add_argument("--instance", action="append", dest="instances")
    parser.add_argument("--bp-time", type=float, default=None,
                        help="B&P 时限（秒）；默认 auto=45%%×IM 耗时")
    parser.add_argument("--bp-nodes", type=int, default=5000)
    parser.add_argument("--no-root-im", action="store_true", help="LB 冲刺：不占 B&P 时限")
    parser.add_argument("--no-lazy-y1", action="store_true", help="y1_sibling_mode=off")
    parser.add_argument(
        "--y1-sibling-mode",
        choices=["off", "fix_y1", "exclude_conflict"],
        default=None,
    )
    parser.add_argument(
        "--bp-im-ratio",
        type=float,
        default=DEFAULT_BP_IM_TIME_RATIO,
        help="auto 时限 = ratio × REF_IM_SEC（须 < 0.5）",
    )
    parser.add_argument(
        "--strategy",
        choices=list(BP_STRATEGIES) + ["all"],
        default="all",
        help="branch strategy or 'all' for A/B/C on filtered instances",
    )
    args = parser.parse_args()

    strategies = list(BP_STRATEGIES) if args.strategy == "all" else [args.strategy]
    instance_names = ensure_instances()
    if args.instances:
        instance_names = [n for n in instance_names if n in args.instances]

    def resolve_bp_time(name: str) -> float:
        if args.bp_time is not None:
            return args.bp_time
        return bp_time_cap_for_instance(name, ratio=args.bp_im_ratio)

    print(
        f"Set-B hard BP | strategies={strategies} BP=auto(ratio={args.bp_im_ratio}) "
        f"nodes={args.bp_nodes} root_im={not args.no_root_im}",
        flush=True,
    )

    rows: list[dict] = []
    for strat in strategies:
        skip_cg_im = strat != strategies[0]
        for name in instance_names:
            bp_limit = resolve_bp_time(name)
            y1_mode = args.y1_sibling_mode
            if y1_mode is None:
                y1_mode = "off" if args.no_lazy_y1 else None
            row = run_case(
                name,
                bp_time_limit=bp_limit,
                bp_max_nodes=args.bp_nodes,
                branch_strategy=strat,
                instances_filter=None,
                skip_cg_im=skip_cg_im,
                root_im=not args.no_root_im,
                y1_sibling_mode=y1_mode,
                lazy_y1_enabled=True if y1_mode == "fix_y1" else (False if y1_mode == "off" else None),
            )
            if row is not None:
                rows.append(row)

    payload = {
        "generation": SET_B_GENERATION,
        "bp_time_limit_sec": args.bp_time,
        "bp_im_ratio": args.bp_im_ratio,
        "bp_max_nodes": args.bp_nodes,
        "strategies": strategies,
        "pricing": "EBRP+fixed_y",
        "cg_fast": BP_CG_KWARGS,
        "instances": rows,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
