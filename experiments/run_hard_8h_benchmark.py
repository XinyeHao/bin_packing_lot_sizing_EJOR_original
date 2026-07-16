"""hard_01–03：LRMP 有效不等式 + B&P 8h 基准（总时限可配）。"""

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
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_MIP_GAP,
    SET_B_CG_CUTS,
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
from model_and_algo.column_generation import ColumnGenerationSolver
from preprocessor.preprocess import load_instance, preprocess

CG_INIT = SET_EXPERIMENT_LIMITS["B"]["cg_init"]
OUTPUT = RESULT_DIR / "set_b_hard_8h_benchmark.json"

METHODS = (
    {"label": "cg_base", "kind": "cg", "cg_kwargs": SET_B_CG_FAST},
    {"label": "cg_cuts", "kind": "cg", "cg_kwargs": SET_B_CG_CUTS},
    {
        "label": "bp_y0",
        "kind": "bp",
        "branch_strategy": "new_policy",
        "y1_sibling_mode": "off",
        "cg_kwargs": SET_B_CG_FAST,
    },
    {
        "label": "bp_excl",
        "kind": "bp",
        "branch_strategy": "new_policy",
        "y1_sibling_mode": "exclude_conflict",
        "cg_kwargs": SET_B_CG_FAST,
    },
    {
        "label": "bp_excl_cuts",
        "kind": "bp",
        "branch_strategy": "new_policy",
        "y1_sibling_mode": "exclude_conflict",
        "cg_kwargs": SET_B_CG_CUTS,
    },
)


def run_cg(data, *, cg_kwargs: dict, time_limit: float) -> dict:
    t0 = time.perf_counter()
    cg = ColumnGenerationSolver(
        data,
        max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        init_time_limit=CG_INIT,
        time_limit=time_limit,
        **cg_kwargs,
    )
    lb, _, _ = cg.run()
    sec = time.perf_counter() - t0
    return {
        "lb": round(lb, 2) if lb != float("-inf") else None,
        "sec": round(sec, 2),
        "converged": cg.converged,
        "stop_reason": cg.stop_reason,
        "same_mt_cuts": len(cg.same_mt_cuts),
        "wip_proc_lb_rows": len(cg.wip_proc_lb_rows),
        "setup_pair_cuts": len(cg.setup_pair_cuts),
    }


def run_bp(
    data,
    name: str,
    *,
    branch_strategy: str,
    y1_sibling_mode: str,
    cg_kwargs: dict,
    bp_time_limit: float,
    bp_max_nodes: int,
) -> dict:
    t0 = time.perf_counter()
    bp = BranchAndPriceSolver(
        data,
        max_nodes=bp_max_nodes,
        cg_max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        cg_init_time_limit=CG_INIT,
        mip_gap=DEFAULT_MIP_GAP,
        time_limit=bp_time_limit,
        cg_kwargs=cg_kwargs,
        use_node_im=False,
        root_im=False,
        final_im=False,
        branch_strategy=branch_strategy,
        y1_sibling_mode=y1_sibling_mode,
    )
    r = bp.solve()
    sec = time.perf_counter() - t0
    ref_ub = REF_IM_UB.get(name)
    return {
        "lb_root": round(r.root_lrmp, 2) if r.root_lrmp else None,
        "lb_best": round(r.best_node_lb, 2) if r.best_node_lb else None,
        "sec": round(sec, 2),
        "status": r.status,
        "nodes": r.nodes_explored,
        "cg_fail": r.nodes_cg_failed,
        "y1_exclude": r.y1_exclude_created,
        "gap_pct": round((ref_ub - r.best_node_lb) / ref_ub * 100, 4)
        if ref_ub and r.best_node_lb
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-hours", type=float, default=8.0)
    parser.add_argument("--cg-sec", type=float, default=180.0)
    parser.add_argument("--bp-im-ratio", type=float, default=0.45)
    parser.add_argument("--bp-nodes", type=int, default=400)
    parser.add_argument("--instance", action="append", dest="instances")
    args = parser.parse_args()

    names = ensure_instances()
    if args.instances:
        names = [n for n in names if n in args.instances]

    total_budget = args.total_hours * 3600.0
    per_instance = total_budget / max(len(names), 1)
    cg_budget = min(args.cg_sec, per_instance * 0.15)
    bp_budget = min(per_instance * 0.35, max(300.0, REF_IM_SEC.get(names[0], 300) * args.bp_im_ratio * 1.5))

    print(
        f"8h benchmark | instances={names} total={args.total_hours}h "
        f"cg={cg_budget:.0f}s bp≈{bp_budget:.0f}s methods={len(METHODS)}",
        flush=True,
    )

    rows: list[dict] = []
    run_start = time.perf_counter()
    for name in names:
        if time.perf_counter() - run_start >= total_budget * 0.98:
            print("Total time budget nearly exhausted, stopping.", flush=True)
            break
        data = preprocess(load_instance(INSTANCES_DIR / f"{name}.json"))
        bp_limit = min(bp_budget, bp_time_cap_for_instance(name, ratio=args.bp_im_ratio) * 2)
        for meth in METHODS:
            if time.perf_counter() - run_start >= total_budget * 0.98:
                break
            print(f"\n=== {name} {meth['label']} ===", flush=True)
            row = {"instance": name, "method": meth["label"]}
            cg_kw = {k: v for k, v in meth["cg_kwargs"].items() if k != "time_limit"}
            if meth["kind"] == "cg":
                stats = run_cg(data, cg_kwargs=cg_kw, time_limit=cg_budget)
                row.update(stats)
                print(
                    f"  CG LB={stats['lb']} cuts=({stats['same_mt_cuts']},"
                    f"{stats['wip_proc_lb_rows']},{stats['setup_pair_cuts']}) "
                    f"conv={stats['converged']} ({stats['sec']}s)",
                    flush=True,
                )
            else:
                stats = run_bp(
                    data,
                    name,
                    branch_strategy=meth["branch_strategy"],
                    y1_sibling_mode=meth["y1_sibling_mode"],
                    cg_kwargs=cg_kw,
                    bp_time_limit=bp_limit,
                    bp_max_nodes=args.bp_nodes,
                )
                row.update(stats)
                print(
                    f"  BP LB_best={stats['lb_best']} gap={stats['gap_pct']}% "
                    f"nodes={stats['nodes']} cg_fail={stats['cg_fail']} "
                    f"exclude={stats['y1_exclude']} ({stats['sec']}s)",
                    flush=True,
                )
            rows.append(row)

    payload = {
        "generation": SET_B_GENERATION,
        "total_hours": args.total_hours,
        "cg_sec_budget": cg_budget,
        "bp_sec_budget": bp_limit if names else None,
        "ref_im_ub": REF_IM_UB,
        "runs": rows,
        "elapsed_sec": round(time.perf_counter() - run_start, 2),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT} elapsed={payload['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
