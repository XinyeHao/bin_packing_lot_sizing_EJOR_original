"""Branch-and-price 测试：demo 算例 + 1 个 Set-B 新算例。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import INSTANCES_DIR, RESULT_DIR
from configuration import (
    DEFAULT_BP_DEMO_MAX_NODES,
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_MIP_GAP,
    SET_B,
    SET_EXPERIMENT_LIMITS,
)
from instance_generator.generator import generate_demo_instance, generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess

OUTPUT = RESULT_DIR / "bp_test_results.json"
SET_B_MAX_NODES = 40
SET_B_CG_ITERS = 300


def ensure_set_b_new01() -> str:
    name = "set_b_new_01"
    path = INSTANCES_DIR / f"{name}.json"
    if not path.exists():
        inst = generate_instance(SET_B, 1, seed=9301)
        inst.name = name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(inst.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return name


def run_case(
    name: str,
    *,
    max_nodes: int,
    cg_iters: int,
    im_limit: float,
    cg_init: float,
) -> dict:
    data = preprocess(load_instance(INSTANCES_DIR / f"{name}.json"))
    print(f"\n=== {name} (max_nodes={max_nodes}) ===", flush=True)

    t0 = time.perf_counter()
    cg = ColumnGenerationSolver(
        data,
        max_iterations=cg_iters,
        init_time_limit=cg_init,
    )
    cg.run()
    cg_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    bp = BranchAndPriceSolver(
        data,
        max_nodes=max_nodes,
        cg_max_iterations=cg_iters,
        cg_init_time_limit=cg_init,
        im_time_limit=im_limit,
        mip_gap=DEFAULT_MIP_GAP,
    )
    bp_result = bp.solve()
    bp_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    im = solve_im(data, time_limit=im_limit, mip_gap=DEFAULT_MIP_GAP, method="im_gurobi")
    im_sec = time.perf_counter() - t2

    lrmp = cg.lrmp_obj
    im_obj = im.objective
    bp_obj = bp_result.objective
    row = {
        "instance": name,
        "lrmp": round(lrmp, 2) if lrmp else None,
        "cg_sec": round(cg_sec, 2),
        "cg_iters": len(cg.iteration_times),
        "bp_obj": round(bp_obj, 2) if bp_obj is not None else None,
        "bp_sec": round(bp_sec, 2),
        "bp_nodes": bp_result.nodes_explored,
        "bp_pruned": bp_result.nodes_pruned,
        "bp_max_depth": bp_result.max_depth,
        "bp_status": bp_result.status,
        "im_obj": round(im_obj, 2) if im_obj else None,
        "im_sec": round(im_sec, 2),
        "cg_gap_pct": round((im_obj - lrmp) / im_obj * 100, 2) if im_obj and lrmp else None,
        "bp_gap_pct": round((im_obj - bp_obj) / im_obj * 100, 2) if im_obj and bp_obj else None,
    }
    print(
        f"  CG: LRMP={row['lrmp']} ({row['cg_sec']}s) | "
        f"BP: obj={row['bp_obj']} nodes={row['bp_nodes']} depth={row['bp_max_depth']} ({row['bp_sec']}s) | "
        f"IM: {row['im_obj']} ({row['im_sec']}s)",
        flush=True,
    )
    return row


def main() -> None:
    generate_demo_instance()
    ensure_set_b_new01()
    cg_init_b = SET_EXPERIMENT_LIMITS["B"]["cg_init"]

    rows = [
        run_case(
            "demo_small",
            max_nodes=DEFAULT_BP_DEMO_MAX_NODES,
            cg_iters=DEFAULT_CG_MAX_ITERATIONS,
            im_limit=300.0,
            cg_init=90.0,
        ),
        run_case(
            "set_b_new_01",
            max_nodes=SET_B_MAX_NODES,
            cg_iters=SET_B_CG_ITERS,
            im_limit=600.0,
            cg_init=cg_init_b,
        ),
    ]

    out = {"cases": rows}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
