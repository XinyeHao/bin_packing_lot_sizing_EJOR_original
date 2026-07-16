"""分析 set_b_hard_03 的 CG 耗时结构。"""

from __future__ import annotations

import statistics as stats
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from checkers.utils import load_processed_instance
from configuration import DEFAULT_CG_MAX_ITERATIONS, SET_B_CG_FAST, SET_EXPERIMENT_LIMITS
from model_and_algo.column_generation import ColumnGenerationSolver


def profile(name: str, *, fast: bool = False) -> None:
    data = load_processed_instance(name)
    kwargs: dict = {"init_time_limit": SET_EXPERIMENT_LIMITS["B"]["cg_init"]}
    if fast:
        kwargs.update(SET_B_CG_FAST)
    cg = ColumnGenerationSolver(data, max_iterations=DEFAULT_CG_MAX_ITERATIONS, **kwargs)
    t0 = time.perf_counter()
    lb, rcs, cols = cg.run()
    total = time.perf_counter() - t0
    details = cg.iteration_time_detail
    n = len(details)
    cols_total = sum(len(v) for v in cols.values())
    master = sum(d["master"] for d in details)
    pricing = sum(d["pricing"] for d in details)
    addcol = sum(d["add_columns"] for d in details)
    newcols = sum(int(d["num_new_columns"]) for d in details)

    print(f"\n=== {name} ===")
    print(f"converged={cg.converged} stop={cg.stop_reason} iters={n} total_sec={total:.2f} lrmp={lb:.2f}")
    print(f"columns_total={cols_total} per_machine={ {m: len(v) for m, v in cols.items()} }")
    print(
        f"time: master={master:.1f}s ({100*master/total:.0f}%) "
        f"pricing={pricing:.1f}s ({100*pricing/total:.0f}%) "
        f"add_col={addcol:.1f}s ({100*addcol/total:.0f}%)"
    )
    print(f"new_columns_added={newcols} avg_new/iter={newcols/n:.2f}")
    print(f"avg_iter={stats.mean(d['total'] for d in details):.3f}s")
    print(f"  first10 total={stats.mean(d['total'] for d in details[:10]):.3f}s "
          f"pricing={stats.mean(d['pricing'] for d in details[:10]):.3f}s "
          f"master={stats.mean(d['master'] for d in details[:10]):.3f}s")
    print(f"  last10  total={stats.mean(d['total'] for d in details[-10:]):.3f}s "
          f"pricing={stats.mean(d['pricing'] for d in details[-10:]):.3f}s "
          f"master={stats.mean(d['master'] for d in details[-10:]):.3f}s")
    print(f"pricing_workers={int(details[0]['pricing_workers']) if details else 0}")
    print(f"subproblem_solves≈{n * len(data.machines)} (EBRP, no fixed_y)")


if __name__ == "__main__":
    for inst in ("set_b_hard_03",):
        print("--- default (full CG) ---")
        profile(inst, fast=False)
        print("--- SET_B_CG_FAST ---")
        profile(inst, fast=True)
