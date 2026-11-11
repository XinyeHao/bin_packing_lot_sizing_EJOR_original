"""Medium seed 31101: odd SRI ranks 3 / 5 / 7.

1) Root-only: measure LRMP lift for (3), (3,5), (3,5,7).
2) Full B&P: SRI-only with cardinalities (3,5,7), compare to pure BP.
"""
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
from model_and_algo.column_generation import ColumnGenerationSolver
from preprocessor.preprocess import preprocess

SEED = 31101
OUT = ROOT / "result" / "sri_31101.json"
SRI_MAX = 12
CAND_MAX = 16


def gap(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return round(100.0 * max(0.0, (ub - lb) / ub), 4)


def _data():
    spec = scale_spec("medium")
    cfg.apply_bench_generation()
    return preprocess(generate_instance(make_set_config(spec), SEED % 1000, SEED)), spec


def _clean(*xs):
    for x in xs:
        try:
            x.dispose()
        except Exception:
            pass
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass


def root_lift(cards: tuple[int, ...]) -> dict:
    data, _ = _data()
    print(f"\n[root] cards={cards}", flush=True)
    t0 = time.perf_counter()
    # baseline without cuts
    base = ColumnGenerationSolver(
        data,
        max_iterations=120,
        pricing_mode="ebrp",
        ebrp_force=True,
        use_gurobi_init=True,
        pricing_inprocess=True,
        skip_cross_check=True,
        lrmp_cuts="none",
        ebrp_time_limit=20.0,
        verbose=False,
    )
    base.run()
    lb0 = float(base.lrmp_obj) if base.converged else None
    _clean(base)

    cuts = {
        "deadline_cover": False,
        "sri": True,
        "sri_max_cuts": SRI_MAX,
        "sri_violation_tol": 1e-6,
        "sri_cardinalities": cards,
        "sri_cand_max": CAND_MAX,
        "tight_deadline": False,
    }
    cg = ColumnGenerationSolver(
        data,
        max_iterations=120,
        pricing_mode="ebrp",
        ebrp_force=True,
        use_gurobi_init=True,
        pricing_inprocess=True,
        skip_cross_check=True,
        lrmp_cuts=cuts,
        ebrp_time_limit=20.0,
        verbose=False,
    )
    cg.run()
    # one extra separation + reoptimize pass if cuts were added mid-run
    n = len(cg.sri_cuts)
    by_r: dict[int, int] = {}
    for c in cg.sri_cuts:
        by_r[len(c.items)] = by_r.get(len(c.items), 0) + 1
    lb1 = float(cg.lrmp_obj) if cg.converged else None
    row = {
        "cardinalities": list(cards),
        "converged": bool(cg.converged),
        "lb_no_cut": lb0,
        "lb_with_sri": lb1,
        "delta": None if lb0 is None or lb1 is None else round(lb1 - lb0, 4),
        "n_sri": n,
        "by_rank": by_r,
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "cut_names": [c.name for c in cg.sri_cuts],
    }
    print(
        f"  LB {lb0} -> {lb1}  delta={row['delta']}  n={n} by_rank={by_r}  "
        f"{row['runtime_sec']}s",
        flush=True,
    )
    _clean(cg, data)
    return row


def run_bp_sri(cards: tuple[int, ...]) -> dict:
    data, spec = _data()
    cuts = {
        "deadline_cover": False,
        "sri": True,
        "sri_max_cuts": SRI_MAX,
        "sri_violation_tol": 1e-6,
        "sri_cardinalities": cards,
        "sri_cand_max": CAND_MAX,
        "tight_deadline": False,
    }
    print(f"\n======== BPC-SRI cards={cards} seed={SEED} limit={spec.time_limit_sec:.0f}s ========", flush=True)
    t0 = time.perf_counter()
    bp = BranchAndPriceSolver(
        data,
        max_nodes=8000,
        time_limit=spec.time_limit_sec,
        mip_gap=0.01,
        branch_strategy="frac_sb",
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
    cg = getattr(res, "cg_solver", None)
    if cg is not None:
        for c in getattr(cg, "sri_cuts", []) or []:
            n_sri += 1
            by_r[len(c.items)] = by_r.get(len(c.items), 0) + 1
    row = {
        "tag": f"sri_{'_'.join(map(str, cards))}",
        "cardinalities": list(cards),
        "status": res.status,
        "ub": res.objective,
        "lb": res.lower_bound,
        "gap_pct": gap(res.objective, res.lower_bound),
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "nodes": res.nodes_explored,
        "nodes_cg_failed": res.nodes_cg_failed,
        "root_lb": res.root_valid_lb,
        "root_lrmp": res.root_lrmp,
        "n_sri": n_sri,
        "by_rank": by_r,
    }
    print(
        f"  -> {row['tag']}: status={row['status']} UB={row['ub']} LB={row['lb']} "
        f"gap={row['gap_pct']}% nodes={row['nodes']} sri={n_sri}{by_r} "
        f"sec={row['runtime_sec']}",
        flush=True,
    )
    _clean(bp, res, data)
    return row


def main():
    bp_ref = None
    bp_path = ROOT / "result" / "bp_31101.json"
    if bp_path.exists():
        bp_ref = json.loads(bp_path.read_text(encoding="utf-8"))

    root_rows = [
        root_lift((3,)),
        root_lift((3, 5)),
        root_lift((3, 5, 7)),
    ]
    tree = run_bp_sri((3, 5, 7))

    out = {
        "seed": SEED,
        "scale": "medium",
        "note": "Odd SRI ranks 3/5/7 on 31101 (no deadline). Root lift + full B&P for (3,5,7).",
        "sri_max_cuts": SRI_MAX,
        "sri_cand_max": CAND_MAX,
        "bp_reference": {
            "ub": (bp_ref or {}).get("ub"),
            "lb": (bp_ref or {}).get("lb"),
            "gap_pct": (bp_ref or {}).get("gap_pct"),
            "runtime_sec": (bp_ref or {}).get("runtime_sec"),
            "nodes": (bp_ref or {}).get("nodes"),
            "root_lb": (bp_ref or {}).get("root_lb"),
        },
        "root_lifts": root_rows,
        "tree_sri_357": tree,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n======== summary ========", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
