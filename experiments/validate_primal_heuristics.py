"""Check saved incumbents and local repair against the original compact model."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import gurobipy as gp
import configuration as cfg
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.branch_and_price import BranchAndPriceSolver
from model_and_algo.im_model import solve_im
from model_and_algo.primal_heuristics import (
    cgfo_restrictions, lp_setup_support, snapshot_integer_master, validate_primal,
)


def main():
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", 1)
    cfg.apply_bench_generation()
    data = preprocess(generate_instance(cfg.SetConfig("small_cg", 12, 2, 0, 8, 3), 1, 10101))
    options = dict(pricing_mode="ebrp", ebrp_force=True, use_gurobi_init=True,
                   pricing_inprocess=True, skip_cross_check=True, lrmp_cuts="none",
                   max_iterations=120, ebrp_time_limit=20.0, verbose=False)
    cg = ColumnGenerationSolver(data, **options)
    cg.run()
    lp_before = cg._model.ObjVal
    obj = cg.solve_rmp_ip(time_limit=5.0, threads=1)
    saved = cg.last_integer_solution
    diagnostic = cg.last_irmp_diagnostics
    assert diagnostic["status"] == "optimal" and diagnostic["mip_gap"] <= 0.01 + 1e-8
    assert abs(diagnostic["objective"] - obj) < 1e-4
    assert diagnostic["bound"] <= obj + 1e-4 and diagnostic["mip_runtime_sec"] >= 0
    assert saved is not None and abs(saved.objective - obj) < 1e-4
    validate_primal(data, saved)
    serialized = json.dumps(saved.to_dict(), sort_keys=True)
    assert cg._model.IsMIP == 0 and abs(cg._model.ObjVal - lp_before) < 1e-4
    assert all(q.VType == "C" for qs in cg._artifacts["Q"].values() for q in qs)
    assert all(abs(row.Pi) < float("inf") for row in cg._artifacts["flow_items"].values())
    cg._model.optimize()
    assert json.dumps(saved.to_dict(), sort_keys=True) == serialized
    assert cg.last_irmp_diagnostics is diagnostic

    support = lp_setup_support(cg)
    fy, fx, released = cgfo_restrictions(data, support, saved, release_count=2)
    assert len(released) == 2
    frozen = set(data.machines) - set(released)
    assert len(fx) == len(frozen) * len(data.cured_items) * len(data.periods)
    repair = solve_im(data, time_limit=5.0, fixed_y=fy, fixed_x=fx, mip_start=saved,
                      threads=1, budget_includes_build=True, primal_focus=True)
    validate_primal(data, repair)
    assert repair.objective <= saved.objective + 1e-4
    for machine in frozen:
        for item in data.cured_items:
            for period in data.periods:
                assert abs(repair.packing.get(machine, {}).get(item, {}).get(period, 0)
                           - saved.packing.get(machine, {}).get(item, {}).get(period, 0)) < 1e-5

    bad = copy.deepcopy(saved)
    item = data.cured_items[0]
    bad.extra["scrap"][item] = 2.0
    try:
        validate_primal(data, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid primal was accepted")

    im = solve_im(data, time_limit=20.0, mip_gap=0.0, threads=1)
    assert im.status == "optimal"
    solver = BranchAndPriceSolver(data, max_nodes=1000, time_limit=60.0, mip_gap=0.0,
                                  branch_strategy="ryan_foster", root_im=False,
                                  final_im=False, use_node_im=False,
                                  strict_cg_certification=True, cg_kwargs=options)
    with patch.object(solver, "_try_root_irmp", wraps=solver._try_root_irmp) as initialize:
        bp = solver.solve()
        assert initialize.call_count <= 1
    assert len(bp.primal_stats["irmp_attempts"]) <= 1
    assert all(x["node_id"] == 0 for x in bp.primal_stats["irmp_attempts"])
    assert bp.status == "optimal" and abs(bp.objective - im.objective) < 1e-4
    assert bp.best_solution is not None
    validate_primal(data, bp.best_solution)
    assert bp.lower_bound <= bp.objective + 1e-5
    print(json.dumps({"passed": True, "checks": 7, "im_optimum": im.objective,
                      "bp_optimum": bp.objective,
                      "covered": ["IRMP original feasibility and cost", "snapshot survives LP restore",
                                  "LP duals restored", "local repair feasible MIP start",
                                  "frozen-machine assignments preserved", "invalid incumbent rejected",
                                  "exact BP agrees with independent compact IM"]}, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
