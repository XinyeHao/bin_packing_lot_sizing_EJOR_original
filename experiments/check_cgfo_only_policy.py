"""Check CGFO-only mode and root-only IRMP initialization in adaptive mode."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_and_algo import branch_and_price as bp


def exercise(*, root=False, stalled=True, repair_improves=False, duplicate=False,
              new_columns=10, strategy="cgfo", incumbent=100.0, expired=False):
    solver = bp.BranchAndPriceSolver(SimpleNamespace(machines=["m0"]),
                                     upper_bound_strategy=strategy)
    solver._solve_started = time.perf_counter() - 1000
    solver._time_irmp = solver._time_cgfo = 0.0
    solver._current_node_count = 1 if root else 200
    solver._current_node_id = 0 if root else 1
    solver._last_primal_improvement_node = 0 if stalled else 199
    solver._nodes_at_last_irmp = solver._nodes_at_last_cgfo = 0
    solver._primal_columns_added = solver._columns_at_last_irmp = 0
    solver._irmp_failures = solver._cgfo_failures = 0
    solver._primal_stats = {"cgfo_calls": 0, "irmp_backup_calls": 0}
    calls = []

    def irmp(cg, value, budget):
        assert root, "IRMP must never run at a tree node"
        calls.append("irmp_root")
        return 95.0

    def repair(cg, value, budget, *, root):
        calls.append("cgfo")
        if not duplicate:
            solver._primal_stats["cgfo_calls"] += 1
        return 90.0 if repair_improves else value

    solver._try_root_irmp = irmp
    solver._try_cgfo_repair = repair
    cg = SimpleNamespace(_model=object(), converged=True,
                         iteration_time_detail=[{"num_new_columns": new_columns}])
    node = SimpleNamespace(node_id=0 if root else 1, lower_bound=0.0)
    deadline = time.perf_counter() + (-1 if expired else 100)
    with patch.object(bp, "_q_integer", return_value=False), patch.object(bp, "_r_integer", return_value=False):
        result = solver._run_upper_bound_heuristics(cg, incumbent, node, deadline)
    return calls, result, solver


def main():
    assert not hasattr(bp.BranchAndPriceSolver, "_try_irmp")
    calls, _, _ = exercise(root=True)
    assert calls == ["cgfo"]
    assert exercise(stalled=False)[0] == []
    assert exercise(repair_improves=True)[0] == ["cgfo"]
    assert exercise()[0] == ["cgfo"]
    assert exercise(duplicate=True)[0] == ["cgfo"]
    assert exercise(stalled=False, incumbent=None)[0] == []
    assert exercise(expired=True, incumbent=None)[0] == []
    assert exercise(root=True, strategy="adaptive")[0] == ["irmp_root", "cgfo"]
    assert exercise(strategy="adaptive")[0] == ["cgfo"]
    assert exercise(stalled=False, strategy="adaptive", incumbent=None)[0] == []
    guard = bp.BranchAndPriceSolver(SimpleNamespace(machines=["m0"]))
    guard._current_node_id = 1
    try:
        guard._try_root_irmp(None, None, 5.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("IRMP accepted a non-root call")
    try:
        exercise(strategy="irmp")
    except ValueError:
        pass
    else:
        raise AssertionError("Removed IRMP strategy was accepted")
    print(json.dumps({"passed": True, "checks": 13,
                      "covered": ["IRMP helper removed from BP", "direct root CGFO",
                                  "repair requires stagnation", "successful local repair",
                                  "failed repair never falls back to IRMP", "duplicate repair skips IRMP",
                                  "no-incumbent never invokes IRMP", "deadline respected",
                                  "IRMP-only strategy rejected", "root IRMP precedes CGFO",
                                  "adaptive tree repair never calls IRMP", "no-incumbent tree skips IRMP",
                                  "non-root IRMP guard"]}))


if __name__ == "__main__":
    main()
