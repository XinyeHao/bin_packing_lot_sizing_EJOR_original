"""Run the synthetic case with the teacher's Ryan–Foster BP solver."""
import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "bin_packing_lot_sizing_EJOR_original-main"


def load_reporter():
    spec = importlib.util.spec_from_file_location(
        "case1_report", CASE / "experiments/run_case1_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--instance", type=Path, default=CASE / "instances/case1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "case1_rf")
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--pricing-limit", type=float, default=60)
    args = parser.parse_args()
    if min(args.time_limit, args.pricing_limit) <= 0:
        parser.error("Time limits must be positive")
    sys.path.insert(0, str(args.solver_dir.resolve()))
    import gurobipy as gp
    from common.data_models import InstanceData
    from model_and_algo import branch_and_price as bp
    from preprocessor.preprocess import preprocess

    reporter = load_reporter()
    raw = reporter.read_json(args.instance)
    data = preprocess(InstanceData.from_dict(raw))
    history = []
    best = {"ub": None, "batches": None}
    started = time.perf_counter()

    def record(ub=None, lb=None):
        previous = history[-1] if history else {"ub": None, "lb": None}
        ub = previous["ub"] if ub is None else ub
        if lb is None or not math.isfinite(lb):
            lb = previous["lb"]
        elif previous["lb"] is not None:
            lb = max(previous["lb"], lb)
        if ub is not None and lb is not None and lb > ub + 1e-5:
            raise ValueError("Global bound exceeds the feasible objective")
        if (ub, lb) != (previous["ub"], previous["lb"]):
            history.append(dict(elapsed_sec=time.perf_counter() - started,
                                ub=ub, lb=lb, lb_certified=lb is not None))

    def save_schedule(objective, starts, spans=None):
        groups = defaultdict(list)
        for machine, item, period in starts:
            groups[machine, raw["item_config"][item], period].append(item)
        batches = []
        for (machine, config, period), items in sorted(groups.items()):
            duration = max(data.lt_it[data.cured_index[i]][period - 1] for i in items)
            if spans is not None and spans[machine][config][period] != duration:
                raise ValueError("Unexpected IM batch span")
            batches.append(dict(machine=machine, config=config, start=period,
                                end=period + duration, items=sorted(items)))
        best.update(ub=objective, batches=batches)
        record(ub=objective)

    original_im = bp.solve_im

    def capture_im(*a, **kw):
        solution = original_im(*a, **kw)
        if solution.objective is not None and (
            best["ub"] is None or solution.objective < best["ub"] - bp.INT_TOL
        ):
            starts = [(m, i, t) for m, items in solution.packing.items()
                      for i, periods in items.items() for t, x in periods.items() if x > 0.5]
            save_schedule(solution.objective, starts, solution.extra["span"])
        return solution

    class CaseSolver(bp.BranchAndPriceSolver):
        def _note_gap_stop(self, incumbent, open_heap, lazy_heap, dfs_stack,
                           nodes_explored, current_lb=None):
            if not self._strict_unresolved_cg:
                record(lb=self._tree_dual_bound(open_heap, lazy_heap, dfs_stack, current_lb))
            return super()._note_gap_stop(
                incumbent, open_heap, lazy_heap, dfs_stack, nodes_explored, current_lb
            )

    # Capture integer LRMP incumbents at the point they are accepted, while
    # the model still contains that node's integer solution.
    import inspect
    import textwrap
    source = textwrap.dedent(inspect.getsource(bp.BranchAndPriceSolver.solve))
    marker = "incumbent = int_cost"
    if source.count(marker) != 1:
        raise RuntimeError("The solver's integer LRMP update has changed")

    def capture_lrmp(cg, objective):
        starts = [(m, i, t) for m in data.machines
                  for col, q in zip(cg.columns[m], cg._artifacts["Q"][m])
                  if q.X > 0.5 for (i, t), x in col.x.items() if x > 0.5]
        save_schedule(objective, starts)

    namespace = dict(vars(bp), capture_lrmp=capture_lrmp)
    source = source.replace(marker, marker + "\n                    capture_lrmp(cg, incumbent)")
    # Match the indentation of the actual update in the dedented solver.
    source = source.replace("\n                    capture_lrmp", "\n" +
                            next(l[:len(l)-len(l.lstrip())] for l in source.splitlines()
                                 if l.strip() == marker) + "capture_lrmp")
    exec(compile(source, bp.__file__, "exec"), namespace)
    namespace["solve_im"] = capture_im
    CaseSolver.solve = namespace["solve"]

    solver = CaseSolver(
        data, max_nodes=8000, time_limit=args.time_limit, mip_gap=0.01,
        branch_strategy="ryan_foster", strict_cg_certification=True,
        use_node_im=False, root_im=False, final_im=False,
        cg_kwargs=dict(pricing_mode="ebrp", ebrp_force=True, use_gurobi_init=True,
                       pricing_inprocess=True, skip_cross_check=True,
                       lrmp_cuts="none", max_iterations=120,
                       ebrp_time_limit=args.pricing_limit),
    )
    bp.solve_im = capture_im
    try:
        result = solver.solve()
    finally:
        bp.solve_im = original_im
    if result.objective is not None and (
        best["ub"] is None or abs(best["ub"] - result.objective) > 1e-5
    ):
        raise RuntimeError("The final BP incumbent has no matching captured schedule")
    lb = result.lower_bound if result.nodes_cg_failed == 0 else result.root_valid_lb
    record(result.objective, lb)
    lb = history[-1]["lb"] if history else None
    output = dict(branch_strategy="ryan_foster", status=result.status,
                  ub=result.objective, lb=lb, runtime_sec=result.runtime_sec,
                  nodes=result.nodes_explored, nodes_cg_failed=result.nodes_cg_failed,
                  root_cg_stop=result.root_cg_stop, bound_history=history,
                  batches=best["batches"], pricing_limit=args.pricing_limit,
                  time_limit=args.time_limit, gurobi_version=gp.gurobi.version())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rf_result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    if best["batches"] is not None:
        reporter.validate_result(raw, output)
        reporter.analyse(raw, output, args.out)
    print(f"Results saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
