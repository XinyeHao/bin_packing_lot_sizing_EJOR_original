"""Diagnostic practical BP runs for selected medium seeds."""
from __future__ import annotations

import argparse, gc, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); os.environ["GRB_CS_QUIET"] = "1"
import gurobipy as gp
import configuration as cfg
from experiments.paper_experiment_config import MODEL_VERSION, PROCESS_ASSUMPTION, make_set_config, scale_spec
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess


def gap(ub, lb):
    if ub is None or lb is None or abs(float(ub)) <= 1e-9: return None
    return 100.0 * max(0.0, (float(ub) - float(lb)) / abs(float(ub)))


def solve(seed: int):
    spec = scale_spec("medium"); raw = data = solver = result = None; t0 = time.perf_counter()
    try:
        cfg.apply_bench_generation()
        raw = generate_instance(make_set_config(spec), seed % 1000, seed); data = preprocess(raw)
        solver = BranchAndPriceSolver(
            data, max_nodes=8000, time_limit=1800.0, mip_gap=0.01,
            branch_strategy="frac_sb", use_node_im=False, root_im=False, final_im=False,
            strict_cg_certification=False,
            cg_kwargs={"pricing_mode":"ebrp", "ebrp_force":True, "use_gurobi_init":True,
                       "pricing_inprocess":True, "skip_cross_check":True, "lrmp_cuts":"none",
                       "max_iterations":120, "ebrp_time_limit":20.0, "verbose":False},
        )
        result = solver.solve(); ub, lb = result.objective, result.lower_bound
        row = {"method":"bp_practical", "scale":"medium", "seed":seed, "T":spec.periods, "I":spec.items,
               "time_limit_sec":1800.0, "strict_cg_certification":False, "pricing_mode":"ebrp",
               "ebrp_time_limit_sec":20.0, "cuts":"none", "status":result.status,
               "ub":ub, "lb":lb, "gap_pct":gap(ub,lb), "runtime_sec":round(time.perf_counter()-t0,3),
               "nodes":result.nodes_explored, "nodes_cg_failed":result.nodes_cg_failed,
               "model_version":MODEL_VERSION, "process_assumption":PROCESS_ASSUMPTION}
    except Exception as exc:
        row={"method":"bp_practical", "scale":"medium", "seed":seed, "status":"error",
             "error":f"{type(exc).__name__}: {exc}", "runtime_sec":round(time.perf_counter()-t0,3)}
    finally:
        for obj in (result, solver, data, raw):
            try: obj.dispose()
            except Exception: pass
        gc.collect()
        try: gp.disposeDefaultEnv()
        except Exception: pass
    out=ROOT/"result"/"exp4"/f"test_bp_practical_{seed}.json"; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,required=True); args=ap.parse_args()
    for s in args.seeds: solve(s)
