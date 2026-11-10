"""One isolated solve for the paper experiment suite.

Called only by paper_experiment_runner.py.  One process per method-instance
pair prevents Gurobi/native memory from leaking into the following instance.
"""
from __future__ import annotations

import argparse, gc, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT); os.environ["GRB_CS_QUIET"] = "1"
import gurobipy as gp
import configuration as cfg
from experiments.paper_experiment_config import MODEL_VERSION, PROCESS_ASSUMPTION, make_set_config, scale_spec
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import preprocess


class CapacityInfeasibleError(ValueError):
    """Raised when a capacity-sensitivity variant cannot process an item."""


def _oversized_items(raw) -> list[str]:
    """Return items that fit on none of their eligible machines."""
    bad = []
    for item in raw.cured_item_ids:
        eligible = raw.item_machines.get(item) or raw.machine_ids
        if not any(raw.machine_capacities[m] + 1e-9 >= raw.tray_lengths[item] for m in eligible):
            bad.append(item)
    return bad

def gap(ub, lb):
    return None if ub is None or lb is None or ub <= 1e-9 else 100.0 * max(0.0, (ub-lb)/ub)

def clean(*xs):
    for x in xs:
        try: x.dispose()
        except Exception: pass
    gc.collect()
    try: gp.disposeDefaultEnv()
    except Exception: pass

def main():
    p=argparse.ArgumentParser()
    p.add_argument("method", choices=("im","bp","bpc","root_ebrp","root_gurobi"))
    p.add_argument("scale", choices=("small","medium","large")); p.add_argument("seed",type=int)
    p.add_argument("output"); p.add_argument("--limit",type=float,default=None); p.add_argument("--capacity-multiplier",type=float,default=1.0)
    p.add_argument("--ebrp-time-limit",type=float,default=20.0,
                   help="Per-node EBRP pricing time limit (seconds).")
    a=p.parse_args(); spec=scale_spec(a.scale); limit=a.limit or spec.time_limit_sec
    # Preserve the paper's original stopping convention: small instances are
    # exact-validation runs, while medium and large instances may stop once a
    # 1% relative MIP gap is reached.
    target_mip_gap = 0.0 if a.scale == "small" else 0.01
    raw=data=solver=result=None
    row={"method":a.method,"scale":a.scale,"seed":a.seed,"T":spec.periods,"I":spec.items,"time_limit_sec":limit,"target_mip_gap":target_mip_gap,"capacity_multiplier":a.capacity_multiplier,"process_assumption":PROCESS_ASSUMPTION,"model_version":MODEL_VERSION,"capacity_feasible":None,"oversized_items":[]}
    try:
        cfg.apply_bench_generation(); raw=generate_instance(make_set_config(spec),a.seed%1000,a.seed)
        if a.capacity_multiplier != 1.0:
            raw.machine_capacities={m: round(q*a.capacity_multiplier,4) for m,q in raw.machine_capacities.items()}
        oversized = _oversized_items(raw)
        row.update(capacity_feasible=not oversized, oversized_items=oversized)
        if oversized:
            raise CapacityInfeasibleError(
                f"{len(oversized)} item(s) fit on no eligible machine after capacity multiplier {a.capacity_multiplier}: {oversized}"
            )
        data=preprocess(raw); t=time.perf_counter()
        if a.method=="im":
            result=solve_im(data,time_limit=limit,mip_gap=target_mip_gap,verbose=False)
            row.update(status=result.status,ub=result.objective,lb=result.lower_bound,gap_pct=gap(result.objective,result.lower_bound),lower_bound_certified=result.lower_bound is not None,gurobi_nodes=(result.extra or {}).get("mip_node_count"))
        elif a.method in ("bp","bpc"):
            cuts="none" if a.method=="bp" else "standard"
            # The paper BPC uses forced EBRP pricing.  In the BP--BPC cut
            # ablation, every setting below is shared; ``lrmp_cuts`` is the
            # sole algorithmic difference (none for BP, standard for BPC).
            # This is also the same BPC configuration used for E2 and E4B.
            solver=BranchAndPriceSolver(data,max_nodes=8000,time_limit=limit,mip_gap=target_mip_gap,branch_strategy="frac_sb",use_node_im=False,root_im=False,final_im=False,strict_cg_certification=True,cg_kwargs={"pricing_mode":"ebrp","ebrp_force":True,"use_gurobi_init":True,"pricing_inprocess":True,"skip_cross_check":True,"lrmp_cuts":cuts,"max_iterations":120,"ebrp_time_limit":a.ebrp_time_limit,"verbose":False})
            result=solver.solve(); certified=(result.nodes_cg_failed==0 and result.lower_bound is not None)
            # E2/E4 are practical fixed-budget experiments.  Always retain
            # the raw tree gap for reporting, while separately flagging
            # whether the bound is fully CG-certified.
            raw_gap_pct = gap(result.objective, result.lower_bound)
            row.update(status=result.status,ub=result.objective,lb=result.lower_bound,
                       gap_pct=raw_gap_pct,
                       raw_gap_pct=raw_gap_pct,
                       certified_gap_pct=raw_gap_pct if certified else None,
                       lower_bound_certified=certified,nodes=result.nodes_explored,
                       nodes_created=result.nodes_created,nodes_pruned=result.nodes_pruned,
                       nodes_cg_failed=result.nodes_cg_failed,root_lb=result.root_valid_lb,
                       root_converged=result.root_cg_converged,root_stop=result.root_cg_stop,
                       cuts=cuts,
                       termination_rule="pending-node heap empty OR raw gap <= 1% OR time limit")
        else:
            mode="ebrp" if a.method=="root_ebrp" else "im_gurobi"
            solver=ColumnGenerationSolver(data,max_iterations=500,time_limit=limit,pricing_mode=mode,ebrp_force=(mode=="ebrp"),use_gurobi_init=True,pricing_inprocess=True,skip_cross_check=True,lrmp_cuts="none",im_pricing_time_limit=90.0,ebrp_time_limit=None,verbose=False)
            solver.run(); details=solver.iteration_time_detail
            row.update(status="converged" if solver.converged else (solver.stop_reason or "not_converged"),lb=solver.lrmp_obj if solver.converged else None,last_valid_lb=getattr(solver,"last_valid_lb",None),lower_bound_certified=solver.converged,iterations=len(solver.iteration_times),columns=sum(len(v) for v in solver.columns.values()),last_min_rc=solver.last_min_rc,pricing_time_sec=round(sum(d.get("pricing",0.0) for d in details),3),master_time_sec=round(sum(d.get("master",0.0) for d in details),3))
        row["runtime_sec"]=round(time.perf_counter()-t,3)
    except CapacityInfeasibleError as e:
        row.update(status="capacity_infeasible",error=str(e))
    except Exception as e:
        row.update(status="error",error=f"{type(e).__name__}: {e}")
    finally:
        clean(result,solver,data,raw)
    Path(a.output).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
