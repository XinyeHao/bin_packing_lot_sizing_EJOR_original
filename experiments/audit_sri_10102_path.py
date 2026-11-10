"""Audit SRI-only B&P against an independent compact-IM reference solution."""
from __future__ import annotations
import json, os, sys, types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
import configuration as cfg
from experiments.paper_experiment_config import SCALES, make_set_config
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import preprocess

SEED=10102; TOL=1e-5
OPTS={"deadline_cover":False,"sri":True,"sri_max_cuts":3,"sri_violation_tol":1e-6}
def compatible(data,node,ref):
    for (u,m,t),v in node.fixed_y.items():
        if int(ref["setup"][data.machines[m]][data.configs[u]][str(data.periods[t])]) != int(v): return False
    for i,v in node.fixed_scrap.items():
        if int(round(ref["scrap"][data.cured_items[i]])) != int(v): return False
    for i,(sense,bound) in node.fixed_wip.items():
        w=data.w_i[i]-ref["scrap"][data.cured_items[i]]
        if (sense=="le" and w>bound+TOL) or (sense=="ge" and w<bound-TOL): return False
    return True
def main():
    cfg.apply_bench_generation(); data=preprocess(generate_instance(make_set_config(SCALES["small"]),102,SEED))
    # Allow diagnostics to be written outside a read-only source checkout.
    diagnostic_dir = Path(os.environ.get("BPC_DIAGNOSTIC_DIR", ROOT / "result" / "diagnostics"))
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    reference_path = diagnostic_dir / "sri_10102_reference.json"
    if reference_path.exists():
        ref = json.loads(reference_path.read_text(encoding="utf-8"))
    else:
        im = solve_im(data,time_limit=900,mip_gap=0.0,method="sri_path_audit_reference")
        if im.objective is None: raise RuntimeError("reference IM has no feasible solution")
        ref = {
            "status": im.status, "objective": im.objective, "lb": im.lower_bound,
            "time": im.runtime_sec,
            "setup": {
                machine: {config: {str(period): value for period, value in periods.items()}
                          for config, periods in configs.items()}
                for machine, configs in im.setup.items()
            },
            "scrap": {item: float(value) for item, value in im.extra["scrap"].items()},
        }
        reference_path.write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")
    s=BranchAndPriceSolver(data,max_nodes=8000,time_limit=900,mip_gap=0.0,branch_strategy="frac_sb",use_node_im=False,root_im=False,final_im=False,strict_cg_certification=True,cg_kwargs={"pricing_mode":"ebrp","ebrp_force":True,"use_gurobi_init":True,"pricing_inprocess":True,"skip_cross_check":True,"lrmp_cuts":OPTS,"max_iterations":120,"ebrp_time_limit":None,"verbose":False})
    trace=[]; original=s._run_node_cg
    def wrapped(self,node,time_limit=None,incumbent=None):
        cg=original(node,time_limit=time_limit,incumbent=incumbent); lb=getattr(cg,"last_valid_lb",None) or cg.lrmp_obj; ok=compatible(data,node,ref)
        trace.append({"node_id":node.node_id,"depth":node.depth,"compatible":ok,"lb":lb,"invalid":bool(ok and lb is not None and lb>ref["objective"]+TOL),"stop":cg.stop_reason,"sri_count":len(cg.sri_cuts),"fixed_y":{str(k):v for k,v in node.fixed_y.items()},"fixed_scrap":{str(k):v for k,v in node.fixed_scrap.items()},"fixed_wip":{str(k):list(v) for k,v in node.fixed_wip.items()}}); return cg
    s._run_node_cg=types.MethodType(wrapped,s); result=s.solve(); invalid=[x for x in trace if x["invalid"]]
    report={"reference":ref,"sri_only":{"status":result.status,"objective":result.objective,"lb":result.lower_bound,"root_lb":result.root_valid_lb,"time":result.runtime_sec,"nodes":result.nodes_explored},"n_nodes":len(trace),"n_compatible":sum(x["compatible"] for x in trace),"invalid_nodes":invalid,"trace":trace}
    out=diagnostic_dir/"sri_10102_path_audit.json"; out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"reference":ref["objective"],"sri_only":result.objective,"invalid_nodes":len(invalid),"first":invalid[0] if invalid else None,"output":str(out)},ensure_ascii=False))
if __name__=="__main__": main()
