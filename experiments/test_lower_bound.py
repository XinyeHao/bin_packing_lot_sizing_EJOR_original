"""Paper-style lower-bound test: IM-LP versus root CG.

Initial-column limits are 60/120/180 s for small/medium/large.  CG stops at
sum of machine reduced costs > -0.1 or 500 iterations, following Hao et al.
The resulting CG value is explicitly labelled approximate unless exact CG
convergence also occurs.
"""
from __future__ import annotations
import argparse,csv,gc,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
os.environ["GRB_CS_QUIET"]="1"
import gurobipy as gp
import configuration as cfg
from experiments.paper_experiment_config import MODEL_VERSION,PROCESS_ASSUMPTION,generation_metadata,make_set_config,scale_spec
from instance_generator.generator import generate_instance
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_lim
from preprocessor.preprocess import preprocess

INIT_LIMIT={"small":60.0,"medium":120.0,"large":180.0}

def clean(*xs):
 for x in xs:
  try: x.dispose()
  except Exception: pass
 gc.collect()
 try: gp.disposeDefaultEnv()
 except Exception: pass

def main():
 p=argparse.ArgumentParser(); p.add_argument("--scale",choices=("small","medium","large"),default="small"); p.add_argument("--seeds",nargs="+",type=int); p.add_argument("--im-lp-time-limit",type=float,default=None,help="None means no solver time limit, as in the reference design."); p.add_argument("--pricing-time-limit",type=float,default=300.0,help="Safety cap for one pricing MIP."); p.add_argument("--output-prefix",default=None); a=p.parse_args()
 spec=scale_spec(a.scale); seeds=a.seeds or list(spec.seeds); init=INIT_LIMIT[a.scale]; prefix=a.output_prefix or f"test_lower_bound_{a.scale}_15"; out=ROOT/"result"; out.mkdir(exist_ok=True); jp=out/f"{prefix}.json"; cp=out/f"{prefix}.csv"
 rows=[]
 for seed in seeds:
  raw=data=cg=None; row={"seed":seed,"scale":a.scale,"T":spec.periods,"I":spec.items,"error":None}
  try:
   cfg.apply_bench_generation(); raw=generate_instance(make_set_config(spec),seed%1000,seed); data=preprocess(raw)
   t=time.perf_counter(); im=solve_lim(data,time_limit=a.im_lp_time_limit,verbose=False); row.update(im_lp_status=im.status,im_lp_lb=im.lower_bound,im_lp_time_sec=round(time.perf_counter()-t,3))
   cg=ColumnGenerationSolver(data,max_iterations=500,init_time_limit=init,time_limit=None,pricing_mode="im_gurobi",use_gurobi_init=True,pricing_inprocess=True,skip_cross_check=True,lrmp_cuts="none",im_pricing_time_limit=a.pricing_time_limit,sum_rc_stop_threshold=-0.1,verbose=True,progress_interval_sec=30.0)
   t=time.perf_counter(); _,rcs,cols=cg.run(); root=cg.lrmp_obj
   row.update(cg_status=cg.stop_reason or ("converged" if cg.converged else "not_converged"),cg_exact_converged=cg.converged,cg_root_value=root,cg_time_sec=round(time.perf_counter()-t,3),cg_iterations=len(rcs),cg_columns=sum(len(v) for v in cols.values()),cg_sum_rc=rcs[-1] if rcs else None,cg_stop_rule="sum_rc > -0.1 or 500 iterations")
   if im.lower_bound is not None and root is not None:
    row["cg_minus_im_lp"]=root-im.lower_bound; row["relative_difference_pct"]=100*(root-im.lower_bound)/max(abs(root),abs(im.lower_bound),1e-6)
  except Exception as e: row["error"]=f"{type(e).__name__}: {e}"
  finally: clean(cg,data,raw)
  rows.append(row); payload={"experiment":"Paper-style approximate root-CG versus compact IM-LP","process_assumption":PROCESS_ASSUMPTION,"model_version":MODEL_VERSION,"generation_settings":generation_metadata(),"scale":a.scale,"seeds":seeds,"initial_column_time_limit_sec":init,"cg_stop":"sum reduced costs > -0.1 or 500 iterations","strictness":"sum-RC stop is approximate, not a certificate of exact DW-LP optimality","rows":rows}; jp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); fields=sorted({k for r in rows for k in r});
  with cp.open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
  print(f"seed={seed}: {row.get('cg_status')}",flush=True)
if __name__=="__main__": main()
