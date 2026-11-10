"""Run paper experiments E2--E4 with resumable, isolated workers."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.paper_experiment_config import MODEL_VERSION,PROCESS_ASSUMPTION,SCALES

def main(kind: str|None=None):
 p=argparse.ArgumentParser(); p.add_argument("--experiment",choices=("upper","cuts","sensitivity"),default=kind or "upper"); p.add_argument("--scales",nargs="+",choices=tuple(SCALES),default=list(SCALES)); p.add_argument("--output-prefix",default=None); a=p.parse_args()
 a.output_prefix=a.output_prefix or f"exp_{a.experiment}_paper"
 out=ROOT/"result"; out.mkdir(exist_ok=True); jp=out/f"{a.output_prefix}.json"; cp=out/f"{a.output_prefix}.csv"; tmp=out/f"_{a.output_prefix}_tmp"; tmp.mkdir(exist_ok=True)
 old=[]
 if jp.exists():
  try: old=json.loads(jp.read_text(encoding="utf-8")).get("rows",[])
  except Exception: pass
 rows={ (r.get("method"),r.get("scale"),r.get("seed"),r.get("capacity_multiplier",1.0)):r for r in old }
 methods=("im","bpc") if a.experiment=="upper" else (("bp","bpc") if a.experiment=="cuts" else ("root_ebrp","root_gurobi"))
 multipliers=(0.8,1.0,1.2) if a.experiment=="sensitivity" else (1.0,)
 for scale in a.scales:
  spec=SCALES[scale]
  for seed in spec.seeds:
   for mult in multipliers:
    for method in methods:
     key=(method,scale,seed,mult)
     if key in rows: continue
     one=tmp/f"{method}_{scale}_{seed}_{mult}.json"; log=tmp/f"{method}_{scale}_{seed}_{mult}.log"
     cmd=[sys.executable,"-B","experiments/paper_experiment_worker.py",method,scale,str(seed),str(one),"--limit",str(spec.time_limit_sec),"--capacity-multiplier",str(mult)]
     with log.open("w",encoding="utf-8") as f:
      try: subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=spec.time_limit_sec+180)
      except subprocess.TimeoutExpired: pass
     row=json.loads(one.read_text(encoding="utf-8")) if one.exists() else {"method":method,"scale":scale,"seed":seed,"capacity_multiplier":mult,"status":"wall_time_limit","error":"worker did not return before parent cap"}
     rows[key]=row
     payload={"experiment":a.experiment,"process_assumption":PROCESS_ASSUMPTION,"model_version":MODEL_VERSION,"scales":{s:{"T":x.periods,"I":x.items,"seeds":list(x.seeds),"time_limit_sec":x.time_limit_sec} for s,x in SCALES.items()},"rows":list(rows.values())}
     jp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
     fields=sorted({k for r in rows.values() for k in r});
     with cp.open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows.values())
     print(f"{a.experiment}: {method} {scale} {seed} done",flush=True)
if __name__=="__main__": main()
