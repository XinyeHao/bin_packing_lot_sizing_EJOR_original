"""Run IM on the 120T 60I 5M instance, 600s time limit."""
import sys, time, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.solver import solve

scfg = SetConfig("small_cg", 120, 5, 0, 60, 5)
inst = generate_instance(scfg, 1, seed=777)
data = preprocess(inst)
T = len(data.periods); I = len(data.cured_items)
M = len(data.machines); U = len(data.configs)
total = sum(len(data.items_by_machine[m]) for m in range(M))
print(f"Instance: T={T} I={I} M={M} U={U} items/m={total/M:.1f}", flush=True)

t0 = time.perf_counter()
result = solve(data, method="im_gurobi", time_limit=600, verbose=True)
elapsed = time.perf_counter() - t0

print(f"\n=== IM Result ===", flush=True)
print(f"Status:   {result.status}", flush=True)
print(f"Best obj: {result.best_obj}", flush=True)
print(f"Best lb:  {result.best_lb}", flush=True)
print(f"Gap:      {result.mip_gap*100 if result.mip_gap else 'N/A'}%", flush=True)
print(f"Time:     {elapsed:.1f}s", flush=True)
print(f"Solver:   {result.solver_time:.1f}s", flush=True)
