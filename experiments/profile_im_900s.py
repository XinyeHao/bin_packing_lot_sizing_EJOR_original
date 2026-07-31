"""Run IM (gap=0, 900s) on the T=35/I=50/LT=(8,9,10) instance."""
import sys, time, gc, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["GRB_CS_QUIET"] = "1"
sys.stderr = open(os.devnull, "w")

import configuration as cfg
cfg.apply_bench_generation()

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.solver import solve

gc.collect(); time.sleep(1.0)

scfg = SetConfig("small_cg", 35, 5, 0, 50, 5)
inst = generate_instance(scfg, 1, seed=888)
data = preprocess(inst)

print(f"T={len(data.periods)} I={len(data.cured_items)} M={len(data.machines)}", flush=True)

t0 = time.perf_counter()
result = solve(data, method="im", time_limit=900, mip_gap=0.0, verbose=False)
t_im = time.perf_counter() - t0

gap_str = f"{result.mip_gap*100:.2f}%" if result.mip_gap else "N/A"
print(f"IM 900s: obj={result.objective} lb={result.lower_bound} gap={gap_str} status={result.status} time={t_im:.1f}s", flush=True)
