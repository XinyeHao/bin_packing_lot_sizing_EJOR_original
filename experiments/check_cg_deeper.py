"""Check if standard-cuts CG converges with more iterations on seed 101."""
import sys, os, gc, time
ROOT = r'C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['GRB_CS_QUIET'] = '1'

import configuration as cfg
cfg.apply_bench_generation()
from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.column_generation import ColumnGenerationSolver

gc.collect(); time.sleep(0.5)
scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)
data = generate_instance(scfg, instance_id=1, seed=101)
data = preprocess(data)
print(f"T={len(data.periods)} I={len(data.cured_items)} M={len(data.machines)}", flush=True)

# Run with standard cuts, many more iterations
cg = ColumnGenerationSolver(
    data,
    pricing_mode='ebrp',
    ebrp_force=True,
    use_gurobi_init=False,
    skip_cross_check=True,
    pricing_inprocess=True,
    verbose=True,
    max_iterations=300,
    time_limit=300.0,
    lrmp_cuts='standard',
)
t0 = time.perf_counter()
obj, rcs, cols = cg.run()
elapsed = time.perf_counter() - t0
print(f"\n--- Results ---")
print(f"  lrmp_obj={obj:.2f}  converged={cg.converged}  stop_reason={cg.stop_reason}")
print(f"  iterations={len(cg.iteration_times)}  columns_total={sum(len(c) for c in cg.columns.values())}")
if cg.last_min_rc is not None:
    print(f"  last_min_rc={cg.last_min_rc:.4f}")
print(f"  last_valid_lb={cg.last_valid_lb:.2f}" if cg.last_valid_lb is not None else "  last_valid_lb=None")
print(f"  elapsed={elapsed:.1f}s")
