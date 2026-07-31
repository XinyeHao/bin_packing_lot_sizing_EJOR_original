"""Test: run no-cuts CG first, then add cuts and re-price from the same pool."""
import sys, os, gc, time, copy
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

# Phase 1: no cuts CG to convergence
print("=== Phase 1: No cuts, EBRP pricing ===", flush=True)
cg1 = ColumnGenerationSolver(
    data, pricing_mode='ebrp', ebrp_force=True, use_gurobi_init=False,
    skip_cross_check=True, pricing_inprocess=True, verbose=False,
    max_iterations=300, time_limit=300.0, lrmp_cuts=None,
)
t0 = time.perf_counter()
obj1, rcs1, cols1 = cg1.run()
t1 = time.perf_counter() - t0
print(f"  LRMP={obj1:.2f}  iter={len(cg1.iteration_times)}  converged={cg1.converged}  stop={cg1.stop_reason}")
print(f"  min_RC={cg1.last_min_rc:.4f}" if cg1.last_min_rc else "  min_RC=None")
print(f"  last_valid_lb={cg1.last_valid_lb:.2f}" if cg1.last_valid_lb else "  last_valid_lb=None")
print(f"  elapsed={t1:.1f}s", flush=True)

# Phase 2: standard cuts, starting from the no-cuts column pool
print("\n=== Phase 2: Standard cuts, starting from no-cuts pool ===", flush=True)
cg2 = ColumnGenerationSolver(
    data, pricing_mode='ebrp', ebrp_force=True, use_gurobi_init=False,
    skip_cross_check=True, pricing_inprocess=True, verbose=True,
    max_iterations=200, time_limit=300.0, lrmp_cuts='standard',
    inherited_columns={
        m: [copy.deepcopy(c) for c in cols]
        for m, cols in cg1.columns.items()
    },
)
t0 = time.perf_counter()
obj2, rcs2, cols2 = cg2.run()
t2 = time.perf_counter() - t0
print(f"  LRMP={obj2:.2f}  iter={len(cg2.iteration_times)}  converged={cg2.converged}  stop={cg2.stop_reason}")
print(f"  min_RC={cg2.last_min_rc:.4f}" if cg2.last_min_rc else "  min_RC=None")
print(f"  last_valid_lb={cg2.last_valid_lb:.2f}" if cg2.last_valid_lb else "  last_valid_lb=None")
print(f"  elapsed={t2:.1f}s", flush=True)

# Summary
print(f"\n=== Summary ===")
print(f"  No cuts:   LRMP={obj1:.2f}  valid_lb={cg1.last_valid_lb:.2f}  iter={len(cg1.iteration_times)}")
print(f"  + cuts:    LRMP={obj2:.2f}  valid_lb={cg2.last_valid_lb:.2f}  iter={len(cg2.iteration_times)}")
print(f"  delta_LB = {cg2.last_valid_lb - cg1.last_valid_lb:+.2f}", flush=True)
