"""Check if root CG converges with standard cuts on seed 101."""
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

for cut_mode in (None, 'standard'):
    gc.collect()
    label = "none" if cut_mode is None else cut_mode
    print(f"\n--- lrmp_cuts={label} ---", flush=True)
    cg = ColumnGenerationSolver(
        data,
        pricing_mode='ebrp',
        ebrp_force=True,
        use_gurobi_init=False,
        skip_cross_check=True,
        pricing_inprocess=True,
        verbose=True,
        max_iterations=500,
        time_limit=600.0,
        lrmp_cuts=cut_mode,
    )
    t0 = time.perf_counter()
    obj, rcs, cols = cg.run()
    elapsed = time.perf_counter() - t0
    print(f"  lrmp_obj={obj:.2f}  converged={cg.converged}  stop_reason={cg.stop_reason}")
    print(f"  iterations={len(cg.iteration_times)}  columns_total={sum(len(c) for c in cg.columns.values())}")
    if cg.last_min_rc is not None:
        print(f"  last_min_rc={cg.last_min_rc:.6f}")
    print(f"  elapsed={elapsed:.1f}s", flush=True)
    del cg; gc.collect()
