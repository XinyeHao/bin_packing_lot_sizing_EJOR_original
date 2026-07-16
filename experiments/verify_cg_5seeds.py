"""Verify CG results for seeds 101-505 (output after each seed completes)."""
import sys, os, gc, time
ROOT = r'C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['GRB_CS_QUIET'] = '1'

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.column_generation import ColumnGenerationSolver

seeds = [101, 202, 303, 404, 505]
scfg = SetConfig('small_cg', 36, 5, 0, 50, 5)
expected = {
    101: 26721.24,
    202: 28935.81,
    303: 27277.66,
    404: 26688.17,
    505: 26541.88,
}

results = []
t0 = time.perf_counter()
for seed in seeds:
    print(f"\n{'='*60}")
    print(f"CG seed={seed}")
    print(f"{'='*60}", flush=True)
    data = generate_instance(scfg, instance_id=seed % 100, seed=seed)
    data = preprocess(data)
    print(f"T={len(data.periods)} I={len(data.cured_items)} M={len(data.machines)}", flush=True)

    cg = ColumnGenerationSolver(
        data,
        pricing_mode='ebrp',
        ebrp_force=True,
        use_gurobi_init=False,
        skip_cross_check=True,
        pricing_inprocess=True,
        verbose=False,
        progress_interval_sec=10.0,
        time_limit=600.0,
    )
    cg_t0 = time.perf_counter()
    cg.run()
    cg_sec = time.perf_counter() - cg_t0

    ok = "OK" if abs(cg.lrmp_obj - expected.get(seed, 0)) < 1.0 else "MISMATCH"
    print(f"  Converged: {cg.converged}  LRMP: {cg.lrmp_obj:.2f} (expected {expected.get(seed, 0):.2f}) [{ok}]", flush=True)
    print(f"  Iterations: {len(cg.iteration_times)}  Time: {cg_sec:.1f}s  Columns: {sum(len(v) for v in cg.columns.values())}", flush=True)
    results.append((seed, cg.converged, cg.lrmp_obj, len(cg.iteration_times), cg_sec))
    gc.collect()

total = time.perf_counter() - t0
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for seed, conv, lrmp, iters, sec in results:
    ok = "OK" if abs(lrmp - expected.get(seed, 0)) < 1.0 else "MISMATCH"
    print(f"  seed={seed}: conv={conv}  LRMP={lrmp:.2f} (exp {expected.get(seed, 0):.2f}) [{ok}]  iters={iters}  time={sec:.1f}s")
print(f"Total: {total:.1f}s", flush=True)
