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
from model_and_algo.column_generation import solve_column_generation

seeds = [202, 303, 404, 505]
scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)

results = []
for seed in seeds:
    gc.collect()
    inst = generate_instance(scfg, 1, seed=seed)
    data = preprocess(inst)
    T = len(data.periods)
    I = len(data.cured_items)
    M = len(data.machines)
    print(f'\n=== seed={seed} T={T} I={I} M={M} ===', flush=True)

    t0 = time.perf_counter()
    r = solve_column_generation(
        data,
        max_iterations=5000,
        verbose=False,
        progress_interval_sec=0.0,
        time_limit=600.0,
        pricing_mode='ebrp',
        use_gurobi_init=False,
        skip_cross_check=True,
        pricing_inprocess=True,
        ebrp_force=True,
    )
    elapsed = time.perf_counter() - t0
    lrmp = r.extra.get('lrmp_obj') if r.extra else None
    niter = r.extra.get('num_iterations') if r.extra else None
    converged = r.extra.get('converged') if r.extra else False
    print(f'  LRMP={lrmp:.2f}  iters={niter}  converged={converged}  time={elapsed:.1f}s', flush=True)
    results.append((seed, lrmp, niter, converged, elapsed))

print('\n=== Summary ===')
print(f'{"seed":>6}  {"LRMP":>10}  {"iters":>6}  {"converged":>10}  {"time":>8}')
for seed, lrmp, niter, converged, elapsed in results:
    print(f'{seed:>6}  {lrmp:>10.2f}  {niter:>6}  {str(converged):>10}  {elapsed:>7.1f}s')

# Expected values from original run:
# seed 202: LRMP=28936, iters=49, 265.5s
# seed 303: LRMP=26671, iters=44, 147.4s
# seed 404: LRMP=34369, iters=54, 502.4s
# seed 505: LRMP=31403, iters=56, 356.0s
