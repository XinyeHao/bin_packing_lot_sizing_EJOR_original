"""Benchmark with parallel pricing across machines."""
import sys, os, gc, time

ROOT = r'C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['GRB_CS_QUIET'] = '1'


def main():
    import configuration as cfg
    cfg.apply_bench_generation()
    from configuration import SetConfig
    from instance_generator.generator import generate_instance
    from preprocessor.preprocess import preprocess
    from model_and_algo.column_generation import solve_column_generation

    seed = 101
    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)

    gc.collect()
    time.sleep(0.2)

    inst = generate_instance(scfg, 1, seed=seed)
    data = preprocess(inst)
    T = len(data.periods)
    I = len(data.cured_items)
    M = len(data.machines)
    print(f'T={T} I={I} M={M}')

    t0 = time.perf_counter()
    r = solve_column_generation(
        data,
        max_iterations=5000,
        verbose=True,
        progress_interval_sec=5.0,
        time_limit=600.0,
        pricing_mode='ebrp',
        use_gurobi_init=False,
        skip_cross_check=True,
        pricing_inprocess=False,   # <--- parallel across machines
        ebrp_force=True,
    )
    elapsed = time.perf_counter() - t0
    lrmp = r.extra.get('lrmp_obj') if r.extra else r.lower_bound
    niter = r.extra.get('num_iterations') if r.extra else None

    print(f'\nFINAL: seed={seed}  LRMP={lrmp:.2f}  iters={niter}  time={elapsed:.1f}s')
    print(f'Baseline (inprocess): 85.8s')
    if elapsed < 85.8:
        print(f'Speedup vs inprocess: {85.8/elapsed:.1f}x')


if __name__ == '__main__':
    main()
