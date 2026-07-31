"""Benchmark CG+EBRP speedup on multiple seeds."""
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

    seeds = [101, 202, 303, 404, 505]
    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)

    results = []
    for seed in seeds:
        gc.collect()
        time.sleep(0.2)

        print(f'\n{"="*70}')
        print(f'seed={seed}')
        print(f'{"="*70}')

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
            progress_interval_sec=0.0,
            time_limit=600.0,
            pricing_mode='ebrp',
            use_gurobi_init=False,
            skip_cross_check=True,
            pricing_inprocess=True,
            ebrp_force=True,
        )
        elapsed = time.perf_counter() - t0
        lrmp = r.extra.get('lrmp_obj') if r.extra else r.lower_bound
        niter = r.extra.get('num_iterations') if r.extra else None

        results.append((seed, lrmp, niter, elapsed))
        print(f'seed={seed}: LRMP={lrmp:.2f} iters={niter} time={elapsed:.1f}s')

    print(f'\n{"="*70}')
    print(f'Summary:')
    print(f'{"="*70}')
    for seed, lrmp, niter, elapsed in results:
        print(f'  seed={seed}: LRMP={lrmp:.2f}  iters={niter}  time={elapsed:.1f}s')
    avg_time = sum(t for _, _, _, t in results) / len(results)
    print(f'  Average time: {avg_time:.1f}s')


if __name__ == '__main__':
    main()
