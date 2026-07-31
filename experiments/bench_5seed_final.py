"""Full 5-seed benchmark with all optimizations."""
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

    # Baseline approximate times from previous testing
    baseline_approx = {101: 113.8, 202: 126.5, 303: 148.2, 404: 169.1, 505: 98.7}

    results = []
    for seed in seeds:
        gc.collect(); time.sleep(0.3)
        print(f'\n======== seed={seed} ========')
        inst = generate_instance(scfg, 1, seed=seed)
        data = preprocess(inst)

        t0 = time.perf_counter()
        r = solve_column_generation(
            data, max_iterations=5000, verbose=False,
            progress_interval_sec=8.0, time_limit=600.0,
            pricing_mode='ebrp', use_gurobi_init=False,
            skip_cross_check=True, pricing_inprocess=True, ebrp_force=True,
        )
        elapsed = time.perf_counter() - t0
        lrmp = r.extra.get('lrmp_obj') if r.extra else r.lower_bound
        niter = r.extra.get('num_iterations') if r.extra else None

        bl = baseline_approx.get(seed, 120.0)
        spd = bl / elapsed if elapsed > 0 else 0
        print(f'RESULT: seed={seed}  LRMP={lrmp:.2f}  iters={niter}  time={elapsed:.1f}s  speedup={spd:.1f}x')
        results.append((seed, lrmp, niter, elapsed, spd))

    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    for s, l, i, t, sp in results:
        print(f'  seed={s}: {t:.1f}s  LRMP={l:.2f}  iters={i}  speedup={sp:.1f}x')
    avg_sp = sum(sp for _, _, _, _, sp in results) / len(results)
    avg_time = sum(t for _, _, _, t, _ in results) / len(results)
    print(f'  Average time: {avg_time:.1f}s')
    print(f'  Average speedup: {avg_sp:.1f}x')


if __name__ == '__main__':
    main()
