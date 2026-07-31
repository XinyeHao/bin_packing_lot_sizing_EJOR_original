"""Profile EBRP timing breakdown without cProfile (recursion-friendly)."""
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
    from model_and_algo.pricing_ebrp import EBRPContext, solve_subproblem_ebrp, EBRPStats
    from model_and_algo.column_generation import solve_column_generation

    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)

    for seed in [101, 202]:
        gc.collect(); time.sleep(0.2)
        inst = generate_instance(scfg, 1, seed=seed)
        data = preprocess(inst)
        T = len(data.periods)
        I = len(data.cured_items)
        M = len(data.machines)

        print(f'\n{"="*60}')
        print(f'seed={seed} T={T} I={I} M={M}')
        print(f'{"="*60}')

        # Build contexts for all machines
        t0 = time.perf_counter()
        from model_and_algo.pricing_ebrp import build_ebrp_contexts
        contexts = build_ebrp_contexts(data)
        t1 = time.perf_counter()
        print(f'EBRP build: {t1-t0:.2f}s')

        for m_idx, ctx in enumerate(contexts):
            total_subsets = sum(len(cache) for cache in ctx._subset_cache)
            print(f'  M{m_idx}: pool_specs={len(ctx.pool_specs)} subsets={total_subsets} events={len(ctx.event_times)}')

        # Run CG with verbose to see per-iteration pricing time
        print(f'\nRunning CG+EBRP...')
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
        print(f'Total: LRMP={lrmp:.2f} iters={niter} time={elapsed:.1f}s')


if __name__ == '__main__':
    main()
